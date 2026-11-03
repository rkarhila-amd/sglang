# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""BreakableCudaGraphBackend — segment-captured graphs with eager break
markers (eager_on_graph decorators on attention / mamba layers).
No torch.compile.
"""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import torch

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.model_executor.runner_backend.base_cuda_graph_backend import (
    BaseCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend.cuda_graph_dedup_mixin import (
    DedupedCudaGraphMixin,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    eager_on_graph,
    enable_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_utils.pool import (
    GraphPoolPrecarve,
    get_or_create_global_graph_memory_pool,
    graph_pool_capture_scope,
    graph_pool_replay_scope,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
        BaseCudaGraphRunner,
    )
    from sglang.srt.model_executor.runner.shape_key import ShapeKey


def _as_logits_output(output: Any):
    """Return ``LogitsProcessorOutput`` when ``output`` is one, else None."""
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

    if isinstance(output, LogitsProcessorOutput):
        return LogitsProcessorOutput
    return None


def _logits_row_count(output: Any) -> Optional[int]:
    """Leading rows shared by the decode logits and the captured hidden states."""
    for name in (
        "next_token_logits",
        "full_logits",
        "hidden_states",
        "last_hidden_states",
    ):
        value = getattr(output, name, None)
        if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] > 0:
            return int(value.shape[0])
    return None


def _logits_row_field_names(output: Any, rows: Optional[int]) -> set:
    if rows is None:
        return set()
    names = set()
    for field in dataclasses.fields(output):
        value = getattr(output, field.name)
        if (
            torch.is_tensor(value)
            and value.ndim >= 1
            and int(value.shape[0]) == rows
        ):
            names.add(field.name)
    return names


class BreakableCudaGraphBackend(DedupedCudaGraphMixin, BaseCudaGraphBackend):
    """Segmented capture: graphs break at attention / mamba boundaries;
    attention metadata is recomputed at replay outside captured segments.
    """

    def __init__(
        self,
        cuda_graph_runner: BaseCudaGraphRunner,
        *,
        enable_memory_saver: bool = False,
        debug_eager: bool = False,
    ) -> None:
        self._cuda_graph_runner = cuda_graph_runner
        self._model_runner = cuda_graph_runner.model_runner
        self._graphs: Dict[Any, BreakableCUDAGraph] = {}
        self._outputs: Dict[Any, Any] = {}
        self._capture_inputs: Dict[Any, Any] = {}
        self._pool = None
        self._device_module = cuda_graph_runner.device_module
        self._tp_group = get_parallel().tp_group
        self._capture_stream: Optional[torch.cuda.Stream] = None
        self._debug_eager = debug_eager
        self._shared_output_buffer: Optional[Any] = None
        # Names of LogitsProcessorOutput fields resized on dim 0. Set when
        # the shared buffer is allocated from the first captured shape.
        self._logits_row_fields: Optional[set] = None
        self._precarve = GraphPoolPrecarve()
        self._memory_saver_adapter: Optional[Any] = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
            and get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        )
        if (
            self._memory_saver_adapter is not None
            and self._memory_saver_adapter.enabled
        ):
            raise NotImplementedError(
                "Breakable CUDA graph is not compatible with memory saver mode"
            )

    @contextmanager
    def capture_session(self, stream: torch.cuda.Stream):
        if self._pool is None:
            self._pool = get_or_create_global_graph_memory_pool(self._device_module)
        set_graph_pool_id(self._pool)
        self._capture_stream = stream
        self._shared_output_buffer = None
        self._logits_row_fields = None
        self.begin_cuda_graph_capture()
        try:
            with self.replay_session():
                yield
        finally:
            try:
                self.end_cuda_graph_capture()
            finally:
                self._capture_stream = None

    def capture_one(
        self,
        shape_key: ShapeKey,
        forward_fn: Callable[[], Any],
        capture_inputs: Optional[Any] = None,
        post_warmup_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        warmup_out = None
        for _ in range(2):
            self._device_module.synchronize()
            self._tp_group.barrier()
            with self._precarve.measure():
                warmup_out = forward_fn()
            if post_warmup_hook is not None:
                post_warmup_hook()

        graph = BreakableCUDAGraph(self.deduped_cuda_graph)
        size = shape_key.size
        if self._shared_output_buffer is None:
            capacity_rows = self._cuda_graph_runner.cuda_graph_output_capacity_rows(
                warmup_out
            )
            if capacity_rows is None:
                capacity_rows = size
            self._shared_output_buffer = self._alloc_full_buffer(
                warmup_out, capacity_rows
            )
        buffer = self._shared_output_buffer
        with (
            graph_pool_capture_scope(),
            BreakableCUDAGraphCapture(
                cuda_graph=graph,
                pool=self._pool,
                stream=self._capture_stream,
            ),
        ):
            self._precarve.mint()
            if self._debug_eager:
                # The whole forward runs as one eager break that writes the
                # output buffer.
                # The warmup ran the same forward, so its output gives the
                # captured rows.
                out_rows = self._output_rows(warmup_out, size)

                @eager_on_graph
                def forward_into_buffer(output_buffer: Any) -> None:
                    self._copy_output_to_buffer(forward_fn(), output_buffer, out_rows)

                forward_into_buffer(buffer)
            else:
                out = forward_fn()
                out_rows = self._output_rows(out, size)
                self._copy_output_to_buffer(out, buffer, out_rows)

        stored = self._slice_output(buffer, out_rows)
        self._graphs[shape_key] = graph
        self._outputs[shape_key] = stored
        # CUDA graphs retain tensor addresses, not Python tensor lifetimes.
        self._capture_inputs[shape_key] = capture_inputs

    def _output_rows(self, output: Any, cap: int) -> int:
        """Leading-dim row count actually produced by the body.

        A body that shards or prunes its output along dim 0 returns fewer than
        ``cap`` rows. Tensors and proxy outputs are clamped to ``cap``.
        ``LogitsProcessorOutput`` returns its real token rows, which can
        exceed a request-count key; the shared buffer is sized to the first
        shape.
        """
        runner_rows = self._cuda_graph_runner.cuda_graph_output_rows(output)
        if runner_rows is not None:
            return runner_rows
        if torch.is_tensor(output):
            return min(cap, output.shape[0])
        if _as_logits_output(output) is not None:
            rows = _logits_row_count(output)
            # Token rows can exceed the request-count key. The shared buffer
            # is sized to the first (largest) shape, so keep the real count.
            return cap if rows is None else rows
        if isinstance(output, PPProxyTensors):
            rows = [t.shape[0] for t in output.tensors.values()]
            return min([cap, *rows])
        if isinstance(output, (list, tuple)) and output:
            return min(self._output_rows(o, cap) for o in output if o is not None)
        return cap

    def _alloc_full_buffer(self, output: Any, size: int) -> Any:
        """A same-structure buffer as ``output`` but with ``size`` leading rows."""
        if output is None:
            return None
        if torch.is_tensor(output):
            return output.new_empty((size, *output.shape[1:]))
        if isinstance(output, PPProxyTensors):
            return PPProxyTensors(
                {
                    key: t.new_empty((size, *t.shape[1:]))
                    for key, t in output.tensors.items()
                }
            )
        if isinstance(output, tuple):
            return tuple(self._alloc_full_buffer(o, size) for o in output)
        if isinstance(output, list):
            return [self._alloc_full_buffer(o, size) for o in output]
        logits_cls = _as_logits_output(output)
        if logits_cls is not None:
            return self._alloc_logits_buffer(output, logits_cls, size)
        raise TypeError(f"Unsupported BCG output type: {type(output)}")

    def _slice_output(self, output: Any, num_tokens: int) -> Any:
        if output is None:
            return None
        if torch.is_tensor(output):
            return output[:num_tokens]
        if isinstance(output, PPProxyTensors):
            return output[:num_tokens]
        if isinstance(output, tuple):
            return tuple(self._slice_output(item, num_tokens) for item in output)
        if isinstance(output, list):
            return [self._slice_output(item, num_tokens) for item in output]
        logits_cls = _as_logits_output(output)
        if logits_cls is not None:
            return self._slice_logits_output(output, logits_cls, num_tokens)
        raise TypeError(f"Unsupported BCG output type: {type(output)}")

    def _copy_output_to_buffer(
        self, output: Any, output_buffer: Any, num_tokens: int
    ) -> None:
        if output is None or output_buffer is None:
            if output is None and output_buffer is None:
                return
            raise ValueError(
                "BCG output structure changed between capture sizes: "
                f"{type(output)} vs {type(output_buffer)}"
            )
        if torch.is_tensor(output) and torch.is_tensor(output_buffer):
            output_buffer[:num_tokens].copy_(output[:num_tokens])
            return
        if isinstance(output, PPProxyTensors) and isinstance(
            output_buffer, PPProxyTensors
        ):
            if output.tensors.keys() != output_buffer.tensors.keys():
                raise ValueError(
                    "BCG output proxy structure changed between capture sizes: "
                    f"{output.tensors.keys()} != {output_buffer.tensors.keys()}"
                )
            for key, tensor in output.tensors.items():
                self._copy_output_to_buffer(
                    tensor, output_buffer.tensors[key], num_tokens
                )
            return
        if (isinstance(output, list) and isinstance(output_buffer, list)) or (
            isinstance(output, tuple) and isinstance(output_buffer, tuple)
        ):
            if len(output) != len(output_buffer):
                raise ValueError(
                    "BCG output sequence structure changed between capture sizes: "
                    f"{len(output)} != {len(output_buffer)}"
                )
            for item, buffer in zip(output, output_buffer):
                self._copy_output_to_buffer(item, buffer, num_tokens)
            return
        if _as_logits_output(output) is not None and _as_logits_output(
            output_buffer
        ) is not None:
            self._copy_logits_output(output, output_buffer, num_tokens)
            return
        raise TypeError(
            "Unsupported BCG output buffer pair: "
            f"{type(output)} vs {type(output_buffer)}"
        )

    def _alloc_logits_buffer(self, output: Any, logits_cls: type, size: int) -> Any:
        """One process-lifetime buffer for decode logits and hidden states.

        The forward returns ``LogitsProcessorOutput``. Graph-pool tensors from
        a segment are not stable across the next shape, so the row fields are
        copied into this buffer during capture and replay returns views of it.
        """
        rows = _logits_row_count(output)
        capacity = size if rows is None else max(size, rows)
        row_fields = _logits_row_field_names(output, rows)
        self._logits_row_fields = row_fields
        kwargs = {}
        for field in dataclasses.fields(output):
            value = getattr(output, field.name)
            if field.name in row_fields:
                kwargs[field.name] = value.new_empty((capacity, *value.shape[1:]))
            elif torch.is_tensor(value):
                kwargs[field.name] = torch.empty_like(value)
            elif value is None:
                kwargs[field.name] = None
            else:
                raise TypeError(
                    "Unsupported BCG logits field "
                    f"{field.name}: {type(value)}"
                )
        return logits_cls(**kwargs)

    def _slice_logits_output(
        self, output: Any, logits_cls: type, num_tokens: int
    ) -> Any:
        row_fields = self._logits_row_fields or set()
        kwargs = {}
        for field in dataclasses.fields(output):
            value = getattr(output, field.name)
            if field.name in row_fields and torch.is_tensor(value):
                kwargs[field.name] = value[:num_tokens]
            else:
                kwargs[field.name] = value
        return logits_cls(**kwargs)

    def _copy_logits_output(
        self, output: Any, output_buffer: Any, num_tokens: int
    ) -> None:
        row_fields = self._logits_row_fields
        if row_fields is None:
            raise ValueError("BCG logits buffer was not allocated")
        for field in dataclasses.fields(output):
            src = getattr(output, field.name)
            dst = getattr(output_buffer, field.name)
            if field.name in row_fields:
                if not (torch.is_tensor(src) and torch.is_tensor(dst)):
                    raise TypeError(
                        "BCG logits row field "
                        f"{field.name} changed: {type(src)} vs {type(dst)}"
                    )
                if src.shape[0] < num_tokens or dst.shape[0] < num_tokens:
                    raise ValueError(
                        f"BCG logits field {field.name} has {tuple(src.shape)} "
                        f"into {tuple(dst.shape)}, need {num_tokens} rows"
                    )
                if src.shape[1:] != dst.shape[1:]:
                    raise ValueError(
                        f"BCG logits field {field.name} trailing shape changed: "
                        f"{tuple(src.shape)} vs {tuple(dst.shape)}"
                    )
                dst[:num_tokens].copy_(src[:num_tokens])
                continue
            if src is None and dst is None:
                continue
            if torch.is_tensor(src) and torch.is_tensor(dst):
                if src.shape != dst.shape:
                    raise ValueError(
                        f"BCG logits field {field.name} shape changed: "
                        f"{tuple(src.shape)} vs {tuple(dst.shape)}"
                    )
                dst.copy_(src)
                continue
            raise TypeError(
                "Unsupported BCG logits field "
                f"{field.name}: {type(src)} vs {type(dst)}"
            )

    def can_run(self, forward_batch: ForwardBatch, shape_key: ShapeKey) -> bool:
        return shape_key in self._graphs

    @contextmanager
    def replay_session(self):
        with enable_breakable_cuda_graph():
            yield

    def replay(
        self,
        shape_key: ShapeKey,
        static_forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any:
        with graph_pool_replay_scope():
            self._graphs[shape_key].replay()
        return self._outputs[shape_key]

    def cleanup(self) -> None:
        self.close()
        self._graphs.clear()
        self._outputs.clear()
        self._capture_inputs.clear()
        self._pool = None
        self._shared_output_buffer = None
        self._logits_row_fields = None
