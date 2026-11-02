"""SHUFFLE 5D KV pool helpers for the AITER attention backend.

This module hosts the attention pathways that are specific to the
``SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d`` (SHUFFLE 5D) physical layout.
They live here rather than inline in
:mod:`sglang.srt.layers.attention.aiter_backend` so the main backend
file keeps focused on the legacy NHD path and on dispatch wiring.

Each entry point takes the :class:`AiterAttnBackend` instance as its
first argument so it can reach the shared per-step metadata
(``forward_metadata``, ``qo_indptr``, ``input_dtype``, …) without
needing to be a method on the class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch


def flydsl_stream_capturing() -> bool:
    """True only while this stream is recording a CUDA graph.

    ``get_is_capture_mode`` is also set for the pre-capture warmup and for
    a breakable eager gap. Those are not stream captures.
    """
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def flydsl_use_graph_bound(
    *, stream_capturing: bool, capture_mode: bool, breakable: bool
) -> bool:
    """Whether uniform ``pa_decode`` should bake a host max-context bound.

    A full CUDA graph, including its pre-capture warmup, must specialize on
    one bound: replay does not re-enter Python, and a different
    ``max_context_length`` compiles a different kernel. A breakable eager
    gap sets the capture-mode flag while the stream is not capturing; that
    launch keeps the measured length.
    """
    if breakable and not stream_capturing:
        return False
    return bool(stream_capturing or capture_mode)


def flydsl_note_scratch(
    cache: dict, pinned: dict, key, value, pin: bool
) -> None:
    """Remember scratch, and keep any entry a captured graph still holds.

    The eager cache stays at 16 keys. Pinned entries are the tensors
    recorded into a CUDA graph; dropping them would free that storage
    while the graph still writes it.
    """
    cache[key] = value
    if pin:
        pinned[key] = value
    if len(cache) <= 16:
        return
    for old in list(cache.keys()):
        if old == key or old in pinned:
            continue
        cache.pop(old)
        return


try:
    # `mha_batch_prefill_func` is re-exported at the aiter top level via
    # `aiter/__init__.py` (`from .ops.mha import *`). Note: a bare
    # `from aiter.mha import ...` does NOT work — that module path only
    # exists as `aiter.ops.mha`.
    from aiter import mha_batch_prefill_func
    from aiter.ops.triton.gluon.pa_decode_gluon import (
        get_recommended_splits,
        pa_decode_gluon,
    )
except ImportError:  # pragma: no cover - import-time guard mirrors aiter_backend
    mha_batch_prefill_func = None
    pa_decode_gluon = None
    get_recommended_splits = None

from sglang.kernels.ops.kvcache.shuffle_5d_cache import (
    launch_gather_shuffle_5d_to_linear,
)
from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype

if TYPE_CHECKING:
    from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def forward_extend_vectorized_5d(
    backend: AiterAttnBackend,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    bs0: int,
    window_size,
    sinks,
) -> torch.Tensor:
    """``forward_extend`` specialization for the SHUFFLE 5D KV pool.

    Two sub-paths, both routing through aiter's 3D LINEAR-mode
    ``mha_batch_prefill_func`` (page_size=1):

    1. Fresh-prompt shortcut: when every request in the batch has zero
       ``extend_prefix_lens`` (first chunk of a fresh prompt, or any
       path bypassing prefix reuse) the fresh ``(k, v)`` inputs ARE the
       full KV stream — skip pool reads entirely and run on bf16
       ``(k, v)`` directly. No descales needed since no data is read
       from the (possibly fp8) cache.

    2. Gather-and-linearize: otherwise gather the per-token K/V from the
       SHUFFLE 5D pool via ``launch_gather_shuffle_5d_to_linear``
       (triton inverse of the SHUFFLE writer) into a contiguous
       ``(T, H, D)`` buffer in the cache's ``store_dtype``, then run the
       same LINEAR prefill. fp8-store layers are forwarded to aiter as
       raw fp8 with the per-tensor descales — aiter's LINEAR-mode kernel
       supports fp8 K/V/Q natively, so no host-side dequant is needed.

    The fallback exists because aiter's paged ``mha_batch_prefill_func``
    lacks a compiled kernel for our
    ``(page_size=64, bf16/fp8, SHUFFLE 5D)`` configuration; calling it
    from the 5D pool aborts with ``"no matching kernel found"``.

    Returns the ``(T, H_q * D_v)`` attention output, ready to be
    returned from ``AiterAttnBackend.forward_extend``.
    """
    # Path 1: fresh-prompt shortcut.
    extend_no_prefix = forward_batch.extend_prefix_lens_cpu is not None and not any(
        forward_batch.extend_prefix_lens_cpu
    )
    if extend_no_prefix:
        k_lin = k.contiguous().view(-1, layer.tp_k_head_num, layer.qk_head_dim)
        v_lin = v.contiguous().view(-1, layer.tp_v_head_num, layer.v_head_dim)
        total_tokens = k_lin.shape[0]
        kv_indices_lin = torch.arange(
            total_tokens, dtype=torch.int32, device=k_lin.device
        )
        kv_indptr_lin = backend.forward_metadata.qo_indptr[:bs0]
        max_q = int(backend.forward_metadata.max_q_len)
        o = mha_batch_prefill_func(
            q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            k_lin,
            v_lin,
            backend.forward_metadata.qo_indptr[:bs0],
            kv_indptr_lin,
            kv_indices_lin,
            max_q,
            max_q,
            softmax_scale=layer.scaling,
            causal=True,
            logits_soft_cap=backend.logits_soft_cap,
            alibi_slopes=None,
            return_lse=False,
            return_attn_probs=False,
            window_size=window_size,
            sink_ptr=sinks,
        )
        if o.dtype != backend.input_dtype:
            o = o.to(backend.input_dtype)
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    # Path 2: gather-and-linearize.
    # SWA layers gather from the SWA sub-pool via swa_page_table;
    # full-attn layers gather from the full sub-pool via kv_indices.
    # Both are per-TOKEN slot id lists populated by
    # ``create_flashinfer_kv_indices_triton`` from ``req_to_token`` (one
    # slot id per logical token), so the first ``seq_lens_sum`` entries
    # of either tensor are exactly the per-token absolute pool slot ids
    # in request-major order — no per-token gather metadata to build on
    # host.
    is_swa_layer = (
        layer.sliding_window_size is not None
        and layer.sliding_window_size > -1
        and backend.forward_metadata.swa_page_table is not None
    )
    total_kv = int(forward_batch.seq_lens_sum)
    if is_swa_layer:
        slot_ids = backend.forward_metadata.swa_page_table[:total_kv]
    else:
        slot_ids = backend.forward_metadata.kv_indices[:total_kv]

    # Resolve the raw 5D K/V buffer for this layer from the view holding it.
    kv = backend.token_to_kv_pool.layer_router.layer(layer.layer_id)
    sub_pool = kv.view
    sub_layer_id = kv.view_layer_id
    k_buf = sub_pool.k_buffer[sub_layer_id - sub_pool.start_layer]
    v_buf = sub_pool.v_buffer[sub_layer_id - sub_pool.start_layer]

    k_lin, v_lin = launch_gather_shuffle_5d_to_linear(k_buf, v_buf, slot_ids)
    # k_lin / v_lin come out in ``store_dtype`` (uint8 for fp8 pools
    # because ``Tensor.index_put`` isn't implemented for fp8 — see
    # ``MHATokenToKVPool`` ctor). Reinterpret them back to the compute
    # dtype so aiter sees matching q/k/v dtypes. The bytes are
    # identical; this is a zero-copy view.
    if sub_pool.store_dtype != sub_pool.dtype:
        k_lin = k_lin.view(sub_pool.dtype)
        v_lin = v_lin.view(sub_pool.dtype)

    # For fp8 K/V we hand the raw fp8 tensors and the layer's per-tensor
    # descales straight to aiter.
    if sub_pool.dtype == fp8_dtype:
        q_local, q_descale_local = backend._quantize_q_fp8(q)
        k_descale_local = (
            layer.k_scale if layer.k_scale is not None else backend.k_scale
        )
        v_descale_local = (
            layer.v_scale if layer.v_scale is not None else backend.v_scale
        )
    else:
        q_local = q
        q_descale_local = None
        k_descale_local = None
        v_descale_local = None

    kv_indptr_lin = backend.forward_metadata.kv_indptr[:bs0]
    kv_indices_lin = torch.arange(total_kv, dtype=torch.int32, device=k_lin.device)
    max_kv = int(backend.forward_metadata.max_kv_len)
    max_q = int(backend.forward_metadata.max_q_len)

    o = mha_batch_prefill_func(
        q_local.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
        k_lin,
        v_lin,
        backend.forward_metadata.qo_indptr[:bs0],
        kv_indptr_lin,
        kv_indices_lin,
        max_q,
        max_kv,
        softmax_scale=layer.scaling,
        causal=True,
        logits_soft_cap=backend.logits_soft_cap,
        alibi_slopes=None,
        return_lse=False,
        return_attn_probs=False,
        window_size=window_size,
        sink_ptr=sinks,
        q_descale=q_descale_local,
        k_descale=k_descale_local,
        v_descale=v_descale_local,
    )
    if o.dtype != backend.input_dtype:
        o = o.to(backend.input_dtype)
    return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)


def forward_decode_vectorized_5d(
    backend: AiterAttnBackend,
    q: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    o: torch.Tensor,
    sinks,
) -> None:
    """``forward_decode`` specialization for the SHUFFLE 5D KV pool.

    Runs ``pa_decode_gluon`` for both full-attention and sliding-window
    layers — when SHUFFLE 5D is active the SWA sub-pool is also
    allocated 5D (see ``SWAKVPool`` ctor), so we keep one decode kernel
    instead of falling back to ``unified_attention`` for SWA layers.

    The choice between the two layer kinds is purely metadata:

    * Full-attn  → ``kv_indices`` page table + ``sliding_window=0`` +
      ``max_part_num`` recommended by aiter heuristics.
    * SWA layer  → ``swa_page_table`` + ``sliding_window=layer.sliding_window_size``
      + ``max_part_num=1`` (SWA windows are small enough that
      splitting does not help).

    fp8 KV requires per-tensor ``key_scale`` / ``value_scale`` to be
    forwarded; without them the kernel reads the fp8 bytes as fp8
    values without any dequant and produces garbage logits.

    Writes the attention output into ``o`` in place (via a stride-0
    safe ``o.view``).
    """
    bs = forward_batch.batch_size
    num_kv_heads = layer.tp_k_head_num
    num_q_heads = layer.tp_q_head_num
    q_group = num_q_heads // num_kv_heads
    is_swa_layer = (
        layer.sliding_window_size is not None and layer.sliding_window_size > -1
    )

    if is_swa_layer:
        block_tables_pa = (
            backend.forward_metadata.swa_page_table
            if backend.forward_metadata.swa_page_table is not None
            else backend.forward_metadata.kv_indices
        )
        ctx_part = 256
        max_part_num = 1
        sliding_window_arg = int(layer.sliding_window_size)
    else:
        block_tables_pa = backend.forward_metadata.kv_indices
        ctx_part = 256
        max_part_num = get_recommended_splits(bs, num_kv_heads)
        sliding_window_arg = 0

    q_in = q.view(-1, num_q_heads, layer.qk_head_dim)
    # Direct view of o as kernel output — saves a per-layer o.copy_ of
    # bs * H_q * D bf16 elementwise.
    o_view = o.view(-1, num_q_heads, layer.v_head_dim)
    exp_sums = torch.empty(
        (bs, num_kv_heads, max_part_num, q_group),
        dtype=torch.float32,
        device=q_in.device,
    )
    max_logits = torch.empty_like(exp_sums)
    temporary_output = torch.empty(
        (bs, num_kv_heads, max_part_num, q_group, layer.qk_head_dim),
        dtype=q_in.dtype,
        device=q_in.device,
    )

    # For fp8 KV cache the kernel needs per-tensor dequant scales
    # (key_scale / value_scale). Without them the fp8 bytes are
    # interpreted as fp8 values with no dequant.
    key_scale = None
    value_scale = None
    if backend.kv_cache_dtype == fp8_dtype:
        key_scale = layer.k_scale if layer.k_scale is not None else backend.k_scale
        value_scale = layer.v_scale if layer.v_scale is not None else backend.v_scale

    pa_decode_gluon(
        output=o_view,
        query=q_in,
        key_cache=k_cache,
        value_cache=v_cache,
        context_lengths=forward_batch.seq_lens,
        block_tables=block_tables_pa,
        softmax_scale=layer.scaling,
        query_length=1,
        max_context_partition_num=max_part_num,
        context_partition_size=ctx_part,
        compute_type=backend.input_dtype,
        key_scale=key_scale,
        value_scale=value_scale,
        exp_sums=exp_sums,
        max_logits=max_logits,
        temporary_output=temporary_output,
        sinks=sinks,
        sliding_window=sliding_window_arg,
        ps=True,
    )


class FlydslPaDecode:
    """FlyDSL ``pa_decode`` on a SHUFFLE 5D cache, plans reused per batch size.

    ``context_lengths`` count tokens the kernel may read, including the query
    tokens. ``query_length`` is the same for every row. ``block_tables`` are
    page ids. A dense plan (``sliding_window=0``) may be refreshed with a new
    query length; the plan itself is tied to one batch size.
    """

    def __init__(self):
        self._plans: dict = {}
        # One scratch set per (heads, capacity, query rows). A ragged verify
        # step launches two widths; swapping a single buffer would allocate
        # on every layer.
        self._scratch_cache: dict = {}
        # Scratch recorded into a CUDA graph. Never evicted.
        self._scratch_pinned: dict = {}

    def run(
        self,
        output: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        context_lengths: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: float,
        query_length: int,
        max_context_length: int,
        pin_scratch: bool = False,
    ) -> None:
        from aiter.ops.flydsl.pa_decode import pa_decode, plan_pa_decode

        if query_length < 1:
            raise ValueError(f"query_length must be positive, got {query_length}")
        if context_lengths.dtype != torch.int32 or not context_lengths.is_contiguous():
            context_lengths = context_lengths.to(dtype=torch.int32).contiguous()
        if block_tables.dtype != torch.int32 or not block_tables.is_contiguous():
            block_tables = block_tables.to(dtype=torch.int32).contiguous()
        batch = int(context_lengths.shape[0])
        if batch < 1:
            raise ValueError("FlyDSL pa_decode needs at least one sequence")
        if key_cache.dim() != 5 or value_cache.dim() != 5:
            raise ValueError(
                "FlyDSL pa_decode expects 5D K/V caches, "
                f"got K {tuple(key_cache.shape)} V {tuple(value_cache.shape)}"
            )
        num_kv_heads = int(key_cache.shape[1])
        num_q_heads = int(query.shape[1])
        head_dim = int(query.shape[-1])
        if num_q_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_q_heads {num_q_heads} must be divisible by "
                f"num_kv_heads {num_kv_heads}"
            )
        if query.shape[0] != batch * query_length:
            raise ValueError(
                f"query rows {query.shape[0]} != batch {batch} * "
                f"query_length {query_length}"
            )
        if not query.is_contiguous():
            query = query.contiguous()
        out = output
        if out.shape != query.shape or not out.is_contiguous():
            out = torch.empty(query.shape, dtype=query.dtype, device=query.device)

        key = (batch, num_kv_heads, query.device)
        plan = plan_pa_decode(
            context_lengths,
            num_kv_heads,
            query_length=query_length,
            sliding_window=0,
            plan=self._plans.get(key),
        )
        self._plans[key] = plan

        rows = query_length * (num_q_heads // num_kv_heads)
        scratch_key = (
            num_kv_heads,
            int(plan.capacity),
            rows,
            head_dim,
            query.dtype,
            query.device,
        )
        if self._scratch_key != scratch_key:
            scalar = (num_kv_heads, int(plan.capacity), rows)
            exp_sums = torch.empty(scalar, dtype=torch.float32, device=query.device)
            self._scratch = (
                exp_sums,
                torch.empty_like(exp_sums),
                torch.empty(*scalar, head_dim, dtype=query.dtype, device=query.device),
            )
        flydsl_note_scratch(
            self._scratch_cache,
            self._scratch_pinned,
            scratch_key,
            cached,
            pin_scratch,
        )
        exp_sums, max_logits, temporary = cached

        pa_decode(
            out,
            query,
            key_cache,
            value_cache,
            context_lengths,
            block_tables,
            float(softmax_scale),
            int(query_length),
            work_plan=plan,
            exp_sums=exp_sums,
            max_logits=max_logits,
            temporary_output=temporary,
            max_context_length=int(max_context_length),
        )
        if out is not output:
            output.copy_(out)


def flydsl_verify_live_batch(q_rows: int, qlen: int, pool_bs: int) -> int:
    """How many leading requests ``q_rows`` cover at one uniform ``qlen``.

    The scheduler pool can be longer than this forward's query. Hybrid verify
    rewrites ``batch_size`` to ``num_tokens // width`` and leaves a trailing
    pool slot in place, so a 24-row query at width 8 is 3 requests even when
    ``req_pool_indices`` still has 4. Those query rows are packed in pool
    order, so the kernel reads the prefix. A ragged width is not a multiple
    of ``qlen`` and is rejected.
    """
    if qlen < 1:
        raise ValueError(f"query length must be positive, got {qlen}")
    if pool_bs < 1:
        raise ValueError(f"verify pool must be positive, got {pool_bs}")
    if q_rows < 1 or q_rows % qlen != 0:
        raise ValueError(
            f"FlyDSL verify query rows {q_rows} are not a positive multiple "
            f"of qlen {qlen} (pool {pool_bs})"
        )
    live = q_rows // qlen
    if live > pool_bs:
        raise ValueError(
            f"FlyDSL verify live batch {live} exceeds pool {pool_bs} "
            f"(query rows {q_rows}, qlen {qlen})"
        )
    return live


def flydsl_verify_width_groups(
    qo_vals: list[int],
    q_rows: int,
    pool_bs: int,
) -> tuple[int, list[tuple[int, list[int]]]]:
    """Live batch and ``(width, request positions)`` groups from ``qo_indptr``.

    The live batch is the first index whose prefix sum equals ``q_rows``.
    Requests that share a width are one ``pa_decode`` launch. The width is
    the indptr delta, never rounded up to the batch max: a 7-token request
    must not read an eighth KV token. Positions within a group stay in pool
    order. Groups are ordered by the first request of that width.
    """
    if q_rows < 1:
        raise ValueError(f"query rows must be positive, got {q_rows}")
    if pool_bs < 1:
        raise ValueError(f"verify pool must be positive, got {pool_bs}")
    if len(qo_vals) < 2:
        raise ValueError("verify qo_indptr is missing")
    limit = min(pool_bs, len(qo_vals) - 1)
    bs = None
    for i in range(1, limit + 1):
        if qo_vals[i] == q_rows:
            bs = i
            break
        if qo_vals[i] > q_rows:
            raise ValueError(
                f"qo_indptr jumps over {q_rows} query rows at request {i} "
                f"(qo={qo_vals[: i + 1]}, pool {pool_bs})"
            )
    if bs is None:
        raise ValueError(
            f"qo_indptr does not cover {q_rows} query rows "
            f"(pool {pool_bs}, prefix {qo_vals[: min(8, len(qo_vals))]})"
        )
    buckets: dict[int, list[int]] = {}
    order: list[int] = []
    for i in range(bs):
        width = qo_vals[i + 1] - qo_vals[i]
        if width < 1:
            raise ValueError(
                f"verify width at request {i} is {width} (qo={qo_vals[: bs + 1]})"
            )
        if width not in buckets:
            order.append(width)
            buckets[width] = []
        buckets[width].append(i)
    return bs, [(width, buckets[width]) for width in order]


def flydsl_group_row_indices(
    qo_vals: list[int], width: int, positions: list[int]
) -> list[int]:
    """Query-row ids for one width group. Not a contiguous slice when a
    different width sits between two members.
    """
    rows: list[int] = []
    for i in positions:
        start = qo_vals[i]
        rows.extend(range(start, start + width))
    return rows


def flydsl_ragged_dense_rows(
    qo_indptr: torch.Tensor,
    verify_lens: torch.Tensor,
    q_rows: int,
    max_q: int,
) -> torch.Tensor:
    """Map packed query rows onto a dense ``[batch, max_q]`` window.

    Token ``t`` of request ``i`` (width ``w``) lands at
    ``i * max_q + (max_q - w + t)``. FlyDSL ``pa_decode`` uses dense causal
    MTP: query index ``j`` reads through context index
    ``context_len - max_q + j``, and the last query is the newest token.
    Right-aligning the real tokens makes query ``max_q - w`` the first real
    token (it reads through the prefix) and query ``max_q - 1`` the last.
    Slots before the real tokens stay zero; their outputs are discarded.
    """
    if q_rows < 1:
        raise ValueError(f"query rows must be positive, got {q_rows}")
    if max_q < 1:
        raise ValueError(f"max_q must be positive, got {max_q}")
    bs = int(verify_lens.shape[0])
    if bs < 1:
        raise ValueError(f"verify batch must be positive, got {bs}")
    if int(qo_indptr.shape[0]) < bs + 1:
        raise RuntimeError(
            f"qo_indptr length {int(qo_indptr.shape[0])} < batch {bs} + 1"
        )
    ends = qo_indptr[1 : bs + 1]
    token_ids = torch.arange(q_rows, device=qo_indptr.device, dtype=torch.int32)
    req = torch.searchsorted(ends, token_ids, right=True).to(torch.long)
    starts = qo_indptr[:bs].index_select(0, req)
    offset = token_ids.to(torch.long) - starts.to(torch.long)
    widths = verify_lens.index_select(0, req).to(torch.long)
    local = (max_q - widths) + offset
    return req * max_q + local


def _flydsl_ragged_static_bufs(backend, bs: int, max_q: int, heads: int, dim: int, like):
    """Dense query, output, and context for one captured ragged shape.

    Allocated on the warmup, before stream capture. A tensor created inside
    the capturing region is not a stable replay address.
    """
    key = (bs, max_q, heads, dim, like.dtype, str(like.device))
    cache = getattr(backend, "_flydsl_ragged_bufs", None)
    if cache is None:
        cache = {}
        backend._flydsl_ragged_bufs = cache
    bufs = cache.get(key)
    if bufs is None:
        rows = bs * max_q
        bufs = (
            torch.empty((rows, heads, dim), dtype=like.dtype, device=like.device),
            torch.empty((rows, heads, dim), dtype=like.dtype, device=like.device),
            torch.empty((bs,), dtype=torch.int32, device=like.device),
        )
        cache[key] = bufs
    return bufs


def flydsl_forward_ragged_padded(
    backend,
    q: torch.Tensor,
    o: torch.Tensor,
    layer,
    forward_batch,
    layout,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    max_context: int,
    max_q: int,
    runner,
) -> None:
    """One ``pa_decode`` at a fixed ``max_q`` for a mixed-width verify batch.

    ``verify_lens``, ``qo_indptr``, and ``seq_lens`` are device tensors the
    compact runner refreshes before replay. This launch does not read a
    length back to the host while the stream is capturing. ``max_q`` and
    ``max_context`` are the host specialization baked into the graph.
    """
    bs = int(layout.verify_lens.shape[0])
    n_q = int(layer.tp_q_head_num)
    dim = int(layer.qk_head_dim)
    q3 = q.reshape(-1, n_q, dim)
    o3 = o.reshape(-1, n_q, int(layer.v_head_dim))
    q_rows = int(q3.shape[0])
    if int(o3.shape[0]) < q_rows:
        raise RuntimeError(
            f"ragged verify output rows {int(o3.shape[0])} < query rows {q_rows}"
        )
    if int(layer.qk_head_dim) != int(layer.v_head_dim):
        raise RuntimeError(
            "FlyDSL ragged verify requires qk_head_dim == v_head_dim, "
            f"got {layer.qk_head_dim} and {layer.v_head_dim}."
        )
    lens = layout.verify_lens[:bs]
    qo = layout.qo_indptr_device
    capturing = flydsl_stream_capturing()
    if not capturing:
        widest = int(lens.max().item())
        if widest > max_q or widest < 1:
            raise RuntimeError(
                f"FlyDSL ragged verify width {widest} is outside 1..{max_q}."
            )
    q_dense, o_dense, context = _flydsl_ragged_static_bufs(
        backend, bs, max_q, n_q, dim, q3
    )
    prefix = forward_batch.seq_lens[:bs]
    src = prefix if prefix.dtype == torch.int32 else prefix.to(dtype=torch.int32)
    context.copy_(src)
    add = lens if lens.dtype == torch.int32 else lens.to(dtype=torch.int32)
    context.add_(add)
    rows = flydsl_ragged_dense_rows(qo, lens, q_rows, max_q)
    q_dense.zero_()
    q_dense.index_copy_(0, rows, q3[:q_rows])
    req_indices = forward_batch.req_pool_indices[:bs]
    table = flydsl_block_table(
        backend.req_to_token, req_indices, int(backend.page_size), int(max_context)
    )
    runner.run(
        o_dense,
        q_dense,
        k_cache,
        v_cache,
        context,
        table,
        float(layer.scaling),
        int(max_q),
        int(max_context),
        pin_scratch=True,
    )
    gathered = o_dense.index_select(0, rows)
    o3[:q_rows].copy_(gathered)


def flydsl_block_table(
    req_to_token: torch.Tensor,
    req_indices: torch.Tensor,
    page_size: int,
    max_context: int,
) -> torch.Tensor:
    """Page ids for logical tokens ``[0, max_context)``.

    ``req_to_token[req, t]`` is the pool slot of logical token ``t``. With
    page size P the slot at column ``k * P`` is the first token of page k,
    which is the same rule as ``AiterAttnBackend._transform_table_1_to_real``.
    The last page may be partial, so the table only needs that page's first
    column. A width of ``max_context`` is enough; the row does not have to
    be padded out to a page boundary.
    """
    if page_size not in (16, 64, 128):
        raise ValueError(f"FlyDSL page size must be 16, 64, or 128, got {page_size}")
    if max_context < 1:
        raise ValueError(f"max_context must be positive, got {max_context}")
    n_pages = (max_context + page_size - 1) // page_size
    last_start = (n_pages - 1) * page_size
    if req_to_token.shape[1] <= last_start:
        raise RuntimeError(
            f"req_to_token width {req_to_token.shape[1]} has no column "
            f"{last_start} for context {max_context} at page size {page_size}"
        )
    starts = torch.arange(n_pages, device=req_to_token.device, dtype=torch.long)
    starts = starts * page_size
    rows = req_to_token.index_select(
        0, req_indices.to(device=req_to_token.device, dtype=torch.long)
    )
    slots = rows.index_select(1, starts)
    # Divide before narrowing: a slot id can exceed int32 on a large pool.
    return (slots.to(dtype=torch.int64) // page_size).to(dtype=torch.int32).contiguous()

# --- mid-stack gap helpers (from G5 tip) ---

def flydsl_verify_width_groups(
    qo_vals: list[int],
    q_rows: int,
    pool_bs: int,
) -> tuple[int, list[tuple[int, list[int]]]]:
    """Live batch and ``(width, request positions)`` groups from ``qo_indptr``.

    The live batch is the first index whose prefix sum equals ``q_rows``.
    Requests that share a width are one ``pa_decode`` launch. The width is
    the indptr delta, never rounded up to the batch max: a 7-token request
    must not read an eighth KV token. Positions within a group stay in pool
    order. Groups are ordered by the first request of that width.
    """
    if q_rows < 1:
        raise ValueError(f"query rows must be positive, got {q_rows}")
    if pool_bs < 1:
        raise ValueError(f"verify pool must be positive, got {pool_bs}")
    if len(qo_vals) < 2:
        raise ValueError("verify qo_indptr is missing")
    limit = min(pool_bs, len(qo_vals) - 1)
    bs = None
    for i in range(1, limit + 1):
        if qo_vals[i] == q_rows:
            bs = i
            break
        if qo_vals[i] > q_rows:
            raise ValueError(
                f"qo_indptr jumps over {q_rows} query rows at request {i} "
                f"(qo={qo_vals[: i + 1]}, pool {pool_bs})"
            )
    if bs is None:
        raise ValueError(
            f"qo_indptr does not cover {q_rows} query rows "
            f"(pool {pool_bs}, prefix {qo_vals[: min(8, len(qo_vals))]})"
        )
    buckets: dict[int, list[int]] = {}
    order: list[int] = []
    for i in range(bs):
        width = qo_vals[i + 1] - qo_vals[i]
        if width < 1:
            raise ValueError(
                f"verify width at request {i} is {width} (qo={qo_vals[: bs + 1]})"
            )
        if width not in buckets:
            order.append(width)
            buckets[width] = []
        buckets[width].append(i)
    return bs, [(width, buckets[width]) for width in order]

def flydsl_group_row_indices(
    qo_vals: list[int], width: int, positions: list[int]
) -> list[int]:
    """Query-row ids for one width group. Not a contiguous slice when a
    different width sits between two members.
    """
    rows: list[int] = []
    for i in positions:
        start = qo_vals[i]
        rows.extend(range(start, start + width))
    return rows

