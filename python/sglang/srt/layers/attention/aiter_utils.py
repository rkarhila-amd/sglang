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
        q_local = q.to(fp8_dtype)
        q_descale_local = (
            layer.k_scale if layer.k_scale is not None else backend.k_scale
        )
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
        self._scratch = None
        self._scratch_key = None

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
            out = torch.empty(
                query.shape, dtype=query.dtype, device=query.device
            )

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
                torch.empty(
                    *scalar, head_dim, dtype=query.dtype, device=query.device
                ),
            )
            self._scratch_key = scratch_key
        exp_sums, max_logits, temporary = self._scratch

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
