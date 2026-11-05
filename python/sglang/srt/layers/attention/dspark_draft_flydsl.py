"""Bidirectional DSpark draft attention on a SHUFFLE 5D KV pool.

The draft layers are ``ENCODER_ONLY``: every query in the block sees the
prefix and every key in the block. FlyDSL ``pa_decode`` with query length
greater than 1 is causal, so the block is packed into the GQA group and
launched at query length 1. One launch reads the pool. A short mixed
block uses that same launch whether or not a CUDA graph is being
captured. Prefill, which has more rows than the draft cap, still gathers
the prefix back to NHD and keeps Triton ``extend_attention_fwd``. The
decode step does not pack.

``max_context_length`` specializes ``pa_decode``. It is rounded up to a
power of two so a book whose prefixes sit in one bucket does not compile
again on every new maximum. Per-row ``context_lengths`` stay exact.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# A prefill chunk is far above this. The draft block is 7 (window 8).
_MAX_UNIFORM_QLEN = 16


def expand_draft_queries(
    q: torch.Tensor,
    qlen: int,
    n_kv: int,
    n_q: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``[B * qlen, n_q, D]`` to ``[B, n_q * qlen, D]``.

    Order inside each KV head is position, then the original group. Query
    length 1 is the identity permutation. ``out`` is a preallocated
    destination so a CUDA graph does not allocate the pack buffer.
    """
    if qlen < 1:
        raise ValueError(f"qlen must be positive, got {qlen}")
    if n_q % n_kv != 0:
        raise ValueError(f"n_q {n_q} is not divisible by n_kv {n_kv}")
    if q.shape[0] % qlen != 0:
        raise ValueError(f"query rows {q.shape[0]} are not a multiple of qlen {qlen}")
    batch = q.shape[0] // qlen
    dim = q.shape[-1]
    group = n_q // n_kv
    x = q.view(batch, qlen, n_kv, group, dim)
    flat = x.permute(0, 2, 1, 3, 4).reshape(batch, n_kv * qlen * group, dim)
    if out is None:
        return flat.contiguous()
    if out.shape != flat.shape:
        raise ValueError(
            f"expand destination {tuple(out.shape)} != {tuple(flat.shape)}"
        )
    out.copy_(flat)
    return out


def collapse_draft_outputs(
    out: torch.Tensor,
    qlen: int,
    n_kv: int,
    n_q: int,
    dest: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Inverse of :func:`expand_draft_queries`. Returns ``[B * qlen, n_q, D]``.

    ``dest`` is a preallocated ``[B * qlen, n_q, D]`` buffer.
    """
    if out.shape[1] != n_q * qlen:
        raise ValueError(
            f"expanded heads {out.shape[1]} != n_q {n_q} * qlen {qlen}"
        )
    batch = out.shape[0]
    dim = out.shape[-1]
    group = n_q // n_kv
    x = out.view(batch, n_kv, qlen, group, dim)
    flat = x.permute(0, 2, 1, 3, 4).reshape(batch * qlen, n_q, dim)
    if dest is None:
        return flat.contiguous()
    if dest.shape != flat.shape:
        raise ValueError(
            f"collapse destination {tuple(dest.shape)} != {tuple(flat.shape)}"
        )
    dest.copy_(flat)
    return dest


def bucket_max_context(n: int) -> int:
    """Smallest power of two that is at least ``n`` and at least one page."""
    if n < 1:
        raise ValueError(f"context must be positive, got {n}")
    bucket = 16
    while bucket < n:
        bucket *= 2
    return bucket


def _refuse_capture() -> None:
    """Gathered prefill still syncs. The uniform block does not call this."""
    from sglang.srt.layers.attention.aiter_utils import flydsl_stream_capturing
    from sglang.srt.model_executor.runner import get_is_capture_mode

    if flydsl_stream_capturing() or get_is_capture_mode():
        raise RuntimeError(
            "SGLANG_DSPARK_DRAFT_FLYDSL prefill does not capture CUDA graphs. "
            "The uniform draft block captures; this gather path still syncs. "
            "Launch with --disable-decode-cuda-graph."
        )


def _draft_graph_bound() -> bool:
    """True for a full-graph warmup and for the stream capture itself.

    Both must specialize ``pa_decode`` on the same host max-context. A
    breakable eager gap keeps the measured length.
    """
    from sglang.srt.layers.attention.aiter_utils import (
        flydsl_stream_capturing,
        flydsl_use_graph_bound,
    )
    from sglang.srt.model_executor.runner import get_is_capture_mode
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
        is_in_breakable_cuda_graph,
    )

    return flydsl_use_graph_bound(
        stream_capturing=flydsl_stream_capturing(),
        capture_mode=bool(get_is_capture_mode()),
        breakable=bool(is_in_breakable_cuda_graph()),
    )


def _capture_max_context(backend) -> int:
    """Host bound baked into a captured draft ``pa_decode``.

    Replay does not re-enter this function. Every context length that
    will run must be at most this value.
    """
    cached = getattr(backend, "_draft_flydsl_capture_max", None)
    if cached is not None:
        return int(cached)
    bound = int(getattr(backend, "max_context_len", 0) or 0)
    table = getattr(backend, "req_to_token", None)
    if table is not None and getattr(table, "ndim", 0) >= 2:
        width = int(table.shape[1])
        bound = width if bound < 1 else min(bound, width)
    if bound < 1:
        raise RuntimeError(
            "DSPARK draft FlyDSL capture max_context is not positive. "
            "The model context length has to be set before capture."
        )
    backend._draft_flydsl_capture_max = bound
    return bound


def _expand_bufs(backend, batch: int, expanded_heads: int, dim: int, like: torch.Tensor):
    """Pack and output buffers for one captured batch. Not evicted."""
    from sglang.srt.layers.attention.aiter_utils import flydsl_note_scratch

    key = (batch, expanded_heads, dim, like.dtype, like.device)
    cache = getattr(backend, "_draft_flydsl_expand", None)
    pinned = getattr(backend, "_draft_flydsl_expand_pinned", None)
    if cache is None:
        cache = {}
        backend._draft_flydsl_expand = cache
    if pinned is None:
        pinned = {}
        backend._draft_flydsl_expand_pinned = pinned
    bufs = cache.get(key)
    if bufs is None:
        shape = (batch, expanded_heads, dim)
        bufs = (
            torch.empty(shape, dtype=like.dtype, device=like.device),
            torch.empty(shape, dtype=like.dtype, device=like.device),
        )
    flydsl_note_scratch(cache, pinned, key, bufs, True)
    return bufs


def _nonce(backend) -> int:
    return int(getattr(backend, "_draft_flydsl_nonce", 0))


def _uniform_qlen_cached(backend, forward_batch, bs: int) -> Optional[int]:
    nonce = _nonce(backend)
    cached = getattr(backend, "_draft_flydsl_qlen", None)
    if cached is not None and cached[0] == nonce:
        return cached[1]
    qlen = _uniform_qlen(forward_batch.extend_seq_lens, bs)
    backend._draft_flydsl_qlen = (nonce, qlen)
    return qlen


def _uniform_qlen(extend_seq_lens: Optional[torch.Tensor], bs: int) -> Optional[int]:
    if extend_seq_lens is None or bs < 1:
        return None
    if int(extend_seq_lens.shape[0]) < bs:
        return None
    first = int(extend_seq_lens[0].item())
    if first < 1 or first > _MAX_UNIFORM_QLEN:
        return None
    rows = extend_seq_lens[:bs]
    if not bool(torch.equal(rows, rows.new_full(rows.shape, first))):
        return None
    return first


def draft_left_align_rows(
    qo_indptr: torch.Tensor,
    q_rows: int,
    max_q: int,
) -> torch.Tensor:
    """Map packed query rows onto a dense ``[batch, max_q]`` window.

    Token ``t`` of request ``i`` lands at ``i * max_q + t``. The draft
    block is bidirectional: those slots are packed into the GQA group and
    launched at query length 1, so every real query sees the whole real
    block. This is not the causal right-align used by target verify
    (``flydsl_ragged_dense_rows``), which would put token ``t`` at
    ``i * max_q + (max_q - width + t)``.
    """
    if q_rows < 1:
        raise ValueError(f"query rows must be positive, got {q_rows}")
    if max_q < 1:
        raise ValueError(f"max_q must be positive, got {max_q}")
    bs = int(qo_indptr.shape[0]) - 1
    if bs < 1:
        raise ValueError(f"draft batch must be positive, got {bs}")
    ends = qo_indptr[1 : bs + 1]
    token_ids = torch.arange(q_rows, device=qo_indptr.device, dtype=torch.int32)
    req = torch.searchsorted(ends, token_ids, right=True).to(torch.long)
    starts = qo_indptr[:bs].index_select(0, req)
    offset = token_ids.to(torch.long) - starts.to(torch.long)
    return req * max_q + offset


def _draft_width_buf(backend, bs: int, device) -> torch.Tensor:
    cache = getattr(backend, "_draft_flydsl_width_bufs", None)
    pinned = getattr(backend, "_draft_flydsl_width_pinned", None)
    if cache is None:
        cache = {}
        backend._draft_flydsl_width_bufs = cache
    if pinned is None:
        pinned = {}
        backend._draft_flydsl_width_pinned = pinned
    key = (bs, str(device))
    buf = cache.get(key)
    if buf is None:
        buf = torch.empty((bs,), dtype=torch.int32, device=device)
    from sglang.srt.layers.attention.aiter_utils import flydsl_note_scratch

    flydsl_note_scratch(cache, pinned, key, buf, _draft_graph_bound())
    return buf


def _draft_qo_buf(backend, bs: int, device) -> torch.Tensor:
    """``bs + 1`` indptr. Allocated outside capture, filled in place."""
    cache = getattr(backend, "_draft_flydsl_qo_bufs", None)
    pinned = getattr(backend, "_draft_flydsl_qo_pinned", None)
    if cache is None:
        cache = {}
        backend._draft_flydsl_qo_bufs = cache
    if pinned is None:
        pinned = {}
        backend._draft_flydsl_qo_pinned = pinned
    key = (bs, str(device))
    buf = cache.get(key)
    if buf is None:
        buf = torch.empty((bs + 1,), dtype=torch.int32, device=device)
    from sglang.srt.layers.attention.aiter_utils import flydsl_note_scratch

    flydsl_note_scratch(cache, pinned, key, buf, _draft_graph_bound())
    return buf


def _draft_slot_bufs(backend, rows: int, heads: int, dim: int, like: torch.Tensor):
    from sglang.srt.layers.attention.aiter_utils import flydsl_note_scratch

    key = (rows, heads, dim, like.dtype, str(like.device))
    cache = getattr(backend, "_draft_flydsl_slot_bufs", None)
    pinned = getattr(backend, "_draft_flydsl_slot_pinned", None)
    if cache is None:
        cache = {}
        backend._draft_flydsl_slot_bufs = cache
    if pinned is None:
        pinned = {}
        backend._draft_flydsl_slot_pinned = pinned
    bufs = cache.get(key)
    if bufs is None:
        bufs = (
            torch.empty((rows, heads, dim), dtype=like.dtype, device=like.device),
            torch.empty((rows, heads, dim), dtype=like.dtype, device=like.device),
        )
    flydsl_note_scratch(cache, pinned, key, bufs, _draft_graph_bound())
    return bufs


def _mixed_draft_widths(forward_batch, bs: int):
    """Per-request draft widths already on the device, or None.

    A ragged verify layout wins. Otherwise the staged extend lengths.
    This does not read a length back to the host.
    """
    spec = getattr(forward_batch, "spec_info", None)
    layout = getattr(spec, "ragged_verify_layout", None) if spec is not None else None
    if layout is not None and int(layout.verify_lens.shape[0]) >= bs:
        return layout.verify_lens[:bs], layout.qo_indptr_device
    for name in ("extend_seq_lens",):
        tensor = getattr(forward_batch, name, None)
        if tensor is not None and int(tensor.shape[0]) >= bs:
            return tensor[:bs], None
    if spec is not None:
        tensor = getattr(spec, "extend_seq_lens_tensor", None)
        if tensor is not None and int(tensor.shape[0]) >= bs:
            return tensor[:bs], None
    return None, None


def _launch_ragged_bidirectional(
    backend,
    q: torch.Tensor,
    o: torch.Tensor,
    layer,
    forward_batch,
    widths: torch.Tensor,
    qo_indptr: Optional[torch.Tensor],
    max_q: int,
) -> torch.Tensor:
    """One query-length-1 ``pa_decode`` for a mixed-width draft block.

    Real tokens are left-aligned in a ``max_q`` window and packed into the
    GQA group, the same way the uniform block becomes bidirectional.
    Context is ``prefix + width``, not ``prefix + max_q``, so a short
    request does not read a padding key. Padding query slots stay zero
    and their outputs are dropped.
    """
    from sglang.srt.layers.attention.aiter_utils import (
        flydsl_block_table,
        flydsl_stream_capturing,
    )

    capturing = flydsl_stream_capturing()
    n_q = int(layer.tp_q_head_num)
    n_kv = int(layer.tp_k_head_num)
    dim = int(layer.qk_head_dim)
    q3 = q.reshape(-1, n_q, dim)
    q_rows = int(q3.shape[0])
    bs = int(widths.shape[0])
    if not capturing:
        widest = int(widths.max().item())
        if widest > max_q or widest < 1:
            raise RuntimeError(
                f"draft ragged width {widest} is outside 1..{max_q}"
            )
        if qo_indptr is not None and int(qo_indptr[bs].item()) != q_rows:
            raise RuntimeError(
                f"draft qo_indptr[{bs}] != query rows {q_rows}"
            )
    k_cache, v_cache = _caches(backend, layer)
    width_buf = _draft_width_buf(backend, bs, q3.device)
    src_w = widths if widths.dtype == torch.int32 else widths.to(dtype=torch.int32)
    width_buf.copy_(src_w)
    if qo_indptr is None:
        qo = _draft_qo_buf(backend, bs, q3.device)
        # A Python scalar store syncs during stream capture. zero_ is a kernel.
        qo.zero_()
        torch.cumsum(width_buf, dim=0, out=qo[1 : bs + 1])
    else:
        qo = qo_indptr
        if qo.dtype != torch.int32:
            staged = _draft_qo_buf(backend, bs, q3.device)
            staged.copy_(qo[: bs + 1].to(dtype=torch.int32))
            qo = staged
    prefix = forward_batch.extend_prefix_lens
    if prefix is None:
        prefix = forward_batch.seq_lens
    context = _static_context_buf(backend, bs, prefix.device)
    src_p = prefix[:bs]
    if src_p.dtype != torch.int32:
        src_p = src_p.to(dtype=torch.int32)
    context.copy_(src_p)
    context.add_(width_buf)
    rows = draft_left_align_rows(qo, q_rows, max_q)
    q_slot, o_slot = _draft_slot_bufs(backend, bs * max_q, n_q, dim, q3)
    q_slot.zero_()
    q_slot.index_copy_(0, rows, q3[:q_rows])
    q_exp, o_exp = _expand_bufs(backend, bs, n_q * max_q, dim, q3)
    expand_draft_queries(q_slot, max_q, n_kv, n_q, out=q_exp)
    spec = _capture_max_context(backend)
    req_indices = forward_batch.req_pool_indices[:bs]
    table = flydsl_block_table(
        backend.req_to_token, req_indices, backend.page_size, spec
    )
    # Same batch coupling as target verify: query_length 1 still moves
    # sequence 0 when another row has a long context
    # (decode-step-batch-bisect/bisect.json, qlen1_bs4_long: max abs
    # 0.0625, 494 elements). Context-0 padding does not. One batch-1
    # launch per capture slot matches the eager gather, which is one
    # sequence at a time. The loop bound is the captured batch.
    runner = _runner(backend)
    for slot in range(bs):
        runner.run(
            o_exp[slot : slot + 1],
            q_exp[slot : slot + 1],
            k_cache,
            v_cache,
            context[slot : slot + 1],
            table[slot : slot + 1],
            float(layer.scaling),
            1,
            spec,
            pin_scratch=True,
        )
    collapse_draft_outputs(o_exp, max_q, n_kv, n_q, dest=o_slot)
    o_view = o.view(-1, n_q, layer.v_head_dim)
    o_view[:q_rows].copy_(o_slot.index_select(0, rows))
    return o


def _caches(backend, layer):
    k_cache, v_cache = backend.kv_layers.read(layer.layer_id).kv()
    if k_cache.dim() != 5 or v_cache.dim() != 5:
        raise RuntimeError(
            "SGLANG_DSPARK_DRAFT_FLYDSL requires the draft KV pool to be "
            "SHUFFLE 5D. The pool builder selects that layout when the flag "
            f"is set. Got K {tuple(k_cache.shape)} V {tuple(v_cache.shape)}."
        )
    return k_cache, v_cache


def _runner(backend):
    from sglang.srt.layers.attention.aiter_utils import FlydslPaDecode

    runner = getattr(backend, "_draft_flydsl_runner", None)
    if runner is None:
        runner = FlydslPaDecode()
        backend._draft_flydsl_runner = runner
    return runner


def _static_context_buf(backend, bs: int, device) -> torch.Tensor:
    """One int32 context buffer per captured batch. Allocated outside capture."""
    cache = getattr(backend, "_draft_flydsl_ctx_bufs", None)
    if cache is None:
        cache = {}
        backend._draft_flydsl_ctx_bufs = cache
    key = (bs, str(device))
    buf = cache.get(key)
    if buf is None:
        buf = torch.empty((bs,), dtype=torch.int32, device=device)
        cache[key] = buf
    return buf


def _context(
    backend, forward_batch, bs: int, qlen: int, graph_bound: bool, capturing: bool
) -> Tuple[torch.Tensor, int]:
    # The hand-built DSPARK block stores the prefix in seq_lens and does not
    # set extend_prefix_lens. The qlen new tokens are already in the pool.
    # A tensor allocated inside capture is not a stable replay address. The
    # graph-bound path fills a buffer created on the warmup, in place.
    prefix = forward_batch.extend_prefix_lens
    if prefix is None:
        prefix = forward_batch.seq_lens
    if prefix is None or int(prefix.shape[0]) < bs:
        raise RuntimeError(
            "SGLANG_DSPARK_DRAFT_FLYDSL uniform block needs extend_prefix_lens "
            "or seq_lens."
        )
    key = (_nonce(backend), bs, qlen, bool(graph_bound), bool(capturing))
    cached = getattr(backend, "_draft_flydsl_ctx", None)
    if cached is not None and cached[0] == key:
        return cached[1], cached[2]
    if graph_bound:
        context = _static_context_buf(backend, bs, prefix.device)
        src = prefix[:bs]
        if src.dtype != torch.int32:
            src = src.to(dtype=torch.int32)
        context.copy_(src)
        context.add_(qlen)
        spec = _capture_max_context(backend)
    else:
        context = (prefix[:bs].to(dtype=torch.int32) + qlen).contiguous()
        true_max = int(context.max().item())
        if true_max < qlen:
            raise RuntimeError(
                f"draft context {true_max} is shorter than the block {qlen}"
            )
        spec = bucket_max_context(true_max)
    backend._draft_flydsl_ctx = (key, context, spec)
    return context, spec


def _launch_uniform(
    backend,
    q: torch.Tensor,
    o: torch.Tensor,
    layer,
    forward_batch,
    qlen: int,
) -> torch.Tensor:
    from sglang.srt.layers.attention.aiter_utils import (
        flydsl_block_table,
        flydsl_stream_capturing,
    )

    capturing = flydsl_stream_capturing()
    graph_bound = _draft_graph_bound()
    n_q = int(layer.tp_q_head_num)
    n_kv = int(layer.tp_k_head_num)
    dim = int(layer.qk_head_dim)
    q3 = q.reshape(-1, n_q, dim)
    bs = q3.shape[0] // qlen
    k_cache, v_cache = _caches(backend, layer)
    if int(k_cache.shape[1]) != n_kv:
        raise RuntimeError(
            f"draft KV heads in the pool ({int(k_cache.shape[1])}) "
            f"!= layer tp_k_head_num {n_kv}"
        )
    context, spec = _context(
        backend, forward_batch, bs, qlen, graph_bound, capturing
    )
    req_indices = forward_batch.req_pool_indices[:bs]
    table = flydsl_block_table(
        backend.req_to_token, req_indices, backend.page_size, spec
    )
    if graph_bound:
        # Warmup allocates these. Capture copies into the same storage.
        q_exp, o_exp = _expand_bufs(backend, bs, n_q * qlen, dim, q3)
        expand_draft_queries(q3, qlen, n_kv, n_q, out=q_exp)
    else:
        q_exp = expand_draft_queries(q3, qlen, n_kv, n_q)
        o_exp = torch.empty_like(q_exp)
    if not capturing and not getattr(backend, "_draft_flydsl_logged", False):
        prefix0 = int(context[0].item()) - qlen
        table_slot = int(
            backend.req_to_token[int(req_indices[0].item()), prefix0 + qlen - 1].item()
        )
        tail = int(forward_batch.out_cache_loc[qlen - 1].item())
        logger.info(
            "DSPARK_DRAFT_FLYDSL uniform page=%s qlen=%s bs=%s q_heads=%s "
            "kv_heads=%s spec_max_context=%s prefix0=%s table_slot=%s "
            "out_cache_tail=%s graph_bound=%s",
            backend.page_size,
            qlen,
            bs,
            n_q * qlen,
            n_kv,
            spec,
            prefix0,
            table_slot,
            tail,
            graph_bound,
        )
        backend._draft_flydsl_logged = True
    _runner(backend).run(
        o_exp,
        q_exp,
        k_cache,
        v_cache,
        context,
        table,
        float(layer.scaling),
        1,
        spec,
        pin_scratch=graph_bound,
    )
    o_view = o.view(-1, n_q, layer.v_head_dim)
    if graph_bound:
        collapse_draft_outputs(o_exp, qlen, n_kv, n_q, dest=o_view)
    else:
        flat = collapse_draft_outputs(o_exp, qlen, n_kv, n_q)
        o_view.copy_(flat)
    return o


def _gather_prefix(k_cache, v_cache, kv_indptr, kv_indices, bs: int):
    from sglang.kernels.ops.kvcache.shuffle_5d_cache import (
        launch_gather_shuffle_5d_to_linear,
    )

    n = int(kv_indptr[bs].item())
    heads = int(k_cache.shape[1])
    head_dim = int(k_cache.shape[2]) * int(k_cache.shape[-1])
    if n <= 0:
        zeros = k_cache.new_zeros((1, heads, head_dim))
        empty = kv_indices.new_empty((0,))
        return zeros, zeros, empty
    slots = kv_indices[:n].to(dtype=torch.int64).contiguous()
    k_lin, v_lin = launch_gather_shuffle_5d_to_linear(k_cache, v_cache, slots)
    idx = torch.arange(n, device=kv_indices.device, dtype=kv_indices.dtype)
    return k_lin, v_lin, idx


def _launch_gathered_extend(
    backend,
    q: torch.Tensor,
    o: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer,
    forward_batch,
    causal: bool,
) -> torch.Tensor:
    """Prefix gathered to NHD. Used for prefill, not for the uniform block."""
    _refuse_capture()
    md = backend.forward_metadata
    bs = int(forward_batch.batch_size)
    k_cache, v_cache = _caches(backend, layer)
    kv_indptr = md.kv_indptr
    kv_indices = md.kv_indices
    if kv_indptr is None or kv_indices is None:
        raise RuntimeError(
            "SGLANG_DSPARK_DRAFT_FLYDSL prefill is missing kv_indptr."
        )
    k_lin, v_lin, idx = _gather_prefix(k_cache, v_cache, kv_indptr, kv_indices, bs)
    if not getattr(backend, "_draft_flydsl_gather_logged", False):
        logger.info(
            "DSPARK_DRAFT_FLYDSL gather-prefix page=%s bs=%s prefix_tokens=%s "
            "(prefill only; the uniform block does not pack)",
            backend.page_size,
            bs,
            int(idx.shape[0]),
        )
        backend._draft_flydsl_gather_logged = True
    n_q = int(layer.tp_q_head_num)
    backend.extend_attention_fwd(
        q.view(-1, n_q, layer.qk_head_dim),
        k.contiguous(),
        v.contiguous(),
        o.view(-1, n_q, layer.v_head_dim),
        k_lin,
        v_lin,
        md.qo_indptr,
        kv_indptr,
        idx,
        md.custom_mask,
        causal,
        md.mask_indptr,
        md.max_extend_len,
        1.0,
        1.0,
        sm_scale=layer.scaling,
        logit_cap=0.0,
        sliding_window_size=-1,
        page_size=1,
        extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
    )
    return o


def _resolve_block_qlen(backend, forward_batch, bs: int, q_rows: int) -> Optional[int]:
    """Width of a uniform DSPARK block, or None when this extend is prefill.

    The hand-built block batch leaves ``extend_seq_lens`` empty. Its spec
    info carries ``draft_token_num`` (7 on this model) and ``seq_lens`` is
    the prefix. A prefill chunk is longer than ``_MAX_UNIFORM_QLEN`` or is
    not ``TARGET_VERIFY``, so it stays on the gather path.
    """
    from sglang.srt.layers.attention.aiter_utils import flydsl_stream_capturing

    # extend_seq_lens[0].item() syncs. A capturing stream uses the host
    # draft_token_num on the spec info instead.
    if not flydsl_stream_capturing():
        cached = _uniform_qlen_cached(backend, forward_batch, bs)
        if cached is not None and bs > 0 and q_rows == bs * cached:
            return cached
    if bs < 1 or not forward_batch.forward_mode.is_target_verify():
        return None
    spec = forward_batch.spec_info
    width = getattr(spec, "draft_token_num", None) if spec is not None else None
    if width is None:
        return None
    width = int(width)
    if width < 1 or width > _MAX_UNIFORM_QLEN:
        return None
    if q_rows != bs * width:
        return None
    return width


def forward_draft_extend(
    backend,
    q: torch.Tensor,
    o: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer,
    forward_batch,
    causal: bool,
) -> torch.Tensor:
    """Draft extend on the SHUFFLE pool. ``k`` and ``v`` are already stored."""
    if causal:
        raise RuntimeError(
            "SGLANG_DSPARK_DRAFT_FLYDSL is the bidirectional draft mask. "
            f"Layer {layer.layer_id} was causal."
        )
    if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
        raise RuntimeError(
            "SGLANG_DSPARK_DRAFT_FLYDSL does not apply a sliding window "
            f"(layer {layer.layer_id} has "
            f"sliding_window_size={layer.sliding_window_size})."
        )
    if getattr(layer, "logit_cap", 0):
        raise RuntimeError(
            "SGLANG_DSPARK_DRAFT_FLYDSL does not apply a logit cap."
        )
    if backend.page_size != 16:
        raise RuntimeError(
            "SGLANG_DSPARK_DRAFT_FLYDSL requires page_size=16 "
            f"(got {backend.page_size})."
        )
    if layer.qk_head_dim != layer.v_head_dim:
        raise RuntimeError(
            "SGLANG_DSPARK_DRAFT_FLYDSL requires qk_head_dim == v_head_dim, "
            f"got {layer.qk_head_dim} and {layer.v_head_dim}."
        )
    bs = int(forward_batch.batch_size)
    q_rows = q.reshape(-1, int(layer.tp_q_head_num), int(layer.qk_head_dim)).shape[0]
    qlen = _resolve_block_qlen(backend, forward_batch, bs, q_rows)
    mask = backend.forward_metadata.custom_mask
    prefix = forward_batch.extend_prefix_lens
    if prefix is None and forward_batch.forward_mode.is_target_verify():
        prefix = forward_batch.seq_lens
    if (
        qlen is not None
        and q_rows == bs * qlen
        and prefix is not None
        and mask is None
    ):
        return _launch_uniform(backend, q, o, layer, forward_batch, qlen)
    # Mixed short blocks. pa_decode with query length > 1 is causal, and
    # this block is bidirectional, so pad to the host draft width, pack
    # into the GQA group, and launch at query length 1. Eager used to
    # fall through to the gather. That path reads kv_indptr, which the
    # FlyDSL flush does not own, and the second eager 8-wide shifted
    # short-7 while this launch stayed put. Prefill (more rows than the
    # cap) still gathers. Graphs warmup and capture already take this
    # branch; dropping the graph-bound check does not change it.
    spec = forward_batch.spec_info
    max_q = getattr(spec, "draft_token_num", None) if spec is not None else None
    max_q = int(max_q or 0)
    widths, staged_qo = _mixed_draft_widths(forward_batch, bs)
    if (
        mask is None
        and prefix is not None
        and widths is not None
        and 1 <= max_q <= _MAX_UNIFORM_QLEN
        and bs <= q_rows <= bs * max_q
    ):
        return _launch_ragged_bidirectional(
            backend,
            q,
            o,
            layer,
            forward_batch,
            widths,
            staged_qo,
            max_q,
        )
    if not getattr(backend, "_draft_flydsl_skip_logged", False):
        backend._draft_flydsl_skip_logged = True
        logger.info(
            "DSPARK_DRAFT_FLYDSL skip-uniform qlen=%s q_rows=%s bs=%s "
            "has_prefix=%s mode=%s mask=%s",
            qlen,
            q_rows,
            bs,
            prefix is not None,
            forward_batch.forward_mode,
            mask is not None,
        )
    return _launch_gathered_extend(
        backend, q, o, k, v, layer, forward_batch, causal
    )
