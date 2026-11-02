"""Bidirectional DSpark draft attention on a SHUFFLE 5D KV pool.

The draft layers are ``ENCODER_ONLY``: every query in the block sees the
prefix and every key in the block. FlyDSL ``pa_decode`` with query length
greater than 1 is causal, so the block is packed into the GQA group and
launched at query length 1. One launch reads the pool. Prefill and any
extend that is not a short uniform block gather the prefix back to NHD
and keep Triton ``extend_attention_fwd``. The decode step does not pack.

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
    q: torch.Tensor, qlen: int, n_kv: int, n_q: int
) -> torch.Tensor:
    """``[B * qlen, n_q, D]`` to ``[B, n_q * qlen, D]``.

    Order inside each KV head is position, then the original group. Query
    length 1 is the identity permutation.
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
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    return x.view(batch, n_kv * qlen * group, dim)


def collapse_draft_outputs(
    out: torch.Tensor, qlen: int, n_kv: int, n_q: int
) -> torch.Tensor:
    """Inverse of :func:`expand_draft_queries`. Returns ``[B * qlen, n_q, D]``."""
    if out.shape[1] != n_q * qlen:
        raise ValueError(
            f"expanded heads {out.shape[1]} != n_q {n_q} * qlen {qlen}"
        )
    batch = out.shape[0]
    dim = out.shape[-1]
    group = n_q // n_kv
    x = out.view(batch, n_kv, qlen, group, dim)
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    return x.view(batch * qlen, n_q, dim)


def bucket_max_context(n: int) -> int:
    """Smallest power of two that is at least ``n`` and at least one page."""
    if n < 1:
        raise ValueError(f"context must be positive, got {n}")
    bucket = 16
    while bucket < n:
        bucket *= 2
    return bucket


def _refuse_capture() -> None:
    from sglang.srt.model_executor.runner import get_is_capture_mode

    capturing = False
    if get_is_capture_mode():
        capturing = True
    elif torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        capturing = True
    if capturing:
        raise RuntimeError(
            "SGLANG_DSPARK_DRAFT_FLYDSL does not capture CUDA graphs. "
            "Launch with --disable-decode-cuda-graph."
        )


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


def _context(backend, forward_batch, bs: int, qlen: int) -> Tuple[torch.Tensor, int]:
    prefix = forward_batch.extend_prefix_lens
    if prefix is None or int(prefix.shape[0]) < bs:
        raise RuntimeError(
            "SGLANG_DSPARK_DRAFT_FLYDSL uniform block needs extend_prefix_lens."
        )
    key = (_nonce(backend), bs, qlen)
    cached = getattr(backend, "_draft_flydsl_ctx", None)
    if cached is not None and cached[0] == key:
        return cached[1], cached[2]
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
    from sglang.srt.layers.attention.aiter_utils import flydsl_block_table

    _refuse_capture()
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
    context, spec = _context(backend, forward_batch, bs, qlen)
    req_indices = forward_batch.req_pool_indices[:bs]
    table = flydsl_block_table(
        backend.req_to_token, req_indices, backend.page_size, spec
    )
    q_exp = expand_draft_queries(q3, qlen, n_kv, n_q)
    o_exp = torch.empty_like(q_exp)
    if not getattr(backend, "_draft_flydsl_logged", False):
        logger.info(
            "DSPARK_DRAFT_FLYDSL uniform page=%s qlen=%s bs=%s q_heads=%s "
            "kv_heads=%s spec_max_context=%s",
            backend.page_size,
            qlen,
            bs,
            n_q * qlen,
            n_kv,
            spec,
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
    )
    flat = collapse_draft_outputs(o_exp, qlen, n_kv, n_q)
    o.view(-1, n_q, layer.v_head_dim).copy_(flat)
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
    qlen = _uniform_qlen_cached(backend, forward_batch, bs)
    q_rows = q.reshape(-1, int(layer.tp_q_head_num), int(layer.qk_head_dim)).shape[0]
    if (
        qlen is not None
        and q_rows == bs * qlen
        and forward_batch.extend_prefix_lens is not None
        and backend.forward_metadata.custom_mask is None
    ):
        return _launch_uniform(backend, q, o, layer, forward_batch, qlen)
    return _launch_gathered_extend(
        backend, q, o, k, v, layer, forward_batch, causal
    )
