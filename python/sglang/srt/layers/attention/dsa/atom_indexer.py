# SPDX-License-Identifier: Apache-2.0
"""ATOM-aligned DSA indexer helpers (aiter cache, MQA logits top-k)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.utils import aiter_can_use_preshuffle_paged_mqa
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype, is_fp8_fnuz
from sglang.srt.utils import get_bool_env_var, is_hip

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_use_aiter_preshuffle = aiter_can_use_preshuffle_paged_mqa()
_is_fp8_fnuz = is_fp8_fnuz()


def use_atom_indexer_impl() -> bool:
    if envs.SGLANG_DSA_INDEXER_IMPL.get() != "atom":
        return False
    if not _use_aiter:
        return False
    return True


def supports_atom_qk_rope_fusion(
    head_dim: int,
    rope_head_dim: int,
    quant_block_size: int,
) -> bool:
    if not envs.SGLANG_DSA_INDEXER_QK_ROPE_FUSION.get():
        return False
    return (
        head_dim == quant_block_size
        and rope_head_dim == head_dim // 2
        and _use_aiter
    )


def atom_indexer_qk_rope_quant_and_cache(
    q_bf16: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    positions: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    k_norm_eps: float,
    quant_block_size: int,
    scale_fmt: Optional[str],
    weights_scale: float,
    is_neox_style: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused Q RoPE + Q FP8 quant + K norm/RoPE + K cache (ATOM path)."""
    from aiter import dtypes, indexer_qk_rope_quant_and_cache

    q_fp8 = torch.empty_like(q_bf16, dtype=dtypes.fp8)
    weights_out = torch.empty(weights.shape, device=weights.device, dtype=torch.float32)
    indexer_qk_rope_quant_and_cache(
        q_bf16,
        q_fp8,
        weights,
        weights_out,
        k,
        kv_cache,
        slot_mapping,
        k_norm_weight,
        k_norm_bias,
        positions,
        cos_cache,
        sin_cache,
        k_norm_eps,
        quant_block_size,
        scale_fmt,
        weights_scale,
        preshuffle=_use_aiter_preshuffle,
        is_neox=is_neox_style,
    )
    return q_fp8, weights_out


def atom_topk_decode_paged(
    logits: torch.Tensor,
    context_lens: torch.Tensor,
    topk_indices: torch.Tensor,
    next_n: int = 1,
) -> torch.Tensor:
    from aiter import top_k_per_row_decode

    num_rows = logits.shape[0]
    top_k_per_row_decode(
        logits,
        next_n,
        context_lens,
        topk_indices,
        num_rows,
        logits.stride(0),
        logits.stride(1),
    )
    return topk_indices


def atom_topk_prefill_ragged(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    from aiter import top_k_per_row_prefill

    num_rows = logits.shape[0]
    k = topk_indices.shape[1]
    top_k_per_row_prefill(
        logits,
        row_starts,
        row_ends,
        topk_indices,
        values=None,
        numRows=num_rows,
        stride0=logits.stride(0),
        stride1=logits.stride(1),
        k=k,
    )
    return topk_indices


def prepare_indexer_kv_cache_view(
    buf: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """View indexer K buffer as paged FP8 cache for aiter kernels."""
    if _is_fp8_fnuz:
        dtype = torch.float8_e4m3fnuz
    else:
        dtype = fp8_dtype
    return buf.view(-1, page_size, 132).view(dtype)
