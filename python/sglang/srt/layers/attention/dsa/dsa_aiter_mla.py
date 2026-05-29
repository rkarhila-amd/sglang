# SPDX-License-Identifier: Apache-2.0
"""Aiter MLA metadata helpers for DSA sparse attention with FP8 KV cache."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention


def init_dsa_aiter_mla_buffers(
    device: torch.device,
    max_bs: int,
    num_heads_padded: int,
    kv_cache_dtype: torch.dtype,
    max_split_per_batch: int = 32,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    from aiter import get_mla_metadata_info_v1

    fast_mode = True
    intra_batch_mode = False
    (
        (work_meta_data_size, work_meta_data_type),
        (work_indptr_size, work_indptr_type),
        (work_info_set_size, work_info_set_type),
        (reduce_indptr_size, reduce_indptr_type),
        (reduce_final_map_size, reduce_final_map_type),
        (reduce_partial_map_size, reduce_partial_map_type),
    ) = get_mla_metadata_info_v1(
        max_bs,
        1,
        num_heads_padded,
        kv_cache_dtype,
        kv_cache_dtype,
        is_sparse=False,
        fast_mode=fast_mode,
        num_kv_splits=max_split_per_batch,
        intra_batch_mode=intra_batch_mode,
    )
    return (
        torch.empty(work_meta_data_size, dtype=work_meta_data_type, device=device),
        torch.empty(work_indptr_size, dtype=work_indptr_type, device=device),
        torch.empty(work_info_set_size, dtype=work_info_set_type, device=device),
        torch.empty(reduce_indptr_size, dtype=reduce_indptr_type, device=device),
        torch.empty(reduce_final_map_size, dtype=reduce_final_map_type, device=device),
        torch.empty(
            reduce_partial_map_size, dtype=reduce_partial_map_type, device=device
        ),
    )


def fill_dsa_aiter_mla_metadata(
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    work_metadata: torch.Tensor,
    work_info_set: torch.Tensor,
    work_indptr: torch.Tensor,
    reduce_indptr: torch.Tensor,
    reduce_final_map: torch.Tensor,
    reduce_partial_map: torch.Tensor,
    num_heads_padded: int,
    kv_cache_dtype: torch.dtype,
    max_q_len: int,
    max_split_per_batch: int = 32,
) -> None:
    from aiter import get_mla_metadata_v1

    get_mla_metadata_v1(
        qo_indptr,
        kv_indptr,
        kv_last_page_len,
        num_heads_padded,
        1,
        False,
        work_metadata,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        max_q_len,
        fast_mode=True,
        is_sparse=False,
        max_split_per_batch=max_split_per_batch,
        intra_batch_mode=False,
    )


def mla_decode_fwd_fp8_sparse(
    q_kernel: torch.Tensor,
    kv_cache: torch.Tensor,
    o_kernel: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    max_seqlen_q: int,
    layer: "RadixAttention",
    q_scale: torch.Tensor,
    kv_scale: torch.Tensor,
    work_metadata: torch.Tensor,
    work_indptr: torch.Tensor,
    work_info_set: torch.Tensor,
    reduce_indptr: torch.Tensor,
    reduce_final_map: torch.Tensor,
    reduce_partial_map: torch.Tensor,
    num_kv_splits: int = 32,
) -> None:
    from aiter.mla import mla_decode_fwd

    mla_decode_fwd(
        q_kernel,
        kv_cache,
        o_kernel,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        max_seqlen_q,
        sm_scale=layer.scaling,
        logit_cap=layer.logit_cap,
        q_scale=q_scale,
        kv_scale=kv_scale,
        work_meta_data=work_metadata,
        work_indptr=work_indptr,
        work_info_set=work_info_set,
        reduce_indptr=reduce_indptr,
        reduce_final_map=reduce_final_map,
        reduce_partial_map=reduce_partial_map,
        num_kv_splits=num_kv_splits,
    )


def get_fp8_kv_scales(layer: "RadixAttention", backend_k_scale: Optional[torch.Tensor]):
    k_scale = getattr(layer, "k_scale", None)
    if k_scale is None:
        k_scale = backend_k_scale
    if k_scale is None and backend_k_scale is None:
        return None, None
    if k_scale is None:
        k_scale = backend_k_scale
    return k_scale, k_scale
