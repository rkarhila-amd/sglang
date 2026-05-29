# SPDX-License-Identifier: Apache-2.0
"""ATOM-aligned Triton helpers for DSA indexer index conversion (ported from ATOM attention_mla)."""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

@triton.jit
def _convert_req_index_to_global_index_kernel(
    qo_indptr,  # int32 [num_requests]
    kv_indptr,  # int32 [num_requests+1]
    page_kv_indptr,  # int32 [num_requests+1]
    kv_indices,  # int32 [num_requests * max_num_blocks_per_req]
    token_indices_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    out_kv_indices,  # int32
    # shapes (compile-time where possible)
    NUM_TOPK_TOKENS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,  # tile width along columns
    # strides (in elements)
    ti_stride0,
    ti_stride1,
):
    # program_id(0) -> batch_id (row)
    # program_id(1) -> tile index along columns
    batch_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    # Each program covers BLOCK_N consecutive columns
    indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load request id for this token (no mask: grid is exact)
    kv_start = tl.load(kv_indptr + batch_id)
    kv_end = tl.load(kv_indptr + batch_id + 1)
    out_kv_start = tl.load(page_kv_indptr + batch_id)
    kv_len = kv_end - kv_start
    qo_start = tl.load(qo_indptr + batch_id)
    qo_end = tl.load(qo_indptr + batch_id + 1)

    for token_id in range(qo_start, qo_end):
        # Load token indices for this tile
        ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
        tok = tl.load(ti_ptr)  # int32

        # Guard block_table access
        valid_mask = (indice_id < kv_len) & (indice_id < NUM_TOPK_TOKENS)
        out_val = tl.load(
            kv_indices + kv_start + tok,
            mask=valid_mask,
            other=0,
        )

        # Store results
        out_ptr_ij = out_kv_indices + out_kv_start + indice_id
        tl.store(
            out_ptr_ij,
            out_val,
            mask=valid_mask,
        )


def triton_convert_req_index_to_global_index(
    qo_indptr: torch.Tensor,  # int32 [num_tokens + 1]
    kv_indptr: torch.Tensor,  # int32 [num_tokens + 1]
    page_kv_indptr: torch.Tensor,  # int32 [num_tokens + 1]
    kv_indices: torch.Tensor,  # int32 [total_kv_seqlen]
    token_indices: torch.Tensor,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    BLOCK_SIZE: int = 1,  # page_block_size = 1 for now
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 128,  # tile width along columns
    out: Optional[torch.Tensor] = None,
):
    """
    out[token_id, indice_id] =
        block_table[req_id[token_id],
            token_indices[token_id, indice_id] // BLOCK_SIZE] * BLOCK_SIZE
        + token_indices[token_id, indice_id] % BLOCK_SIZE

    Only when token_indices[token_id, indice_id] == -1 do we output -1.
    For safety, we also output -1 if the derived block_id would be
        out-of-bounds.
    """
    assert kv_indices.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert token_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0, (
        f"NUM_TOPK_TOKENS ({NUM_TOPK_TOKENS}) must be divisible by"
        f"BLOCK_N ({BLOCK_N})"
    )

    num_batch = kv_indptr.shape[0] - 1
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N

    # Ensure contiguous tensors on the same device
    qo_indptr_c = qo_indptr.contiguous()
    kv_indptr_c = kv_indptr.contiguous()
    kv_indices_c = kv_indices.contiguous()
    token_indices_c = token_indices.contiguous()
    page_kv_indptr_c = page_kv_indptr.contiguous()
    # NOTE: MTP (max_seqlen_q > 1) uses triton_convert_req_index_to_global_index_dsa_prefill instead
    if out is not None:
        new_kv_indices = out[: kv_indices.shape[0]]
    else:
        new_kv_indices = torch.empty_like(kv_indices)

    # Strides in elements
    ti_stride0, ti_stride1 = token_indices_c.stride()

    # Exact 2D grid: tokens × column tiles
    grid = (num_batch, tiles_per_row)

    _convert_req_index_to_global_index_kernel[grid](
        qo_indptr_c,
        kv_indptr_c,
        page_kv_indptr_c,
        kv_indices_c,
        token_indices_c,
        new_kv_indices,
        # shapes / constexprs
        NUM_TOPK_TOKENS,
        BLOCK_SIZE,
        BLOCK_N,
        # strides
        ti_stride0,
        ti_stride1,
    )
    return new_kv_indices


@triton.jit
def _convert_req_index_to_global_index_dsa_prefill_kernel(
    dsa_qo_indptr,  # int32 [num_tokens + 1]
    dsa_kv_indptr,  # int32 [num_tokens + 1]
    token_to_seq_idxs,  # int32 [num_tokens]
    topk_indices,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    block_table,  # int32 [num_req, max_num_blocks_per_req]
    cu_seqlens_q,  # int32 [num_tokens + 1]
    out_kv_indices,  # int32
    # shapes (compile-time where possible)
    NUM_TOPK_TOKENS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,  # tile width along columns
    # strides (in elements)
    ti_stride0: tl.int64,  # topk_indices stride 0
    ti_stride1: tl.constexpr,  # topk_indices stride 1
    bt_stride0: tl.int64,  # block_table stride 0
    bt_stride1: tl.constexpr,  # block_table stride 1
):
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    col_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

    req_id = tl.load(token_to_seq_idxs + token_id)  # int32

    kv_start = tl.load(dsa_kv_indptr + token_id)
    kv_end = tl.load(dsa_kv_indptr + token_id + 1)
    kv_len = kv_end - kv_start

    # Load token indices for this tile
    indice = tl.load(
        topk_indices + token_id * ti_stride0 + col_id * ti_stride1
    )  # int32
    pre_seqlens_q = tl.load(cu_seqlens_q + req_id)

    seq_token_idx = indice - pre_seqlens_q
    block_id = seq_token_idx // PAGE_SIZE
    inblock_offset = seq_token_idx % PAGE_SIZE

    # Guard block_table access
    store_mask = (col_id < kv_len) & (col_id < NUM_TOPK_TOKENS)
    valid_mask = store_mask & (indice >= 0)
    physical_block = tl.load(
        block_table + req_id * bt_stride0 + block_id * bt_stride1,
        mask=valid_mask,
        other=-1,
    )
    out_val = tl.where(valid_mask, physical_block * PAGE_SIZE + inblock_offset, -1)

    # Store results
    out_ptr_ij = out_kv_indices + kv_start + col_id
    tl.store(
        out_ptr_ij,
        out_val,
        mask=store_mask,
    )


def triton_convert_req_index_to_global_index_dsa_prefill(
    dsa_qo_indptr: torch.Tensor,  # int32 [num_tokens + 1]
    dsa_kv_indptr: torch.Tensor,  # int32 [num_tokens + 1]
    token_to_seq_idxs: torch.Tensor,  # int32 [num_tokens]
    topk_indices: torch.Tensor,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    block_table: torch.Tensor,  # int32 [num_req, max_num_blocks_per_req]
    cu_seqlens_q: torch.Tensor,  # int32 [num_tokens + 1]
    # dsa_kv_indices: torch.Tensor,  # int32 [total_kv_seqlen]           -->>>     output for this kernel
    PAGE_SIZE: int = 1,
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 1024,  # tile width along columns
    out: Optional[torch.Tensor] = None,
):

    assert topk_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0, (
        f"NUM_TOPK_TOKENS ({NUM_TOPK_TOKENS}) must be divisible by"
        f"BLOCK_N ({BLOCK_N})"
    )

    num_tokens = dsa_qo_indptr.shape[0] - 1
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N

    total_out = num_tokens * NUM_TOPK_TOKENS
    if out is not None:
        new_kv_indices = out[:total_out]
    else:
        new_kv_indices = torch.empty(
            total_out, dtype=torch.int32, device=topk_indices.device
        )

    # Strides in elements
    ti_stride0, ti_stride1 = topk_indices.stride()
    bt_stride0, bt_stride1 = block_table.stride()

    grid = (num_tokens, tiles_per_row)

    _convert_req_index_to_global_index_dsa_prefill_kernel[grid](
        dsa_qo_indptr,
        dsa_kv_indptr,
        token_to_seq_idxs,
        topk_indices,
        block_table,
        cu_seqlens_q,
        new_kv_indices,
        # shapes / constexprs
        NUM_TOPK_TOKENS,
        PAGE_SIZE,
        BLOCK_N,
        # strides
        ti_stride0,
        ti_stride1,
        bt_stride0,
        bt_stride1,
    )
    return new_kv_indices


@triton.jit
def _gather_kv_indices_sparse_kernel(
    sparse_kv_indptr,
    token_to_seq_idxs,
    topk_indices,
    kv_indices,
    kv_indptr,
    out_kv_indices,
    NUM_TOPK_TOKENS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ti_stride0: tl.int64,
    ti_stride1: tl.constexpr,
):
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)
    col_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

    req_id = tl.load(token_to_seq_idxs + token_id)

    out_start = tl.load(sparse_kv_indptr + token_id)
    out_end = tl.load(sparse_kv_indptr + token_id + 1)
    kv_len = out_end - out_start

    pos = tl.load(topk_indices + token_id * ti_stride0 + col_id * ti_stride1)

    kv_base = tl.load(kv_indptr + req_id)
    kv_end = tl.load(kv_indptr + req_id + 1)
    req_kv_len = kv_end - kv_base

    store_mask = (col_id < kv_len) & (col_id < NUM_TOPK_TOKENS)
    valid_mask = store_mask & (pos >= 0) & (pos < req_kv_len)

    out_val = tl.load(
        kv_indices + kv_base + pos,
        mask=valid_mask,
        other=0,
    )

    tl.store(
        out_kv_indices + out_start + col_id,
        out_val,
        mask=store_mask,
    )


def triton_gather_kv_indices_sparse(
    sparse_kv_indptr: torch.Tensor,
    token_to_seq_idxs: torch.Tensor,
    topk_indices: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 1024,
    out: Optional[torch.Tensor] = None,
):
    assert topk_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0

    # MTP decode can carry metadata tensors padded to a larger query layout
    # than the number of rows produced by the current indexer call. Keep all
    # per-token inputs aligned to the actual valid intersection before launch;
    # otherwise the kernel may read past topk_indices.
    num_tokens = min(
        token_to_seq_idxs.shape[0],
        topk_indices.shape[0],
        sparse_kv_indptr.shape[0] - 1,
    )
    sparse_kv_indptr = sparse_kv_indptr[: num_tokens + 1]
    token_to_seq_idxs = token_to_seq_idxs[:num_tokens]
    topk_indices = topk_indices[:num_tokens]
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N

    total_out = num_tokens * NUM_TOPK_TOKENS
    if out is not None:
        out_buf = out[:total_out]
    else:
        out_buf = torch.empty(total_out, dtype=torch.int32, device=topk_indices.device)

    ti_stride0, ti_stride1 = topk_indices.stride()
    grid = (num_tokens, tiles_per_row)

    _gather_kv_indices_sparse_kernel[grid](
        sparse_kv_indptr,
        token_to_seq_idxs,
        topk_indices,
        kv_indices,
        kv_indptr,
        out_buf,
        NUM_TOPK_TOKENS,
        BLOCK_N,
        ti_stride0,
        ti_stride1,
    )
    return out_buf
