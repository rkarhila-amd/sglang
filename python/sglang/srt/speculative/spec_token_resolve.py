"""Resolve EAGLE spec-v2 accepted tokens for scheduler commit."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Sequence

from sglang.srt.debug.moe_path_debug import spec_path_debug_enabled

if TYPE_CHECKING:
    from sglang.srt.managers.utils import GenerationBatchResult

logger = logging.getLogger(__name__)

# Matches verify_tree_greedy tests: unwritten predict slots stay unfilled.
UNFILLED_PREDICT_TOKEN = -1


def sanitize_accept_tokens(tokens: Sequence[int]) -> List[int]:
    """Drop trailing unfilled predict slots (sentinel -1)."""
    out: List[int] = []
    for token_id in tokens:
        if token_id == UNFILLED_PREDICT_TOKEN:
            break
        out.append(int(token_id))
    return out


def resolve_spec_accept_tokens(
    flat_predict: Sequence[int],
    req_idx: int,
    accept_len: int,
    stride: int,
    accept_index_row: Optional[Sequence[int]] = None,
) -> List[int]:
    """Gather accepted tokens for one request from the verify predict buffer."""
    if accept_index_row is not None:
        indices = [idx for idx in accept_index_row[:accept_len] if idx >= 0]
        tokens = [flat_predict[idx] for idx in indices]
    else:
        start = req_idx * stride
        tokens = list(flat_predict[start : start + accept_len])
    return sanitize_accept_tokens(tokens)


def resolve_spec_accept_tokens_from_result(
    result: GenerationBatchResult,
    req_idx: int,
    accept_len: int,
    *,
    flat_predict: Optional[Sequence[int]] = None,
    accept_index_rows: Optional[Sequence[Sequence[int]]] = None,
) -> List[int]:
    stride = result.speculative_num_draft_tokens
    assert stride is not None
    if flat_predict is None:
        assert result.next_token_ids is not None
        flat_predict = result.next_token_ids.tolist()
    if accept_index_rows is None and result.accept_index is not None:
        assert result.accept_index.is_cpu
        accept_index_rows = result.accept_index.tolist()
    row = accept_index_rows[req_idx] if accept_index_rows is not None else None
    return resolve_spec_accept_tokens(
        flat_predict,
        req_idx,
        accept_len,
        stride,
        accept_index_row=row,
    )


def log_spec_accept_resolve(
    *,
    req_idx: int,
    accept_len: int,
    stride: int,
    tokens: Sequence[int],
    accept_index_row: Optional[Sequence[int]] = None,
    batch_size: int = 0,
    cuda_graph: bool = False,
    rid: Optional[str] = None,
) -> None:
    if not spec_path_debug_enabled():
        return
    zero_count = sum(1 for t in tokens if t == 0)
    used_gather = accept_index_row is not None
    expected_len = accept_len
    actual_len = len(tokens)
    if (
        zero_count == 0
        and actual_len == expected_len
        and not any(t == UNFILLED_PREDICT_TOKEN for t in tokens)
    ):
        return
    logger.warning(
        "[SPEC_ACCEPT] req=%s idx=%d bs=%d cg=%s gather=%s "
        "accept_len=%d stride=%d resolved_len=%d zero_ids=%d tokens=%s indices=%s",
        rid or req_idx,
        req_idx,
        batch_size,
        cuda_graph,
        used_gather,
        accept_len,
        stride,
        actual_len,
        zero_count,
        list(tokens)[:16],
        list(accept_index_row)[:16] if accept_index_row is not None else None,
    )
