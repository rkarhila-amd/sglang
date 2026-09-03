"""Deduped speculative/MoE path logging for concurrency debugging."""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_spec_seen: set[str] = set()
_moe_seen: set[str] = set()


def spec_path_debug_enabled() -> bool:
    return os.environ.get("SGLANG_SPEC_PATH_DEBUG", "0") == "1"


def moe_path_debug_enabled() -> bool:
    return os.environ.get("SGLANG_MOE_PATH_DEBUG", "0") == "1"


def log_spec_path(
    *,
    phase: str,
    forward_mode: str,
    bs: int,
    num_tokens_per_req: int,
    cuda_graph: bool,
    extra: Optional[str] = None,
) -> None:
    if not spec_path_debug_enabled():
        return
    moe_m = bs * num_tokens_per_req if num_tokens_per_req else bs
    key = f"{phase}|{forward_mode}|bs={bs}|w={num_tokens_per_req}|cg={cuda_graph}"
    if key in _spec_seen:
        return
    _spec_seen.add(key)
    msg = (
        f"[SPEC_PATH] {phase}: forward_mode={forward_mode} bs={bs} "
        f"num_tokens_per_req={num_tokens_per_req} moe_M≈{moe_m} cuda_graph={cuda_graph}"
    )
    if extra:
        msg += f" {extra}"
    logger.info(msg)


def log_moe_runner(
    *,
    M: int,
    activation: str,
    gate_mode: int,
    quant_type: str,
    swiglu_limit: float,
    forward_mode: Optional[str] = None,
    layer_id: Optional[int] = None,
) -> None:
    if not moe_path_debug_enabled():
        return
    key = (
        f"M={M}|act={activation}|gate={gate_mode}|qt={quant_type}|"
        f"fm={forward_mode}|layer={layer_id}"
    )
    if key in _moe_seen:
        return
    _moe_seen.add(key)
    layer_part = f" layer={layer_id}" if layer_id is not None else ""
    fm_part = f" forward_mode={forward_mode}" if forward_mode else ""
    logger.info(
        "[MOE_PATH] sglang aiter_runner%s%s: M=%d activation=%s gate_mode=%d "
        "quant_type=%s swiglu_limit=%s",
        layer_part,
        fm_part,
        M,
        activation,
        gate_mode,
        quant_type,
        swiglu_limit,
    )
