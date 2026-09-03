"""Bisect NaN / numeric drift on fused shared-expert MoE (expert slot 128+).

Enable with::

    export SGLANG_FUSED_MOE_BISECT_DEBUG=1
    export SGLANG_FUSED_MOE_BISECT_LAYER_IDS=3,4
    export SGLANG_FUSED_MOE_BISECT_TARGET_M=128
    export SGLANG_FUSED_MOE_BISECT_LOG_ALL=1   # log finite rows once per site

Logs ``[FUSED_MOE_BISECT]`` on TARGET_VERIFY forwards when fusion is active.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

_seen: set[str] = set()
_log_count = 0


def fused_moe_bisect_enabled() -> bool:
    return os.environ.get("SGLANG_FUSED_MOE_BISECT_DEBUG", "0") == "1"


def _target_layer_ids() -> set[int]:
    raw = os.environ.get("SGLANG_FUSED_MOE_BISECT_LAYER_IDS", "3,4")
    return {int(x.strip()) for x in raw.split(",") if x.strip()}


def _target_m_set() -> set[int]:
    raw = os.environ.get("SGLANG_FUSED_MOE_BISECT_TARGET_M", "128")
    if not raw.strip():
        return set()
    return {int(x.strip()) for x in raw.split(",") if x.strip()}


def _target_step_set() -> set[int]:
    raw = os.environ.get("SGLANG_FUSED_MOE_BISECT_TARGET_STEPS", "")
    if not raw.strip():
        return set()
    return {int(x.strip()) for x in raw.split(",") if x.strip()}


def _max_logs() -> int:
    return int(os.environ.get("SGLANG_FUSED_MOE_BISECT_MAX_LOGS", "800"))


def _log_all() -> bool:
    return os.environ.get("SGLANG_FUSED_MOE_BISECT_LOG_ALL", "1") == "1"


def _in_cuda_graph_capture() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _verify_step() -> int:
    try:
        from sglang.srt.speculative.spec_verify_debug import get_verify_step

        return get_verify_step()
    except Exception:
        return 0


def _bisect_max_steps() -> int:
    raw = os.environ.get("SGLANG_FUSED_MOE_BISECT_MAX_STEPS", "")
    if raw.strip():
        return int(raw.strip())
    try:
        from sglang.srt.speculative.spec_verify_debug import (
            _max_steps as spec_verify_max_steps,
        )

        return spec_verify_max_steps()
    except Exception:
        return 5


def _step_allowed() -> bool:
    steps = _target_step_set()
    if steps:
        return _verify_step() in steps
    return _verify_step() < _bisect_max_steps()


def _tp_rank() -> int:
    try:
        from sglang.srt.distributed import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())
    except Exception:
        return 0


def _tp_size() -> int:
    try:
        from sglang.srt.distributed import get_tensor_model_parallel_world_size

        return int(get_tensor_model_parallel_world_size())
    except Exception:
        return 1


def _tensor_stats(t: torch.Tensor) -> dict:
    t = t.detach()
    if t.numel() == 0:
        return {"shape": list(t.shape), "empty": True}
    finite = torch.isfinite(t)
    out = {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "nan": int(torch.isnan(t).sum().item()),
        "inf": int(torch.isinf(t).sum().item()),
        "non_finite": int((~finite).sum().item()),
    }
    if finite.any():
        tf = t[finite]
        out["min"] = float(tf.min().item())
        out["max"] = float(tf.max().item())
        out["mean"] = float(tf.float().mean().item())
    return out


def _bad_row_indices(tensor: torch.Tensor, max_rows: int = 128) -> list[int]:
    t = tensor.detach().reshape(tensor.shape[0], -1)
    row_bad = (~torch.isfinite(t)).any(dim=1)
    return torch.nonzero(row_bad, as_tuple=False).flatten().tolist()[:max_rows]


def _is_target_verify(forward_batch) -> bool:
    if forward_batch is None:
        return True
    try:
        return bool(forward_batch.forward_mode.is_target_verify())
    except Exception:
        return False


def _m_allowed(m: int) -> bool:
    targets = _target_m_set()
    if not targets:
        return True
    return m in targets


def log_fused_moe_bisect(
    *,
    site: str,
    layer_id: int,
    tensor: Optional[torch.Tensor],
    forward_batch=None,
    extra: Optional[str] = None,
) -> None:
    global _log_count
    if not fused_moe_bisect_enabled() or _in_cuda_graph_capture():
        return
    if tensor is None or layer_id not in _target_layer_ids():
        return
    if not _is_target_verify(forward_batch):
        return
    if not _step_allowed():
        return
    if tensor.dim() < 1:
        return

    m = int(tensor.shape[0])
    if not _m_allowed(m):
        return

    stats = _tensor_stats(tensor)
    non_finite = int(stats.get("non_finite", 0))
    step = _verify_step()

    key = f"{step}|{layer_id}|{site}|M={m}|{'bad' if non_finite else 'ok'}"
    if non_finite == 0:
        if not _log_all():
            return
        ok_key = f"{step}|{layer_id}|{site}|M={m}|ok_once"
        if ok_key in _seen:
            return
        _seen.add(ok_key)
    else:
        if key in _seen:
            return
        _seen.add(key)

    if _log_count >= _max_logs():
        return
    _log_count += 1

    parts = [
        f"[FUSED_MOE_BISECT] verify_step={step} layer={layer_id} tp={_tp_rank()}/{_tp_size()} site={site} M={m}",
        f"stats={stats}",
    ]
    if non_finite > 0:
        parts.append(f"bad_rows={_bad_row_indices(tensor)}")
    if extra:
        parts.append(extra)
    level = logging.WARNING if non_finite > 0 else logging.INFO
    logger.log(level, " ".join(parts))


def _topk_tensors(topk_output) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    if TopKOutputChecker.format_is_standard(topk_output):
        return topk_output.topk_ids, topk_output.topk_weights
    return None, None


def log_fused_moe_topk(
    *,
    layer_id: int,
    topk_output: Any,
    fused_expert_base_id: int,
    num_fused_shared_experts: int,
    forward_batch=None,
) -> None:
    if not fused_moe_bisect_enabled() or _in_cuda_graph_capture():
        return
    if layer_id not in _target_layer_ids():
        return
    if not _is_target_verify(forward_batch):
        return
    if not _step_allowed():
        return

    topk_ids, topk_weights = _topk_tensors(topk_output)
    if topk_ids is None or topk_weights is None:
        return

    m = int(topk_ids.shape[0])
    if not _m_allowed(m):
        return

    step = _verify_step()
    fused_ids = list(
        range(fused_expert_base_id, fused_expert_base_id + num_fused_shared_experts)
    )
    ids_cpu = topk_ids.detach().cpu()
    w_cpu = topk_weights.detach().cpu()

    rows_with_fused: list[int] = []
    fused_weight_by_row: dict[int, float] = {}
    for row in range(m):
        row_ids = ids_cpu[row].tolist()
        for fid in fused_ids:
            if fid in row_ids:
                rows_with_fused.append(row)
                idx = row_ids.index(fid)
                fused_weight_by_row[row] = float(w_cpu[row, idx].item())
                break

    non_finite_ids = int((~torch.isfinite(topk_weights)).sum().item())
    parts = [
        f"[FUSED_MOE_BISECT] verify_step={step} layer={layer_id} tp={_tp_rank()}/{_tp_size()} site=topk_fused_slot M={m}",
        f"fused_expert_ids={fused_ids}",
        f"rows_with_fused={len(rows_with_fused)}/{m}",
        f"topk_weight_non_finite={non_finite_ids}",
    ]
    if fused_weight_by_row:
        sample = dict(list(fused_weight_by_row.items())[:8])
        parts.append(f"fused_weight_sample={sample}")
        vals = list(fused_weight_by_row.values())
        parts.append(
            f"fused_weight_min={min(vals):.6g} fused_weight_max={max(vals):.6g}"
        )

    key = f"{step}|{layer_id}|topk|M={m}"
    if key in _seen:
        return
    _seen.add(key)

    global _log_count
    if _log_count >= _max_logs():
        return
    _log_count += 1
    logger.info(" ".join(parts))
