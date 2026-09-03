"""Bisect NaN source upstream of shared_experts.gate_up_proj (MiniMax MoE).

Enable with::

    export SGLANG_SHARED_EXPERTS_BISECT_DEBUG=1
    export SGLANG_SHARED_EXPERTS_BISECT_LAYER_IDS=3   # default: 3
    export SGLANG_SHARED_EXPERTS_BISECT_TARGET_M=128  # TARGET_VERIFY rows
    export SGLANG_SHARED_EXPERTS_BISECT_TARGET_STEPS=1  # optional verify steps
    export SGLANG_SHARED_EXPERTS_BISECT_TARGET_ROWS=60,61,62,63  # optional rows

Logs one line per (layer, site, M) on TARGET_VERIFY forwards. Non-finite rows
always log; finite rows log once per site when
``SGLANG_SHARED_EXPERTS_BISECT_LOG_ALL=1``.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_seen: set[str] = set()
_log_count = 0


def shared_experts_bisect_enabled() -> bool:
    return os.environ.get("SGLANG_SHARED_EXPERTS_BISECT_DEBUG", "0") == "1"


def _target_layer_ids() -> set[int]:
    raw = os.environ.get("SGLANG_SHARED_EXPERTS_BISECT_LAYER_IDS", "3")
    return {int(x.strip()) for x in raw.split(",") if x.strip()}


def _target_m_set() -> set[int]:
    raw = os.environ.get("SGLANG_SHARED_EXPERTS_BISECT_TARGET_M", "128")
    if not raw.strip():
        return set()
    return {int(x.strip()) for x in raw.split(",") if x.strip()}


def _target_step_set() -> set[int]:
    raw = os.environ.get("SGLANG_SHARED_EXPERTS_BISECT_TARGET_STEPS", "")
    if not raw.strip():
        return set()
    return {int(x.strip()) for x in raw.split(",") if x.strip()}


def _target_row_set() -> set[int]:
    raw = os.environ.get("SGLANG_SHARED_EXPERTS_BISECT_TARGET_ROWS", "")
    if not raw.strip():
        return set()
    return {int(x.strip()) for x in raw.split(",") if x.strip()}


def _max_logs() -> int:
    return int(os.environ.get("SGLANG_SHARED_EXPERTS_BISECT_MAX_LOGS", "500"))


def _log_all() -> bool:
    return os.environ.get("SGLANG_SHARED_EXPERTS_BISECT_LOG_ALL", "1") == "1"


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
        return -1


def _tensor_stats(t: torch.Tensor) -> dict:
    t = t.detach()
    if t.numel() == 0:
        return {"shape": list(t.shape), "dtype": str(t.dtype), "empty": True}
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
    return out


def _bad_row_indices(tensor: torch.Tensor, max_rows: int = 128) -> list[int]:
    t = tensor.detach().reshape(tensor.shape[0], -1)
    row_bad = (~torch.isfinite(t)).any(dim=1)
    return torch.nonzero(row_bad, as_tuple=False).flatten().tolist()[:max_rows]


def _row_stats(tensor: torch.Tensor, rows: set[int]) -> dict:
    out = {}
    for row in sorted(rows):
        if 0 <= row < tensor.shape[0]:
            out[row] = _tensor_stats(tensor[row : row + 1])
    return out


def _m_allowed(m: int) -> bool:
    targets = _target_m_set()
    if targets:
        return m in targets
    return m >= 8


def _step_allowed() -> bool:
    targets = _target_step_set()
    if not targets:
        return True
    return _verify_step() in targets


_dump_done: set[str] = set()


def _tp_rank() -> int:
    try:
        from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())
    except Exception:
        return 0


def _tp_size() -> int:
    try:
        from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_world_size

        return int(get_tensor_model_parallel_world_size())
    except Exception:
        return 1


def _dump_path_for_rank(base: str) -> str:
    if os.environ.get("SGLANG_SHARED_EXPERTS_BISECT_DUMP_PER_TP", "0") != "1":
        return base
    root, ext = os.path.splitext(base)
    if not ext:
        return f"{base}_tp{_tp_rank()}.pt"
    return f"{root}_tp{_tp_rank()}{ext}"


def _maybe_dump_tensors(
    *,
    site: str,
    layer_id: int,
    step: int,
    tensor: torch.Tensor,
    bundle: Optional[dict] = None,
) -> Optional[str]:
    dump_path = _dump_path_for_rank(
        os.environ.get("SGLANG_SHARED_EXPERTS_BISECT_DUMP", "").strip()
    )
    if not dump_path:
        return None
    target_step = int(os.environ.get("SGLANG_SHARED_EXPERTS_BISECT_DUMP_STEP", "1"))
    if step != target_step:
        return None
    if layer_id not in _target_layer_ids():
        return None

    dump_on = os.environ.get("SGLANG_SHARED_EXPERTS_BISECT_DUMP_MODE", "always")
    non_finite = int(_tensor_stats(tensor).get("non_finite", 0))
    if dump_on == "bad_only" and non_finite == 0 and site not in {
        "shared_gate_up_in",
        "shared_swiglu_out",
    }:
        return None

    key = f"{dump_path}|{step}|{layer_id}"
    if bundle is None:
        bundle = {}
    if os.path.exists(dump_path):
        try:
            existing = torch.load(dump_path, map_location="cpu")
            if isinstance(existing, dict):
                bundle = {**existing, **bundle}
        except Exception:
            pass

    bundle.setdefault("verify_step", step)
    bundle.setdefault("layer_id", layer_id)
    bundle.setdefault("tp_rank", _tp_rank())
    bundle.setdefault("tp_size", _tp_size())
    bundle[site] = tensor.detach().cpu()

    dump_bad = os.environ.get("SGLANG_SHARED_EXPERTS_BISECT_DUMP_ON_BAD", "1") == "1"
    should_write = site in {"shared_gate_up_in", "shared_swiglu_out"} or (
        dump_bad and non_finite > 0
    )
    if not should_write:
        return None

    flag = f"{key}|{site}"
    if flag in _dump_done and site not in {"shared_down_out", "moe_shared_out"}:
        return None
    _dump_done.add(flag)

    try:
        os.makedirs(os.path.dirname(dump_path) or ".", exist_ok=True)
        torch.save(bundle, dump_path)
        return dump_path
    except Exception as exc:
        return f"dump_error:{exc}"


def log_shared_experts_bisect(
    *,
    site: str,
    layer_id: int,
    tensor: Optional[torch.Tensor],
    forward_batch=None,
    extra: Optional[str] = None,
) -> None:
    global _log_count
    if not shared_experts_bisect_enabled() or _in_cuda_graph_capture():
        return
    if tensor is None or layer_id not in _target_layer_ids():
        return
    if forward_batch is not None and not forward_batch.forward_mode.is_target_verify():
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
    watch_rows = _target_row_set()
    watch_hit = bool(watch_rows) and any(
        _tensor_stats(tensor[r : r + 1]).get("non_finite", 0) > 0
        for r in watch_rows
        if 0 <= r < tensor.shape[0]
    )

    step = _verify_step()
    dumped = _maybe_dump_tensors(
        site=site, layer_id=layer_id, step=step, tensor=tensor
    )

    key = f"{step}|{layer_id}|{site}|M={m}|{'bad' if non_finite else 'ok'}"
    if non_finite == 0 and not watch_rows:
        if not _log_all():
            return
        ok_key = f"{step}|{layer_id}|{site}|M={m}|ok_once"
        if ok_key in _seen:
            return
        _seen.add(ok_key)
    elif non_finite == 0 and watch_rows:
        ok_key = f"{step}|{layer_id}|{site}|M={m}|watch_once"
        if ok_key in _seen:
            return
        _seen.add(ok_key)
    else:
        if key in _seen and not watch_hit:
            return
        _seen.add(key)

    if _log_count >= _max_logs():
        return
    _log_count += 1

    parts = [
        f"[SHARED_EXPERTS_BISECT] verify_step={step} layer={layer_id} tp={_tp_rank()}/{_tp_size()} site={site} M={m}",
        f"stats={stats}",
    ]
    if non_finite > 0:
        parts.append(f"bad_rows={_bad_row_indices(tensor)}")
    if watch_rows:
        parts.append(f"watch_rows={_row_stats(tensor, watch_rows)}")
    if dumped:
        parts.append(f"dumped={dumped}")
    if extra:
        parts.append(extra)
    level = logging.WARNING if (non_finite > 0 or watch_hit) else logging.INFO
    logger.log(level, " ".join(parts))


_down_weight_dump_done: set[int] = set()
_load_weight_dump_done: set[int] = set()


def maybe_dump_down_proj_weights_at_load(*, linear_module) -> None:
    """Dump shared_experts down_proj weights once per TP rank at model load."""
    dump_dir = os.environ.get("SGLANG_LAYER3_DOWN_WEIGHT_DUMP", "").strip()
    if not dump_dir:
        return
    prefix = getattr(linear_module, "prefix", "") or ""
    if "layers.3" not in prefix or "shared_experts.down_proj" not in prefix:
        return
    tp_rank = _tp_rank()
    if tp_rank in _load_weight_dump_done:
        return
    weight = getattr(linear_module, "weight", None)
    scale = getattr(linear_module, "weight_scale", None)
    if weight is None or scale is None:
        return
    block = int(os.environ.get("SGLANG_LAYER3_DOWN_WEIGHT_BLOCK", "57"))
    block_size = int(os.environ.get("SGLANG_LAYER3_DOWN_WEIGHT_BLOCK_SIZE", "64"))
    row0 = block * block_size
    row1 = row0 + block_size
    bundle = {
        "layer_id": 3,
        "tp_rank": tp_rank,
        "tp_size": _tp_size(),
        "prefix": prefix,
        "block": block,
        "block_size": block_size,
        "row_range": [row0, row1],
        "weight_shape": list(weight.shape),
        "weight_scale_shape": list(scale.shape),
        "weight_block": weight[row0:row1].detach().cpu(),
        "weight_scale_block": scale[row0:row1].detach().cpu(),
    }
    try:
        os.makedirs(dump_dir, exist_ok=True)
        out = os.path.join(dump_dir, f"runtime_down_tp{tp_rank}_block{block}.pt")
        torch.save(bundle, out)
        if not os.path.isfile(out):
            raise OSError(f"torch.save did not create {out}")
        _load_weight_dump_done.add(tp_rank)
        logger.warning(
            "[SHARED_EXPERTS_BISECT] load-time dump down_proj tp=%s block=%s prefix=%s -> %s",
            tp_rank,
            block,
            prefix,
            out,
        )
    except Exception as exc:
        logger.error(
            "[SHARED_EXPERTS_BISECT] load-time down_proj dump failed tp=%s: %s",
            tp_rank,
            exc,
        )


_down_weight_dump_done_fwd: set[int] = set()


def maybe_dump_down_proj_weights(*, layer_id: int, linear_module) -> None:
    """One-shot dump of shared_experts down_proj MXFP4 weights per TP rank."""
    dump_dir = os.environ.get("SGLANG_LAYER3_DOWN_WEIGHT_DUMP", "").strip()
    if not dump_dir:
        return
    target_layer = int(os.environ.get("SGLANG_LAYER3_DOWN_WEIGHT_LAYER", "3"))
    if layer_id != target_layer:
        return
    tp_rank = _tp_rank()
    if tp_rank in _down_weight_dump_done_fwd:
        return
    weight = getattr(linear_module, "weight", None)
    scale = getattr(linear_module, "weight_scale", None)
    if weight is None or scale is None:
        return
    block = int(os.environ.get("SGLANG_LAYER3_DOWN_WEIGHT_BLOCK", "57"))
    block_size = int(os.environ.get("SGLANG_LAYER3_DOWN_WEIGHT_BLOCK_SIZE", "64"))
    row0 = block * block_size
    row1 = row0 + block_size
    bundle = {
        "layer_id": layer_id,
        "tp_rank": tp_rank,
        "tp_size": _tp_size(),
        "block": block,
        "block_size": block_size,
        "row_range": [row0, row1],
        "weight_shape": list(weight.shape),
        "weight_scale_shape": list(scale.shape),
        "weight_block": weight[row0:row1].detach().cpu(),
        "weight_scale_block": scale[row0:row1].detach().cpu(),
    }
    try:
        os.makedirs(dump_dir, exist_ok=True)
        out = os.path.join(dump_dir, f"runtime_down_tp{tp_rank}_block{block}.pt")
        torch.save(bundle, out)
        if not os.path.isfile(out):
            raise OSError(f"torch.save did not create {out}")
        _down_weight_dump_done_fwd.add(tp_rank)
        logger.warning(
            "[SHARED_EXPERTS_BISECT] dumped runtime down_proj weights tp=%s block=%s shape_w=%s shape_s=%s -> %s",
            tp_rank,
            block,
            list(weight.shape),
            list(scale.shape),
            out,
        )
    except Exception as exc:
        logger.error(
            "[SHARED_EXPERTS_BISECT] failed runtime down_proj weight dump tp=%s: %s",
            tp_rank,
            exc,
        )
