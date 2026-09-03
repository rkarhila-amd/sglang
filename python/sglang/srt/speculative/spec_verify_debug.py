"""Debug logging for EAGLE target-verify sampling (logits / argmax / accept)."""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_verify_step = 0
_layer_nan_first: dict[tuple[int, str], int] = {}
_minimax_tv_seen: set[tuple[int, int, int]] = set()
_lmhead_seen: set[str] = set()


def spec_verify_debug_enabled() -> bool:
    return os.environ.get("SGLANG_SPEC_VERIFY_DEBUG", "0") == "1"


def spec_verify_layer_debug_enabled() -> bool:
    return spec_verify_debug_enabled() or os.environ.get(
        "SGLANG_SPEC_VERIFY_LAYER_DEBUG", "0"
    ) == "1"


def _max_steps() -> int:
    return int(os.environ.get("SGLANG_SPEC_VERIFY_DEBUG_MAX_STEPS", "5"))


def _min_bs() -> int:
    return int(os.environ.get("SGLANG_SPEC_VERIFY_DEBUG_MIN_BS", "16"))


def _target_bs_set() -> set[int]:
    raw = os.environ.get("SGLANG_SPEC_VERIFY_TARGET_BS", "")
    if not raw.strip():
        return set()
    return {int(x.strip()) for x in raw.split(",") if x.strip()}


def _all_layers_enabled() -> bool:
    return os.environ.get("SGLANG_SPEC_VERIFY_ALL_LAYERS", "0") == "1"


def _milestone_layers(num_layers: int = 63) -> set[int]:
    if _all_layers_enabled():
        return set(range(num_layers))
    raw = os.environ.get("SGLANG_SPEC_VERIFY_LAYER_IDS", "")
    if raw.strip():
        return {int(x.strip()) for x in raw.split(",") if x.strip()}
    layers = {0, 1, 2, 15, 31, num_layers - 1}
    layers.update(range(0, num_layers, 8))
    return {i for i in layers if 0 <= i < num_layers}


def _layer_should_log(layer_id: int, non_finite: int, num_layers: int = 63) -> bool:
    """Log milestone layers always; any layer when it has NaN/Inf."""
    if non_finite > 0:
        return True
    return layer_id in _milestone_layers(num_layers)


def _batch_size(forward_batch) -> int:
    if forward_batch is None:
        return 0
    if getattr(forward_batch, "seq_lens", None) is not None:
        return int(forward_batch.seq_lens.shape[0])
    return 0


def _bs_allowed(bs: int) -> bool:
    targets = _target_bs_set()
    if targets:
        return bs in targets
    return bs >= _min_bs()


def _target_step_set() -> set[int]:
    raw = os.environ.get("SGLANG_SPEC_VERIFY_TARGET_STEPS", "")
    if not raw.strip():
        return set()
    return {int(x.strip()) for x in raw.split(",") if x.strip()}


def _step_allowed() -> bool:
    steps = _target_step_set()
    if not steps:
        return True
    return _verify_step in steps


def _lmhead_max_steps() -> int:
    raw = os.environ.get("SGLANG_SPEC_VERIFY_LMHEAD_MAX_STEPS", "")
    if raw.strip():
        return int(raw.strip())
    return _max_steps()


def _lmhead_log_all() -> bool:
    return os.environ.get("SGLANG_SPEC_VERIFY_LMHEAD_LOG_ALL", "1") == "1"


def _lmhead_should_log(bs: int) -> bool:
    if not spec_verify_layer_debug_enabled() or _in_cuda_graph_capture():
        return False
    if _verify_step >= _lmhead_max_steps():
        return False
    if not _step_allowed():
        return False
    return _bs_allowed(bs)


def _should_log(bs: int) -> bool:
    if not spec_verify_debug_enabled():
        return False
    if _verify_step >= _max_steps():
        return False
    if not _step_allowed():
        return False
    return _bs_allowed(bs)


def _in_cuda_graph_capture() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


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
    """Row indices (dim 0) with any non-finite element."""
    t = tensor.detach().reshape(tensor.shape[0], -1)
    row_bad = (~torch.isfinite(t)).any(dim=1)
    return torch.nonzero(row_bad, as_tuple=False).flatten().tolist()[:max_rows]


def _rows_to_req_draft(rows: list[int], draft_token_num: int = 4) -> list[dict]:
    out = []
    for row in rows:
        out.append(
            {
                "row": row,
                "req": row // draft_token_num,
                "draft_pos": row % draft_token_num,
            }
        )
    return out


def _seq_lens_cpu_list(forward_batch, limit: int = 64) -> Optional[list]:
    sl = getattr(forward_batch, "seq_lens_cpu", None)
    if sl is None:
        return None
    try:
        return [int(x) for x in sl[:limit].tolist()]
    except Exception:
        return None


def _prefix_lens_cpu_list(forward_batch, limit: int = 64) -> Optional[list]:
    pl = getattr(forward_batch, "prefix_lens", None)
    if pl is None:
        return None
    try:
        return [int(x) for x in pl[:limit].detach().cpu().tolist()]
    except Exception:
        return None


def log_eagle_verify_sampling(
    *,
    phase: str,
    bs: int,
    draft_token_num: int,
    next_token_logits: Optional[torch.Tensor] = None,
    target_predict: Optional[torch.Tensor] = None,
    candidates: Optional[torch.Tensor] = None,
    predict: Optional[torch.Tensor] = None,
    accept_lens: Optional[torch.Tensor] = None,
    forward_mode: Optional[str] = None,
    extra: Optional[str] = None,
) -> None:
    global _verify_step
    if not spec_verify_debug_enabled() or _in_cuda_graph_capture():
        return
    if not _should_log(bs):
        return

    step = _verify_step
    if phase == "post_verify":
        _verify_step += 1

    rows = min(bs, 4)
    parts = [
        f"[SPEC_VERIFY] step={step} phase={phase} bs={bs} draft_token_num={draft_token_num}",
    ]
    if forward_mode is not None:
        parts.append(f"forward_mode={forward_mode}")
    if extra:
        parts.append(extra)

    if next_token_logits is not None:
        parts.append(f"logits={_tensor_stats(next_token_logits)}")
        logits_stats = _tensor_stats(next_token_logits)
        if int(logits_stats.get("non_finite", 0)) > 0:
            parts.append(f"logits_bad_rows={_bad_row_indices(next_token_logits)}")
        flat = next_token_logits.reshape(bs * draft_token_num, -1)
        for i in range(rows):
            row = flat[i * draft_token_num]
            argmax = int(row.argmax().item())
            parts.append(
                f"req{i}_logits0 argmax={argmax} max={float(row.max().item()):.4g} "
                f"tok0={float(row[0].item()):.4g} tok1={float(row[1].item()):.4g}"
            )

    if target_predict is not None:
        tp = target_predict.reshape(bs, draft_token_num).detach().cpu()
        for i in range(rows):
            parts.append(f"req{i}_target_predict={tp[i].tolist()}")

    if candidates is not None:
        cand = candidates.detach().cpu()
        for i in range(rows):
            parts.append(f"req{i}_candidates={cand[i].tolist()}")

    if predict is not None:
        flat = predict.detach().cpu()
        stride = draft_token_num
        for i in range(rows):
            sl = flat[i * stride : (i + 1) * stride].tolist()
            parts.append(f"req{i}_predict_block={sl}")

    if accept_lens is not None:
        al = accept_lens.detach().cpu().tolist()
        parts.append(f"accept_lens_head={al[:rows]}")
        if len(al) > rows:
            parts.append(f"accept_lens_all={al}")

    logger.warning(" ".join(parts))


def log_target_verify_activation(
    *,
    site: str,
    layer_id: int,
    tensor: Optional[torch.Tensor],
    forward_batch,
    extra: Optional[str] = None,
) -> None:
    """Layer-wise NaN/Inf probe for TARGET_VERIFY forwards."""
    if not spec_verify_layer_debug_enabled() or _in_cuda_graph_capture():
        return
    if tensor is None or not forward_batch.forward_mode.is_target_verify():
        return

    bs = _batch_size(forward_batch)
    if not _bs_allowed(bs):
        return
    if _verify_step >= _max_steps():
        return
    if not _step_allowed():
        return

    stats = _tensor_stats(tensor)
    non_finite = int(stats.get("non_finite", 0))
    if not _layer_should_log(layer_id, non_finite):
        return

    key = (bs, site)
    first_nan = _layer_nan_first.get(key)
    if stats.get("non_finite", 0) > 0 and first_nan is None:
        _layer_nan_first[key] = layer_id
        first_nan = layer_id

    parts = [
        f"[SPEC_LAYER] step={_verify_step} site={site} layer={layer_id} bs={bs}",
        f"stats={stats}",
    ]
    if first_nan is not None:
        parts.append(f"first_nonfinite_layer={first_nan}")
    if non_finite > 0 and layer_id == first_nan:
        bad_rows = _bad_row_indices(tensor)
        draft = _draft_token_num()
        parts.append(f"bad_rows={bad_rows}")
        parts.append(f"bad_row_map={_rows_to_req_draft(bad_rows, draft)}")
        seq_lens = _seq_lens_cpu_list(forward_batch)
        if seq_lens is not None:
            bad_reqs = sorted({r // draft for r in bad_rows})
            parts.append(f"seq_lens_bad_reqs={[seq_lens[i] for i in bad_reqs if i < len(seq_lens)]}")
            parts.append(f"seq_lens_all={seq_lens}")
        prefix_lens = _prefix_lens_cpu_list(forward_batch)
        if prefix_lens is not None:
            parts.append(f"prefix_lens_all={prefix_lens}")
        ext_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if ext_cpu is not None:
            parts.append(f"extend_seq_lens_cpu={list(ext_cpu[:bs])}")
    if extra:
        parts.append(extra)
    logger.warning(" ".join(parts))


def get_verify_step() -> int:
    return _verify_step


def _draft_token_num() -> int:
    return int(os.environ.get("SGLANG_SPEC_VERIFY_DRAFT_TOKENS", "4"))


def _infer_verify_bs(tensor: Optional[torch.Tensor]) -> int:
    if tensor is None or tensor.ndim == 0 or tensor.shape[0] == 0:
        return 0
    draft = _draft_token_num()
    if tensor.shape[0] % draft != 0:
        return 0
    return tensor.shape[0] // draft


def log_target_verify_lm_head(
    *,
    site: str,
    tensor: Optional[torch.Tensor],
    logits_metadata,
    extra: Optional[str] = None,
) -> None:
    if not logits_metadata.forward_mode.is_target_verify():
        return
    bs = _infer_verify_bs(tensor)
    if not _lmhead_should_log(bs):
        return
    stats = _tensor_stats(tensor) if tensor is not None else {"missing": True}
    non_finite = int(stats.get("non_finite", 0))
    step = _verify_step

    if non_finite == 0 and not _lmhead_log_all():
        key = f"{step}|{site}|bs={bs}|ok"
        if key in _lmhead_seen:
            return
        _lmhead_seen.add(key)

    key = (bs, site)
    first_nan = _layer_nan_first.get(key)
    if non_finite > 0 and first_nan is None:
        _layer_nan_first[key] = -1
        first_nan = -1
    parts = [
        f"[SPEC_LMHEAD] step={step} site={site} bs={bs}",
        f"stats={stats}",
    ]
    if non_finite > 0:
        parts.append(f"bad_rows={_bad_row_indices(tensor)}")
        if tensor is not None and tensor.ndim >= 2:
            draft = _draft_token_num()
            flat = tensor.reshape(bs * draft, -1)
            for i in range(min(bs, 4)):
                row = flat[i * draft]
                if torch.isfinite(row).all():
                    parts.append(
                        f"req{i}_argmax={int(row.argmax().item())} "
                        f"max={float(row.max().item()):.4g}"
                    )
    if first_nan is not None:
        parts.append("first_nonfinite_lmhead=1")
    if extra:
        parts.append(extra)
    level = logging.WARNING if non_finite > 0 else logging.INFO
    logger.log(level, " ".join(parts))


def log_minimax_target_verify_forward(
    *,
    layer_id: int,
    forward_batch,
    q: torch.Tensor,
) -> None:
    if not spec_verify_layer_debug_enabled() or _in_cuda_graph_capture():
        return
    bs = _batch_size(forward_batch)
    if not _bs_allowed(bs):
        return
    if not _layer_should_log(layer_id, 0):
        return
    key = (bs, int(q.shape[0]), layer_id)
    if key in _minimax_tv_seen:
        return
    _minimax_tv_seen.add(key)
    ext = getattr(forward_batch, "extend_seq_lens", None)
    ext_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
    logger.warning(
        "[SPEC_LAYER] minimax_sparse TARGET_VERIFY pre_attn layer=%d bs=%d q_tokens=%d "
        "q=%s seq_lens_cpu_head=%s extend_seq_lens=%s extend_seq_lens_cpu=%s",
        layer_id,
        bs,
        int(q.shape[0]),
        _tensor_stats(q),
        list(forward_batch.seq_lens_cpu[:4])
        if forward_batch.seq_lens_cpu is not None
        else None,
        ext.detach().cpu().tolist()[:4] if ext is not None else None,
        ext_cpu[:4] if ext_cpu is not None else None,
    )
