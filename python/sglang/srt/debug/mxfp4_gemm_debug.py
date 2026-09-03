"""NaN/Inf logging for Quark MXFP4 GEMM kernels in real inference.

Enable with::

    export SGLANG_MXFP4_PRE_QUANT_DEBUG=1   # gemm_afp4wfp4_pre_quant only
    export SGLANG_MXFP4_GEMM_DEBUG=1          # also gemm_afp4wfp4 + act quant

Optional filters (same spirit as spec-verify debug)::

    export SGLANG_MXFP4_GEMM_MIN_M=16
    export SGLANG_MXFP4_GEMM_TARGET_M=32,128
    export SGLANG_MXFP4_GEMM_MAX_LOGS=2000
    export SGLANG_MXFP4_GEMM_LOG_EACH_LAYER=1   # one before-gemm line per layer at each M
    export SGLANG_MXFP4_GEMM_LOG_ALL=0      # set 1 to log every non-milestone row too

Logs are suppressed during CUDA graph capture (``.item()`` unsafe).
Non-finite rows always log when filters match.
"""
from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

import torch

logger = logging.getLogger(__name__)

_log_count = 0
_seen_bad: set[str] = set()
_layer_milestone_seen: set[str] = set()
_tls = threading.local()


@dataclass
class _Mxfp4GemmContext:
    layer_prefix: Optional[str] = None
    path: Optional[str] = None


def pre_quant_debug_enabled() -> bool:
    return os.environ.get("SGLANG_MXFP4_PRE_QUANT_DEBUG", "0") == "1"


def gemm_debug_enabled() -> bool:
    return pre_quant_debug_enabled() or os.environ.get(
        "SGLANG_MXFP4_GEMM_DEBUG", "0"
    ) == "1"


def _max_logs() -> int:
    return int(os.environ.get("SGLANG_MXFP4_GEMM_MAX_LOGS", "2000"))


def _min_m() -> int:
    return int(os.environ.get("SGLANG_MXFP4_GEMM_MIN_M", "8"))


def _target_m_set() -> set[int]:
    raw = os.environ.get("SGLANG_MXFP4_GEMM_TARGET_M", "")
    if not raw.strip():
        return set()
    return {int(x.strip()) for x in raw.split(",") if x.strip()}


def _log_all() -> bool:
    return os.environ.get("SGLANG_MXFP4_GEMM_LOG_ALL", "0") == "1"


def _log_each_layer() -> bool:
    return os.environ.get("SGLANG_MXFP4_GEMM_LOG_EACH_LAYER", "1") == "1"


def _in_cuda_graph_capture() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _m_allowed(m: int) -> bool:
    targets = _target_m_set()
    if targets:
        return m in targets
    return m >= _min_m()


def _tensor_nan_stats(t: torch.Tensor) -> dict:
    t = t.detach()
    if t.numel() == 0:
        return {"shape": list(t.shape), "dtype": str(t.dtype), "empty": True}
    if not t.is_floating_point():
        return {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "nan": 0,
            "inf": 0,
            "non_finite": 0,
            "note": "non_float_skipped",
        }
    nan = int(torch.isnan(t).sum().item())
    inf = int(torch.isinf(t).sum().item())
    non_finite = nan + inf
    out = {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "nan": nan,
        "inf": inf,
        "non_finite": non_finite,
    }
    if non_finite < t.numel():
        finite = torch.isfinite(t)
        if finite.any():
            tf = t[finite]
            out["min"] = float(tf.min().item())
            out["max"] = float(tf.max().item())
    return out


def _current_context() -> _Mxfp4GemmContext:
    ctx = getattr(_tls, "ctx", None)
    if ctx is None:
        return _Mxfp4GemmContext()
    return ctx


@contextmanager
def mxfp4_gemm_debug_context(
    *,
    layer_prefix: Optional[str] = None,
    path: Optional[str] = None,
) -> Iterator[None]:
    prev = getattr(_tls, "ctx", None)
    _tls.ctx = _Mxfp4GemmContext(layer_prefix=layer_prefix, path=path)
    try:
        yield
    finally:
        if prev is None:
            delattr(_tls, "ctx")
        else:
            _tls.ctx = prev


def _should_emit(
    *,
    kernel: str,
    phase: str,
    m: int,
    stats: dict,
    layer_prefix: Optional[str],
) -> bool:
    global _log_count
    if _log_count >= _max_logs():
        return False
    if not _m_allowed(m):
        return False
    non_finite = int(stats.get("non_finite", 0))
    if non_finite > 0:
        key = f"{kernel}|{phase}|{layer_prefix}|M={m}|bad"
        if key in _seen_bad:
            return False
        _seen_bad.add(key)
        return True
    return _log_all()


def _emit(
    *,
    kernel: str,
    phase: str,
    m: int,
    stats: dict,
    layer_prefix: Optional[str] = None,
    path: Optional[str] = None,
    tensor: Optional[torch.Tensor] = None,
) -> None:
    global _log_count
    if _in_cuda_graph_capture():
        return
    ctx = _current_context()
    layer_prefix = layer_prefix or ctx.layer_prefix
    path = path or ctx.path
    if not _should_emit(
        kernel=kernel, phase=phase, m=m, stats=stats, layer_prefix=layer_prefix
    ):
        return
    _log_count += 1
    level = logging.WARNING if stats.get("non_finite", 0) else logging.INFO
    extra = ""
    if stats.get("non_finite", 0) and tensor is not None and tensor.dim() >= 1:
        row_bad = (~torch.isfinite(tensor.detach().reshape(tensor.shape[0], -1))).any(
            dim=1
        )
        bad_rows = torch.nonzero(row_bad, as_tuple=False).flatten().tolist()[:32]
        extra = f" bad_rows={bad_rows}"
    logger.log(
        level,
        "[MXFP4_GEMM] %s %s M=%d layer=%s path=%s stats=%s%s",
        kernel,
        phase,
        m,
        layer_prefix or "?",
        path or "?",
        stats,
        extra,
    )


def log_mxfp4_per_layer_before_gemm(
    x: torch.Tensor,
    *,
    kernel: str,
    layer_prefix: Optional[str] = None,
    path: Optional[str] = None,
) -> None:
    """Log bf16 (or pre-gemm) activations once per layer before MXFP4 GEMM.

    ``kernel`` is ``gemm_afp4wfp4_pre_quant`` or ``gemm_afp4wfp4``.
    With ``SGLANG_MXFP4_GEMM_LOG_EACH_LAYER=1`` (default), emits one line per
    (layer, kernel, M) even when finite. Non-finite rows always log.
    """
    if not gemm_debug_enabled() or _in_cuda_graph_capture():
        return
    if x.dim() < 1:
        return
    m = int(x.shape[0])
    if not _m_allowed(m):
        return
    ctx = _current_context()
    layer_prefix = layer_prefix or ctx.layer_prefix
    path = path or ctx.path
    stats = _tensor_nan_stats(x)
    non_finite = int(stats.get("non_finite", 0))
    milestone_key = f"milestone|{kernel}|before|{layer_prefix}|M={m}"
    if non_finite == 0:
        if not _log_each_layer():
            return
        if milestone_key in _layer_milestone_seen:
            return
        _layer_milestone_seen.add(milestone_key)
    _emit(
        kernel=kernel,
        phase="before",
        m=m,
        stats=stats,
        layer_prefix=layer_prefix,
        path=path,
    )


def log_mxfp4_pre_quant_boundary(
    phase: str,
    *,
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    layer_prefix: Optional[str] = None,
    path: Optional[str] = None,
) -> None:
    if not pre_quant_debug_enabled() or _in_cuda_graph_capture():
        return
    m = int(x.shape[0]) if x.dim() >= 1 else 0
    if phase == "before":
        _emit(
            kernel="gemm_afp4wfp4_pre_quant",
            phase="before",
            m=m,
            stats=_tensor_nan_stats(x),
            layer_prefix=layer_prefix,
            path=path,
        )
    elif phase == "after" and y is not None:
        _emit(
            kernel="gemm_afp4wfp4_pre_quant",
            phase="after",
            m=m,
            stats=_tensor_nan_stats(y),
            layer_prefix=layer_prefix,
            path=path,
        )


def log_mxfp4_gemm_boundary(
    phase: str,
    *,
    kernel: str,
    t: torch.Tensor,
    layer_prefix: Optional[str] = None,
    path: Optional[str] = None,
) -> None:
    if not gemm_debug_enabled() or _in_cuda_graph_capture():
        return
    m = int(t.shape[0]) if t.dim() >= 1 else 0
    _emit(
        kernel=kernel,
        phase=phase,
        m=m,
        stats=_tensor_nan_stats(t),
        layer_prefix=layer_prefix,
        path=path,
        tensor=t,
    )
