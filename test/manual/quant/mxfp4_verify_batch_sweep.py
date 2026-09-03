"""MXFP4 kernel batch-row sweep for EAGLE TARGET_VERIFY shapes.

Exercises the MXFP4-only ops that differ from the working MXFP8 path:

1. ``dynamic_mxfp4_quant`` — activation quant (per-forward)
2. ``gemm_afp4wfp4`` / ``gemm_afp4wfp4_pre_quant`` — Quark dense linear
3. ``aiter.fused_moe`` with ``QuantType.per_1x32`` + clamped SwiGLU (MiniMax-M3)

Default row counts ``M`` are ``{8, 16, 32, 64, 128}``. ``M=128`` matches
TARGET_VERIFY at ``cuda_graph_max_bs=32`` with 4 draft tokens per sequence
(``32 * 4`` token rows).

Each kernel is checked for:

- finite outputs (no NaN/Inf)
- agreement with a dequant / PyTorch reference within tolerance

Requires gfx95 (MI35x) + aiter. Skips automatically on other hardware.

Run (when a GPU is free)::

    cd python && python -m pytest ../test/manual/quant/mxfp4_verify_batch_sweep.py -v -s

Or from minimax-dspark::

    bash run_smci355_mxfp4_kernel_sweep.sh
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Iterable

import pytest
import torch

from sglang.srt.layers.moe.utils import get_moe_weight_sizes
from sglang.srt.layers.quantization.quark.utils import e8m0_to_f32, mxfp4_to_f32
from sglang.srt.utils.common import is_gfx95_supported, is_hip

# MiniMax-M3-MXFP4 production dims (from smci355 MOE_PATH logs).
MINIMAX_HIDDEN_SIZE = 6144
MINIMAX_INTERMEDIATE_SIZE = 6144 // 8  # 768
MINIMAX_NUM_EXPERTS = 128
MINIMAX_TOP_K = 4
MINIMAX_SWIGLU_LIMIT = float(os.environ.get("SGLANG_MXFP4_SWEEP_SWIGLU_LIMIT", "7.0"))

# EAGLE TARGET_VERIFY row counts: bs * draft_tokens; 32*4 = 128 is the failing case.
DEFAULT_BATCH_ROWS = (8, 16, 32, 64, 128)

OCP_MX_BLOCK_SIZE = 32
_GEMM_RTOL = 0.08
_GEMM_ATOL = 0.08
_MOE_RTOL = 0.12
_MOE_ATOL = 0.12
_QUANT_RTOL = 1.0
_QUANT_ATOL = 1.0

pytestmark = pytest.mark.skipif(
    not (is_hip() and is_gfx95_supported() and torch.cuda.is_available()),
    reason="MXFP4 verify batch sweep requires gfx95 ROCm + CUDA device",
)


def _parse_batch_rows() -> tuple[int, ...]:
    raw = os.environ.get("SGLANG_MXFP4_SWEEP_M", "")
    if not raw.strip():
        return DEFAULT_BATCH_ROWS
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


BATCH_ROWS = _parse_batch_rows()


@dataclass(frozen=True)
class SweepResult:
    kernel: str
    m: int
    ok: bool
    max_abs_err: float
    nan_count: int
    inf_count: int
    note: str = ""

    def as_line(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        return (
            f"{status}  {self.kernel:24s}  M={self.m:4d}  "
            f"max_abs_err={self.max_abs_err:.4e}  nan={self.nan_count}  inf={self.inf_count}"
            + (f"  ({self.note})" if self.note else "")
        )


def _require_aiter_kernels():
    try:
        from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
        from aiter.ops.triton.gemm_afp4wfp4_pre_quant_atomic import (
            gemm_afp4wfp4_pre_quant,
        )
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
    except ImportError as exc:
        pytest.skip(f"aiter MXFP4 kernels unavailable: {exc}")
    return dynamic_mxfp4_quant, gemm_afp4wfp4, gemm_afp4wfp4_pre_quant


def _dequant_mxfp4_matrix(x_q: torch.Tensor, x_s: torch.Tensor) -> torch.Tensor:
    """Dequantize packed MXFP4 weights/activations to fp32."""
    if hasattr(torch, "float4_e2m1fn_x2") and x_q.dtype == torch.float4_e2m1fn_x2:
        x_q = x_q.view(torch.uint8)
    x_f32 = mxfp4_to_f32(x_q, is_3d=x_q.dim() == 3)
    scales = x_s.repeat_interleave(OCP_MX_BLOCK_SIZE, dim=-1).to(torch.float32)
    return x_f32 * e8m0_to_f32(scales)


def _torch_gemm_mxfp4_ref(
    x_bf16: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    x_q, x_s = _require_aiter_kernels()[0](x_bf16)
    x_f32 = _dequant_mxfp4_matrix(x_q, x_s)
    w_f32 = _dequant_mxfp4_matrix(w_q, w_s)
    return torch.mm(x_f32, w_f32.T).to(out_dtype)


def _finite_stats(t: torch.Tensor) -> tuple[int, int]:
    if t.numel() == 0:
        return 0, 0
    nan_count = int(torch.isnan(t).sum().item())
    inf_count = int(torch.isinf(t).sum().item())
    return nan_count, inf_count


def _check_tensor(
    kernel: str,
    m: int,
    got: torch.Tensor,
    ref: torch.Tensor,
    rtol: float,
    atol: float,
    note: str = "",
) -> SweepResult:
    nan_count, inf_count = _finite_stats(got)
    if nan_count or inf_count:
        return SweepResult(kernel, m, False, float("inf"), nan_count, inf_count, note)
    if got.shape != ref.shape:
        return SweepResult(
            kernel,
            m,
            False,
            float("inf"),
            nan_count,
            inf_count,
            f"shape mismatch got={tuple(got.shape)} ref={tuple(ref.shape)}",
        )
    diff = (got.float() - ref.float()).abs()
    max_abs_err = float(diff.max().item()) if diff.numel() else 0.0
    ok = bool(torch.allclose(got, ref, rtol=rtol, atol=atol))
    return SweepResult(kernel, m, ok, max_abs_err, nan_count, inf_count, note)


def sweep_dynamic_mxfp4_quant(
    m_values: Iterable[int] = BATCH_ROWS,
    k: int = MINIMAX_HIDDEN_SIZE,
    seed: int = 0,
) -> list[SweepResult]:
    dynamic_mxfp4_quant, _, _ = _require_aiter_kernels()
    device = "cuda"
    results: list[SweepResult] = []
    for m in m_values:
        torch.manual_seed(seed + m)
        x_bf16 = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        x_q, x_s = dynamic_mxfp4_quant(x_bf16)
        nan_count, inf_count = _finite_stats(x_s)
        if nan_count or inf_count:
            results.append(
                SweepResult("dynamic_mxfp4_quant", m, False, float("inf"), nan_count, inf_count)
            )
            continue
        x_roundtrip = _dequant_mxfp4_matrix(x_q, x_s).to(torch.bfloat16)
        nan_rt, inf_rt = _finite_stats(x_roundtrip)
        if nan_rt or inf_rt:
            results.append(
                SweepResult(
                    "dynamic_mxfp4_quant",
                    m,
                    False,
                    float("inf"),
                    nan_rt,
                    inf_rt,
                    note="roundtrip non-finite",
                )
            )
            continue
        results.append(
            _check_tensor(
                "dynamic_mxfp4_quant",
                m,
                x_roundtrip,
                x_bf16,
                _QUANT_RTOL,
                _QUANT_ATOL,
                note="bf16 roundtrip",
            )
        )
    return results


def sweep_gemm_afp4wfp4(
    m_values: Iterable[int] = BATCH_ROWS,
    n: int = MINIMAX_INTERMEDIATE_SIZE * 2,
    k: int = MINIMAX_HIDDEN_SIZE,
    seed: int = 0,
) -> list[SweepResult]:
    dynamic_mxfp4_quant, gemm_afp4wfp4, _ = _require_aiter_kernels()
    device = "cuda"
    dtype = torch.bfloat16
    results: list[SweepResult] = []

    torch.manual_seed(seed)
    w_bf16 = torch.randn(n, k, device=device, dtype=dtype)
    w_q, w_s = dynamic_mxfp4_quant(w_bf16)

    for m in m_values:
        torch.manual_seed(seed + 10_000 + m)
        x_bf16 = torch.randn(m, k, device=device, dtype=dtype)
        x_q, x_s = dynamic_mxfp4_quant(x_bf16)
        y = torch.empty(m, n, device=device, dtype=dtype)
        gemm_afp4wfp4(x_q, w_q, x_s, w_s, dtype, y)
        ref = _torch_gemm_mxfp4_ref(x_bf16, w_q, w_s, dtype)
        results.append(_check_tensor("gemm_afp4wfp4", m, y, ref, _GEMM_RTOL, _GEMM_ATOL))
    return results


def sweep_gemm_afp4wfp4_pre_quant(
    m_values: Iterable[int] = BATCH_ROWS,
    n: int = MINIMAX_INTERMEDIATE_SIZE * 2,
    k: int = MINIMAX_HIDDEN_SIZE,
    seed: int = 0,
) -> list[SweepResult]:
    """``gemm_afp4wfp4_pre_quant`` path (fused quant inside GEMM)."""
    dynamic_mxfp4_quant, gemm_afp4wfp4, gemm_afp4wfp4_pre_quant = _require_aiter_kernels()
    device = "cuda"
    dtype = torch.bfloat16
    results: list[SweepResult] = []

    torch.manual_seed(seed)
    w_bf16 = torch.randn(n, k, device=device, dtype=dtype)
    w_q, w_s = dynamic_mxfp4_quant(w_bf16)

    for m in m_values:
        torch.manual_seed(seed + 20_000 + m)
        x_bf16 = torch.randn(m, k, device=device, dtype=dtype)
        y = torch.empty(m, n, device=device, dtype=dtype)
        gemm_afp4wfp4_pre_quant(x_bf16, w_q, w_s, dtype, y)
        x_q, x_s = dynamic_mxfp4_quant(x_bf16)
        y_ref = torch.empty(m, n, device=device, dtype=dtype)
        gemm_afp4wfp4(x_q, w_q, x_s, w_s, dtype, y_ref)
        results.append(
            _check_tensor(
                "gemm_afp4wfp4_pre_quant",
                m,
                y,
                y_ref,
                _GEMM_RTOL,
                _GEMM_ATOL,
                note="vs explicit quant+gemm",
            )
        )
    return results


def _minimax_moe_weight_shapes() -> tuple[int, int, int]:
    w13_up_dim, w2_down_dim, _ = get_moe_weight_sizes(
        MINIMAX_INTERMEDIATE_SIZE,
        is_concat=True,
        is_packed=True,
        is_aiter_moe=True,
    )
    return w13_up_dim, w2_down_dim, MINIMAX_HIDDEN_SIZE


def _prepare_minimax_moe_weights(
    num_experts: int,
    seed: int,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build shuffled MXFP4 MoE weights matching ``QuarkW4A4MXFp4MoE`` post-load."""
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle

    dynamic_mxfp4_quant, _, _ = _require_aiter_kernels()
    w13_up_dim, w2_down_dim, hidden = _minimax_moe_weight_shapes()

    torch.manual_seed(seed)
    w13_list = []
    w13_scale_list = []
    w2_list = []
    w2_scale_list = []
    for _ in range(num_experts):
        w13_bf16 = torch.randn(w13_up_dim, hidden, device=device, dtype=torch.bfloat16)
        w2_bf16 = torch.randn(hidden, w2_down_dim * 2, device=device, dtype=torch.bfloat16)
        w13_q, w13_s = dynamic_mxfp4_quant(w13_bf16)
        w2_q, w2_s = dynamic_mxfp4_quant(w2_bf16)
        w13_list.append(w13_q)
        w13_scale_list.append(w13_s)
        w2_list.append(w2_q)
        w2_scale_list.append(w2_s)

    w13_weight = torch.stack(w13_list)
    w13_weight_scale = torch.stack(w13_scale_list)
    w2_weight = torch.stack(w2_list)
    w2_weight_scale = torch.stack(w2_scale_list)

    # Mirror QuarkW4A4MXFp4MoE.process_weights_after_loading.
    s0, s1, _ = w13_weight_scale.shape
    w13_weight_scale = e8m0_shuffle(w13_weight_scale.view(s0 * s1, -1)).view(
        s0, s1, -1
    )
    s0, s1, _ = w2_weight_scale.shape
    w2_weight_scale = e8m0_shuffle(w2_weight_scale.view(s0 * s1, -1)).view(s0, s1, -1)
    w13_weight = shuffle_weight(w13_weight.contiguous(), (16, 16))
    w2_weight = shuffle_weight(w2_weight.contiguous(), (16, 16))

    if hasattr(torch, "float4_e2m1fn_x2"):
        w13_weight = w13_weight.view(torch.float4_e2m1fn_x2)
        w2_weight = w2_weight.view(torch.float4_e2m1fn_x2)

    return w13_weight, w13_weight_scale, w2_weight, w2_weight_scale


def _torch_clamped_swiglu_moe_ref(
    hidden_bf16: torch.Tensor,
    w13_q: torch.Tensor,
    w13_s: torch.Tensor,
    w2_q: torch.Tensor,
    w2_s: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    swiglu_limit: float,
) -> torch.Tensor:
    """Slow per-token reference for SEPARATED gate/up MXFP4 MoE."""
    m, k = hidden_bf16.shape
    top_k = topk_ids.shape[1]
    out = torch.zeros(m, k, device=hidden_bf16.device, dtype=torch.bfloat16)

    w13_f32 = _dequant_mxfp4_matrix(w13_q, w13_s)
    w2_f32 = _dequant_mxfp4_matrix(w2_q, w2_s)

    for row in range(m):
        acc = torch.zeros(k, device=hidden_bf16.device, dtype=torch.float32)
        h = hidden_bf16[row].float()
        for j in range(top_k):
            e = int(topk_ids[row, j].item())
            w = float(topk_weights[row, j].item())
            gemm1 = h @ w13_f32[e].T
            half = gemm1.shape[0] // 2
            gate, up = gemm1[:half], gemm1[half:]
            if swiglu_limit > 0:
                gate = gate.clamp(max=swiglu_limit)
                up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
            mid = torch.nn.functional.silu(gate) * up
            acc = acc + w * (mid @ w2_f32[e].T)
        out[row] = acc.to(torch.bfloat16)
    return out


def _moe_gate_mode() -> str:
    from sglang.srt.environ import envs

    if envs.SGLANG_USE_AITER_MOE_GU_ITLV.get():
        from aiter.ops.flydsl.moe_common import GateMode

        return GateMode.INTERLEAVE.value
    from aiter.ops.flydsl.moe_common import GateMode

    return GateMode.SEPARATED.value


def sweep_fused_moe_mxfp4(
    m_values: Iterable[int] = BATCH_ROWS,
    num_experts: int = MINIMAX_NUM_EXPERTS,
    top_k: int = MINIMAX_TOP_K,
    swiglu_limit: float = MINIMAX_SWIGLU_LIMIT,
    seed: int = 0,
    *,
    reference: bool = True,
) -> list[SweepResult]:
    """``aiter.fused_moe`` PER_1X32 path used by MiniMax-M3 MXFP4 MoE."""
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    device = "cuda"
    results: list[SweepResult] = []
    w13, w13_s, w2, w2_s = _prepare_minimax_moe_weights(num_experts, seed, device)

    # Reference MoE is O(M * top_k * E); use fewer experts when comparing accuracy.
    ref_experts = int(os.environ.get("SGLANG_MXFP4_SWEEP_MOE_REF_EXPERTS", "8"))
    if reference and ref_experts < num_experts:
        w13_ref, w13_s_ref, w2_ref, w2_s_ref = _prepare_minimax_moe_weights(
            ref_experts, seed + 99, device
        )
    else:
        w13_ref, w13_s_ref, w2_ref, w2_s_ref = w13, w13_s, w2, w2_s
        ref_experts = num_experts

    gate_mode = _moe_gate_mode()

    for m in m_values:
        torch.manual_seed(seed + 30_000 + m)
        hidden = torch.randn(m, MINIMAX_HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
        logits = torch.randn(m, num_experts, device=device, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(torch.softmax(logits, dim=-1), k=top_k, dim=-1)
        topk_weights = topk_weights.to(torch.float32)
        topk_ids = topk_ids.to(torch.int32)

        if ref_experts < num_experts:
            topk_ids = topk_ids % ref_experts

        out = fused_moe(
            hidden_states=hidden,
            w1=w13_ref if reference else w13,
            w2=w2_ref if reference else w2,
            topk_weight=topk_weights,
            topk_ids=topk_ids,
            quant_type=QuantType.per_1x32,
            activation=ActivationType.Swiglu,
            w1_scale=w13_s_ref if reference else w13_s,
            w2_scale=w2_s_ref if reference else w2_s,
            swiglu_limit=swiglu_limit,
            gate_mode=gate_mode,
        )

        nan_count, inf_count = _finite_stats(out)
        if nan_count or inf_count:
            results.append(
                SweepResult(
                    "fused_moe_mxfp4",
                    m,
                    False,
                    float("inf"),
                    nan_count,
                    inf_count,
                    note=f"E={num_experts} gate_mode={gate_mode}",
                )
            )
            continue

        if reference:
            ref = _torch_clamped_swiglu_moe_ref(
                hidden,
                w13_ref,
                w13_s_ref,
                w2_ref,
                w2_s_ref,
                topk_ids,
                topk_weights,
                swiglu_limit,
            )
            results.append(
                _check_tensor(
                    "fused_moe_mxfp4",
                    m,
                    out,
                    ref,
                    _MOE_RTOL,
                    _MOE_ATOL,
                    note=f"E={ref_experts} gate_mode={gate_mode}",
                )
            )
        else:
            results.append(
                SweepResult(
                    "fused_moe_mxfp4",
                    m,
                    True,
                    0.0,
                    0,
                    0,
                    note=f"finite-only E={num_experts} gate_mode={gate_mode}",
                )
            )

    # Second pass at full expert count: NaN probe only (too slow for py ref).
    if reference and num_experts > ref_experts:
        for m in m_values:
            torch.manual_seed(seed + 40_000 + m)
            hidden = torch.randn(
                m, MINIMAX_HIDDEN_SIZE, device=device, dtype=torch.bfloat16
            )
            logits = torch.randn(m, num_experts, device=device, dtype=torch.float32)
            topk_weights, topk_ids = torch.topk(
                torch.softmax(logits, dim=-1), k=top_k, dim=-1
            )
            out = fused_moe(
                hidden_states=hidden,
                w1=w13,
                w2=w2,
                topk_weight=topk_weights.to(torch.float32),
                topk_ids=topk_ids.to(torch.int32),
                quant_type=QuantType.per_1x32,
                activation=ActivationType.Swiglu,
                w1_scale=w13_s,
                w2_scale=w2_s,
                swiglu_limit=swiglu_limit,
                gate_mode=gate_mode,
            )
            nan_count, inf_count = _finite_stats(out)
            ok = nan_count == 0 and inf_count == 0
            results.append(
                SweepResult(
                    "fused_moe_mxfp4_fullE",
                    m,
                    ok,
                    0.0,
                    nan_count,
                    inf_count,
                    note=f"finite-only E={num_experts}",
                )
            )
    return results


def run_full_sweep(print_table: bool = True) -> list[SweepResult]:
    """Run all sweeps and optionally print a summary table."""
    all_results: list[SweepResult] = []
    all_results.extend(sweep_dynamic_mxfp4_quant())
    all_results.extend(sweep_gemm_afp4wfp4())
    all_results.extend(sweep_gemm_afp4wfp4_pre_quant())
    all_results.extend(sweep_fused_moe_mxfp4())
    if print_table:
        print("\n=== MXFP4 verify batch sweep ===")
        print(f"M values: {BATCH_ROWS}")
        print(f"hidden={MINIMAX_HIDDEN_SIZE} inter={MINIMAX_INTERMEDIATE_SIZE}")
        print(f"swiglu_limit={MINIMAX_SWIGLU_LIMIT} gate_mode={_moe_gate_mode()}")
        for row in all_results:
            print(row.as_line())
        failed = [r for r in all_results if not r.ok]
        print(f"\n{len(all_results) - len(failed)}/{len(all_results)} passed")
        if failed:
            print("Failures:")
            for row in failed:
                print(f"  {row.as_line()}")
    return all_results


# ---------------------------------------------------------------------------
# Pytest entry points (one test per kernel family)
# ---------------------------------------------------------------------------


class TestMxfp4VerifyBatchSweep:
    def test_dynamic_mxfp4_quant_sweep(self):
        results = sweep_dynamic_mxfp4_quant()
        assert all(r.ok for r in results), "\n".join(r.as_line() for r in results if not r.ok)

    def test_gemm_afp4wfp4_sweep(self):
        results = sweep_gemm_afp4wfp4()
        assert all(r.ok for r in results), "\n".join(r.as_line() for r in results if not r.ok)

    def test_gemm_afp4wfp4_pre_quant_sweep(self):
        results = sweep_gemm_afp4wfp4_pre_quant()
        assert all(r.ok for r in results), "\n".join(r.as_line() for r in results if not r.ok)

    def test_fused_moe_mxfp4_sweep(self):
        results = sweep_fused_moe_mxfp4()
        assert all(r.ok for r in results), "\n".join(r.as_line() for r in results if not r.ok)


if __name__ == "__main__":
    results = run_full_sweep(print_table=True)
    sys.exit(0 if all(r.ok for r in results) else 1)
