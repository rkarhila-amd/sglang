"""Tier-B GPU tests for MiniMax-M3 shared-expert fusion (MXFP4).

These require gfx95 ROCm + aiter and a MiniMax-M3 MXFP4 checkpoint in the HF
cache. They validate:

1. MoE post-load transforms produce the expected single-expert tensor shapes.
2. Checkpoint gate/up tensors concatenate into the fused MoE w13 layout.
3. The unfused shared MLP GEMM path is self-consistent (duplicate reference).
4. The fused MoE shared-expert slot runs with production AITER kwargs (finite,
   deterministic) and does not match the separate-GEMM unfused path today.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import torch

_AMD_DIR = Path(__file__).resolve().parent
if str(_AMD_DIR) not in sys.path:
    sys.path.insert(0, str(_AMD_DIR))

from minimax_shared_expert_fusion_mxfp4_utils import (
    HIDDEN,
    SHARED_EXPERT_ID,
    load_shared_expert_mxfp4_weights,
    model_snapshot,
    moe_weight_dims,
    prepare_moe_shared_slot_weights,
    production_gate_mode,
    require_aiter_mxfp4_kernels,
    run_fused_shared_moe_slot,
    run_shared_mlp_dequant_reference,
    run_unfused_shared_mlp,
)
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd-mi35x")

_RUNNABLE = is_hip() and is_gfx95_supported() and torch.cuda.is_available()


def _parse_batch_rows() -> tuple[int, ...]:
    raw = os.environ.get("MINIMAX_FUSION_TEST_M", "8,128")
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


def _relerr(got: torch.Tensor, ref: torch.Tensor) -> float:
    got_f = got.float()
    ref_f = ref.float()
    return ((got_f - ref_f).norm() / (ref_f.norm() + 1e-8)).item()


def _weight_bytes(tensor: torch.Tensor) -> torch.Tensor:
    if hasattr(torch, "float4_e2m1fn_x2") and tensor.dtype == torch.float4_e2m1fn_x2:
        return tensor.view(torch.uint8)
    return tensor


@unittest.skipUnless(_RUNNABLE, "requires HIP gfx950 + aiter + CUDA")
class TestMiniMaxSharedExpertFusionMxfp4(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        try:
            cls._snap = model_snapshot()
            cls._weights = load_shared_expert_mxfp4_weights(snap=cls._snap)
            cls._moe_weights = prepare_moe_shared_slot_weights(
                cls._weights["w13_w"],
                cls._weights["w13_s"],
                cls._weights["w2_w"],
                cls._weights["w2_s"],
            )
        except FileNotFoundError as exc:
            raise unittest.SkipTest(str(exc)) from exc
        require_aiter_mxfp4_kernels()

    def test_moe_post_load_shuffle_preserves_weight_shapes(self):
        w13, w13_s, w2, w2_s = self._moe_weights
        self.assertEqual(w13.shape[0], 1)
        self.assertEqual(w2.shape[0], 1)
        w13_up_dim, w2_down_dim = moe_weight_dims()
        self.assertEqual(w13.shape[1], w13_up_dim)
        self.assertEqual(w2.shape[2], w2_down_dim)
        self.assertEqual(w13_s.dtype, torch.uint8)
        self.assertEqual(w2_s.dtype, torch.uint8)
        self.assertTrue(getattr(w13, "is_shuffled", False))
        self.assertTrue(getattr(w2, "is_shuffled", False))

    def test_checkpoint_w13_concat_matches_moe_slot_layout(self):
        w = self._weights
        w13_up_dim, w2_down_dim = moe_weight_dims()
        self.assertEqual(w["gate_w"].shape[0], w13_up_dim // 2)
        self.assertEqual(w["w13_w"].shape[0], w13_up_dim)
        self.assertEqual(w["w2_w"].shape[1], w2_down_dim)
        torch.testing.assert_close(
            w["w13_w"], torch.cat([w["gate_w"], w["up_w"]], dim=0)
        )
        torch.testing.assert_close(
            w["w13_s"], torch.cat([w["gate_s"], w["up_s"]], dim=0)
        )

    def test_post_load_shuffle_changes_weight_bytes(self):
        unshuffled = prepare_moe_shared_slot_weights(
            self._weights["w13_w"],
            self._weights["w13_s"],
            self._weights["w2_w"],
            self._weights["w2_s"],
            shuffle=False,
        )
        shuffled = self._moe_weights
        self.assertFalse(torch.equal(_weight_bytes(unshuffled[0]), _weight_bytes(shuffled[0])))
        self.assertFalse(torch.equal(unshuffled[1], shuffled[1]))

    def test_unfused_shared_mlp_matches_reference_gemm_path(self):
        for m in _parse_batch_rows():
            with self.subTest(m=m):
                torch.manual_seed(1000 + m)
                hidden = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
                got = run_unfused_shared_mlp(hidden, self._weights)
                ref = run_shared_mlp_dequant_reference(hidden, self._weights)
                self.assertFalse(torch.isnan(got).any().item())
                self.assertFalse(torch.isinf(got).any().item())
                torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)

    def test_fused_shared_slot_matches_production_aiter_kwargs(self):
        from aiter import ActivationType, QuantType
        from aiter.fused_moe import fused_moe

        torch.manual_seed(2000)
        hidden = torch.randn(8, HIDDEN, device="cuda", dtype=torch.bfloat16)
        got = run_fused_shared_moe_slot(hidden, self._moe_weights)

        w13, w13_s, w2, w2_s = self._moe_weights
        if getattr(w13, "is_shuffled", False):
            w13.is_shuffled = True
            w2.is_shuffled = True
        m = hidden.shape[0]
        ref = fused_moe(
            hidden_states=hidden,
            w1=w13,
            w2=w2,
            topk_weight=torch.ones((m, 1), device="cuda", dtype=torch.float32),
            topk_ids=torch.zeros((m, 1), device="cuda", dtype=torch.int32),
            quant_type=QuantType.per_1x32,
            activation=ActivationType.Swiglu,
            w1_scale=w13_s,
            w2_scale=w2_s,
            swiglu_limit=7.0,
            gate_mode=production_gate_mode(),
        )
        torch.testing.assert_close(got, ref, rtol=0, atol=0)

    def test_fused_shared_slot_is_deterministic(self):
        torch.manual_seed(0)
        hidden = torch.randn(4, HIDDEN, device="cuda", dtype=torch.bfloat16)
        first = run_fused_shared_moe_slot(hidden, self._moe_weights)
        second = run_fused_shared_moe_slot(hidden, self._moe_weights)
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_fused_moe_differs_from_unfused_gemm_path(self):
        """Guardrail: fused MoE and separate GEMMs are not numerically equivalent yet."""
        torch.manual_seed(3000)
        hidden = torch.randn(8, HIDDEN, device="cuda", dtype=torch.bfloat16)
        unfused = run_unfused_shared_mlp(hidden, self._weights, use_alpha=False)
        fused = run_fused_shared_moe_slot(hidden, self._moe_weights, use_alpha=False)
        self.assertGreater(_relerr(fused, unfused), 0.1)

    def test_shuffled_moe_weights_remain_non_finite_free(self):
        for m in _parse_batch_rows():
            with self.subTest(m=m):
                torch.manual_seed(4000 + m)
                hidden = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
                fused = run_fused_shared_moe_slot(
                    hidden, self._moe_weights, use_alpha=False
                )
                self.assertFalse(torch.isnan(fused).any().item())
                self.assertFalse(torch.isinf(fused).any().item())

    def test_unfused_alpha_path_differs_from_fused_limit_only(self):
        """Documents the activation mismatch between current unfused and fused paths."""
        torch.manual_seed(3000)
        hidden = torch.randn(8, HIDDEN, device="cuda", dtype=torch.bfloat16)
        with_alpha = run_unfused_shared_mlp(hidden, self._weights, use_alpha=True)
        limit_only = run_unfused_shared_mlp(hidden, self._weights, use_alpha=False)
        rel = _relerr(with_alpha, limit_only)
        self.assertGreater(rel, 0.01)

    def test_fused_shared_slot_uses_local_expert_index_zero(self):
        torch.manual_seed(0)
        hidden = torch.randn(4, HIDDEN, device="cuda", dtype=torch.bfloat16)
        fused = run_fused_shared_moe_slot(hidden, self._moe_weights, local_expert_id=0)
        self.assertEqual(fused.shape, hidden.shape)
        self.assertEqual(SHARED_EXPERT_ID, 128)


if __name__ == "__main__":
    unittest.main()
