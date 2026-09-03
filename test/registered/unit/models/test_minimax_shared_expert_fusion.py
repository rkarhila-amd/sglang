"""Tier-A CPU tests for MiniMax-M3 shared-expert fusion.

These pin two contracts that are easy to break silently when fusion is enabled:

1. ``load_weights`` must remap ``mlp.shared_experts.{gate,up,down}_proj*`` into the
   extra fused MoE expert slot (``experts.{num_local_experts}``) with the correct
   w1 / w3 / w2 shard ids.
2. MiniMax's TopK contract (``sqrtsoftplus``, ``apply_routed_scaling_factor_on_output``,
   one fused shared expert) must leave the shared column at unit weight so the fused
   expert contributes 1.0x after ``routed_scaling_factor`` scaling.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.topk import TopKConfig, biased_topk_impl, select_experts
from sglang.srt.models.minimax_m3 import MiniMaxM3SparseForCausalLM
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Representative MiniMax-M3 MoE routing shape (see minimax-dspark helpers).
_NUM_LOCAL_EXPERTS = 128
_NUM_EXPERTS_PER_TOK = 4
_NUM_FUSED_SHARED = 1
_TOP_K = _NUM_EXPERTS_PER_TOK + _NUM_FUSED_SHARED
_ROUTED_SCALING_FACTOR = 2.0
_LAYER_ID = 3


class _RecordingExpertParam:
    def __init__(self, full_name: str, records: list):
        self.full_name = full_name
        self.records = records

    def weight_loader(
        self,
        _param,
        loaded_weight,
        candidate_name,
        *,
        shard_id,
        expert_id,
    ):
        self.records.append(
            {
                "tensor": loaded_weight,
                "candidate_name": candidate_name,
                "shard_id": shard_id,
                "expert_id": expert_id,
                "param_name": self.full_name,
            }
        )


class _FusionLoadModel(nn.Module):
    """Minimal stand-in for ``MiniMaxM3SparseForCausalLM`` during ``load_weights``."""

    def __init__(self, *, layer_id: int = _LAYER_ID):
        super().__init__()
        self.num_fused_shared_experts = _NUM_FUSED_SHARED
        self.config = SimpleNamespace(
            num_local_experts=_NUM_LOCAL_EXPERTS,
            num_mtp_modules=0,
        )
        self.model = SimpleNamespace(start_layer=0, end_layer=32)
        self.records: list[dict] = []
        prefix = f"model.layers.{layer_id}.mlp"
        self._params = {
            f"{prefix}.experts.w13_weight": _RecordingExpertParam(
                f"{prefix}.experts.w13_weight", self.records
            ),
            f"{prefix}.experts.w13_weight_scale": _RecordingExpertParam(
                f"{prefix}.experts.w13_weight_scale", self.records
            ),
            f"{prefix}.experts.w2_weight": _RecordingExpertParam(
                f"{prefix}.experts.w2_weight", self.records
            ),
            f"{prefix}.experts.w2_weight_scale": _RecordingExpertParam(
                f"{prefix}.experts.w2_weight_scale", self.records
            ),
        }

    def named_parameters(self):
        return list(self._params.items())


def _minimax_topk_config() -> TopKConfig:
    return TopKConfig(
        top_k=_TOP_K,
        renormalize=True,
        num_fused_shared_experts=_NUM_FUSED_SHARED,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=_ROUTED_SCALING_FACTOR,
        apply_routed_scaling_factor_on_output=True,
        allow_routed_experts_capture=False,
    )


class TestMiniMaxSharedExpertFusionLoad(CustomTestCase):
    def _load(self, ckpt_name: str, tensor: torch.Tensor) -> list[dict]:
        model = _FusionLoadModel()
        MiniMaxM3SparseForCausalLM.load_weights(
            model,
            [(ckpt_name, tensor)],
        )
        return model.records

    def test_gate_proj_remaps_into_fused_w13_w1_shard(self):
        weight = torch.tensor([1.0])
        records = self._load(
            "model.layers.3.block_sparse_moe.shared_experts.gate_proj.weight",
            weight,
        )
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertIs(rec["tensor"], weight)
        self.assertEqual(rec["shard_id"], "w1")
        self.assertEqual(rec["expert_id"], _NUM_LOCAL_EXPERTS)
        self.assertEqual(
            rec["param_name"],
            "model.layers.3.mlp.experts.w13_weight",
        )

    def test_up_proj_remaps_into_fused_w13_w3_shard(self):
        weight = torch.tensor([2.0])
        records = self._load(
            "model.layers.3.mlp.shared_experts.up_proj.weight",
            weight,
        )
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec["shard_id"], "w3")
        self.assertEqual(rec["expert_id"], _NUM_LOCAL_EXPERTS)

    def test_down_proj_remaps_into_fused_w2_shard(self):
        weight = torch.tensor([3.0])
        records = self._load(
            "model.layers.3.mlp.shared_experts.down_proj.weight",
            weight,
        )
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec["shard_id"], "w2")
        self.assertEqual(rec["expert_id"], _NUM_LOCAL_EXPERTS)
        self.assertEqual(rec["param_name"], "model.layers.3.mlp.experts.w2_weight")

    def test_mxfp4_scales_follow_the_same_remap(self):
        gate_scale = torch.tensor([4.0])
        down_scale = torch.tensor([5.0])
        model = _FusionLoadModel()
        MiniMaxM3SparseForCausalLM.load_weights(
            model,
            [
                (
                    "model.layers.3.mlp.shared_experts.gate_proj.weight_scale",
                    gate_scale,
                ),
                (
                    "model.layers.3.mlp.shared_experts.down_proj.weight_scale",
                    down_scale,
                ),
            ],
        )
        self.assertEqual(len(model.records), 2)
        by_shard = {rec["shard_id"]: rec for rec in model.records}
        self.assertEqual(by_shard["w1"]["expert_id"], _NUM_LOCAL_EXPERTS)
        self.assertEqual(by_shard["w2"]["expert_id"], _NUM_LOCAL_EXPERTS)
        self.assertIs(by_shard["w1"]["tensor"], gate_scale)
        self.assertIs(by_shard["w2"]["tensor"], down_scale)


class TestMiniMaxSharedExpertFusionTopK(CustomTestCase):
    def _run_biased_topk_reference(self, *, num_tokens: int = 2):
        torch.manual_seed(0)
        hidden_states = torch.randn(num_tokens, 16)
        router_logits = torch.randn(num_tokens, _NUM_LOCAL_EXPERTS)
        correction_bias = torch.randn(_NUM_LOCAL_EXPERTS)
        routed_topk = _TOP_K - _NUM_FUSED_SHARED

        return biased_topk_impl(
            hidden_states=hidden_states,
            gating_output=router_logits,
            correction_bias=correction_bias,
            topk=routed_topk,
            renormalize=True,
            scoring_func="sqrtsoftplus",
            num_fused_shared_experts=_NUM_FUSED_SHARED,
            routed_scaling_factor=_ROUTED_SCALING_FACTOR,
            apply_routed_scaling_factor_on_output=True,
        )

    def test_fused_shared_column_weight_nets_to_unit(self):
        topk_weights, _topk_ids = self._run_biased_topk_reference()
        # On the aiter path ``select_experts`` passes routed-only topk here; the
        # shared expert reuses the last routed column before append.
        self.assertEqual(topk_weights.shape[-1], _TOP_K - _NUM_FUSED_SHARED)
        shared = topk_weights[:, -1]
        self.assertTrue(torch.allclose(shared, torch.ones_like(shared), atol=1e-5, rtol=0))

    def test_routed_columns_sum_to_routed_scaling_factor(self):
        topk_weights, _topk_ids = self._run_biased_topk_reference(num_tokens=1)
        routed_sum = topk_weights[0, :-1].sum()
        self.assertAlmostEqual(
            routed_sum.item(),
            _ROUTED_SCALING_FACTOR,
            places=4,
        )

    def test_select_experts_appends_shared_slot_on_aiter_path(self):
        """HIP aiter: routed topk then ``fused_append_shared_experts`` -> top_k columns."""

        def _routed_only_fused_gate(
            gating_output,
            correction_bias,
            *,
            topk,
            scoring_func,
            num_fused_shared_experts,
            renormalize,
            routed_scaling_factor,
            apply_routed_scaling_factor_on_output,
        ):
            hidden_states = torch.empty(
                gating_output.shape[0], 1, dtype=gating_output.dtype, device=gating_output.device
            )
            return biased_topk_impl(
                hidden_states=hidden_states,
                gating_output=gating_output,
                correction_bias=correction_bias,
                topk=topk,
                renormalize=renormalize,
                scoring_func=scoring_func,
                num_fused_shared_experts=0,
                routed_scaling_factor=routed_scaling_factor,
                apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
            )

        torch.manual_seed(1)
        device = get_device()
        num_tokens = 3
        hidden_states = torch.randn(num_tokens, 16, device=device)
        router_logits = torch.randn(num_tokens, _NUM_LOCAL_EXPERTS, device=device)
        correction_bias = torch.randn(_NUM_LOCAL_EXPERTS, device=device)
        topk_config = _minimax_topk_config()
        topk_config.correction_bias = correction_bias

        with (
            patch.object(topk_module, "_use_aiter", True),
            patch(
                "sglang.kernels.ops.moe.moe_fused_gate.moe_fused_gate",
                side_effect=_routed_only_fused_gate,
            ),
        ):
            out = select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_config=topk_config,
                layer_id=_LAYER_ID,
            )

        topk_weights = out.topk_weights
        self.assertEqual(topk_weights.shape, (num_tokens, _TOP_K))
        self.assertTrue(
            torch.allclose(topk_weights[:, -1], torch.ones(num_tokens, device=device), atol=1e-5, rtol=0)
        )
        self.assertTrue(
            torch.all(
                (out.topk_ids[:, -1] >= _NUM_LOCAL_EXPERTS)
                & (out.topk_ids[:, -1] < _NUM_LOCAL_EXPERTS + _NUM_FUSED_SHARED)
            )
        )


if __name__ == "__main__":
    unittest.main()
