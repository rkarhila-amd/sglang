"""Helpers for MiniMax-M3 shared-expert fusion MXFP4 GPU tests (Tier B)."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from typing import Optional

import torch

from sglang.srt.layers.moe.utils import get_moe_weight_sizes
from sglang.srt.layers.quantization.quark.utils import e8m0_to_f32, mxfp4_to_f32

LAYER = int(os.environ.get("MINIMAX_FUSION_TEST_LAYER", "3"))
HIDDEN = 6144
INTERMEDIATE = 3072
NUM_ROUTED_EXPERTS = 128
SHARED_EXPERT_ID = NUM_ROUTED_EXPERTS
SWIGLU_ALPHA = 1.702
SWIGLU_LIMIT = 7.0
OCP_MX_BLOCK_SIZE = 32


def model_snapshot() -> Path:
    cache = Path(os.environ.get("HF_HUB_CACHE", "/hf-cache/hub"))
    for repo in (
        "amd--MiniMax-M3-MXFP4",
        "MiniMaxAI--MiniMax-M3-MXFP4",
        "MiniMaxAI--MiniMax-M3",
    ):
        snaps = cache / f"models--{repo}" / "snapshots"
        if snaps.is_dir():
            return sorted(snaps.iterdir())[-1]
    raise FileNotFoundError(
        "MiniMax-M3 MXFP4 checkpoint not found in HF cache "
        f"(searched under {cache})"
    )


def load_weight_index(snap: Path) -> dict:
    index_path = snap / "model.safetensors.index.json"
    if index_path.is_file():
        return json.loads(index_path.read_text())
    single = snap / "model.safetensors"
    if single.is_file():
        return {"weight_map": {"__single__": "model.safetensors"}}
    raise FileNotFoundError(f"No safetensors index in {snap}")


def get_checkpoint_tensor(snap: Path, index: dict, name: str) -> torch.Tensor:
    from safetensors import safe_open

    shard = index["weight_map"][name]
    if shard == "__single__":
        shard = "model.safetensors"
    with safe_open(str(snap / shard), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def shard_tensor(
    tensor: torch.Tensor, tp_rank: int, tp_size: int, dim: int = 0
) -> torch.Tensor:
    if tp_size <= 1:
        return tensor
    size = tensor.shape[dim]
    if size % tp_size != 0:
        raise ValueError(f"dim {dim} size {size} not divisible by tp_size {tp_size}")
    chunk = size // tp_size
    start = tp_rank * chunk
    end = (tp_rank + 1) * chunk
    if dim == 0:
        return tensor[start:end]
    if dim == 1:
        return tensor[:, start:end]
    raise ValueError(f"unsupported shard dim {dim}")


def shared_expert_prefix(layer: int = LAYER) -> str:
    return f"language_model.model.layers.{layer}.block_sparse_moe.shared_experts"


def load_shared_expert_mxfp4_weights(
    *,
    layer: int = LAYER,
    tp_rank: int = 0,
    tp_size: int = 1,
    device: str = "cuda",
    snap: Optional[Path] = None,
    index: Optional[dict] = None,
) -> dict[str, torch.Tensor]:
    snap = snap or model_snapshot()
    index = index or load_weight_index(snap)
    prefix = shared_expert_prefix(layer)

    def load(name_suffix: str) -> torch.Tensor:
        key = f"{prefix}.{name_suffix}"
        if key not in index["weight_map"]:
            raise KeyError(f"Missing checkpoint tensor {key}")
        return get_checkpoint_tensor(snap, index, key)

    gate_w = shard_tensor(load("gate_proj.weight"), tp_rank, tp_size, dim=0)
    gate_s = shard_tensor(load("gate_proj.weight_scale"), tp_rank, tp_size, dim=0)
    up_w = shard_tensor(load("up_proj.weight"), tp_rank, tp_size, dim=0)
    up_s = shard_tensor(load("up_proj.weight_scale"), tp_rank, tp_size, dim=0)
    down_w = shard_tensor(load("down_proj.weight"), tp_rank, tp_size, dim=1)
    down_s = shard_tensor(load("down_proj.weight_scale"), tp_rank, tp_size, dim=1)

    return {
        "gate_w": gate_w.to(device),
        "gate_s": gate_s.to(device),
        "up_w": up_w.to(device),
        "up_s": up_s.to(device),
        "down_w": down_w.to(device),
        "down_s": down_s.to(device),
        "w13_w": torch.cat([gate_w, up_w], dim=0).to(device),
        "w13_s": torch.cat([gate_s, up_s], dim=0).to(device),
        "w2_w": down_w.to(device),
        "w2_s": down_s.to(device),
    }


def require_aiter_mxfp4_kernels():
    try:
        from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.shuffle import shuffle_weight
        from aiter.utility.fp4_utils import e8m0_shuffle
    except ImportError as exc:
        raise unittest.SkipTest(f"aiter MXFP4 kernels unavailable: {exc}") from exc
    return gemm_afp4wfp4, dynamic_mxfp4_quant, shuffle_weight, e8m0_shuffle


def dequant_mxfp4_matrix(x_q: torch.Tensor, x_s: torch.Tensor) -> torch.Tensor:
    if hasattr(torch, "float4_e2m1fn_x2") and x_q.dtype == torch.float4_e2m1fn_x2:
        x_q = x_q.view(torch.uint8)
    x_f32 = mxfp4_to_f32(x_q, is_3d=x_q.dim() == 3)
    scales = x_s.repeat_interleave(OCP_MX_BLOCK_SIZE, dim=-1).to(torch.float32)
    return x_f32 * e8m0_to_f32(scales)


def prepare_moe_shared_slot_weights(
    w13_w: torch.Tensor,
    w13_s: torch.Tensor,
    w2_w: torch.Tensor,
    w2_s: torch.Tensor,
    *,
    shuffle: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the same post-load transforms as ``QuarkW4A4MXFp4MoE`` (single expert)."""
    w13 = w13_w.unsqueeze(0).contiguous()
    w2 = w2_w.unsqueeze(0).contiguous()
    w13_scale = w13_s.unsqueeze(0).contiguous()
    w2_scale = w2_s.unsqueeze(0).contiguous()

    if not shuffle:
        if hasattr(torch, "float4_e2m1fn_x2") and w13.dtype == torch.uint8:
            w13 = w13.view(torch.float4_e2m1fn_x2)
            w2 = w2.view(torch.float4_e2m1fn_x2)
        return w13, w13_scale, w2, w2_scale

    _, _, shuffle_weight, e8m0_shuffle = require_aiter_mxfp4_kernels()

    s0, s1, _ = w13_scale.shape
    w13_scale = e8m0_shuffle(w13_scale.view(s0 * s1, -1)).view(s0, s1, -1)
    s0, s1, _ = w2_scale.shape
    w2_scale = e8m0_shuffle(w2_scale.view(s0 * s1, -1)).view(s0, s1, -1)

    w13 = shuffle_weight(w13.contiguous(), (16, 16))
    w2 = shuffle_weight(w2.contiguous(), (16, 16))

    if hasattr(torch, "float4_e2m1fn_x2"):
        w13 = w13.view(torch.float4_e2m1fn_x2)
        w2 = w2.view(torch.float4_e2m1fn_x2)

    w13.is_shuffled = True
    w2.is_shuffled = True
    return w13, w13_scale, w2, w2_scale


def moe_weight_dims(intermediate: int = INTERMEDIATE) -> tuple[int, int]:
    w13_up_dim, w2_down_dim, _ = get_moe_weight_sizes(
        intermediate,
        is_concat=True,
        is_packed=True,
        is_aiter_moe=True,
    )
    return w13_up_dim, w2_down_dim


def swiglu_shared_activation(gate_up: torch.Tensor, *, use_alpha: bool = True) -> torch.Tensor:
    if use_alpha:
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            swiglu_no_interleaved_with_alpha_and_limit,
        )

        return swiglu_no_interleaved_with_alpha_and_limit(
            gate_up, SWIGLU_ALPHA, SWIGLU_LIMIT
        )

    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.clamp(max=SWIGLU_LIMIT)
    up = up.clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    return torch.nn.functional.silu(gate) * up


def run_unfused_shared_mlp(
    hidden_bf16: torch.Tensor,
    weights: dict[str, torch.Tensor],
    *,
    use_alpha: bool = True,
) -> torch.Tensor:
    gemm_afp4wfp4, dynamic_mxfp4_quant, _, _ = require_aiter_mxfp4_kernels()
    dtype = hidden_bf16.dtype
    m = hidden_bf16.shape[0]

    x_q, x_s = dynamic_mxfp4_quant(hidden_bf16.contiguous())
    gate_up = torch.empty(m, weights["w13_w"].shape[0], device=hidden_bf16.device, dtype=dtype)
    gemm_afp4wfp4(x_q, weights["w13_w"], x_s, weights["w13_s"], dtype, gate_up)
    mid = swiglu_shared_activation(gate_up, use_alpha=use_alpha)

    mid_q, mid_s = dynamic_mxfp4_quant(mid.contiguous())
    out = torch.empty(m, HIDDEN, device=hidden_bf16.device, dtype=dtype)
    gemm_afp4wfp4(mid_q, weights["down_w"], mid_s, weights["down_s"], dtype, out)
    return out


def production_gate_mode() -> int:
    """Mirror ``AiterRunner.run`` gate_mode selection for clamped-SwiGLU MXFP4."""
    from aiter.ops.flydsl.moe_common import GateMode

    from sglang.srt.environ import envs

    return (
        GateMode.INTERLEAVE.value
        if envs.SGLANG_USE_AITER_MOE_GU_ITLV.get()
        else GateMode.SEPARATED.value
    )


def run_fused_shared_moe_slot(
    hidden_bf16: torch.Tensor,
    moe_weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    local_expert_id: int = 0,
    use_alpha: bool = False,
) -> torch.Tensor:
    """Run fused MoE with only the shared-expert weights loaded.

    ``local_expert_id`` is the index into the ``[E, ...]`` weight tensors passed
    to ``fused_moe`` (0 when this helper is called with a single-expert slice).
    Kwargs mirror ``AiterRunner`` for MiniMax-M3 MXFP4 (``swiglu_limit`` path).
    """
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    w13, w13_s, w2, w2_s = moe_weights
    if getattr(w13, "is_shuffled", False):
        w13.is_shuffled = True
        w2.is_shuffled = True

    m = hidden_bf16.shape[0]
    topk_ids = torch.full(
        (m, 1), local_expert_id, device=hidden_bf16.device, dtype=torch.int32
    )
    topk_weights = torch.ones((m, 1), device=hidden_bf16.device, dtype=torch.float32)

    kwargs = dict(
        swiglu_limit=SWIGLU_LIMIT,
        gate_mode=production_gate_mode(),
    )
    if use_alpha:
        kwargs["beta"] = float(SWIGLU_ALPHA)

    return fused_moe(
        hidden_states=hidden_bf16,
        w1=w13,
        w2=w2,
        topk_weight=topk_weights,
        topk_ids=topk_ids,
        quant_type=QuantType.per_1x32,
        activation=ActivationType.Swiglu,
        w1_scale=w13_s,
        w2_scale=w2_s,
        **kwargs,
    )


def run_shared_mlp_dequant_reference(
    hidden_bf16: torch.Tensor,
    weights: dict[str, torch.Tensor],
    *,
    use_alpha: bool = True,
) -> torch.Tensor:
    """Slow reference that mirrors the unfused quant GEMM path stage-by-stage."""
    gemm_afp4wfp4, dynamic_mxfp4_quant, _, _ = require_aiter_mxfp4_kernels()
    dtype = hidden_bf16.dtype
    m = hidden_bf16.shape[0]

    x_q, x_s = dynamic_mxfp4_quant(hidden_bf16.contiguous())
    gate_up = torch.empty(m, weights["w13_w"].shape[0], device=hidden_bf16.device, dtype=dtype)
    gemm_afp4wfp4(x_q, weights["w13_w"], x_s, weights["w13_s"], dtype, gate_up)

    mid = swiglu_shared_activation(gate_up, use_alpha=use_alpha)
    mid_q, mid_s = dynamic_mxfp4_quant(mid.contiguous())
    out = torch.empty(m, HIDDEN, device=hidden_bf16.device, dtype=dtype)
    gemm_afp4wfp4(mid_q, weights["down_w"], mid_s, weights["down_s"], dtype, out)
    return out
