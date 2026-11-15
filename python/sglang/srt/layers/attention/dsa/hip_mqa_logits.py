"""Pick the HIP fp8 MQA-logits kernel of the DSA prefill indexer.

Default: aiter's Triton `fp8_mqa_logits`. With SGLANG_DSA_INDEXER_FLYDSL_MQA_LOGITS=1: aiter's FlyDSL
`flydsl_fp8_mqa_logits` (same signature, `aiter/ops/flydsl/fp8_mqa_logits_kernels.py`). A missing FlyDSL kernel
raises instead of falling back, so an A/B run can never silently measure the Triton kernel twice.
"""
import logging
from functools import lru_cache

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def _resolve(flydsl: bool):
    if flydsl:
        from aiter.ops.flydsl.fp8_mqa_logits_kernels import flydsl_fp8_mqa_logits

        logger.info("DSA indexer HIP MQA logits: aiter FlyDSL kernel (SGLANG_DSA_INDEXER_FLYDSL_MQA_LOGITS=1)")
        return flydsl_fp8_mqa_logits
    from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits

    return fp8_mqa_logits


def resolve_hip_fp8_mqa_logits():
    """Return a callable (q, kv, kv_scale, weights, ks, ke, clean_logits=...) -> fp32 logits."""
    return _resolve(bool(envs.SGLANG_DSA_INDEXER_FLYDSL_MQA_LOGITS.get()))
