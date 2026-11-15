"""CPU test of the SGLANG_DSA_INDEXER_FLYDSL_MQA_LOGITS switch (dsa/hip_mqa_logits.py): default = Triton kernel,
flag = FlyDSL kernel, missing FlyDSL raises (no silent fallback). aiter and sglang.srt.environ are stubbed."""
import importlib.util
import os
import sys
import types
import unittest

_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "python", "sglang", "srt", "layers", "attention", "dsa", "hip_mqa_logits.py")


class _Flag:
    def __init__(self):
        self.v = False

    def get(self):
        return self.v


def _mod(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


def _load(flag, with_flydsl=True):
    for n in ("aiter", "aiter.ops", "aiter.ops.triton", "aiter.ops.flydsl", "aiter.ops.triton.fp8_mqa_logits",
              "aiter.ops.flydsl.fp8_mqa_logits_kernels", "sglang", "sglang.srt", "sglang.srt.environ"):
        sys.modules.pop(n, None)
    _mod("sglang"); _mod("sglang.srt")
    _mod("sglang.srt.environ", envs=types.SimpleNamespace(SGLANG_DSA_INDEXER_FLYDSL_MQA_LOGITS=flag))
    _mod("aiter"); _mod("aiter.ops"); _mod("aiter.ops.triton"); _mod("aiter.ops.flydsl")
    _mod("aiter.ops.triton.fp8_mqa_logits", fp8_mqa_logits=lambda *a, **k: "triton")
    if with_flydsl:
        _mod("aiter.ops.flydsl.fp8_mqa_logits_kernels", flydsl_fp8_mqa_logits=lambda *a, **k: "flydsl")
    spec = importlib.util.spec_from_file_location("hip_mqa_under_test", _SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class T(unittest.TestCase):
    def test_default_triton(self):
        f = _Flag(); m = _load(f)
        self.assertEqual(m.resolve_hip_fp8_mqa_logits()(), "triton")

    def test_flag_flydsl_and_toggle(self):
        f = _Flag(); m = _load(f)
        self.assertEqual(m.resolve_hip_fp8_mqa_logits()(), "triton")
        f.v = True
        self.assertEqual(m.resolve_hip_fp8_mqa_logits()(), "flydsl")
        f.v = False
        self.assertEqual(m.resolve_hip_fp8_mqa_logits()(), "triton")

    def test_missing_flydsl_raises(self):
        f = _Flag(); f.v = True; m = _load(f, with_flydsl=False)
        with self.assertRaises(ImportError):
            m.resolve_hip_fp8_mqa_logits()


if __name__ == "__main__":
    unittest.main()
