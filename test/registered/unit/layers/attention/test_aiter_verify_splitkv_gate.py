"""CPU gate for SGLANG_AITER_VERIFY_SPLITKV on aiter target verify.

Does not launch a server or a kernel. ``_verify_splitkv_eligible`` only
reads backend flags, the layer, and capture mode.
"""

import os
import sys
import types
import unittest
from contextlib import contextmanager
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_RUNNER = "sglang.srt.model_executor.runner"


@contextmanager
def _stub_capture(enabled: bool):
    """Avoid importing the CUDA-graph runner just to read capture mode."""
    previous = sys.modules.get(_RUNNER)
    stub = types.ModuleType(_RUNNER)
    stub.get_is_capture_mode = lambda: enabled
    sys.modules[_RUNNER] = stub
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop(_RUNNER, None)
        else:
            sys.modules[_RUNNER] = previous


def _layer(**overrides):
    fields = {
        "logit_cap": None,
        "sliding_window_size": None,
        "qk_head_dim": 256,
        "v_head_dim": 256,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _backend(**overrides):
    backend = AiterAttnBackend.__new__(AiterAttnBackend)
    backend.use_mla = False
    backend.kv_cache_is_vectorized_5d = False
    backend.page_size = 1
    backend.topk = 1
    backend.forward_metadata = SimpleNamespace(
        qo_indptr=torch.zeros(2, dtype=torch.int32),
        custom_mask=None,
    )
    for name, value in overrides.items():
        setattr(backend, name, value)
    return backend


class TestAiterVerifySplitKvGate(CustomTestCase):
    def _eligible(self, backend, layer=None, sinks=None, capture=False):
        os.environ.pop("SGLANG_AITER_VERIFY_SPLITKV", None)
        with _stub_capture(capture):
            return backend._verify_splitkv_eligible(
                _layer() if layer is None else layer,
                sinks,
            )

    def test_default_env_is_eligible(self):
        self.assertTrue(self._eligible(_backend()))

    def test_env_zero_and_false_disable(self):
        backend = _backend()
        with _stub_capture(False):
            os.environ["SGLANG_AITER_VERIFY_SPLITKV"] = "0"
            self.assertFalse(backend._verify_splitkv_eligible(_layer(), None))
            os.environ["SGLANG_AITER_VERIFY_SPLITKV"] = "false"
            self.assertFalse(backend._verify_splitkv_eligible(_layer(), None))
            os.environ["SGLANG_AITER_VERIFY_SPLITKV"] = "true"
            self.assertTrue(backend._verify_splitkv_eligible(_layer(), None))
        os.environ.pop("SGLANG_AITER_VERIFY_SPLITKV", None)

    def test_backend_flags_reject(self):
        cases = (
            {"use_mla": True},
            {"kv_cache_is_vectorized_5d": True},
            {"page_size": 16},
            {"topk": 2},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                self.assertFalse(self._eligible(_backend(**overrides)))

    def test_topk_none_stays_eligible(self):
        self.assertTrue(self._eligible(_backend(topk=None)))

    def test_layer_and_metadata_reject(self):
        cases = {
            "sinks": (_backend(), _layer(), object()),
            "logit_cap": (_backend(), _layer(logit_cap=1.0), None),
            "sliding_window": (_backend(), _layer(sliding_window_size=128), None),
            "head_dim": (_backend(), _layer(v_head_dim=128), None),
            "no_metadata": (_backend(forward_metadata=None), _layer(), None),
            "no_qo": (
                _backend(
                    forward_metadata=SimpleNamespace(qo_indptr=None, custom_mask=None)
                ),
                _layer(),
                None,
            ),
            "custom_mask": (
                _backend(
                    forward_metadata=SimpleNamespace(
                        qo_indptr=torch.zeros(2, dtype=torch.int32),
                        custom_mask=torch.zeros(1, dtype=torch.int32),
                    )
                ),
                _layer(),
                None,
            ),
        }
        for name, (backend, layer, sinks) in cases.items():
            with self.subTest(name=name):
                self.assertFalse(self._eligible(backend, layer, sinks))

    def test_sliding_window_disabled_stays_eligible(self):
        self.assertTrue(self._eligible(_backend(), _layer(sliding_window_size=-1)))

    def test_capture_mode_rejects(self):
        self.assertFalse(self._eligible(_backend(), capture=True))

    def test_forward_returns_false_when_gated_off(self):
        backend = _backend()
        os.environ["SGLANG_AITER_VERIFY_SPLITKV"] = "0"
        try:
            self.assertFalse(
                backend._forward_target_verify_splitkv(None, None, _layer(), None, None)
            )
        finally:
            os.environ.pop("SGLANG_AITER_VERIFY_SPLITKV", None)

    def test_forward_returns_false_during_capture(self):
        os.environ.pop("SGLANG_AITER_VERIFY_SPLITKV", None)
        backend = _backend()
        with _stub_capture(True):
            self.assertFalse(
                backend._forward_target_verify_splitkv(None, None, _layer(), None, None)
            )


if __name__ == "__main__":
    unittest.main()
