"""CPU checks for the FlyDSL verify gate and page table.

Does not launch a kernel. The page table is the slot-to-page map the
serving path builds from ``req_to_token``.
"""

import os
import sys
import types
import unittest
from contextlib import contextmanager
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
from sglang.srt.layers.attention.aiter_utils import flydsl_block_table
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_RUNNER = "sglang.srt.model_executor.runner"


@contextmanager
def _stub_capture(enabled: bool):
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
        "layer_id": 0,
        "logit_cap": None,
        "sliding_window_size": None,
        "qk_head_dim": 256,
        "v_head_dim": 256,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _backend(**overrides):
    backend = AiterAttnBackend.__new__(AiterAttnBackend)
    backend.kv_cache_is_vectorized_5d = True
    backend.page_size = 16
    backend.kv_cache_dtype = torch.bfloat16
    for name, value in overrides.items():
        setattr(backend, name, value)
    return backend


class TestAiterVerifyFlydslGate(CustomTestCase):
    def test_env_default_off(self):
        os.environ.pop("SGLANG_AITER_VERIFY_FLYDSL", None)
        from sglang.srt.environ import envs

        self.assertFalse(envs.SGLANG_AITER_VERIFY_FLYDSL.get())

    def test_ready_layer_passes(self):
        _backend()._flydsl_require_layer(_layer(), None)

    def test_layer_gates_raise(self):
        cases = {
            "nhd": (_backend(kv_cache_is_vectorized_5d=False), _layer(), None),
            "page": (_backend(page_size=64), _layer(), None),
            "fp8": (_backend(kv_cache_dtype=torch.float8_e4m3fnuz), _layer(), None),
            "sinks": (_backend(), _layer(), object()),
            "logit_cap": (_backend(), _layer(logit_cap=1.0), None),
            "sliding": (_backend(), _layer(sliding_window_size=128), None),
            "head": (_backend(), _layer(v_head_dim=128), None),
        }
        for name, (backend, layer, sinks) in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(RuntimeError):
                    backend._flydsl_require_layer(layer, sinks)

    def test_sliding_window_disabled_passes(self):
        _backend()._flydsl_require_layer(_layer(sliding_window_size=-1), None)

    def test_capture_raises(self):
        with _stub_capture(True):
            with self.assertRaises(RuntimeError):
                _backend()._flydsl_require_eager()

    def test_eager_passes(self):
        with _stub_capture(False):
            _backend()._flydsl_require_eager()


class TestFlydslBlockTable(CustomTestCase):
    def test_partial_page_uses_existing_columns(self):
        # Context 108 is not a multiple of 16. The last page starts at 96.
        page = 16
        context = 108
        base = 10000
        width = context
        table_src = torch.arange(width, dtype=torch.int64) + base
        req = table_src.view(1, width)
        got = flydsl_block_table(req, torch.tensor([0]), page, context)
        n_pages = (context + page - 1) // page
        starts = torch.arange(n_pages) * page
        expect = ((base + starts) // page).to(torch.int32).view(1, n_pages)
        self.assertEqual(tuple(got.shape), (1, n_pages))
        self.assertTrue(torch.equal(got, expect))

    def test_short_row_raises(self):
        req = torch.arange(96, dtype=torch.int64).view(1, 96)
        with self.assertRaises(RuntimeError):
            flydsl_block_table(req, torch.tensor([0]), 16, 108)

    def test_two_requests_keep_their_pages(self):
        page = 16
        context = 32
        req = torch.tensor(
            [
                [16, 17, 18, 19] + [0] * 28,
                [48, 49, 50, 51] + [0] * 28,
            ],
            dtype=torch.int64,
        )
        # Only the page-start columns matter: 0 and 16.
        req[0, 0] = 160
        req[0, 16] = 176
        req[1, 0] = 320
        req[1, 16] = 336
        got = flydsl_block_table(req, torch.tensor([1, 0]), page, context)
        self.assertTrue(torch.equal(got[0], torch.tensor([20, 21], dtype=torch.int32)))
        self.assertTrue(torch.equal(got[1], torch.tensor([10, 11], dtype=torch.int32)))


if __name__ == "__main__":
    unittest.main()
