"""CPU checks for the FlyDSL verify gate and page table.

Does not launch a kernel. The page table is the slot-to-page map the
serving path builds from ``req_to_token``.
"""

import os
import sys
import types
import unittest
import unittest.mock
from contextlib import contextmanager
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
from sglang.srt.layers.attention.aiter_utils import (
    flydsl_block_table,
    flydsl_group_row_indices,
    flydsl_note_scratch,
    flydsl_use_graph_bound,
    flydsl_verify_live_batch,
    flydsl_verify_width_groups,
)
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
        os.environ.pop("SGLANG_AITER_VERIFY_FLYDSL_EXTEND", None)
        from sglang.srt.environ import envs

        self.assertFalse(envs.SGLANG_AITER_VERIFY_FLYDSL.get())
        self.assertFalse(envs.SGLANG_AITER_VERIFY_FLYDSL_EXTEND.get())

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

    def test_stream_capture_raises(self):
        # Ragged and extend still sync. Only an actual stream capture refuses
        # them. The session capture flag is also set in a breakable eager gap.
        import sglang.srt.layers.attention.aiter_backend as backend_mod

        previous = backend_mod.flydsl_stream_capturing
        backend_mod.flydsl_stream_capturing = lambda: True
        try:
            with self.assertRaises(RuntimeError):
                _backend()._flydsl_require_eager()
        finally:
            backend_mod.flydsl_stream_capturing = previous

    def test_capture_mode_flag_does_not_raise(self):
        with _stub_capture(True):
            _backend()._flydsl_require_eager()

    def test_eager_passes(self):
        with _stub_capture(False):
            _backend()._flydsl_require_eager()

    def test_ragged_verify_graph_opts_in(self):
        # The graph path is the padded window. The grouped eager path still
        # syncs and is not what this flag selects.
        self.assertTrue(AiterAttnBackend.supports_ragged_verify_graph)


class TestFlydslGraphBound(CustomTestCase):
    def test_full_graph_uses_the_bound(self):
        self.assertTrue(
            flydsl_use_graph_bound(
                stream_capturing=False, capture_mode=True, breakable=False
            )
        )
        self.assertTrue(
            flydsl_use_graph_bound(
                stream_capturing=True, capture_mode=True, breakable=False
            )
        )

    def test_breakable_gap_keeps_the_measured_length(self):
        self.assertFalse(
            flydsl_use_graph_bound(
                stream_capturing=False, capture_mode=True, breakable=True
            )
        )

    def test_eager_keeps_the_measured_length(self):
        self.assertFalse(
            flydsl_use_graph_bound(
                stream_capturing=False, capture_mode=False, breakable=False
            )
        )

    def test_capture_max_is_the_shorter_of_context_and_table(self):
        backend = _backend()
        backend.max_context_len = 8192
        backend.req_to_token = torch.zeros(4, 1000)
        self.assertEqual(backend._flydsl_capture_max_context(), 1000)
        backend._flydsl_capture_max_cached = None
        backend.req_to_token = torch.zeros(4, 9000)
        self.assertEqual(backend._flydsl_capture_max_context(), 8192)

    def test_pinned_scratch_is_not_evicted(self):
        cache: dict = {}
        pinned: dict = {}
        for i in range(16):
            flydsl_note_scratch(cache, pinned, i, i, pin=True)
        self.assertEqual(len(cache), 16)
        flydsl_note_scratch(cache, pinned, 16, 16, pin=False)
        self.assertEqual(len(cache), 17)
        self.assertIn(0, cache)
        flydsl_note_scratch(cache, pinned, 17, 17, pin=False)
        self.assertNotIn(16, cache)
        self.assertIn(0, cache)
        self.assertIn(17, cache)


class TestFlydslVerifyLiveBatch(CustomTestCase):
    def test_uniform_matches_pool(self):
        self.assertEqual(flydsl_verify_live_batch(32, 8, 4), 4)

    def test_trailing_pool_slot_uses_query_prefix(self):
        # GSM8K conc4 crashed with q=24 and a 4-wide plan (need 32).
        self.assertEqual(flydsl_verify_live_batch(24, 8, 4), 3)

    def test_ragged_rows_raise(self):
        with self.assertRaises(ValueError):
            flydsl_verify_live_batch(22, 8, 4)

    def test_live_wider_than_pool_raises(self):
        with self.assertRaises(ValueError):
            flydsl_verify_live_batch(32, 8, 3)


class TestFlydslExtendPlan(CustomTestCase):
    def test_ragged_103_rows_plan_from_qo_indptr(self):
        # Conc16 crashed in pa_decode: 103 query rows, pool 13, qlen 8.
        # Extend does not need one width. Twelve 8s plus a 7 is 103.
        widths = [8] * 12 + [7]
        qo = [0]
        for width in widths:
            qo.append(qo[-1] + width)
        backend = _backend()
        backend.forward_metadata = SimpleNamespace(
            qo_indptr=torch.tensor(qo, dtype=torch.int32)
        )
        self.assertEqual(backend._flydsl_extend_plan(103, 13), (13, 8))

    def test_missing_qo_indptr_raises(self):
        backend = _backend()
        backend.forward_metadata = SimpleNamespace(qo_indptr=None)
        with self.assertRaises(RuntimeError):
            backend._flydsl_extend_plan(8, 1)


def _qo(widths):
    qo = [0]
    for width in widths:
        qo.append(qo[-1] + width)
    return qo


class TestFlydslVerifyWidthGroups(CustomTestCase):
    def test_127_rows_pool_16_qlen_8_does_not_pad(self):
        # Compact conc16 died at 127 query rows, pool 16, max qlen 8.
        # 15 requests of width 8 and one of width 7. The short request
        # must stay width 7.
        widths = [8] * 15 + [7]
        qo = _qo(widths)
        self.assertEqual(qo[-1], 127)
        bs, groups = flydsl_verify_width_groups(qo, 127, 16)
        self.assertEqual(bs, 16)
        self.assertEqual(groups, [(8, list(range(15))), (7, [15])])
        self.assertEqual(sum(w * len(ix) for w, ix in groups), 127)
        self.assertEqual(
            flydsl_group_row_indices(qo, 7, [15]), list(range(120, 127))
        )

    def test_middle_width_is_not_a_contiguous_slice(self):
        widths = [8, 7, 8]
        qo = _qo(widths)
        _bs, groups = flydsl_verify_width_groups(qo, 23, 3)
        self.assertEqual(groups, [(8, [0, 2]), (7, [1])])
        self.assertEqual(
            flydsl_group_row_indices(qo, 8, [0, 2]),
            list(range(8)) + list(range(15, 23)),
        )
        self.assertEqual(flydsl_group_row_indices(qo, 7, [1]), list(range(8, 15)))

    def test_multiple_of_max_can_still_be_ragged(self):
        # 8+8+4+4 = 24, which divides the max width. The first 3 requests
        # do not cover the query.
        qo = _qo([8, 8, 4, 4])
        bs, groups = flydsl_verify_width_groups(qo, 24, 4)
        self.assertEqual(bs, 4)
        self.assertEqual(groups, [(8, [0, 1]), (4, [2, 3])])

    def test_indptr_that_skips_the_query_raises(self):
        with self.assertRaises(ValueError):
            flydsl_verify_width_groups([0, 8, 16, 32], 24, 3)


class _KV:
    def read(self, layer_id):
        return self

    def kv(self):
        return (torch.zeros(1), torch.zeros(1))


class TestFlydslGroupedVerifyLaunch(CustomTestCase):
    def setUp(self):
        super().setUp()
        os.environ.pop("SGLANG_AITER_VERIFY_FLYDSL_EXTEND", None)
        os.environ.pop("SGLANG_AITER_VERIFY_FLYDSL_DEBUG", None)
        self.calls = []
        self.heads = 2
        self.dim = 4

    def _run(self, widths, q_rows, pool_bs, qlen, seq, pool_ids=None, graph_bound=False):
        qo = _qo(widths)
        if pool_ids is None:
            pool_ids = list(range(pool_bs))
        q = torch.arange(q_rows * self.heads * self.dim, dtype=torch.float32).reshape(
            q_rows, self.heads, self.dim
        )
        q = q.clone()
        q[:, 0, 0] = torch.arange(q_rows)
        o = torch.zeros_like(q)
        backend = _backend()
        backend.forward_metadata = SimpleNamespace(
            qo_indptr=torch.tensor(qo, dtype=torch.int32),
            max_q_len=qlen,
        )
        backend._flydsl_nonce = 7
        backend.token_to_kv_pool = SimpleNamespace(layer_router=_KV())
        backend._flydsl_require_eager = lambda: None
        backend._flydsl_require_layer = lambda *args, **kwargs: None
        heads, dim = self.heads, self.dim

        def fake_launch(
            q_in,
            o_in,
            layer,
            fb,
            context,
            query_length,
            max_context,
            k_cache,
            v_cache,
            req_indices=None,
        ):
            q3 = q_in.reshape(-1, heads, dim)
            self.calls.append(
                {
                    "qlen": query_length,
                    "rows": int(q3.shape[0]),
                    "q_ids": [int(v) for v in q3[:, 0, 0].tolist()],
                    "ctx": [int(v) for v in context.tolist()],
                    "max_context": max_context,
                    "req": None
                    if req_indices is None
                    else [int(v) for v in req_indices.tolist()],
                }
            )
            o_in.reshape(-1, heads, dim).fill_(float(query_length))

        backend._flydsl_launch = fake_launch
        if graph_bound:
            backend._flydsl_graph_bound = lambda: True
            backend._flydsl_capture_max_context = lambda: 4096
        fb = SimpleNamespace(
            req_pool_indices=torch.tensor(pool_ids, dtype=torch.int64),
            seq_lens=torch.tensor(seq, dtype=torch.int32),
            seq_lens_cpu=None,
        )
        layer = _layer(
            tp_q_head_num=self.heads,
            qk_head_dim=self.dim,
            v_head_dim=self.dim,
        )
        backend._forward_target_verify_flydsl(
            q, o, layer, fb, None, torch.zeros(1), torch.zeros(1)
        )
        return o

    def test_127_launches_width_8_and_width_7(self):
        widths = [8] * 15 + [7]
        seq = [1000 + i for i in range(16)]
        pool_ids = list(range(100, 116))
        o = self._run(widths, 127, 16, 8, seq, pool_ids)
        self.assertEqual([c["qlen"] for c in self.calls], [8, 7])
        self.assertEqual(self.calls[0]["rows"], 120)
        self.assertEqual(self.calls[1]["rows"], 7)
        self.assertEqual(self.calls[0]["q_ids"], list(range(120)))
        self.assertEqual(self.calls[1]["q_ids"], list(range(120, 127)))
        self.assertEqual(self.calls[0]["ctx"], [1000 + i + 8 for i in range(15)])
        self.assertEqual(self.calls[1]["ctx"], [1015 + 7])
        self.assertEqual(self.calls[1]["req"], [115])
        self.assertTrue(torch.all(o.reshape(127, self.heads, self.dim)[:120] == 8))
        self.assertTrue(torch.all(o.reshape(127, self.heads, self.dim)[120:] == 7))

    def test_middle_short_request_scatters_back(self):
        o = self._run([8, 7, 8], 23, 3, 8, [10, 20, 30], [5, 6, 7])
        self.assertEqual([c["qlen"] for c in self.calls], [8, 7])
        self.assertEqual(self.calls[0]["q_ids"], list(range(8)) + list(range(15, 23)))
        self.assertEqual(self.calls[1]["q_ids"], list(range(8, 15)))
        self.assertEqual(self.calls[0]["ctx"], [18, 38])
        self.assertEqual(self.calls[1]["ctx"], [27])
        self.assertEqual(self.calls[0]["req"], [5, 7])
        self.assertEqual(self.calls[1]["req"], [6])
        flat = o[:, 0, 0]
        self.assertTrue(torch.equal(flat[:8], torch.full((8,), 8.0)))
        self.assertTrue(torch.equal(flat[8:15], torch.full((7,), 7.0)))
        self.assertTrue(torch.equal(flat[15:], torch.full((8,), 8.0)))

    def test_uniform_width_stays_one_launch(self):
        # Trailing pool slot: 24 query rows, 4 pool entries, width 8.
        o = self._run([8, 8, 8, 8], 24, 4, 8, [10, 20, 30, 40])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0]["qlen"], 8)
        self.assertEqual(self.calls[0]["rows"], 24)
        self.assertEqual(self.calls[0]["ctx"], [18, 28, 38])
        self.assertEqual(self.calls[0]["max_context"], 38)
        self.assertIsNone(self.calls[0]["req"])
        self.assertTrue(torch.all(o[:, 0, 0] == 8))

    def test_graph_bound_keeps_lengths_and_bakes_host_max(self):
        # Eager max of this batch is 38. A captured graph has to specialize
        # on the host bound instead, without padding the per-request lengths.
        o = self._run(
            [8, 8, 8, 8], 24, 4, 8, [10, 20, 30, 40], graph_bound=True
        )
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0]["ctx"], [18, 28, 38])
        self.assertEqual(self.calls[0]["max_context"], 4096)
        self.assertTrue(torch.all(o[:, 0, 0] == 8))

    def test_sum_divisible_by_max_still_groups(self):
        o = self._run([8, 8, 4, 4], 24, 4, 8, [10, 20, 30, 40])
        self.assertEqual([c["qlen"] for c in self.calls], [8, 4])
        self.assertEqual(self.calls[0]["rows"], 16)
        self.assertEqual(self.calls[1]["rows"], 8)
        self.assertEqual(self.calls[0]["ctx"], [18, 28])
        self.assertEqual(self.calls[1]["ctx"], [34, 44])
        flat = o[:, 0, 0]
        self.assertTrue(torch.equal(flat[:16], torch.full((16,), 8.0)))
        self.assertTrue(torch.equal(flat[16:], torch.full((8,), 4.0)))


class TestFlydslBreakableGap(CustomTestCase):
    def _layer6(self):
        layer = _layer()
        layer.tp_q_head_num = 6
        return layer

    def test_gap_method_is_a_graph_break(self):
        fn = AiterAttnBackend._flydsl_grouped_verify_gap
        self.assertTrue(callable(getattr(fn, "run_eager", None)))

    def test_gap_outside_capture_calls_packed_row_base(self):
        # The breakable gap launches packed Q/O per occupied width.
        backend = _backend()
        seen = {}

        def packed(self, q, o, layer, forward_batch, layout):
            seen["rows"] = int(q.shape[0])
            seen["layout"] = layout

        q = torch.zeros(12, 6, 256)
        layout = SimpleNamespace(
            verify_lens=torch.tensor([4, 4, 4], dtype=torch.int32),
            qo_indptr_device=torch.tensor([0, 4, 8, 12], dtype=torch.int32),
        )
        fb = SimpleNamespace(
            req_pool_indices=torch.arange(3),
            spec_info=SimpleNamespace(ragged_verify_layout=layout),
        )
        with unittest.mock.patch.object(
            AiterAttnBackend, "_flydsl_packed_attention", packed
        ):
            backend._flydsl_grouped_verify_gap(
                q, torch.zeros_like(q), self._layer6(), fb
            )
        self.assertEqual(seen["rows"], 12)
        self.assertIs(seen["layout"], layout)

    def test_packed_attention_has_no_pytorch_gather(self):
        import inspect

        src = inspect.getsource(AiterAttnBackend._flydsl_packed_attention)
        for bad in ("index_select", "index_copy_", ".copy_("):
            self.assertNotIn(bad, src)

    def _dispatch(self, *, breakable: bool):
        backend = _backend()
        backend.forward_metadata = SimpleNamespace(max_q_len=8, qo_indptr=None)
        calls = []

        def gap(self, q, o, layer, forward_batch):
            calls.append("gap")

        def padded(self, q, o, layer, forward_batch, layout):
            calls.append("padded")

        q = torch.zeros(12, 6, 256)
        layout = SimpleNamespace()
        fb = SimpleNamespace(
            req_pool_indices=torch.arange(4),
            spec_info=SimpleNamespace(draft_token_num=8, ragged_verify_layout=layout),
        )
        with (
            unittest.mock.patch.object(
                AiterAttnBackend, "_flydsl_grouped_verify_gap", gap
            ),
            unittest.mock.patch.object(
                AiterAttnBackend, "_forward_target_verify_flydsl_padded", padded
            ),
            unittest.mock.patch(
                "sglang.srt.layers.attention.aiter_backend.is_in_breakable_cuda_graph",
                return_value=breakable,
            ),
            _stub_capture(True),
        ):
            backend._forward_target_verify_flydsl(
                q, torch.zeros_like(q), self._layer6(), fb, None, torch.zeros(1), torch.zeros(1)
            )
        return calls

    def test_grouped_qo_prefers_the_layout_indptr(self):
        # Metadata's uniform arange is input_rows // slots. 24 rows on 4
        # slots is width 5 and sums to 20, short of the packed query.
        # The layout indptr is what the gap has to group on.
        backend = _backend()
        uniform = torch.arange(0, 25, 5)
        layout_qo = torch.tensor([0, 8, 14, 20, 24], dtype=torch.int32)
        backend.forward_metadata = SimpleNamespace(qo_indptr=uniform)
        fb = SimpleNamespace(
            spec_info=SimpleNamespace(
                ragged_verify_layout=SimpleNamespace(qo_indptr_device=layout_qo)
            )
        )
        got = backend._flydsl_grouped_qo(fb)
        self.assertTrue(torch.equal(got, layout_qo))

    def test_grouped_qo_falls_back_to_metadata(self):
        backend = _backend()
        uniform = torch.arange(0, 25, 5)
        backend.forward_metadata = SimpleNamespace(qo_indptr=uniform)
        fb = SimpleNamespace(spec_info=None)
        got = backend._flydsl_grouped_qo(fb)
        self.assertTrue(torch.equal(got, uniform))

    def _gap_q_rows(self, q_rows: int, pool_bs: int, qo):
        backend = _backend()
        seen = {}

        def packed(self, q, o, layer, forward_batch, layout):
            seen["q_rows"] = int(q.shape[0])
            seen["pool_bs"] = pool_bs

        q = torch.zeros(q_rows, 6, 256)
        fb = SimpleNamespace(
            req_pool_indices=torch.arange(pool_bs),
            spec_info=SimpleNamespace(
                ragged_verify_layout=SimpleNamespace(
                    qo_indptr_device=qo,
                    verify_lens=torch.ones(pool_bs, dtype=torch.int32),
                )
            ),
        )
        with unittest.mock.patch.object(
            AiterAttnBackend, "_flydsl_packed_attention", packed
        ):
            backend._flydsl_grouped_verify_gap(
                q, torch.zeros_like(q), self._layer6(), fb
            )
        return seen

    def test_gap_slices_to_the_staged_indptr_not_the_tier(self):
        # 912d636 died on GSM8K p32: graph tier 256 rows, pool 32, staged
        # prefix [0, 6, 14, ...] never reached 256. The padded window must
        # see only the covered rows, or searchsorted walks into the ghost tail.
        qo = torch.tensor([0, 6, 14, 21, 28], dtype=torch.int32)
        self.assertEqual(
            self._gap_q_rows(32, 4, qo),
            {"q_rows": 28, "pool_bs": 4},
        )

    def test_gap_keeps_q_rows_when_the_indptr_covers_the_tier(self):
        qo = torch.tensor([0, 8, 16, 24, 32], dtype=torch.int32)
        self.assertEqual(
            self._gap_q_rows(32, 4, qo),
            {"q_rows": 32, "pool_bs": 4},
        )

    def test_gap_keeps_q_rows_when_the_indptr_runs_past_the_buffer(self):
        qo = torch.tensor([0, 10, 20, 40], dtype=torch.int32)
        self.assertEqual(
            self._gap_q_rows(12, 3, qo),
            {"q_rows": 12, "pool_bs": 3},
        )

    def test_grouped_step_uses_the_real_lens_not_the_bucket_fill(self):
        # 18faffe launched the filled last row. short-4 logprobs matched
        # the 0fbbca3 graphs file (max abs 0.7435) and not cea4337.
        from sglang.srt.speculative.ragged_verify import remember_real_verify_lens

        backend = _backend()
        layout = SimpleNamespace(
            qo_indptr_device=torch.tensor([0, 5, 13, 21], dtype=torch.int32)
        )
        remember_real_verify_lens(layout, torch.tensor([5, 8, 6], dtype=torch.int32))
        fb = SimpleNamespace(
            req_pool_indices=torch.arange(3),
            seq_lens=torch.tensor([10, 20, 30], dtype=torch.int32),
            spec_info=SimpleNamespace(ragged_verify_layout=layout),
        )
        packed = backend._flydsl_grouped_step(21, 3, torch.zeros(21, 6, 256), fb)
        sig = [(width, int(req.shape[0])) for width, _, _, _, req in packed]
        self.assertEqual(sig, [(5, 1), (8, 1), (6, 1)])
        self.assertEqual(packed[-1][1].tolist(), list(range(13, 19)))

    def test_breakable_layout_takes_the_grouped_gap(self):
        self.assertEqual(self._dispatch(breakable=True), ["gap"])

    def test_full_graph_layout_stays_padded(self):
        self.assertEqual(self._dispatch(breakable=False), ["padded"])

    def test_eager_layout_uses_packed(self):
        backend = _backend()
        backend.forward_metadata = SimpleNamespace(max_q_len=8, qo_indptr=None)
        calls = []

        def packed(self, q, o, layer, forward_batch):
            calls.append("packed")

        def padded(self, q, o, layer, forward_batch, layout):
            calls.append("padded")

        q = torch.zeros(12, 6, 256)
        layout = SimpleNamespace()
        fb = SimpleNamespace(
            req_pool_indices=torch.arange(4),
            spec_info=SimpleNamespace(draft_token_num=8, ragged_verify_layout=layout),
        )
        with (
            unittest.mock.patch.object(
                AiterAttnBackend, "_flydsl_packed_verify", packed
            ),
            unittest.mock.patch.object(
                AiterAttnBackend, "_forward_target_verify_flydsl_padded", padded
            ),
            unittest.mock.patch(
                "sglang.srt.layers.attention.aiter_backend.is_in_breakable_cuda_graph",
                return_value=False,
            ),
            _stub_capture(False),
        ):
            backend._forward_target_verify_flydsl(
                q,
                torch.zeros_like(q),
                self._layer6(),
                fb,
                None,
                torch.zeros(1),
                torch.zeros(1),
            )
        self.assertEqual(calls, ["packed"])


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
