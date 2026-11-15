"""CPU test of the row split used by SGLANG_DSA_INDEXER_TP_ROW_SPLIT (dsa_dcp.dcp_split_rows / dcp_all_gather_rows_ with an explicit group).

Simulates W attn-TP ranks in one process: every rank scores only its stride of rows, then the strides are all-gathered in place.
The reassembled top-k must equal the unsplit top-k for every row count, including row counts that are not a multiple of W
and ranks that own no rows. The heavy kernel imports of dsa_dcp.py are stubbed, so this needs only torch.
"""
import importlib.util
import os
import sys
import types
import unittest

import torch

_DCP = os.path.join(
    os.path.dirname(__file__), "..", "..", "python", "sglang", "srt", "layers", "attention", "dsa", "dsa_dcp.py"
)


def _load_dsa_dcp():
    for name in (
        "msgspec",
        "sglang",
        "sglang.kernels",
        "sglang.kernels.ops",
        "sglang.kernels.ops.attention",
        "sglang.kernels.ops.attention.dsa",
        "sglang.kernels.ops.attention.dsa.transform_index",
        "sglang.kernels.ops.attention.fixup_zero_kv",
        "sglang.srt",
        "sglang.srt.runtime_context",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["msgspec"].Struct = object
    sys.modules["sglang.kernels.ops.attention.dsa.transform_index"].transform_index_page_table_dcp = None
    sys.modules["sglang.kernels.ops.attention.fixup_zero_kv"].fixup_zero_kv_rows = None
    sys.modules["sglang.srt.runtime_context"].get_parallel = lambda: (_ for _ in ()).throw(AssertionError("DCP group must not be used"))
    spec = importlib.util.spec_from_file_location("dsa_dcp_under_test", _DCP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeGroup:
    def __init__(self, world, rank, bufs):
        self.world_size, self.rank_in_group, self.bufs = world, rank, bufs

    def all_gather_into_tensor(self, out, inp):
        per = inp.shape[0]
        for r in range(self.world_size):
            out[r * per : (r + 1) * per] = self.bufs[r][r * per : (r + 1) * per]


class TestRowSplit(unittest.TestCase):
    def test_split_matches_unsplit(self):
        dcp = _load_dsa_dcp()
        torch.manual_seed(0)
        topk, keys = 16, 300
        for world in (2, 4):
            for rows in (1, 3, 5, 8, 63, 64, 65, 1001):
                logits = torch.randn(rows, keys)
                ref = torch.topk(logits, topk, dim=1).indices.to(torch.int32)
                bufs = [torch.full((-(-rows // world) * world, topk), -7, dtype=torch.int32) for _ in range(world)]
                groups = [FakeGroup(world, r, bufs) for r in range(world)]
                covered = []
                for r in range(world):  # phase 1: every rank scores its own stride
                    per, slot, lo, hi = dcp.dcp_split_rows(rows, groups[r])
                    self.assertEqual(per, -(-rows // world))
                    self.assertEqual(slot, r * per)
                    covered += list(range(lo, hi))
                    if hi > lo:
                        bufs[r][lo:hi] = torch.topk(logits[lo:hi], topk, dim=1).indices.to(torch.int32)
                self.assertEqual(covered, list(range(rows)))  # strides tile the rows exactly once
                for r in range(world):  # phase 2: in-place all-gather on every rank
                    per, slot, _, _ = dcp.dcp_split_rows(rows, groups[r])
                    dcp.dcp_all_gather_rows_(bufs[r], per, slot, groups[r])
                    self.assertTrue(torch.equal(bufs[r][:rows], ref), f"world={world} rows={rows} rank={r}")


if __name__ == "__main__":
    unittest.main()
