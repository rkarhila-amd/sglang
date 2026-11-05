"""CPU checks for the DSPARK draft GQA expand.

The bidirectional block is packed into the GQA group so FlyDSL pa_decode
can run at query length 1. These tests do not launch a kernel.
"""

import unittest

import torch

from sglang.srt.layers.attention.dspark_draft_flydsl import (
    bucket_max_context,
    collapse_draft_outputs,
    expand_draft_queries,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDsparkDraftFlydslPack(CustomTestCase):
    def test_qlen_one_is_identity(self):
        q = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
        expanded = expand_draft_queries(q, qlen=1, n_kv=2, n_q=4)
        self.assertTrue(torch.equal(expanded, q))
        collapsed = collapse_draft_outputs(expanded, qlen=1, n_kv=2, n_q=4)
        self.assertTrue(torch.equal(collapsed, q))

    def test_roundtrip_draft_block(self):
        # TP4 draft rank: 8 q / 2 kv, query 7, head dim 128.
        batch, qlen, n_kv, n_q, dim = 3, 7, 2, 8, 128
        q = torch.randn(batch * qlen, n_q, dim)
        expanded = expand_draft_queries(q, qlen=qlen, n_kv=n_kv, n_q=n_q)
        self.assertEqual(expanded.shape, (batch, n_q * qlen, dim))
        collapsed = collapse_draft_outputs(expanded, qlen=qlen, n_kv=n_kv, n_q=n_q)
        self.assertTrue(torch.equal(collapsed, q))
        packed = torch.empty_like(expanded)
        expand_draft_queries(q, qlen=qlen, n_kv=n_kv, n_q=n_q, out=packed)
        self.assertTrue(torch.equal(packed, expanded))
        back = torch.empty_like(q)
        collapse_draft_outputs(packed, qlen=qlen, n_kv=n_kv, n_q=n_q, dest=back)
        self.assertTrue(torch.equal(back, q))

    def test_position_then_group_inside_each_kv_head(self):
        batch, qlen, n_kv, group, dim = 1, 2, 2, 2, 1
        n_q = n_kv * group
        q = torch.arange(batch * qlen * n_q * dim, dtype=torch.float32).reshape(
            batch * qlen, n_q, dim
        )
        expanded = expand_draft_queries(q, qlen=qlen, n_kv=n_kv, n_q=n_q)
        # Rows are (pos, head). Packed order is (kv, pos, group):
        # kv0 pos0, kv0 pos1, kv1 pos0, kv1 pos1.
        self.assertTrue(
            torch.equal(
                expanded.view(-1),
                torch.tensor([0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0]),
            )
        )

    def test_bucket_is_a_power_of_two_page(self):
        self.assertEqual(bucket_max_context(1), 16)
        self.assertEqual(bucket_max_context(16), 16)
        self.assertEqual(bucket_max_context(17), 32)
        self.assertEqual(bucket_max_context(64), 64)
        self.assertEqual(bucket_max_context(93007), 131072)
        with self.assertRaises(ValueError):
            bucket_max_context(0)


if __name__ == "__main__":
    unittest.main()
