"""Unit tests for EAGLE spec token resolution."""

import unittest

from sglang.srt.speculative.spec_token_resolve import (
    UNFILLED_PREDICT_TOKEN,
    resolve_spec_accept_tokens,
    sanitize_accept_tokens,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSpecTokenResolve(CustomTestCase):
    def test_sanitize_stops_at_sentinel(self):
        self.assertEqual(
            sanitize_accept_tokens([101, 102, UNFILLED_PREDICT_TOKEN, 999]),
            [101, 102],
        )

    def test_linear_slice_drops_trailing_sentinel(self):
        flat = [101, UNFILLED_PREDICT_TOKEN, UNFILLED_PREDICT_TOKEN, 0, 0, 0]
        self.assertEqual(
            resolve_spec_accept_tokens(flat, req_idx=0, accept_len=3, stride=3),
            [101],
        )

    def test_gather_via_accept_index(self):
        flat = [UNFILLED_PREDICT_TOKEN] * 6
        flat[2] = 201
        flat[4] = 202
        flat[5] = 203
        tokens = resolve_spec_accept_tokens(
            flat,
            req_idx=0,
            accept_len=3,
            stride=6,
            accept_index_row=[2, 4, 5],
        )
        self.assertEqual(tokens, [201, 202, 203])

    def test_gather_skips_negative_index_padding(self):
        flat = list(range(100, 112))
        flat[8] = 201
        flat[10] = 202
        tokens = resolve_spec_accept_tokens(
            flat,
            req_idx=1,
            accept_len=3,
            stride=6,
            accept_index_row=[8, -1, 10],
        )
        self.assertEqual(tokens, [201, 202])


if __name__ == "__main__":
    unittest.main()
