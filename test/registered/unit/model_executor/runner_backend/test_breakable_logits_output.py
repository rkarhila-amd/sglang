"""CPU tests for breakable-graph LogitsProcessorOutput buffers."""

import unittest

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _backend() -> BreakableCudaGraphBackend:
    return BreakableCudaGraphBackend.__new__(BreakableCudaGraphBackend)


class TestBreakableLogitsOutput(CustomTestCase):
    def test_logits_and_hidden_rows_share_one_buffer(self):
        backend = _backend()
        logits = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        hidden = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        output = LogitsProcessorOutput(
            next_token_logits=logits,
            hidden_states=hidden,
        )
        buffer = backend._alloc_full_buffer(output, 4)
        self.assertEqual(buffer.next_token_logits.shape, (4, 3))
        self.assertEqual(buffer.hidden_states.shape, (4, 2))
        self.assertIsNone(buffer.last_hidden_states)

        backend._copy_output_to_buffer(output, buffer, 4)
        stored = backend._slice_output(buffer, 4)
        torch.testing.assert_close(stored.next_token_logits, logits)
        torch.testing.assert_close(stored.hidden_states, hidden)

        # A shorter shape writes a prefix and leaves the tail of this buffer.
        short = LogitsProcessorOutput(
            next_token_logits=torch.full((2, 3), 9.0),
            hidden_states=torch.full((2, 2), 8.0),
        )
        backend._copy_output_to_buffer(short, buffer, 2)
        sliced = backend._slice_output(buffer, 2)
        torch.testing.assert_close(sliced.next_token_logits, short.next_token_logits)
        torch.testing.assert_close(sliced.hidden_states, short.hidden_states)
        torch.testing.assert_close(buffer.next_token_logits[2:], logits[2:])
        torch.testing.assert_close(buffer.hidden_states[2:], hidden[2:])

    def test_token_rows_grow_the_buffer_past_the_request_key(self):
        backend = _backend()
        backend._cuda_graph_runner = type(
            "_Runner",
            (),
            {"cuda_graph_output_rows": staticmethod(lambda output: None)},
        )()
        output = LogitsProcessorOutput(
            next_token_logits=torch.ones(6, 2),
            hidden_states=None,
        )
        self.assertEqual(backend._output_rows(output, 2), 6)
        buffer = backend._alloc_full_buffer(output, 2)
        self.assertEqual(buffer.next_token_logits.shape, (6, 2))
        backend._copy_output_to_buffer(output, buffer, 6)
        stored = backend._slice_output(buffer, 6)
        self.assertEqual(stored.next_token_logits.shape, (6, 2))
        self.assertIsNone(stored.hidden_states)

    def test_non_row_tensor_is_copied_whole(self):
        backend = _backend()
        embeds = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        output = LogitsProcessorOutput(
            next_token_logits=torch.ones(4, 2),
            mm_input_embeds=embeds,
        )
        buffer = backend._alloc_full_buffer(output, 4)
        self.assertEqual(buffer.mm_input_embeds.shape, (2, 3))
        backend._copy_output_to_buffer(output, buffer, 4)
        stored = backend._slice_output(buffer, 4)
        self.assertEqual(stored.next_token_logits.shape, (4, 2))
        torch.testing.assert_close(stored.mm_input_embeds, embeds)

    def test_python_logits_field_is_rejected(self):
        backend = _backend()
        output = LogitsProcessorOutput(
            next_token_logits=torch.ones(2, 2),
            next_token_top_logprobs_val=[[0.0]],
        )
        with self.assertRaises(TypeError):
            backend._alloc_full_buffer(output, 2)

    def test_plain_tensor_buffer_is_unchanged(self):
        backend = _backend()
        source = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        buffer = backend._alloc_full_buffer(source, 4)
        buffer.zero_()
        backend._copy_output_to_buffer(source, buffer, 2)
        torch.testing.assert_close(buffer[:2], source[:2])
        self.assertEqual(buffer[2:].count_nonzero().item(), 0)

    def test_unknown_output_still_raises(self):
        backend = _backend()
        with self.assertRaises(TypeError):
            backend._alloc_full_buffer(object(), 2)


if __name__ == "__main__":
    unittest.main()
