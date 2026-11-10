import dataclasses
import unittest

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    alloc_logits_processor_output,
    copy_logits_processor_output,
    logits_processor_output_rows,
    slice_logits_processor_output,
)


class TestBcgLogitsOutput(unittest.TestCase):
    def test_buffer_keeps_token_rows_above_the_graph_key(self):
        src = LogitsProcessorOutput(
            next_token_logits=torch.arange(12, dtype=torch.float32).reshape(3, 4),
            hidden_states=torch.arange(6, dtype=torch.float32).reshape(3, 2),
        )
        self.assertEqual(logits_processor_output_rows(src), 3)
        # Graph key 2 is smaller than the 3 token rows verify can return.
        buf = alloc_logits_processor_output(src, 2)
        self.assertEqual(buf.next_token_logits.shape, (3, 4))
        self.assertEqual(buf.hidden_states.shape, (3, 2))
        self.assertIsNone(buf.next_token_logprobs)
        copy_logits_processor_output(src, buf, 3)
        sliced = slice_logits_processor_output(buf, 2)
        self.assertTrue(
            torch.equal(sliced.next_token_logits, src.next_token_logits[:2])
        )
        self.assertTrue(torch.equal(sliced.hidden_states, src.hidden_states[:2]))
        # The shared buffer itself still holds the third row.
        self.assertTrue(torch.equal(buf.next_token_logits, src.next_token_logits))

    def test_alloc_grows_to_the_requested_capacity(self):
        src = LogitsProcessorOutput(
            next_token_logits=torch.zeros(3, 4),
            hidden_states=torch.zeros(3, 2),
        )
        buf = alloc_logits_processor_output(src, 8)
        self.assertEqual(tuple(buf.next_token_logits.shape), (8, 4))
        self.assertEqual(tuple(buf.hidden_states.shape), (8, 2))

    def test_non_logits_object_is_not_treated_as_rows(self):
        other_cls = dataclasses.make_dataclass(
            "Other", [("next_token_logits", torch.Tensor)]
        )
        other = other_cls(torch.zeros(2, 2))
        self.assertIsNone(logits_processor_output_rows(other))


if __name__ == "__main__":
    unittest.main()
