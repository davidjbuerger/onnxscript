# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for exporting ``nn.EmbeddingBag`` with ``padding_idx``."""

from __future__ import annotations

import unittest

import numpy as np
import onnx
import onnxruntime as ort
import parameterized
import torch

_NUM_EMBEDDINGS = 40
_EMBEDDING_DIM = 5


def _max_loop_depth(graph: onnx.GraphProto) -> int:
    """Return how deeply ``Loop`` nodes are nested in ``graph`` (0 if there is none)."""
    depth = 0
    for node in graph.node:
        inner = max(
            (
                _max_loop_depth(attr.g)
                for attr in node.attribute
                if attr.type == onnx.AttributeProto.GRAPH
            ),
            default=0,
        )
        depth = max(depth, inner + (1 if node.op_type == "Loop" else 0))
    return depth


class _Bag(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.bag = torch.nn.EmbeddingBag(_NUM_EMBEDDINGS, _EMBEDDING_DIM, **kwargs)

    def forward(self, indices, offsets):
        return self.bag(indices, offsets)


class _WeightedBag(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.bag = torch.nn.EmbeddingBag(_NUM_EMBEDDINGS, _EMBEDDING_DIM, **kwargs)

    def forward(self, indices, offsets, per_sample_weights):
        return self.bag(indices, offsets, per_sample_weights=per_sample_weights)


def _export_and_run(module: torch.nn.Module, args: tuple[torch.Tensor, ...]):
    program = torch.onnx.export(module, args, dynamo=True, report=False)
    assert program is not None
    model = program.model_proto
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    feeds = {i.name: a.numpy() for i, a in zip(session.get_inputs(), args)}
    return model, session.run(None, feeds)[0]


def _ragged_inputs(padding_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Bags with padding entries, an all-padding bag and an empty bag."""
    p = padding_index
    bags = [[3, p, 5, p, 9], [p, p, p], [], [1, 2, p, 4]]
    indices = torch.tensor([i for bag in bags for i in bag], dtype=torch.int64)
    offsets = torch.tensor(
        np.cumsum([0] + [len(bag) for bag in bags[:-1]]), dtype=torch.int64
    )
    return indices, offsets


class EmbeddingBagPaddingIdxTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            (f"{mode}_padding_idx_{padding_idx}", mode, padding_idx)
            for mode in ("sum", "mean", "max")
            for padding_idx in (7, -1)
        ]
    )
    def test_matches_torch(self, _, mode: str, padding_idx: int):
        torch.manual_seed(0)
        module = _Bag(mode=mode, padding_idx=padding_idx).eval()
        normalized = padding_idx % _NUM_EMBEDDINGS
        args = _ragged_inputs(normalized)
        _, result = _export_and_run(module, args)
        expected = module(*args).detach().numpy()
        self.assertEqual(result.shape, expected.shape)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_per_sample_weights_match_torch(self):
        torch.manual_seed(0)
        module = _WeightedBag(mode="sum", padding_idx=7).eval()
        indices, offsets = _ragged_inputs(7)
        args = (indices, offsets, torch.randn(indices.shape[0]))
        _, result = _export_and_run(module, args)
        expected = module(*args).detach().numpy()
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    @parameterized.parameterized.expand([("sum",), ("mean",), ("max",)])
    def test_no_loop_over_indices(self, mode: str):
        # The padding index is handled with a mask. A ``Loop`` nested inside the
        # per-bag ``Loop`` runs once per index, which is extremely slow on large
        # inputs (hundreds of milliseconds on CPU, seconds on GPU for 80k indices).
        torch.manual_seed(0)
        module = _Bag(mode=mode, padding_idx=7).eval()
        model, _ = _export_and_run(module, _ragged_inputs(7))
        self.assertLessEqual(_max_loop_depth(model.graph), 1)


if __name__ == "__main__":
    unittest.main()
