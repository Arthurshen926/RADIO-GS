import pytest
import torch

from radio_gs.v4.object_memory import (
    DenseObjectAssignments,
    ObjectCodebook,
    SparseObjectAssignments,
)
from radio_gs.v4.query import QueryPacket


def test_top2_assignment_preserves_unknown_and_dropped_mass():
    dense = torch.tensor([[0.5, 0.3, 0.1], [0.0, 0.0, 0.0]])
    assignments = SparseObjectAssignments.from_dense(
        dense, unknown_weight=torch.tensor([0.1, 1.0]), top_k=2
    )
    assert assignments.token_ids[0].tolist() == [0, 1]
    assert assignments.weights[0].tolist() == pytest.approx([0.5, 0.3])
    assert float(assignments.unknown_weight[0]) == pytest.approx(0.2)
    assert float(assignments.unknown_weight[1]) == 1.0
    assert assignments.to_dense()[0].tolist() == pytest.approx([0.5, 0.3, 0.0])


def test_one_token_selection_produces_one_element_posterior():
    assignments = SparseObjectAssignments.from_dense(
        torch.tensor([[0.8, 0.1], [0.0, 0.7]]),
        unknown_weight=torch.tensor([0.1, 0.3]),
    )
    posterior = assignments.element_posterior(
        QueryPacket("single_instance"),
        torch.tensor([0.9, 0.0]),
        null_probability=0.1,
    )
    assert posterior.foreground.tolist() == pytest.approx([0.72, 0.0])
    assert posterior.assignment_unknown.tolist() == pytest.approx([0.09, 0.27])
    codebook = ObjectCodebook.from_assignments(
        torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]), assignments
    )
    assert codebook.centres.shape == (2, 3)
    assert bool((codebook.confidence >= 0).all())


def test_query_posterior_requires_explicit_null_simplex():
    assignments = SparseObjectAssignments.from_dense(torch.tensor([[0.8, 0.1]]))
    with pytest.raises(ValueError, match="simplex"):
        assignments.element_posterior(
            QueryPacket("single_instance"),
            torch.tensor([0.8, 0.1]),
            null_probability=0.2,
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        assignments.element_posterior(
            QueryPacket("single_instance"),
            torch.tensor([1.1, 0.0]),
            null_probability=-0.1,
        )


def test_training_assignments_remain_dense_until_explicit_compression():
    dense = DenseObjectAssignments.from_logits(
        torch.tensor([[2.0, 1.0, 0.5], [0.0, 0.0, 0.0]]),
        torch.tensor([-1.0, 1.0]),
    )
    assert dense.token_probability.shape == (2, 3)
    assert torch.allclose(
        dense.token_probability.sum(-1) + dense.unknown_probability,
        torch.ones(2),
    )
    assert dense.compress(top_k=2).token_ids.shape == (2, 2)


def test_token_confidence_ignores_far_away_unknown_elements():
    points = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [100.0, 0.0, 0.0]])
    assignments = SparseObjectAssignments.from_dense(
        torch.tensor([[1.0], [1.0], [0.0]]),
        unknown_weight=torch.tensor([0.0, 0.0, 1.0]),
    )
    codebook = ObjectCodebook.from_assignments(points, assignments)
    assert float(codebook.confidence[0]) == pytest.approx(1.0)
