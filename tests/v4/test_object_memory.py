import pytest
import torch

from radio_gs.v4.object_memory import ObjectCodebook, SparseObjectAssignments


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
    posterior = assignments.element_posterior(torch.tensor([1.0, 0.0]))
    assert posterior.tolist() == pytest.approx([0.8, 0.0])
    codebook = ObjectCodebook.from_assignments(
        torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]), assignments
    )
    assert codebook.centres.shape == (2, 3)
    assert bool((codebook.confidence >= 0).all())
