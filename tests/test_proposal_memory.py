import pytest
import torch

from radio_gs.models.proposal_memory import (
    build_proposal_memory_from_labels,
    build_voxel_proposal_labels,
    propagate_logits_with_proposals,
)


def test_build_proposal_memory_pools_values_with_confidence_weights():
    values = torch.tensor(
        [
            [1.0, 0.0],
            [3.0, 2.0],
            [0.0, 4.0],
            [2.0, 6.0],
            [99.0, 99.0],
        ]
    )
    labels = torch.tensor([2, 2, 7, 7, -1])
    confidence = torch.tensor([1.0, 3.0, 1.0, 1.0, 100.0])

    memory = build_proposal_memory_from_labels(values, labels, confidence=confidence)

    assert memory.proposal_ids.tolist() == [2, 7]
    assert torch.allclose(memory.pooled_values[0], torch.tensor([2.5, 1.5]))
    assert torch.allclose(memory.pooled_values[1], torch.tensor([1.0, 5.0]))
    assert memory.counts.tolist() == [2, 2]
    assert torch.allclose(memory.weight_sums, torch.tensor([4.0, 2.0]))


def test_propagate_logits_with_proposals_residual_blends_assigned_rows_only():
    logits = torch.tensor(
        [
            [4.0, 0.0],
            [0.0, 4.0],
            [0.0, 3.0],
            [10.0, 10.0],
        ]
    )
    labels = torch.tensor([0, 0, 1, -1])

    propagated, stats = propagate_logits_with_proposals(
        logits,
        labels,
        alpha=0.5,
    )

    assert stats["enabled"] is True
    assert stats["num_proposals"] == 2
    assert torch.allclose(propagated[0], torch.tensor([3.0, 1.0]))
    assert torch.allclose(propagated[1], torch.tensor([1.0, 3.0]))
    assert torch.allclose(propagated[2], logits[2])
    assert torch.allclose(propagated[3], logits[3])


def test_propagate_logits_with_proposals_can_gate_low_margin_rows():
    logits = torch.tensor(
        [
            [5.0, 0.0],
            [0.2, 0.0],
            [0.0, 4.0],
        ]
    )
    labels = torch.tensor([0, 0, 1])

    propagated, stats = propagate_logits_with_proposals(
        logits,
        labels,
        alpha=1.0,
        gate="low_margin",
        margin_threshold=0.5,
    )

    assert stats["enabled"] is True
    assert stats["num_assigned"] == 1
    assert torch.allclose(propagated[0], logits[0])
    assert torch.allclose(propagated[1], torch.tensor([2.6, 0.0]))
    assert torch.allclose(propagated[2], logits[2])


def test_propagate_logits_with_proposals_can_gate_low_confidence_rows():
    logits = torch.tensor(
        [
            [8.0, 0.0],
            [0.2, 0.1],
            [0.0, 4.0],
        ]
    )
    labels = torch.tensor([0, 0, 0])

    propagated, stats = propagate_logits_with_proposals(
        logits,
        labels,
        alpha=1.0,
        gate="low_confidence",
        confidence_threshold=0.60,
    )

    pooled = logits.mean(dim=0)
    assert stats["gate"] == "low_confidence"
    assert stats["confidence_threshold"] == 0.60
    assert stats["num_assigned"] == 1
    assert torch.allclose(propagated[0], logits[0])
    assert torch.allclose(propagated[1], pooled)
    assert torch.allclose(propagated[2], logits[2])


def test_build_voxel_proposal_labels_groups_points_by_voxel():
    xyz = torch.tensor(
        [
            [0.01, 0.01, 0.01],
            [0.09, 0.02, 0.01],
            [0.21, 0.01, 0.01],
            [0.20, 0.08, 0.01],
        ],
        dtype=torch.float32,
    )

    labels = build_voxel_proposal_labels(xyz, voxel_size=0.1)

    assert labels.shape == (4,)
    assert labels[0].item() == labels[1].item()
    assert labels[2].item() == labels[3].item()
    assert labels[0].item() != labels[2].item()


def test_proposal_memory_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same number"):
        build_proposal_memory_from_labels(
            torch.zeros(3, 2),
            torch.zeros(2, dtype=torch.long),
        )
