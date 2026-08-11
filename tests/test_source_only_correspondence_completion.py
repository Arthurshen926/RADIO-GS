import torch

from radio_gs.querying.query_spec import PrimitiveUnaryEvidence
from radio_gs.querying.source_only_correspondence_completion import (
    source_only_one_hop_correspondence_completion,
)
from radio_gs.querying.support_solver import PrimitiveSupportGraph


def _graph(*, appearance=None, boundary=None):
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    app = torch.ones(4) if appearance is None else torch.tensor(appearance).float()
    bnd = torch.ones(4) if boundary is None else torch.tensor(boundary).float()
    return PrimitiveSupportGraph(
        edge_index=edges,
        edge_weight=torch.ones(4),
        raw_affinity=torch.ones(4),
        local_sigma=torch.ones(3),
        num_nodes=3,
        edge_channels={"appearance": app, "boundary": bnd},
    )


def test_completes_only_exact_abstain_rows_and_preserves_anchors_bitwise():
    observation = PrimitiveUnaryEvidence(
        torch.tensor([0.8, 0.0, -0.7]),
        "test",
        torch.tensor([0.8, 0.0, 0.7]),
    )
    visibility = torch.tensor([[1, 1], [1, 1], [0, 1]], dtype=torch.bool)
    completed, diagnostics = source_only_one_hop_correspondence_completion(
        _graph(),
        observation,
        torch.tensor([0.9, 0.8, 0.1]),
        visibility,
        hard_seed_threshold=0.2,
    )

    assert torch.equal(completed.values[[0, 2]], observation.values[[0, 2]])
    assert torch.equal(
        completed.confidence[[0, 2]], observation.confidence[[0, 2]]
    )
    assert completed.confidence[1] > 0
    assert diagnostics.observed_rows == 2
    assert diagnostics.positive_anchor_rows == 1
    assert diagnostics.negative_anchor_rows == 1
    assert diagnostics.completed_rows == 1


def test_requires_covisibility_and_both_typed_relations():
    observation = PrimitiveUnaryEvidence(
        torch.tensor([1.0, 0.0, 0.0]),
        "test",
        torch.tensor([1.0, 0.0, 0.0]),
    )
    no_covisibility = torch.tensor(
        [[1, 0], [0, 1], [0, 1]], dtype=torch.bool
    )
    completed, diagnostics = source_only_one_hop_correspondence_completion(
        _graph(),
        observation,
        torch.tensor([1.0, 0.9, 0.5]),
        no_covisibility,
        hard_seed_threshold=0.2,
    )
    assert completed.confidence[1] == 0
    assert diagnostics.completed_rows == 0

    visibility = torch.ones((3, 2), dtype=torch.bool)
    completed, diagnostics = source_only_one_hop_correspondence_completion(
        _graph(boundary=[0.0, 0.0, 1.0, 1.0]),
        observation,
        torch.tensor([1.0, 0.9, 0.5]),
        visibility,
        hard_seed_threshold=0.2,
    )
    assert completed.confidence[1] == 0
    assert diagnostics.completed_rows == 0


def test_query_independent_reliability_scales_confidence_not_probability():
    observation = PrimitiveUnaryEvidence(
        torch.tensor([1.0, 0.0, 0.0]),
        "test",
        torch.tensor([1.0, 0.0, 0.0]),
    )
    visibility = torch.ones((3, 2), dtype=torch.bool)
    full, _ = source_only_one_hop_correspondence_completion(
        _graph(),
        observation,
        torch.tensor([1.0, 0.75, 0.5]),
        visibility,
        hard_seed_threshold=0.2,
        query_independent_reliability=torch.ones(3),
    )
    half, _ = source_only_one_hop_correspondence_completion(
        _graph(),
        observation,
        torch.tensor([1.0, 0.75, 0.5]),
        visibility,
        hard_seed_threshold=0.2,
        query_independent_reliability=torch.tensor([1.0, 0.5, 1.0]),
    )
    assert torch.allclose(half.confidence[1], 0.5 * full.confidence[1])
    assert torch.allclose(
        half.values[1] / half.confidence[1],
        full.values[1] / full.confidence[1],
    )
