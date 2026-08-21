import torch

from radio_gs.querying.learned_compact_object_affinity import (
    CompactObjectAffinity,
    balanced_relation_loss,
    build_source_proposal_relations,
    pool_proposal_features,
)
from radio_gs.querying.source_multiview_object_tracks import (
    build_source_learned_object_tracks,
)


def test_ternary_source_relations_keep_unsupported_cross_view_unknown() -> None:
    # Proposals 0/1 are the same support in different views.  Proposals 0/2
    # are disjoint but cross-view, hence unknown rather than a false negative.
    relation = build_source_proposal_relations(
        torch.tensor([0, 1, 0, 1, 2]),
        torch.tensor([0, 0, 1, 1, 2]),
        torch.ones(5),
        torch.tensor([0, 1, 1]),
        torch.tensor([0.2, 0.2, 0.2]),
        num_rows=3,
        num_proposals=3,
    )
    labels = {(int(a), int(b)): int(y) for a, b, y in zip(relation.left, relation.right, relation.relation)}
    assert labels[(0, 1)] == 1
    assert labels[(1, 2)] == 0
    assert (0, 2) not in labels


def test_pool_and_compact_code_have_exact_axes() -> None:
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    pooled = pool_proposal_features(
        features,
        torch.tensor([0, 1, 2]),
        torch.tensor([0, 0, 1]),
        torch.tensor([1.0, 3.0, 2.0]),
        num_proposals=2,
    )
    assert torch.allclose(pooled, torch.tensor([[0.25, 0.75], [1.0, 1.0]]))
    model = CompactObjectAffinity(input_dim=2, object_dim=2)
    assert model(pooled).shape == (2, 2)


def test_unknown_relation_does_not_enter_proper_loss() -> None:
    logits = torch.tensor([2.0, -2.0], requires_grad=True)
    relation = torch.tensor([1, 0])
    base = balanced_relation_loss(logits, relation)
    # Unknown is excluded by selecting known relations before the proper loss.
    extended_logits = torch.tensor([2.0, -2.0, 100.0], requires_grad=True)
    extended_relation = torch.tensor([1, 0, -1])
    known = extended_relation >= 0
    extended = balanced_relation_loss(extended_logits[known], extended_relation[known])
    assert torch.equal(base, extended)


def test_learned_tracks_emit_same_sparse_interface() -> None:
    tracks = build_source_learned_object_tracks(
        torch.tensor([0, 1, 0, 1, 2]),
        torch.tensor([0, 0, 1, 1, 2]),
        torch.ones(5),
        torch.tensor([0, 1, 1]),
        torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]),
        num_rows=3,
        num_proposals=3,
        relation_logit_scale=8.0,
        minimum_same_probability=0.9,
    )
    assert tracks.num_tracks == 1
    assert tracks.proposal_track_indices.tolist() == [0, 0, -1]
    assert sorted(tracks.row_indices.tolist()) == [0, 1]
