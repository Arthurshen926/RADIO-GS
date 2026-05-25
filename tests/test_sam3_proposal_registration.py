import torch

from radio_gs.models.sam3_proposal_registration import (
    build_sam3_mask_memberships,
    fuse_scores_with_query_sam3_proposals,
    fuse_scores_with_sam3_proposals,
)


def test_build_memberships_from_logits_keeps_confident_pairs():
    logits = torch.tensor([[[4.0, -4.0], [-4.0, 4.0]]])
    pixels_xy = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    scores = torch.tensor([0.8])

    memberships = build_sam3_mask_memberships(
        logits,
        pixels_xy,
        scores=scores,
        min_probability=0.5,
    )

    assert memberships.row_indices.tolist() == [0, 1]
    assert memberships.proposal_indices.tolist() == [0, 0]
    assert torch.all(memberships.weights > 0.5)
    assert memberships.num_rows == 2
    assert memberships.num_proposals == 1


def test_build_memberships_filters_invisible_and_out_of_bounds_rows():
    logits = torch.full((1, 4, 4), 8.0)
    pixels_xy = torch.tensor([[1.0, 1.0], [5.0, 1.0], [2.0, 2.0]])
    visibility = torch.tensor([1.0, 1.0, 0.0])

    memberships = build_sam3_mask_memberships(
        logits,
        pixels_xy,
        visibility=visibility,
        min_probability=0.5,
    )

    assert memberships.row_indices.tolist() == [0]
    assert memberships.proposal_indices.tolist() == [0]


def test_build_memberships_uses_dense_proposal_ids_after_topk_masks():
    logits = torch.full((3, 2, 2), -8.0)
    logits[:, 0, 0] = 8.0
    scores = torch.tensor([0.1, 0.9, 0.8])

    memberships = build_sam3_mask_memberships(
        logits,
        torch.tensor([[0.0, 0.0]]),
        scores=scores,
        mask_query_indices=torch.tensor([7, 8, 9]),
        min_probability=0.5,
        max_masks=2,
        proposal_offset=10,
    )

    assert memberships.proposal_indices.tolist() == [10, 11]
    assert memberships.num_proposals == 2
    assert memberships.proposal_query_indices.tolist() == [8, 9]


def test_fuse_scores_uses_proposal_pooled_scores_for_low_margin_rows():
    scores = torch.tensor([[0.55, 0.50], [0.90, 0.10]])
    row = torch.tensor([0, 1])
    prop = torch.tensor([0, 0])
    weights = torch.tensor([1.0, 1.0])

    fused, stats = fuse_scores_with_sam3_proposals(
        scores,
        row,
        prop,
        weights,
        alpha=0.5,
        gate="low_margin",
        margin_threshold=0.1,
    )

    assert fused[0, 0] > scores[0, 0]
    assert torch.allclose(fused[1], scores[1])
    assert stats["enabled"] is True
    assert stats["num_proposals"] == 1
    assert stats["num_assigned"] == 1


def test_empty_memberships_return_original_scores():
    scores = torch.randn(3, 2)

    fused, stats = fuse_scores_with_sam3_proposals(
        scores,
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        torch.empty(0),
        alpha=0.5,
    )

    assert torch.allclose(fused, scores)
    assert stats["enabled"] is False
    assert stats["num_memberships"] == 0


def test_query_conditioned_fusion_only_updates_matching_query_column():
    scores = torch.tensor(
        [
            [0.20, 0.90],
            [0.80, 0.10],
        ],
        dtype=torch.float32,
    )
    row = torch.tensor([0, 1])
    prop = torch.tensor([0, 0])
    weights = torch.tensor([1.0, 1.0])
    proposal_query_indices = torch.tensor([0])

    fused, stats = fuse_scores_with_query_sam3_proposals(
        scores,
        row,
        prop,
        weights,
        proposal_query_indices,
        alpha=0.5,
        gate="all",
    )

    assert fused[0, 0] > scores[0, 0]
    assert fused[1, 0] < scores[1, 0]
    assert torch.allclose(fused[:, 1], scores[:, 1])
    assert stats["query_conditioned"] is True
    assert stats["num_assigned"] == 2
