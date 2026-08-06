from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from radio_gs.evaluation.source_query_response_hard_negatives import (
    build_multiview_teacher_targets,
    build_negative_authority,
    evaluate_source_query_response,
    mine_scene_global_hard_negatives,
    validate_negative_authority,
)


def test_multiview_response_is_log_mean_exp_and_consensus_is_query_independent() -> None:
    descriptors = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.6, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        dim=-1,
    )
    rows = torch.tensor([0, 0, 1], dtype=torch.int64)
    text = F.normalize(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), dim=-1)
    consensus, response, counts = build_multiview_teacher_targets(
        descriptors, rows, text, region_count=2, temperature=0.05
    )
    expected = 0.05 * (
        torch.logsumexp(torch.tensor([1.0, 0.8]) / 0.05, dim=0)
        - torch.log(torch.tensor(2.0))
    )
    assert torch.allclose(response[0, 0], expected)
    assert torch.equal(response[1], torch.tensor([0.0, 1.0]))
    assert torch.equal(counts, torch.tensor([2, 1]))
    assert torch.allclose(
        consensus[0], F.normalize(descriptors[:2].mean(dim=0), dim=0)
    )


def test_perfect_single_view_response_has_perfect_profiles_and_rankings() -> None:
    accepted = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        ),
        dim=-1,
    )
    text = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        dim=-1,
    )
    teacher = accepted @ text.T
    result = evaluate_source_query_response(
        accepted,
        teacher,
        text,
        scale_indices=torch.tensor([0, 0, 1, 1]),
        view_counts=torch.ones(4, dtype=torch.int64),
    )
    assert result["row_metrics"]["response_profile_cosine"]["mean"] == pytest.approx(1.0)
    assert result["row_metrics"]["response_mae"]["mean"] < 1e-7
    rank = result["query_rank_metrics"]
    assert rank["spearman"]["mean"] == pytest.approx(1.0)
    assert rank["top1_agreement"]["mean"] == pytest.approx(1.0)
    assert rank["top5_overlap"]["mean"] == pytest.approx(1.0)


def _miner_fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    consensus = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.90, 0.4359, 0.0],
                [0.85, 0.0, 0.5268],
                [0.0, 1.0, 0.0],
                [0.0, 0.90, 0.4359],
                [0.0, 0.0, 1.0],
            ]
        ),
        dim=-1,
    )
    response = torch.tensor(
        [
            [0.9, 0.1, -0.2, 0.0],
            [0.85, 0.12, -0.18, 0.01],
            [0.82, 0.08, -0.1, 0.04],
            [0.1, 0.9, 0.0, -0.2],
            [0.12, 0.84, 0.04, -0.18],
            [-0.1, 0.0, 0.9, 0.1],
        ],
        dtype=torch.float32,
    )
    region_rows = torch.tensor(
        [[0, 10], [1, 11], [2, 12], [3, 13], [4, 14], [5, 15]],
        dtype=torch.int64,
    )
    token_mask = torch.ones_like(region_rows, dtype=torch.bool)
    return consensus, response, region_rows, token_mask


def test_miner_is_spatially_disjoint_deterministic_and_authority_is_fail_closed() -> None:
    consensus, response, region_rows, token_mask = _miner_fixture()
    first = mine_scene_global_hard_negatives(
        consensus,
        response,
        region_rows,
        token_mask,
        scale_indices=torch.tensor([0, 0, 1, 1, 2, 2]),
        block_rows=2,
    )
    second = mine_scene_global_hard_negatives(
        consensus,
        response,
        region_rows,
        token_mask,
        scale_indices=torch.tensor([0, 0, 1, 1, 2, 2]),
        block_rows=3,
    )
    integer_channels = {
        "anchor_region_indices",
        "negative_region_indices",
        "source_codes",
        "teacher_similarity_ranks",
        "response_nearest_ranks",
        "anchor_scale_indices",
        "negative_scale_indices",
        "row_offsets",
    }
    assert all(torch.equal(first[key], second[key]) for key in integer_channels)
    assert torch.allclose(first["teacher_cosines"], second["teacher_cosines"])
    assert torch.allclose(
        first["response_profile_cosines"], second["response_profile_cosines"]
    )
    assert not torch.equal(
        first["anchor_region_indices"], first["negative_region_indices"]
    )
    authority = build_negative_authority(
        scene_id="scene0001_00",
        canonical_region_indices=torch.arange(6),
        region_fingerprints=[f"region-{index}" for index in range(6)],
        channels=first,
        input_authority={"fixture": {"sha256": "a" * 64}},
    )
    validate_negative_authority(
        authority, region_rows=region_rows, token_mask=token_mask
    )

    tampered_rows = region_rows.clone()
    anchor = int(first["anchor_region_indices"][0])
    negative = int(first["negative_region_indices"][0])
    tampered_rows[negative, 0] = tampered_rows[anchor, 0]
    with pytest.raises(ValueError, match="spatial overlap"):
        validate_negative_authority(
            authority, region_rows=tampered_rows, token_mask=token_mask
        )


def test_miner_rejects_constant_centered_response_profile() -> None:
    consensus, response, region_rows, token_mask = _miner_fixture()
    response[2] = 1.0
    with pytest.raises(ValueError, match="constant"):
        mine_scene_global_hard_negatives(
            consensus,
            response,
            region_rows,
            token_mask,
            scale_indices=torch.zeros(6, dtype=torch.int64),
        )
