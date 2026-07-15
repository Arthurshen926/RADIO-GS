import pytest
import torch

from radio_gs.scripts.build_gaussian_multiview_teacher_cache import (
    _load_responsibility_cache,
    accumulate_contribution_mean_channel_chunked,
    merge_topk_view_observations,
)


def test_topk_view_fusion_preserves_whole_observation_vectors() -> None:
    current = torch.tensor([[[1.0, 3.0], [10.0, 30.0]]])
    responsibility = torch.tensor([[0.1, 0.3]])
    new_observation = torch.tensor([[2.0, 20.0]])

    features, scores = merge_topk_view_observations(
        current,
        responsibility,
        new_observation,
        torch.tensor([0.2]),
    )

    torch.testing.assert_close(scores, torch.tensor([[0.3, 0.2]]))
    torch.testing.assert_close(
        features,
        torch.tensor([[[3.0, 2.0], [30.0, 20.0]]]),
    )


def test_topk_view_fusion_keeps_channels_from_same_ranked_view() -> None:
    current = torch.zeros(2, 3, 1)
    responsibility = torch.full((2, 1), -float("inf"))
    observation = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    features, scores = merge_topk_view_observations(
        current,
        responsibility,
        observation,
        torch.tensor([0.5, 0.8]),
    )

    torch.testing.assert_close(features[..., 0], observation)
    torch.testing.assert_close(scores[:, 0], torch.tensor([0.5, 0.8]))


def test_shared_responsibility_cache_is_feature_independent_and_fail_closed(
    tmp_path,
) -> None:
    contract = {
        "selected_frame_indices": [3],
        "feature_height": 2,
        "feature_width": 3,
        "gaussian_state_sha256": "geometry",
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }
    path = tmp_path / "responsibility.pt"
    torch.save(
        {
            "schema_version": 1,
            "metadata": contract,
            "assignments": [
                {
                    "gaussian_ids": torch.tensor([0, 2], dtype=torch.int32),
                    "pixel_ids": torch.tensor([1, 5], dtype=torch.int32),
                    "weights": torch.tensor([0.25, 0.75]),
                }
            ],
        },
        path,
    )

    assignments, digest = _load_responsibility_cache(
        path,
        expected_contract=contract,
        num_gaussians=3,
    )

    assert len(assignments) == 1
    assert assignments[0]["gaussian_ids"].tolist() == [0, 2]
    assert len(digest) == 64
    with pytest.raises(ValueError, match="contract differs"):
        _load_responsibility_cache(
            path,
            expected_contract={**contract, "gaussian_state_sha256": "other"},
            num_gaussians=3,
        )


def test_chunked_contribution_mean_matches_full_weighted_scatter() -> None:
    feature = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [[9.0, 10.0], [11.0, 12.0]]]
    )
    gids = torch.tensor([0, 1, 0, 2])
    pids = torch.tensor([0, 1, 3, 2])
    weights = torch.tensor([1.0, 0.5, 2.0, 0.25])
    sums = torch.zeros(3, 3)
    counts = torch.zeros(3)

    frame_counts = accumulate_contribution_mean_channel_chunked(
        feature,
        gids,
        pids,
        weights,
        sums,
        counts,
        channel_chunk_size=2,
    )

    flat = feature.reshape(3, 4).t()
    expected = torch.zeros_like(sums)
    expected.index_add_(0, gids, flat[pids] * weights[:, None])
    expected_counts = torch.zeros(3).index_add_(0, gids, weights)
    torch.testing.assert_close(sums, expected)
    torch.testing.assert_close(counts, expected_counts)
    torch.testing.assert_close(frame_counts, expected_counts)
