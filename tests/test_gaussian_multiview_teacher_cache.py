import json

import pytest
import torch

from radio_gs.scripts.build_gaussian_multiview_teacher_cache import (
    _load_extracted_capability_maps,
    _load_responsibility_cache,
    _resolve_extracted_capability_source,
    accumulate_contribution_mean_channel_chunked,
    merge_topk_view_observations,
    raster_fusion_reliability,
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


def test_mean_resultant_reliability_measures_directional_agreement() -> None:
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.5, 0.0],
            [0.0, 0.0],
        ]
    )
    reliability = raster_fusion_reliability(
        features,
        torch.tensor([True, True, False]),
        torch.tensor([4, 2, 0]),
        num_views=4,
        mode="mean_resultant",
        normalized_observations=True,
    ).float()

    torch.testing.assert_close(
        reliability,
        torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [0.5, 0.5, 1.0],
                [0.0, 0.0, 0.0],
            ]
        ),
    )


def test_mean_resultant_reliability_rejects_unnormalized_observations() -> None:
    with pytest.raises(ValueError, match="normalized observations"):
        raster_fusion_reliability(
            torch.ones(1, 2),
            torch.ones(1, dtype=torch.bool),
            torch.ones(1),
            num_views=1,
            mode="mean_resultant",
            normalized_observations=False,
        )


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
    alias_contract = {
        **contract,
        "config": "/different/feature/source.yaml",
        "selected_dataset_indices": [0],
    }
    assignments, _ = _load_responsibility_cache(
        path,
        expected_contract=alias_contract,
        num_gaussians=3,
    )
    assert len(assignments) == 1
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


def test_chunked_contribution_mean_reuses_cpu_transfer_staging() -> None:
    feature = torch.tensor(
        [[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]
    )
    gids = torch.tensor([0, 1])
    pids = torch.tensor([0, 1])
    weights = torch.tensor([0.25, 0.75])
    sums = torch.zeros(2, 3)
    counts = torch.zeros(2)
    sum_staging = torch.empty(2, 2)
    count_staging = torch.empty(2)

    frame_counts = accumulate_contribution_mean_channel_chunked(
        feature,
        gids,
        pids,
        weights,
        sums,
        counts,
        channel_chunk_size=2,
        cpu_sum_staging=sum_staging,
        cpu_count_staging=count_staging,
    )

    assert frame_counts.data_ptr() == count_staging.data_ptr()
    torch.testing.assert_close(counts, torch.tensor([0.25, 0.75]))
    torch.testing.assert_close(
        sums,
        torch.tensor([[0.25, 0.75, 1.25], [1.50, 3.00, 4.50]]),
    )


def test_official_extracted_capability_source_requires_the_matching_manifest(
    tmp_path,
) -> None:
    """A direct SAM/DINO MPR must come from the official extractor output.

    This protects the high-spatial-fidelity route from silently falling back to
    an arbitrary tensor directory whose rows happen to have a compatible
    dimensionality.
    """

    manifest = {
        "radio": {"version": "c-radio_v4-h"},
        "features": {
            "adaptors": [
                {
                    "name": "dino_v3_7b",
                    "subdir": "dino_v3_7b",
                    "dim": 4096,
                    "grid": [30, 40],
                    "dtype": "float16",
                },
                {
                    "name": "sam3",
                    "subdir": "sam3",
                    "dim": 1024,
                    "grid": [30, 40],
                    "dtype": "float16",
                },
            ]
        },
    }
    (tmp_path / "frame_manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "sam3").mkdir()

    source = _resolve_extracted_capability_source(tmp_path, "sam3")

    assert source["subdir"] == "sam3"
    assert source["adaptor_name"] == "sam3"
    assert source["output_dim"] == 1024
    assert source["native_grid"] == [30, 40]

    manifest["features"]["adaptors"] = manifest["features"]["adaptors"][:1]
    (tmp_path / "frame_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="does not declare"):
        _resolve_extracted_capability_source(tmp_path, "sam3")


def test_extracted_capability_maps_are_selected_by_raw_frame_id_and_then_resampled(
    tmp_path,
    monkeypatch,
) -> None:
    feature_root = tmp_path / "features"
    (feature_root / "sam3").mkdir(parents=True)
    pose_dir = tmp_path / "pose"
    pose_dir.mkdir()
    torch.save(torch.arange(1024, dtype=torch.float16).reshape(1024, 1, 1), feature_root / "sam3" / "rgb_7.pt")
    (pose_dir / "7.txt").write_text(
        "1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n"
    )
    (feature_root / "frame_manifest.json").write_text(
        json.dumps(
            {
                "radio": {"version": "c-radio_v4-h"},
                "features": {
                    "adaptors": [
                        {
                            "name": "sam3",
                            "subdir": "sam3",
                            "dim": 1024,
                            "grid": [1, 1],
                            "dtype": "float16",
                        }
                    ]
                },
            }
        )
    )

    def reject_full_selection_stack(*args, **kwargs):
        raise AssertionError("official capability loader must preallocate, not stack")

    monkeypatch.setattr(torch, "stack", reject_full_selection_stack)
    maps, source = _load_extracted_capability_maps(
        feature_dir=feature_root,
        feature_space="sam3",
        pose_file=None,
        pose_dir=str(pose_dir),
        feature_size=(2, 3),
        dataset_type="scannet",
        selected_frame_indices=[7],
    )

    assert maps.shape == (1, 1024, 2, 3)
    torch.testing.assert_close(
        maps[0].norm(dim=0), torch.ones(2, 3), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(maps[0, 1, 0, 0] / maps[0, 2, 0, 0], torch.tensor(0.5))
    assert source["subdir"] == "sam3"
