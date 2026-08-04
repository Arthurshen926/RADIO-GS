import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from PIL import Image

from radio_gs.scripts import extract_radio_features as extraction
from radio_gs.scripts.build_gaussian_multiview_teacher_cache import (
    _load_extracted_capability_maps,
    _load_responsibility_cache,
    _resolve_extracted_capability_source,
    accumulate_contribution_mean_channel_chunked,
    estimate_capability_mpr_cpu_bytes,
    finalize_registered_mean_chunked,
    merge_topk_view_observations,
    prepare_raster_view_features,
    raster_fusion_reliability,
    validate_raster_reliability_policy,
)
from radio_gs.scripts.eval_lerf_direct_3d_selection import (
    select_top_raster_hits_per_gaussian,
)


def _strict_feature_bundle(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    frame_id: int,
) -> tuple[Path, Path, str]:
    """Create one real, strictly resumable extractor bundle without a GPU."""

    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(1, 2, 3)).save(
        image_dir / f"rgb_{frame_id}.png"
    )
    radio_repo = tmp_path / "RADIO"
    radio_repo.mkdir(parents=True, exist_ok=True)
    (radio_repo / "hubconf.py").write_text("# test runtime\n")
    checkpoint = tmp_path / "radio.pth"
    checkpoint.write_bytes(b"test-radio-checkpoint")
    output = tmp_path / "features"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        extraction,
        "_load_radio_model",
        lambda *_args, **_kwargs: (object(), object()),
    )

    def fake_preprocess(paths, _height, _width, _device):
        ids = [int(path.stem.rsplit("_", 1)[-1]) for path in paths]
        return torch.tensor(ids, dtype=torch.float32).reshape(-1, 1, 1, 1)

    def fake_forward(
        _model,
        _conditioner,
        images,
        _amp,
        _patch_h,
        _patch_w,
        adaptor_names=None,
    ):
        batch = int(images.shape[0])
        assert list(adaptor_names or []) == ["dino_v3_7b", "sam3"]
        return (
            torch.ones(batch, 2),
            torch.ones(batch, 8, 1, 1),
            {
                "dino_v3_7b": torch.ones(batch, 4096, 1, 1),
                "sam3": torch.arange(1024, dtype=torch.float32)
                .reshape(1, 1024, 1, 1)
                .repeat(batch, 1, 1, 1),
            },
        )

    monkeypatch.setattr(extraction, "_load_and_preprocess", fake_preprocess)
    monkeypatch.setattr(extraction, "_run_radio_batch", fake_forward)
    monkeypatch.setattr(extraction, "_thermal_pause", lambda *_args: None)
    extraction.extract(
        Namespace(
            scene="scene0001_00",
            image_dir=str(image_dir),
            output_dir=str(output),
            radio_repo=str(radio_repo),
            radio_version="c-radio_v4-h",
            radio_checkpoint=str(checkpoint),
            batch_size=1,
            frame_stride=1,
            max_frames=None,
            frame_id_mode="auto",
            exclude_image_stem=[],
            exclude_image_stems_file="",
            extract_adaptors=True,
            adaptor_names="dino_v3_7b,sam3",
            resolution_scale=1.0,
            sliding_window=False,
            tile_size=1024,
            tile_overlap=128,
            device="cpu",
            amp=False,
            skip_pca_stats=True,
            resume_partial=True,
            radio_thermal_pacing_seconds_per_image=0.0,
        )
    )
    manifest = json.loads((output / "frame_manifest.json").read_text())
    return output, image_dir, str(manifest["output_bundle_sha256"])


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


def test_every_raster_lift_uses_the_declared_pixel_normalization() -> None:
    feature_map = torch.tensor([[[[3.0]], [[4.0]]]])

    normalized = prepare_raster_view_features(
        feature_map, normalize_each_view=True
    )
    unchanged = prepare_raster_view_features(
        feature_map, normalize_each_view=False
    )

    torch.testing.assert_close(
        normalized[:, :, 0, 0], torch.tensor([[0.6, 0.8]])
    )
    torch.testing.assert_close(unchanged, feature_map)


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


def test_canonical_contract_resolves_before_mean_resultant_validation() -> None:
    args = Namespace(
        observation_contract="canonical-mpr-v1",
        aggregation_mode="center",
        raster_reliability_mode="mean_resultant",
        normalize_each_view=False,
        max_views=1,
        registration_weight_mode="uniform",
        raster_view_fusion="view_mean",
        depth_tolerance=1.0,
        relative_depth_tolerance=1.0,
        alpha_threshold=1.0,
        robust_mpr=True,
    )

    validate_raster_reliability_policy(args)

    assert args.aggregation_mode == "raster_gaussian_top1"
    assert args.normalize_each_view is True


def test_legacy_mean_resultant_still_requires_normalized_raster_inputs() -> None:
    args = Namespace(
        observation_contract="legacy",
        aggregation_mode="raster_gaussian_top1",
        raster_reliability_mode="mean_resultant",
        normalize_each_view=False,
    )

    with pytest.raises(ValueError, match="requires --normalize-each-view"):
        validate_raster_reliability_policy(args)


def test_marginal_responsibility_policy_is_exact_and_threshold_free() -> None:
    args = Namespace(
        observation_contract="legacy",
        aggregation_mode="raster_marginal_responsibility",
        raster_reliability_mode="mean_resultant",
        normalize_each_view=True,
        raster_view_fusion="contribution_mean",
        registration_weight_mode="alpha_depth",
        alpha_threshold=0.0,
        responsibility_cache="",
        save_responsibility_cache="",
    )

    validate_raster_reliability_policy(args)

    assert args.registration_weight_mode == (
        "exact_front_to_back_marginal_responsibility"
    )


def test_marginal_responsibility_policy_rejects_post_alpha_threshold() -> None:
    args = Namespace(
        observation_contract="legacy",
        aggregation_mode="raster_marginal_responsibility",
        raster_reliability_mode="legacy_valid",
        normalize_each_view=True,
        raster_view_fusion="contribution_mean",
        registration_weight_mode="alpha_depth",
        alpha_threshold=0.02,
        responsibility_cache="",
        save_responsibility_cache="",
    )

    with pytest.raises(ValueError, match="forbids post-compositor"):
        validate_raster_reliability_policy(args)


def test_exact_center_uncertainty_preserves_adjoint_target_policy() -> None:
    args = Namespace(
        observation_contract="legacy",
        aggregation_mode="raster_exact_center_uncertainty",
        raster_reliability_mode="mean_resultant",
        normalize_each_view=True,
        raster_view_fusion="contribution_mean",
        registration_weight_mode="alpha_depth",
        alpha_threshold=0.0,
        responsibility_cache="",
        save_responsibility_cache="",
    )

    validate_raster_reliability_policy(args)

    assert args.registration_weight_mode == "exact_front_to_back_adjoint_center"


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
    candidate_gaussian_ids = torch.tensor([0, 2, 0], dtype=torch.int32)
    candidate_pixel_ids = torch.tensor([1, 1, 5], dtype=torch.int32)
    candidate_weights = torch.tensor([0.25, 0.75, 0.1])
    producer_keep = select_top_raster_hits_per_gaussian(
        candidate_gaussian_ids,
        candidate_weights,
        n_gaussians=3,
    )
    path = tmp_path / "responsibility.pt"
    torch.save(
        {
            "schema_version": 1,
            "metadata": contract,
            "assignments": [
                {
                    # Gaussian-top-1 lifting permits distinct Gaussians to
                    # share a feature pixel.
                    "gaussian_ids": candidate_gaussian_ids[producer_keep],
                    "pixel_ids": candidate_pixel_ids[producer_keep],
                    "weights": candidate_weights[producer_keep],
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


def test_shared_responsibility_cache_accepts_producer_top1_float_ties(
    tmp_path,
) -> None:
    contract = {
        "selected_frame_indices": [3],
        "feature_height": 2,
        "feature_width": 3,
    }
    path = tmp_path / "responsibility.pt"
    torch.save(
        {
            "schema_version": 1,
            "metadata": contract,
            "assignments": [
                {
                    # This is the rare boundary case emitted by the producer's
                    # ``weight >= max_weight - 1e-8`` predicate.
                    "gaussian_ids": torch.tensor([0, 0, 1], dtype=torch.int32),
                    "pixel_ids": torch.tensor([1, 2, 2], dtype=torch.int32),
                    "weights": torch.tensor(
                        [0.25, 0.25 - 8e-9, 0.5], dtype=torch.float32
                    ),
                }
            ],
        },
        path,
    )

    assignments, _ = _load_responsibility_cache(
        path,
        expected_contract=contract,
        num_gaussians=2,
    )

    assert assignments[0]["gaussian_ids"].tolist() == [0, 0, 1]
    assert assignments[0]["pixel_ids"].tolist() == [1, 2, 2]


def test_shared_responsibility_cache_rejects_non_top1_duplicate_gaussian(
    tmp_path,
) -> None:
    contract = {
        "selected_frame_indices": [3],
        "feature_height": 2,
        "feature_width": 3,
    }
    path = tmp_path / "responsibility.pt"
    torch.save(
        {
            "schema_version": 1,
            "metadata": contract,
            "assignments": [
                {
                    "gaussian_ids": torch.tensor([0, 0], dtype=torch.int32),
                    "pixel_ids": torch.tensor([1, 2], dtype=torch.int32),
                    "weights": torch.tensor([0.25, 0.5], dtype=torch.float32),
                }
            ],
        },
        path,
    )

    with pytest.raises(ValueError, match="outside the top-1 tie tolerance"):
        _load_responsibility_cache(
            path,
            expected_contract=contract,
            num_gaussians=1,
        )


def test_shared_responsibility_cache_rejects_duplicate_gaussian_pixel_pair(
    tmp_path,
) -> None:
    contract = {
        "selected_frame_indices": [3],
        "feature_height": 2,
        "feature_width": 3,
    }
    path = tmp_path / "responsibility.pt"
    torch.save(
        {
            "schema_version": 1,
            "metadata": contract,
            "assignments": [
                {
                    "gaussian_ids": torch.tensor([0, 0], dtype=torch.int32),
                    "pixel_ids": torch.tensor([1, 1], dtype=torch.int32),
                    "weights": torch.tensor([0.5, 0.5], dtype=torch.float32),
                }
            ],
        },
        path,
    )

    with pytest.raises(ValueError, match="repeats Gaussian/pixel pairs"):
        _load_responsibility_cache(
            path,
            expected_contract=contract,
            num_gaussians=1,
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
    monkeypatch,
) -> None:
    """A direct SAM/DINO MPR must come from the official extractor output.

    This protects the high-spatial-fidelity route from silently falling back to
    an arbitrary tensor directory whose rows happen to have a compatible
    dimensionality.
    """

    feature_root, image_dir, bundle_sha256 = _strict_feature_bundle(
        tmp_path,
        monkeypatch,
        frame_id=0,
    )
    manifest_path = feature_root / "frame_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    checkpoint_sha256 = str(manifest["radio"]["checkpoint_sha256"])

    source = _resolve_extracted_capability_source(
        feature_root,
        "sam3",
        expected_scene="scene0001_00",
        expected_image_dir=image_dir,
        expected_frame_indices=[0],
        expected_output_bundle_sha256=bundle_sha256,
    )

    assert source["subdir"] == "sam3"
    assert source["adaptor_name"] == "sam3"
    assert source["output_dim"] == 1024
    assert source["native_grid"] == [1, 1]
    assert source["radio_checkpoint_sha256"] == checkpoint_sha256
    assert source["scene"] == "scene0001_00"

    with pytest.raises(ValueError, match="not bound"):
        _resolve_extracted_capability_source(
            feature_root,
            "sam3",
            expected_radio_checkpoint_sha256="another",
            expected_output_bundle_sha256=bundle_sha256,
        )
    with pytest.raises(ValueError, match="different scene"):
        _resolve_extracted_capability_source(
            feature_root,
            "sam3",
            expected_scene="scene0002_00",
            expected_output_bundle_sha256=bundle_sha256,
        )

    manifest["features"]["adaptors"] = manifest["features"]["adaptors"][:1]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest|signature"):
        _resolve_extracted_capability_source(
            feature_root,
            "sam3",
            expected_output_bundle_sha256=bundle_sha256,
        )


def test_extracted_capability_maps_are_selected_by_raw_frame_id_and_then_resampled(
    tmp_path,
    monkeypatch,
) -> None:
    feature_root, _image_dir, bundle_sha256 = _strict_feature_bundle(
        tmp_path,
        monkeypatch,
        frame_id=7,
    )
    pose_dir = tmp_path / "pose"
    pose_dir.mkdir()
    (pose_dir / "7.txt").write_text(
        "1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n"
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
        expected_output_bundle_sha256=bundle_sha256,
    )

    assert maps.shape == (1, 1024, 2, 3)
    assert maps.dtype == torch.float16
    torch.testing.assert_close(
        maps[0].float().norm(dim=0),
        torch.ones(2, 3),
        atol=5e-4,
        rtol=5e-4,
    )
    torch.testing.assert_close(
        (
            maps[0, 1, 0, 0].float()
            / maps[0, 2, 0, 0].float()
        ),
        torch.tensor(0.5),
        atol=5e-4,
        rtol=5e-4,
    )
    assert source["subdir"] == "sam3"


def test_registered_mean_finalization_is_row_chunked_and_exact() -> None:
    sums = torch.tensor(
        [[2.0, 4.0], [0.0, 0.0], [9.0, 3.0]],
        dtype=torch.float32,
    )
    counts = torch.tensor([2.0, 0.0, 3.0], dtype=torch.float32)

    features, valid = finalize_registered_mean_chunked(
        sums,
        counts,
        row_chunk_size=1,
    )

    assert features.dtype == torch.float16
    assert torch.equal(valid, torch.tensor([True, False, True]))
    torch.testing.assert_close(
        features.float(),
        torch.tensor([[1.0, 2.0], [0.0, 0.0], [3.0, 1.0]]),
    )


def test_capability_mpr_memory_estimate_accounts_for_half_maps_and_output() -> None:
    estimate = estimate_capability_mpr_cpu_bytes(
        num_views=120,
        channels=4096,
        height=60,
        width=80,
        num_gaussians=300_000,
        aggregation_mode="raster_gaussian_top1",
        raster_view_fusion="contribution_mean",
        raster_topk=3,
        raster_channel_chunk_size=256,
    )

    assert estimate["teacher_maps_float16"] == 120 * 4096 * 60 * 80 * 2
    assert estimate["registered_sum_float32"] == 300_000 * 4096 * 4
    assert estimate["final_features_float16"] == 300_000 * 4096 * 2
    assert estimate["estimated_peak_bytes"] == sum(
        value
        for key, value in estimate.items()
        if key != "estimated_peak_bytes"
    )
