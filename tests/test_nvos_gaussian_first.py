import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from radio_gs.field import FeatureSpaceSignature
from radio_gs.querying.query_compilers import (
    _deterministic_prototypes,
    compile_registered_primitive_seeds,
)
from radio_gs.scripts.eval_nvos_gaussian_first import (
    _dataset_protocol_contract,
    _joint_signed_observation_seeds,
    _load_training_poses,
    _registered_solver_masses,
    _require_bipolar_solver_support,
    _render_registered_stage_maps,
    _resolve_observed_feature_path,
    _scaled_raster_shape,
    _valid_normalized_score_map,
    _weighted_spherical_prototypes,
)


def test_training_pose_loader_filters_target_view_overlap(tmp_path) -> None:
    pose = tmp_path / "pose.txt"
    np.savetxt(pose, np.eye(4, dtype=np.float32))
    (tmp_path / "train_frame_ids.json").write_text(json.dumps({"frame_ids": [7, 8]}))
    (tmp_path / "feature_pose_mapping.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "feature_frame_id": 7,
                        "camera_name": "target",
                        "pose_path": str(pose),
                    },
                    {
                        "feature_frame_id": 8,
                        "camera_name": "reference",
                        "pose_path": str(pose),
                    },
                ]
            }
        )
    )
    poses = _load_training_poses(tmp_path, ["target"])
    assert len(poses) == 1
    torch.testing.assert_close(poses[0], torch.eye(4))


def test_training_pose_loader_uses_resolved_camera_name_not_annotation_alias(tmp_path) -> None:
    pose = tmp_path / "pose.txt"
    np.savetxt(pose, np.eye(4, dtype=np.float32))
    (tmp_path / "train_frame_ids.json").write_text(json.dumps({"frame_ids": [1, 2]}))
    (tmp_path / "feature_pose_mapping.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "feature_frame_id": 1,
                        "camera_name": "IMG_4027",
                        "pose_path": str(pose),
                    },
                    {
                        "feature_frame_id": 2,
                        "camera_name": "IMG_4026",
                        "pose_path": str(pose),
                    },
                ]
            }
        )
    )

    # The annotation alias is image001, but the frozen protocol resolver maps
    # it to IMG_4027 before calling the loader.
    poses = _load_training_poses(tmp_path, ["IMG_4027"])

    assert len(poses) == 1


def test_observed_feature_path_uses_frozen_camera_mapping(tmp_path) -> None:
    feature_dir = tmp_path / "radio_features" / "backbone"
    feature_dir.mkdir(parents=True)
    feature_path = feature_dir / "rgb_7.pt"
    torch.save(torch.zeros(2, 3, 4), feature_path)
    (tmp_path / "feature_pose_mapping.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "feature_frame_id": 7,
                        "camera_name": "IMG_4027",
                        "colmap_camera_name": "IMG_4027",
                    }
                ]
            }
        )
    )
    (tmp_path / "radio_features" / "frame_manifest.json").write_text(
        json.dumps(
            {
                "frames": [
                    {"source_rank": 7, "frame_idx": 7, "saved_stem": "rgb_7"}
                ]
            }
        )
    )

    assert _resolve_observed_feature_path(tmp_path, "IMG_4027") == feature_path


def test_single_prototype_matches_weighted_mean() -> None:
    rows = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
    weights = torch.tensor([3.0, 1.0])
    actual = _weighted_spherical_prototypes(rows, weights, 1)
    expected = F.normalize(torch.tensor([3.0, 1.0]), dim=0)
    torch.testing.assert_close(actual[0], expected)


def test_multiple_prototypes_preserve_distinct_prompt_appearances() -> None:
    rows = F.normalize(
        torch.tensor([[1.0, 0.02], [1.0, -0.02], [0.02, 1.0], [-0.02, 1.0]]),
        dim=-1,
    )
    centers = _weighted_spherical_prototypes(rows, torch.ones(4), 2)
    similarities = rows @ centers.T
    assert torch.all(similarities.amax(dim=1) > 0.99)
    assert abs(float(centers[0] @ centers[1])) < 0.1


def test_sparse_registered_compiler_uses_continuous_primitive_seeds() -> None:
    signature = FeatureSpaceSignature(
        radio_version="c-radio_v4-h",
        radio_checkpoint_sha256="radio",
        raw_feature_dim=1280,
        adaptor_name="official",
        adaptor_output_dim=2,
        token_type="primitive",
    )
    query = compile_registered_primitive_seeds(
        torch.tensor([0.1, 0.8, 0.0]),
        torch.tensor([0.0, 0.0, 1.0]),
        appearance_features=torch.eye(3, 2),
        boundary_features=torch.eye(3, 2),
        appearance_signature=signature,
        boundary_signature=signature,
        prototype_count=1,
    )
    torch.testing.assert_close(
        query.positive_seeds.weights, torch.tensor([0.125, 1.0, 0.0])
    )
    assert query.negative_seeds is not None


def test_native_prompt_raster_shape_is_not_tied_to_feature_resolution() -> None:
    assert _scaled_raster_shape(756, 1008, 1.0) == (756, 1008)
    assert _scaled_raster_shape(756, 1008, 0.5) == (378, 504)


def test_joint_signed_registered_seeds_leave_conflicting_mass_neutral() -> None:
    positive, negative = _joint_signed_observation_seeds(
        torch.tensor([0.7, 0.0, -0.5, 0.0]),
        torch.tensor([0.9, 0.8, 0.7, 0.0]),
        support_threshold=0.0,
    )

    torch.testing.assert_close(positive, torch.tensor([0.7, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(negative, torch.tensor([0.0, 0.0, 0.5, 0.0]))


def test_historical_registered_seed_construction_preserves_positive_tie() -> None:
    positive, negative = _registered_solver_masses(
        torch.tensor([0.4]),
        torch.tensor([0.4]),
        support_threshold=0.0,
        construction="winner_take_all",
    )

    torch.testing.assert_close(positive, torch.tensor([0.4]))
    torch.testing.assert_close(negative, torch.tensor([0.0]))


def test_capability_filter_must_preserve_both_prompt_signs() -> None:
    with pytest.raises(RuntimeError, match="Capability-valid.*neg=0"):
        _require_bipolar_solver_support(
            torch.tensor([0.5, 0.0]),
            torch.zeros(2),
            label="Capability-valid",
        )


def test_registered_stage_renderer_reuses_only_the_actual_final_stage() -> None:
    values = {
        "unary_prior": torch.tensor([1.0]),
        "propagated": torch.tensor([2.0]),
        "connected": torch.tensor([3.0]),
    }
    rendered = _render_registered_stage_maps(
        values,
        final_stage="propagated",
        final_rendered=np.array([20.0], dtype=np.float32),
        render=lambda tensor: np.array([float(tensor.item() * 10.0)]),
    )

    np.testing.assert_array_equal(rendered["unary_prior"], np.array([10.0]))
    np.testing.assert_array_equal(rendered["propagated"], np.array([20.0]))
    np.testing.assert_array_equal(rendered["connected"], np.array([30.0]))


def test_dataset_protocol_contract_excludes_method_score_semantics() -> None:
    manifest = {
        "benchmark": "nvos",
        "protocol": {
            "cohort": ["scene"],
            "dataset_version": "v1",
            "task": "segmentation",
            "prompt_type": "fixed_scribble",
            "prompt_support": "complete",
            "prompt_asset_sha256": {
                "scene": {"positive": "p", "negative": "n"}
            },
            "prediction_representation": "continuous_margin",
            "score_semantics": "cosine_margin",
            "threshold": {"value": 0.0},
        },
        "scenes": [
            {
                "scene_id": "scene",
                "prompt": {
                    "type": "positive_negative_scribbles",
                    "frame_id": "prompt",
                },
                "prompt_frame_ids": ["prompt"],
                "calibration_frame_ids": [],
                "evaluation_frame_ids": ["target"],
                "excluded_training_frame_ids": ["target"],
                "training_frames": [{"frame_id": "prompt"}],
                "target_rgb_policy": "forbidden",
                "frames": [
                    {
                        "frame_id": "target",
                        "ground_truth_sha256": "ground-truth",
                    }
                ],
            }
        ],
    }

    original = _dataset_protocol_contract(manifest)
    manifest["protocol"]["prediction_representation"] = "posterior"
    manifest["protocol"]["score_semantics"] = "foreground_probability"
    manifest["protocol"]["threshold"] = {"value": 0.5}

    assert _dataset_protocol_contract(manifest) == original
    manifest["protocol"]["prompt_asset_sha256"]["scene"]["positive"] = "changed"
    assert _dataset_protocol_contract(manifest) != original


def test_valid_normalized_score_map_uses_only_supported_compositing_mass() -> None:
    rendered = torch.tensor(
        [
            [[0.20, 0.00], [0.45, 0.10]],
            [[0.25, 0.00], [0.50, 0.20]],
        ]
    )
    actual = _valid_normalized_score_map(rendered)
    torch.testing.assert_close(
        actual,
        torch.tensor([[0.8, 0.0], [0.9, 0.5]]),
    )


def test_valid_normalized_score_map_interpolates_to_total_alpha_score() -> None:
    rendered = torch.tensor(
        [
            [[0.20, 0.45]],
            [[0.25, 0.50]],
        ]
    )

    total_alpha = _valid_normalized_score_map(rendered, coverage_power=1.0)

    torch.testing.assert_close(total_alpha, rendered[0])


def test_sparse_prototypes_match_prefiltered_reference_for_half_bank() -> None:
    features = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=torch.float16,
        ).float(),
        dim=-1,
    ).half()
    weights = torch.tensor([0.0, 0.7, 0.0, 0.2, 0.0])

    actual_features, actual_masses = _deterministic_prototypes(
        features, weights, count=2, chunk_size=1
    )
    active = weights > 0
    expected_features, expected_masses = _deterministic_prototypes(
        features[active].float(), weights[active], count=2
    )

    torch.testing.assert_close(actual_features, expected_features)
    torch.testing.assert_close(actual_masses, expected_masses)


def test_spherical_mean_fps_anchors_with_weighted_mean() -> None:
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    weights = torch.tensor([0.5, 0.4, 0.1])
    prototypes, masses = _deterministic_prototypes(
        features,
        weights,
        count=2,
        chunk_size=1,
        strategy="spherical_mean_fps",
    )
    expected_mean = F.normalize(
        (F.normalize(features, dim=-1) * weights[:, None]).sum(dim=0), dim=0
    )
    torch.testing.assert_close(prototypes[0], expected_mean)
    torch.testing.assert_close(masses.sum(), torch.tensor(1.0))
