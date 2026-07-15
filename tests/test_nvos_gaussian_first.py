import json

import numpy as np
import torch
import torch.nn.functional as F

from radio_gs.field import FeatureSpaceSignature
from radio_gs.querying.query_compilers import (
    _deterministic_prototypes,
    compile_registered_primitive_seeds,
)
from radio_gs.scripts.eval_nvos_gaussian_first import (
    _load_training_poses,
    _resolve_observed_feature_path,
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
