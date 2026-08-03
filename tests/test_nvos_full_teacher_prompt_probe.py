from pathlib import Path
from copy import deepcopy

import pytest
import torch

from radio_gs.interfaces.prompt_responsibility_cache import (
    COMPOSITOR_CONTRACT,
    PromptResponsibilityAuthority,
    sha256_file,
    tensor_sha256,
)
from radio_gs.scripts.analyze_nvos_full_teacher_prompt_adjoint_cycle import (
    FULL_TEACHER_ARTIFACT_KEYS,
    METHOD_CONTRACT,
    OFFICIAL_RADIO_CHECKPOINT_SHA256,
    _json_sha256,
    _validate_reference_rgb_authority,
    validate_full_teacher_completion_payload,
)


SHA = "a" * 64


def _authority(manifest_sha256: str) -> PromptResponsibilityAuthority:
    return PromptResponsibilityAuthority(
        scene_id="fern",
        frame_id="reference",
        camera_name="reference",
        colmap_camera_name="reference",
        geometry_checkpoint_sha256=SHA,
        geometry_xyz_sha256=SHA,
        pose_sha256=SHA,
        intrinsics_sha256=SHA,
        height=2,
        width=3,
        num_gaussians=4,
        alpha_threshold=0.0,
        compositor_contract=COMPOSITOR_CONTRACT,
        target_rgb_opened=False,
        target_mask_opened=False,
        source_sha256={"benchmark_manifest": manifest_sha256},
    )


def _manifest(source: Path, target: Path) -> dict:
    return {
        "protocol": {
            "benchmark": "NVOS",
            "prompt_type": "fixed_positive_negative_scribbles",
            "target_rgb_at_query": "forbidden",
            "target_rgb_during_field_training": "forbidden",
            "target_mask_use": "scoring_only",
        },
        "scenes": [
            {
                "scene_id": "fern",
                "prompt": {
                    "frame_id": "reference",
                    "type": "positive_negative_scribbles",
                },
                "prompt_frame_ids": ["reference"],
                "evaluation_frame_ids": ["target"],
                "excluded_training_frame_ids": ["target"],
                "target_rgb_policy": "excluded_from_field_training_and_query",
                "frames": [
                    {
                        "frame_id": "reference",
                        "camera_name": "reference",
                        "rgb_path": str(source),
                        "ground_truth": None,
                        "gt_mask_path": None,
                    },
                    {
                        "frame_id": "target",
                        "camera_name": "target",
                        "rgb_path": str(target),
                        "ground_truth": "target-mask.png",
                        "gt_mask_path": "target-mask.png",
                    },
                ],
            }
        ],
    }


def _full_teacher_payload(authority, source):
    tensors = {
        "low_posterior": torch.full((1, 1), 0.5),
        "dense_native": torch.full((2, 3), 0.5),
        "anchored_native": torch.full((2, 3), 0.5),
        "primitive_probability": torch.full((4,), 0.5),
        "primitive_visible": torch.ones(4, dtype=torch.bool),
        "cycle_native": torch.full((2, 3), 0.5),
        "cycle_supported": torch.ones((2, 3), dtype=torch.bool),
    }
    digests = {name: tensor_sha256(value) for name, value in sorted(tensors.items())}
    payload = {
        "schema_version": 1,
        "artifact_type": "nvos_full_teacher_dino_dense_prompt_exact_adjoint_cycle",
        "scene_id": "fern",
        "method_contract": METHOD_CONTRACT,
        "method_contract_sha256": _json_sha256(METHOD_CONTRACT),
        "responsibility_authority_sha256": authority.digest,
        "responsibility_file_sha256": "b" * 64,
        "prompt_feature_sha256": "c" * 64,
        "radio_checkpoint_sha256": OFFICIAL_RADIO_CHECKPOINT_SHA256,
        "benchmark_manifest_sha256": SHA,
        "source_rgb_path": str(source.resolve()),
        "source_rgb_sha256": sha256_file(source),
        "source_rgb_shape": [2, 3],
        "source_frame_id": "reference",
        "source_feature_grid_shape": [1, 1],
        "radio_source_tree_sha256": "d" * 64,
        "tensors": tensors,
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "tensor_sha256": digests,
        "tensor_bundle_sha256": _json_sha256(digests),
    }
    assert set(payload) == FULL_TEACHER_ARTIFACT_KEYS
    return payload


def test_full_teacher_contract_changes_only_prompt_feature_source():
    assert METHOD_CONTRACT["prompt_feature_source"] == (
        "frozen_reference_rgb_full_2d_teacher"
    )
    assert METHOD_CONTRACT["dense_rule"] == (
        "class_count_normalized_all_scribble_logmeanexp_cosine"
    )
    assert METHOD_CONTRACT["adjoint"] == "u=(W.T@y)/(W.T@1)_visible"
    assert METHOD_CONTRACT["graph"] == "none"
    assert METHOD_CONTRACT["target_rgb"] == "never_opened"


def test_reference_rgb_authority_accepts_only_bound_non_target_source(tmp_path):
    source = tmp_path / "reference.jpg"
    target = tmp_path / "target.jpg"
    source.write_bytes(b"source-rgb")
    target.write_bytes(b"target-rgb")
    manifest = _manifest(source, target)
    record = _validate_reference_rgb_authority(
        manifest=manifest,
        manifest_sha256=SHA,
        authority=_authority(SHA),
        source_rgb=source,
        expected_source_rgb_sha256=sha256_file(source),
    )
    assert record["reference_rgb_protocol_permitted"] is True
    assert record["source_rgb_sha256"] == sha256_file(source)


def test_full_teacher_completion_validator_binds_source_and_every_tensor(tmp_path):
    source = tmp_path / "reference.jpg"
    source.write_bytes(b"source-rgb")
    authority = _authority(SHA)
    payload = _full_teacher_payload(authority, source)
    tensors = validate_full_teacher_completion_payload(
        payload,
        authority=authority,
        expected_responsibility_file_sha256="b" * 64,
        expected_primitive_sha256=payload["tensor_sha256"]["primitive_probability"],
        expected_source_rgb_path=source,
        expected_benchmark_manifest_sha256=SHA,
    )
    assert tensors["primitive_probability"].shape == (4,)


@pytest.mark.parametrize(
    "tamper",
    ["extra_key", "contract", "source", "source_hash", "grid", "primitive", "bundle"],
)
def test_full_teacher_completion_validator_fails_closed(tmp_path, tamper):
    source = tmp_path / "reference.jpg"
    source.write_bytes(b"source-rgb")
    authority = _authority(SHA)
    payload = _full_teacher_payload(authority, source)
    expected_primitive = payload["tensor_sha256"]["primitive_probability"]
    if tamper == "extra_key":
        payload["unexpected"] = False
    elif tamper == "contract":
        payload["method_contract"] = {**METHOD_CONTRACT, "temperature": 0.08}
    elif tamper == "source":
        payload["source_rgb_path"] = str(tmp_path / "other.jpg")
    elif tamper == "source_hash":
        payload["source_rgb_sha256"] = "e" * 64
    elif tamper == "grid":
        payload["source_feature_grid_shape"] = [1, 2]
    elif tamper == "primitive":
        payload["tensors"]["primitive_probability"][0] = 0.25
    elif tamper == "bundle":
        payload["tensor_bundle_sha256"] = "e" * 64
    with pytest.raises(ValueError):
        validate_full_teacher_completion_payload(
            payload,
            authority=authority,
            expected_responsibility_file_sha256="b" * 64,
            expected_primitive_sha256=expected_primitive,
            expected_source_rgb_path=source,
            expected_benchmark_manifest_sha256=SHA,
        )


@pytest.mark.parametrize("tamper", ["path", "hash", "target_role", "target_policy"])
def test_reference_rgb_authority_fails_closed(tmp_path, tamper):
    source = tmp_path / "reference.jpg"
    target = tmp_path / "target.jpg"
    other = tmp_path / "other.jpg"
    source.write_bytes(b"source-rgb")
    target.write_bytes(b"target-rgb")
    other.write_bytes(b"other-rgb")
    manifest = _manifest(source, target)
    candidate = source
    expected = sha256_file(source)
    if tamper == "path":
        candidate = other
    elif tamper == "hash":
        expected = "f" * 64
    elif tamper == "target_role":
        manifest["scenes"][0]["prompt"]["frame_id"] = "target"
    elif tamper == "target_policy":
        manifest["scenes"][0]["target_rgb_policy"] = "allowed"
    with pytest.raises(ValueError):
        _validate_reference_rgb_authority(
            manifest=manifest,
            manifest_sha256=SHA,
            authority=_authority(SHA),
            source_rgb=candidate,
            expected_source_rgb_sha256=expected,
        )
