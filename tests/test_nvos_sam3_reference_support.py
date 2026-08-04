from pathlib import Path

import pytest
import torch

from radio_gs.interfaces.prompt_responsibility_cache import (
    COMPOSITOR_CONTRACT,
    PromptResponsibilityAuthority,
    tensor_sha256,
)
from radio_gs.scripts.build_nvos_sam3_reference_support import (
    ARTIFACT_KEYS,
    ARTIFACT_TYPE,
    METHOD_CONTRACT,
    REGISTRATION_SHA256,
    _json_sha256,
    validate_nvos_sam3_support_payload,
)


SHA = "a" * 64


def _authority() -> PromptResponsibilityAuthority:
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
        source_sha256={
            "positive_scribble": "b" * 64,
            "negative_scribble": "c" * 64,
        },
    )


def _payload(authority: PromptResponsibilityAuthority) -> dict:
    tensors = {
        "primitive_probability": torch.tensor([1.0, 0.0, 0.7, 0.0]),
        "query_compatibility": torch.tensor([0.8, 0.2, 0.7, 0.0]),
        "signed_reference_evidence": torch.tensor([0.3, -0.2, 0.0, 0.0]),
        "reference_weight": torch.tensor([2.0, 3.0, 1.0, 0.0]),
        "completed_positive_mass": torch.tensor([2.0, 0.0, 1.0, 0.0]),
        "raw_negative_mass": torch.tensor([0.0, 3.0, 0.0, 0.0]),
        "capability_valid": torch.tensor([True, True, True, False]),
        "positive_exclusive": torch.tensor([True, False, False, False]),
        "negative_exclusive": torch.tensor([False, True, False, False]),
        "conflict": torch.tensor([False, False, False, True]),
    }
    tensors = {name: value.contiguous() for name, value in tensors.items()}
    digests = {name: tensor_sha256(value) for name, value in sorted(tensors.items())}
    payload = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "scene_id": authority.scene_id,
        "method_contract": METHOD_CONTRACT,
        "method_contract_sha256": _json_sha256(METHOD_CONTRACT),
        "experiment_registration_path": "/registration.json",
        "experiment_registration_sha256": REGISTRATION_SHA256,
        "responsibility_report_path": "/responsibility.json",
        "responsibility_report_sha256": "d" * 64,
        "responsibility_file_sha256": "e" * 64,
        "responsibility_authority_sha256": authority.digest,
        "responsibility_tensor_bundle_sha256": "f" * 64,
        "reference_completion_path": "/completion.pt",
        "reference_completion_sha256": "1" * 64,
        "reference_completion_receipt_path": "/completion.json",
        "reference_completion_receipt_sha256": "2" * 64,
        "reference_completion_tensor_bundle_sha256": "3" * 64,
        "source_rgb_path": "/source.jpg",
        "source_rgb_sha256": "4" * 64,
        "completed_positive_sha256": "5" * 64,
        "raw_positive_sha256": "6" * 64,
        "raw_negative_sha256": "7" * 64,
        "capability_cache_path": "/capability.pt",
        "capability_sidecar_sha256": "8" * 64,
        "field_checkpoint_sha256": "9" * 64,
        "support_graph_path": "/graph.pt",
        "support_graph_sha256": "0" * 64,
        "knn_cache_path": "/knn.pt",
        "knn_cache_sha256": "a" * 64,
        "feature_hash_sha256": "b" * 64,
        "tensors": tensors,
        "tensor_sha256": digests,
        "tensor_bundle_sha256": _json_sha256(digests),
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
    }
    assert set(payload) == ARTIFACT_KEYS
    return payload


def test_sam3_reference_support_contract_is_source_only_and_target_blind():
    assert METHOD_CONTRACT["track"] == "source_reference_RGB_official_SAM3_completion"
    assert "raw_official_negative" in METHOD_CONTRACT["negative_observation"]
    assert METHOD_CONTRACT["target_dependent_tuning"] is False
    assert METHOD_CONTRACT["connected_selection"] == "none"


def test_sam3_reference_support_validator_accepts_bound_payload():
    authority = _authority()
    payload = _payload(authority)
    primitive = validate_nvos_sam3_support_payload(
        payload,
        authority=authority,
        expected_responsibility_file_sha256="e" * 64,
        expected_completion_sha256="1" * 64,
        expected_primitive_sha256=payload["tensor_sha256"]["primitive_probability"],
    )
    torch.testing.assert_close(primitive, payload["tensors"]["primitive_probability"])


@pytest.mark.parametrize(
    "tamper",
    ["schema", "registration", "completion", "target", "anchor", "tensor", "bundle"],
)
def test_sam3_reference_support_validator_fails_closed(tamper):
    authority = _authority()
    payload = _payload(authority)
    if tamper == "schema":
        payload["extra"] = False
    elif tamper == "registration":
        payload["experiment_registration_sha256"] = "c" * 64
    elif tamper == "completion":
        payload["reference_completion_sha256"] = "d" * 64
    elif tamper == "target":
        payload["target_mask_opened"] = True
    elif tamper == "anchor":
        payload["tensors"]["primitive_probability"][0] = 0.9
    elif tamper == "tensor":
        payload["tensors"]["completed_positive_mass"][2] += 0.1
    elif tamper == "bundle":
        payload["tensor_bundle_sha256"] = "d" * 64
    with pytest.raises(ValueError):
        validate_nvos_sam3_support_payload(
            payload,
            authority=authority,
            expected_responsibility_file_sha256="e" * 64,
            expected_completion_sha256="1" * 64,
        )


def test_repository_registration_hash_matches_sam3_support_binding():
    import hashlib

    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "paper"
        / "artifacts"
        / "nvos_official_sam3_reference_completion_registration_20260803.json"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == REGISTRATION_SHA256
