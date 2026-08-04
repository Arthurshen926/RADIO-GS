from pathlib import Path

import numpy as np
import pytest
import torch

from radio_gs.interfaces.prompt_responsibility_cache import (
    COMPOSITOR_CONTRACT,
    PromptResponsibilityAuthority,
    tensor_sha256,
)
from radio_gs.querying.sam3_reference_completion import (
    entropy_reliability_soft_observation,
)
from radio_gs.scripts.build_nvos_sam3_raw_semantic_gated_support import (
    ARTIFACT_KEYS,
    ARTIFACT_TYPE,
    METHOD_CONTRACT,
    REGISTRATION_SHA256,
    _json_sha256,
    form_semantic_completed_initial_unary,
    validate_nvos_sam3_raw_semantic_gated_payload,
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
    q = np.asarray([[0.0, 0.1, 0.5], [0.9, 1.0, 0.2]], dtype=np.float32)
    raw_positive = np.asarray([[0, 0, 1], [0, 0, 0]], dtype=bool)
    raw_negative = np.asarray([[0, 0, 0], [0, 1, 0]], dtype=bool)
    reliability, observation = entropy_reliability_soft_observation(
        q, raw_positive, raw_negative
    )
    pixel_tensors = {
        "aggregate_probability": torch.from_numpy(q),
        "binary_entropy_reliability": torch.from_numpy(reliability),
        "soft_positive_observation": torch.from_numpy(observation),
        "raw_positive": torch.from_numpy(raw_positive),
        "raw_negative": torch.from_numpy(raw_negative),
    }
    pixel_digests = {
        name: tensor_sha256(value) for name, value in sorted(pixel_tensors.items())
    }
    completed_probability = torch.tensor([0.4, 0.6, 0.8, 0.2])
    compatibility = torch.tensor([0.9, 0.2, 0.75, 0.0])
    valid = torch.tensor([True, True, True, False])
    positive = torch.tensor([True, False, False, False])
    negative = torch.tensor([False, True, False, False])
    conflict = torch.tensor([False, False, False, True])
    initial = form_semantic_completed_initial_unary(
        completed_probability, compatibility, valid, positive, negative
    )
    raw_positive_mass = torch.tensor([2.0, 0.0, 0.0, 1.0])
    raw_negative_mass = torch.tensor([0.0, 3.0, 0.0, 1.0])
    tensors = {
        "primitive_probability": torch.tensor([1.0, 0.0, 0.7, 0.0]),
        "raw_query_compatibility": compatibility,
        "raw_signed_reference_evidence": torch.tensor([0.5, -0.5, 0.0, 0.0]),
        "raw_reference_weight": raw_positive_mass + raw_negative_mass,
        "completed_positive_probability": completed_probability,
        "semantic_completed_initial_unary": initial,
        "completed_positive_mass": torch.tensor([1.6, 3.6, 4.0, 0.8]),
        "raw_positive_mass": raw_positive_mass,
        "raw_negative_mass": raw_negative_mass,
        "visible_mass": torch.tensor([4.0, 6.0, 5.0, 4.0]),
        "capability_valid": valid,
        "positive_exclusive": positive,
        "negative_exclusive": negative,
        "conflict": conflict,
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
        "capability_cache_path": "/capability.pt",
        "capability_sidecar_sha256": "5" * 64,
        "field_checkpoint_sha256": "6" * 64,
        "support_graph_path": "/graph.pt",
        "support_graph_sha256": "7" * 64,
        "knn_cache_path": "/knn.pt",
        "knn_cache_sha256": "8" * 64,
        "feature_hash_sha256": "9" * 64,
        "pixel_tensors": pixel_tensors,
        "pixel_tensor_sha256": pixel_digests,
        "pixel_tensor_bundle_sha256": _json_sha256(pixel_digests),
        "tensors": tensors,
        "tensor_sha256": digests,
        "tensor_bundle_sha256": _json_sha256(digests),
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
    }
    assert set(payload) == ARTIFACT_KEYS
    return payload


def test_semantic_initial_unary_is_parameter_free_product_with_hard_anchors():
    output = form_semantic_completed_initial_unary(
        torch.tensor([0.5, 0.8, 0.4, 0.9]),
        torch.tensor([0.2, 0.7, 0.5, 0.9]),
        torch.tensor([True, True, True, False]),
        torch.tensor([True, False, False, False]),
        torch.tensor([False, True, False, False]),
    )
    torch.testing.assert_close(output, torch.tensor([1.0, 0.0, 0.2, 0.0]))


def test_raw_semantic_contract_forbids_completed_refit_and_new_scalar():
    assert METHOD_CONTRACT["compatibility_refit_from_completed_evidence"] == "forbidden"
    assert METHOD_CONTRACT["new_scalar"] == "forbidden"
    assert METHOD_CONTRACT["query_edge_gate"] == "sqrt(P_raw_i_times_P_raw_j)"
    assert METHOD_CONTRACT["target_dependent_tuning"] is False
    assert METHOD_CONTRACT["connected_selection"] == "none"


def test_raw_semantic_validator_recomputes_primitive_product():
    authority = _authority()
    payload = _payload(authority)
    primitive = validate_nvos_sam3_raw_semantic_gated_payload(
        payload,
        authority=authority,
        expected_responsibility_file_sha256="e" * 64,
        expected_completion_sha256="1" * 64,
        expected_primitive_sha256=payload["tensor_sha256"]["primitive_probability"],
    )
    torch.testing.assert_close(primitive, payload["tensors"]["primitive_probability"])


@pytest.mark.parametrize(
    "tamper",
    [
        "initial_formula", "raw_weight", "completed_probability", "primitive",
        "registration", "target", "schema",
    ],
)
def test_raw_semantic_validator_fails_closed(tamper):
    authority = _authority()
    payload = _payload(authority)
    if tamper == "initial_formula":
        payload["tensors"]["semantic_completed_initial_unary"][2] += 0.01
    elif tamper == "raw_weight":
        payload["tensors"]["raw_reference_weight"][2] += 0.01
    elif tamper == "completed_probability":
        payload["tensors"]["completed_positive_probability"][2] += 0.01
    elif tamper == "primitive":
        payload["tensors"]["primitive_probability"][2] += 0.01
    elif tamper == "registration":
        payload["experiment_registration_sha256"] = "0" * 64
    elif tamper == "target":
        payload["target_metric_computed"] = True
    elif tamper == "schema":
        payload["extra"] = False
    with pytest.raises(ValueError):
        validate_nvos_sam3_raw_semantic_gated_payload(
            payload,
            authority=authority,
            expected_responsibility_file_sha256="e" * 64,
            expected_completion_sha256="1" * 64,
        )


def test_repository_registration_hash_matches_raw_semantic_binding():
    import hashlib

    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "paper"
        / "artifacts"
        / "nvos_sam3_raw_semantic_gated_followup_registration_20260803.json"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == REGISTRATION_SHA256
