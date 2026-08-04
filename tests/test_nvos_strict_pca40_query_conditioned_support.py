import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from radio_gs.interfaces.prompt_responsibility_cache import (
    COMPOSITOR_CONTRACT,
    PromptResponsibilityAuthority,
    sha256_file,
    tensor_sha256,
)
from radio_gs.scripts.build_nvos_strict_pca40_query_conditioned_support import (
    ARTIFACT_KEYS,
    ARTIFACT_TYPE,
    METHOD_CONTRACT,
    REGISTRATION_SHA256,
    _json_sha256,
    validate_nvos_strict_pca40_support_payload,
)
from radio_gs.scripts.eval_nvos_strict_pca40_query_conditioned_support import (
    _load_strict_pca40_selector,
    _strict_pca40_readout_contract,
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
        "capability_valid": torch.tensor([True, True, True, False]),
        "positive_exclusive": torch.tensor([True, False, False, False]),
        "negative_exclusive": torch.tensor([False, True, False, False]),
        "conflict": torch.tensor([False, False, False, True]),
    }
    digests = {
        name: tensor_sha256(value) for name, value in sorted(tensors.items())
    }
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
        "relation_cache_path": "/relation.pt",
        "relation_cache_sha256": "1" * 64,
        "relation_cache_receipt_sha256": "2" * 64,
        "relation_feature_sha256": "3" * 64,
        "source_feature_sha256": "4" * 64,
        "source_xyz_sha256": "5" * 64,
        "capability_sidecar_sha256": "6" * 64,
        "field_checkpoint_sha256": "7" * 64,
        "support_graph_path": "/graph.pt",
        "support_graph_sha256": "8" * 64,
        "knn_cache_path": "/knn.pt",
        "knn_cache_sha256": "9" * 64,
        "tensors": tensors,
        "tensor_sha256": digests,
        "tensor_bundle_sha256": _json_sha256(digests),
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
    }
    assert set(payload) == ARTIFACT_KEYS
    return payload


def test_pca40_method_contract_is_fixed_and_discloses_non_native_boundary():
    assert METHOD_CONTRACT["track"] == "strict_raw_positive_negative_scribble"
    assert "PCA40" in METHOD_CONTRACT["relation_feature"]
    assert "not_native_DINOv2" in METHOD_CONTRACT["relation_feature_role"]
    assert METHOD_CONTRACT["feature_bandwidth"] == 2.0
    assert METHOD_CONTRACT["regularizer_bandwidth"] == 4.0
    assert METHOD_CONTRACT["threshold"] == 0.5
    assert METHOD_CONTRACT["target_dependent_tuning"] is False


def test_pca40_support_validator_accepts_frozen_payload():
    authority = _authority()
    payload = _payload(authority)
    primitive = validate_nvos_strict_pca40_support_payload(
        payload,
        authority=authority,
        expected_responsibility_file_sha256="e" * 64,
        expected_primitive_sha256=payload["tensor_sha256"]["primitive_probability"],
    )
    torch.testing.assert_close(primitive, payload["tensors"]["primitive_probability"])


@pytest.mark.parametrize(
    "tensor_name",
    [
        "primitive_probability",
        "query_compatibility",
        "signed_reference_evidence",
        "reference_weight",
        "capability_valid",
        "positive_exclusive",
        "negative_exclusive",
        "conflict",
    ],
)
def test_pca40_support_validator_rejects_every_tensor_tamper(tensor_name):
    authority = _authority()
    payload = _payload(authority)
    value = payload["tensors"][tensor_name]
    if value.dtype == torch.bool:
        value[2] = ~value[2]
    else:
        value[2] += 0.01
    with pytest.raises(ValueError):
        validate_nvos_strict_pca40_support_payload(
            payload,
            authority=authority,
            expected_responsibility_file_sha256="e" * 64,
        )


@pytest.mark.parametrize(
    "tamper",
    ["schema", "registration", "method", "target", "anchor", "partition", "bundle"],
)
def test_pca40_support_validator_fails_closed(tamper):
    authority = _authority()
    payload = _payload(authority)
    if tamper == "schema":
        payload["extra"] = False
    elif tamper == "registration":
        payload["experiment_registration_sha256"] = SHA
    elif tamper == "method":
        payload["method_contract"] = {**METHOD_CONTRACT, "feature_bandwidth": 4.0}
    elif tamper == "target":
        payload["target_metric_computed"] = True
    elif tamper == "anchor":
        payload["tensors"]["primitive_probability"][0] = 0.9
    elif tamper == "partition":
        payload["tensors"]["negative_exclusive"][0] = True
    elif tamper == "bundle":
        payload["tensor_bundle_sha256"] = SHA
    with pytest.raises(ValueError):
        validate_nvos_strict_pca40_support_payload(
            payload,
            authority=authority,
            expected_responsibility_file_sha256="e" * 64,
        )


def test_nvos_pca40_registration_hash_is_frozen():
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "paper"
        / "artifacts"
        / "nvos_strict_cradio_dino_pca40_experiment_registration_20260803.json"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == REGISTRATION_SHA256


def test_pca40_evaluator_loads_only_hash_bound_selector(tmp_path):
    authority = _authority()
    payload = _payload(authority)
    path = tmp_path / "selector.pt"
    torch.save(payload, path)
    args = SimpleNamespace(
        completion=str(path),
        expected_completion_sha256=sha256_file(path),
        expected_primitive_sha256=payload["tensor_sha256"]["primitive_probability"],
    )
    primitive, frozen = _load_strict_pca40_selector(
        args,
        authority,
        expected_responsibility_file_sha256="e" * 64,
        expected_responsibility_tensor_bundle_sha256="f" * 64,
        expected_benchmark_manifest_sha256="0" * 64,
        expected_source_rgb_path=tmp_path / "unused.jpg",
    )
    assert frozen["artifact_type"] == ARTIFACT_TYPE
    torch.testing.assert_close(primitive, payload["tensors"]["primitive_probability"])


def test_pca40_readout_contract_discloses_non_native_relation():
    contract = _strict_pca40_readout_contract(_payload(_authority()))
    assert "PCA40" in contract["graph"]
    assert "not_native_LUDVIG_DINOv2" in contract["compatibility_boundary"]
    assert contract["threshold"] == 0.5
    assert contract["connected_selection"] == "none"
