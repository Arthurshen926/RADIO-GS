from copy import deepcopy
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
from radio_gs.querying.query_specific_propagation_cv import (
    ACTION_HASH256_DIFFUSION,
    ACTION_STRONG_UNARY,
    audit_signed_cv_population,
)
from radio_gs.scripts.build_nvos_signed_scribble_cv_selector import (
    ARTIFACT_KEYS,
    ARTIFACT_TYPE,
    CV_CONTRACT,
    METHOD_CONTRACT,
    REGISTRATION_SHA256,
    _json_sha256,
    _metrics_from_payload_tensors,
    validate_nvos_cv_selector_payload,
)
from radio_gs.scripts.eval_nvos_signed_scribble_cv_selector import (
    _cv_readout_contract,
    _load_cv_selector,
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
        height=20,
        width=30,
        num_gaussians=600,
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
    count = authority.num_gaussians
    rows = torch.arange(count)
    labels = rows % 2 == 0
    evidence = torch.where(labels, 0.5, -0.5).float()
    weight = torch.linspace(0.25, 2.0, count).float()
    valid = torch.ones(count, dtype=torch.bool)
    folds, reports = audit_signed_cv_population(rows, evidence, weight)
    query = torch.where(labels, 0.8, 0.2).float()
    positive = torch.zeros(count, dtype=torch.bool)
    negative = torch.zeros(count, dtype=torch.bool)
    positive[0] = True
    negative[1] = True
    primitive = query.clone()
    primitive[positive] = 1
    primitive[negative] = 0
    oof_unary = torch.where(labels, 0.9, 0.1).float()
    oof_diffusion = torch.where(labels, 0.6, 0.4).float()
    tensors = {
        "primitive_probability": primitive,
        "query_compatibility": query,
        "signed_reference_evidence": evidence,
        "reference_weight": weight,
        "capability_valid": valid,
        "positive_exclusive": positive,
        "negative_exclusive": negative,
        "conflict": torch.zeros(count, dtype=torch.bool),
        "cv_fold_ids": folds,
        "cv_oof_strong_unary": oof_unary,
        "cv_oof_hash256_diffusion": oof_diffusion,
    }
    tensors = {name: value.contiguous() for name, value in tensors.items()}
    digests = {name: tensor_sha256(value) for name, value in sorted(tensors.items())}
    payload = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "scene_id": authority.scene_id,
        "method_contract": METHOD_CONTRACT,
        "method_contract_sha256": _json_sha256(METHOD_CONTRACT),
        "cv_contract": CV_CONTRACT,
        "cv_contract_sha256": _json_sha256(CV_CONTRACT),
        "selected_action": ACTION_STRONG_UNARY,
        "cv_metrics": _metrics_from_payload_tensors(tensors),
        "cv_fold_reports": reports,
        "experiment_registration_path": "/registration.json",
        "experiment_registration_sha256": REGISTRATION_SHA256,
        "base_selector_path": "/base.pt",
        "base_selector_sha256": "d" * 64,
        "base_selector_method_contract_sha256": "e" * 64,
        "base_primitive_probability_sha256": "f" * 64,
        "base_query_compatibility_sha256": digests["query_compatibility"],
        "responsibility_report_path": "/responsibility.json",
        "responsibility_report_sha256": "1" * 64,
        "responsibility_file_sha256": "2" * 64,
        "responsibility_authority_sha256": authority.digest,
        "responsibility_tensor_bundle_sha256": "3" * 64,
        "capability_cache_path": "/capability.pt",
        "capability_sidecar_sha256": "4" * 64,
        "field_checkpoint_sha256": "5" * 64,
        "support_graph_path": "/graph.pt",
        "support_graph_sha256": "6" * 64,
        "knn_cache_path": "/knn.pt",
        "knn_cache_sha256": "7" * 64,
        "feature_hash_sha256": "8" * 64,
        "tensors": tensors,
        "tensor_sha256": digests,
        "tensor_bundle_sha256": _json_sha256(digests),
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
    }
    assert set(payload) == ARTIFACT_KEYS
    return payload


def test_cv_selector_contract_is_target_blind_and_only_selects_registered_actions():
    assert METHOD_CONTRACT["target_dependent_tuning"] is False
    assert METHOD_CONTRACT["connected_selection"] == "none"
    assert METHOD_CONTRACT["candidate_actions"] == [
        ACTION_STRONG_UNARY,
        ACTION_HASH256_DIFFUSION,
    ]
    assert METHOD_CONTRACT["heldout_evidence"].startswith("exact_zero")


def test_cv_selector_validator_recomputes_folds_metrics_and_decision():
    authority = _authority()
    payload = _payload(authority)
    primitive = validate_nvos_cv_selector_payload(
        payload,
        authority=authority,
        expected_responsibility_file_sha256="2" * 64,
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
        "cv_fold_ids",
        "cv_oof_strong_unary",
        "cv_oof_hash256_diffusion",
    ],
)
def test_cv_selector_validator_rejects_every_tensor_tamper(tensor_name):
    authority = _authority()
    payload = _payload(authority)
    value = payload["tensors"][tensor_name]
    if value.dtype == torch.bool:
        value[20] = ~value[20]
    elif value.dtype == torch.int64:
        value[20] = (value[20] + 1) % 3
    else:
        value[20] += 0.01
    with pytest.raises(ValueError):
        validate_nvos_cv_selector_payload(
            payload,
            authority=authority,
            expected_responsibility_file_sha256="2" * 64,
        )


@pytest.mark.parametrize(
    "tamper",
    [
        "schema",
        "registration",
        "method",
        "cv_contract",
        "target_flag",
        "selected_action",
        "metrics",
        "fold_report",
        "base_query_digest",
        "bundle",
    ],
)
def test_cv_selector_validator_fails_closed(tamper):
    authority = _authority()
    payload = _payload(authority)
    if tamper == "schema":
        payload["extra"] = False
    elif tamper == "registration":
        payload["experiment_registration_sha256"] = "9" * 64
    elif tamper == "method":
        payload["method_contract"] = {**METHOD_CONTRACT, "iterations": 99}
    elif tamper == "cv_contract":
        payload["cv_contract"] = {**CV_CONTRACT, "minimum_class_rows": 31}
    elif tamper == "target_flag":
        payload["target_mask_opened"] = True
    elif tamper == "selected_action":
        payload["selected_action"] = ACTION_HASH256_DIFFUSION
    elif tamper == "metrics":
        payload["cv_metrics"] = deepcopy(payload["cv_metrics"])
        payload["cv_metrics"][ACTION_STRONG_UNARY][
            "responsibility_balanced_log_loss"
        ] += 0.01
    elif tamper == "fold_report":
        payload["cv_fold_reports"] = deepcopy(payload["cv_fold_reports"])
        payload["cv_fold_reports"][0]["heldout_positive_rows"] += 1
    elif tamper == "base_query_digest":
        payload["base_query_compatibility_sha256"] = "9" * 64
    elif tamper == "bundle":
        payload["tensor_bundle_sha256"] = "9" * 64
    with pytest.raises(ValueError):
        validate_nvos_cv_selector_payload(
            payload,
            authority=authority,
            expected_responsibility_file_sha256="2" * 64,
        )


def test_cv_selector_rejects_selected_diffusion_that_differs_from_frozen_base():
    authority = _authority()
    payload = _payload(authority)
    labels = torch.arange(authority.num_gaussians) % 2 == 0
    payload["tensors"]["cv_oof_strong_unary"] = torch.full(
        (authority.num_gaussians,), 0.5
    )
    payload["tensors"]["cv_oof_hash256_diffusion"] = torch.where(
        labels, 0.9, 0.1
    ).float()
    payload["tensors"]["primitive_probability"] = torch.where(
        labels, 0.9, 0.1
    ).float()
    payload["tensors"]["primitive_probability"][
        payload["tensors"]["positive_exclusive"]
    ] = 1
    payload["tensors"]["primitive_probability"][
        payload["tensors"]["negative_exclusive"]
    ] = 0
    payload["selected_action"] = ACTION_HASH256_DIFFUSION
    payload["cv_metrics"] = _metrics_from_payload_tensors(payload["tensors"])
    digests = {
        name: tensor_sha256(value)
        for name, value in sorted(payload["tensors"].items())
    }
    payload["tensor_sha256"] = digests
    payload["tensor_bundle_sha256"] = _json_sha256(digests)
    payload["base_primitive_probability_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="selected diffusion"):
        validate_nvos_cv_selector_payload(
            payload,
            authority=authority,
            expected_responsibility_file_sha256="2" * 64,
        )


def test_repository_registration_hash_matches_cv_binding():
    import hashlib

    path = (
        Path(__file__).resolve().parents[1]
        / "paper"
        / "artifacts"
        / "nvos_signed_scribble_propagation_cv_registration_20260803.json"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == REGISTRATION_SHA256


def test_cv_evaluator_adapter_loads_only_frozen_validated_artifact(tmp_path):
    authority = _authority()
    payload = _payload(authority)
    path = tmp_path / "selector.pt"
    torch.save(payload, path)
    args = SimpleNamespace(
        completion=str(path),
        expected_completion_sha256=sha256_file(path),
        expected_primitive_sha256=payload["tensor_sha256"]["primitive_probability"],
    )
    primitive, frozen = _load_cv_selector(
        args,
        authority,
        expected_responsibility_file_sha256="2" * 64,
        expected_responsibility_tensor_bundle_sha256="3" * 64,
        expected_benchmark_manifest_sha256="0" * 64,
        expected_source_rgb_path=tmp_path / "unused.jpg",
    )
    assert frozen["selected_action"] == ACTION_STRONG_UNARY
    torch.testing.assert_close(primitive, payload["tensors"]["primitive_probability"])


def test_cv_evaluator_receipt_discloses_reference_only_selection():
    contract = _cv_readout_contract(_payload(_authority()))
    assert contract["selected_action"] == ACTION_STRONG_UNARY
    assert "reference_only" in contract["selection"]
    assert contract["target_used_for_selection"] is False
    assert contract["threshold"] == 0.5
    assert contract["connected_selection"] == "none"
