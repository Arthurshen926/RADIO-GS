from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from radio_gs.scripts.materialize_registered_evidence_external_prediction import (
    _write_supported_calibration,
)

from radio_gs.interfaces.capability_cache import CanonicalCapabilityBank
from radio_gs.interfaces.factorized_primitive_state import FactorizedPrimitiveState
from radio_gs.interfaces.prompt_responsibility_cache import (
    PromptResponsibilityAuthority,
    build_prompt_responsibility_cache,
)
from radio_gs.querying.global_prompt_logit_calibrator import (
    GlobalPromptLogitCalibratorV2,
)
from radio_gs.querying.registered_evidence_external_adapter import (
    EXECUTION_AUTHORITY_SCHEMA,
    RENDERED_PROBABILITY_ROUNDOFF_TOLERANCE,
    build_external_registered_features,
    calibrate_rendered_supported_probability,
    infer_registered_primitive_unary,
    render_then_calibrate,
    validate_external_execution_authority,
    validate_external_promotion_documents,
)
from radio_gs.querying.registered_evidence_to_unary import (
    RegisteredEvidenceToUnaryV1,
)


_SHA = "0" * 64


def _assets():
    authority = PromptResponsibilityAuthority(
        scene_id="sentinel",
        frame_id="source",
        camera_name="source",
        colmap_camera_name="source",
        geometry_checkpoint_sha256=_SHA,
        geometry_xyz_sha256=_SHA,
        pose_sha256=_SHA,
        intrinsics_sha256=_SHA,
        height=2,
        width=3,
        num_gaussians=3,
        source_sha256={"geometry": _SHA},
    )
    # Pixel 5 is intentionally unsupported by exact W.
    responsibility = build_prompt_responsibility_cache(
        authority=authority,
        gaussian_ids=torch.tensor([0, 1, 2, 0, 1]),
        pixel_ids=torch.tensor([0, 1, 2, 3, 4]),
        weights=torch.tensor([0.8, 0.7, 0.6, 0.5, 0.4]),
    )
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    valid = torch.ones(3, dtype=torch.bool)
    capability = CanonicalCapabilityBank(
        xyz=xyz,
        valid=valid,
        appearance=torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
        boundary=torch.tensor([[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]]),
        signatures={},
        metadata={},
    )
    state = FactorizedPrimitiveState(
        xyz=xyz,
        valid=valid,
        global_rows=torch.arange(3),
        semantic_direction=torch.zeros(3, 1280, dtype=torch.float16),
        predicted_log_amplitude=torch.zeros(3),
        directional_dispersion=torch.tensor([0.1, 0.2, 0.3]),
        log_amplitude_std=torch.tensor([0.2, 0.3, 0.4]),
        observation_evidence=torch.tensor([0.5, 0.6, 0.7]),
        visibility_purity_value=torch.tensor([0.0, 0.0, 0.0]),
        visibility_purity_known=torch.zeros(3, dtype=torch.bool),
        metadata={
            "geometry_fingerprint": {
                "num_gaussians": 3,
                "xyz_sha256": _SHA,
            }
        },
    )
    return responsibility, capability, state


def test_full_mask_requires_only_exact_w_supported_pixels() -> None:
    responsibility, capability, state = _assets()
    positive = torch.tensor([[1, 0, 0], [0, 0, 0]], dtype=torch.bool)
    negative = torch.tensor([[0, 1, 1], [1, 1, 0]], dtype=torch.bool)
    features, statistics = build_external_registered_features(
        source_responsibility=responsibility,
        positive_mask=positive,
        negative_mask=negative,
        prompt_mode="full_mask",
        capability=capability,
        factorized_state=state,
    )
    assert features.values.shape == (3, 13)
    assert torch.equal(statistics["visible_mass"], responsibility.visible_mass.float())
    assert torch.allclose(features.labeled_coverage, torch.ones(3))


def test_full_mask_rejects_unlabeled_exact_w_supported_pixel() -> None:
    responsibility, capability, state = _assets()
    positive = torch.tensor([[1, 0, 0], [0, 0, 0]], dtype=torch.bool)
    negative = torch.tensor([[0, 1, 1], [1, 0, 0]], dtype=torch.bool)
    try:
        build_external_registered_features(
            source_responsibility=responsibility,
            positive_mask=positive,
            negative_mask=negative,
            prompt_mode="full_mask",
            capability=capability,
            factorized_state=state,
        )
    except ValueError as error:
        assert "supported pixel" in str(error)
    else:
        raise AssertionError("unlabeled exact-W support was accepted")


def test_v1_inference_and_v2_calibration_preserve_order_and_unsupported_zero() -> None:
    responsibility, capability, state = _assets()
    positive = torch.tensor([[1, 0, 0], [0, 0, 0]], dtype=torch.bool)
    negative = torch.tensor([[0, 1, 1], [1, 1, 0]], dtype=torch.bool)
    features, _ = build_external_registered_features(
        source_responsibility=responsibility,
        positive_mask=positive,
        negative_mask=negative,
        prompt_mode="full_mask",
        capability=capability,
        factorized_state=state,
    )
    primitive = infer_registered_primitive_unary(
        RegisteredEvidenceToUnaryV1(), features, device="cpu", chunk_size=2
    )
    assert torch.equal(primitive.probability, features.analytic_probability)

    calls: list[torch.Tensor] = []

    def renderer(probability: torch.Tensor):
        calls.append(probability.clone())
        raw = torch.tensor(
            [
                [float(probability[0]), float(probability[:2].mean())],
                [0.0, float(0.2 + 0.1 * probability[2])],
            ]
        )
        support = torch.tensor([[True, True], [False, True]])
        return raw, support

    calibrated = render_then_calibrate(
        primitive.probability,
        renderer=renderer,
        calibrator=GlobalPromptLogitCalibratorV2(),
    )
    assert len(calls) == 1
    assert torch.equal(calls[0], primitive.probability)
    assert calibrated.calibrated_probability[1, 0].item() == 0.0
    assert torch.allclose(
        calibrated.calibrated_probability[calibrated.supported],
        calibrated.raw_probability[calibrated.supported],
        atol=1e-6,
        rtol=0,
    )


def test_calibration_rejects_nonzero_unsupported_probability() -> None:
    try:
        calibrate_rendered_supported_probability(
            GlobalPromptLogitCalibratorV2(),
            torch.tensor([[0.2, 0.4]]),
            torch.tensor([[True, False]]),
        )
    except ValueError as error:
        assert "unsupported" in str(error)
    else:
        raise AssertionError("nonzero unsupported score was accepted")


def test_calibration_clamps_only_finite_roundoff_and_preserves_unsupported_zero() -> (
    None
):
    calibrated = calibrate_rendered_supported_probability(
        GlobalPromptLogitCalibratorV2(),
        torch.tensor([[-5e-6, 1.0 + 5e-6], [0.0, 0.4]], dtype=torch.float32),
        torch.tensor([[True, True], [False, True]]),
    )
    assert calibrated.raw_probability[0, 0].item() == 0.0
    assert calibrated.raw_probability[0, 1].item() == 1.0
    assert calibrated.raw_probability[1, 0].item() == 0.0
    assert calibrated.calibrated_probability[1, 0].item() == 0.0
    audit = calibrated.strict_domain_audit
    assert audit["render_roundoff_tolerance"] == (
        RENDERED_PROBABILITY_ROUNDOFF_TOLERANCE
    )
    assert audit["raw_total_count"] == 4
    assert audit["raw_nonfinite_count"] == 0
    assert -1e-5 <= audit["raw_finite_minimum"] < 0
    assert 1 < audit["raw_finite_maximum"] <= 1.0 + 1e-5
    assert audit["raw_below_zero_count"] == 1
    assert audit["raw_above_one_count"] == 1
    assert audit["raw_values_clamped"] == 2
    assert 0 < audit["raw_maximum_below_zero_excess"] <= 1e-5
    assert 0 < audit["raw_maximum_above_one_excess"] <= 1e-5


def test_calibration_rejects_large_roundoff_and_nonfinite_values() -> None:
    invalid_values = (-2e-5, 1.0 + 2e-5, float("nan"), float("inf"))
    for invalid in invalid_values:
        try:
            calibrate_rendered_supported_probability(
                GlobalPromptLogitCalibratorV2(),
                torch.tensor([[invalid, 0.4]], dtype=torch.float32),
                torch.tensor([[True, True]]),
            )
        except ValueError as error:
            if torch.isfinite(torch.tensor(invalid)):
                assert "roundoff tolerance" in str(error)
            else:
                assert "must be finite" in str(error)
        else:
            raise AssertionError(f"invalid rendered probability {invalid} accepted")


def test_materializer_writes_supported_calibration_contract(tmp_path) -> None:
    calibrated = calibrate_rendered_supported_probability(
        GlobalPromptLogitCalibratorV2(),
        torch.tensor([[0.2, 0.0], [0.8, 0.5]], dtype=torch.float32),
        torch.tensor([[True, False], [True, True]]),
    )
    record = _write_supported_calibration(tmp_path, "target", calibrated)
    assert record["supported_pixels"] == 3
    support = torch.from_numpy(np.load(record["supported"]["path"], allow_pickle=False))
    assert torch.equal(support, calibrated.supported.to(torch.uint8))


def test_feature_builder_rejects_geometry_authority_mismatch() -> None:
    responsibility, capability, state = _assets()
    mismatched = replace(
        state,
        metadata={
            "geometry_fingerprint": {
                "num_gaussians": 3,
                "xyz_sha256": "1" * 64,
            }
        },
    )
    try:
        build_external_registered_features(
            source_responsibility=responsibility,
            positive_mask=torch.tensor([[1, 0, 0], [0, 0, 0]], dtype=torch.bool),
            negative_mask=torch.tensor([[0, 1, 1], [1, 1, 0]], dtype=torch.bool),
            prompt_mode="full_mask",
            capability=capability,
            factorized_state=mismatched,
        )
    except ValueError as error:
        assert "geometry authorities" in str(error)
    else:
        raise AssertionError("mismatched geometry authority was accepted")


def test_feature_builder_rejects_global_row_authority_mismatch() -> None:
    responsibility, capability, state = _assets()
    mismatched = replace(state, global_rows=torch.tensor([0, 2, 1]))
    try:
        build_external_registered_features(
            source_responsibility=responsibility,
            positive_mask=torch.tensor([[1, 0, 0], [0, 0, 0]], dtype=torch.bool),
            negative_mask=torch.tensor([[0, 1, 1], [1, 1, 0]], dtype=torch.bool),
            prompt_mode="full_mask",
            capability=capability,
            factorized_state=mismatched,
        )
    except ValueError as error:
        assert "global rows" in str(error)
    else:
        raise AssertionError("mismatched global-row authority was accepted")


def _execution_authority() -> dict:
    record = {"path": "/frozen", "sha256": _SHA}
    records = {
        name: dict(record)
        for name in (
            "manifest",
            "source_responsibility_cache",
            "source_responsibility_report",
            "source_positive_mask",
            "source_negative_mask",
            "capability_bank",
            "factorized_primitive_state",
            "v1_checkpoint",
            "v2_checkpoint",
            "carrier_config",
            "carrier_checkpoint",
            "camera_map",
        )
    }
    return {
        "schema": EXECUTION_AUTHORITY_SCHEMA,
        "schema_version": 1,
        "status": "authorized_after_clean_v2_promotion",
        "scene_id": "sentinel",
        "protocol_hash": _SHA,
        "prompt_mode": "signed_scribble",
        "source_frame_id": "source",
        "target_frame_ids": ["target"],
        "records": records,
        "capability_source": "canonical_radio_field_official_frozen_capability_views",
        "promotion": {
            "v1_result": dict(record),
            "v2_result": dict(record),
            "cross_scene_confirmation_receipt": dict(record),
            "cross_scene_confirmation_result": dict(record),
            "v2_clean_gate_passed": True,
            "v2_cross_scene_gate_passed": True,
            "benchmark_execution_authorized": True,
        },
        "source_access": {
            "target_rgb_opened": False,
            "target_mask_opened": False,
            "target_metric_computed": False,
            "target_distribution_fit": False,
            "per_scene_parameter_fit": False,
        },
    }


def test_execution_authority_fails_closed_without_clean_promotion() -> None:
    authority = _execution_authority()
    assert validate_external_execution_authority(authority)["scene_id"] == "sentinel"
    authority["promotion"]["v2_clean_gate_passed"] = False
    try:
        validate_external_execution_authority(authority)
    except ValueError as error:
        assert "did not authorize" in str(error)
    else:
        raise AssertionError("unpromoted V2 checkpoint was accepted")


def _promotion_documents():
    result_sha = "2" * 64
    v2_source = {
        "promotion_gate_passed": True,
        "decision": "eligible_for_cross_scene_confirmation",
        "graph": "off",
        "connected_selection": "off",
    }
    receipt = {
        "schema": (
            "radio_gs.prompt_unary." "cross_scene_clean_confirmation_result_receipt.v1"
        ),
        "status": "scene0002_independent_confirmation_complete",
        "scene_id": "scene0002_00",
        "decision": {
            "V2": (
                "only existing candidate eligible to continue: original whole gate "
                "and independent scene0002 macro cross-scene gate both pass"
            )
        },
        "artifacts": {"result": {"path": "/raw", "sha256": result_sha}},
        "source_access": {
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_queries_opened": False,
            "benchmark_labels_opened": False,
            "benchmark_metrics_opened": False,
            "per_scene_tuning": False,
        },
    }
    result = {
        "status": "cross_scene_clean_confirmation_complete",
        "scene_id": "scene0002_00",
        "fit_or_parameter_update": False,
        "candidate_eligibility": {
            "V2": {
                "eligible": True,
                "reason": (
                    "passed_original_whole_gate_and_independent_cross_scene_gate"
                ),
            },
            "V3": {"eligible": False},
            "V4": {"eligible": False},
        },
        "source_gate_states": {
            "V2": {
                "promotion_gate_passed": True,
                "decision": "eligible_for_cross_scene_confirmation",
            }
        },
        "source_access": {
            "scene0002_labels_enter_fit": False,
            "scene0002_target_rgb_opened": False,
            "per_scene_tuning": False,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_queries_opened": False,
            "benchmark_labels_opened": False,
            "benchmark_metrics_opened": False,
        },
    }
    return v2_source, receipt, result, result_sha


def test_promotion_documents_require_independent_cross_scene_gate() -> None:
    source, receipt, result, digest = _promotion_documents()
    validate_external_promotion_documents(
        v2_source_result=source,
        cross_scene_receipt=receipt,
        cross_scene_result=result,
        expected_cross_scene_result_sha256=digest,
    )
    result["candidate_eligibility"]["V2"]["eligible"] = False
    try:
        validate_external_promotion_documents(
            v2_source_result=source,
            cross_scene_receipt=receipt,
            cross_scene_result=result,
            expected_cross_scene_result_sha256=digest,
        )
    except ValueError as error:
        assert "frozen V2 gate" in str(error)
    else:
        raise AssertionError("failed cross-scene candidate was accepted")
