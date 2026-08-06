"""Target-blind external adapter for the registered-evidence unary method.

This module owns no benchmark evaluator, metric, or Gaussian renderer.  It
only composes three frozen operations behind explicit authorities:

1. exact source-view ``W.T`` evidence registration;
2. :class:`RegisteredEvidenceToUnaryV1` primitive inference; and
3. post-render, supported-pixel-only global logit calibration.

The renderer is deliberately supplied by the caller.  This keeps the method
adapter independent of benchmark code while making the scientifically
important ordering (primitive inference, render, then calibration) testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import torch
import torch.nn.functional as F

from radio_gs.interfaces.capability_cache import CanonicalCapabilityBank
from radio_gs.interfaces.factorized_primitive_state import FactorizedPrimitiveState
from radio_gs.interfaces.prompt_responsibility_cache import PromptResponsibilityCache
from radio_gs.querying.global_prompt_logit_calibrator import (
    GlobalPromptLogitCalibratorV2,
)
from radio_gs.querying.registered_evidence_to_unary import (
    RegisteredEvidenceFeatures,
    RegisteredEvidenceToUnaryV1,
    build_registered_evidence_features,
)


ADAPTER_SCHEMA = "radio_gs.registered_evidence_external_prediction.v1"
EXECUTION_AUTHORITY_SCHEMA = (
    "radio_gs.registered_evidence_external_execution_authority.v2"
)
PROMPT_MODES = ("signed_scribble", "full_mask")
RENDERED_PROBABILITY_ROUNDOFF_TOLERANCE = 1e-5


@dataclass(frozen=True)
class RegisteredPrimitivePrediction:
    """Frozen full-row V1 primitive output."""

    probability: torch.Tensor
    confidence: torch.Tensor
    analytic_probability: torch.Tensor
    bounded_logit_residual: torch.Tensor


@dataclass(frozen=True)
class SupportedCalibration:
    """One rendered frame after the frozen V2 supported-pixel transform."""

    raw_probability: torch.Tensor
    supported: torch.Tensor
    calibrated_probability: torch.Tensor
    strict_domain_audit: Mapping[str, int | float]


RenderCallback = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def _full_state(
    compact: torch.Tensor,
    global_rows: torch.Tensor,
    num_rows: int,
    *,
    boolean: bool = False,
) -> torch.Tensor:
    value = torch.as_tensor(compact).detach().cpu().reshape(-1)
    rows = torch.as_tensor(global_rows).detach().long().cpu().reshape(-1)
    if value.shape != rows.shape:
        raise ValueError("compact factorized state and global rows differ")
    result = torch.zeros(num_rows, dtype=torch.bool if boolean else torch.float32)
    result[rows] = value.bool() if boolean else value.float()
    return result


def _prototype_margin(
    capability: torch.Tensor,
    positive_weight: torch.Tensor,
    negative_weight: torch.Tensor,
) -> torch.Tensor:
    rows = torch.as_tensor(capability).detach().float().cpu()
    positive = torch.as_tensor(positive_weight).detach().float().cpu().reshape(-1)
    negative = torch.as_tensor(negative_weight).detach().float().cpu().reshape(-1)
    if (
        rows.ndim != 2
        or positive.shape != rows.shape[:1]
        or negative.shape != rows.shape[:1]
    ):
        raise ValueError("prototype weights differ from capability rows")
    if float(positive.sum()) <= 0 or float(negative.sum()) <= 0:
        raise ValueError("source prompt lacks positive or negative capability mass")
    rows = F.normalize(rows, dim=1, eps=1e-8)
    positive_prototype = F.normalize(positive @ rows, dim=0, eps=1e-8)
    negative_prototype = F.normalize(negative @ rows, dim=0, eps=1e-8)
    return rows @ (positive_prototype - negative_prototype)


def _validate_source_masks(
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
    cache: PromptResponsibilityCache,
    *,
    prompt_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"prompt_mode must be one of: {', '.join(PROMPT_MODES)}")
    shape = (cache.authority.height, cache.authority.width)
    positive = torch.as_tensor(positive_mask).detach().bool().cpu()
    negative = torch.as_tensor(negative_mask).detach().bool().cpu()
    if positive.shape != shape or negative.shape != shape:
        raise ValueError("source masks differ from exact-W native shape")
    if bool((positive & negative).any()):
        raise ValueError("source positive and negative masks overlap")
    if not bool(positive.any()) or not bool(negative.any()):
        raise ValueError("source prompt must contain positive and negative pixels")
    if prompt_mode == "full_mask":
        pixel_supported = torch.zeros(
            cache.authority.height * cache.authority.width, dtype=torch.bool
        )
        pixel_supported[cache.pixel_ids] = True
        labeled = (positive | negative).reshape(-1)
        if not bool(labeled[pixel_supported].all()):
            raise ValueError(
                "full_mask prompt must label every exact-W supported pixel"
            )
    return positive, negative


def build_external_registered_features(
    *,
    source_responsibility: PromptResponsibilityCache,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
    prompt_mode: str,
    capability: CanonicalCapabilityBank,
    factorized_state: FactorizedPrimitiveState,
) -> tuple[RegisteredEvidenceFeatures, dict[str, torch.Tensor]]:
    """Construct the frozen 13-D V1 input without reading any target asset."""

    positive, negative = _validate_source_masks(
        positive_mask,
        negative_mask,
        source_responsibility,
        prompt_mode=prompt_mode,
    )
    num_rows = int(source_responsibility.authority.num_gaussians)
    if (
        capability.xyz.shape != (num_rows, 3)
        or capability.valid.shape != (num_rows,)
        or capability.num_gaussians != num_rows
    ):
        raise ValueError("capability and source exact-W row counts differ")
    if (
        factorized_state.xyz.shape != (num_rows, 3)
        or factorized_state.valid.shape != (num_rows,)
        or factorized_state.global_rows.ndim != 1
    ):
        raise ValueError("factorized state and source exact-W row counts differ")
    if not torch.equal(
        capability.xyz.float().cpu(), factorized_state.xyz.float().cpu()
    ):
        raise ValueError("capability and factorized-state xyz differ")
    if not torch.equal(
        capability.valid.bool().cpu(), factorized_state.valid.bool().cpu()
    ):
        raise ValueError("capability and factorized-state valid rows differ")
    expected_rows = torch.where(capability.valid.bool().cpu())[0]
    if not torch.equal(expected_rows, factorized_state.global_rows.long().cpu()):
        raise ValueError("capability and factorized-state global rows differ")
    geometry = factorized_state.metadata.get("geometry_fingerprint")
    if (
        not isinstance(geometry, Mapping)
        or int(geometry.get("num_gaussians", -1)) != num_rows
        or geometry.get("xyz_sha256")
        != source_responsibility.authority.geometry_xyz_sha256
    ):
        raise ValueError(
            "source exact-W and factorized-state geometry authorities differ"
        )

    foreground = source_responsibility.adjoint(positive).weighted_sum.float()
    background = source_responsibility.adjoint(negative).weighted_sum.float()
    visible = source_responsibility.visible_mass.float().clone()
    global_rows = capability.global_rows.long().cpu()
    banks = capability.valid_feature_banks()
    dino_compact = _prototype_margin(
        banks["appearance"], foreground[global_rows], background[global_rows]
    )
    sam_compact = _prototype_margin(
        banks["boundary"], foreground[global_rows], background[global_rows]
    )
    dino_margin = torch.zeros(num_rows, dtype=torch.float32)
    sam_margin = torch.zeros(num_rows, dtype=torch.float32)
    dino_margin[global_rows] = dino_compact
    sam_margin[global_rows] = sam_compact

    state_rows = factorized_state.global_rows.long().cpu()
    dispersion = _full_state(
        factorized_state.directional_dispersion, state_rows, num_rows
    )
    amplitude_std = _full_state(
        factorized_state.log_amplitude_std, state_rows, num_rows
    )
    evidence = _full_state(factorized_state.observation_evidence, state_rows, num_rows)
    purity = _full_state(factorized_state.visibility_purity_value, state_rows, num_rows)
    purity_known = _full_state(
        factorized_state.visibility_purity_known,
        state_rows,
        num_rows,
        boolean=True,
    )
    features = build_registered_evidence_features(
        foreground_mass=foreground,
        background_mass=background,
        visible_mass=visible,
        dino_margin=dino_margin,
        sam_margin=sam_margin,
        directional_dispersion=dispersion,
        log_amplitude_std=amplitude_std,
        observation_evidence=evidence,
        visibility_purity_value=purity,
        visibility_purity_known=purity_known,
        capability_valid=capability.valid,
        source_view_support=(visible > 0).float(),
    )
    return features, {
        "foreground_mass": foreground.contiguous(),
        "background_mass": background.contiguous(),
        "visible_mass": visible.contiguous(),
        "dino_margin": dino_margin.contiguous(),
        "sam_margin": sam_margin.contiguous(),
    }


def _subset_features(
    features: RegisteredEvidenceFeatures, start: int, stop: int, device: torch.device
) -> RegisteredEvidenceFeatures:
    return RegisteredEvidenceFeatures(
        values=features.values[start:stop].to(device),
        analytic_probability=features.analytic_probability[start:stop].to(device),
        registered_probability=features.registered_probability[start:stop].to(device),
        labeled_coverage=features.labeled_coverage[start:stop].to(device),
        capability_valid=features.capability_valid[start:stop].to(device),
    )


@torch.inference_mode()
def infer_registered_primitive_unary(
    head: RegisteredEvidenceToUnaryV1,
    features: RegisteredEvidenceFeatures,
    *,
    device: torch.device | str,
    chunk_size: int = 65536,
) -> RegisteredPrimitivePrediction:
    """Run the frozen V1 head in bounded chunks and return full global rows."""

    target_device = torch.device(device)
    size = int(chunk_size)
    if size <= 0:
        raise ValueError("chunk_size must be positive")
    count = int(features.values.shape[0])
    head = head.to(target_device).eval().requires_grad_(False)
    outputs = {
        name: torch.empty(count, dtype=torch.float32)
        for name in (
            "probability",
            "confidence",
            "analytic_probability",
            "bounded_logit_residual",
        )
    }
    for start in range(0, count, size):
        stop = min(start + size, count)
        prediction = head(_subset_features(features, start, stop, target_device))
        outputs["probability"][
            start:stop
        ] = prediction.foreground_probability.float().cpu()
        outputs["confidence"][start:stop] = prediction.confidence.float().cpu()
        outputs["analytic_probability"][
            start:stop
        ] = prediction.analytic_probability.float().cpu()
        outputs["bounded_logit_residual"][
            start:stop
        ] = prediction.bounded_logit_residual.float().cpu()
    for name, value in outputs.items():
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"V1 primitive {name} contains NaN or infinity")
    probability = outputs["probability"]
    if bool(((probability <= 0) | (probability >= 1)).any()):
        raise ValueError("V1 primitive probability must be strictly inside (0,1)")
    return RegisteredPrimitivePrediction(
        probability=probability.contiguous(),
        confidence=outputs["confidence"].contiguous(),
        analytic_probability=outputs["analytic_probability"].contiguous(),
        bounded_logit_residual=outputs["bounded_logit_residual"].contiguous(),
    )


@torch.inference_mode()
def calibrate_rendered_supported_probability(
    calibrator: GlobalPromptLogitCalibratorV2,
    raw_probability: torch.Tensor,
    supported: torch.Tensor,
    *,
    device: torch.device | str = "cpu",
) -> SupportedCalibration:
    """Apply V2 after render and leave unsupported pixels at exact zero."""

    raw = torch.as_tensor(raw_probability).detach().float().cpu()
    support = torch.as_tensor(supported).detach().bool().cpu()
    if raw.ndim != 2 or support.shape != raw.shape:
        raise ValueError("rendered probability/support must align as [H,W]")
    finite = torch.isfinite(raw)
    if not bool(finite.all()):
        raise ValueError("rendered probability must be finite")
    raw_finite_minimum = float(raw.min())
    raw_finite_maximum = float(raw.max())
    below_zero = raw < 0
    above_one = raw > 1
    maximum_below_zero_excess = (
        float((-raw[below_zero]).max()) if bool(below_zero.any()) else 0.0
    )
    maximum_above_one_excess = (
        float((raw[above_one] - 1.0).max()) if bool(above_one.any()) else 0.0
    )
    if (
        maximum_below_zero_excess > RENDERED_PROBABILITY_ROUNDOFF_TOLERANCE
        or maximum_above_one_excess > RENDERED_PROBABILITY_ROUNDOFF_TOLERANCE
    ):
        raise ValueError("rendered probability exceeds the numeric roundoff tolerance")
    if not bool(support.any()):
        raise ValueError("rendered frame has no supported pixels")
    if bool(raw[~support].ne(0).any()):
        raise ValueError("unsupported rendered probability must be exact zero")
    raw = raw.clamp(0.0, 1.0)
    if bool(raw[~support].ne(0).any()):
        raise RuntimeError("numeric clamp changed unsupported pixels")
    target_device = torch.device(device)
    calibrator = calibrator.to(target_device).eval().requires_grad_(False)
    supported_probability = raw[support].to(target_device)
    audit: dict[str, int | float] = dict(
        calibrator.strict_domain_audit(supported_probability)
    )
    audit.update(
        {
            "render_roundoff_tolerance": (RENDERED_PROBABILITY_ROUNDOFF_TOLERANCE),
            "raw_total_count": int(raw.numel()),
            "raw_nonfinite_count": 0,
            "raw_finite_minimum": raw_finite_minimum,
            "raw_finite_maximum": raw_finite_maximum,
            "raw_below_zero_count": int(below_zero.sum()),
            "raw_above_one_count": int(above_one.sum()),
            "raw_maximum_below_zero_excess": maximum_below_zero_excess,
            "raw_maximum_above_one_excess": maximum_above_one_excess,
            "raw_values_clamped": int((below_zero | above_one).sum()),
        }
    )
    calibrated = torch.zeros_like(raw)
    calibrated[support] = calibrator(supported_probability).float().cpu()
    if not bool(torch.isfinite(calibrated).all()) or bool(
        ((calibrated < 0) | (calibrated > 1)).any()
    ):
        raise ValueError("calibrated rendered probability is invalid")
    if bool(calibrated[~support].ne(0).any()):
        raise RuntimeError("calibration changed unsupported pixels")
    return SupportedCalibration(
        raw_probability=raw.contiguous(),
        supported=support.contiguous(),
        calibrated_probability=calibrated.contiguous(),
        strict_domain_audit=audit,
    )


def render_then_calibrate(
    primitive_probability: torch.Tensor,
    *,
    renderer: RenderCallback,
    calibrator: GlobalPromptLogitCalibratorV2,
    calibration_device: torch.device | str = "cpu",
) -> SupportedCalibration:
    """Make the non-commuting render-before-calibration order explicit."""

    raw, supported = renderer(primitive_probability)
    return calibrate_rendered_supported_probability(
        calibrator,
        raw,
        supported,
        device=calibration_device,
    )


def validate_external_execution_authority(value: object) -> dict[str, object]:
    """Validate a per-scene authority created only after the clean V2 gate."""

    if not isinstance(value, Mapping):
        raise ValueError("external execution authority must be a mapping")
    authority = dict(value)
    required = {
        "schema",
        "schema_version",
        "status",
        "scene_id",
        "protocol_hash",
        "prompt_mode",
        "source_frame_id",
        "target_frame_ids",
        "records",
        "capability_source",
        "promotion",
        "source_access",
    }
    if set(authority) != required:
        raise ValueError("external execution authority fields differ")
    if (
        authority.get("schema") != EXECUTION_AUTHORITY_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("status") != "authorized_after_clean_v2_promotion"
        or authority.get("prompt_mode") not in PROMPT_MODES
        or not str(authority.get("scene_id", "")).strip()
        or not str(authority.get("protocol_hash", "")).strip()
        or not str(authority.get("source_frame_id", "")).strip()
    ):
        raise ValueError("external execution authority header differs")
    frames = authority.get("target_frame_ids")
    if (
        not isinstance(frames, list)
        or not frames
        or len({str(frame) for frame in frames}) != len(frames)
        or any(not str(frame).strip() for frame in frames)
    ):
        raise ValueError("external execution target frames differ")
    records = authority.get("records")
    required_records = {
        "manifest",
        "source_responsibility_cache",
        "source_responsibility_report",
        "source_positive_mask",
        "capability_bank",
        "factorized_primitive_state",
        "v1_checkpoint",
        "v2_checkpoint",
        "carrier_config",
        "carrier_checkpoint",
        "camera_map",
    }
    if authority["prompt_mode"] == "signed_scribble":
        required_records.add("source_negative_mask")
    if not isinstance(records, Mapping) or set(records) != required_records:
        raise ValueError("external execution file records differ")
    for label, record in records.items():
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise ValueError(f"external execution {label} record differs")
    promotion = authority.get("promotion")
    if not isinstance(promotion, Mapping) or set(promotion) != {
        "v1_result",
        "v2_result",
        "cross_scene_confirmation_receipt",
        "cross_scene_confirmation_result",
        "v2_clean_gate_passed",
        "v2_cross_scene_gate_passed",
        "benchmark_execution_authorized",
    }:
        raise ValueError("external execution promotion authority differs")
    if (
        promotion.get("v2_clean_gate_passed") is not True
        or promotion.get("v2_cross_scene_gate_passed") is not True
        or promotion.get("benchmark_execution_authorized") is not True
    ):
        raise ValueError(
            "clean and cross-scene V2 promotion did not authorize benchmark execution"
        )
    for label in (
        "v1_result",
        "v2_result",
        "cross_scene_confirmation_receipt",
        "cross_scene_confirmation_result",
    ):
        record = promotion.get(label)
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise ValueError(f"external execution {label} result record differs")
    source_access = authority.get("source_access")
    forbidden = (
        "target_rgb_opened",
        "target_mask_opened",
        "target_metric_computed",
        "target_distribution_fit",
        "per_scene_parameter_fit",
    )
    if not isinstance(source_access, Mapping) or set(source_access) != set(forbidden):
        raise ValueError("external execution source-access declaration differs")
    if any(source_access.get(name) is not False for name in forbidden):
        raise ValueError("external execution authority is target-contaminated")
    if authority.get("capability_source") not in {
        "canonical_radio_field_official_frozen_capability_views",
        "exact_capability_mpr_official_frozen_capability_views",
    }:
        raise ValueError("external execution capability source differs")
    return authority


def validate_external_promotion_documents(
    *,
    v2_source_result: object,
    cross_scene_receipt: object,
    cross_scene_result: object,
    expected_cross_scene_result_sha256: str,
) -> None:
    """Validate the two-stage clean promotion before benchmark prediction.

    File SHA verification is deliberately owned by the caller.  This pure
    validator checks that the bound documents actually encode the frozen
    scene0001 gate and the independent scene0002 confirmation, rather than
    trusting self-declared booleans in an execution authority.
    """

    if not isinstance(v2_source_result, Mapping):
        raise ValueError("V2 source promotion result must be a mapping")
    if (
        v2_source_result.get("promotion_gate_passed") is not True
        or v2_source_result.get("decision") != "eligible_for_cross_scene_confirmation"
        or v2_source_result.get("graph") != "off"
        or v2_source_result.get("connected_selection") != "off"
    ):
        raise ValueError("V2 source promotion document did not pass its frozen gate")

    if not isinstance(cross_scene_receipt, Mapping):
        raise ValueError("cross-scene promotion receipt must be a mapping")
    if (
        cross_scene_receipt.get("schema")
        != "radio_gs.prompt_unary.cross_scene_clean_confirmation_result_receipt.v1"
        or cross_scene_receipt.get("status")
        != "scene0002_independent_confirmation_complete"
        or cross_scene_receipt.get("scene_id") != "scene0002_00"
        or cross_scene_receipt.get("decision", {}).get("V2")
        != (
            "only existing candidate eligible to continue: original whole gate "
            "and independent scene0002 macro cross-scene gate both pass"
        )
    ):
        raise ValueError("cross-scene promotion receipt did not authorize V2")
    receipt_result = cross_scene_receipt.get("artifacts", {}).get("result")
    if (
        not isinstance(receipt_result, Mapping)
        or receipt_result.get("sha256") != expected_cross_scene_result_sha256
    ):
        raise ValueError("cross-scene receipt binds another raw result")
    receipt_access = cross_scene_receipt.get("source_access")
    if not isinstance(receipt_access, Mapping) or any(
        receipt_access.get(name) is not False
        for name in (
            "benchmark_images_opened",
            "benchmark_masks_opened",
            "benchmark_queries_opened",
            "benchmark_labels_opened",
            "benchmark_metrics_opened",
            "per_scene_tuning",
        )
    ):
        raise ValueError("cross-scene promotion receipt is benchmark-contaminated")

    if not isinstance(cross_scene_result, Mapping):
        raise ValueError("cross-scene promotion result must be a mapping")
    eligibility = cross_scene_result.get("candidate_eligibility")
    source_gates = cross_scene_result.get("source_gate_states")
    result_access = cross_scene_result.get("source_access")
    if (
        cross_scene_result.get("status") != "cross_scene_clean_confirmation_complete"
        or cross_scene_result.get("scene_id") != "scene0002_00"
        or cross_scene_result.get("fit_or_parameter_update") is not False
        or not isinstance(eligibility, Mapping)
        or eligibility.get("V2", {}).get("eligible") is not True
        or eligibility.get("V2", {}).get("reason")
        != "passed_original_whole_gate_and_independent_cross_scene_gate"
        or eligibility.get("V3", {}).get("eligible") is not False
        or eligibility.get("V4", {}).get("eligible") is not False
        or not isinstance(source_gates, Mapping)
        or source_gates.get("V2", {}).get("promotion_gate_passed") is not True
        or source_gates.get("V2", {}).get("decision")
        != "eligible_for_cross_scene_confirmation"
        or not isinstance(result_access, Mapping)
        or result_access.get("scene0002_labels_enter_fit") is not False
        or result_access.get("scene0002_target_rgb_opened") is not False
        or result_access.get("per_scene_tuning") is not False
        or any(
            result_access.get(name) is not False
            for name in (
                "benchmark_images_opened",
                "benchmark_masks_opened",
                "benchmark_queries_opened",
                "benchmark_labels_opened",
                "benchmark_metrics_opened",
            )
        )
    ):
        raise ValueError("cross-scene raw result did not pass the frozen V2 gate")
