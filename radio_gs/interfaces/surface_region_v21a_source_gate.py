"""Fail-closed source-only promotion gate for the V2.1A rescue candidate."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import re
from typing import Any

import torch

from radio_gs.interfaces import surface_region_v21_source_gate as v21_gate
from radio_gs.interfaces.surface_region_summary import (
    surface_region_state_dict_sha256,
)
from radio_gs.interfaces.surface_region_typed_context_training import (
    accepted_v2_authority,
)
from radio_gs.models.surface_region_typed_context_residual import (
    SurfaceRegionAcceptedV2TypedContextResidualV1,
)
from radio_gs.scripts import (
    train_surface_region_typed_context_response_listwise_v21a_rescue as rescue,
)
from radio_gs.scripts import train_surface_region_typed_context_residual as v1_trainer
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
)


PROMOTION_SCHEMA = "radio_gs.surface_region_v21a_source_promotion_evidence.v1"
PROMOTION_CHAIN_SCHEMA = "radio_gs.surface_region_v21a_source_promotion_chain.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RESPONSE_KEYS = (
    "response_auxiliary_loss",
    "response_absolute_relevance_loss",
    "response_continuous_pairwise_relevance_loss",
)


def _finite(value: object, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _sha(value: object, *, label: str) -> str:
    result = str(value)
    if _SHA256.fullmatch(result) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return result


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    path = str(value["path"])
    digest = _sha(value["sha256"], label=f"{label} digest")
    if not path.startswith("/"):
        raise ValueError(f"{label} path must be absolute")
    return {"path": path, "sha256": digest}


def _validated_record(value: object, *, label: str) -> dict[str, str]:
    record = _record(value, label=label)
    path = validate_file_record(record, label=label)
    if str(path) != record["path"]:
        raise ValueError(f"{label} path is not canonical")
    return record


def _open_execution_authority(
    value: object,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    verified = _validated_record(value, label="V2.1A execution authority")
    raw, digest, source = load_json_object(
        verified["path"],
        expected_sha256=verified["sha256"],
        label="V2.1A execution authority",
    )
    authority = rescue.validate_execution_authority(raw)
    expected_implementations = rescue._implementation_records()
    if any(
        authority[name] != expected_implementations[name]
        for name in expected_implementations
    ):
        raise ValueError("V2.1A execution implementation authority differs")
    prepared = rescue.prepare_inputs(source, expected_sha256=digest)
    prepared_execution = dict(prepared.execution)
    prepared_execution.pop("verified_path", None)
    prepared_execution.pop("verified_sha256", None)
    if prepared_execution != authority:
        raise ValueError("V2.1A complete source execution reload differs")
    return authority, {"path": str(source), "sha256": digest}, prepared.registry


def _validate_response_summary(value: object, *, label: str) -> dict[str, Any]:
    required = {
        "scene_count",
        "scene_macro_auxiliary_loss",
        "scene_macro_absolute_relevance_loss",
        "scene_macro_continuous_pairwise_relevance_loss",
        "scene_macro_active_over_eligible_coverage",
        "minimum_active_over_eligible_coverage",
        "scene_macro_pair_trainable_endpoint_coverage",
        "minimum_pair_trainable_endpoint_coverage",
        "all_authority_pairs_retained",
        "per_scene",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError(f"{label} response summary differs")
    result = dict(value)
    per_scene = result["per_scene"]
    if (
        result["scene_count"] != len(rescue.VALIDATION_SCENES)
        or result["all_authority_pairs_retained"] is not True
        or not isinstance(per_scene, Mapping)
        or tuple(per_scene) != rescue.VALIDATION_SCENES
    ):
        raise ValueError(f"{label} validation scene axis differs")
    rows: dict[str, dict[str, Any]] = {}
    for scene in rescue.VALIDATION_SCENES:
        raw = per_scene[scene]
        required_row = {
            *_RESPONSE_KEYS,
            "active_rows",
            "eligible_rows",
            "active_eligible_rows",
            "active_over_eligible_coverage",
            "angle_radians_p50",
            "angle_radians_p95",
            "angle_cap_fraction",
            "response_authority_hard_negative_pairs",
            "response_pairwise_objective_hard_negative_pairs",
            "response_triplet_objective_hard_negative_pairs",
            "response_pair_trainable_endpoint_coverage",
        }
        if not isinstance(raw, Mapping) or not required_row.issubset(raw):
            raise ValueError(f"{label} {scene} diagnostics differ")
        v21_gate._finite_scalar_tree(raw, label=f"{label}.{scene}")
        row = dict(raw)
        losses = [_finite(row[name], label=f"{label}.{scene}.{name}") for name in _RESPONSE_KEYS]
        if any(item < 0 for item in losses):
            raise ValueError(f"{label} {scene} response loss is negative")
        eligible = int(row["eligible_rows"])
        active = int(row["active_rows"])
        active_eligible = int(row["active_eligible_rows"])
        coverage = _finite(
            row["active_over_eligible_coverage"],
            label=f"{label}.{scene}.active coverage",
        )
        pair_coverage = _finite(
            row["response_pair_trainable_endpoint_coverage"],
            label=f"{label}.{scene}.pair coverage",
        )
        authority_pairs = int(row["response_authority_hard_negative_pairs"])
        if (
            eligible <= 0
            or not 0 <= active_eligible <= active
            or active != active_eligible
            or active_eligible > eligible
            or not math.isclose(
                coverage, active_eligible / eligible, rel_tol=0.0, abs_tol=1e-9
            )
            or not 0.0 <= pair_coverage <= 1.0
            or authority_pairs <= 0
            or int(row["response_pairwise_objective_hard_negative_pairs"])
            != authority_pairs
            or int(row["response_triplet_objective_hard_negative_pairs"])
            != authority_pairs
        ):
            raise ValueError(f"{label} {scene} denominator or coverage differs")
        p50 = _finite(row["angle_radians_p50"], label=f"{label}.{scene}.p50")
        p95 = _finite(row["angle_radians_p95"], label=f"{label}.{scene}.p95")
        cap_fraction = _finite(
            row["angle_cap_fraction"], label=f"{label}.{scene}.cap fraction"
        )
        if not 0.0 <= p50 <= p95 <= v1_trainer.MAX_ANGLE_RADIANS + 2e-6 or not 0.0 <= cap_fraction <= 1.0:
            raise ValueError(f"{label} {scene} angle diagnostics differ")
        rows[scene] = row
    macro_fields = {
        "scene_macro_auxiliary_loss": "response_auxiliary_loss",
        "scene_macro_absolute_relevance_loss": "response_absolute_relevance_loss",
        "scene_macro_continuous_pairwise_relevance_loss": (
            "response_continuous_pairwise_relevance_loss"
        ),
        "scene_macro_active_over_eligible_coverage": (
            "active_over_eligible_coverage"
        ),
        "scene_macro_pair_trainable_endpoint_coverage": (
            "response_pair_trainable_endpoint_coverage"
        ),
    }
    for declared, field in macro_fields.items():
        expected = sum(float(rows[scene][field]) for scene in rows) / len(rows)
        if not math.isclose(
            _finite(result[declared], label=f"{label}.{declared}"),
            expected,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"{label} {declared} differs")
    expected_active_min = min(
        float(row["active_over_eligible_coverage"]) for row in rows.values()
    )
    expected_pair_min = min(
        float(row["response_pair_trainable_endpoint_coverage"])
        for row in rows.values()
    )
    if not math.isclose(
        _finite(
            result["minimum_active_over_eligible_coverage"],
            label=f"{label}.minimum active coverage",
        ),
        expected_active_min,
        rel_tol=0.0,
        abs_tol=1e-9,
    ) or not math.isclose(
        _finite(
            result["minimum_pair_trainable_endpoint_coverage"],
            label=f"{label}.minimum pair coverage",
        ),
        expected_pair_min,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(f"{label} minimum coverage differs")
    result["per_scene"] = rows
    return result


def _validate_validation(value: object, *, label: str) -> dict[str, Any]:
    required = {
        "v1_non_regression",
        "response_listwise_v21a",
        "validation_no_grad",
        "benchmark_opened",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError(f"{label} validation fields differ")
    result = dict(value)
    if (
        result["validation_no_grad"] is not True
        or result["benchmark_opened"] is not False
    ):
        raise ValueError(f"{label} validation protocol differs")
    v21_gate._validate_v1_non_regression(
        result["v1_non_regression"], label=f"{label}.v1"
    )
    result["response_listwise_v21a"] = _validate_response_summary(
        result["response_listwise_v21a"], label=label
    )
    if any(
        int(result["response_listwise_v21a"]["per_scene"][scene]["active_rows"])
        != int(result["v1_non_regression"]["per_scene"][scene]["active_rows"])
        for scene in rescue.VALIDATION_SCENES
    ):
        raise ValueError(f"{label} response/V1 active rows differ")
    return result


def _validate_training(value: object, *, label: str) -> dict[str, Any]:
    required = {
        "optimizer_step_completed",
        "scene_count",
        "equal_scene_weight",
        "complete_scene_forward",
        "pairwise_any_trainable_endpoint_filter",
        "triplet_anchor_trainable_filter",
        "gradient_norms_preclip",
        "global_gradient_norm_postclip",
        "response_listwise_v21a",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError(f"{label} training fields differ")
    result = dict(value)
    if (
        result["optimizer_step_completed"] is not True
        or result["scene_count"] != len(rescue.TRAIN_SCENES)
        or float(result["equal_scene_weight"]) != 1.0 / len(rescue.TRAIN_SCENES)
        or result["complete_scene_forward"] is not True
        or result["pairwise_any_trainable_endpoint_filter"] is not True
        or result["triplet_anchor_trainable_filter"] is not True
    ):
        raise ValueError(f"{label} optimizer protocol differs")
    gradients = result["gradient_norms_preclip"]
    names = {
        "descriptor_projection",
        "context_projection",
        "scalar_projection",
        "fusion_projection",
        "residual_projection",
        "global",
    }
    if not isinstance(gradients, Mapping) or set(gradients) != names:
        raise ValueError(f"{label} gradient component axis differs")
    if any(_finite(value, label=f"{label}.{name}") < 0 for name, value in gradients.items()):
        raise ValueError(f"{label} gradient norm is negative")
    postclip = _finite(
        result["global_gradient_norm_postclip"], label=f"{label}.postclip"
    )
    if postclip < 0 or postclip > v1_trainer.MAX_GRADIENT_NORM + 1e-5:
        raise ValueError(f"{label} postclip norm differs")
    # Training has four scenes rather than the validation pair, so validate
    # only its finite diagnostics here; held-out coverage is recomputed below.
    v21_gate._finite_scalar_tree(
        result["response_listwise_v21a"], label=f"{label}.response"
    )
    return result


def _validate_history(value: object) -> list[dict[str, Any]]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != rescue.OPTIMIZER_STEPS + 1
    ):
        raise ValueError("V2.1A history must contain epoch zero plus 30 steps")
    result: list[dict[str, Any]] = []
    for epoch, raw in enumerate(value):
        required = {"epoch", "training", "validation", "model_state_dict_sha256"}
        if not isinstance(raw, Mapping) or set(raw) != required or raw["epoch"] != epoch:
            raise ValueError("V2.1A history row differs")
        training = raw["training"]
        if epoch == 0:
            if training is not None:
                raise ValueError("V2.1A epoch zero must not be an optimizer step")
        else:
            training = _validate_training(training, label=f"epoch {epoch}")
        result.append(
            {
                **dict(raw),
                "training": training,
                "validation": _validate_validation(
                    raw["validation"], label=f"epoch {epoch}"
                ),
                "model_state_dict_sha256": _sha(
                    raw["model_state_dict_sha256"], label=f"epoch {epoch} state"
                ),
            }
        )
    return result


def _diagnostic_records(value: object) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != {"best_raw_aux", "final"}:
        raise ValueError("V2.1A diagnostic state roles differ")
    result: dict[str, dict[str, Any]] = {}
    for role in ("best_raw_aux", "final"):
        raw = value[role]
        if not isinstance(raw, Mapping) or set(raw) != {
            "epoch",
            "model_state_dict_sha256",
            "file",
        }:
            raise ValueError(f"V2.1A {role} diagnostic record differs")
        epoch = raw["epoch"]
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch <= 0:
            raise ValueError(f"V2.1A {role} epoch differs")
        result[role] = {
            "epoch": epoch,
            "model_state_dict_sha256": _sha(
                raw["model_state_dict_sha256"], label=f"{role} state"
            ),
            "file": _record(raw["file"], label=f"{role} file"),
        }
    return result


def validate_source_promotion_evidence(value: object) -> dict[str, Any]:
    required = {
        "schema",
        "schema_version",
        "status",
        "training_contract",
        "training_contract_sha256",
        "implementation",
        "objective_adapter",
        "loss_implementation",
        "preregistration",
        "execution_authority_addendum",
        "execution_authority",
        "checkpoint",
        "normalization_authority",
        "certificate",
        "diagnostic_states",
        "optimizer_steps_completed",
        "selected_epoch",
        "selection_status",
        "selected_validation",
        "best_raw_aux_epoch",
        "final_epoch",
        "history",
        "source_access",
        "benchmark_opened",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("V2.1A source result fields differ")
    result = dict(value)
    contract = rescue.training_contract()
    if (
        result["schema"] != rescue.RESULT_SCHEMA
        or result["schema_version"] != 1
        or result["status"]
        not in {
            "source_only_rescue_promotion_candidate_complete",
            "source_only_rescue_diagnostic_complete_no_promotion_checkpoint",
        }
        or result["training_contract"] != contract
        or result["training_contract_sha256"] != canonical_json_sha256(contract)
        or result["optimizer_steps_completed"] != rescue.OPTIMIZER_STEPS
        or result["final_epoch"] != rescue.OPTIMIZER_STEPS
        or result["source_access"] != rescue.source_access()
        or result["benchmark_opened"] is not False
    ):
        raise ValueError("V2.1A source result identity differs")
    for name in (
        "implementation",
        "objective_adapter",
        "loss_implementation",
        "preregistration",
        "execution_authority_addendum",
        "execution_authority",
        "normalization_authority",
        "certificate",
    ):
        result[name] = _record(result[name], label=name)
    history = _validate_history(result["history"])
    selected = rescue.select_promotion_epoch(history)
    best_raw = rescue.select_best_raw_aux_epoch(history)
    result["diagnostic_states"] = _diagnostic_records(result["diagnostic_states"])
    if (
        result["selected_epoch"] != selected
        or result["best_raw_aux_epoch"] != best_raw
        or result["diagnostic_states"]["best_raw_aux"]["epoch"] != best_raw
        or result["diagnostic_states"]["final"]["epoch"] != rescue.OPTIMIZER_STEPS
    ):
        raise ValueError("V2.1A source selection differs")
    passed = selected is not None
    if passed:
        checks = rescue.promotion_checks(history, selected)
        if (
            result["checkpoint"] is None
            or result["selection_status"]
            != "promotion_eligible_minimum_auxiliary"
            or result["status"]
            != "source_only_rescue_promotion_candidate_complete"
            or result["selected_validation"] != result["history"][selected]["validation"]
        ):
            raise ValueError("V2.1A promoted result bindings differ")
        result["checkpoint"] = _record(result["checkpoint"], label="checkpoint")
    else:
        checks = {
            name: False
            for name in (
                "selected_epoch_positive",
                "auxiliary_macro_strictly_improved",
                "absolute_relevance_macro_strictly_improved",
                "absolute_relevance_every_scene_non_regression",
                "continuous_pairwise_macro_strictly_improved",
                "v1_fidelity_non_regression",
                "minimum_active_over_eligible_coverage_at_least_95pct",
                "minimum_pair_trainable_endpoint_coverage_at_least_95pct",
            )
        }
        if (
            result["checkpoint"] is not None
            or result["selected_validation"] is not None
            or result["selection_status"]
            != "no_epoch_satisfied_all_promotion_constraints"
            or result["status"]
            != "source_only_rescue_diagnostic_complete_no_promotion_checkpoint"
        ):
            raise ValueError("V2.1A diagnostic-only result bindings differ")
    return {
        "schema": PROMOTION_SCHEMA,
        "schema_version": 1,
        "selected_epoch": selected,
        "checks": checks,
        "coverage_threshold": rescue.COVERAGE_THRESHOLD,
        "passed": passed and all(checks.values()),
        "optimizer_steps_completed": rescue.OPTIMIZER_STEPS,
        "target_execution_authorized": False,
        "benchmark_opened": False,
    }


def _validate_model_state(
    state: object, normalization: Mapping[str, Any]
) -> tuple[dict[str, torch.Tensor], str, dict[str, Any]]:
    model = SurfaceRegionAcceptedV2TypedContextResidualV1(
        scalar_median=normalization["median"],
        scalar_robust_scale=normalization["robust_scale"],
        max_angle_radians=v1_trainer.MAX_ANGLE_RADIANS,
        max_alpha=v1_trainer.MAX_ALPHA,
    )
    expected = model.state_dict()
    if not isinstance(state, Mapping) or set(state) != set(expected):
        raise ValueError("V2.1A model state axis differs")
    frozen: dict[str, torch.Tensor] = {}
    for name, reference in expected.items():
        raw = state[name]
        if not torch.is_tensor(raw):
            raise ValueError(f"V2.1A model state {name} is not a tensor")
        tensor = raw.detach().cpu().contiguous()
        if (
            tensor.dtype != reference.dtype
            or tensor.shape != reference.shape
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError(f"V2.1A model state {name} differs")
        frozen[name] = tensor
    if not torch.equal(frozen["scalar_median"], normalization["median"]) or not torch.equal(
        frozen["scalar_robust_scale"], normalization["robust_scale"]
    ):
        raise ValueError("V2.1A state normalization buffers differ")
    digest = surface_region_state_dict_sha256(frozen)
    model.load_state_dict(frozen, strict=True)
    return frozen, digest, model.architecture()


def _open_diagnostic_state(
    role: str,
    record: Mapping[str, str],
    *,
    normalization: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    verified = _validated_record(record, label=f"V2.1A {role} diagnostic")
    raw, digest, path = load_torch_mapping(
        verified["path"],
        expected_sha256=verified["sha256"],
        map_location="cpu",
        label=f"V2.1A {role} diagnostic",
    )
    required = {
        "schema",
        "schema_version",
        "role",
        "epoch",
        "model_class",
        "model_state_dict",
        "model_state_dict_sha256",
        "normalization_authority",
        "execution_authority",
        "implementation",
        "objective_adapter",
        "loss_implementation",
        "preregistration",
        "execution_authority_addendum",
        "source_access",
        "benchmark_opened",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError(f"V2.1A {role} diagnostic fields differ")
    payload = dict(raw)
    _state, state_sha, _architecture = _validate_model_state(
        payload["model_state_dict"], normalization
    )
    if (
        payload["schema"] != rescue.DIAGNOSTIC_STATE_SCHEMA
        or payload["schema_version"] != 1
        or payload["role"] != role
        or payload["model_class"] != SurfaceRegionAcceptedV2TypedContextResidualV1.__name__
        or payload["model_state_dict_sha256"] != state_sha
        or payload["source_access"] != rescue.source_access()
        or payload["benchmark_opened"] is not False
    ):
        raise ValueError(f"V2.1A {role} diagnostic identity differs")
    return payload, {"path": str(path), "sha256": digest}


def validate_source_rescue_chain(
    path: str | Path,
    *,
    expected_sha256: str,
    require_promotion: bool = True,
) -> dict[str, Any]:
    raw, result_sha, result_path = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="V2.1A source rescue result",
    )
    promotion = validate_source_promotion_evidence(raw)
    expected_records = rescue._implementation_records()
    for name, expected in expected_records.items():
        if _validated_record(raw[name], label=f"V2.1A {name}") != expected:
            raise ValueError(f"V2.1A {name} authority differs")
    prereg, _prereg_sha, _prereg_path = load_json_object(
        expected_records["preregistration"]["path"],
        expected_sha256=expected_records["preregistration"]["sha256"],
        label="V2.1A preregistration",
    )
    if (
        prereg.get("schema")
        != "radio_gs.source_global_response_listwise_v21a_rescue_preregistration.v1"
        or prereg.get("status") != "preregistered_before_implementation_or_execution"
        or prereg.get("benchmark_execution_authorized") is not False
        or prereg.get("benchmark_opened") is not False
    ):
        raise ValueError("V2.1A preregistration identity differs")
    addendum, _addendum_sha, _addendum_path = load_json_object(
        expected_records["execution_authority_addendum"]["path"],
        expected_sha256=expected_records["execution_authority_addendum"]["sha256"],
        label="V2.1A independent execution-authority addendum",
    )
    if (
        addendum.get("schema")
        != "radio_gs.source_global_response_listwise_v21a_execution_authority_addendum.v1"
        or addendum.get("status")
        != "preregistered_clarification_before_builder_implementation"
        or addendum.get("benchmark_execution_authorized") is not False
        or addendum.get("benchmark_opened") is not False
        or addendum.get("base_preregistration")
        != expected_records["preregistration"]
    ):
        raise ValueError("V2.1A execution-authority addendum identity differs")
    execution, execution_record, registry = _open_execution_authority(
        raw["execution_authority"]
    )
    normalization_record = _validated_record(
        raw["normalization_authority"], label="V2.1A normalization"
    )
    normalization_raw, normalization_sha, normalization_path = load_torch_mapping(
        normalization_record["path"],
        expected_sha256=normalization_record["sha256"],
        map_location="cpu",
        label="V2.1A normalization",
    )
    normalization = v21_gate.validate_normalization_authority(normalization_raw)
    actual_normalization = {
        "path": str(normalization_path),
        "sha256": normalization_sha,
    }
    diagnostics: dict[str, tuple[dict[str, Any], dict[str, str]]] = {}
    for role in ("best_raw_aux", "final"):
        diagnostics[role] = _open_diagnostic_state(
            role,
            raw["diagnostic_states"][role]["file"],
            normalization=normalization,
        )
    for role, (payload, record) in diagnostics.items():
        declared = raw["diagnostic_states"][role]
        if (
            record != declared["file"]
            or payload["epoch"] != declared["epoch"]
            or payload["model_state_dict_sha256"]
            != declared["model_state_dict_sha256"]
            or payload["normalization_authority"] != actual_normalization
            or payload["execution_authority"] != execution_record
            or any(payload[name] != expected_records[name] for name in expected_records)
            or payload["model_state_dict_sha256"]
            != raw["history"][payload["epoch"]]["model_state_dict_sha256"]
        ):
            raise ValueError(f"V2.1A {role} diagnostic chain differs")

    certificate_record = _validated_record(
        raw["certificate"], label="V2.1A certificate"
    )
    certificate, certificate_sha, certificate_path = load_json_object(
        certificate_record["path"],
        expected_sha256=certificate_record["sha256"],
        label="V2.1A certificate",
    )
    certificate_required = {
        "schema",
        "schema_version",
        "training_contract",
        "training_contract_sha256",
        "implementation",
        "objective_adapter",
        "loss_implementation",
        "preregistration",
        "execution_authority_addendum",
        "execution_authority",
        "cohort_authority",
        "pilot_cohort_region_view_registry",
        "pilot_cohort_region_view_registry_authority_sha256",
        "benchmark_exclusion_manifest",
        "input_records_by_split",
        "optimizer_steps_completed",
        "selected_epoch",
        "selected_validation",
        "model_state_dict_sha256",
        "normalization_authority",
        "normalization_content_authority_sha256",
        "diagnostic_states",
        "source_access",
        "benchmark_opened",
        "content_sha256",
    }
    if not isinstance(certificate, Mapping) or set(certificate) != certificate_required:
        raise ValueError("V2.1A certificate fields differ")
    content = dict(certificate)
    declared_content = content.pop("content_sha256", None)
    if (
        certificate.get("schema") != rescue.CERTIFICATE_SCHEMA
        or certificate.get("schema_version") != 1
        or certificate.get("training_contract") != rescue.training_contract()
        or certificate.get("training_contract_sha256")
        != rescue.TRAINING_CONTRACT_SHA256
        or canonical_json_sha256(content) != declared_content
        or certificate.get("optimizer_steps_completed") != rescue.OPTIMIZER_STEPS
        or certificate.get("execution_authority") != execution_record
        or certificate.get("normalization_authority") != actual_normalization
        or certificate.get("diagnostic_states") != raw["diagnostic_states"]
        or certificate.get("selected_epoch") != raw["selected_epoch"]
        or certificate.get("selected_validation") != raw["selected_validation"]
        or certificate.get("source_access") != rescue.source_access()
        or certificate.get("benchmark_opened") is not False
        or any(certificate.get(name) != expected_records[name] for name in expected_records)
    ):
        raise ValueError("V2.1A certificate identity differs")
    source_manifest, _teacher_manifest = rescue.pilot.pilot_shard.derive_pilot_global_manifests(
        registry
    )
    normalization_train = [
        {
            "scene_id": row["scene_id"],
            "training_shard": row["training_shard"],
            "adaptive_context": row["adaptive_context"],
        }
        for row in execution["source_train"]
    ]
    if (
        certificate.get("cohort_authority") != execution["cohort_authority"]
        or certificate.get("pilot_cohort_region_view_registry")
        != execution["pilot_cohort_region_view_registry"]
        or certificate.get("pilot_cohort_region_view_registry_authority_sha256")
        != registry["authority_sha256"]
        or certificate.get("benchmark_exclusion_manifest")
        != execution["benchmark_exclusion_manifest"]
        or certificate.get("input_records_by_split", {}).get("source_train")
        != execution["source_train"]
        or certificate.get("input_records_by_split", {}).get("source_validation")
        != execution["source_validation"]
        or normalization["train_input_records"] != normalization_train
        or normalization["source_state_cohort_authority_sha256"]
        != source_manifest["authority_sha256"]
        or certificate.get("normalization_content_authority_sha256")
        != rescue.pilot.pilot_normalization_authority_sha256(normalization)
    ):
        raise ValueError("V2.1A source lineage differs")

    checkpoint_record: dict[str, str] | None = None
    selected_epoch = raw["selected_epoch"]
    if selected_epoch is not None:
        checkpoint_record = _validated_record(
            raw["checkpoint"], label="V2.1A checkpoint"
        )
        checkpoint, checkpoint_sha, checkpoint_path = load_torch_mapping(
            checkpoint_record["path"],
            expected_sha256=checkpoint_record["sha256"],
            map_location="cpu",
            label="V2.1A checkpoint",
        )
        checkpoint_required = {
            "schema",
            "schema_version",
            "model_class",
            "model_architecture",
            "accepted_v2_authority",
            "model_state_dict",
            "model_state_dict_sha256",
            "normalization_authority",
            "certificate",
            "selected_epoch",
            "implementation",
            "objective_adapter",
            "loss_implementation",
            "preregistration",
            "execution_authority_addendum",
            "source_access",
        }
        if not isinstance(checkpoint, Mapping) or set(checkpoint) != checkpoint_required:
            raise ValueError("V2.1A checkpoint fields differ")
        state, state_sha, architecture = _validate_model_state(
            checkpoint.get("model_state_dict"), normalization
        )
        actual_certificate = {
            "path": str(certificate_path),
            "sha256": certificate_sha,
        }
        if (
            checkpoint.get("schema") != rescue.CHECKPOINT_SCHEMA
            or checkpoint.get("schema_version") != 1
            or checkpoint.get("model_class")
            != SurfaceRegionAcceptedV2TypedContextResidualV1.__name__
            or checkpoint.get("model_architecture") != architecture
            or checkpoint.get("accepted_v2_authority") != accepted_v2_authority()
            or checkpoint.get("model_state_dict_sha256") != state_sha
            or checkpoint.get("normalization_authority") != actual_normalization
            or checkpoint.get("certificate") != actual_certificate
            or checkpoint.get("selected_epoch") != selected_epoch
            or checkpoint.get("source_access") != rescue.source_access()
            or any(checkpoint.get(name) != expected_records[name] for name in expected_records)
            or state_sha
            != raw["history"][selected_epoch]["model_state_dict_sha256"]
            or certificate.get("model_state_dict_sha256") != state_sha
        ):
            raise ValueError("V2.1A checkpoint chain differs")
        checkpoint_record = {"path": str(checkpoint_path), "sha256": checkpoint_sha}
    elif certificate.get("model_state_dict_sha256") is not None:
        raise ValueError("V2.1A diagnostic certificate exposes a deployment state")
    if require_promotion and promotion["passed"] is not True:
        raise ValueError("V2.1A source promotion did not pass")
    return {
        "schema": PROMOTION_CHAIN_SCHEMA,
        "schema_version": 1,
        "source_result": {"path": str(result_path), "sha256": result_sha},
        "execution_authority": execution_record,
        "normalization_authority": actual_normalization,
        "certificate": {"path": str(certificate_path), "sha256": certificate_sha},
        "checkpoint": checkpoint_record,
        "diagnostic_states": raw["diagnostic_states"],
        "selected_epoch": selected_epoch,
        "promotion": promotion,
        "source_promotion_authorized": promotion["passed"] is True,
        "target_execution_authorized": False,
        "benchmark_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--allow-no-promotion", action="store_true")
    args = parser.parse_args()
    result = validate_source_rescue_chain(
        args.result,
        expected_sha256=args.expected_sha256,
        require_promotion=not args.allow_no_promotion,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "PROMOTION_CHAIN_SCHEMA",
    "PROMOTION_SCHEMA",
    "validate_source_promotion_evidence",
    "validate_source_rescue_chain",
]
