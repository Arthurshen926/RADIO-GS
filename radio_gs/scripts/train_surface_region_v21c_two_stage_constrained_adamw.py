#!/usr/bin/env python3
"""Run the source-only V2.1C Stage-I audit or authorized Stage-II training."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any

import torch

from radio_gs.interfaces import (
    surface_region_v21b_reliability_conditioned_residual as v21b_interface,
)
from radio_gs.interfaces.surface_region_typed_context_training import (
    accepted_v2_authority,
)
from radio_gs.optimization import adamw_two_constraint_projection_v21c as projection
from radio_gs.scripts import (
    train_surface_region_typed_context_residual as v1_trainer,
)
from radio_gs.scripts import (
    train_surface_region_v21b_conditioned_rank256_exact4x2 as v21b,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


ARTIFACT_TYPE = "surface_region_v21c_two_stage_constrained_adamw_source_pilot"
EXECUTION_AUTHORITY_SCHEMA = (
    "radio_gs.surface_region_v21c_two_stage_constrained_adamw_"
    "execution_authority.v1"
)
STAGE_I_RESULT_SCHEMA = "radio_gs.surface_region_v21c_gradient_conflict_audit.v1"
STAGE_II_STATE_ARCHIVE_SCHEMA = (
    "radio_gs.surface_region_v21c_constrained_adamw_state_archive.v1"
)
STAGE_II_CHECKPOINT_SCHEMA = (
    "radio_gs.surface_region_v21c_constrained_adamw_checkpoint.v1"
)
STAGE_II_CERTIFICATE_SCHEMA = (
    "radio_gs.surface_region_v21c_constrained_adamw_source_certificate.v1"
)
STAGE_II_RESULT_SCHEMA = (
    "radio_gs.surface_region_v21c_constrained_adamw_source_result.v1"
)
NORMALIZATION_SCHEMA = "radio_gs.surface_region_v21c_train4_normalization.v1"
STAGE_I = "stage_i_gradient_conflict_audit"
STAGE_II = "stage_ii_constrained_adamw"
OPTIMIZER_STEPS = 30
MINIMUM_CONFLICT_STEPS = 16
CONFLICT_DOT_TOLERANCE = 1e-12
MINIMUM_RELATIVE_IMPROVEMENT = 0.005
MINIMUM_ACTIVE_COVERAGE = 0.95
MINIMUM_PAIR_COVERAGE = 0.95
TRAIN_SCENES = v21b.TRAIN_SCENES
VALIDATION_SCENES = v21b.VALIDATION_SCENES
PREREGISTRATION = Path(
    "paper/artifacts/"
    "surface_region_v21c_two_stage_constrained_adamw_"
    "preregistration_20260807.json"
)
PREREGISTRATION_SHA256 = (
    "6b6d13e65619fd0a37bd46918f2ad405edb10ebcba683aa872d3d0f2cf36fda0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class V21CInputs:
    execution: dict[str, Any]
    parent: v21b.V21BInputs
    stage_i_result: dict[str, Any] | None


def source_access() -> dict[str, bool]:
    return {
        **v21b.source_access(),
        "stage_i_gradient_conflict_audit_opened": True,
        "stage_ii_projection_conditioned_on_frozen_stage_i": True,
        "target_query_or_benchmark_opened": False,
    }


def training_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "parent_v21b_training_contract_sha256": v21b.TRAINING_CONTRACT_SHA256,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "cohort": {
            "source_train": list(TRAIN_SCENES),
            "source_validation": list(VALIDATION_SCENES),
            "scene_and_physical_space_disjoint": True,
        },
        "stages": {
            "stage_i": {
                "optimizer_steps": OPTIMIZER_STEPS,
                "parameter_subset": projection.PARAMETER_SUBSET_SELECTION,
                "gradient_order": list(projection.GRADIENT_ORDER),
                "conflict_dot_tolerance": CONFLICT_DOT_TOLERANCE,
                "minimum_conflict_steps": MINIMUM_CONFLICT_STEPS,
                "projection_authorized": False,
            },
            "stage_ii": {
                "optimizer_steps": OPTIMIZER_STEPS,
                "early_stopping": False,
                "trigger": "frozen_stage_i_strict_majority_conflict",
                "projection": (
                    "min_norm_to_adamw_candidate_subject_to_nonnegative_"
                    "absolute_and_pairwise_gradient_dot"
                ),
                "adamw_moments": "advanced_unprojected",
                "decoupled_weight_decay": "included_in_projected_candidate",
            },
        },
        "promotion": {
            "minimum_relative_improvement": MINIMUM_RELATIVE_IMPROVEMENT,
            "macro_metrics": ["auxiliary", "absolute", "pairwise"],
            "absolute_every_validation_scene_non_regression": True,
            "pairwise_every_validation_scene_non_regression": True,
            "v1_non_regression": True,
            "active_row_coverage_minimum": MINIMUM_ACTIVE_COVERAGE,
            "pair_endpoint_coverage_minimum": MINIMUM_PAIR_COVERAGE,
            "selection": ["minimum_auxiliary_loss", "earliest_step"],
        },
        "source_access": source_access(),
    }


TRAINING_CONTRACT_SHA256 = canonical_json_sha256(training_contract())


_CODE_RECORD_FIELDS = (
    "trainer_implementation",
    "execution_builder_implementation",
    "projection_implementation",
    "source_gate_implementation",
    "preregistration",
)


def _expected_code_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[2]
    return {
        "trainer_implementation": Path(__file__).resolve(),
        "execution_builder_implementation": root
        / "radio_gs/scripts/build_surface_region_v21c_execution_authority.py",
        "projection_implementation": Path(projection.__file__).resolve(),
        "source_gate_implementation": root
        / "radio_gs/interfaces/surface_region_v21c_source_gate.py",
        "preregistration": root / PREREGISTRATION,
    }


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"V2.1C {label} file record differs")
    path = str(value["path"])
    digest = str(value["sha256"])
    if not path.startswith("/") or _SHA256.fullmatch(digest) is None:
        raise ValueError(f"V2.1C {label} file record differs")
    return {"path": path, "sha256": digest}


def validate_execution_authority(value: object) -> dict[str, Any]:
    required = {
        "schema",
        "schema_version",
        "status",
        "stage",
        *_CODE_RECORD_FIELDS,
        "parent_v21b_execution_authority",
        "stage_i_audit_result",
        "training_authorized",
        "projection_authorized",
        "benchmark_execution_authorized",
        "source_access",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("V2.1C execution authority fields differ")
    authority = dict(value)
    stage = authority["stage"]
    if stage not in (STAGE_I, STAGE_II):
        raise ValueError("V2.1C execution stage differs")
    expected_status = (
        "authorized_source_only_stage_i_audit"
        if stage == STAGE_I
        else "authorized_source_only_stage_ii_after_positive_audit"
    )
    if (
        authority["schema"] != EXECUTION_AUTHORITY_SCHEMA
        or authority["schema_version"] != 1
        or authority["status"] != expected_status
        or authority["training_authorized"] is not True
        or authority["projection_authorized"] is not (stage == STAGE_II)
        or authority["benchmark_execution_authorized"] is not False
        or authority["source_access"] != source_access()
    ):
        raise ValueError("V2.1C execution authority header differs")
    for name in (*_CODE_RECORD_FIELDS, "parent_v21b_execution_authority"):
        authority[name] = _record(authority[name], label=name)
    if stage == STAGE_I:
        if authority["stage_i_audit_result"] is not None:
            raise ValueError("V2.1C Stage-I authority cannot bind an audit result")
    else:
        authority["stage_i_audit_result"] = _record(
            authority["stage_i_audit_result"], label="stage_i_audit_result"
        )
    return authority


def _finite_tree(value: object, *, label: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, Mapping):
        for name, item in value.items():
            _finite_tree(item, label=f"{label}.{name}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _finite_tree(item, label=f"{label}[{index}]")
        return
    if isinstance(value, int):
        return
    if not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")


def validate_stage_i_audit_result(value: object) -> dict[str, Any]:
    required = {
        "schema",
        "schema_version",
        "status",
        "training_contract_sha256",
        "execution_authority",
        "parent_v21b_execution_authority",
        "parameter_subset",
        "optimizer",
        "history",
        "trigger",
        "source_access",
        "benchmark_opened",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("V2.1C Stage-I audit result fields differ")
    result = dict(value)
    if (
        result["schema"] != STAGE_I_RESULT_SCHEMA
        or result["schema_version"] != 1
        or result["status"] != "source_only_stage_i_gradient_conflict_audit_complete"
        or result["training_contract_sha256"] != TRAINING_CONTRACT_SHA256
        or result["source_access"] != source_access()
        or result["benchmark_opened"] is not False
    ):
        raise ValueError("V2.1C Stage-I audit result header differs")
    result["execution_authority"] = _record(
        result["execution_authority"], label="Stage-I execution authority"
    )
    result["parent_v21b_execution_authority"] = _record(
        result["parent_v21b_execution_authority"], label="parent V2.1B authority"
    )
    subset = result["parameter_subset"]
    if (
        not isinstance(subset, Mapping)
        or subset.get("selection") != projection.PARAMETER_SUBSET_SELECTION
        or not isinstance(subset.get("parameter_records"), list)
        or canonical_json_sha256(subset["parameter_records"])
        != subset.get("parameter_records_sha256")
        or int(subset.get("parameter_count", 0)) != len(subset["parameter_records"])
        or int(subset.get("vector_numel", 0))
        != sum(int(item["numel"]) for item in subset["parameter_records"])
    ):
        raise ValueError("V2.1C Stage-I parameter subset differs")
    optimizer = result["optimizer"]
    if not isinstance(optimizer, Mapping) or optimizer != {
        "name": "AdamW",
        "learning_rate": v1_trainer.LEARNING_RATE,
        "weight_decay": v1_trainer.WEIGHT_DECAY,
        "maximum_gradient_norm": v1_trainer.MAX_GRADIENT_NORM,
        "ordinary_candidate_applied": True,
        "projection_applied": False,
    }:
        raise ValueError("V2.1C Stage-I optimizer declaration differs")
    history = result["history"]
    if (
        not isinstance(history, list)
        or len(history) != OPTIMIZER_STEPS
        or [row.get("step") for row in history] != list(
            range(1, OPTIMIZER_STEPS + 1)
        )
    ):
        raise ValueError("V2.1C Stage-I history must contain exactly 30 steps")
    conflict_steps = []
    for row in history:
        if not isinstance(row, Mapping) or set(row) != {
            "step",
            "training",
            "validation",
            "model_state_dict_sha256",
        }:
            raise ValueError("V2.1C Stage-I history row fields differ")
        training = row["training"]
        if not isinstance(training, Mapping) or set(training) != {
            "scene_count",
            "equal_scene_weight",
            "per_scene",
            "gradient_evidence",
            "adamw_candidate_evidence",
        }:
            raise ValueError("V2.1C Stage-I training evidence fields differ")
        evidence = training["gradient_evidence"]
        if (
            not isinstance(evidence, Mapping)
            or evidence.get("gradient_order") != list(projection.GRADIENT_ORDER)
            or len(evidence.get("gram", [])) != 3
            or len(evidence.get("cosine", [])) != 3
        ):
            raise ValueError("V2.1C Stage-I gradient evidence differs")
        candidate = training["adamw_candidate_evidence"]
        if not isinstance(candidate, Mapping) or not isinstance(
            candidate.get("constraint_conflict"), bool
        ):
            raise ValueError("V2.1C Stage-I candidate evidence differs")
        if candidate["constraint_conflict"]:
            conflict_steps.append(int(row["step"]))
        _finite_tree(training, label=f"Stage-I step {row['step']}")
        if _SHA256.fullmatch(str(row["model_state_dict_sha256"])) is None:
            raise ValueError("V2.1C Stage-I model-state digest differs")
    trigger = result["trigger"]
    expected_trigger = {
        "audited_steps": OPTIMIZER_STEPS,
        "minimum_conflict_steps": MINIMUM_CONFLICT_STEPS,
        "conflict_steps": conflict_steps,
        "conflict_step_count": len(conflict_steps),
        "strict_majority_conflict_confirmed": (
            len(conflict_steps) >= MINIMUM_CONFLICT_STEPS
        ),
        "stage_ii_authorized": len(conflict_steps) >= MINIMUM_CONFLICT_STEPS,
    }
    if trigger != expected_trigger:
        raise ValueError("V2.1C Stage-I trigger differs")
    return result


def prepare_inputs(path: str | Path, *, expected_sha256: str) -> V21CInputs:
    raw, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="V2.1C execution authority",
    )
    authority = validate_execution_authority(raw)
    for name, expected in _expected_code_paths().items():
        observed = validate_file_record(authority[name], label=f"V2.1C {name}")
        if observed != expected.resolve():
            raise ValueError(f"V2.1C authority binds another {name}")
    if authority["preregistration"]["sha256"] != PREREGISTRATION_SHA256:
        raise ValueError("V2.1C preregistration digest differs")
    parent_record = authority["parent_v21b_execution_authority"]
    parent = v21b.prepare_inputs(
        validate_file_record(parent_record, label="V2.1C parent V2.1B authority"),
        expected_sha256=parent_record["sha256"],
    )
    audit: dict[str, Any] | None = None
    if authority["stage"] == STAGE_II:
        audit_record = authority["stage_i_audit_result"]
        audit_raw, _audit_digest, _audit_path = load_json_object(
            validate_file_record(audit_record, label="V2.1C Stage-I audit result"),
            expected_sha256=audit_record["sha256"],
            label="V2.1C Stage-I audit result",
        )
        audit = validate_stage_i_audit_result(audit_raw)
        if audit["trigger"]["stage_ii_authorized"] is not True:
            raise ValueError("V2.1C Stage-II is not authorized by Stage-I")
        if audit["parent_v21b_execution_authority"] != parent_record:
            raise ValueError("V2.1C Stage-I and Stage-II parent authority differs")
    authority["verified_path"] = str(source)
    authority["verified_sha256"] = digest
    return V21CInputs(execution=authority, parent=parent, stage_i_result=audit)


def build_normalization(inputs: V21CInputs) -> dict[str, Any]:
    result = v21b.build_pilot_normalization(
        [item.base for item in inputs.parent.train],
        source_state_cohort_authority_sha256=inputs.parent.source_state_manifest[
            "authority_sha256"
        ],
    )
    result["schema"] = NORMALIZATION_SCHEMA
    result["source_access"] = source_access()
    return result


def _initialize(
    inputs: V21CInputs, device: torch.device
) -> tuple[dict[str, Any], torch.nn.Module, torch.optim.AdamW]:
    normalization = build_normalization(inputs)
    torch.manual_seed(v1_trainer.SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(v1_trainer.SEED)
    model = v21b_interface.build_model_from_source_normalization(normalization).to(
        device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=v1_trainer.LEARNING_RATE,
        weight_decay=v1_trainer.WEIGHT_DECAY,
    )
    return normalization, model, optimizer


def _candidate_evidence(
    candidate: torch.Tensor,
    gradients: Mapping[str, torch.Tensor],
    *,
    reconstruction: float,
) -> dict[str, Any]:
    candidate64 = candidate.detach().double()
    dots = {
        name: float(gradients[name].detach().double() @ candidate64)
        for name in ("absolute", "pairwise")
    }
    norm = float(torch.linalg.vector_norm(candidate64))
    cosine = {}
    for name in ("absolute", "pairwise"):
        gradient_norm = float(
            torch.linalg.vector_norm(gradients[name].detach().double())
        )
        cosine[name] = (
            dots[name] / (norm * gradient_norm)
            if norm > 0 and gradient_norm > 0
            else 0.0
        )
    return {
        "descent_displacement_convention": "theta_before_minus_theta_after",
        "candidate_norm": norm,
        "gradient_dot_candidate": dots,
        "gradient_cosine_candidate": cosine,
        "constraint_conflict": any(
            value < -CONFLICT_DOT_TOLERANCE for value in dots.values()
        ),
        "conflict_dot_tolerance": CONFLICT_DOT_TOLERANCE,
        "adamw_candidate_reconstruction_max_abs_error": reconstruction,
        "adamw_moments_advanced": True,
        "decoupled_weight_decay_included": True,
    }


def _training_gradients(
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    inputs: V21CInputs,
    normalization: Mapping[str, Any],
    device: torch.device,
) -> tuple[
    tuple[tuple[str, torch.nn.Parameter], ...],
    dict[str, torch.Tensor],
    dict[str, dict[str, float | int | bool]],
]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    named = projection.trainable_named_parameters(model)
    fit = inputs.parent.fit.embeddings.to(device)
    accumulated: dict[str, torch.Tensor] | None = None
    per_scene: dict[str, dict[str, float | int | bool]] = {}
    scene_count = len(inputs.parent.train)
    for binding in inputs.parent.train:
        total, metrics = v21b._objective(
            model,
            binding,
            normalization,
            fit,
            inputs.parent.canonical_negative,
            inputs.parent.compositional,
            inputs.parent.relations,
            device,
        )
        objectives = {
            "combined": total / scene_count,
            "absolute": metrics["response_absolute_relevance_loss"] / scene_count,
            "pairwise": metrics[
                "response_continuous_pairwise_relevance_loss"
            ]
            / scene_count,
        }
        scene_gradients = {
            name: projection.objective_gradient(
                objective, named, retain_graph=True
            )
            for name, objective in objectives.items()
        }
        if accumulated is None:
            accumulated = {
                name: value.detach().clone()
                for name, value in scene_gradients.items()
            }
        else:
            for name in projection.GRADIENT_ORDER:
                accumulated[name].add_(scene_gradients[name].detach())
        objectives["combined"].backward()
        per_scene[binding.scene_id] = {
            "combined_objective": float(total.detach().cpu()),
            **v21b.v2_trainer._float_metrics(metrics),
        }
    if accumulated is None:
        raise RuntimeError("V2.1C source training cohort is empty")
    torch.nn.utils.clip_grad_norm_(
        tuple(parameter for _, parameter in named),
        v1_trainer.MAX_GRADIENT_NORM,
        error_if_nonfinite=True,
    )
    return named, accumulated, per_scene


def train_one_stage_i_step(
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    inputs: V21CInputs,
    normalization: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    named, gradients, per_scene = _training_gradients(
        model, optimizer, inputs, normalization, device
    )
    before = projection.flatten_parameter_values(named).clone()
    predicted = projection.predict_adamw_descent_direction(optimizer, named)
    optimizer.step()
    actual = before - projection.flatten_parameter_values(named)
    reconstruction = float((predicted - actual).abs().max().detach().cpu())
    return {
        "scene_count": len(inputs.parent.train),
        "equal_scene_weight": 1.0 / len(inputs.parent.train),
        "per_scene": per_scene,
        "gradient_evidence": projection.gradient_geometry(gradients),
        "adamw_candidate_evidence": _candidate_evidence(
            actual, gradients, reconstruction=reconstruction
        ),
    }


def train_one_stage_ii_step(
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    inputs: V21CInputs,
    normalization: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    named, gradients, per_scene = _training_gradients(
        model, optimizer, inputs, normalization, device
    )
    geometry = projection.gradient_geometry(gradients)
    projected = projection.commit_projected_adamw_step(
        optimizer,
        named,
        gradients["absolute"],
        gradients["pairwise"],
    )
    # The committed update must meet the exact two protected first-order
    # constraints; KKT evidence is retained in every history row.
    if projected["kkt"]["passed"] is not True:
        raise RuntimeError("V2.1C committed projection lacks a KKT certificate")
    return {
        "scene_count": len(inputs.parent.train),
        "equal_scene_weight": 1.0 / len(inputs.parent.train),
        "per_scene": per_scene,
        "gradient_evidence": geometry,
        "projected_adamw_evidence": projected,
    }


def evaluate(
    model: torch.nn.Module,
    inputs: V21CInputs,
    normalization: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return v21b.evaluate(model, inputs.parent, normalization, device)


def promotion_checks(
    validation: Mapping[str, Any], epoch_zero: Mapping[str, Any]
) -> dict[str, Any]:
    parent = v21b.promotion_checks(validation, epoch_zero)
    checks = dict(parent["checks"])
    response = validation["response_listwise_v21b"]
    baseline = epoch_zero["response_listwise_v21b"]
    checks["pairwise_every_scene_non_regression"] = all(
        float(
            response["per_scene"][scene][
                "response_continuous_pairwise_relevance_loss"
            ]
        )
        <= float(
            baseline["per_scene"][scene][
                "response_continuous_pairwise_relevance_loss"
            ]
        )
        for scene in VALIDATION_SCENES
    )
    return {
        "relative_improvement_from_step_zero": dict(
            parent["relative_improvement_from_step_zero"]
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }


def attach_promotion(
    validation: Mapping[str, Any], epoch_zero: Mapping[str, Any]
) -> dict[str, Any]:
    result = dict(validation)
    result["promotion"] = promotion_checks(result, epoch_zero)
    result["selection_eligible"] = bool(result["promotion"]["passed"])
    return result


def select_promotion_step(history: Sequence[Mapping[str, Any]]) -> int | None:
    if not history or [int(row.get("step", -1)) for row in history] != list(
        range(len(history))
    ):
        raise ValueError("V2.1C history must be contiguous from step zero")
    eligible = [
        row
        for row in history
        if row.get("validation", {}).get("selection_eligible") is True
    ]
    if not eligible:
        return None
    return int(
        min(
            eligible,
            key=lambda row: (
                float(
                    row["validation"]["response_listwise_v21b"][
                        "scene_macro_auxiliary_loss"
                    ]
                ),
                int(row["step"]),
            ),
        )["step"]
    )


def _authority_record(inputs: V21CInputs) -> dict[str, str]:
    return {
        "path": inputs.execution["verified_path"],
        "sha256": inputs.execution["verified_sha256"],
    }


def run_stage_i(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"V2.1C Stage-I result exists: {output}")
    inputs = prepare_inputs(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
    )
    if inputs.execution["stage"] != STAGE_I:
        raise ValueError("V2.1C Stage-I command requires a Stage-I authority")
    device = torch.device(str(args.device))
    normalization, model, optimizer = _initialize(inputs, device)
    named = projection.trainable_named_parameters(model)
    subset = projection.parameter_subset_manifest(named)
    history: list[dict[str, Any]] = []
    for step in range(1, OPTIMIZER_STEPS + 1):
        training = train_one_stage_i_step(
            model, optimizer, inputs, normalization, device
        )
        validation = evaluate(model, inputs, normalization, device)
        row = {
            "step": step,
            "training": training,
            "validation": validation,
            "model_state_dict_sha256": v21b._state_sha(v21b._state_copy(model)),
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    conflict_steps = [
        int(row["step"])
        for row in history
        if row["training"]["adamw_candidate_evidence"]["constraint_conflict"]
        is True
    ]
    confirmed = len(conflict_steps) >= MINIMUM_CONFLICT_STEPS
    result = {
        "schema": STAGE_I_RESULT_SCHEMA,
        "schema_version": 1,
        "status": "source_only_stage_i_gradient_conflict_audit_complete",
        "training_contract_sha256": TRAINING_CONTRACT_SHA256,
        "execution_authority": _authority_record(inputs),
        "parent_v21b_execution_authority": dict(
            inputs.execution["parent_v21b_execution_authority"]
        ),
        "parameter_subset": subset,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": v1_trainer.LEARNING_RATE,
            "weight_decay": v1_trainer.WEIGHT_DECAY,
            "maximum_gradient_norm": v1_trainer.MAX_GRADIENT_NORM,
            "ordinary_candidate_applied": True,
            "projection_applied": False,
        },
        "history": history,
        "trigger": {
            "audited_steps": OPTIMIZER_STEPS,
            "minimum_conflict_steps": MINIMUM_CONFLICT_STEPS,
            "conflict_steps": conflict_steps,
            "conflict_step_count": len(conflict_steps),
            "strict_majority_conflict_confirmed": confirmed,
            "stage_ii_authorized": confirmed,
        },
        "source_access": source_access(),
        "benchmark_opened": False,
    }
    validate_stage_i_audit_result(result)
    write_frozen_json(output, result)
    return result


def _stage_ii_output_paths(output: Path) -> dict[str, Path]:
    return {
        "checkpoint": output,
        "normalization": output.with_suffix(output.suffix + ".normalization.pt"),
        "state_archive": output.with_suffix(output.suffix + ".epoch_states.pt"),
        "certificate": output.with_suffix(output.suffix + ".certificate.json"),
        "result": output.with_suffix(output.suffix + ".json"),
    }


def _write_stage_ii_outputs(
    output: Path,
    normalization: Mapping[str, Any],
    inputs: V21CInputs,
    history: Sequence[Mapping[str, Any]],
    saved_states: Mapping[int, Mapping[str, torch.Tensor]],
    selected_step: int | None,
) -> dict[str, Any]:
    paths = _stage_ii_output_paths(output)
    existing = [path for path in paths.values() if path.exists() or path.is_symlink()]
    if existing:
        raise FileExistsError(
            "V2.1C Stage-II first-writer outputs already exist: "
            + ", ".join(str(path) for path in existing)
        )
    normalization_path = write_torch_noclobber(paths["normalization"], normalization)
    normalization_record = file_record(normalization_path)
    eligible_steps = [
        int(row["step"])
        for row in history
        if row["validation"]["selection_eligible"] is True
    ]
    expected_saved = [0, *eligible_steps]
    if list(saved_states) != expected_saved:
        raise RuntimeError("V2.1C saved-state coverage differs")
    state_hashes = {
        str(step): v21b._state_sha(saved_states[step]) for step in saved_states
    }
    archive = {
        "schema": STAGE_II_STATE_ARCHIVE_SCHEMA,
        "schema_version": 1,
        "training_contract_sha256": TRAINING_CONTRACT_SHA256,
        "execution_authority": _authority_record(inputs),
        "stage_i_audit_result": dict(inputs.execution["stage_i_audit_result"]),
        "normalization_authority": normalization_record,
        "saved_steps": expected_saved,
        "promotion_eligible_steps": eligible_steps,
        "model_state_dict_sha256_by_step": state_hashes,
        "model_state_dict_by_step": {
            str(step): dict(saved_states[step]) for step in saved_states
        },
        "source_access": source_access(),
    }
    archive_path = write_torch_noclobber(paths["state_archive"], archive)
    archive_record = file_record(archive_path)

    checkpoint_record: dict[str, str] | None = None
    certificate_record: dict[str, str] | None = None
    selected_validation: Mapping[str, Any] | None = None
    if selected_step is not None:
        selected_state = saved_states[selected_step]
        selected_sha = state_hashes[str(selected_step)]
        selected_validation = history[selected_step]["validation"]
        certificate = {
            "schema": STAGE_II_CERTIFICATE_SCHEMA,
            "schema_version": 1,
            "training_contract": training_contract(),
            "training_contract_sha256": TRAINING_CONTRACT_SHA256,
            "execution_authority": _authority_record(inputs),
            "stage_i_audit_result": dict(inputs.execution["stage_i_audit_result"]),
            "selected_step": selected_step,
            "selected_validation": dict(selected_validation),
            "model_state_dict_sha256": selected_sha,
            "normalization_authority": normalization_record,
            "state_archive": archive_record,
            "source_access": source_access(),
            "benchmark_opened": False,
        }
        certificate["content_sha256"] = canonical_json_sha256(certificate)
        certificate_record = file_record(
            write_frozen_json(paths["certificate"], certificate)
        )
        reconstructed = v21b_interface.build_model_from_source_normalization(
            normalization
        )
        reconstructed.load_state_dict(selected_state, strict=True)
        checkpoint = {
            "schema": STAGE_II_CHECKPOINT_SCHEMA,
            "schema_version": 1,
            "model_class": type(reconstructed).__name__,
            "model_architecture": reconstructed.architecture(),
            "accepted_v2_authority": accepted_v2_authority(),
            "model_state_dict": dict(selected_state),
            "model_state_dict_sha256": selected_sha,
            "normalization_authority": normalization_record,
            "certificate": certificate_record,
            "state_archive": archive_record,
            "selected_step": selected_step,
            "source_access": source_access(),
        }
        checkpoint_record = file_record(
            write_torch_noclobber(paths["checkpoint"], checkpoint)
        )
    passed = selected_step is not None
    result = {
        "schema": STAGE_II_RESULT_SCHEMA,
        "schema_version": 1,
        "status": (
            "source_only_v21c_promotion_candidate_complete"
            if passed
            else "source_only_v21c_complete_no_eligible_promotion"
        ),
        "training_contract": training_contract(),
        "training_contract_sha256": TRAINING_CONTRACT_SHA256,
        "execution_authority": _authority_record(inputs),
        "stage_i_audit_result": dict(inputs.execution["stage_i_audit_result"]),
        "normalization_authority": normalization_record,
        "state_archive": archive_record,
        "checkpoint": checkpoint_record,
        "certificate": certificate_record,
        "selected_step": selected_step,
        "selected_validation": (
            dict(selected_validation) if selected_validation is not None else None
        ),
        "promotion_candidate_available": passed,
        "history": list(history),
        "source_access": source_access(),
        "benchmark_opened": False,
    }
    write_frozen_json(paths["result"], result)
    return result


def run_stage_ii(args: argparse.Namespace) -> dict[str, Any]:
    inputs = prepare_inputs(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
    )
    if inputs.execution["stage"] != STAGE_II or inputs.stage_i_result is None:
        raise ValueError("V2.1C Stage-II command requires a triggered authority")
    device = torch.device(str(args.device))
    normalization, model, optimizer = _initialize(inputs, device)
    subset = projection.parameter_subset_manifest(
        projection.trainable_named_parameters(model)
    )
    if subset != inputs.stage_i_result["parameter_subset"]:
        raise ValueError("V2.1C Stage-II parameter subset differs from Stage-I")
    epoch_zero_raw = evaluate(model, inputs, normalization, device)
    epoch_zero = attach_promotion(epoch_zero_raw, epoch_zero_raw)
    zero_state = v21b._state_copy(model)
    history: list[dict[str, Any]] = [
        {
            "step": 0,
            "training": None,
            "validation": epoch_zero,
            "model_state_dict_sha256": v21b._state_sha(zero_state),
        }
    ]
    saved_states: dict[int, dict[str, torch.Tensor]] = {0: zero_state}
    for step in range(1, OPTIMIZER_STEPS + 1):
        training = train_one_stage_ii_step(
            model, optimizer, inputs, normalization, device
        )
        validation = attach_promotion(
            evaluate(model, inputs, normalization, device), epoch_zero
        )
        state = v21b._state_copy(model)
        row = {
            "step": step,
            "training": training,
            "validation": validation,
            "model_state_dict_sha256": v21b._state_sha(state),
        }
        history.append(row)
        if validation["selection_eligible"] is True:
            saved_states[step] = state
        print(json.dumps(row, sort_keys=True), flush=True)
    if len(history) != OPTIMIZER_STEPS + 1:
        raise RuntimeError("V2.1C Stage-II did not complete exactly 30 steps")
    selected_step = select_promotion_step(history)
    if selected_step is not None:
        model.load_state_dict(saved_states[selected_step], strict=True)
        restored = attach_promotion(
            evaluate(model, inputs, normalization, device), epoch_zero
        )
        if restored != history[selected_step]["validation"]:
            raise RuntimeError("V2.1C restored selected validation differs")
    model.cpu()
    return _write_stage_ii_outputs(
        Path(args.output).expanduser().resolve(),
        normalization,
        inputs,
        history,
        saved_states,
        selected_step,
    )


def synthetic_dry_run() -> dict[str, Any]:
    candidate = torch.tensor([-1.0, -1.0])
    absolute = torch.tensor([1.0, 0.0])
    pairwise = torch.tensor([0.0, 1.0])
    projected, evidence = projection.project_two_halfspaces(
        candidate, absolute, pairwise
    )
    return {
        "schema": "radio_gs.surface_region_v21c_synthetic_dry_run.v1",
        "projected": projected.tolist(),
        "kkt": evidence["kkt"],
        "minimum_conflict_steps": MINIMUM_CONFLICT_STEPS,
        "optimizer_steps": OPTIMIZER_STEPS,
        "stage_ii_direct_execution_authorized": False,
        "benchmark_opened": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("synthetic-dry-run")
    validate = commands.add_parser("validate-authority")
    validate.add_argument("--execution-authority", required=True)
    validate.add_argument("--expected-execution-authority-sha256", required=True)
    for name in ("audit-stage-i", "train-stage-ii"):
        run = commands.add_parser(name)
        run.add_argument("--execution-authority", required=True)
        run.add_argument("--expected-execution-authority-sha256", required=True)
        run.add_argument("--output", required=True)
        run.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "synthetic-dry-run":
        result = synthetic_dry_run()
    elif args.command == "validate-authority":
        inputs = prepare_inputs(
            args.execution_authority,
            expected_sha256=args.expected_execution_authority_sha256,
        )
        result = {
            "status": "source_only_v21c_authority_validated",
            "stage": inputs.execution["stage"],
            "source_train": [item.scene_id for item in inputs.parent.train],
            "source_validation": [
                item.scene_id for item in inputs.parent.validation
            ],
            "stage_ii_triggered": inputs.stage_i_result is not None,
            "benchmark_opened": False,
        }
    elif args.command == "audit-stage-i":
        result = run_stage_i(args)
    else:
        result = run_stage_ii(args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "EXECUTION_AUTHORITY_SCHEMA",
    "MINIMUM_CONFLICT_STEPS",
    "OPTIMIZER_STEPS",
    "PREREGISTRATION_SHA256",
    "STAGE_I",
    "STAGE_II",
    "STAGE_I_RESULT_SCHEMA",
    "STAGE_II_RESULT_SCHEMA",
    "TRAINING_CONTRACT_SHA256",
    "attach_promotion",
    "build_parser",
    "prepare_inputs",
    "promotion_checks",
    "run_stage_i",
    "run_stage_ii",
    "select_promotion_step",
    "source_access",
    "synthetic_dry_run",
    "train_one_stage_i_step",
    "train_one_stage_ii_step",
    "training_contract",
    "validate_execution_authority",
    "validate_stage_i_audit_result",
]
