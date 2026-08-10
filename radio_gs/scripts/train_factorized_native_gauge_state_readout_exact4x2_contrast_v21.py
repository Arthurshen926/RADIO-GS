#!/usr/bin/env python3
"""Contrast V2.1 with a variance-qualified pair-correlation comparator.

V2 and its frozen result remain immutable.  V2.1 changes only the source gate:
a baseline pair correlation is a valid non-regression comparator only when the
baseline itself passes the fixed variance/spread requirement.  A collapsed
baseline instead requires absolute correlation, strictly improved pair MAE,
and every existing anti-collapse and visual-probe check.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.interfaces import factorized_native_gauge_state_readout as interface
from radio_gs.models import source_contrast_preservation as contrast
from radio_gs.models.factorized_native_gauge_state_readout import DIRECTION_ONLY
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.scripts import (
    train_factorized_native_gauge_state_readout_exact4x2 as legacy,
)
from radio_gs.scripts import (
    train_factorized_native_gauge_state_readout_exact4x2_contrast_v2 as v2,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


EXECUTION_AUTHORITY_SCHEMA = (
    "radio_gs.factorized_native_gauge_state_readout_exact4x2_"
    "contrast_v21_execution_authority.v1"
)
CHECKPOINT_SCHEMA = (
    "radio_gs.factorized_native_gauge_state_readout_contrast_checkpoint.v2_1"
)
RESULT_SCHEMA = (
    "radio_gs.factorized_native_gauge_state_readout_contrast_source_result.v2_1"
)
TRAIN_SCENES = v2.TRAIN_SCENES
VALIDATION_SCENES = v2.VALIDATION_SCENES
OPTIMIZER_STEPS = v2.OPTIMIZER_STEPS
EVALUATION_INTERVAL = v2.EVALUATION_INTERVAL
PAIR_MAE_STRICT_IMPROVEMENT_TOLERANCE = v2.RAW_NON_REGRESSION_TOLERANCE


@dataclass(frozen=True)
class PreparedInputs:
    authority: dict[str, Any]
    base_v2: v2.PreparedInputs


def source_access() -> dict[str, bool]:
    return dict(v2.source_access())


def training_contract() -> dict[str, Any]:
    contract = deepcopy(v2.training_contract())
    contract["schema_version"] = 21
    contract["artifact_type"] = (
        "factorized_native_contrast_preserving_source_only_v2_1"
    )
    contract["contrast_v2_modified"] = False
    promotion = contract["promotion"]
    promotion.pop(
        "every_scene_pair_gram_mae_and_correlation_non_regression", None
    )
    promotion["pair_geometry_baseline_rule"] = {
        "variance_qualified_baseline": {
            "definition": (
                "baseline_student_to_teacher_spread_ratio_at_least_0p75"
            ),
            "requirements": [
                "absolute_pair_correlation_at_least_0p20",
                "pair_correlation_non_regression",
                "pair_mae_non_regression",
            ],
        },
        "variance_unqualified_baseline": {
            "requirements": [
                "absolute_pair_correlation_at_least_0p20",
                "pair_mae_strict_improvement",
                "all_existing_raw_residual_spread_and_visual_probe_gates",
            ],
            "correlation_non_regression": "not_a_valid_comparator",
        },
        "fixed_global_thresholds_only": True,
    }
    return contract


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    path = str(value["path"])
    digest = legacy.pilot_shard._require_sha256(value["sha256"], label=label)
    if not Path(path).is_absolute():
        raise ValueError(f"{label} file record differs")
    return {"path": path, "sha256": digest}


def validate_execution_authority(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("contrast V2.1 execution authority must be a mapping")
    authority = dict(value)
    required = {
        "schema",
        "schema_version",
        "status",
        "trainer_implementation",
        "base_v2_execution_authority",
        "training_contract_sha256",
        "authorized_arm",
        "training_authorized",
        "benchmark_execution_authorized",
        "source_access",
    }
    if (
        set(authority) != required
        or authority.get("schema") != EXECUTION_AUTHORITY_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("status")
        != "authorized_source_only_contrast_v21_exact4train_2validation"
        or authority.get("training_contract_sha256")
        != canonical_json_sha256(training_contract())
        or authority.get("authorized_arm") != DIRECTION_ONLY
        or authority.get("training_authorized") is not True
        or authority.get("benchmark_execution_authorized") is not False
        or authority.get("source_access") != source_access()
    ):
        raise ValueError("contrast V2.1 execution authority header differs")
    for name in ("trainer_implementation", "base_v2_execution_authority"):
        authority[name] = _record(authority[name], label=name)
    return authority


def prepare_inputs(path: str | Path, *, expected_sha256: str) -> PreparedInputs:
    raw, digest, source_path = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="factorized-native contrast V2.1 execution authority",
    )
    authority = validate_execution_authority(raw)
    observed = validate_file_record(
        authority["trainer_implementation"], label="V2.1 trainer implementation"
    )
    if observed != Path(__file__).resolve():
        raise ValueError("contrast V2.1 authority binds another trainer")
    base = authority["base_v2_execution_authority"]
    base_v2 = v2.prepare_inputs(base["path"], expected_sha256=base["sha256"])
    authority["verified_path"] = str(source_path)
    authority["verified_sha256"] = digest
    return PreparedInputs(authority=authority, base_v2=base_v2)


def _conditional_pair_geometry(
    validation: Mapping[str, Any], baseline: Mapping[str, Any]
) -> tuple[bool, dict[str, Any]]:
    per_scene: dict[str, Any] = {}
    for scene in VALIDATION_SCENES:
        current = validation["per_scene"][scene]
        reference = baseline["per_scene"][scene]
        baseline_spread = float(reference["student_to_teacher_spread_ratio"])
        baseline_qualified = baseline_spread >= contrast.MINIMUM_SPREAD_RATIO
        current_mae = float(current["teacher_centered_pair_gram_mae"])
        baseline_mae = float(reference["teacher_centered_pair_gram_mae"])
        current_correlation = float(
            current["teacher_centered_pair_gram_correlation"]
        )
        baseline_correlation = float(
            reference["teacher_centered_pair_gram_correlation"]
        )
        absolute_correlation = (
            current_correlation >= v2.MINIMUM_PAIR_GRAM_CORRELATION
        )
        if baseline_qualified:
            comparator = "variance_qualified_non_regression"
            mae_check = (
                current_mae
                <= baseline_mae + v2.RAW_NON_REGRESSION_TOLERANCE
            )
            correlation_check = (
                current_correlation + v2.RAW_NON_REGRESSION_TOLERANCE
                >= baseline_correlation
            )
        else:
            comparator = "variance_unqualified_strict_mae_improvement"
            mae_check = (
                current_mae + PAIR_MAE_STRICT_IMPROVEMENT_TOLERANCE
                < baseline_mae
            )
            correlation_check = True
        passed = absolute_correlation and mae_check and correlation_check
        per_scene[scene] = {
            "baseline_variance_qualified": baseline_qualified,
            "baseline_spread_ratio": baseline_spread,
            "comparator": comparator,
            "baseline_pair_mae": baseline_mae,
            "candidate_pair_mae": current_mae,
            "pair_mae_check": mae_check,
            "absolute_pair_correlation_threshold": (
                v2.MINIMUM_PAIR_GRAM_CORRELATION
            ),
            "baseline_pair_correlation": baseline_correlation,
            "candidate_pair_correlation": current_correlation,
            "absolute_pair_correlation_check": absolute_correlation,
            "correlation_non_regression_check": correlation_check,
            "passed": passed,
        }
    return all(row["passed"] for row in per_scene.values()), per_scene


def attach_selection(
    validation: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    selected = v2.attach_selection(validation, baseline)
    checks = dict(selected["selection"]["checks"])
    checks.pop("every_scene_pair_geometry_non_regression")
    pair_passed, pair_audit = _conditional_pair_geometry(validation, baseline)
    checks["every_scene_pair_geometry_conditional_baseline"] = pair_passed
    selection = dict(selected["selection"])
    selection["checks"] = checks
    selection["pair_geometry_baseline_audit"] = pair_audit
    selection["eligible"] = all(checks.values())
    return {**dict(selected), "selection": selection}


def select_step(history: Sequence[Mapping[str, Any]]) -> int | None:
    return v2.select_step(history)


def train(args: argparse.Namespace) -> dict[str, Any]:
    contract = training_contract()
    output = Path(args.output).expanduser().resolve()
    result_path = output.with_suffix(output.suffix + ".json")
    normalization_path = output.with_suffix(output.suffix + ".normalization.pt")
    reference_path = output.with_suffix(output.suffix + ".contrast_reference.pt")
    if any(
        path.exists() or path.is_symlink()
        for path in (output, result_path, normalization_path, reference_path)
    ):
        raise FileExistsError("contrast V2.1 first-writer output already exists")
    prepared = prepare_inputs(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
    )
    source = prepared.base_v2.source
    train_scenes = [legacy.load_scene(binding) for binding in source.train]
    validation_scenes = [legacy.load_scene(binding) for binding in source.validation]
    normalization = interface.build_source_normalization(
        [scene.state for scene in train_scenes],
        source_state_cohort_authority_sha256=source.registry["authority_sha256"],
    )
    reference = v2.build_contrast_reference(
        train_scenes, cohort_sha256=source.registry["authority_sha256"]
    )
    teacher_center = reference["teacher_center"]
    device = torch.device(str(args.device))
    torch.manual_seed(v2.SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(v2.SEED)
    model = interface.build_model(DIRECTION_ONLY, normalization).to(device)
    head = SigLIP2SummaryHead.from_radio_checkpoint(
        source.authority["verified_radio_path"],
        expected_sha256=source.authority["official_radio_checkpoint"]["sha256"],
    ).to(device).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=v2.LEARNING_RATE, weight_decay=v2.WEIGHT_DECAY
    )
    baseline = v2.evaluate(model, head, validation_scenes, teacher_center, device)
    zero_state = v2._state_copy(model)
    history: list[dict[str, Any]] = [
        {
            "step": 0,
            "training_scene_objective": None,
            "validation": attach_selection(baseline, baseline),
            "model_state_dict_sha256": v2._state_sha(zero_state),
        }
    ]
    saved_states: dict[int, dict[str, torch.Tensor]] = {}
    last_training: dict[str, Any] | None = None
    for step in range(1, OPTIMIZER_STEPS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        scene_objective: dict[str, Any] = {}
        for scene in train_scenes:
            rows = legacy._cyclic_batch(legacy._scene_rows(scene), step=step)
            loss, parts = v2._batch_loss(
                model, head, scene, rows, teacher_center, device
            )
            (loss / len(train_scenes)).backward()
            scene_objective[scene.binding.scene_id] = {
                "total": float(loss.detach().cpu()),
                **parts,
            }
        torch.nn.utils.clip_grad_norm_(
            tuple(model.parameters()),
            v2.MAX_GRADIENT_NORM,
            error_if_nonfinite=True,
        )
        optimizer.step()
        last_training = {"step": step, "per_scene": scene_objective}
        if step % EVALUATION_INTERVAL != 0:
            continue
        validation = attach_selection(
            v2.evaluate(model, head, validation_scenes, teacher_center, device),
            baseline,
        )
        state = v2._state_copy(model)
        entry = {
            "step": step,
            "training_scene_objective": scene_objective,
            "validation": validation,
            "model_state_dict_sha256": v2._state_sha(state),
        }
        history.append(entry)
        if validation["selection"]["eligible"] is True:
            saved_states[step] = state
        print(json.dumps(entry, sort_keys=True), flush=True)
    selected = select_step(history)
    normalization_file = write_torch_noclobber(normalization_path, normalization)
    reference_file = write_torch_noclobber(reference_path, reference)
    checkpoint_record: dict[str, str] | None = None
    if selected is not None:
        selected_state = saved_states[selected]
        checkpoint = {
            "schema": CHECKPOINT_SCHEMA,
            "schema_version": 21,
            "training_contract": contract,
            "training_contract_sha256": canonical_json_sha256(contract),
            "interface_contract_sha256": interface.INTERFACE_CONTRACT_SHA256,
            "model_architecture": model.architecture(
                interface.INTERFACE_CONTRACT_SHA256
            ),
            "model_state_dict": selected_state,
            "model_state_dict_sha256": v2._state_sha(selected_state),
            "normalization": file_record(normalization_file),
            "contrast_reference": file_record(reference_file),
            "execution_authority": {
                "path": prepared.authority["verified_path"],
                "sha256": prepared.authority["verified_sha256"],
            },
            "selected_step": selected,
            "source_access": source_access(),
        }
        checkpoint_file = write_torch_noclobber(output, checkpoint)
        checkpoint_record = file_record(checkpoint_file)
    report = {
        "schema": RESULT_SCHEMA,
        "schema_version": 21,
        "status": (
            "source_only_contrast_v21_promotion_candidate_complete"
            if selected is not None
            else "source_only_contrast_v21_complete_no_eligible_candidate"
        ),
        "arm": DIRECTION_ONLY,
        "training_contract": contract,
        "training_contract_sha256": canonical_json_sha256(contract),
        "execution_authority": {
            "path": prepared.authority["verified_path"],
            "sha256": prepared.authority["verified_sha256"],
        },
        "normalization": file_record(normalization_file),
        "contrast_reference": file_record(reference_file),
        "checkpoint": checkpoint_record,
        "selected_step": selected,
        "history": history,
        "last_training_step": last_training,
        "benchmark_opened": False,
        "source_access": source_access(),
    }
    write_frozen_json(result_path, report)
    return report


def synthetic_dry_run() -> dict[str, Any]:
    baseline = {
        "student_to_teacher_spread_ratio": 0.05,
        "teacher_centered_pair_gram_mae": 0.5,
        "teacher_centered_pair_gram_correlation": 0.8,
    }
    candidate = {
        "student_to_teacher_spread_ratio": 0.9,
        "teacher_centered_pair_gram_mae": 0.3,
        "teacher_centered_pair_gram_correlation": 0.6,
    }
    pair_passed, audit = _conditional_pair_geometry(
        {"per_scene": {scene: candidate for scene in VALIDATION_SCENES}},
        {"per_scene": {scene: baseline for scene in VALIDATION_SCENES}},
    )
    return {
        "schema": "radio_gs.factorized_native_contrast_v21_synthetic_dry_run.v1",
        "collapsed_baseline_pair_gate_passed": pair_passed,
        "pair_comparators": {
            scene: audit[scene]["comparator"] for scene in VALIDATION_SCENES
        },
        "contrast_v2_modified": False,
        "benchmark_opened": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("synthetic-dry-run")
    validate = commands.add_parser("validate-authority")
    validate.add_argument("--execution-authority", required=True)
    validate.add_argument("--expected-execution-authority-sha256", required=True)
    run = commands.add_parser("train")
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
        prepared = prepare_inputs(
            args.execution_authority,
            expected_sha256=args.expected_execution_authority_sha256,
        )
        result = {
            "status": "source_only_contrast_v21_authority_validated",
            "source_train": [
                item.scene_id for item in prepared.base_v2.source.train
            ],
            "source_validation": [
                item.scene_id for item in prepared.base_v2.source.validation
            ],
            "benchmark_opened": False,
        }
    else:
        result = train(args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "CHECKPOINT_SCHEMA",
    "EXECUTION_AUTHORITY_SCHEMA",
    "RESULT_SCHEMA",
    "attach_selection",
    "build_parser",
    "prepare_inputs",
    "select_step",
    "source_access",
    "synthetic_dry_run",
    "train",
    "training_contract",
    "validate_execution_authority",
]
