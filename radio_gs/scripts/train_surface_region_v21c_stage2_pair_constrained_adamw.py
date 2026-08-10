#!/usr/bin/env python3
"""Train independently authorized V2.1C Stage-II after pair-conflict audit."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import torch

from radio_gs.interfaces import surface_region_v21c_stage2_pair_trigger as trigger
from radio_gs.interfaces import (
    surface_region_v21b_reliability_conditioned_residual as v21b_interface,
)
from radio_gs.optimization import constrained_adamw_update_v21c_stage2 as update
from radio_gs.optimization import adamw_two_constraint_projection_v21c as frozen_projection
from radio_gs.scripts import (
    train_surface_region_typed_context_residual as v1_trainer,
)
from radio_gs.scripts import (
    train_surface_region_v21b_conditioned_rank256_exact4x2 as v21b,
)
from radio_gs.scripts import (
    train_surface_region_v21c_two_stage_constrained_adamw as frozen_stage_i,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_json_object,
    validate_file_record,
)


EXECUTION_AUTHORITY_SCHEMA = (
    "radio_gs.surface_region_v21c_stage2_pair_constrained_adamw_"
    "execution_authority.v1"
)
TRAINING_ADDENDUM = Path(
    "paper/artifacts/"
    "surface_region_v21c_stage2_pair_conflict_trigger_addendum_20260807.json"
)
TRAINING_ADDENDUM_SHA256 = (
    "0c78aa38f18cf6d415f93bc6517128b9eac062a4c8146183ae4304095442946e"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class StageIIInputs:
    execution: dict[str, Any]
    parent: v21b.V21BInputs
    stage_i_result: dict[str, Any]
    pair_trigger_evidence: dict[str, Any]


def source_access() -> dict[str, bool]:
    return {
        **frozen_stage_i.source_access(),
        "pair_conflict_majority_trigger_opened": True,
        "pair_candidate_cosine_median_trigger_opened": True,
    }


def training_contract() -> dict[str, Any]:
    base = frozen_stage_i.training_contract()
    return {
        **base,
        "artifact_type": (
            "surface_region_v21c_stage2_pair_constrained_adamw_source_pilot"
        ),
        "stage_ii_addendum_sha256": TRAINING_ADDENDUM_SHA256,
        "strong_pair_trigger": {
            "minimum_conflict_steps": trigger.MINIMUM_PAIR_CONFLICT_STEPS,
            "pair_dot_threshold": trigger.PAIR_DOT_THRESHOLD,
            "pair_candidate_cosine_median_maximum_exclusive": (
                trigger.PAIR_COSINE_MEDIAN_MAXIMUM_EXCLUSIVE
            ),
            "absolute_only_conflict_authorizes": False,
        },
        "source_access": source_access(),
    }


TRAINING_CONTRACT_SHA256 = canonical_json_sha256(training_contract())


_CODE_RECORD_FIELDS = (
    "trainer_implementation",
    "execution_builder_implementation",
    "update_policy_implementation",
    "pair_trigger_implementation",
    "source_gate_implementation",
    "frozen_stage_i_trainer_implementation",
    "frozen_projection_implementation",
    "parent_preregistration",
    "training_addendum",
)


def _expected_code_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[2]
    return {
        "trainer_implementation": Path(__file__).resolve(),
        "execution_builder_implementation": root
        / "radio_gs/scripts/build_surface_region_v21c_stage2_execution_authority.py",
        "update_policy_implementation": Path(update.__file__).resolve(),
        "pair_trigger_implementation": Path(trigger.__file__).resolve(),
        "source_gate_implementation": root
        / "radio_gs/interfaces/surface_region_v21c_stage2_source_gate.py",
        "frozen_stage_i_trainer_implementation": Path(
            frozen_stage_i.__file__
        ).resolve(),
        "frozen_projection_implementation": Path(
            frozen_projection.__file__
        ).resolve(),
        "parent_preregistration": root / frozen_stage_i.PREREGISTRATION,
        "training_addendum": root / TRAINING_ADDENDUM,
    }


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"V2.1C Stage-II {label} file record differs")
    record = {"path": str(value["path"]), "sha256": str(value["sha256"])}
    if not record["path"].startswith("/") or _SHA256.fullmatch(record["sha256"]) is None:
        raise ValueError(f"V2.1C Stage-II {label} file record differs")
    return record


def validate_execution_authority(value: object) -> dict[str, Any]:
    required = {
        "schema",
        "schema_version",
        "status",
        *_CODE_RECORD_FIELDS,
        "parent_v21b_execution_authority",
        "stage_i_execution_authority",
        "stage_i_audit_result",
        "pair_trigger_evidence",
        "training_authorized",
        "projection_authorized",
        "benchmark_execution_authorized",
        "source_access",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("V2.1C Stage-II authority fields differ")
    authority = dict(value)
    if (
        authority["schema"] != EXECUTION_AUTHORITY_SCHEMA
        or authority["schema_version"] != 1
        or authority["status"]
        != "authorized_source_only_stage_ii_after_pair_conflict_majority"
        or authority["training_authorized"] is not True
        or authority["projection_authorized"] is not True
        or authority["benchmark_execution_authorized"] is not False
        or authority["source_access"] != source_access()
    ):
        raise ValueError("V2.1C Stage-II authority header differs")
    for name in (
        *_CODE_RECORD_FIELDS,
        "parent_v21b_execution_authority",
        "stage_i_execution_authority",
        "stage_i_audit_result",
    ):
        authority[name] = _record(authority[name], label=name)
    if not isinstance(authority["pair_trigger_evidence"], Mapping):
        raise ValueError("V2.1C Stage-II pair-trigger evidence differs")
    authority["pair_trigger_evidence"] = dict(authority["pair_trigger_evidence"])
    return authority


def prepare_inputs(path: str | Path, *, expected_sha256: str) -> StageIIInputs:
    raw, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="V2.1C Stage-II execution authority",
    )
    authority = validate_execution_authority(raw)
    for name, expected in _expected_code_paths().items():
        observed = validate_file_record(
            authority[name], label=f"V2.1C Stage-II {name}"
        )
        if observed != expected.resolve():
            raise ValueError(f"V2.1C Stage-II authority binds another {name}")
    if (
        authority["parent_preregistration"]["sha256"]
        != frozen_stage_i.PREREGISTRATION_SHA256
        or authority["training_addendum"]["sha256"]
        != TRAINING_ADDENDUM_SHA256
    ):
        raise ValueError("V2.1C Stage-II preregistration/addendum differs")
    parent_record = authority["parent_v21b_execution_authority"]
    parent = v21b.prepare_inputs(
        validate_file_record(parent_record, label="V2.1C Stage-II V2.1B parent"),
        expected_sha256=parent_record["sha256"],
    )
    audit_record = authority["stage_i_audit_result"]
    audit_raw, _audit_sha, _audit_path = load_json_object(
        validate_file_record(audit_record, label="V2.1C Stage-I audit"),
        expected_sha256=audit_record["sha256"],
        label="V2.1C Stage-I audit",
    )
    audit = frozen_stage_i.validate_stage_i_audit_result(audit_raw)
    evidence = trigger.require_authorized(audit)
    if evidence != authority["pair_trigger_evidence"]:
        raise ValueError("V2.1C Stage-II pair-trigger evidence replay differs")
    if (
        audit["parent_v21b_execution_authority"] != parent_record
        or audit["execution_authority"] != authority["stage_i_execution_authority"]
    ):
        raise ValueError("V2.1C Stage-II Stage-I lineage differs")
    authority["verified_path"] = str(source)
    authority["verified_sha256"] = digest
    return StageIIInputs(
        execution=authority,
        parent=parent,
        stage_i_result=audit,
        pair_trigger_evidence=evidence,
    )


def train_one_step(
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    inputs: StageIIInputs,
    normalization: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    named, gradients, per_scene = frozen_stage_i._training_gradients(
        model, optimizer, inputs, normalization, device
    )
    geometry = frozen_projection.gradient_geometry(gradients)
    certificate = update.commit(optimizer, named, gradients)
    return {
        "scene_count": len(inputs.parent.train),
        "equal_scene_weight": 1.0 / len(inputs.parent.train),
        "per_scene": per_scene,
        "gradient_evidence": geometry,
        "projected_adamw_evidence": certificate,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    paths = frozen_stage_i._stage_ii_output_paths(output)
    existing = [path for path in paths.values() if path.exists() or path.is_symlink()]
    if existing:
        raise FileExistsError(
            "V2.1C Stage-II first-writer outputs already exist: "
            + ", ".join(str(path) for path in existing)
        )
    inputs = prepare_inputs(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
    )
    device = torch.device(str(args.device))
    normalization = frozen_stage_i.build_normalization(inputs)
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
    subset = frozen_projection.parameter_subset_manifest(
        frozen_projection.trainable_named_parameters(model)
    )
    if subset != inputs.stage_i_result["parameter_subset"]:
        raise ValueError("V2.1C Stage-II parameter subset differs from Stage-I")
    epoch_zero_raw = frozen_stage_i.evaluate(model, inputs, normalization, device)
    epoch_zero = frozen_stage_i.attach_promotion(epoch_zero_raw, epoch_zero_raw)
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
    for step in range(1, frozen_stage_i.OPTIMIZER_STEPS + 1):
        training = train_one_step(model, optimizer, inputs, normalization, device)
        validation = frozen_stage_i.attach_promotion(
            frozen_stage_i.evaluate(model, inputs, normalization, device), epoch_zero
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
    selected_step = frozen_stage_i.select_promotion_step(history)
    if selected_step is not None:
        model.load_state_dict(saved_states[selected_step], strict=True)
        restored = frozen_stage_i.attach_promotion(
            frozen_stage_i.evaluate(model, inputs, normalization, device), epoch_zero
        )
        if restored != history[selected_step]["validation"]:
            raise RuntimeError("V2.1C Stage-II restored validation differs")
    model.cpu()
    return frozen_stage_i._write_stage_ii_outputs(
        output,
        normalization,
        inputs,
        history,
        saved_states,
        selected_step,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-authority")
    validate.add_argument("--execution-authority", required=True)
    validate.add_argument("--expected-execution-authority-sha256", required=True)
    train = commands.add_parser("train")
    train.add_argument("--execution-authority", required=True)
    train.add_argument("--expected-execution-authority-sha256", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate-authority":
        inputs = prepare_inputs(
            args.execution_authority,
            expected_sha256=args.expected_execution_authority_sha256,
        )
        result = {
            "status": "source_only_v21c_stage2_authority_validated",
            "pair_trigger_evidence": inputs.pair_trigger_evidence,
            "benchmark_opened": False,
        }
    else:
        result = run(args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "EXECUTION_AUTHORITY_SCHEMA",
    "TRAINING_ADDENDUM_SHA256",
    "TRAINING_CONTRACT_SHA256",
    "StageIIInputs",
    "build_parser",
    "prepare_inputs",
    "run",
    "source_access",
    "train_one_step",
    "training_contract",
    "validate_execution_authority",
]
