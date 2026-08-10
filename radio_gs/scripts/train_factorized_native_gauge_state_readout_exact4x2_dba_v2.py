#!/usr/bin/env python3
"""Train source-only precision-constrained ranking DBA-v2."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import torch

from radio_gs.interfaces import factorized_native_gauge_state_readout as readout
from radio_gs.losses import factorized_native_source_precision_ranking as precision_rank
from radio_gs.models.factorized_native_gauge_state_readout import DIRECTION_ONLY
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.scripts import (
    train_factorized_native_gauge_state_readout_exact4x2 as legacy,
)
from radio_gs.scripts import (
    train_factorized_native_gauge_state_readout_exact4x2_contrast_v2 as contrast_v2,
)
from radio_gs.scripts import (
    train_factorized_native_gauge_state_readout_exact4x2_dba_v1 as dba_v1,
)
from radio_gs.scripts import train_surface_region_typed_context_residual as sparse_teacher
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


EXECUTION_AUTHORITY_SCHEMA = (
    "radio_gs.factorized_native_precision_ranking_dba_v2_"
    "execution_authority.v1"
)
CHECKPOINT_SCHEMA = "radio_gs.factorized_native_precision_ranking_checkpoint.v2"
RESULT_SCHEMA = "radio_gs.factorized_native_precision_ranking_source_result.v2"
SCHEMA_VERSION = 2
TRAIN_SCENES = dba_v1.TRAIN_SCENES
VALIDATION_SCENES = dba_v1.VALIDATION_SCENES
OPTIMIZER_STEPS = dba_v1.OPTIMIZER_STEPS
EVALUATION_INTERVAL = dba_v1.EVALUATION_INTERVAL
BATCH_ROWS = dba_v1.BATCH_ROWS
LEARNING_RATE = dba_v1.LEARNING_RATE
WEIGHT_DECAY = dba_v1.WEIGHT_DECAY
MAX_GRADIENT_NORM = dba_v1.MAX_GRADIENT_NORM
DBA_AUXILIARY_WEIGHT = dba_v1.DBA_AUXILIARY_WEIGHT
SEED = dba_v1.SEED
DESIGN_PATH = (
    Path(__file__).resolve().parents[2]
    / "paper/artifacts/factorized_native_source_precision_ranking_dba_v2_design_20260807.md"
)
_DEPENDENCY_PATHS = {
    "precision_ranking_loss": Path(precision_rank.__file__).resolve(),
    "dba_v1_frozen_source_protocol": Path(dba_v1.__file__).resolve(),
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def source_access() -> dict[str, bool]:
    return dict(dba_v1.source_access())


def training_contract() -> dict[str, Any]:
    contract = deepcopy(dba_v1.training_contract())
    contract["schema"] = RESULT_SCHEMA
    contract["schema_version"] = SCHEMA_VERSION
    contract["candidate"] = {
        "initialization": "exact_promoted_contrast_v2_1_direction_only_checkpoint",
        "dba_v1_checkpoint_used": False,
        "new_inference_parameters": False,
        "scene_parameters": False,
        "query_parameters": False,
    }
    contract["objective"] = {
        "visual": "complete_immutable_contrast_v2_1_visual_objective",
        "boundary_primary": (
            "balanced_all_teacher_positive_and_top_3_per_positive_"
            "teacher_negative_zero_boundary_softplus"
        ),
        "minimum_precision": precision_rank.MINIMUM_PRECISION,
        "hard_negatives_per_positive": (
            precision_rank.HARD_NEGATIVES_PER_POSITIVE
        ),
        "hard_negative_selection": (
            "deterministic_highest_student_inference_margin_teacher_negatives"
        ),
        "soft_fidelity": {
            "domain": "same_positive_and_selected_hard_negative_units",
            "loss": "class_balanced_smooth_l1_probability_to_teacher_probability",
            "beta": precision_rank.SMOOTH_L1_BETA,
            "weight": precision_rank.SOFT_FIDELITY_WEIGHT,
        },
        "boundary_pairwise_rank": {
            "pairs": "positive_to_deterministic_hard_negative_quantiles",
            "loss": "softplus_10_times_negative_minus_positive_margin",
            "weight": precision_rank.BOUNDARY_RANK_WEIGHT,
        },
        "global_order": {
            "pairs": "teacher_probability_sorted_lower_upper_quantiles",
            "pair_cap": precision_rank.GLOBAL_ORDER_PAIR_CAP,
            "minimum_teacher_probability_gap": (
                precision_rank.MINIMUM_TEACHER_ORDER_GAP
            ),
            "loss": "softplus_negative_10_times_student_margin_order_gap",
            "weight": precision_rank.GLOBAL_ORDER_WEIGHT,
        },
        "boundary_auxiliary_weight_on_complete_dba_v2_loss": (
            DBA_AUXILIARY_WEIGHT
        ),
        "equal_scene_gradient_accumulation": True,
    }
    contract["promotion"]["inherited_unchanged_from_dba_v1"] = True
    return contract


TRAINING_CONTRACT_SHA256 = canonical_json_sha256(training_contract())


@dataclass(frozen=True)
class PreparedInputs:
    authority: dict[str, Any]
    base: dba_v1.PreparedInputs


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    path, digest = str(value["path"]), str(value["sha256"])
    if not Path(path).is_absolute() or _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} file record differs")
    return {"path": path, "sha256": digest}


def _canonical_output(value: object) -> str:
    raw = str(value)
    resolved = str(Path(raw).expanduser().resolve())
    if raw != resolved:
        raise ValueError("DBA-v2 output must be canonical absolute")
    return resolved


def validate_execution_authority_header(value: object) -> dict[str, Any]:
    required = {
        "schema",
        "schema_version",
        "status",
        "implementation",
        "implementation_dependencies",
        "design_preregistration",
        "base_dba_v1_execution_authority",
        "training_contract_sha256",
        "training_output",
        "training_authorized",
        "target_execution_authorized",
        "query_execution_authorized",
        "metric_execution_authorized",
        "source_access",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("DBA-v2 execution authority fields differ")
    authority = dict(value)
    if (
        authority.get("schema") != EXECUTION_AUTHORITY_SCHEMA
        or authority.get("schema_version") != SCHEMA_VERSION
        or authority.get("status")
        != "authorized_source_only_precision_ranking_dba_v2_exact4train_2validation"
        or authority.get("training_contract_sha256") != TRAINING_CONTRACT_SHA256
        or authority.get("training_authorized") is not True
        or authority.get("target_execution_authorized") is not False
        or authority.get("query_execution_authorized") is not False
        or authority.get("metric_execution_authorized") is not False
        or authority.get("source_access") != source_access()
    ):
        raise ValueError("DBA-v2 execution authority header differs")
    authority["implementation"] = _record(
        authority["implementation"], label="DBA-v2 implementation"
    )
    dependencies = authority.get("implementation_dependencies")
    if not isinstance(dependencies, Mapping) or set(dependencies) != set(
        _DEPENDENCY_PATHS
    ):
        raise ValueError("DBA-v2 implementation dependencies differ")
    authority["implementation_dependencies"] = {
        name: _record(dependencies[name], label=f"DBA-v2 dependency {name}")
        for name in sorted(_DEPENDENCY_PATHS)
    }
    for name in ("design_preregistration", "base_dba_v1_execution_authority"):
        authority[name] = _record(authority[name], label=f"DBA-v2 {name}")
    authority["training_output"] = _canonical_output(authority["training_output"])
    return authority


def prepare_inputs(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_output: str | Path | None = None,
) -> PreparedInputs:
    raw, digest, source = load_json_object(
        path, expected_sha256=expected_sha256, label="DBA-v2 execution authority"
    )
    authority = validate_execution_authority_header(raw)
    observed = validate_file_record(
        authority["implementation"], label="DBA-v2 implementation"
    )
    if observed != Path(__file__).resolve():
        raise ValueError("DBA-v2 authority binds another trainer")
    for name, expected in _DEPENDENCY_PATHS.items():
        observed = validate_file_record(
            authority["implementation_dependencies"][name],
            label=f"DBA-v2 dependency {name}",
        )
        if observed != expected:
            raise ValueError(f"DBA-v2 dependency differs: {name}")
    design = validate_file_record(
        authority["design_preregistration"], label="DBA-v2 design preregistration"
    )
    if design != DESIGN_PATH.resolve():
        raise ValueError("DBA-v2 authority binds another design")
    base_record = authority["base_dba_v1_execution_authority"]
    validate_file_record(base_record, label="DBA-v2 base DBA-v1 authority")
    base = dba_v1.prepare_inputs(
        base_record["path"], expected_sha256=base_record["sha256"]
    )
    if expected_output is not None and authority["training_output"] != str(
        Path(expected_output).expanduser().resolve()
    ):
        raise ValueError("DBA-v2 authority binds another output")
    authority["verified_record"] = {"path": str(source), "sha256": digest}
    return PreparedInputs(authority=authority, base=base)


def _batch_objective(
    model: torch.nn.Module,
    head: torch.nn.Module,
    scene: legacy.LoadedScene,
    rows: torch.Tensor,
    teacher_center: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    descriptor = legacy._descriptor_batch(model, head, scene, rows, device)
    teacher, mask = sparse_teacher.gather_sparse_teacher_batch(scene.shard, rows)
    teacher_device = teacher.to(device)
    mask_device = mask.to(device)
    visual_loss, visual_parts = contrast_v2.contrast.contrast_preserving_objective(
        descriptor, teacher_device, mask_device, teacher_center.to(device)
    )
    aligned = precision_rank.source_precision_constrained_ranking_loss(
        descriptor, teacher_device, mask_device, positive, negative
    )
    total = visual_loss + DBA_AUXILIARY_WEIGHT * aligned.loss
    parts: dict[str, float | int] = {
        "total": float(total.detach().cpu()),
        "visual_total": float(visual_loss.detach().cpu()),
        "dba_v2_complete": float(aligned.loss.detach().cpu()),
        "dba_v2_weighted": float(
            (DBA_AUXILIARY_WEIGHT * aligned.loss).detach().cpu()
        ),
        "dba_v2_hard_boundary": float(aligned.hard_boundary_loss.detach().cpu()),
        "dba_v2_soft_fidelity": float(aligned.soft_fidelity_loss.detach().cpu()),
        "dba_v2_boundary_pairwise_rank": float(
            aligned.boundary_pairwise_rank_loss.detach().cpu()
        ),
        "dba_v2_global_order": float(aligned.global_order_loss.detach().cpu()),
        "dba_v2_teacher_positive_pairs": aligned.teacher_positive_pairs,
        "dba_v2_teacher_negative_pairs": aligned.teacher_negative_pairs,
        "dba_v2_selected_hard_negative_pairs": (
            aligned.selected_hard_negative_pairs
        ),
        "dba_v2_global_order_pairs": aligned.global_order_pairs,
    }
    parts.update(
        {
            f"visual_{name}": float(value.detach().cpu())
            for name, value in visual_parts.items()
        }
    )
    return total, parts


def train(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    result_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or output.is_symlink() or result_path.exists() or result_path.is_symlink():
        raise FileExistsError("DBA-v2 first-writer output already exists")
    prepared = prepare_inputs(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
        expected_output=output,
    )
    base = prepared.base
    exclusion_raw, _, _ = load_json_object(
        base.authority["benchmark_exclusion_manifest"]["path"],
        expected_sha256=base.authority["benchmark_exclusion_manifest"]["sha256"],
        label="DBA-v2 benchmark exclusion manifest",
    )
    dba_v1.exclusion.validate_benchmark_exclusion_manifest(exclusion_raw)
    train_scenes = [legacy.load_scene(binding) for binding in base.source.train]
    validation_scenes = [
        legacy.load_scene(binding) for binding in base.source.validation
    ]
    for scene in (*train_scenes, *validation_scenes):
        if legacy._scene_rows(scene).numel() != dba_v1.REGIONS_PER_SCENE:
            raise ValueError("DBA-v2 requires all 4096 canonical rows")

    device = torch.device(str(args.device))
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    gate = base.source_gate
    model = readout.build_model(DIRECTION_ONLY, gate["normalization"])
    model.load_state_dict(gate["checkpoint"]["model_state_dict"], strict=True)
    model = model.to(device)
    head = SigLIP2SummaryHead.from_radio_checkpoint(
        gate["official_radio_checkpoint"]["path"],
        expected_sha256=gate["official_radio_checkpoint"]["sha256"],
    ).to(device).eval().requires_grad_(False)
    positive, negative = dba_v1._load_text_assets(base.authority, device)
    teacher_center = gate["contrast_reference"]["teacher_center"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    baseline = dba_v1.evaluate(
        model, head, validation_scenes, teacher_center, positive, negative, device
    )
    zero_state = contrast_v2._state_copy(model)
    history: list[dict[str, Any]] = [
        {
            "step": 0,
            "training_scene_objective": None,
            "validation": dba_v1.attach_selection(baseline, baseline),
            "model_state_dict_sha256": contrast_v2._state_sha(zero_state),
        }
    ]
    saved_states: dict[int, dict[str, torch.Tensor]] = {}
    last_training: dict[str, Any] | None = None
    for step in range(1, OPTIMIZER_STEPS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        per_scene: dict[str, Any] = {}
        for scene in train_scenes:
            rows = legacy._cyclic_batch(legacy._scene_rows(scene), step=step)
            total, parts = _batch_objective(
                model,
                head,
                scene,
                rows,
                teacher_center,
                positive,
                negative,
                device,
            )
            (total / len(train_scenes)).backward()
            per_scene[scene.binding.scene_id] = parts
        torch.nn.utils.clip_grad_norm_(
            tuple(model.parameters()), MAX_GRADIENT_NORM, error_if_nonfinite=True
        )
        optimizer.step()
        last_training = {"step": step, "per_scene": per_scene}
        if step % EVALUATION_INTERVAL != 0:
            continue
        validation = dba_v1.attach_selection(
            dba_v1.evaluate(
                model,
                head,
                validation_scenes,
                teacher_center,
                positive,
                negative,
                device,
            ),
            baseline,
        )
        state = contrast_v2._state_copy(model)
        entry = {
            "step": step,
            "training_scene_objective": per_scene,
            "validation": validation,
            "model_state_dict_sha256": contrast_v2._state_sha(state),
        }
        history.append(entry)
        if validation["selection"]["eligible"] is True:
            saved_states[step] = state
        print(json.dumps(entry, sort_keys=True), flush=True)

    selected = dba_v1.select_step(history)
    checkpoint_record: dict[str, str] | None = None
    if selected is not None:
        selected_state = saved_states[selected]
        checkpoint = {
            "schema": CHECKPOINT_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "training_contract": training_contract(),
            "training_contract_sha256": TRAINING_CONTRACT_SHA256,
            "interface_contract_sha256": readout.INTERFACE_CONTRACT_SHA256,
            "model_architecture": model.architecture(readout.INTERFACE_CONTRACT_SHA256),
            "model_state_dict": selected_state,
            "model_state_dict_sha256": contrast_v2._state_sha(selected_state),
            "warm_start_source_contrast_v21_result": dict(
                base.authority["source_contrast_v21_result"]
            ),
            "warm_start_source_contrast_v21_checkpoint": dict(
                gate["result"]["checkpoint"]
            ),
            "normalization": dict(gate["result"]["normalization"]),
            "contrast_reference": dict(gate["result"]["contrast_reference"]),
            "fit_text_bank": dict(base.authority["fit_text_bank"]),
            "canonical_negative_bank": dict(base.authority["canonical_negative_bank"]),
            "execution_authority": dict(prepared.authority["verified_record"]),
            "selected_step": selected,
            "source_access": source_access(),
        }
        checkpoint_record = file_record(write_torch_noclobber(output, checkpoint))
    report = {
        "schema": RESULT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": (
            "source_only_dba_v2_promotion_candidate_complete"
            if selected is not None
            else "source_only_dba_v2_complete_no_eligible_candidate"
        ),
        "arm": DIRECTION_ONLY,
        "training_contract": training_contract(),
        "training_contract_sha256": TRAINING_CONTRACT_SHA256,
        "execution_authority": dict(prepared.authority["verified_record"]),
        "input_authority": {
            "base_dba_v1_execution_authority": dict(
                prepared.authority["base_dba_v1_execution_authority"]
            ),
            "source_contrast_v21_result": dict(
                base.authority["source_contrast_v21_result"]
            ),
            "source_contrast_v21_checkpoint": dict(gate["result"]["checkpoint"]),
            "fit_text_bank": dict(base.authority["fit_text_bank"]),
            "canonical_negative_bank": dict(base.authority["canonical_negative_bank"]),
            "benchmark_exclusion_manifest": dict(
                base.authority["benchmark_exclusion_manifest"]
            ),
        },
        "checkpoint": checkpoint_record,
        "selected_step": selected,
        "history": history,
        "last_training_step": last_training,
        "target_query_or_metric_authorized": False,
        "benchmark_opened": False,
        "source_access": source_access(),
    }
    write_frozen_json(result_path, report)
    return report


def synthetic_dry_run() -> dict[str, Any]:
    return {
        "schema": "radio_gs.factorized_native_dba_v2_synthetic_dry_run.v1",
        "hard_negatives_per_positive": precision_rank.HARD_NEGATIVES_PER_POSITIVE,
        "minimum_precision": precision_rank.MINIMUM_PRECISION,
        "global_order_pair_cap": precision_rank.GLOBAL_ORDER_PAIR_CAP,
        "training_steps": OPTIMIZER_STEPS,
        "rows_per_scene_per_step": BATCH_ROWS,
        "complete_rows_per_scene": OPTIMIZER_STEPS * BATCH_ROWS,
        "evaluation_steps": [
            0,
            *range(EVALUATION_INTERVAL, OPTIMIZER_STEPS + 1, EVALUATION_INTERVAL),
        ],
        "promotion_gate_bitwise_inherited_from_dba_v1": (
            training_contract()["promotion"]["inherited_unchanged_from_dba_v1"]
        ),
        "target_query_or_benchmark_opened": False,
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
            "status": "source_only_dba_v2_authority_validated",
            "source_train": [item.scene_id for item in prepared.base.source.train],
            "source_validation": [
                item.scene_id for item in prepared.base.source.validation
            ],
            "warm_start_selected_step": prepared.base.source_gate["selected_step"],
            "target_query_or_benchmark_opened": False,
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
    "TRAINING_CONTRACT_SHA256",
    "build_parser",
    "prepare_inputs",
    "source_access",
    "synthetic_dry_run",
    "train",
    "training_contract",
    "validate_execution_authority_header",
]
