#!/usr/bin/env python3
"""Train the opt-in direction-only readout with contrast-preserving supervision.

V1 remains frozen.  This source-only V2 removes a source-train teacher centroid
before supervising residual directions and pair geometry, and promotes only on
the frozen source-validation scenes.  It has no target or benchmark command.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from radio_gs.interfaces import factorized_native_gauge_state_readout as interface
from radio_gs.models import source_contrast_preservation as contrast
from radio_gs.models.factorized_native_gauge_state_readout import (
    DIRECTION_ONLY,
    FactorizedNativeGaugeStateReadout,
)
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.scripts import (
    train_factorized_native_gauge_state_readout_exact4x2 as legacy,
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
    "radio_gs.factorized_native_gauge_state_readout_exact4x2_"
    "contrast_v2_execution_authority.v1"
)
CHECKPOINT_SCHEMA = (
    "radio_gs.factorized_native_gauge_state_readout_contrast_checkpoint.v2"
)
RESULT_SCHEMA = (
    "radio_gs.factorized_native_gauge_state_readout_contrast_source_result.v2"
)
CONTRAST_REFERENCE_SCHEMA = (
    "radio_gs.factorized_native_teacher_contrast_reference.v1"
)
TRAIN_SCENES = legacy.TRAIN_SCENES
VALIDATION_SCENES = legacy.VALIDATION_SCENES
OPTIMIZER_STEPS = 60
EVALUATION_INTERVAL = 5
BATCH_ROWS = 64
EVAL_BATCH_ROWS = 256
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
MAX_GRADIENT_NORM = 1.0
RAW_NON_REGRESSION_TOLERANCE = 1e-7
MINIMUM_RELATIVE_RAW_MEAN_IMPROVEMENT = 0.002
MINIMUM_CENTERED_RESIDUAL_IMPROVEMENT = 0.02
MINIMUM_PAIR_GRAM_CORRELATION = 0.20
MINIMUM_ABSOLUTE_PROBE_CORRELATION = 0.20
MINIMUM_ABSOLUTE_PROBE_STD_RATIO = 0.75
MAXIMUM_ABSOLUTE_PROBE_STD_RATIO = 1.25
SEED = 0


@dataclass(frozen=True)
class PreparedInputs:
    authority: dict[str, Any]
    source: legacy.PreparedInputs


def source_access() -> dict[str, bool]:
    return dict(legacy.source_access())


def training_contract() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "artifact_type": "factorized_native_contrast_preserving_source_only",
        "legacy_v1_modified": False,
        "arm": DIRECTION_ONLY,
        "cohort": {
            "source_train": list(TRAIN_SCENES),
            "source_validation": list(VALIDATION_SCENES),
            "scene_and_physical_space_disjoint": True,
            "validation_contribution_to_teacher_center": False,
        },
        "input": {
            "base_v1_exact4x2_authority": "caller_sha_bound",
            "unit_visual_teacher_only": True,
            "raw_radio_vector": "prohibited",
            "query_or_text": "prohibited",
            "target_or_benchmark": "prohibited",
        },
        "model": {
            "class": "FactorizedNativeGaugeStateReadout",
            "arm": DIRECTION_ONLY,
            "scene_parameters": False,
            "output": "frozen_official_siglip2_summary_descriptor",
        },
        "teacher_center": {
            "fit": "equal_scene_mean_of_source_train_region_prototypes",
            "gauge": "arithmetic_mean_of_unit_descriptors_not_renormalized",
            "heldout_validation_contribution": False,
        },
        "objective": {
            "raw_all_view_cosine_weight": contrast.RAW_COSINE_WEIGHT,
            "teacher_centered_residual_cosine_weight": (
                contrast.CENTERED_RESIDUAL_WEIGHT
            ),
            "teacher_centered_pair_gram_weight": contrast.CENTERED_GRAM_WEIGHT,
            "absolute_visual_probe_calibration_weight": (
                contrast.ABSOLUTE_VISUAL_PROBE_WEIGHT
            ),
            "variance_noncollapse_weight": contrast.SPREAD_FLOOR_WEIGHT,
            "minimum_student_to_teacher_spread_ratio": (
                contrast.MINIMUM_SPREAD_RATIO
            ),
            "equal_scene_gradient_accumulation": True,
            "query_free": True,
        },
        "optimizer": {
            "name": "AdamW",
            "steps": OPTIMIZER_STEPS,
            "batch_rows_per_scene_per_step": BATCH_ROWS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "maximum_gradient_norm": MAX_GRADIENT_NORM,
            "validation_interval_steps": EVALUATION_INTERVAL,
        },
        "promotion": {
            "source_validation_only": True,
            "required_scenes": list(VALIDATION_SCENES),
            "raw_relative_mean_improvement": (
                MINIMUM_RELATIVE_RAW_MEAN_IMPROVEMENT
            ),
            "centered_residual_absolute_improvement": (
                MINIMUM_CENTERED_RESIDUAL_IMPROVEMENT
            ),
            "every_scene_minimum_spread_ratio": contrast.MINIMUM_SPREAD_RATIO,
            "every_scene_minimum_pair_gram_correlation": (
                MINIMUM_PAIR_GRAM_CORRELATION
            ),
            "absolute_score_calibration_proxy": (
                "student_to_teacher_visual_probe_raw_cosine_response"
            ),
            "every_scene_absolute_visual_probe_correlation": (
                MINIMUM_ABSOLUTE_PROBE_CORRELATION
            ),
            "every_scene_absolute_visual_probe_std_ratio_interval": [
                MINIMUM_ABSOLUTE_PROBE_STD_RATIO,
                MAXIMUM_ABSOLUTE_PROBE_STD_RATIO,
            ],
            "every_scene_raw_and_centered_p05_non_regression": True,
            "every_scene_pair_gram_mae_and_correlation_non_regression": True,
            "ranking": [
                "maximum_macro_teacher_centered_residual_cosine",
                "maximum_macro_teacher_centered_pair_gram_correlation",
                "maximum_macro_raw_all_view_cosine",
                "earliest_step",
            ],
        },
        "benchmark_execution_authorized": False,
        "source_access": source_access(),
    }


_CODE_RECORD_FIELDS = (
    "trainer_implementation",
    "contrast_objective_implementation",
    "model_implementation",
    "interface_implementation",
)


def _expected_code_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[2]
    return {
        "trainer_implementation": Path(__file__).resolve(),
        "contrast_objective_implementation": Path(contrast.__file__).resolve(),
        "model_implementation": (
            root / "radio_gs/models/factorized_native_gauge_state_readout.py"
        ),
        "interface_implementation": (
            root / "radio_gs/interfaces/factorized_native_gauge_state_readout.py"
        ),
    }


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
        raise ValueError("contrast V2 execution authority must be a mapping")
    authority = dict(value)
    required = {
        "schema",
        "schema_version",
        "status",
        *_CODE_RECORD_FIELDS,
        "base_v1_execution_authority",
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
        != "authorized_source_only_contrast_v2_exact4train_2validation"
        or authority.get("training_contract_sha256")
        != canonical_json_sha256(training_contract())
        or authority.get("authorized_arm") != DIRECTION_ONLY
        or authority.get("training_authorized") is not True
        or authority.get("benchmark_execution_authorized") is not False
        or authority.get("source_access") != source_access()
    ):
        raise ValueError("contrast V2 execution authority header differs")
    for name in (*_CODE_RECORD_FIELDS, "base_v1_execution_authority"):
        authority[name] = _record(authority[name], label=name)
    return authority


def prepare_inputs(path: str | Path, *, expected_sha256: str) -> PreparedInputs:
    raw, digest, source_path = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="factorized-native contrast V2 execution authority",
    )
    authority = validate_execution_authority(raw)
    for name, expected_path in _expected_code_paths().items():
        observed = validate_file_record(authority[name], label=name)
        if observed != expected_path.resolve():
            raise ValueError(f"contrast V2 authority binds another {name}")
    base = authority["base_v1_execution_authority"]
    source = legacy.prepare_inputs(base["path"], expected_sha256=base["sha256"])
    if (
        tuple(item.scene_id for item in source.train) != TRAIN_SCENES
        or tuple(item.scene_id for item in source.validation) != VALIDATION_SCENES
    ):
        raise ValueError("contrast V2 base source cohort differs")
    authority["verified_path"] = str(source_path)
    authority["verified_sha256"] = digest
    return PreparedInputs(authority=authority, source=source)


def _scene_teacher_prototypes(scene: legacy.LoadedScene) -> torch.Tensor:
    rows = legacy._scene_rows(scene)
    chunks: list[torch.Tensor] = []
    for start in range(0, rows.numel(), EVAL_BATCH_ROWS):
        teachers, mask = sparse_teacher.gather_sparse_teacher_batch(
            scene.shard, rows[start : start + EVAL_BATCH_ROWS]
        )
        chunks.append(contrast.teacher_prototype(teachers, mask).cpu())
    return torch.cat(chunks).contiguous()


@torch.no_grad()
def build_contrast_reference(
    train_scenes: Sequence[legacy.LoadedScene], *, cohort_sha256: str
) -> dict[str, Any]:
    if tuple(scene.binding.scene_id for scene in train_scenes) != TRAIN_SCENES:
        raise ValueError("contrast center requires the exact source-train cohort")
    prototypes = [_scene_teacher_prototypes(scene) for scene in train_scenes]
    center = contrast.fit_equal_scene_teacher_center(prototypes)
    reference = {
        "schema": CONTRAST_REFERENCE_SCHEMA,
        "schema_version": 1,
        "fit_scenes": list(TRAIN_SCENES),
        "heldout_validation_scenes": list(VALIDATION_SCENES),
        "equal_scene_weighting": True,
        "teacher_center": center,
        "teacher_center_norm": float(center.norm()),
        "teacher_center_squared_norm": float(center.square().sum()),
        "source_cohort_authority_sha256": str(cohort_sha256),
        "validation_contribution": False,
        "benchmark_opened": False,
        "source_access": source_access(),
    }
    if not 0.0 < reference["teacher_center_norm"] < 1.0:
        raise ValueError("source-train common component is invalid")
    return reference


def _batch_loss(
    model: FactorizedNativeGaugeStateReadout,
    head: SigLIP2SummaryHead,
    scene: legacy.LoadedScene,
    rows: torch.Tensor,
    teacher_center: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    descriptor = legacy._descriptor_batch(model, head, scene, rows, device)
    teachers, mask = sparse_teacher.gather_sparse_teacher_batch(scene.shard, rows)
    loss, parts = contrast.contrast_preserving_objective(
        descriptor,
        teachers.to(device),
        mask.to(device),
        teacher_center.to(device),
    )
    return loss, {name: float(value.detach().cpu()) for name, value in parts.items()}


@torch.no_grad()
def evaluate(
    model: FactorizedNativeGaugeStateReadout,
    head: SigLIP2SummaryHead,
    scenes: Sequence[legacy.LoadedScene],
    teacher_center: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    per_scene: dict[str, dict[str, Any]] = {}
    for scene in scenes:
        rows = legacy._scene_rows(scene)
        descriptors: list[torch.Tensor] = []
        prototypes: list[torch.Tensor] = []
        raw_cosines: list[torch.Tensor] = []
        for start in range(0, rows.numel(), EVAL_BATCH_ROWS):
            selected = rows[start : start + EVAL_BATCH_ROWS]
            descriptor = legacy._descriptor_batch(
                model, head, scene, selected, device
            )
            teachers, mask = sparse_teacher.gather_sparse_teacher_batch(
                scene.shard, selected
            )
            teachers_device = teachers.to(device)
            mask_device = mask.to(device)
            cosine = torch.einsum(
                "bd,bvd->bv", descriptor, F.normalize(teachers_device, dim=-1)
            )
            descriptors.append(descriptor.cpu())
            prototypes.append(contrast.teacher_prototype(teachers, mask).cpu())
            raw_cosines.append(
                ((cosine * mask_device).sum(dim=1) / mask_device.sum(dim=1)).cpu()
            )
        per_scene[scene.binding.scene_id] = contrast.contrast_metrics_from_prototypes(
            torch.cat(descriptors),
            torch.cat(prototypes),
            teacher_center.cpu(),
            row_all_view_cosine=torch.cat(raw_cosines),
        )
    macro_keys = (
        "mean_all_view_cosine",
        "p05_row_mean_all_view_cosine",
        "mean_teacher_centered_residual_cosine",
        "p05_teacher_centered_residual_cosine",
        "student_to_teacher_spread_ratio",
        "teacher_centered_pair_gram_mae",
        "teacher_centered_pair_gram_correlation",
        "absolute_visual_probe_response_mae",
        "absolute_visual_probe_response_correlation",
        "absolute_visual_probe_response_std_ratio",
    )
    result: dict[str, Any] = {
        "scene_count": len(per_scene),
        "per_scene": per_scene,
        "validation_no_grad": not torch.is_grad_enabled(),
        "benchmark_opened": False,
    }
    for key in macro_keys:
        result[f"macro_{key}"] = sum(
            float(row[key]) for row in per_scene.values()
        ) / len(per_scene)
    return result


def attach_selection(
    validation: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    if set(validation.get("per_scene", {})) != set(VALIDATION_SCENES) or set(
        baseline.get("per_scene", {})
    ) != set(VALIDATION_SCENES):
        raise ValueError("contrast V2 gate requires both heldout source scenes")
    raw = float(validation["macro_mean_all_view_cosine"])
    raw_baseline = float(baseline["macro_mean_all_view_cosine"])
    raw_relative = (raw - raw_baseline) / max(abs(raw_baseline), 1e-12)
    residual_improvement = float(
        validation["macro_mean_teacher_centered_residual_cosine"]
    ) - float(baseline["macro_mean_teacher_centered_residual_cosine"])
    checks = {
        "raw_macro_relative_improvement_at_least_0p2_percent": (
            raw_relative >= MINIMUM_RELATIVE_RAW_MEAN_IMPROVEMENT
        ),
        "centered_residual_macro_absolute_improvement_at_least_0p02": (
            residual_improvement >= MINIMUM_CENTERED_RESIDUAL_IMPROVEMENT
        ),
        "every_scene_raw_mean_and_p05_non_regression": all(
            float(validation["per_scene"][scene][metric])
            + RAW_NON_REGRESSION_TOLERANCE
            >= float(baseline["per_scene"][scene][metric])
            for scene in VALIDATION_SCENES
            for metric in ("mean_all_view_cosine", "p05_row_mean_all_view_cosine")
        ),
        "every_scene_centered_mean_and_p05_non_regression": all(
            float(validation["per_scene"][scene][metric])
            + RAW_NON_REGRESSION_TOLERANCE
            >= float(baseline["per_scene"][scene][metric])
            for scene in VALIDATION_SCENES
            for metric in (
                "mean_teacher_centered_residual_cosine",
                "p05_teacher_centered_residual_cosine",
            )
        ),
        "every_scene_centroid_dispersion_at_least_0p75_teacher": all(
            float(
                validation["per_scene"][scene][
                    "student_to_teacher_spread_ratio"
                ]
            )
            >= contrast.MINIMUM_SPREAD_RATIO
            for scene in VALIDATION_SCENES
        ),
        "every_scene_pair_gram_correlation_at_least_0p20": all(
            float(
                validation["per_scene"][scene][
                    "teacher_centered_pair_gram_correlation"
                ]
            )
            >= MINIMUM_PAIR_GRAM_CORRELATION
            for scene in VALIDATION_SCENES
        ),
        "every_scene_pair_geometry_non_regression": all(
            float(
                validation["per_scene"][scene][
                    "teacher_centered_pair_gram_mae"
                ]
            )
            <= float(
                baseline["per_scene"][scene][
                    "teacher_centered_pair_gram_mae"
                ]
            )
            + RAW_NON_REGRESSION_TOLERANCE
            and float(
                validation["per_scene"][scene][
                    "teacher_centered_pair_gram_correlation"
                ]
            )
            + RAW_NON_REGRESSION_TOLERANCE
            >= float(
                baseline["per_scene"][scene][
                    "teacher_centered_pair_gram_correlation"
                ]
            )
            for scene in VALIDATION_SCENES
        ),
        "every_scene_absolute_visual_probe_calibrated": all(
            float(
                validation["per_scene"][scene][
                    "absolute_visual_probe_response_mae"
                ]
            )
            <= float(
                baseline["per_scene"][scene][
                    "absolute_visual_probe_response_mae"
                ]
            )
            + RAW_NON_REGRESSION_TOLERANCE
            and float(
                validation["per_scene"][scene][
                    "absolute_visual_probe_response_correlation"
                ]
            )
            >= MINIMUM_ABSOLUTE_PROBE_CORRELATION
            and float(
                validation["per_scene"][scene][
                    "absolute_visual_probe_response_correlation"
                ]
            )
            + RAW_NON_REGRESSION_TOLERANCE
            >= float(
                baseline["per_scene"][scene][
                    "absolute_visual_probe_response_correlation"
                ]
            )
            and MINIMUM_ABSOLUTE_PROBE_STD_RATIO
            <= float(
                validation["per_scene"][scene][
                    "absolute_visual_probe_response_std_ratio"
                ]
            )
            <= MAXIMUM_ABSOLUTE_PROBE_STD_RATIO
            for scene in VALIDATION_SCENES
        ),
        "every_scene_has_two_rows": all(
            int(validation["per_scene"][scene]["eligible_rows"]) >= 2
            for scene in VALIDATION_SCENES
        ),
    }
    return {
        **dict(validation),
        "selection": {
            "raw_relative_macro_mean_improvement": raw_relative,
            "centered_residual_absolute_macro_mean_improvement": (
                residual_improvement
            ),
            "checks": checks,
            "eligible": all(checks.values()),
        },
    }


def select_step(history: Sequence[Mapping[str, Any]]) -> int | None:
    if not history or int(history[0].get("step", -1)) != 0:
        raise ValueError("contrast V2 evaluation history must start at step zero")
    steps = [int(row.get("step", -1)) for row in history]
    expected = [0, *range(EVALUATION_INTERVAL, OPTIMIZER_STEPS + 1, EVALUATION_INTERVAL)]
    if steps != expected:
        raise ValueError("contrast V2 evaluation history schedule differs")
    eligible = [
        row
        for row in history
        if row.get("validation", {}).get("selection", {}).get("eligible") is True
    ]
    if not eligible:
        return None
    selected = max(
        eligible,
        key=lambda row: (
            float(
                row["validation"][
                    "macro_mean_teacher_centered_residual_cosine"
                ]
            ),
            float(
                row["validation"][
                    "macro_teacher_centered_pair_gram_correlation"
                ]
            ),
            float(row["validation"]["macro_mean_all_view_cosine"]),
            -int(row["step"]),
        ),
    )
    return int(selected["step"])


def _state_copy(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return legacy._state_copy(model)


def _state_sha(state: Mapping[str, torch.Tensor]) -> str:
    return legacy._state_sha(state)


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
        raise FileExistsError("contrast V2 first-writer output already exists")
    prepared = prepare_inputs(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
    )
    train_scenes = [legacy.load_scene(binding) for binding in prepared.source.train]
    validation_scenes = [
        legacy.load_scene(binding) for binding in prepared.source.validation
    ]
    normalization = interface.build_source_normalization(
        [scene.state for scene in train_scenes],
        source_state_cohort_authority_sha256=prepared.source.registry[
            "authority_sha256"
        ],
    )
    reference = build_contrast_reference(
        train_scenes,
        cohort_sha256=prepared.source.registry["authority_sha256"],
    )
    teacher_center = reference["teacher_center"]
    device = torch.device(str(args.device))
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    model = interface.build_model(DIRECTION_ONLY, normalization).to(device)
    head = SigLIP2SummaryHead.from_radio_checkpoint(
        prepared.source.authority["verified_radio_path"],
        expected_sha256=prepared.source.authority["official_radio_checkpoint"][
            "sha256"
        ],
    ).to(device).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    baseline = evaluate(model, head, validation_scenes, teacher_center, device)
    zero_state = _state_copy(model)
    history: list[dict[str, Any]] = [
        {
            "step": 0,
            "training_scene_objective": None,
            "validation": attach_selection(baseline, baseline),
            "model_state_dict_sha256": _state_sha(zero_state),
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
            loss, parts = _batch_loss(
                model, head, scene, rows, teacher_center, device
            )
            (loss / len(train_scenes)).backward()
            scene_objective[scene.binding.scene_id] = {
                "total": float(loss.detach().cpu()),
                **parts,
            }
        torch.nn.utils.clip_grad_norm_(
            tuple(model.parameters()), MAX_GRADIENT_NORM, error_if_nonfinite=True
        )
        optimizer.step()
        last_training = {"step": step, "per_scene": scene_objective}
        if step % EVALUATION_INTERVAL != 0:
            continue
        validation = attach_selection(
            evaluate(model, head, validation_scenes, teacher_center, device),
            baseline,
        )
        state = _state_copy(model)
        entry = {
            "step": step,
            "training_scene_objective": scene_objective,
            "validation": validation,
            "model_state_dict_sha256": _state_sha(state),
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
            "schema_version": 2,
            "training_contract": contract,
            "training_contract_sha256": canonical_json_sha256(contract),
            "interface_contract_sha256": interface.INTERFACE_CONTRACT_SHA256,
            "model_architecture": model.architecture(
                interface.INTERFACE_CONTRACT_SHA256
            ),
            "model_state_dict": selected_state,
            "model_state_dict_sha256": _state_sha(selected_state),
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
        "schema_version": 2,
        "status": (
            "source_only_contrast_v2_promotion_candidate_complete"
            if selected is not None
            else "source_only_contrast_v2_complete_no_eligible_candidate"
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
    torch.manual_seed(SEED)
    common = F.normalize(torch.randn(16), dim=0) * 0.95
    residual = F.normalize(torch.randn(8, 16), dim=-1) * 0.25
    teacher = F.normalize(common + residual, dim=-1)
    views = teacher[:, None].repeat(1, 2, 1)
    mask = torch.ones(8, 2, dtype=torch.bool)
    loss, parts = contrast.contrast_preserving_objective(
        teacher.clone().requires_grad_(True), views, mask, common
    )
    return {
        "schema": "radio_gs.factorized_native_contrast_v2_synthetic_dry_run.v1",
        "loss_finite": bool(torch.isfinite(loss)),
        "objective_components": sorted(parts),
        "train_scenes": list(TRAIN_SCENES),
        "validation_scenes": list(VALIDATION_SCENES),
        "evaluation_steps": [
            0,
            *range(EVALUATION_INTERVAL, OPTIMIZER_STEPS + 1, EVALUATION_INTERVAL),
        ],
        "legacy_v1_modified": False,
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
            "status": "source_only_contrast_v2_authority_validated",
            "source_train": [item.scene_id for item in prepared.source.train],
            "source_validation": [
                item.scene_id for item in prepared.source.validation
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
    "CONTRAST_REFERENCE_SCHEMA",
    "EXECUTION_AUTHORITY_SCHEMA",
    "RESULT_SCHEMA",
    "attach_selection",
    "build_contrast_reference",
    "build_parser",
    "evaluate",
    "prepare_inputs",
    "select_step",
    "source_access",
    "synthetic_dry_run",
    "train",
    "training_contract",
    "validate_execution_authority",
]
