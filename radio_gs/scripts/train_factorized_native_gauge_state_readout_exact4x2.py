#!/usr/bin/env python3
"""Train one query-free factorized-native gauge/state readout arm on exact 4+2.

This is an independent opt-in candidate.  It does not load, modify, or replace
the accepted V2 readout and it has no target/benchmark execution command.
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
from radio_gs.interfaces.factorized_primitive_state import (
    FactorizedPrimitiveState,
    load_factorized_primitive_state,
)
from radio_gs.interfaces.surface_region_summary import surface_region_state_dict_sha256
from radio_gs.models.factorized_native_gauge_state_readout import (
    FACTORIZED_NATIVE_READOUT_ARMS,
    FactorizedNativeGaugeStateReadout,
)
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.scripts import materialize_full_scalar_clean_training_shard as pilot_shard
from radio_gs.scripts import train_surface_region_full_scalar_residual as base_trainer
from radio_gs.scripts import train_surface_region_typed_context_residual as sparse_teacher
from radio_gs.scripts import (
    train_surface_region_typed_context_response_listwise_v21_pilot as exact4x2,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


EXECUTION_AUTHORITY_SCHEMA = (
    "radio_gs.factorized_native_gauge_state_readout_exact4x2_"
    "execution_authority.v1"
)
CHECKPOINT_SCHEMA = "radio_gs.factorized_native_gauge_state_readout_checkpoint.v1"
RESULT_SCHEMA = "radio_gs.factorized_native_gauge_state_readout_source_result.v1"
TRAIN_SCENES = exact4x2.TRAIN_SCENES
VALIDATION_SCENES = exact4x2.VALIDATION_SCENES
OPTIMIZER_STEPS = 30
BATCH_ROWS = 64
EVAL_BATCH_ROWS = 256
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
MAX_GRADIENT_NORM = 1.0
NON_REGRESSION_TOLERANCE = 1e-7
MINIMUM_RELATIVE_MEAN_IMPROVEMENT = 0.002
SEED = 0


@dataclass(frozen=True)
class SceneBinding:
    split: str
    scene_id: str
    training_shard: dict[str, str]
    accepted_region_authority: dict[str, str]
    factorized_state: dict[str, str]


@dataclass(frozen=True)
class PreparedInputs:
    authority: dict[str, Any]
    train: tuple[SceneBinding, ...]
    validation: tuple[SceneBinding, ...]
    registry: dict[str, Any]


@dataclass(frozen=True)
class LoadedScene:
    binding: SceneBinding
    shard: dict[str, Any]
    accepted: dict[str, Any]
    state: FactorizedPrimitiveState


def source_access() -> dict[str, bool]:
    return dict(interface.source_access())


def training_contract(arm: str) -> dict[str, Any]:
    arm_value = str(arm)
    if arm_value not in FACTORIZED_NATIVE_READOUT_ARMS:
        raise ValueError("unsupported factorized-native readout arm")
    return {
        "schema_version": 1,
        "artifact_type": "factorized_native_gauge_state_readout_source_only",
        "interface_contract_sha256": interface.INTERFACE_CONTRACT_SHA256,
        "arm": arm_value,
        "cohort": {
            "source_train": list(TRAIN_SCENES),
            "source_validation": list(VALIDATION_SCENES),
            "registry": "frozen_exact4train_2validation",
            "scene_and_physical_space_disjoint": True,
        },
        "input": {
            "teacher_shard_schema": pilot_shard.PILOT_TRAINING_SHARD_SCHEMA,
            "registry_schema": pilot_shard.PILOT_COHORT_REGISTRY_SCHEMA,
            "accepted_regions": "caller_sha_bound_canonical_v2_region_rows_only",
            "factorized_state": "caller_sha_bound_exact_marginal_schema_v2",
            "raw_radio_vector": "prohibited",
            "query_or_text": "prohibited",
        },
        "model": {
            "class": "FactorizedNativeGaugeStateReadout",
            "output": "1280_summary_token_then_frozen_official_siglip2_head",
            "descriptor_dim": 1536,
            "scene_parameters": False,
        },
        "normalization": {
            "fit": "source_train_compact_state_rows_only",
            "known_value_aware": True,
            "validation_contribution": False,
        },
        "objective": {
            "all_view_cosine": 1.0,
            "query_free": True,
            "equal_scene_gradient_accumulation": True,
        },
        "optimizer": {
            "name": "AdamW",
            "steps": OPTIMIZER_STEPS,
            "batch_rows_per_scene_per_step": BATCH_ROWS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "maximum_gradient_norm": MAX_GRADIENT_NORM,
        },
        "selection": {
            "validation_only": True,
            "step_zero_candidate": True,
            "every_scene_mean_and_p05_non_regression": True,
            "minimum_relative_macro_mean_improvement": (
                MINIMUM_RELATIVE_MEAN_IMPROVEMENT
            ),
            "ranking": ["maximum_macro_mean_cosine", "maximum_macro_p05", "earliest_step"],
        },
        "legacy_accepted_v2_default_changed": False,
        "benchmark_execution_authorized": False,
        "source_access": source_access(),
    }


_CODE_RECORD_FIELDS = (
    "trainer_implementation",
    "model_implementation",
    "interface_implementation",
    "pilot_asset_loader_implementation",
)


def _expected_code_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[2]
    return {
        "trainer_implementation": Path(__file__).resolve(),
        "model_implementation": root / "radio_gs/models/factorized_native_gauge_state_readout.py",
        "interface_implementation": (
            root / "radio_gs/interfaces/factorized_native_gauge_state_readout.py"
        ),
        "pilot_asset_loader_implementation": Path(pilot_shard.__file__).resolve(),
    }


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    path = str(value["path"])
    digest = pilot_shard._require_sha256(value["sha256"], label=label)
    if not path:
        raise ValueError(f"{label} file record differs")
    return {"path": path, "sha256": digest}


def _scene_records(value: object, *, split: str) -> tuple[SceneBinding, ...]:
    expected = TRAIN_SCENES if split == "source_train" else VALIDATION_SCENES
    if not isinstance(value, list) or len(value) != len(expected):
        raise ValueError(f"{split} requires the exact frozen scene cohort")
    result: list[SceneBinding] = []
    required = {
        "scene_id",
        "training_shard",
        "accepted_region_authority",
        "factorized_state",
    }
    for raw, scene in zip(value, expected):
        if not isinstance(raw, Mapping) or set(raw) != required or raw.get("scene_id") != scene:
            raise ValueError(f"{split} scene records differ")
        result.append(
            SceneBinding(
                split=split,
                scene_id=scene,
                training_shard=_record(raw["training_shard"], label="training shard"),
                accepted_region_authority=_record(
                    raw["accepted_region_authority"], label="accepted region authority"
                ),
                factorized_state=_record(raw["factorized_state"], label="factorized state"),
            )
        )
    return tuple(result)


def validate_execution_authority(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("factorized-native execution authority must be a mapping")
    authority = dict(value)
    required = {
        "schema",
        "schema_version",
        "status",
        *_CODE_RECORD_FIELDS,
        "cohort_authority",
        "pilot_cohort_region_view_registry",
        "benchmark_exclusion_manifest",
        "official_radio_checkpoint",
        "source_train",
        "source_validation",
        "authorized_arms",
        "training_authorized",
        "benchmark_execution_authorized",
        "source_access",
    }
    if (
        set(authority) != required
        or authority.get("schema") != EXECUTION_AUTHORITY_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("status") != "authorized_source_only_exact4train_2validation"
        or authority.get("authorized_arms") != list(FACTORIZED_NATIVE_READOUT_ARMS)
        or authority.get("training_authorized") is not True
        or authority.get("benchmark_execution_authorized") is not False
        or authority.get("source_access") != source_access()
    ):
        raise ValueError("factorized-native execution authority header differs")
    for name in (
        *_CODE_RECORD_FIELDS,
        "cohort_authority",
        "pilot_cohort_region_view_registry",
        "benchmark_exclusion_manifest",
        "official_radio_checkpoint",
    ):
        authority[name] = _record(authority[name], label=name)
    authority["source_train"] = [
        {
            "scene_id": item.scene_id,
            "training_shard": item.training_shard,
            "accepted_region_authority": item.accepted_region_authority,
            "factorized_state": item.factorized_state,
        }
        for item in _scene_records(authority["source_train"], split="source_train")
    ]
    authority["source_validation"] = [
        {
            "scene_id": item.scene_id,
            "training_shard": item.training_shard,
            "accepted_region_authority": item.accepted_region_authority,
            "factorized_state": item.factorized_state,
        }
        for item in _scene_records(
            authority["source_validation"], split="source_validation"
        )
    ]
    return authority


def _validated_record(record: Mapping[str, str], *, label: str) -> Path:
    return validate_file_record(record, label=label)


def prepare_inputs(path: str | Path, *, expected_sha256: str) -> PreparedInputs:
    raw, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="factorized-native source execution authority",
    )
    authority = validate_execution_authority(raw)
    for name, expected in _expected_code_paths().items():
        observed = _validated_record(authority[name], label=name)
        if observed != expected.resolve():
            raise ValueError(f"execution authority binds another {name}")
    radio = _validated_record(
        authority["official_radio_checkpoint"], label="official RADIO checkpoint"
    )
    if (
        authority["official_radio_checkpoint"]["sha256"]
        != pilot_shard.OFFICIAL_RADIO_CHECKPOINT_SHA256
    ):
        raise ValueError("official RADIO checkpoint singleton differs")

    cohort, cohort_file = base_trainer.load_cohort_authority(
        _validated_record(authority["cohort_authority"], label="cohort authority"),
        expected_sha256=authority["cohort_authority"]["sha256"],
    )
    registry_raw, registry_digest, registry_path = load_json_object(
        _validated_record(
            authority["pilot_cohort_region_view_registry"], label="pilot registry"
        ),
        expected_sha256=authority["pilot_cohort_region_view_registry"]["sha256"],
        label="pilot exact4x2 registry",
    )
    registry = pilot_shard.validate_pilot_cohort_region_view_registry(
        registry_raw,
        cohort_authority=cohort,
        cohort_authority_file_sha256=cohort_file["sha256"],
    )
    if authority["pilot_cohort_region_view_registry"] != {
        "path": str(registry_path),
        "sha256": registry_digest,
    }:
        raise ValueError("pilot registry verified file record differs")
    # Validate the exclusion file itself and its binding from the parent cohort.
    exclusion, exclusion_file = base_trainer._load_json_manifest(
        _validated_record(
            authority["benchmark_exclusion_manifest"], label="benchmark exclusion"
        ),
        expected_sha256=authority["benchmark_exclusion_manifest"]["sha256"],
        validator=base_trainer.validate_benchmark_exclusion_manifest,
        label="benchmark exclusion manifest",
    )
    if (
        cohort["benchmark_exclusion"]["manifest_authority_sha256"]
        != exclusion["authority_sha256"]
        or cohort["benchmark_exclusion"]["manifest_file_sha256"]
        != exclusion_file["sha256"]
    ):
        raise ValueError("benchmark exclusion differs from cohort authority")

    registry_by_scene = {
        str(item["scene_id"]): item for item in registry["scene_records"]
    }
    train = _scene_records(authority["source_train"], split="source_train")
    validation = _scene_records(
        authority["source_validation"], split="source_validation"
    )
    for binding in (*train, *validation):
        registry_scene = registry_by_scene.get(binding.scene_id)
        if registry_scene is None:
            raise ValueError("execution scene is absent from exact4x2 registry")
        for record, expected_file_sha, label in (
            (
                binding.accepted_region_authority,
                registry_scene["accepted_region_authority_file_sha256"],
                "accepted region authority",
            ),
            (
                binding.factorized_state,
                registry_scene["factorized_state_file_sha256"],
                "factorized state",
            ),
        ):
            _validated_record(record, label=f"{binding.scene_id} {label}")
            if record["sha256"] != expected_file_sha:
                raise ValueError(f"{binding.scene_id} {label} differs from registry")
        _validated_record(
            binding.training_shard, label=f"{binding.scene_id} training shard"
        )
    authority["verified_path"] = str(source)
    authority["verified_sha256"] = digest
    authority["verified_radio_path"] = str(radio)
    return PreparedInputs(authority, train, validation, registry)


def load_scene(binding: SceneBinding) -> LoadedScene:
    shard, shard_record = pilot_shard.load_pilot_training_shard(
        binding.training_shard["path"],
        expected_sha256=binding.training_shard["sha256"],
        expected_split=binding.split,
    )
    accepted_raw, accepted_sha, accepted_path = load_torch_mapping(
        binding.accepted_region_authority["path"],
        expected_sha256=binding.accepted_region_authority["sha256"],
        map_location="cpu",
        label="factorized-native accepted region authority",
    )
    accepted = pilot_shard.validate_accepted_region_authority(accepted_raw)
    state = load_factorized_primitive_state(
        binding.factorized_state["path"],
        expected_sha256=binding.factorized_state["sha256"],
    )
    geometry = accepted["input_authority"]["geometry_authority"]
    expected_region_ids = [
        pilot_shard.stable_region_id(binding.scene_id, fingerprint)
        for fingerprint in accepted["region_fingerprints"]
    ]
    if (
        shard_record != binding.training_shard
        or {"path": str(accepted_path), "sha256": accepted_sha}
        != binding.accepted_region_authority
        or accepted["scene_id"] != binding.scene_id
        or set(shard["scene_ids"]) != {binding.scene_id}
        or shard["region_row_ids"] != expected_region_ids
        or geometry["factorized_primitive_state_file_sha256"] != state.sha256
        or state.sha256 != binding.factorized_state["sha256"]
        or state.metadata.get("query_independent") is not True
        or any(
            state.metadata.get(key) is not False
            for key in (
                "benchmark_images_opened",
                "benchmark_masks_opened",
                "text_queries_opened",
            )
        )
    ):
        raise ValueError("factorized-native scene lineage or row alignment differs")
    if not torch.equal(shard["accepted_v2_e0"], accepted["accepted_v2_e0"]):
        raise ValueError("teacher shard and accepted authority selected rows differ")
    return LoadedScene(binding, shard, accepted, state)


def _scene_rows(scene: LoadedScene) -> torch.Tensor:
    rows = torch.where(scene.shard["eligible"])[0]
    if rows.numel() < 2:
        raise ValueError("factorized-native scene lacks two eligible teacher rows")
    return rows


def _region_inputs(scene: LoadedScene, rows: torch.Tensor):
    selected = torch.as_tensor(rows).detach().long().cpu()
    return interface.gather_factorized_native_region_inputs(
        scene.state,
        scene.accepted["region_rows"][selected],
        scene.accepted["token_mask"][selected],
        scene.accepted["anchor_index"][selected],
    )


def _descriptor_batch(
    model: FactorizedNativeGaugeStateReadout,
    head: SigLIP2SummaryHead,
    scene: LoadedScene,
    rows: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    inputs = _region_inputs(scene, rows)
    summary = model(
        inputs.unit_direction.to(device),
        inputs.log_amplitude.to(device),
        inputs.state.to(device),
        inputs.state_known_mask.to(device),
        token_mask=inputs.token_mask.to(device),
        anchor_index=inputs.anchor_index.to(device),
    )
    return F.normalize(head(summary[:, None])[:, 0].float(), dim=-1)


def _batch_loss(
    model: FactorizedNativeGaugeStateReadout,
    head: SigLIP2SummaryHead,
    scene: LoadedScene,
    rows: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    descriptor = _descriptor_batch(model, head, scene, rows, device)
    teachers, teacher_mask = sparse_teacher.gather_sparse_teacher_batch(
        scene.shard, rows
    )
    cosine = (descriptor[:, None] * teachers.to(device)).sum(dim=-1)
    row_mean = (cosine * teacher_mask.to(device)).sum(dim=1) / teacher_mask.sum(
        dim=1
    ).to(device)
    return 1.0 - row_mean.mean()


@torch.no_grad()
def evaluate(
    model: FactorizedNativeGaugeStateReadout,
    head: SigLIP2SummaryHead,
    scenes: Sequence[LoadedScene],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    per_scene: dict[str, dict[str, float | int]] = {}
    for scene in scenes:
        values: list[torch.Tensor] = []
        eligible = _scene_rows(scene)
        for start in range(0, eligible.numel(), EVAL_BATCH_ROWS):
            rows = eligible[start : start + EVAL_BATCH_ROWS]
            descriptor = _descriptor_batch(model, head, scene, rows, device)
            teachers, teacher_mask = sparse_teacher.gather_sparse_teacher_batch(
                scene.shard, rows
            )
            cosine = (descriptor[:, None] * teachers.to(device)).sum(dim=-1)
            values.append(
                (
                    (cosine * teacher_mask.to(device)).sum(dim=1)
                    / teacher_mask.sum(dim=1).to(device)
                ).cpu()
            )
        row_cosine = torch.cat(values)
        per_scene[scene.binding.scene_id] = {
            "eligible_rows": int(row_cosine.numel()),
            "mean_all_view_cosine": float(row_cosine.mean()),
            "p05_row_mean_all_view_cosine": float(
                torch.quantile(row_cosine, 0.05)
            ),
        }
    return {
        "scene_count": len(per_scene),
        "macro_mean_all_view_cosine": sum(
            float(row["mean_all_view_cosine"]) for row in per_scene.values()
        )
        / len(per_scene),
        "macro_p05_row_mean_all_view_cosine": sum(
            float(row["p05_row_mean_all_view_cosine"])
            for row in per_scene.values()
        )
        / len(per_scene),
        "per_scene": per_scene,
        "validation_no_grad": not torch.is_grad_enabled(),
        "benchmark_opened": False,
    }


def attach_selection(
    validation: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    current_mean = float(validation["macro_mean_all_view_cosine"])
    baseline_mean = float(baseline["macro_mean_all_view_cosine"])
    relative = (current_mean - baseline_mean) / max(abs(baseline_mean), 1e-12)
    checks = {
        "macro_mean_relative_improvement_at_least_0p2_percent": (
            relative >= MINIMUM_RELATIVE_MEAN_IMPROVEMENT
        ),
        "macro_p05_non_regression": float(
            validation["macro_p05_row_mean_all_view_cosine"]
        )
        + NON_REGRESSION_TOLERANCE
        >= float(baseline["macro_p05_row_mean_all_view_cosine"]),
        "every_scene_mean_and_p05_non_regression": all(
            float(validation["per_scene"][scene][metric])
            + NON_REGRESSION_TOLERANCE
            >= float(baseline["per_scene"][scene][metric])
            for scene in VALIDATION_SCENES
            for metric in (
                "mean_all_view_cosine",
                "p05_row_mean_all_view_cosine",
            )
        ),
        "every_scene_has_two_rows": all(
            int(validation["per_scene"][scene]["eligible_rows"]) >= 2
            for scene in VALIDATION_SCENES
        ),
    }
    return {
        **dict(validation),
        "selection": {
            "relative_macro_mean_improvement": relative,
            "checks": checks,
            "eligible": all(checks.values()),
        },
    }


def select_step(history: Sequence[Mapping[str, Any]]) -> int | None:
    if not history or [int(row.get("step", -1)) for row in history] != list(
        range(len(history))
    ):
        raise ValueError("factorized-native history must be contiguous")
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
            float(row["validation"]["macro_mean_all_view_cosine"]),
            float(row["validation"]["macro_p05_row_mean_all_view_cosine"]),
            -int(row["step"]),
        ),
    )
    return int(selected["step"])


def _state_copy(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().contiguous().clone()
        for name, value in model.state_dict().items()
    }


def _state_sha(state: Mapping[str, torch.Tensor]) -> str:
    return surface_region_state_dict_sha256(state)


def _cyclic_batch(rows: torch.Tensor, *, step: int) -> torch.Tensor:
    count = int(rows.numel())
    width = min(BATCH_ROWS, count)
    start = ((int(step) - 1) * BATCH_ROWS) % count
    offsets = torch.arange(width)
    return rows[(start + offsets) % count]


def train(args: argparse.Namespace) -> dict[str, Any]:
    arm = str(args.arm)
    contract = training_contract(arm)
    output = Path(args.output).expanduser().resolve()
    result_path = output.with_suffix(output.suffix + ".json")
    normalization_path = output.with_suffix(output.suffix + ".normalization.pt")
    if any(
        path.exists() or path.is_symlink()
        for path in (output, result_path, normalization_path)
    ):
        raise FileExistsError("factorized-native first-writer output already exists")
    prepared = prepare_inputs(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
    )
    train_scenes = [load_scene(binding) for binding in prepared.train]
    validation_scenes = [load_scene(binding) for binding in prepared.validation]
    normalization = interface.build_source_normalization(
        [scene.state for scene in train_scenes],
        source_state_cohort_authority_sha256=prepared.registry["authority_sha256"],
    )
    device = torch.device(str(args.device))
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    model = interface.build_model(arm, normalization).to(device)
    head = SigLIP2SummaryHead.from_radio_checkpoint(
        prepared.authority["verified_radio_path"],
        expected_sha256=prepared.authority["official_radio_checkpoint"]["sha256"],
    ).to(device).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    baseline = evaluate(model, head, validation_scenes, device)
    zero_state = _state_copy(model)
    history: list[dict[str, Any]] = [
        {
            "step": 0,
            "training_scene_loss": None,
            "validation": attach_selection(baseline, baseline),
            "model_state_dict_sha256": _state_sha(zero_state),
        }
    ]
    saved_states: dict[int, dict[str, torch.Tensor]] = {0: zero_state}
    for step in range(1, OPTIMIZER_STEPS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        scene_losses: dict[str, float] = {}
        for scene in train_scenes:
            rows = _cyclic_batch(_scene_rows(scene), step=step)
            loss = _batch_loss(model, head, scene, rows, device)
            (loss / len(train_scenes)).backward()
            scene_losses[scene.binding.scene_id] = float(loss.detach().cpu())
        torch.nn.utils.clip_grad_norm_(
            tuple(model.parameters()), MAX_GRADIENT_NORM, error_if_nonfinite=True
        )
        optimizer.step()
        validation = attach_selection(
            evaluate(model, head, validation_scenes, device), baseline
        )
        state = _state_copy(model)
        history.append(
            {
                "step": step,
                "training_scene_loss": scene_losses,
                "validation": validation,
                "model_state_dict_sha256": _state_sha(state),
            }
        )
        if validation["selection"]["eligible"] is True:
            saved_states[step] = state
        print(json.dumps(history[-1], sort_keys=True), flush=True)
    selected = select_step(history)
    checkpoint_record: dict[str, str] | None = None
    normalization_file = write_torch_noclobber(normalization_path, normalization)
    if selected is not None:
        selected_state = saved_states[selected]
        checkpoint = {
            "schema": CHECKPOINT_SCHEMA,
            "schema_version": 1,
            "training_contract": contract,
            "training_contract_sha256": canonical_json_sha256(contract),
            "interface_contract_sha256": interface.INTERFACE_CONTRACT_SHA256,
            "model_architecture": model.architecture(interface.INTERFACE_CONTRACT_SHA256),
            "model_state_dict": selected_state,
            "model_state_dict_sha256": _state_sha(selected_state),
            "normalization": file_record(normalization_file),
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
        "schema_version": 1,
        "status": (
            "source_only_promotion_candidate_complete"
            if selected is not None
            else "source_only_complete_no_eligible_candidate"
        ),
        "arm": arm,
        "training_contract": contract,
        "training_contract_sha256": canonical_json_sha256(contract),
        "execution_authority": {
            "path": prepared.authority["verified_path"],
            "sha256": prepared.authority["verified_sha256"],
        },
        "normalization": file_record(normalization_file),
        "checkpoint": checkpoint_record,
        "selected_step": selected,
        "history": history,
        "benchmark_opened": False,
        "source_access": source_access(),
    }
    write_frozen_json(result_path, report)
    return report


def synthetic_dry_run() -> dict[str, Any]:
    normalization = {
        "schema": interface.NORMALIZATION_SCHEMA,
        "schema_version": interface.NORMALIZATION_SCHEMA_VERSION,
        "interface_contract_sha256": interface.INTERFACE_CONTRACT_SHA256,
        "source_state_cohort_authority_sha256": "a" * 64,
        "state_names": list(interface.FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES),
        "state_names_sha256": interface.FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES_SHA256,
        "state_median": torch.zeros(6),
        "state_robust_scale": torch.ones(6),
        "log_amplitude_median": torch.zeros(1),
        "log_amplitude_robust_scale": torch.ones(1),
        "known_count_by_state_column": [2] * 6,
        "fit_scene_count": 4,
        "source_access": source_access(),
    }
    direction = F.normalize(torch.randn(2, 3, 1280), dim=-1)
    amplitude = torch.zeros(2, 3)
    state = torch.zeros(2, 3, 6)
    known = torch.ones(2, 3, 6, dtype=torch.bool)
    mask = torch.ones(2, 3, dtype=torch.bool)
    outputs = {}
    for arm in FACTORIZED_NATIVE_READOUT_ARMS:
        model = interface.build_model(arm, normalization)
        outputs[arm] = list(
            model(
                direction,
                amplitude,
                state,
                known,
                token_mask=mask,
                anchor_index=torch.zeros(2, dtype=torch.long),
            ).shape
        )
    return {
        "schema": "radio_gs.factorized_native_gauge_state_synthetic_dry_run.v1",
        "arms": outputs,
        "optimizer_steps": OPTIMIZER_STEPS,
        "train_scenes": list(TRAIN_SCENES),
        "validation_scenes": list(VALIDATION_SCENES),
        "raw_vector_reconstruction": False,
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
    run.add_argument("--arm", choices=FACTORIZED_NATIVE_READOUT_ARMS, required=True)
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
            "status": "source_only_exact4x2_authority_validated",
            "source_train": [item.scene_id for item in prepared.train],
            "source_validation": [item.scene_id for item in prepared.validation],
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
    "SceneBinding",
    "attach_selection",
    "build_parser",
    "load_scene",
    "prepare_inputs",
    "select_step",
    "source_access",
    "synthetic_dry_run",
    "train",
    "training_contract",
    "validate_execution_authority",
]
