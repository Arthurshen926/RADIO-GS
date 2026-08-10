#!/usr/bin/env python3
"""Run the preregistered source-only V2.1A full-30-step rescue candidate."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import torch

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
    train_surface_region_typed_context_response_listwise_v2 as v2_trainer,
)
from radio_gs.scripts import (
    train_surface_region_typed_context_response_listwise_v21_pilot as pilot,
)
from radio_gs.scripts import (
    train_surface_region_typed_context_response_listwise_v21a as v21a_adapter,
)
from radio_gs.scripts import train_surface_region_typed_context_residual as v1_trainer
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    write_frozen_json,
    write_torch_noclobber,
)


ARTIFACT_TYPE = "surface_region_typed_context_response_listwise_v21a_source_rescue"
RESULT_SCHEMA = "radio_gs.surface_region_v21a_rescue_result.v1"
CHECKPOINT_SCHEMA = "radio_gs.surface_region_v21a_rescue_checkpoint.v1"
DIAGNOSTIC_STATE_SCHEMA = "radio_gs.surface_region_v21a_diagnostic_state.v1"
CERTIFICATE_SCHEMA = "radio_gs.surface_region_v21a_rescue_certificate.v1"
EXECUTION_AUTHORITY_SCHEMA = (
    "radio_gs.surface_region_v21a_rescue_execution_authority.v1"
)
PREREGISTRATION = Path(
    "paper/artifacts/"
    "source_global_response_listwise_v21a_anchor_triplet_full30_rescue_"
    "preregistration_20260807.json"
)
EXECUTION_AUTHORITY_ADDENDUM = Path(
    "paper/artifacts/"
    "source_global_response_listwise_v21a_independent_execution_authority_"
    "addendum_20260807.json"
)
OPTIMIZER_STEPS = 30
COVERAGE_THRESHOLD = 0.95
TRAIN_SCENES = pilot.TRAIN_SCENES
VALIDATION_SCENES = pilot.VALIDATION_SCENES
PilotInputs = pilot.PilotInputs


def source_access() -> dict[str, bool]:
    return pilot.source_access()


def training_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "base_v21_training_contract_sha256": pilot.TRAINING_CONTRACT_SHA256,
        "cohort": {
            "source_train": list(TRAIN_SCENES),
            "source_validation": list(VALIDATION_SCENES),
            "same_frozen_v21_execution_authority_and_assets": True,
        },
        "objective_intervention": {
            "continuous_pairwise_training_denominator": (
                "authority_pairs_with_at_least_one_trainable_endpoint"
            ),
            "triplet_training_denominator": (
                "authority_pairs_with_trainable_anchor"
            ),
            "validation_denominator": "all_authority_pairs",
            "all_other_v21_terms_unchanged": True,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": v1_trainer.LEARNING_RATE,
            "weight_decay": v1_trainer.WEIGHT_DECAY,
            "fixed_optimizer_steps": OPTIMIZER_STEPS,
            "early_stopping": False,
            "maximum_gradient_norm": v1_trainer.MAX_GRADIENT_NORM,
            "equal_scene_gradient_accumulation": True,
        },
        "selection": {
            "split": "source_validation",
            "candidate_epochs": list(range(1, OPTIMIZER_STEPS + 1)),
            "all_promotion_constraints_required_before_ranking": True,
            "minimum_active_over_eligible_coverage": COVERAGE_THRESHOLD,
            "minimum_any_endpoint_pair_coverage": COVERAGE_THRESHOLD,
            "primary": "minimum_scene_macro_v21_auxiliary_loss",
            "tie_break": [
                "minimum_scene_macro_absolute_relevance_loss",
                "minimum_scene_macro_continuous_pairwise_relevance_loss",
                "maximum_v1_mean_all_view_cosine",
                "maximum_v1_p05_row_mean_all_view_cosine",
                "maximum_v1_relation_fidelity",
                "earliest_epoch",
            ],
        },
        "diagnostics": {
            "every_epoch": True,
            "angle": ["p50", "p95", "cap_fraction"],
            "routing": "active_over_eligible_and_pair_coverage",
            "gradient_norms": (
                "five_projection_components_plus_global_preclip_and_postclip"
            ),
            "saved_states": ["best_raw_aux", "final", "promotion_selected_if_any"],
        },
        "source_access": source_access(),
        "benchmark_opened": False,
    }


TRAINING_CONTRACT_SHA256 = canonical_json_sha256(training_contract())


def _base_authority_projection(authority: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": pilot.EXECUTION_AUTHORITY_SCHEMA,
        "schema_version": 1,
        "status": "authorized_source_only_4train_2validation_v21_pilot",
        "implementation": dict(authority["parent_implementation"]),
        "active_pair_addendum": dict(authority["active_pair_addendum"]),
        "cohort_authority": dict(authority["cohort_authority"]),
        "pilot_cohort_region_view_registry": dict(
            authority["pilot_cohort_region_view_registry"]
        ),
        "benchmark_exclusion_manifest": dict(
            authority["benchmark_exclusion_manifest"]
        ),
        "fit_text_bank": dict(authority["fit_text_bank"]),
        "canonical_negative_bank": dict(authority["canonical_negative_bank"]),
        "compositional_banks": {
            name: dict(row) for name, row in authority["compositional_banks"].items()
        },
        "typed_relation_authority": dict(authority["typed_relation_authority"]),
        "source_train": [dict(row) for row in authority["source_train"]],
        "source_validation": [dict(row) for row in authority["source_validation"]],
        "training_authorized": True,
        "benchmark_execution_authorized": False,
        "source_access": pilot.source_access(),
    }


def validate_execution_authority(value: object) -> dict[str, Any]:
    required = {
        "schema",
        "schema_version",
        "status",
        "implementation",
        "objective_adapter",
        "loss_implementation",
        "preregistration",
        "execution_authority_addendum",
        "base_v21_asset_execution_authority",
        "parent_implementation",
        "active_pair_addendum",
        "cohort_authority",
        "pilot_cohort_region_view_registry",
        "benchmark_exclusion_manifest",
        "fit_text_bank",
        "canonical_negative_bank",
        "compositional_banks",
        "typed_relation_authority",
        "source_train",
        "source_validation",
        "training_contract_sha256",
        "training_authorized",
        "benchmark_execution_authorized",
        "source_access",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("V2.1A execution authority fields differ")
    authority = dict(value)
    if (
        authority["schema"] != EXECUTION_AUTHORITY_SCHEMA
        or authority["schema_version"] != 1
        or authority["status"]
        != "authorized_source_only_v21a_anchor_triplet_full30_rescue"
        or authority["training_contract_sha256"] != TRAINING_CONTRACT_SHA256
        or authority["training_authorized"] is not True
        or authority["benchmark_execution_authorized"] is not False
        or authority["source_access"] != source_access()
    ):
        raise ValueError("V2.1A execution authority identity differs")
    records = (
        "implementation",
        "objective_adapter",
        "loss_implementation",
        "preregistration",
        "execution_authority_addendum",
        "base_v21_asset_execution_authority",
        "parent_implementation",
        "active_pair_addendum",
        "cohort_authority",
        "pilot_cohort_region_view_registry",
        "benchmark_exclusion_manifest",
        "fit_text_bank",
        "canonical_negative_bank",
    )
    for name in records:
        authority[name] = pilot._record(authority[name], label=f"V2.1A {name}")
    if authority["implementation"] == authority["parent_implementation"]:
        raise ValueError("V2.1A must not reuse the baseline implementation authority")
    projected = pilot.validate_execution_authority(
        _base_authority_projection(authority)
    )
    authority["compositional_banks"] = projected["compositional_banks"]
    authority["typed_relation_authority"] = projected["typed_relation_authority"]
    authority["source_train"] = projected["source_train"]
    authority["source_validation"] = projected["source_validation"]
    return authority


def prepare_inputs(path: str | Path, *, expected_sha256: str) -> PilotInputs:
    raw, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="V2.1A rescue execution authority",
    )
    authority = validate_execution_authority(raw)
    expected_implementations = _implementation_records()
    if any(
        authority[name] != expected_implementations[name]
        for name in expected_implementations
    ):
        raise ValueError("V2.1A execution implementation binding differs")
    parent_record = authority["base_v21_asset_execution_authority"]
    parent_raw, parent_sha, parent_path = load_json_object(
        parent_record["path"],
        expected_sha256=parent_record["sha256"],
        label="V2.1A parent V2.1 asset execution authority",
    )
    parent = pilot.validate_execution_authority(parent_raw)
    if (
        parent != _base_authority_projection(authority)
        or parent_record != {"path": str(parent_path), "sha256": parent_sha}
    ):
        raise ValueError("V2.1A direct asset projection differs from V2.1 parent")
    # The parent authority is an immutable asset manifest only.  It may bind
    # an older baseline trainer implementation, so V2.1A replays every source
    # asset through the current independently bound loaders without treating
    # the baseline implementation record as V2.1A training authorization.
    cohort, cohort_file, _exclusion_file = pilot._validate_subset_authorities(
        _base_authority_projection(authority)
    )
    registry_record = authority["pilot_cohort_region_view_registry"]
    registry_raw, registry_digest, registry_path = load_json_object(
        pilot._validated_path(
            registry_record, label="V2.1A pilot cohort region/view registry"
        ),
        expected_sha256=registry_record["sha256"],
        label="V2.1A pilot cohort region/view registry",
    )
    registry = pilot.pilot_shard.validate_pilot_cohort_region_view_registry(
        registry_raw,
        cohort_authority=cohort,
        cohort_authority_file_sha256=cohort_file["sha256"],
    )
    if registry_record != {"path": str(registry_path), "sha256": registry_digest}:
        raise ValueError("V2.1A pilot registry verified record differs")
    source_state_manifest, teacher_manifest = (
        pilot.pilot_shard.derive_pilot_global_manifests(registry)
    )
    registry_scene_records = {
        str(record["scene_id"]): record for record in registry["scene_records"]
    }
    expected_lineage = {
        "pilot_cohort_region_view_registry_file_sha256": registry_digest,
        "pilot_cohort_region_view_registry_authority_sha256": registry[
            "authority_sha256"
        ],
        "source_state_cohort_authority_sha256": source_state_manifest[
            "authority_sha256"
        ],
        "source_state_manifest_file_sha256": pilot._frozen_json_file_sha256(
            source_state_manifest
        ),
        "cohort_authority_sha256": cohort["authority_sha256"],
        "cohort_authority_file_sha256": cohort_file["sha256"],
        "teacher_authority_sha256": teacher_manifest["authority_sha256"],
        "teacher_manifest_file_sha256": pilot._frozen_json_file_sha256(
            teacher_manifest
        ),
    }
    fit = pilot.v2_trainer.load_fit_text_bank(
        pilot._validated_path(authority["fit_text_bank"], label="V2.1A fit bank"),
        expected_sha256=authority["fit_text_bank"]["sha256"],
    )
    canonical = pilot.load_frozen_canonical_negative_bank(
        pilot._validated_path(
            authority["canonical_negative_bank"], label="V2.1A negative bank"
        ),
        expected_file_sha256=authority["canonical_negative_bank"]["sha256"],
    )
    compositional = tuple(
        pilot.load_frozen_compositional_generic_bank(
            pilot._validated_path(
                {
                    "path": authority["compositional_banks"][name]["path"],
                    "sha256": authority["compositional_banks"][name]["sha256"],
                },
                label=f"V2.1A {name}",
            ),
            expected_file_sha256=authority["compositional_banks"][name]["sha256"],
            component_id=name,
            loss_weight=pilot.COMPONENT_WEIGHTS[name],
        )
        for name in pilot.COMPONENT_WEIGHTS
    )
    relation_record = authority["typed_relation_authority"]
    relations = pilot.load_frozen_typed_text_relation_authority(
        pilot._validated_path(
            {"path": relation_record["path"], "sha256": relation_record["sha256"]},
            label="V2.1A typed relation authority",
        ),
        expected_file_sha256=relation_record["sha256"],
    )
    if relations.content_authority_sha256 != relation_record[
        "content_authority_sha256"
    ]:
        raise ValueError("V2.1A typed relation content authority differs")

    def bind(split: str) -> tuple[v2_trainer.ResponseSceneBinding, ...]:
        records = authority[split]
        bases, summaries = pilot._base_bindings(
            records,
            split=split,
            expected_lineage=expected_lineage,
            registry_scene_records=registry_scene_records,
        )
        result: list[v2_trainer.ResponseSceneBinding] = []
        for base, item, summary in zip(bases, records, summaries):
            response = v2_trainer.bind_response_scene(base, item, fit_text_bank=fit)
            pilot._validate_response_shard_channel_summary(
                response,
                summary,
                registry_scene_records[response.scene_id],
            )
            result.append(response)
        return tuple(result)

    authority["verified_path"] = str(source)
    authority["verified_sha256"] = digest
    return PilotInputs(
        execution=authority,
        train=bind("source_train"),
        validation=bind("source_validation"),
        fit=fit,
        canonical_negative=canonical,
        compositional=compositional,
        relations=relations,
        cohort=cohort,
        registry=registry,
        source_state_manifest=source_state_manifest,
    )


def _objective(
    model: SurfaceRegionAcceptedV2TypedContextResidualV1,
    binding: v2_trainer.ResponseSceneBinding,
    normalization: Mapping[str, Any],
    fit: torch.Tensor,
    inputs: PilotInputs,
    device: torch.device,
    *,
    training: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int | bool]]:
    scene = pilot.load_pilot_bound_scene(binding)
    routing = pilot._pilot_routing(scene, normalization)
    return v21a_adapter.complete_scene_objective_v21a(
        model,
        scene,
        normalization,
        fit,
        binding.response_authority,
        inputs.canonical_negative,
        device,
        compositional_banks=inputs.compositional,
        relation_authority=inputs.relations,
        training=training,
        routing_masks=routing,
    )


def _component_gradient_norms(model: torch.nn.Module) -> dict[str, float]:
    groups = {
        "descriptor_projection": "descriptor_projection.",
        "context_projection": "context_projection.",
        "scalar_projection": "scalar_projection.",
        "fusion_projection": "fusion_projection.",
        "residual_projection": "residual_projection.",
    }
    squared = {name: 0.0 for name in groups}
    global_squared = 0.0
    for parameter_name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().double().square().sum().cpu())
        global_squared += value
        for group, prefix in groups.items():
            if parameter_name.startswith(prefix):
                squared[group] += value
                break
    return {
        **{name: value**0.5 for name, value in squared.items()},
        "global": global_squared**0.5,
    }


def _scene_macro(per_scene: Mapping[str, Mapping[str, Any]], name: str) -> float:
    return sum(float(row[name]) for row in per_scene.values()) / len(per_scene)


def _response_summary(
    per_scene: Mapping[str, Mapping[str, float | int | bool]],
    *,
    validation: bool,
) -> dict[str, Any]:
    return {
        "scene_count": len(per_scene),
        "scene_macro_auxiliary_loss": _scene_macro(
            per_scene, "response_auxiliary_loss"
        ),
        "scene_macro_absolute_relevance_loss": _scene_macro(
            per_scene, "response_absolute_relevance_loss"
        ),
        "scene_macro_continuous_pairwise_relevance_loss": _scene_macro(
            per_scene, "response_continuous_pairwise_relevance_loss"
        ),
        "scene_macro_active_over_eligible_coverage": _scene_macro(
            per_scene, "active_over_eligible_coverage"
        ),
        "minimum_active_over_eligible_coverage": min(
            float(row["active_over_eligible_coverage"])
            for row in per_scene.values()
        ),
        "scene_macro_pair_trainable_endpoint_coverage": _scene_macro(
            per_scene, "response_pair_trainable_endpoint_coverage"
        ),
        "minimum_pair_trainable_endpoint_coverage": min(
            float(row["response_pair_trainable_endpoint_coverage"])
            for row in per_scene.values()
        ),
        "all_authority_pairs_retained": bool(validation),
        "per_scene": dict(per_scene),
    }


def train_one_step(
    model: SurfaceRegionAcceptedV2TypedContextResidualV1,
    optimizer: torch.optim.Optimizer,
    inputs: PilotInputs,
    normalization: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    fit = inputs.fit.embeddings.to(device)
    per_scene: dict[str, dict[str, float | int | bool]] = {}
    for binding in inputs.train:
        total, metrics = _objective(
            model,
            binding,
            normalization,
            fit,
            inputs,
            device,
            training=True,
        )
        (total / len(inputs.train)).backward()
        row = {
            "combined_objective": float(total.detach().cpu()),
            **v2_trainer._float_metrics(metrics),
        }
        if int(row["response_triplet_objective_hard_negative_pairs"]) > int(
            row["response_pairwise_objective_hard_negative_pairs"]
        ):
            raise RuntimeError("triplet denominator exceeds pairwise denominator")
        per_scene[binding.scene_id] = row
    preclip = _component_gradient_norms(model)
    torch.nn.utils.clip_grad_norm_(
        tuple(model.parameters()),
        v1_trainer.MAX_GRADIENT_NORM,
        error_if_nonfinite=True,
    )
    postclip = _component_gradient_norms(model)
    optimizer.step()
    return {
        "optimizer_step_completed": True,
        "scene_count": len(inputs.train),
        "equal_scene_weight": 1.0 / len(inputs.train),
        "complete_scene_forward": True,
        "pairwise_any_trainable_endpoint_filter": True,
        "triplet_anchor_trainable_filter": True,
        "gradient_norms_preclip": preclip,
        "global_gradient_norm_postclip": postclip["global"],
        "response_listwise_v21a": _response_summary(
            per_scene, validation=False
        ),
    }


@torch.no_grad()
def evaluate(
    model: SurfaceRegionAcceptedV2TypedContextResidualV1,
    inputs: PilotInputs,
    normalization: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    fit = inputs.fit.embeddings.to(device)
    per_scene: dict[str, dict[str, float | int | bool]] = {}
    for binding in inputs.validation:
        total, metrics = _objective(
            model,
            binding,
            normalization,
            fit,
            inputs,
            device,
            training=False,
        )
        row = {
            "combined_objective": float(total.detach().cpu()),
            **v2_trainer._float_metrics(metrics),
        }
        authority_pairs = int(row["response_authority_hard_negative_pairs"])
        if (
            int(row["response_pairwise_objective_hard_negative_pairs"])
            != authority_pairs
            or int(row["response_triplet_objective_hard_negative_pairs"])
            != authority_pairs
        ):
            raise RuntimeError("V2.1A validation did not retain every authority pair")
        per_scene[binding.scene_id] = row
    v1 = pilot._evaluate_v1(model, inputs.validation, normalization, device)
    return {
        "v1_non_regression": v1,
        "response_listwise_v21a": _response_summary(per_scene, validation=True),
        "validation_no_grad": not torch.is_grad_enabled(),
        "benchmark_opened": False,
    }


def promotion_checks(
    history: Sequence[Mapping[str, Any]], epoch: int
) -> dict[str, bool]:
    if not history or int(history[0].get("epoch", -1)) != 0:
        raise ValueError("V2.1A promotion requires epoch zero")
    if epoch <= 0 or epoch >= len(history):
        raise ValueError("V2.1A promotion epoch is outside history")
    zero = history[0]["validation"]
    candidate = history[epoch]["validation"]
    zero_response = zero["response_listwise_v21a"]
    response = candidate["response_listwise_v21a"]
    zero_scene = zero_response["per_scene"]
    candidate_scene = response["per_scene"]
    absolute_every_scene = all(
        float(candidate_scene[scene]["response_absolute_relevance_loss"])
        <= float(zero_scene[scene]["response_absolute_relevance_loss"])
        for scene in VALIDATION_SCENES
    )
    return {
        "selected_epoch_positive": epoch > 0,
        "auxiliary_macro_strictly_improved": float(
            response["scene_macro_auxiliary_loss"]
        )
        < float(zero_response["scene_macro_auxiliary_loss"]),
        "absolute_relevance_macro_strictly_improved": float(
            response["scene_macro_absolute_relevance_loss"]
        )
        < float(zero_response["scene_macro_absolute_relevance_loss"]),
        "absolute_relevance_every_scene_non_regression": absolute_every_scene,
        "continuous_pairwise_macro_strictly_improved": float(
            response["scene_macro_continuous_pairwise_relevance_loss"]
        )
        < float(zero_response["scene_macro_continuous_pairwise_relevance_loss"]),
        "v1_fidelity_non_regression": candidate["v1_non_regression"][
            "non_regression_passed"
        ]
        is True,
        "minimum_active_over_eligible_coverage_at_least_95pct": float(
            response["minimum_active_over_eligible_coverage"]
        )
        >= COVERAGE_THRESHOLD,
        "minimum_pair_trainable_endpoint_coverage_at_least_95pct": float(
            response["minimum_pair_trainable_endpoint_coverage"]
        )
        >= COVERAGE_THRESHOLD,
    }


def _selection_rank(row: Mapping[str, Any]) -> tuple[float, ...]:
    response = row["validation"]["response_listwise_v21a"]
    candidate = row["validation"]["v1_non_regression"]["candidate"]
    return (
        -float(response["scene_macro_auxiliary_loss"]),
        -float(response["scene_macro_absolute_relevance_loss"]),
        -float(response["scene_macro_continuous_pairwise_relevance_loss"]),
        float(candidate["mean_all_view_cosine"]),
        float(candidate["p05_row_mean_all_view_cosine"]),
        float(candidate["relation_fidelity"]),
        -int(row["epoch"]),
    )


def _validate_history_axis(history: Sequence[Mapping[str, Any]]) -> None:
    if not history or [int(row.get("epoch", -1)) for row in history] != list(
        range(len(history))
    ):
        raise ValueError("V2.1A history must be contiguous from epoch zero")


def select_promotion_epoch(history: Sequence[Mapping[str, Any]]) -> int | None:
    _validate_history_axis(history)
    candidates = [
        row
        for row in history[1:]
        if all(promotion_checks(history, int(row["epoch"])).values())
    ]
    return None if not candidates else int(max(candidates, key=_selection_rank)["epoch"])


def select_best_raw_aux_epoch(history: Sequence[Mapping[str, Any]]) -> int:
    _validate_history_axis(history)
    if len(history) < 2:
        raise ValueError("V2.1A raw-aux selection requires a trained epoch")
    return int(max(history[1:], key=_selection_rank)["epoch"])


def _state_copy(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().contiguous().clone()
        for name, value in model.state_dict().items()
    }


def _state_sha(state: Mapping[str, torch.Tensor]) -> str:
    return surface_region_state_dict_sha256(state)


def _output_paths(output: Path) -> dict[str, Path]:
    return {
        "checkpoint": output,
        "normalization": output.with_suffix(output.suffix + ".normalization.pt"),
        "certificate": output.with_suffix(output.suffix + ".certificate.json"),
        "result": output.with_suffix(output.suffix + ".json"),
        "best_raw_aux": output.with_suffix(output.suffix + ".best_raw_aux.pt"),
        "final": output.with_suffix(output.suffix + ".final.pt"),
    }


def _require_outputs_absent(paths: Mapping[str, Path]) -> None:
    resolved = [path.resolve() for path in paths.values()]
    if len(set(resolved)) != len(resolved):
        raise ValueError("V2.1A output paths must be unique")
    existing = [
        path for path in paths.values() if path.exists() or path.is_symlink()
    ]
    if existing:
        raise FileExistsError(
            "V2.1A first-writer outputs already exist: "
            + ", ".join(str(path) for path in existing)
        )


def _implementation_records() -> dict[str, dict[str, str]]:
    root = Path(__file__).resolve().parents[2]
    prereg = (root / PREREGISTRATION).resolve()
    addendum = (root / EXECUTION_AUTHORITY_ADDENDUM).resolve()
    from radio_gs.losses import source_global_response_listwise_loss_v21a as loss

    return {
        "implementation": file_record(Path(__file__).resolve()),
        "objective_adapter": file_record(Path(v21a_adapter.__file__).resolve()),
        "loss_implementation": file_record(Path(loss.__file__).resolve()),
        "preregistration": file_record(prereg),
        "execution_authority_addendum": file_record(addendum),
    }


def _diagnostic_payload(
    *,
    role: str,
    epoch: int,
    state: Mapping[str, torch.Tensor],
    normalization_record: Mapping[str, str],
    execution_record: Mapping[str, str],
    implementations: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    return {
        "schema": DIAGNOSTIC_STATE_SCHEMA,
        "schema_version": 1,
        "role": role,
        "epoch": epoch,
        "model_class": SurfaceRegionAcceptedV2TypedContextResidualV1.__name__,
        "model_state_dict": dict(state),
        "model_state_dict_sha256": _state_sha(state),
        "normalization_authority": dict(normalization_record),
        "execution_authority": dict(execution_record),
        **{name: dict(record) for name, record in implementations.items()},
        "source_access": source_access(),
        "benchmark_opened": False,
    }


def _write_outputs(
    output: Path,
    inputs: PilotInputs,
    normalization: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    states: Mapping[str, tuple[int, Mapping[str, torch.Tensor]]],
) -> dict[str, Any]:
    paths = _output_paths(output)
    normalization_path = write_torch_noclobber(paths["normalization"], normalization)
    normalization_record = file_record(normalization_path)
    execution_record = {
        "path": inputs.execution["verified_path"],
        "sha256": inputs.execution["verified_sha256"],
    }
    implementations = _implementation_records()
    diagnostic_records: dict[str, dict[str, Any]] = {}
    for role in ("best_raw_aux", "final"):
        epoch, state = states[role]
        path = write_torch_noclobber(
            paths[role],
            _diagnostic_payload(
                role=role,
                epoch=epoch,
                state=state,
                normalization_record=normalization_record,
                execution_record=execution_record,
                implementations=implementations,
            ),
        )
        diagnostic_records[role] = {
            "epoch": epoch,
            "model_state_dict_sha256": _state_sha(state),
            "file": file_record(path),
        }

    selected_epoch = select_promotion_epoch(history)
    selected_validation = (
        None if selected_epoch is None else history[selected_epoch]["validation"]
    )
    selected_state = None if selected_epoch is None else states["selected"][1]
    selected_state_sha = None if selected_state is None else _state_sha(selected_state)
    certificate = {
        "schema": CERTIFICATE_SCHEMA,
        "schema_version": 1,
        "training_contract": training_contract(),
        "training_contract_sha256": TRAINING_CONTRACT_SHA256,
        **implementations,
        "execution_authority": execution_record,
        "cohort_authority": dict(inputs.execution["cohort_authority"]),
        "pilot_cohort_region_view_registry": dict(
            inputs.execution["pilot_cohort_region_view_registry"]
        ),
        "pilot_cohort_region_view_registry_authority_sha256": inputs.registry[
            "authority_sha256"
        ],
        "benchmark_exclusion_manifest": dict(
            inputs.execution["benchmark_exclusion_manifest"]
        ),
        "input_records_by_split": {
            "source_train": pilot._input_records(inputs.train),
            "source_validation": pilot._input_records(inputs.validation),
        },
        "optimizer_steps_completed": len(history) - 1,
        "selected_epoch": selected_epoch,
        "selected_validation": selected_validation,
        "model_state_dict_sha256": selected_state_sha,
        "normalization_authority": normalization_record,
        "normalization_content_authority_sha256": (
            pilot.pilot_normalization_authority_sha256(normalization)
        ),
        "diagnostic_states": diagnostic_records,
        "source_access": source_access(),
        "benchmark_opened": False,
    }
    certificate["content_sha256"] = canonical_json_sha256(certificate)
    certificate_path = write_frozen_json(paths["certificate"], certificate)

    checkpoint_record: dict[str, str] | None = None
    if selected_epoch is not None and selected_state is not None:
        checkpoint = {
            "schema": CHECKPOINT_SCHEMA,
            "schema_version": 1,
            "model_class": SurfaceRegionAcceptedV2TypedContextResidualV1.__name__,
            "model_architecture": SurfaceRegionAcceptedV2TypedContextResidualV1(
                scalar_median=normalization["median"],
                scalar_robust_scale=normalization["robust_scale"],
                max_angle_radians=v1_trainer.MAX_ANGLE_RADIANS,
                max_alpha=v1_trainer.MAX_ALPHA,
            ).architecture(),
            "accepted_v2_authority": accepted_v2_authority(),
            "model_state_dict": dict(selected_state),
            "model_state_dict_sha256": selected_state_sha,
            "normalization_authority": normalization_record,
            "certificate": file_record(certificate_path),
            "selected_epoch": selected_epoch,
            **implementations,
            "source_access": source_access(),
        }
        checkpoint_path = write_torch_noclobber(paths["checkpoint"], checkpoint)
        checkpoint_record = file_record(checkpoint_path)

    report = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "status": (
            "source_only_rescue_promotion_candidate_complete"
            if selected_epoch is not None
            else "source_only_rescue_diagnostic_complete_no_promotion_checkpoint"
        ),
        "training_contract": training_contract(),
        "training_contract_sha256": TRAINING_CONTRACT_SHA256,
        **implementations,
        "execution_authority": execution_record,
        "checkpoint": checkpoint_record,
        "normalization_authority": normalization_record,
        "certificate": file_record(certificate_path),
        "diagnostic_states": diagnostic_records,
        "optimizer_steps_completed": len(history) - 1,
        "selected_epoch": selected_epoch,
        "selection_status": (
            "promotion_eligible_minimum_auxiliary"
            if selected_epoch is not None
            else "no_epoch_satisfied_all_promotion_constraints"
        ),
        "selected_validation": selected_validation,
        "best_raw_aux_epoch": select_best_raw_aux_epoch(history),
        "final_epoch": len(history) - 1,
        "history": list(history),
        "source_access": source_access(),
        "benchmark_opened": False,
    }
    write_frozen_json(paths["result"], report)
    return report


def train(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    paths = _output_paths(output)
    _require_outputs_absent(paths)
    inputs = prepare_inputs(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
    )
    normalization = pilot.build_pilot_normalization(
        [item.base for item in inputs.train],
        source_state_cohort_authority_sha256=inputs.source_state_manifest[
            "authority_sha256"
        ],
    )
    device = torch.device(str(args.device))
    torch.manual_seed(v1_trainer.SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(v1_trainer.SEED)
    model = SurfaceRegionAcceptedV2TypedContextResidualV1(
        scalar_median=normalization["median"],
        scalar_robust_scale=normalization["robust_scale"],
        max_angle_radians=v1_trainer.MAX_ANGLE_RADIANS,
        max_alpha=v1_trainer.MAX_ALPHA,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=v1_trainer.LEARNING_RATE,
        weight_decay=v1_trainer.WEIGHT_DECAY,
    )
    epoch_zero = evaluate(model, inputs, normalization, device)
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "training": None,
            "validation": epoch_zero,
            "model_state_dict_sha256": v1_trainer._state_sha(model),
        }
    ]
    best_raw_epoch = -1
    best_raw_state: dict[str, torch.Tensor] | None = None
    best_selected_epoch: int | None = None
    best_selected_state: dict[str, torch.Tensor] | None = None
    for step in range(1, OPTIMIZER_STEPS + 1):
        training = train_one_step(model, optimizer, inputs, normalization, device)
        validation = evaluate(model, inputs, normalization, device)
        row = {
            "epoch": step,
            "training": training,
            "validation": validation,
            "model_state_dict_sha256": v1_trainer._state_sha(model),
        }
        history.append(row)
        raw_epoch = select_best_raw_aux_epoch(history)
        if raw_epoch == step:
            best_raw_epoch = step
            best_raw_state = _state_copy(model)
        selected_epoch = select_promotion_epoch(history)
        if selected_epoch == step:
            best_selected_epoch = step
            best_selected_state = _state_copy(model)
        print(json.dumps(row, sort_keys=True), flush=True)
    if len(history) != OPTIMIZER_STEPS + 1:
        raise RuntimeError("V2.1A did not complete exactly 30 optimizer steps")
    selected_epoch = select_promotion_epoch(history)
    raw_epoch = select_best_raw_aux_epoch(history)
    if raw_epoch != best_raw_epoch or best_raw_state is None:
        raise RuntimeError("V2.1A best-raw-aux state tracking differs")
    if selected_epoch != best_selected_epoch or (
        selected_epoch is not None and best_selected_state is None
    ):
        raise RuntimeError("V2.1A promotion-selected state tracking differs")
    final_state = _state_copy(model)
    states: dict[str, tuple[int, Mapping[str, torch.Tensor]]] = {
        "best_raw_aux": (raw_epoch, best_raw_state),
        "final": (OPTIMIZER_STEPS, final_state),
    }
    if selected_epoch is not None and best_selected_state is not None:
        states["selected"] = (selected_epoch, best_selected_state)
    model.cpu()
    return _write_outputs(output, inputs, normalization, history, states)


def synthetic_dry_run() -> dict[str, Any]:
    return {
        "schema": "radio_gs.surface_region_v21a_rescue_synthetic_dry_run.v1",
        "optimizer_steps": OPTIMIZER_STEPS,
        "early_stopping": False,
        "pairwise_denominator": "any_trainable_endpoint",
        "triplet_denominator": "trainable_anchor_only",
        "coverage_threshold": COVERAGE_THRESHOLD,
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
        inputs = prepare_inputs(
            args.execution_authority,
            expected_sha256=args.expected_execution_authority_sha256,
        )
        result = {
            "status": "source_only_v21a_rescue_authority_validated",
            "source_train": [item.scene_id for item in inputs.train],
            "source_validation": [item.scene_id for item in inputs.validation],
            "benchmark_opened": False,
        }
    else:
        result = train(args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "COVERAGE_THRESHOLD",
    "OPTIMIZER_STEPS",
    "promotion_checks",
    "select_best_raw_aux_epoch",
    "select_promotion_epoch",
    "train_one_step",
    "training_contract",
]
