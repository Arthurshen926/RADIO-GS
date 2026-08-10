#!/usr/bin/env python3
"""Train the preregistered source-only V2.1B rank-256 exact-4+2 pilot."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from radio_gs.interfaces import (
    surface_region_v21b_reliability_conditioned_residual as v21b_interface,
)
from radio_gs.interfaces.surface_region_summary import (
    surface_region_state_dict_sha256,
)
from radio_gs.interfaces.surface_region_typed_context_training import (
    accepted_v2_authority,
)
from radio_gs.losses import source_global_response_listwise_loss_v21 as v21_loss
from radio_gs.losses import source_global_response_listwise_loss_v21b as v21b_loss
from radio_gs.losses.source_typed_text_relation_loss_v21 import (
    FrozenTypedTextRelationAuthority,
    load_frozen_typed_text_relation_authority,
)
from radio_gs.models.surface_region_v21b_reliability_conditioned_residual import (
    SurfaceRegionAcceptedV2ReliabilityConditionedResidualV21B,
)
from radio_gs.scripts import materialize_full_scalar_clean_training_shard as pilot_shard
from radio_gs.scripts import train_surface_region_full_scalar_residual as base_trainer
from radio_gs.scripts import train_surface_region_typed_context_residual as v1_trainer
from radio_gs.scripts import (
    train_surface_region_typed_context_response_listwise_v2 as v2_trainer,
)
from radio_gs.scripts import (
    train_surface_region_typed_context_response_listwise_v21_pilot as asset_pilot,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


ARTIFACT_TYPE = "surface_region_v21b_conditioned_rank256_exact4x2_source_pilot"
EXECUTION_AUTHORITY_SCHEMA = (
    "radio_gs.surface_region_v21b_conditioned_rank256_exact4x2_"
    "execution_authority.v1"
)
NORMALIZATION_SCHEMA = "radio_gs.surface_region_v21b_train4_normalization.v1"
STATE_ARCHIVE_SCHEMA = "radio_gs.surface_region_v21b_promotion_state_archive.v1"
CHECKPOINT_SCHEMA = "radio_gs.surface_region_v21b_conditioned_rank256_checkpoint.v1"
CERTIFICATE_SCHEMA = "radio_gs.surface_region_v21b_source_certificate.v1"
RESULT_SCHEMA = "radio_gs.surface_region_v21b_source_result.v1"
TRAIN_SCENES = asset_pilot.TRAIN_SCENES
VALIDATION_SCENES = asset_pilot.VALIDATION_SCENES
COMPONENT_WEIGHTS = dict(asset_pilot.COMPONENT_WEIGHTS)
PRIMARY_WEIGHT = asset_pilot.PRIMARY_WEIGHT
OPTIMIZER_STEPS = 30
MINIMUM_RELATIVE_IMPROVEMENT = 0.005
MINIMUM_ACTIVE_COVERAGE = 0.95
MINIMUM_PAIR_COVERAGE = 0.95
MODEL_PREREGISTRATION = Path(
    "paper/artifacts/"
    "surface_region_v21b_reliability_conditioned_rank256_"
    "preregistration_20260807.json"
)
MODEL_PREREGISTRATION_SHA256 = (
    "5397e29ece42b03c8af0b77865a9c8327920c59f0be050dbd99eb8ca91ece60a"
)
TRAINING_ADDENDUM = Path(
    "paper/artifacts/"
    "surface_region_v21b_exact4x2_source_training_and_promotion_"
    "addendum_20260807.json"
)
TRAINING_ADDENDUM_SHA256 = (
    "c6441bc63abee9726781db6a9aa083635f170b0b20524cce81b2d474728bba8b"
)


@dataclass(frozen=True)
class V21BInputs:
    execution: dict[str, Any]
    train: tuple[v2_trainer.ResponseSceneBinding, ...]
    validation: tuple[v2_trainer.ResponseSceneBinding, ...]
    fit: v2_trainer.FitTextBank
    canonical_negative: v21_loss.FrozenCanonicalNegativeBank
    compositional: tuple[v21_loss.FrozenCompositionalGenericBank, ...]
    relations: FrozenTypedTextRelationAuthority
    cohort: dict[str, Any]
    registry: dict[str, Any]
    source_state_manifest: dict[str, Any]


def source_access() -> dict[str, bool]:
    return {
        **v21b_interface.source_access(),
        "generic_target_blind_text_bank_opened": True,
        "canonical_generic_negative_bank_opened": True,
        "typed_relation_authority_opened": True,
        "source_validation_used_for_selection": True,
    }


def training_contract() -> dict[str, Any]:
    config = v21_loss.recommended_v21_config()
    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "model_interface_contract_sha256": (
            v21b_interface.SURFACE_REGION_V21B_INTERFACE_CONTRACT_SHA256
        ),
        "model_preregistration_sha256": MODEL_PREREGISTRATION_SHA256,
        "training_addendum_sha256": TRAINING_ADDENDUM_SHA256,
        "cohort": {
            "source_train": list(TRAIN_SCENES),
            "source_validation": list(VALIDATION_SCENES),
            "scene_and_physical_space_disjoint": True,
        },
        "input_authority": {
            "pilot_training_shard_schema": pilot_shard.PILOT_TRAINING_SHARD_SCHEMA,
            "pilot_training_shard_contract_sha256": (
                pilot_shard.PILOT_TRAINING_SHARD_CONTRACT_SHA256
            ),
            "pilot_cohort_region_view_registry_schema": (
                pilot_shard.PILOT_COHORT_REGISTRY_SCHEMA
            ),
            "pilot_cohort_region_view_registry_contract_sha256": (
                canonical_json_sha256(pilot_shard.pilot_cohort_registry_contract())
            ),
            "legacy_24plus8_assets_accepted": False,
        },
        "model": {
            "class": (
                "SurfaceRegionAcceptedV2ReliabilityConditionedResidualV21B"
            ),
            "hidden_rank": 256,
            "angular_budget_radians": "0.15_plus_0.60_times_reliability",
            "fallback": "inactive_or_ood_bitwise_accepted_v2",
        },
        "hard_negative_denominators": {
            "pairwise": "any_endpoint_trainable",
            "triplet": "anchor_trainable_only",
            "same_policy_in_training_and_validation": True,
        },
        "response": {
            "teacher_multiview_temperature": config.response_temperature,
            "inference_logit_scale": config.inference_logit_scale,
            "auxiliary_weight": config.auxiliary_weight,
            "component_weights": {
                "object_noun_primary": PRIMARY_WEIGHT,
                **COMPONENT_WEIGHTS,
            },
            "typed_relation_authority_required": True,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": v1_trainer.LEARNING_RATE,
            "weight_decay": v1_trainer.WEIGHT_DECAY,
            "maximum_gradient_norm": v1_trainer.MAX_GRADIENT_NORM,
            "optimizer_steps": OPTIMIZER_STEPS,
            "early_stopping": False,
            "equal_scene_gradient_accumulation": True,
        },
        "promotion": {
            "v1_non_regression": True,
            "relative_improvement_from_step_zero": {
                "minimum": MINIMUM_RELATIVE_IMPROVEMENT,
                "metrics": [
                    "scene_macro_auxiliary_loss",
                    "scene_macro_absolute_relevance_loss",
                    "scene_macro_continuous_pairwise_relevance_loss",
                ],
            },
            "absolute_relevance_every_scene_non_regression": True,
            "every_scene_active_row_coverage_minimum": MINIMUM_ACTIVE_COVERAGE,
            "every_scene_pair_endpoint_coverage_minimum": MINIMUM_PAIR_COVERAGE,
            "selection": ["minimum_auxiliary_loss", "earliest_step"],
            "no_eligible_state": "diagnostic_only_no_checkpoint_or_certificate",
        },
        "state_retention": "step_zero_plus_every_promotion_eligible_state",
        "source_access": source_access(),
    }


TRAINING_CONTRACT_SHA256 = canonical_json_sha256(training_contract())


def _record(value: object, *, label: str) -> dict[str, str]:
    return asset_pilot._record(value, label=label)


def _scene_records(value: object, *, split: str) -> list[dict[str, Any]]:
    expected = TRAIN_SCENES if split == "source_train" else VALIDATION_SCENES
    if not isinstance(value, list) or len(value) != len(expected):
        raise ValueError(f"V2.1B {split} requires the exact fixed cohort")
    records = [dict(item) for item in value if isinstance(item, Mapping)]
    required = {
        "scene_id",
        "training_shard",
        "adaptive_context",
        "hard_negative_authority",
        "hard_negative_content_authority_sha256",
    }
    if (
        len(records) != len(expected)
        or tuple(str(item.get("scene_id")) for item in records) != expected
        or any(set(item) != required for item in records)
    ):
        raise ValueError(f"V2.1B {split} scene records differ")
    return records


_CODE_RECORD_FIELDS = (
    "trainer_implementation",
    "execution_builder_implementation",
    "source_gate_implementation",
    "asset_loader_implementation",
    "model_implementation",
    "model_interface_implementation",
    "loss_implementation",
    "frozen_v21_loss_implementation",
    "model_preregistration",
    "training_addendum",
)


def validate_execution_authority(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("V2.1B execution authority must be a mapping")
    authority = dict(value)
    required = {
        "schema",
        "schema_version",
        "status",
        *_CODE_RECORD_FIELDS,
        "cohort_authority",
        "pilot_cohort_region_view_registry",
        "benchmark_exclusion_manifest",
        "fit_text_bank",
        "canonical_negative_bank",
        "compositional_banks",
        "typed_relation_authority",
        "source_train",
        "source_validation",
        "training_authorized",
        "benchmark_execution_authorized",
        "source_access",
    }
    if set(authority) != required:
        raise ValueError("V2.1B execution authority fields differ")
    if (
        authority.get("schema") != EXECUTION_AUTHORITY_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("status")
        != "authorized_source_only_v21b_exact4train_2validation"
        or authority.get("training_authorized") is not True
        or authority.get("benchmark_execution_authorized") is not False
        or authority.get("source_access") != source_access()
    ):
        raise ValueError("V2.1B execution authority header differs")
    for name in (
        *_CODE_RECORD_FIELDS,
        "cohort_authority",
        "pilot_cohort_region_view_registry",
        "benchmark_exclusion_manifest",
        "fit_text_bank",
        "canonical_negative_bank",
    ):
        authority[name] = _record(authority[name], label=name)
    banks = authority["compositional_banks"]
    if not isinstance(banks, Mapping) or set(banks) != set(COMPONENT_WEIGHTS):
        raise ValueError("V2.1B compositional component set differs")
    normalized_banks: dict[str, dict[str, Any]] = {}
    for name, weight in COMPONENT_WEIGHTS.items():
        raw = banks[name]
        if not isinstance(raw, Mapping) or set(raw) != {
            "path",
            "sha256",
            "loss_weight",
        } or float(raw["loss_weight"]) != weight:
            raise ValueError(f"V2.1B {name} bank record differs")
        normalized_banks[name] = {
            **_record(
                {"path": raw["path"], "sha256": raw["sha256"]},
                label=name,
            ),
            "loss_weight": weight,
        }
    authority["compositional_banks"] = normalized_banks
    relation = authority["typed_relation_authority"]
    if not isinstance(relation, Mapping) or set(relation) != {
        "path",
        "sha256",
        "content_authority_sha256",
    }:
        raise ValueError("V2.1B typed relation record differs")
    authority["typed_relation_authority"] = {
        **_record(
            {"path": relation["path"], "sha256": relation["sha256"]},
            label="typed relation authority",
        ),
        "content_authority_sha256": v2_trainer._sha(
            relation["content_authority_sha256"],
            label="typed relation content authority",
        ),
    }
    authority["source_train"] = _scene_records(
        authority["source_train"], split="source_train"
    )
    authority["source_validation"] = _scene_records(
        authority["source_validation"], split="source_validation"
    )
    return authority


def _validated_path(record: Mapping[str, str], *, label: str) -> Path:
    return validate_file_record(record, label=label)


def _frozen_json_file_sha256(value: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        dict(value), indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    return hashlib.sha256(serialized).hexdigest()


def _expected_code_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[2]
    return {
        "trainer_implementation": Path(__file__).resolve(),
        "execution_builder_implementation": (
            root / "radio_gs/scripts/build_surface_region_v21b_execution_authority.py"
        ),
        "source_gate_implementation": (
            root / "radio_gs/interfaces/surface_region_v21b_source_gate.py"
        ),
        "asset_loader_implementation": Path(asset_pilot.__file__).resolve(),
        "model_implementation": Path(
            SurfaceRegionAcceptedV2ReliabilityConditionedResidualV21B.__module__
            .replace(".", "/")
            + ".py"
        ),
        "model_interface_implementation": Path(v21b_interface.__file__).resolve(),
        "loss_implementation": Path(v21b_loss.__file__).resolve(),
        "frozen_v21_loss_implementation": Path(v21_loss.__file__).resolve(),
        "model_preregistration": root / MODEL_PREREGISTRATION,
        "training_addendum": root / TRAINING_ADDENDUM,
    }


def _resolved_expected_code_paths() -> dict[str, Path]:
    values = _expected_code_paths()
    # ``__module__`` above intentionally avoids importing a second alias; make
    # its repository-relative path absolute here.
    if not values["model_implementation"].is_absolute():
        values["model_implementation"] = (
            Path(__file__).resolve().parents[2] / values["model_implementation"]
        )
    return {name: path.resolve() for name, path in values.items()}


def prepare_inputs(path: str | Path, *, expected_sha256: str) -> V21BInputs:
    raw, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="V2.1B source execution authority",
    )
    authority = validate_execution_authority(raw)
    for name, expected in _resolved_expected_code_paths().items():
        observed = _validated_path(authority[name], label=f"V2.1B {name}")
        if observed != expected:
            raise ValueError(f"V2.1B authority binds another {name}")
    if authority["model_preregistration"]["sha256"] != MODEL_PREREGISTRATION_SHA256:
        raise ValueError("V2.1B model preregistration digest differs")
    if authority["training_addendum"]["sha256"] != TRAINING_ADDENDUM_SHA256:
        raise ValueError("V2.1B training addendum digest differs")

    cohort, cohort_file, _ = asset_pilot._validate_subset_authorities(authority)
    registry_record = authority["pilot_cohort_region_view_registry"]
    registry_raw, registry_digest, registry_path = load_json_object(
        _validated_path(registry_record, label="V2.1B pilot registry"),
        expected_sha256=registry_record["sha256"],
        label="V2.1B pilot registry",
    )
    registry = pilot_shard.validate_pilot_cohort_region_view_registry(
        registry_raw,
        cohort_authority=cohort,
        cohort_authority_file_sha256=cohort_file["sha256"],
    )
    if registry_record != {"path": str(registry_path), "sha256": registry_digest}:
        raise ValueError("V2.1B pilot registry file record differs")
    source_state_manifest, teacher_manifest = (
        pilot_shard.derive_pilot_global_manifests(registry)
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
        "source_state_manifest_file_sha256": _frozen_json_file_sha256(
            source_state_manifest
        ),
        "cohort_authority_sha256": cohort["authority_sha256"],
        "cohort_authority_file_sha256": cohort_file["sha256"],
        "teacher_authority_sha256": teacher_manifest["authority_sha256"],
        "teacher_manifest_file_sha256": _frozen_json_file_sha256(teacher_manifest),
    }
    fit = v2_trainer.load_fit_text_bank(
        _validated_path(authority["fit_text_bank"], label="V2.1B primary fit bank"),
        expected_sha256=authority["fit_text_bank"]["sha256"],
    )
    canonical = v21_loss.load_frozen_canonical_negative_bank(
        _validated_path(
            authority["canonical_negative_bank"], label="V2.1B negative bank"
        ),
        expected_file_sha256=authority["canonical_negative_bank"]["sha256"],
    )
    compositional = tuple(
        v21_loss.load_frozen_compositional_generic_bank(
            _validated_path(
                {
                    "path": authority["compositional_banks"][name]["path"],
                    "sha256": authority["compositional_banks"][name]["sha256"],
                },
                label=f"V2.1B {name}",
            ),
            expected_file_sha256=authority["compositional_banks"][name]["sha256"],
            component_id=name,
            loss_weight=COMPONENT_WEIGHTS[name],
        )
        for name in COMPONENT_WEIGHTS
    )
    relation_record = authority["typed_relation_authority"]
    relations = load_frozen_typed_text_relation_authority(
        _validated_path(
            {"path": relation_record["path"], "sha256": relation_record["sha256"]},
            label="V2.1B typed relation authority",
        ),
        expected_file_sha256=relation_record["sha256"],
    )
    if relations.content_authority_sha256 != relation_record[
        "content_authority_sha256"
    ]:
        raise ValueError("V2.1B typed relation content authority differs")

    def bind(split: str) -> tuple[v2_trainer.ResponseSceneBinding, ...]:
        records = authority[split]
        bases, summaries = asset_pilot._base_bindings(
            records,
            split=split,
            expected_lineage=expected_lineage,
            registry_scene_records=registry_scene_records,
        )
        result: list[v2_trainer.ResponseSceneBinding] = []
        for base, item, summary in zip(bases, records, summaries):
            response = v2_trainer.bind_response_scene(base, item, fit_text_bank=fit)
            asset_pilot._validate_response_shard_channel_summary(
                response,
                summary,
                registry_scene_records[response.scene_id],
            )
            result.append(response)
        return tuple(result)

    authority["verified_path"] = str(source)
    authority["verified_sha256"] = digest
    return V21BInputs(
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


def build_pilot_normalization(
    bindings: Sequence[v1_trainer.SceneBinding],
    *,
    source_state_cohort_authority_sha256: str,
) -> dict[str, Any]:
    normalization = asset_pilot.build_pilot_normalization(
        bindings,
        source_state_cohort_authority_sha256=source_state_cohort_authority_sha256,
    )
    normalization["schema"] = NORMALIZATION_SCHEMA
    normalization["source_access"] = source_access()
    return normalization


def normalization_content_authority_sha256(
    normalization: Mapping[str, Any],
) -> str:
    return asset_pilot.pilot_normalization_authority_sha256(normalization)


def _objective(
    model: SurfaceRegionAcceptedV2ReliabilityConditionedResidualV21B,
    binding: v2_trainer.ResponseSceneBinding,
    normalization: Mapping[str, Any],
    fit: torch.Tensor,
    canonical: v21_loss.FrozenCanonicalNegativeBank,
    compositional: Sequence[v21_loss.FrozenCompositionalGenericBank],
    relations: FrozenTypedTextRelationAuthority,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int | bool]]:
    scene = asset_pilot.load_pilot_bound_scene(binding)
    declared, effective_ood, active = asset_pilot._pilot_routing(
        scene, normalization
    )
    if int(active.sum()) < 2:
        raise ValueError("V2.1B complete scene requires at least two active rows")
    result = v21b_interface.forward_complete_scene(
        model,
        scene,
        declared_active_mask=declared,
        effective_ood_mask=effective_ood,
        device=device,
    )
    base = scene["accepted_v2_e0"].to(device)
    fallback = ~active.to(device)
    fallback_equal = torch.equal(
        result.semantic_descriptor[fallback], base[fallback]
    )
    if not fallback_equal:
        raise RuntimeError("V2.1B fallback is not bitwise AcceptedV2")
    active_rows = torch.where(active)[0]
    teachers, teacher_mask = v1_trainer.gather_sparse_teacher_batch(
        scene, active_rows
    )
    base_loss, base_metrics = v1_trainer.typed_context_objective(
        result.semantic_descriptor[active.to(device)],
        teachers.to(device),
        teacher_mask.to(device),
        scene["typed_context_statistics"][active_rows].to(device),
        boundary_threshold=float(normalization["source_boundary_score_median"]),
    )
    total, response_metrics = v21b_loss.source_global_response_listwise_loss_v21b(
        base_loss,
        result.semantic_descriptor,
        scene["official_multiview_siglip2_teacher_pair_descriptors"].to(device),
        scene["official_multiview_siglip2_teacher_pair_region_indices"],
        fit,
        binding.response_authority.payload["canonical_region_indices"],
        binding.response_authority,
        canonical,
        accepted_v2_file_sha256=binding.response_authority.accepted_v2_file_sha256,
        teacher_file_sha256=binding.response_authority.teacher_file_sha256,
        teacher_pair_descriptors_sha256=(
            binding.response_authority.teacher_pair_descriptors_sha256
        ),
        fit_text_bank_file_sha256=binding.response_authority.fit_text_bank_file_sha256,
        compositional_banks=compositional,
        relation_authority=relations,
        trainable_region_mask=active,
        config=v21_loss.recommended_v21_config(),
    )
    row_count = len(scene["region_row_ids"])
    return total, {
        "base_objective": base_loss,
        **{f"base_{name}": value for name, value in base_metrics.items()},
        **{f"response_{name}": value for name, value in response_metrics.items()},
        "complete_canonical_rows": row_count,
        "active_rows": int(active.sum()),
        "immutable_rows": int((~active).sum()),
        "active_row_coverage": result.semantic_descriptor.new_tensor(
            int(active.sum()) / row_count
        ),
        "fallback_bitwise_accepted_v2_e0": fallback_equal,
    }


def train_one_step(
    model: SurfaceRegionAcceptedV2ReliabilityConditionedResidualV21B,
    optimizer: torch.optim.Optimizer,
    inputs: V21BInputs,
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
            inputs.canonical_negative,
            inputs.compositional,
            inputs.relations,
            device,
        )
        (total / len(inputs.train)).backward()
        per_scene[binding.scene_id] = {
            "combined_objective": float(total.detach().cpu()),
            **v2_trainer._float_metrics(metrics),
        }
    torch.nn.utils.clip_grad_norm_(
        tuple(model.parameters()),
        v1_trainer.MAX_GRADIENT_NORM,
        error_if_nonfinite=True,
    )
    optimizer.step()
    return {
        "scene_count": len(inputs.train),
        "equal_scene_weight": 1.0 / len(inputs.train),
        "complete_scene_forward": True,
        "pairwise_any_endpoint_filter": True,
        "triplet_anchor_only_filter": True,
        "per_scene": per_scene,
    }


@torch.no_grad()
def evaluate(
    model: SurfaceRegionAcceptedV2ReliabilityConditionedResidualV21B,
    inputs: V21BInputs,
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
            inputs.canonical_negative,
            inputs.compositional,
            inputs.relations,
            device,
        )
        per_scene[binding.scene_id] = {
            "combined_objective": float(total.detach().cpu()),
            **v2_trainer._float_metrics(metrics),
        }
    v1 = asset_pilot._evaluate_v1(model, inputs.validation, normalization, device)

    def macro(name: str) -> float:
        return sum(float(row[name]) for row in per_scene.values()) / len(per_scene)

    response = {
        "scene_count": len(per_scene),
        "scene_macro_auxiliary_loss": macro("response_auxiliary_loss"),
        "scene_macro_absolute_relevance_loss": macro(
            "response_absolute_relevance_loss"
        ),
        "scene_macro_continuous_pairwise_relevance_loss": macro(
            "response_continuous_pairwise_relevance_loss"
        ),
        "scene_macro_active_row_coverage": macro("active_row_coverage"),
        "scene_macro_pair_trainable_endpoint_coverage": macro(
            "response_pair_trainable_endpoint_coverage"
        ),
        "per_scene": per_scene,
    }
    return {
        "v1_non_regression": v1,
        "response_listwise_v21b": response,
        "validation_no_grad": not torch.is_grad_enabled(),
        "benchmark_opened": False,
    }


def _relative_improvement(epoch_zero: float, candidate: float) -> float:
    baseline = float(epoch_zero)
    current = float(candidate)
    if not math.isfinite(baseline) or not math.isfinite(current) or baseline < 0:
        raise ValueError("V2.1B promotion losses must be finite and nonnegative")
    return (baseline - current) / max(abs(baseline), 1e-12)


def promotion_checks(
    validation: Mapping[str, Any],
    epoch_zero: Mapping[str, Any],
) -> dict[str, Any]:
    response = validation["response_listwise_v21b"]
    baseline = epoch_zero["response_listwise_v21b"]
    names = (
        "scene_macro_auxiliary_loss",
        "scene_macro_absolute_relevance_loss",
        "scene_macro_continuous_pairwise_relevance_loss",
    )
    relative = {
        name: _relative_improvement(baseline[name], response[name]) for name in names
    }
    current_scene = response["per_scene"]
    baseline_scene = baseline["per_scene"]
    checks = {
        "v1_fidelity_non_regression": validation["v1_non_regression"][
            "non_regression_passed"
        ]
        is True,
        "auxiliary_relative_improvement_at_least_0p5_percent": (
            relative["scene_macro_auxiliary_loss"] >= MINIMUM_RELATIVE_IMPROVEMENT
        ),
        "absolute_relevance_relative_improvement_at_least_0p5_percent": (
            relative["scene_macro_absolute_relevance_loss"]
            >= MINIMUM_RELATIVE_IMPROVEMENT
        ),
        "continuous_pairwise_relative_improvement_at_least_0p5_percent": (
            relative["scene_macro_continuous_pairwise_relevance_loss"]
            >= MINIMUM_RELATIVE_IMPROVEMENT
        ),
        "absolute_relevance_every_scene_non_regression": all(
            float(current_scene[scene]["response_absolute_relevance_loss"])
            <= float(baseline_scene[scene]["response_absolute_relevance_loss"])
            for scene in VALIDATION_SCENES
        ),
        "active_row_coverage_every_scene_at_least_95_percent": all(
            float(current_scene[scene]["active_row_coverage"])
            >= MINIMUM_ACTIVE_COVERAGE
            for scene in VALIDATION_SCENES
        ),
        "pair_endpoint_coverage_every_scene_at_least_95_percent": all(
            float(
                current_scene[scene][
                    "response_pair_trainable_endpoint_coverage"
                ]
            )
            >= MINIMUM_PAIR_COVERAGE
            for scene in VALIDATION_SCENES
        ),
    }
    return {
        "relative_improvement_from_step_zero": relative,
        "checks": checks,
        "passed": all(checks.values()),
    }


def attach_promotion(
    validation: Mapping[str, Any],
    epoch_zero: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(validation)
    result["promotion"] = promotion_checks(result, epoch_zero)
    result["selection_eligible"] = bool(result["promotion"]["passed"])
    return result


def select_promotion_step(history: Sequence[Mapping[str, Any]]) -> int | None:
    if not history or [int(row.get("step", -1)) for row in history] != list(
        range(len(history))
    ):
        raise ValueError("V2.1B history must be contiguous from step zero")
    eligible = [
        row
        for row in history
        if row.get("validation", {}).get("selection_eligible") is True
    ]
    if not eligible:
        return None
    selected = min(
        eligible,
        key=lambda row: (
            float(
                row["validation"]["response_listwise_v21b"][
                    "scene_macro_auxiliary_loss"
                ]
            ),
            int(row["step"]),
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


def _input_records(
    bindings: Sequence[v2_trainer.ResponseSceneBinding],
) -> list[dict[str, Any]]:
    return v2_trainer.input_records(bindings)


def _output_paths(output: Path) -> dict[str, Path]:
    return {
        "checkpoint": output,
        "normalization": output.with_suffix(output.suffix + ".normalization.pt"),
        "state_archive": output.with_suffix(output.suffix + ".epoch_states.pt"),
        "certificate": output.with_suffix(output.suffix + ".certificate.json"),
        "result": output.with_suffix(output.suffix + ".json"),
    }


def _preflight_outputs(paths: Mapping[str, Path]) -> None:
    resolved = [path.resolve(strict=False) for path in paths.values()]
    if len(set(resolved)) != len(resolved):
        raise ValueError("V2.1B output paths must be unique")
    existing = [path for path in paths.values() if path.exists() or path.is_symlink()]
    if existing:
        raise FileExistsError(
            "V2.1B first-writer outputs already exist: "
            + ", ".join(str(path) for path in existing)
        )


def _write_outputs(
    output: Path,
    normalization: Mapping[str, Any],
    inputs: V21BInputs,
    history: Sequence[Mapping[str, Any]],
    saved_states: Mapping[int, Mapping[str, torch.Tensor]],
    selected_step: int | None,
) -> dict[str, Any]:
    paths = _output_paths(output)
    normalization_path = write_torch_noclobber(paths["normalization"], normalization)
    normalization_record = file_record(normalization_path)
    normalization_content = normalization_content_authority_sha256(normalization)
    eligible_steps = [
        int(row["step"])
        for row in history
        if row["validation"]["selection_eligible"] is True
    ]
    expected_saved = [0, *eligible_steps]
    if list(saved_states) != expected_saved:
        raise RuntimeError("V2.1B saved-state coverage differs")
    state_hashes = {str(step): _state_sha(saved_states[step]) for step in saved_states}
    for step in saved_states:
        if state_hashes[str(step)] != history[step]["model_state_dict_sha256"]:
            raise RuntimeError("V2.1B saved state differs from history")
    archive = {
        "schema": STATE_ARCHIVE_SCHEMA,
        "schema_version": 1,
        "training_contract_sha256": TRAINING_CONTRACT_SHA256,
        "execution_authority": {
            "path": inputs.execution["verified_path"],
            "sha256": inputs.execution["verified_sha256"],
        },
        "normalization_authority": normalization_record,
        "normalization_content_authority_sha256": normalization_content,
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
        selected_state_sha = state_hashes[str(selected_step)]
        selected_validation = history[selected_step]["validation"]
        certificate = {
            "schema": CERTIFICATE_SCHEMA,
            "schema_version": 1,
            "training_contract": training_contract(),
            "training_contract_sha256": TRAINING_CONTRACT_SHA256,
            "execution_authority": {
                "path": inputs.execution["verified_path"],
                "sha256": inputs.execution["verified_sha256"],
            },
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
                "source_train": _input_records(inputs.train),
                "source_validation": _input_records(inputs.validation),
            },
            "selected_step": selected_step,
            "selected_validation": dict(selected_validation),
            "model_state_dict_sha256": selected_state_sha,
            "normalization_authority": normalization_record,
            "normalization_content_authority_sha256": normalization_content,
            "state_archive": archive_record,
            "source_access": source_access(),
            "benchmark_opened": False,
        }
        certificate["content_sha256"] = canonical_json_sha256(certificate)
        certificate_path = write_frozen_json(paths["certificate"], certificate)
        certificate_record = file_record(certificate_path)
        model = v21b_interface.build_model_from_source_normalization(normalization)
        model.load_state_dict(selected_state, strict=True)
        checkpoint = {
            "schema": CHECKPOINT_SCHEMA,
            "schema_version": 1,
            "model_class": type(model).__name__,
            "model_architecture": model.architecture(),
            "accepted_v2_authority": accepted_v2_authority(),
            "model_state_dict": dict(selected_state),
            "model_state_dict_sha256": selected_state_sha,
            "normalization_authority": normalization_record,
            "certificate": certificate_record,
            "state_archive": archive_record,
            "selected_step": selected_step,
            "source_access": source_access(),
        }
        checkpoint_path = write_torch_noclobber(paths["checkpoint"], checkpoint)
        checkpoint_record = file_record(checkpoint_path)

    passed = selected_step is not None
    report = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "status": (
            "source_only_v21b_promotion_candidate_complete"
            if passed
            else "source_only_v21b_complete_no_eligible_promotion"
        ),
        "training_contract": training_contract(),
        "training_contract_sha256": TRAINING_CONTRACT_SHA256,
        "execution_authority": {
            "path": inputs.execution["verified_path"],
            "sha256": inputs.execution["verified_sha256"],
        },
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
    write_frozen_json(paths["result"], report)
    return report


def train(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    paths = _output_paths(output)
    _preflight_outputs(paths)
    inputs = prepare_inputs(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
    )
    normalization = build_pilot_normalization(
        [item.base for item in inputs.train],
        source_state_cohort_authority_sha256=inputs.source_state_manifest[
            "authority_sha256"
        ],
    )
    device = torch.device(str(args.device))
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
    epoch_zero_raw = evaluate(model, inputs, normalization, device)
    epoch_zero = attach_promotion(epoch_zero_raw, epoch_zero_raw)
    zero_state = _state_copy(model)
    history: list[dict[str, Any]] = [
        {
            "step": 0,
            "training": None,
            "validation": epoch_zero,
            "model_state_dict_sha256": _state_sha(zero_state),
        }
    ]
    saved_states: dict[int, dict[str, torch.Tensor]] = {0: zero_state}
    for step in range(1, OPTIMIZER_STEPS + 1):
        training = train_one_step(model, optimizer, inputs, normalization, device)
        validation = attach_promotion(
            evaluate(model, inputs, normalization, device),
            epoch_zero,
        )
        state = _state_copy(model)
        row = {
            "step": step,
            "training": training,
            "validation": validation,
            "model_state_dict_sha256": _state_sha(state),
        }
        history.append(row)
        if validation["selection_eligible"] is True:
            saved_states[step] = state
        print(json.dumps(row, sort_keys=True), flush=True)
    if len(history) != OPTIMIZER_STEPS + 1:
        raise RuntimeError("V2.1B did not complete exactly 30 optimizer steps")
    selected_step = select_promotion_step(history)
    if selected_step is not None:
        model.load_state_dict(saved_states[selected_step], strict=True)
        restored = attach_promotion(
            evaluate(model, inputs, normalization, device),
            epoch_zero,
        )
        if restored != history[selected_step]["validation"]:
            raise RuntimeError("V2.1B restored selected validation differs")
    model.cpu()
    return _write_outputs(
        output,
        normalization,
        inputs,
        history,
        saved_states,
        selected_step,
    )


def synthetic_dry_run() -> dict[str, Any]:
    trainable = torch.tensor([True, False, False, True])
    anchors = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    negatives = torch.tensor([1, 0, 3, 2], dtype=torch.int64)
    pairwise, triplet = v21b_loss.hard_negative_denominator_masks(
        trainable, anchors, negatives
    )
    return {
        "schema": "radio_gs.surface_region_v21b_synthetic_dry_run.v1",
        "optimizer_steps": OPTIMIZER_STEPS,
        "authority_pairs": int(anchors.numel()),
        "pairwise_any_endpoint_pairs": int(pairwise.sum()),
        "triplet_anchor_trainable_pairs": int(triplet.sum()),
        "hidden_rank": 256,
        "minimum_relative_improvement": MINIMUM_RELATIVE_IMPROVEMENT,
        "minimum_active_coverage": MINIMUM_ACTIVE_COVERAGE,
        "minimum_pair_coverage": MINIMUM_PAIR_COVERAGE,
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
            "status": "source_only_v21b_authority_validated",
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
    "CERTIFICATE_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "EXECUTION_AUTHORITY_SCHEMA",
    "NORMALIZATION_SCHEMA",
    "RESULT_SCHEMA",
    "STATE_ARCHIVE_SCHEMA",
    "TRAINING_CONTRACT_SHA256",
    "V21BInputs",
    "attach_promotion",
    "build_parser",
    "evaluate",
    "prepare_inputs",
    "promotion_checks",
    "select_promotion_step",
    "source_access",
    "synthetic_dry_run",
    "train",
    "training_contract",
    "validate_execution_authority",
]
