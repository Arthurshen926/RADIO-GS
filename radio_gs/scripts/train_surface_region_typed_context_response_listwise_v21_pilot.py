#!/usr/bin/env python3
"""Run the fixed source-only V2.1 four-train/two-validation pilot.

This entry point opens no benchmark input.  It reuses the frozen V1 scene
loader, routing, normalization, validation, model and optimizer constants,
plus the frozen V2 hard-negative authority loader.  The only new training
path is the preregistered V2.1 complete-scene objective.
"""

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
import torch.nn.functional as F

from radio_gs.interfaces.surface_region_typed_context_training import (
    accepted_v2_authority,
)
from radio_gs.losses.source_global_response_listwise_loss_v21 import (
    FrozenCanonicalNegativeBank,
    FrozenCompositionalGenericBank,
    load_frozen_canonical_negative_bank,
    load_frozen_compositional_generic_bank,
    recommended_v21_config,
)
from radio_gs.losses.source_typed_text_relation_loss_v21 import (
    FrozenTypedTextRelationAuthority,
    load_frozen_typed_text_relation_authority,
)
from radio_gs.models.surface_region_typed_context_residual import (
    SurfaceRegionAcceptedV2TypedContextResidualV1,
)
from radio_gs.scripts import train_surface_region_full_scalar_residual as base_trainer
from radio_gs.scripts import materialize_full_scalar_clean_training_shard as pilot_shard
from radio_gs.scripts import train_surface_region_typed_context_residual as v1_trainer
from radio_gs.scripts import (
    train_surface_region_typed_context_response_listwise_v2 as v2_trainer,
)
from radio_gs.scripts import (
    train_surface_region_typed_context_response_listwise_v21 as v21_adapter,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    sha256_file,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


ARTIFACT_TYPE = "surface_region_typed_context_response_listwise_v21_source_pilot"
EXECUTION_AUTHORITY_SCHEMA = (
    "radio_gs.surface_region_typed_context_response_listwise_v21_pilot_"
    "execution_authority.v1"
)
CHECKPOINT_SCHEMA = (
    "radio_gs.surface_region_typed_context_response_listwise_v21_pilot_checkpoint.v1"
)
TRAIN_SCENES = (
    "scene0001_00",
    "scene0002_00",
    "scene0003_00",
    "scene0005_00",
)
VALIDATION_SCENES = ("scene0004_00", "scene0008_00")
COMPONENT_WEIGHTS = {
    "synonym_relation": 0.20,
    "lexical_sibling_relation": 0.20,
    "counterfactual_attributes": 0.30,
    "high_precision_part_of": 0.05,
}
PRIMARY_WEIGHT = 0.25
ACTIVE_PAIR_ADDENDUM = Path(
    "paper/artifacts/"
    "source_global_response_listwise_loss_v21_active_pair_denominator_"
    "addendum_20260807.json"
)


@dataclass(frozen=True)
class PilotInputs:
    execution: dict[str, Any]
    train: tuple[v2_trainer.ResponseSceneBinding, ...]
    validation: tuple[v2_trainer.ResponseSceneBinding, ...]
    fit: v2_trainer.FitTextBank
    canonical_negative: FrozenCanonicalNegativeBank
    compositional: tuple[FrozenCompositionalGenericBank, ...]
    relations: FrozenTypedTextRelationAuthority
    cohort: dict[str, Any]
    registry: dict[str, Any]
    source_state_manifest: dict[str, Any]


def source_access() -> dict[str, bool]:
    return {
        **v21_adapter.source_access(),
        "typed_relation_authority_opened": True,
        "source_validation_used_for_selection": True,
    }


def training_contract() -> dict[str, Any]:
    config = recommended_v21_config()
    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "cohort": {
            "source_train": list(TRAIN_SCENES),
            "source_validation": list(VALIDATION_SCENES),
            "fixed_before_training": True,
            "scene_and_physical_space_disjoint": True,
        },
        "input_authority": {
            "pilot_training_shard_schema": (
                pilot_shard.PILOT_TRAINING_SHARD_SCHEMA
            ),
            "pilot_training_shard_contract_sha256": (
                pilot_shard.PILOT_TRAINING_SHARD_CONTRACT_SHA256
            ),
            "pilot_cohort_region_view_registry_schema": (
                pilot_shard.PILOT_COHORT_REGISTRY_SCHEMA
            ),
            "pilot_cohort_region_view_registry_contract_sha256": (
                canonical_json_sha256(
                    pilot_shard.pilot_cohort_registry_contract()
                )
            ),
            "one_registry_file_and_content_authority_for_all_six_shards": True,
            "legacy_24plus8_shard_or_registry_schema_accepted": False,
        },
        "forward": "complete_canonical_scene_rows",
        "identity": {
            "epoch_zero_candidate": True,
            "zero_final_projection": True,
            "inactive_or_ood_bitwise_accepted_v2_e0": True,
        },
        "normalization": {
            "fit_split": "fixed_source_train_four_only",
            "validation_contribution": False,
            "statistics": "coordinate_median_and_1p4826_mad",
            "ood_envelope": "maximum_train_robust_linf_and_exact_constant_coordinates",
            "frozen_v1_24_scene_normalization_modified": False,
        },
        "v21": {
            "teacher_multiview_temperature": config.response_temperature,
            "inference_logit_scale": config.inference_logit_scale,
            "auxiliary_weight": config.auxiliary_weight,
            "component_weights": {
                "object_noun_primary": PRIMARY_WEIGHT,
                **COMPONENT_WEIGHTS,
            },
            "typed_relation_authority_required": True,
            "training_pair_denominator": (
                "authority_pairs_with_at_least_one_trainable_endpoint"
            ),
            "validation_pair_denominator": "all_authority_pairs",
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": v1_trainer.LEARNING_RATE,
            "weight_decay": v1_trainer.WEIGHT_DECAY,
            "epochs": v1_trainer.EPOCHS,
            "patience": v1_trainer.PATIENCE,
            "maximum_gradient_norm": v1_trainer.MAX_GRADIENT_NORM,
            "equal_scene_gradient_accumulation": True,
        },
        "selection": {
            "split": "source_validation",
            "benchmark_read": False,
            "v1_non_regression_required": True,
            "primary": "minimum_scene_macro_v21_auxiliary_loss",
            "tie_break": [
                "maximum_v1_mean_all_view_cosine",
                "maximum_v1_p05_row_mean_all_view_cosine",
                "maximum_v1_relation_fidelity",
                "earliest_epoch",
            ],
        },
        "source_access": source_access(),
    }


TRAINING_CONTRACT_SHA256 = canonical_json_sha256(training_contract())


def build_pilot_normalization(
    bindings: Sequence[v1_trainer.SceneBinding],
    *,
    source_state_cohort_authority_sha256: str,
) -> dict[str, Any]:
    if tuple(item.scene_id for item in bindings) != TRAIN_SCENES:
        raise ValueError("pilot normalization requires the fixed four train scenes")
    values: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for binding in bindings:
        scene = load_pilot_scene(binding)
        values.append(v1_trainer._combined_scalars(scene).float().cpu())
        masks.append((scene["eligible"] & scene["typed_context_valid"]).bool().cpu())
    combined = torch.cat(values)
    mask = torch.cat(masks)
    selected = combined[mask]
    if selected.ndim != 2 or selected.shape[1] != 30 or selected.shape[0] < 2:
        raise ValueError("pilot normalization has insufficient train-only rows")
    median = selected.median(dim=0).values
    deviation = (selected - median).abs()
    mad = deviation.median(dim=0).values
    minimum = selected.min(dim=0).values
    maximum = selected.max(dim=0).values
    constant = minimum == maximum
    scale = torch.where(mad > 0, mad * 1.4826, torch.ones_like(mad))
    normalized = (selected - median) / scale
    variable = ~constant
    robust_linf = (
        normalized[:, variable].abs().amax(dim=1)
        if bool(variable.any())
        else torch.zeros(selected.shape[0])
    )
    boundary = v1_trainer.boundary_score(selected[:, 18:])
    return {
        "schema": "radio_gs.v21_pilot_train4_normalization.v1",
        "schema_version": 1,
        "fit_split": "fixed_source_train_four_only",
        "source_state_cohort_authority_sha256": str(
            source_state_cohort_authority_sha256
        ),
        "train_input_records": v1_trainer.input_records(bindings),
        "source_count": int(mask.sum()),
        "median": median.contiguous(),
        "mad": mad.contiguous(),
        "robust_scale": scale.contiguous(),
        "constant_coordinate_mask": constant.contiguous(),
        "source_max_robust_linf": float(robust_linf.max()),
        "source_boundary_score_median": float(boundary.median()),
        "validation_contribution": False,
        "source_access": source_access(),
    }


def pilot_normalization_authority_sha256(normalization: Mapping[str, Any]) -> str:
    content = dict(normalization)
    for name in ("median", "mad", "robust_scale", "constant_coordinate_mask"):
        tensor = torch.as_tensor(content[name]).detach().cpu().contiguous()
        content[name] = {
            "tensor_channel_sha256": base_trainer._tensor_channel_sha256(tensor)
        }
    return canonical_json_sha256(content)


def _pilot_routing(
    scene: Mapping[str, Any], normalization: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = v1_trainer._combined_scalars(scene).float().cpu()
    median = torch.as_tensor(normalization["median"]).float().cpu()
    scale = torch.as_tensor(normalization["robust_scale"]).float().cpu()
    constant = torch.as_tensor(normalization["constant_coordinate_mask"]).bool().cpu()
    if (
        values.ndim != 2
        or values.shape[1] != 30
        or median.shape != (30,)
        or scale.shape != (30,)
        or constant.shape != (30,)
        or bool((scale <= 0).any())
    ):
        raise ValueError("pilot normalization/routing dimensions differ")
    normalized = (values - median) / scale
    variable = ~constant
    score = (
        normalized[:, variable].abs().amax(dim=1)
        if bool(variable.any())
        else torch.zeros(values.shape[0])
    )
    constant_deviation = (
        (values[:, constant] != median[constant]).any(dim=1)
        if bool(constant.any())
        else torch.zeros(values.shape[0], dtype=torch.bool)
    )
    ood = constant_deviation | (score > float(normalization["source_max_robust_linf"]))
    eligible = scene["eligible"].bool().cpu()
    declared = scene["typed_context_valid"].bool().cpu()
    effective_ood = ood | ~eligible
    active = declared & ~effective_ood
    return declared, effective_ood, active


def _record(value: object, *, label: str) -> dict[str, str]:
    return v2_trainer._record(value, label=label)


def _scene_records(value: object, *, split: str) -> list[dict[str, Any]]:
    expected = TRAIN_SCENES if split == "source_train" else VALIDATION_SCENES
    if not isinstance(value, list) or len(value) != len(expected):
        raise ValueError(f"{split} requires exactly {len(expected)} scenes")
    records = [dict(item) for item in value if isinstance(item, Mapping)]
    if len(records) != len(expected):
        raise ValueError(f"{split} scene records differ")
    if tuple(str(item.get("scene_id")) for item in records) != expected:
        raise ValueError(f"{split} must equal the fixed pilot cohort")
    required = {
        "scene_id",
        "training_shard",
        "adaptive_context",
        "hard_negative_authority",
        "hard_negative_content_authority_sha256",
    }
    if any(set(item) != required for item in records):
        raise ValueError(f"{split} scene execution fields differ")
    return records


def validate_execution_authority(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("V2.1 pilot execution authority must be a mapping")
    authority = dict(value)
    required = {
        "schema",
        "schema_version",
        "status",
        "implementation",
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
        "training_authorized",
        "benchmark_execution_authorized",
        "source_access",
    }
    if set(authority) != required:
        raise ValueError("V2.1 pilot execution authority fields differ")
    if (
        authority.get("schema") != EXECUTION_AUTHORITY_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("status")
        != "authorized_source_only_4train_2validation_v21_pilot"
        or authority.get("training_authorized") is not True
        or authority.get("benchmark_execution_authorized") is not False
        or authority.get("source_access") != source_access()
    ):
        raise ValueError("V2.1 pilot execution authority header differs")
    for name in (
        "implementation",
        "active_pair_addendum",
        "cohort_authority",
        "pilot_cohort_region_view_registry",
        "benchmark_exclusion_manifest",
        "fit_text_bank",
        "canonical_negative_bank",
    ):
        authority[name] = _record(authority[name], label=name)
    banks = authority["compositional_banks"]
    if not isinstance(banks, Mapping) or set(banks) != set(COMPONENT_WEIGHTS):
        raise ValueError("V2.1 pilot compositional component set differs")
    normalized_banks: dict[str, dict[str, Any]] = {}
    for name, expected_weight in COMPONENT_WEIGHTS.items():
        raw = banks[name]
        if not isinstance(raw, Mapping) or set(raw) != {
            "path",
            "sha256",
            "loss_weight",
        }:
            raise ValueError(f"V2.1 pilot {name} record differs")
        if float(raw["loss_weight"]) != expected_weight:
            raise ValueError(f"V2.1 pilot {name} weight differs")
        normalized_banks[name] = {
            **_record({"path": raw["path"], "sha256": raw["sha256"]}, label=name),
            "loss_weight": expected_weight,
        }
    authority["compositional_banks"] = normalized_banks
    relation = authority["typed_relation_authority"]
    if not isinstance(relation, Mapping) or set(relation) != {
        "path",
        "sha256",
        "content_authority_sha256",
    }:
        raise ValueError("V2.1 pilot typed relation record differs")
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


def load_pilot_training_shard(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_split: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Load only the independently versioned exact-4+2 pilot shard."""

    return pilot_shard.load_pilot_training_shard(
        path,
        expected_sha256=expected_sha256,
        expected_split=expected_split,
    )


def load_pilot_scene(binding: v1_trainer.SceneBinding) -> dict[str, Any]:
    """Pilot equivalent of V1 ``load_scene`` with no 24+8 loader fallback."""

    shard, shard_record = load_pilot_training_shard(
        binding.training_shard["path"],
        expected_sha256=binding.training_shard["sha256"],
        expected_split=binding.split,
    )
    context, context_record = v1_trainer._load_context(
        binding.adaptive_context["path"],
        expected_sha256=binding.adaptive_context["sha256"],
    )
    scene = v1_trainer._validate_scene_pair(shard, context)
    if (
        scene != binding.scene_id
        or shard_record != binding.training_shard
        or context_record != binding.adaptive_context
    ):
        raise ValueError("pilot typed-context scene binding changed")
    return {
        **shard,
        "pooled_context_radio_direction": context[
            "pooled_context_radio_direction"
        ].float(),
        "typed_context_statistics": context["typed_context_statistics"].float(),
        "typed_context_valid": context["typed_context_valid"].bool(),
        "context_token_count": context["context_token_count"].long(),
        "context_input_authority": dict(context["input_authority"]),
    }


def load_pilot_bound_scene(
    binding: v2_trainer.ResponseSceneBinding,
) -> dict[str, Any]:
    """Pilot response binding loader; never re-enters the legacy shard loader."""

    scene = load_pilot_scene(binding.base)
    payload = binding.response_authority.payload
    inputs = payload["input_authority"]
    accepted_channels = inputs["accepted_v2"]["channel_sha256"]
    teacher_channels = inputs["official_multiview_siglip2_teacher"][
        "channel_sha256"
    ]
    checks = {
        "accepted_v2_e0": (
            base_trainer._tensor_channel_sha256(scene["accepted_v2_e0"]),
            accepted_channels["accepted_v2_e0"],
        ),
        "teacher_pair_descriptors": (
            base_trainer._tensor_channel_sha256(
                scene["official_multiview_siglip2_teacher_pair_descriptors"]
            ),
            teacher_channels["pair_descriptors"],
        ),
        "teacher_pair_region_indices": (
            base_trainer._tensor_channel_sha256(
                scene["official_multiview_siglip2_teacher_pair_region_indices"]
            ),
            teacher_channels["pair_region_indices"],
        ),
        "canonical_region_indices": (
            scene["sampling_audit"]["canonical_region_indices_sha256"],
            accepted_channels["canonical_region_indices"],
        ),
        "hard_negative_canonical_region_indices": (
            base_trainer._tensor_channel_sha256(
                payload["canonical_region_indices"]
            ),
            accepted_channels["canonical_region_indices"],
        ),
    }
    for label, (observed, expected) in checks.items():
        if observed != expected:
            raise ValueError(f"pilot scene {label} differs from hard-negative authority")
    return scene


def _base_bindings(
    records: Sequence[Mapping[str, Any]],
    *,
    split: str,
    expected_lineage: Mapping[str, object],
    registry_scene_records: Mapping[str, Mapping[str, Any]],
) -> tuple[list[v1_trainer.SceneBinding], list[dict[str, str]]]:
    bindings: list[v1_trainer.SceneBinding] = []
    channel_summaries: list[dict[str, str]] = []
    for item in records:
        shard_record = _record(item["training_shard"], label="training shard")
        context_record = _record(item["adaptive_context"], label="adaptive context")
        shard, observed_shard = load_pilot_training_shard(
            shard_record["path"],
            expected_sha256=shard_record["sha256"],
            expected_split=split,
        )
        context, observed_context = v1_trainer._load_context(
            context_record["path"], expected_sha256=context_record["sha256"]
        )
        scene_id = v1_trainer._validate_scene_pair(shard, context)
        lineage = shard["lineage"]
        registry_scene = registry_scene_records.get(scene_id)
        if (
            scene_id != str(item["scene_id"])
            or observed_shard != shard_record
            or observed_context != context_record
            or registry_scene is None
            or any(
                lineage.get(name) != expected
                for name, expected in expected_lineage.items()
            )
        ):
            raise ValueError("V2.1 pilot source scene binding differs")
        context_inputs = context["input_authority"]
        if (
            context_inputs["accepted_v2_canonical_region_authority"]["sha256"]
            != registry_scene["accepted_region_authority_file_sha256"]
            or context_inputs["factorized_primitive_state"]["sha256"]
            != registry_scene["factorized_state_file_sha256"]
        ):
            raise ValueError(
                f"scene {scene_id} adaptive context differs from pilot registry files"
            )
        bindings.append(
            v1_trainer.SceneBinding(split, scene_id, observed_shard, observed_context)
        )
        channel_summaries.append(
            {
                "accepted_v2_e0": shard["channel_sha256"]["accepted_v2_e0"],
                "teacher_pair_descriptors": shard["channel_sha256"][
                    "official_multiview_siglip2_teacher_pair_descriptors"
                ],
                "teacher_pair_region_indices": shard["channel_sha256"][
                    "official_multiview_siglip2_teacher_pair_region_indices"
                ],
                "canonical_region_indices": shard["sampling_audit"][
                    "canonical_region_indices_sha256"
                ],
            }
        )
        del shard, context
    return bindings, channel_summaries


def _validate_response_shard_channel_summary(
    binding: v2_trainer.ResponseSceneBinding,
    summary: Mapping[str, str],
    registry_scene: Mapping[str, Any],
) -> None:
    """Close HN-to-pilot-shard channel binding before training or promotion."""

    payload = binding.response_authority.payload
    inputs = payload["input_authority"]
    accepted_channels = inputs["accepted_v2"]["channel_sha256"]
    teacher_channels = inputs["official_multiview_siglip2_teacher"][
        "channel_sha256"
    ]
    expected = {
        "accepted_v2_e0": accepted_channels["accepted_v2_e0"],
        "teacher_pair_descriptors": teacher_channels["pair_descriptors"],
        "teacher_pair_region_indices": teacher_channels["pair_region_indices"],
        "canonical_region_indices": accepted_channels[
            "canonical_region_indices"
        ],
    }
    if (
        dict(summary) != expected
        or base_trainer._tensor_channel_sha256(
            payload["canonical_region_indices"]
        )
        != accepted_channels["canonical_region_indices"]
        or inputs["accepted_v2"]["sha256"]
        != registry_scene["accepted_region_authority_file_sha256"]
        or inputs["official_multiview_siglip2_teacher"]["sha256"]
        != registry_scene["teacher_observation_authority_file_sha256"]
    ):
        raise ValueError(
            f"scene {binding.scene_id} hard-negative channels differ from pilot shard"
        )


def _validate_subset_authorities(
    authority: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    cohort, cohort_file = base_trainer.load_cohort_authority(
        _validated_path(authority["cohort_authority"], label="cohort authority"),
        expected_sha256=authority["cohort_authority"]["sha256"],
    )
    exclusion, exclusion_file = base_trainer._load_json_manifest(
        _validated_path(
            authority["benchmark_exclusion_manifest"],
            label="benchmark exclusion manifest",
        ),
        expected_sha256=authority["benchmark_exclusion_manifest"]["sha256"],
        label="benchmark exclusion manifest",
        validator=base_trainer.validate_benchmark_exclusion_manifest,
    )
    if not set(TRAIN_SCENES).issubset(cohort["source_train_scene_ids"]):
        raise ValueError("pilot train scenes are outside the clean cohort authority")
    if not set(VALIDATION_SCENES).issubset(cohort["source_validation_scene_ids"]):
        raise ValueError(
            "pilot validation scenes are outside the clean cohort authority"
        )
    scenes = set(TRAIN_SCENES) | set(VALIDATION_SCENES)
    spaces = {base_trainer.canonical_physical_space_id(item) for item in scenes}
    if (
        len(spaces) != len(scenes)
        or scenes & set(exclusion["scene_ids"])
        or spaces & set(exclusion["physical_space_ids"])
    ):
        raise ValueError("pilot cohort overlaps or aliases a benchmark physical space")
    if cohort["benchmark_exclusion"] != {
        "manifest_authority_sha256": exclusion["authority_sha256"],
        "manifest_file_sha256": exclusion_file["sha256"],
    }:
        raise ValueError("pilot cohort/exclusion authority binding differs")
    return cohort, cohort_file, exclusion_file


def prepare_inputs(path: str | Path, *, expected_sha256: str) -> PilotInputs:
    raw, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="V2.1 source pilot execution authority",
    )
    authority = validate_execution_authority(raw)
    implementation = _validated_path(
        authority["implementation"], label="V2.1 pilot implementation"
    )
    if implementation != Path(__file__).resolve():
        raise ValueError("execution authority binds another V2.1 pilot implementation")
    addendum = _validated_path(
        authority["active_pair_addendum"], label="active-pair addendum"
    )
    if addendum != (Path(__file__).resolve().parents[2] / ACTIVE_PAIR_ADDENDUM):
        raise ValueError("execution authority binds another active-pair addendum")
    cohort, cohort_file, _exclusion_file = _validate_subset_authorities(authority)
    registry_record = authority["pilot_cohort_region_view_registry"]
    registry_raw, registry_digest, registry_path = load_json_object(
        _validated_path(registry_record, label="pilot cohort region/view registry"),
        expected_sha256=registry_record["sha256"],
        label="pilot cohort region/view registry",
    )
    registry = pilot_shard.validate_pilot_cohort_region_view_registry(
        registry_raw,
        cohort_authority=cohort,
        cohort_authority_file_sha256=cohort_file["sha256"],
    )
    if {
        "path": str(registry_path),
        "sha256": registry_digest,
    } != registry_record:
        raise ValueError("pilot registry verified file record differs")
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
        "teacher_manifest_file_sha256": _frozen_json_file_sha256(
            teacher_manifest
        ),
    }
    fit = v2_trainer.load_fit_text_bank(
        _validated_path(authority["fit_text_bank"], label="primary fit bank"),
        expected_sha256=authority["fit_text_bank"]["sha256"],
    )
    canonical = load_frozen_canonical_negative_bank(
        _validated_path(
            authority["canonical_negative_bank"], label="canonical-negative bank"
        ),
        expected_file_sha256=authority["canonical_negative_bank"]["sha256"],
    )
    compositional = tuple(
        load_frozen_compositional_generic_bank(
            _validated_path(
                {
                    "path": authority["compositional_banks"][name]["path"],
                    "sha256": authority["compositional_banks"][name]["sha256"],
                },
                label=name,
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
            {
                "path": relation_record["path"],
                "sha256": relation_record["sha256"],
            },
            label="typed relation authority",
        ),
        expected_file_sha256=relation_record["sha256"],
    )
    if (
        relations.content_authority_sha256
        != relation_record["content_authority_sha256"]
    ):
        raise ValueError("typed relation content authority differs")

    def bind(split: str) -> tuple[v2_trainer.ResponseSceneBinding, ...]:
        records = authority[split]
        bases, summaries = _base_bindings(
            records,
            split=split,
            expected_lineage=expected_lineage,
            registry_scene_records=registry_scene_records,
        )
        result: list[v2_trainer.ResponseSceneBinding] = []
        for base, item, summary in zip(bases, records, summaries):
            response = v2_trainer.bind_response_scene(
                base, item, fit_text_bank=fit
            )
            _validate_response_shard_channel_summary(
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
    canonical: FrozenCanonicalNegativeBank,
    compositional: Sequence[FrozenCompositionalGenericBank],
    relations: FrozenTypedTextRelationAuthority,
    device: torch.device,
    *,
    training: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int | bool]]:
    scene = load_pilot_bound_scene(binding)
    routing = _pilot_routing(scene, normalization)
    return v21_adapter.complete_scene_objective_v21(
        model,
        scene,
        normalization,
        fit,
        binding.response_authority,
        canonical,
        device,
        compositional_banks=compositional,
        relation_authority=relations,
        exclude_both_immutable_pairs=training,
        routing_masks=routing,
    )


def train_one_epoch(
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
            inputs.canonical_negative,
            inputs.compositional,
            inputs.relations,
            device,
            training=True,
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
        "both_immutable_pair_filter": True,
        "per_scene": per_scene,
        "scene_macro_auxiliary_loss": sum(
            float(row["response_auxiliary_loss"]) for row in per_scene.values()
        )
        / len(per_scene),
        "scene_macro_pair_trainable_endpoint_coverage": sum(
            float(row["response_pair_trainable_endpoint_coverage"])
            for row in per_scene.values()
        )
        / len(per_scene),
    }


@torch.no_grad()
def _evaluate_v1_scene(
    model: SurfaceRegionAcceptedV2TypedContextResidualV1,
    scene: Mapping[str, Any],
    normalization: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    declared, effective_ood, active = _pilot_routing(scene, normalization)
    if int(active.sum()) < 2:
        raise ValueError("pilot validation scene requires two active rows")
    bases: list[torch.Tensor] = []
    candidates: list[torch.Tensor] = []
    prototypes: list[torch.Tensor] = []
    base_cosines: list[torch.Tensor] = []
    candidate_cosines: list[torch.Tensor] = []
    fallback_equal = True
    row_count = len(scene["region_row_ids"])
    for start in range(0, row_count, v1_trainer.EVAL_BATCH_ROWS):
        rows = torch.arange(start, min(start + v1_trainer.EVAL_BATCH_ROWS, row_count))
        base = scene["accepted_v2_e0"][rows].to(device)
        result = model.forward_with_diagnostics(
            base,
            scene["pooled_context_radio_direction"][rows].to(device),
            scene["raw_full_scalar_summary"][rows].to(device),
            scene["typed_context_statistics"][rows].to(device),
            active_mask=declared[rows].to(device),
            ood_mask=effective_ood[rows].to(device),
        )
        fallback = ~active[rows].to(device)
        fallback_equal = fallback_equal and torch.equal(
            result.semantic_descriptor[fallback], base[fallback]
        )
        selected = active[rows]
        if bool(selected.any()):
            active_rows = rows[selected]
            teachers, teacher_mask = v1_trainer.gather_sparse_teacher_batch(
                scene, active_rows
            )
            prototype = v1_trainer.teacher_prototype(teachers, teacher_mask)
            base_active = base[selected.to(device)].cpu()
            candidate_active = result.semantic_descriptor[selected.to(device)].cpu()
            base_view = torch.einsum(
                "bd,bvd->bv", F.normalize(base_active, dim=-1), teachers
            )
            candidate_view = torch.einsum(
                "bd,bvd->bv", F.normalize(candidate_active, dim=-1), teachers
            )
            base_cosines.append((base_view * teacher_mask).sum(1) / teacher_mask.sum(1))
            candidate_cosines.append(
                (candidate_view * teacher_mask).sum(1) / teacher_mask.sum(1)
            )
            bases.append(base_active)
            candidates.append(candidate_active)
            prototypes.append(prototype)
    base_rows = torch.cat(bases)
    candidate_rows = torch.cat(candidates)
    prototype_rows = torch.cat(prototypes)
    base_cosine = torch.cat(base_cosines)
    candidate_cosine = torch.cat(candidate_cosines)
    if base_rows.shape[0] > v1_trainer.RELATION_EVAL_ROWS:
        selected = (
            torch.linspace(0, base_rows.shape[0] - 1, v1_trainer.RELATION_EVAL_ROWS)
            .round()
            .long()
            .unique()
        )
        base_relation = base_rows[selected]
        candidate_relation = candidate_rows[selected]
        prototype_relation = prototype_rows[selected]
    else:
        base_relation = base_rows
        candidate_relation = candidate_rows
        prototype_relation = prototype_rows
    base_metrics = {
        "mean_all_view_cosine": float(base_cosine.mean()),
        "p05_row_mean_all_view_cosine": v1_trainer._lower_quantile(base_cosine, 0.05),
        "relation_fidelity": v1_trainer._relation_fidelity(
            base_relation, prototype_relation
        ),
    }
    candidate_metrics = {
        "mean_all_view_cosine": float(candidate_cosine.mean()),
        "p05_row_mean_all_view_cosine": v1_trainer._lower_quantile(
            candidate_cosine, 0.05
        ),
        "relation_fidelity": v1_trainer._relation_fidelity(
            candidate_relation, prototype_relation
        ),
    }
    delta = {
        name: candidate_metrics[name] - base_metrics[name] for name in base_metrics
    }
    return {
        "base": base_metrics,
        "candidate": candidate_metrics,
        "candidate_minus_base": delta,
        "active_rows": int(active.sum()),
        "inactive_fallback_rows": int((~active).sum()),
        "fallback_bitwise_accepted_v2_e0": bool(fallback_equal),
        "relation_evaluation_rows": int(base_relation.shape[0]),
        "validation_no_grad": not torch.is_grad_enabled(),
    }


@torch.no_grad()
def _evaluate_v1(
    model: SurfaceRegionAcceptedV2TypedContextResidualV1,
    bindings: Sequence[v2_trainer.ResponseSceneBinding],
    normalization: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    per_scene = {
        binding.scene_id: _evaluate_v1_scene(
            model,
            load_pilot_bound_scene(binding),
            normalization,
            device,
        )
        for binding in bindings
    }
    names = (
        "mean_all_view_cosine",
        "p05_row_mean_all_view_cosine",
        "relation_fidelity",
    )
    base = {
        name: sum(row["base"][name] for row in per_scene.values()) / len(per_scene)
        for name in names
    }
    candidate = {
        name: sum(row["candidate"][name] for row in per_scene.values()) / len(per_scene)
        for name in names
    }
    delta = {name: candidate[name] - base[name] for name in names}
    scene_mean = [
        row["candidate_minus_base"]["mean_all_view_cosine"]
        for row in per_scene.values()
    ]
    checks = {
        "macro_mean_all_view_cosine": (
            delta["mean_all_view_cosine"] >= -v1_trainer.NON_REGRESSION_TOLERANCE
        ),
        "macro_p05_row_mean_all_view_cosine": (
            delta["p05_row_mean_all_view_cosine"]
            >= -v1_trainer.NON_REGRESSION_TOLERANCE
        ),
        "macro_relation_fidelity": (
            delta["relation_fidelity"] >= -v1_trainer.NON_REGRESSION_TOLERANCE
        ),
        "paired_scene_worst_mean_delta": (
            min(scene_mean) >= -v1_trainer.NON_REGRESSION_TOLERANCE
        ),
        "every_scene_two_active_rows": all(
            row["active_rows"] >= 2 for row in per_scene.values()
        ),
        "fallback_bitwise_accepted_v2_e0": all(
            row["fallback_bitwise_accepted_v2_e0"] for row in per_scene.values()
        ),
    }
    return {
        "aggregation": "scene_macro",
        "base": base,
        "candidate": candidate,
        "candidate_minus_base": delta,
        "paired_scene_mean_delta": {
            "minimum": min(scene_mean),
            "p05": sorted(scene_mean)[
                int(math.floor(0.05 * max(0, len(scene_mean) - 1)))
            ],
            "maximum": max(scene_mean),
        },
        "per_scene": per_scene,
        "non_regression_checks": checks,
        "non_regression_passed": all(checks.values()),
        "validation_no_grad": not torch.is_grad_enabled(),
        "global_or_split_teacher_densification": False,
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
            inputs.canonical_negative,
            inputs.compositional,
            inputs.relations,
            device,
            training=False,
        )
        row = {
            "combined_objective": float(total.detach().cpu()),
            **v2_trainer._float_metrics(metrics),
        }
        if (
            row["response_objective_hard_negative_pairs"]
            != row["response_authority_hard_negative_pairs"]
        ):
            raise RuntimeError("validation did not retain every authority pair")
        per_scene[binding.scene_id] = row
    v1 = _evaluate_v1(model, inputs.validation, normalization, device)
    response = {
        "scene_count": len(per_scene),
        "scene_macro_auxiliary_loss": sum(
            float(row["response_auxiliary_loss"]) for row in per_scene.values()
        )
        / len(per_scene),
        "scene_macro_pair_trainable_endpoint_coverage": sum(
            float(row["response_pair_trainable_endpoint_coverage"])
            for row in per_scene.values()
        )
        / len(per_scene),
        "all_authority_pairs_retained": True,
        "per_scene": per_scene,
    }
    return {
        "v1_non_regression": v1,
        "response_listwise_v21": response,
        "selection_eligible": bool(v1["non_regression_passed"]),
        "validation_no_grad": not torch.is_grad_enabled(),
        "benchmark_opened": False,
    }


def select_best_epoch(history: Sequence[Mapping[str, Any]]) -> int:
    if not history or [int(row.get("epoch", -1)) for row in history] != list(
        range(len(history))
    ):
        raise ValueError("V2.1 pilot history must be contiguous from epoch zero")
    eligible = [
        row
        for row in history
        if row.get("validation", {}).get("selection_eligible") is True
    ]
    if not eligible:
        raise RuntimeError("V2.1 pilot has no source-validation-eligible checkpoint")

    def rank(row: Mapping[str, Any]) -> tuple[float, float, float, float, int]:
        validation = row["validation"]
        response = validation["response_listwise_v21"]
        candidate = validation["v1_non_regression"]["candidate"]
        return (
            -float(response["scene_macro_auxiliary_loss"]),
            float(candidate["mean_all_view_cosine"]),
            float(candidate["p05_row_mean_all_view_cosine"]),
            float(candidate["relation_fidelity"]),
            -int(row["epoch"]),
        )

    return int(max(eligible, key=rank)["epoch"])


def _state_copy(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().contiguous().clone()
        for name, value in model.state_dict().items()
    }


def _input_records(
    bindings: Sequence[v2_trainer.ResponseSceneBinding],
) -> list[dict[str, Any]]:
    return v2_trainer.input_records(bindings)


def _write_outputs(
    output: Path,
    model: SurfaceRegionAcceptedV2TypedContextResidualV1,
    normalization: Mapping[str, Any],
    inputs: PilotInputs,
    history: Sequence[Mapping[str, Any]],
    selected_epoch: int,
    selected_validation: Mapping[str, Any],
) -> dict[str, Any]:
    normalization_path = write_torch_noclobber(
        output.with_suffix(output.suffix + ".normalization.pt"), normalization
    )
    state = _state_copy(model)
    state_sha = v1_trainer._state_sha(model)
    certificate = {
        "schema": (
            "radio_gs.surface_region_typed_context_response_listwise_v21_"
            "pilot_certificate.v1"
        ),
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
        "selected_epoch": selected_epoch,
        "selected_validation": dict(selected_validation),
        "model_state_dict_sha256": state_sha,
        "normalization_authority": file_record(normalization_path),
        "normalization_content_authority_sha256": (
            pilot_normalization_authority_sha256(normalization)
        ),
        "source_access": source_access(),
        "benchmark_opened": False,
    }
    certificate["content_sha256"] = canonical_json_sha256(certificate)
    certificate_path = write_frozen_json(
        output.with_suffix(output.suffix + ".certificate.json"), certificate
    )
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": 1,
        "model_class": type(model).__name__,
        "model_architecture": model.architecture(),
        "accepted_v2_authority": accepted_v2_authority(),
        "model_state_dict": state,
        "model_state_dict_sha256": state_sha,
        "normalization_authority": file_record(normalization_path),
        "certificate": file_record(certificate_path),
        "selected_epoch": selected_epoch,
        "source_access": source_access(),
    }
    checkpoint_path = write_torch_noclobber(output, checkpoint)
    report = {
        "schema": (
            "radio_gs.surface_region_typed_context_response_listwise_v21_"
            "pilot_result.v1"
        ),
        "schema_version": 1,
        "status": "source_only_pilot_complete_no_benchmark_execution",
        "training_contract": training_contract(),
        "training_contract_sha256": TRAINING_CONTRACT_SHA256,
        "execution_authority": {
            "path": inputs.execution["verified_path"],
            "sha256": inputs.execution["verified_sha256"],
        },
        "checkpoint": file_record(checkpoint_path),
        "normalization_authority": file_record(normalization_path),
        "certificate": file_record(certificate_path),
        "selected_epoch": selected_epoch,
        "automatic_fallback_to_epoch_zero": selected_epoch == 0,
        "selected_validation": dict(selected_validation),
        "history": list(history),
        "source_access": source_access(),
        "benchmark_opened": False,
    }
    write_frozen_json(output.with_suffix(output.suffix + ".json"), report)
    return report


def train(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    outputs = (
        output,
        output.with_suffix(output.suffix + ".normalization.pt"),
        output.with_suffix(output.suffix + ".certificate.json"),
        output.with_suffix(output.suffix + ".json"),
    )
    existing = [path for path in outputs if path.exists() or path.is_symlink()]
    if existing:
        raise FileExistsError(
            "V2.1 pilot first-writer outputs already exist: "
            + ", ".join(str(path) for path in existing)
        )
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
    if not epoch_zero["selection_eligible"]:
        raise RuntimeError("V2.1 pilot epoch-zero identity did not pass")
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "training": None,
            "validation": epoch_zero,
            "model_state_dict_sha256": v1_trainer._state_sha(model),
        }
    ]
    best_epoch = 0
    best_state = _state_copy(model)
    stale = 0
    for epoch in range(1, v1_trainer.EPOCHS + 1):
        training = train_one_epoch(model, optimizer, inputs, normalization, device)
        validation = evaluate(model, inputs, normalization, device)
        row = {
            "epoch": epoch,
            "training": training,
            "validation": validation,
            "model_state_dict_sha256": v1_trainer._state_sha(model),
        }
        history.append(row)
        selected = select_best_epoch(history)
        if selected == epoch:
            best_epoch = epoch
            best_state = _state_copy(model)
            stale = 0
        else:
            stale += 1
        print(json.dumps(row, sort_keys=True), flush=True)
        if stale >= v1_trainer.PATIENCE:
            break
    selected_epoch = select_best_epoch(history)
    if selected_epoch != best_epoch:
        raise RuntimeError("V2.1 pilot selected/best epoch differs")
    model.load_state_dict(best_state, strict=True)
    selected_validation = evaluate(model, inputs, normalization, device)
    if selected_validation != history[selected_epoch]["validation"]:
        raise RuntimeError("V2.1 pilot restored validation metrics differ")
    model.cpu()
    return _write_outputs(
        output,
        model,
        normalization,
        inputs,
        history,
        selected_epoch,
        selected_validation,
    )


def synthetic_dry_run() -> dict[str, Any]:
    anchors = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    negatives = torch.tensor([1, 0, 3, 2], dtype=torch.int64)
    trainable = torch.tensor([True, False, False, False])
    retained = trainable[anchors] | trainable[negatives]
    history = [
        {
            "epoch": 0,
            "validation": {
                "selection_eligible": True,
                "response_listwise_v21": {"scene_macro_auxiliary_loss": 0.5},
                "v1_non_regression": {
                    "candidate": {
                        "mean_all_view_cosine": 0.5,
                        "p05_row_mean_all_view_cosine": 0.4,
                        "relation_fidelity": 0.3,
                    }
                },
            },
        },
        {
            "epoch": 1,
            "validation": {
                "selection_eligible": True,
                "response_listwise_v21": {"scene_macro_auxiliary_loss": 0.4},
                "v1_non_regression": {
                    "candidate": {
                        "mean_all_view_cosine": 0.5,
                        "p05_row_mean_all_view_cosine": 0.4,
                        "relation_fidelity": 0.3,
                    }
                },
            },
        },
    ]
    return {
        "schema": "radio_gs.v21_pilot_synthetic_dry_run.v1",
        "complete_canonical_rows": 4,
        "authority_pairs": int(anchors.numel()),
        "training_objective_pairs": int(retained.sum()),
        "both_immutable_pairs_excluded": int((~retained).sum()),
        "validation_objective_pairs": int(anchors.numel()),
        "selected_epoch": select_best_epoch(history),
        "temperature": recommended_v21_config().response_temperature,
        "component_weights": {
            "object_noun_primary": PRIMARY_WEIGHT,
            **COMPONENT_WEIGHTS,
        },
        "typed_relation_authority_required": True,
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
            "status": "source_only_v21_pilot_authority_validated",
            "source_train": [item.scene_id for item in inputs.train],
            "source_validation": [item.scene_id for item in inputs.validation],
            "benchmark_opened": False,
        }
    else:
        result = train(args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
