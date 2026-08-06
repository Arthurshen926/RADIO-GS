#!/usr/bin/env python3
"""Train the preregistered scene-global response/listwise V2 residual.

This is a new authority-driven trainer.  The frozen V1 trainer, model,
evaluator, renderer, and response/listwise loss are imported but never
modified.  Each source scene is forwarded in its complete canonical row
order; only query and sealed hard-negative axes are internally chunked by the
frozen auxiliary loss.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.evaluation.source_query_response_hard_negatives import (
    validate_negative_authority,
)
from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces.surface_region_typed_context_training import (
    build_typed_context_training_certificate,
    typed_context_normalization_authority_sha256,
    typed_context_training_source_access,
    write_typed_context_checkpoint,
)
from radio_gs.losses.source_global_response_listwise_loss import (
    FrozenSourceResponseAuthority,
    load_frozen_source_response_authority,
    recommended_v2_config,
    source_global_response_listwise_loss,
)
from radio_gs.models.surface_region_typed_context_residual import (
    SurfaceRegionAcceptedV2TypedContextResidualV1,
)
from radio_gs.scripts import train_surface_region_full_scalar_residual as base_trainer
from radio_gs.scripts import train_surface_region_typed_context_residual as v1_trainer
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_torch_mapping,
    sha256_file,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


ARTIFACT_TYPE = "surface_region_typed_context_response_listwise_v2"
EXECUTION_AUTHORITY_SCHEMA = (
    "radio_gs.surface_region_typed_context_response_listwise_v2_execution_authority.v1"
)
PREREGISTRATION_ARTIFACT = (
    "surface_region_response_listwise_v2_execution_preregistration"
)
V1_TRAINER_SHA256 = "02e9ba0124134b8d4bfe0ee6d76ae73fdde32fc8e7a54dc9efc8db5b0845f680"
V1_MODEL_SHA256 = "06287aa06223da788a2f1b95fc6906cb8496e36d722eff594d1d397c243ae36f"
FROZEN_LOSS_SHA256 = "552e7bf0e4d83e9346af731e6ce9eaf891968b14f32b49f728b5188c5e012ae7"


def source_access() -> dict[str, bool]:
    return {
        **typed_context_training_source_access(),
        "generic_target_blind_text_bank_opened": True,
        "benchmark_text_queries_opened": False,
    }


def training_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "oracle_decision": "bounded_single_center_has_headroom_and_k2_is_not_justified",
        "frozen_v1": {
            "trainer_sha256": V1_TRAINER_SHA256,
            "model_sha256": V1_MODEL_SHA256,
            "training_contract_sha256": v1_trainer.TRAINING_CONTRACT_SHA256,
            "max_angle_radians": v1_trainer.MAX_ANGLE_RADIANS,
            "max_alpha": v1_trainer.MAX_ALPHA,
            "zero_final_projection": True,
            "inactive_or_ood_bitwise_fallback": True,
        },
        "frozen_auxiliary": {
            "implementation_sha256": FROZEN_LOSS_SHA256,
            "configuration": recommended_v2_config().__dict__,
            "new_learnable_parameters": False,
        },
        "scene_unit": {
            "forward": "complete canonical scene rows",
            "base_objective": "frozen V1 typed_context_objective over all active rows",
            "auxiliary_objective": "frozen source_global_response_listwise_loss",
            "partial_scene": "reject",
            "equal_scene_gradient_accumulation": True,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": v1_trainer.LEARNING_RATE,
            "weight_decay": v1_trainer.WEIGHT_DECAY,
            "epochs": v1_trainer.EPOCHS,
            "patience": v1_trainer.PATIENCE,
            "maximum_gradient_norm": v1_trainer.MAX_GRADIENT_NORM,
        },
        "selection": {
            "split": "source_validation",
            "required_v1_non_regression": True,
            "primary": "minimum scene_macro_source_global_auxiliary_loss",
            "tie_break": [
                "maximum_v1_mean_all_view_cosine",
                "maximum_v1_p05_row_mean_all_view_cosine",
                "maximum_v1_relation_fidelity",
                "earliest_epoch",
            ],
            "epoch_zero_identity_candidate": True,
        },
        "source_access": source_access(),
    }


TRAINING_CONTRACT_SHA256 = canonical_json_sha256(training_contract())


@dataclass(frozen=True)
class FitTextBank:
    embeddings: torch.Tensor
    record: dict[str, str]
    embedding_tensor_sha256: str


@dataclass(frozen=True)
class ResponseSceneBinding:
    base: v1_trainer.SceneBinding
    hard_negative: dict[str, str]
    hard_negative_content_authority_sha256: str
    response_authority: FrozenSourceResponseAuthority

    @property
    def scene_id(self) -> str:
        return self.base.scene_id


def _sha(value: object, *, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return digest


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must be an exact file record")
    path = str(value["path"])
    digest = _sha(value["sha256"], label=f"{label} SHA-256")
    return {"path": path, "sha256": digest}


def _validate_preregistration(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("response/listwise V2 preregistration must be a mapping")
    prereg = dict(value)
    if (
        prereg.get("artifact") != PREREGISTRATION_ARTIFACT
        or prereg.get("schema_version") != 1
        or prereg.get("status") != "sealed_before_v2_trainer_implementation_or_training"
        or prereg.get("execution_gate", {}).get(
            "one_scene_source_only_dry_loss_authorized"
        )
        is not True
        or prereg.get("execution_gate", {}).get("benchmark_execution_authorized")
        is not False
    ):
        raise ValueError("response/listwise V2 preregistration differs")
    frozen = prereg.get("frozen_v1", {})
    auxiliary = prereg.get("frozen_auxiliary_loss", {})
    if (
        frozen.get("trainer", {}).get("sha256") != V1_TRAINER_SHA256
        or frozen.get("model", {}).get("sha256") != V1_MODEL_SHA256
        or auxiliary.get("sha256") != FROZEN_LOSS_SHA256
        or frozen.get("max_angle_radians") != v1_trainer.MAX_ANGLE_RADIANS
        or frozen.get("zero_final_projection") is not True
    ):
        raise ValueError("response/listwise V2 frozen implementation differs")
    repository = Path(__file__).resolve().parents[2]
    frozen_records = {
        "V1 trainer": frozen["trainer"],
        "V1 model": frozen["model"],
        "source-global response loss": auxiliary,
    }
    for label, record in frozen_records.items():
        implementation = (repository / str(record["path"])).resolve()
        if sha256_file(implementation) != str(record["sha256"]):
            raise ValueError(f"{label} changed after response/listwise preregistration")
    return prereg


def load_fit_text_bank(path: str | Path, *, expected_sha256: str) -> FitTextBank:
    payload, digest, source = load_torch_mapping(
        path,
        expected_sha256=expected_sha256,
        map_location="cpu",
        label="target-blind generic fit text bank",
    )
    embeddings = payload.get("embeddings")
    queries = payload.get("queries")
    query_count = len(queries) if isinstance(queries, list) else -1
    if (
        payload.get("split") != "fit"
        or query_count != 806
        or payload.get("benchmark_vocabulary_opened") is not False
        or payload.get("uses_benchmark_vocabulary_for_construction") is not False
        or not torch.is_tensor(embeddings)
        or embeddings.dtype != torch.float32
        or embeddings.ndim != 2
        or embeddings.shape[0] != 806
        or not bool(torch.isfinite(embeddings).all())
    ):
        raise ValueError("generic fit text bank contract differs")
    observed_tensor_sha = tensor_sha256(embeddings)
    if observed_tensor_sha != payload.get("embedding_tensor_sha256"):
        raise ValueError("generic fit text embedding tensor authority differs")
    return FitTextBank(
        embeddings=embeddings.detach().cpu().contiguous(),
        record={"path": str(source), "sha256": digest},
        embedding_tensor_sha256=observed_tensor_sha,
    )


def _load_response_authority(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_content_authority_sha256: str,
    expected_scene_id: str,
) -> FrozenSourceResponseAuthority:
    raw, _, _ = load_torch_mapping(
        path,
        expected_sha256=expected_file_sha256,
        map_location="cpu",
        label=f"{expected_scene_id} hard-negative authority preflight",
    )
    payload = validate_negative_authority(raw)
    inputs = payload["input_authority"]
    accepted = inputs["accepted_v2"]
    teacher = inputs["official_multiview_siglip2_teacher"]
    fit = inputs["fit_text_bank"]
    return load_frozen_source_response_authority(
        path,
        expected_file_sha256=expected_file_sha256,
        expected_content_authority_sha256=expected_content_authority_sha256,
        expected_scene_id=expected_scene_id,
        expected_accepted_v2_file_sha256=accepted["sha256"],
        expected_teacher_file_sha256=teacher["sha256"],
        expected_teacher_pair_descriptors_sha256=teacher["channel_sha256"][
            "pair_descriptors"
        ],
        expected_fit_text_bank_file_sha256=fit["sha256"],
    )


def bind_response_scene(
    base: v1_trainer.SceneBinding,
    value: Mapping[str, Any],
    *,
    fit_text_bank: FitTextBank,
) -> ResponseSceneBinding:
    if set(value) != {
        "scene_id",
        "training_shard",
        "adaptive_context",
        "hard_negative_authority",
        "hard_negative_content_authority_sha256",
    }:
        raise ValueError("response/listwise scene execution record differs")
    if str(value["scene_id"]) != base.scene_id:
        raise ValueError("response/listwise scene identity differs")
    if _record(value["training_shard"], label="training shard") != base.training_shard:
        raise ValueError("response/listwise training shard record differs")
    if (
        _record(value["adaptive_context"], label="adaptive context")
        != base.adaptive_context
    ):
        raise ValueError("response/listwise adaptive-context record differs")
    negative = _record(
        value["hard_negative_authority"], label="hard-negative authority"
    )
    content = _sha(
        value["hard_negative_content_authority_sha256"],
        label="hard-negative content authority",
    )
    authority = _load_response_authority(
        negative["path"],
        expected_file_sha256=negative["sha256"],
        expected_content_authority_sha256=content,
        expected_scene_id=base.scene_id,
    )
    if authority.fit_text_bank_file_sha256 != fit_text_bank.record["sha256"]:
        raise ValueError("scene hard-negative authority binds another fit text bank")
    if (
        authority.fit_text_embedding_tensor_sha256
        != fit_text_bank.embedding_tensor_sha256
    ):
        raise ValueError("scene hard-negative authority binds another fit text tensor")
    return ResponseSceneBinding(base, negative, content, authority)


def load_bound_scene(binding: ResponseSceneBinding) -> dict[str, Any]:
    scene = v1_trainer.load_scene(binding.base)
    payload = binding.response_authority.payload
    inputs = payload["input_authority"]
    accepted_channels = inputs["accepted_v2"]["channel_sha256"]
    teacher_channels = inputs["official_multiview_siglip2_teacher"]["channel_sha256"]
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
            payload["canonical_region_indices_sha256"],
        ),
    }
    for label, (observed, expected) in checks.items():
        if observed != expected:
            raise ValueError(f"scene {label} differs from hard-negative authority")
    return scene


def complete_scene_objective(
    model: SurfaceRegionAcceptedV2TypedContextResidualV1,
    scene: Mapping[str, Any],
    normalization: Mapping[str, Any],
    fit_text_embeddings: torch.Tensor,
    authority: FrozenSourceResponseAuthority,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int | bool]]:
    """Forward every canonical row and combine frozen V1 and V2 losses."""

    declared, effective_ood, active = v1_trainer._routing(scene, normalization)
    row_count = len(scene["region_row_ids"])
    if active.shape != (row_count,) or int(active.sum()) < 2:
        raise ValueError("complete source scene requires at least two active rows")
    base = scene["accepted_v2_e0"].to(device)
    result = model.forward_with_diagnostics(
        base,
        scene["pooled_context_radio_direction"].to(device),
        scene["raw_full_scalar_summary"].to(device),
        scene["typed_context_statistics"].to(device),
        active_mask=declared.to(device),
        ood_mask=effective_ood.to(device),
    )
    fallback = ~active.to(device)
    fallback_equal = torch.equal(result.semantic_descriptor[fallback], base[fallback])
    if not fallback_equal:
        raise RuntimeError("inactive/OOD rows are not bitwise AcceptedV2 e0")
    active_rows = torch.where(active)[0]
    teachers, teacher_mask = v1_trainer.gather_sparse_teacher_batch(scene, active_rows)
    base_loss, base_metrics = v1_trainer.typed_context_objective(
        result.semantic_descriptor[active.to(device)],
        teachers.to(device),
        teacher_mask.to(device),
        scene["typed_context_statistics"][active_rows].to(device),
        boundary_threshold=float(normalization["source_boundary_score_median"]),
    )
    total, response_metrics = source_global_response_listwise_loss(
        base_loss,
        result.semantic_descriptor,
        scene["official_multiview_siglip2_teacher_pair_descriptors"].to(device),
        scene["official_multiview_siglip2_teacher_pair_region_indices"],
        fit_text_embeddings,
        authority.payload["canonical_region_indices"],
        authority,
        accepted_v2_file_sha256=authority.accepted_v2_file_sha256,
        teacher_file_sha256=authority.teacher_file_sha256,
        teacher_pair_descriptors_sha256=(authority.teacher_pair_descriptors_sha256),
        fit_text_bank_file_sha256=authority.fit_text_bank_file_sha256,
        config=recommended_v2_config(),
    )
    if not bool(torch.isfinite(total.detach())):
        raise RuntimeError("complete-scene response/listwise objective is nonfinite")
    return total, {
        "base_objective": base_loss,
        **{f"base_{name}": value for name, value in base_metrics.items()},
        **{f"response_{name}": value for name, value in response_metrics.items()},
        "complete_canonical_rows": row_count,
        "active_rows": int(active.sum()),
        "fallback_bitwise_accepted_v2_e0": fallback_equal,
    }


def _float_metrics(metrics: Mapping[str, object]) -> dict[str, float | int | bool]:
    result: dict[str, float | int | bool] = {}
    for name, value in metrics.items():
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(f"metric {name} is not scalar")
            result[name] = float(value.detach().cpu())
        elif isinstance(value, bool):
            result[name] = value
        elif isinstance(value, int):
            result[name] = value
        else:
            result[name] = float(value)  # type: ignore[arg-type]
    return result


def train_one_epoch(
    model: SurfaceRegionAcceptedV2TypedContextResidualV1,
    optimizer: torch.optim.Optimizer,
    bindings: Sequence[ResponseSceneBinding],
    normalization: Mapping[str, Any],
    fit_text_embeddings: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    per_scene: dict[str, dict[str, float | int | bool]] = {}
    for binding in bindings:
        scene = load_bound_scene(binding)
        total, metrics = complete_scene_objective(
            model,
            scene,
            normalization,
            fit_text_embeddings,
            binding.response_authority,
            device,
        )
        (total / len(bindings)).backward()
        per_scene[binding.scene_id] = {
            "combined_objective": float(total.detach().cpu()),
            **_float_metrics(metrics),
        }
        del scene, total, metrics
    torch.nn.utils.clip_grad_norm_(
        tuple(model.parameters()),
        v1_trainer.MAX_GRADIENT_NORM,
        error_if_nonfinite=True,
    )
    optimizer.step()
    return {
        "scene_count": len(bindings),
        "equal_scene_weight": 1.0 / len(bindings),
        "scene_forward_unit": "complete_canonical_rows",
        "per_scene": per_scene,
        "macro_combined_objective": sum(
            float(value["combined_objective"]) for value in per_scene.values()
        )
        / len(per_scene),
        "macro_response_auxiliary_loss": sum(
            float(value["response_auxiliary_loss"]) for value in per_scene.values()
        )
        / len(per_scene),
    }


@torch.no_grad()
def evaluate_response(
    model: SurfaceRegionAcceptedV2TypedContextResidualV1,
    bindings: Sequence[ResponseSceneBinding],
    normalization: Mapping[str, Any],
    fit_text_embeddings: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    per_scene: dict[str, dict[str, float | int | bool]] = {}
    for binding in bindings:
        scene = load_bound_scene(binding)
        total, metrics = complete_scene_objective(
            model,
            scene,
            normalization,
            fit_text_embeddings,
            binding.response_authority,
            device,
        )
        per_scene[binding.scene_id] = {
            "combined_objective": float(total.detach().cpu()),
            **_float_metrics(metrics),
        }
        del scene, total, metrics
    return {
        "scene_count": len(bindings),
        "scene_macro_auxiliary_loss": sum(
            float(value["response_auxiliary_loss"]) for value in per_scene.values()
        )
        / len(per_scene),
        "all_scenes_complete_canonical_rows": all(
            int(value["complete_canonical_rows"]) > 0 for value in per_scene.values()
        ),
        "all_scenes_bitwise_fallback": all(
            bool(value["fallback_bitwise_accepted_v2_e0"])
            for value in per_scene.values()
        ),
        "per_scene": per_scene,
    }


@torch.no_grad()
def evaluate(
    model: SurfaceRegionAcceptedV2TypedContextResidualV1,
    bindings: Sequence[ResponseSceneBinding],
    normalization: Mapping[str, Any],
    fit_text_embeddings: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    v1_validation = v1_trainer.evaluate(
        model, [item.base for item in bindings], normalization, device
    )
    response = evaluate_response(
        model, bindings, normalization, fit_text_embeddings, device
    )
    passed = (
        v1_validation["non_regression_passed"]
        and response["all_scenes_complete_canonical_rows"]
        and response["all_scenes_bitwise_fallback"]
    )
    return {
        "v1_non_regression": v1_validation,
        "response_listwise": response,
        "selection_eligible": bool(passed),
        "validation_no_grad": not torch.is_grad_enabled(),
    }


def select_best_epoch(history: Sequence[Mapping[str, Any]]) -> int:
    if not history or [int(item.get("epoch", -1)) for item in history] != list(
        range(len(history))
    ):
        raise ValueError(
            "response/listwise V2 history must be contiguous from epoch zero"
        )
    eligible = [
        item
        for item in history
        if item.get("validation", {}).get("selection_eligible") is True
    ]
    if not eligible:
        raise RuntimeError("response/listwise V2 has no non-regressing checkpoint")

    def rank(item: Mapping[str, Any]) -> tuple[float, float, float, float, int]:
        validation = item["validation"]
        response = validation["response_listwise"]
        candidate = validation["v1_non_regression"]["candidate"]
        return (
            -float(response["scene_macro_auxiliary_loss"]),
            float(candidate["mean_all_view_cosine"]),
            float(candidate["p05_row_mean_all_view_cosine"]),
            float(candidate["relation_fidelity"]),
            -int(item["epoch"]),
        )

    return int(max(eligible, key=rank)["epoch"])


def _validate_execution_authority(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("response/listwise execution authority must be a mapping")
    authority = dict(value)
    required = {
        "schema",
        "schema_version",
        "status",
        "preregistration",
        "implementation",
        "fit_text_bank",
        "cohort_authority",
        "source_state_manifest",
        "teacher_manifest",
        "benchmark_exclusion_manifest",
        "source_train",
        "source_validation",
        "full_training_authorized",
        "benchmark_execution_authorized",
        "source_access",
    }
    if set(authority) != required:
        raise ValueError("response/listwise execution authority fields differ")
    if (
        authority.get("schema") != EXECUTION_AUTHORITY_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("status") != "authorized_after_complete_32_scene_preflight"
        or authority.get("full_training_authorized") is not True
        or authority.get("benchmark_execution_authorized") is not False
        or authority.get("source_access") != source_access()
    ):
        raise ValueError("response/listwise execution authority header differs")
    for name in (
        "preregistration",
        "implementation",
        "fit_text_bank",
        "cohort_authority",
        "source_state_manifest",
        "teacher_manifest",
        "benchmark_exclusion_manifest",
    ):
        authority[name] = _record(authority[name], label=name)
    for split, expected in (
        ("source_train", base_trainer.TRAIN_SCENE_COUNT),
        ("source_validation", base_trainer.VALIDATION_SCENE_COUNT),
    ):
        records = authority[split]
        if not isinstance(records, list) or len(records) != expected:
            raise ValueError(f"{split} requires exactly {expected} response scenes")
        scenes = [
            str(item.get("scene_id", ""))
            for item in records
            if isinstance(item, Mapping)
        ]
        if (
            len(scenes) != expected
            or scenes != sorted(scenes)
            or len(set(scenes)) != expected
        ):
            raise ValueError(f"{split} response scene order differs")
    return authority


def _validated_record_path(record: Mapping[str, str], *, label: str) -> Path:
    return validate_file_record(record, label=label)


def _prepare_training_authority(path: str | Path, *, expected_sha256: str) -> tuple[
    dict[str, Any],
    list[ResponseSceneBinding],
    list[ResponseSceneBinding],
    FitTextBank,
    tuple[Any, ...],
    dict[str, Any],
]:
    value, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="response/listwise V2 execution authority",
    )
    authority = _validate_execution_authority(value)
    prereg_path = _validated_record_path(
        authority["preregistration"], label="response/listwise preregistration"
    )
    prereg, _, _ = load_json_object(
        prereg_path,
        expected_sha256=authority["preregistration"]["sha256"],
        label="response/listwise V2 preregistration",
    )
    _validate_preregistration(prereg)
    implementation = _validated_record_path(
        authority["implementation"], label="response/listwise V2 implementation"
    )
    if implementation != Path(__file__).resolve():
        raise ValueError("execution authority binds another V2 implementation")
    fit_path = _validated_record_path(
        authority["fit_text_bank"], label="generic fit text bank"
    )
    fit = load_fit_text_bank(
        fit_path, expected_sha256=authority["fit_text_bank"]["sha256"]
    )

    manifest_args = argparse.Namespace(
        cohort_authority=_validated_record_path(
            authority["cohort_authority"], label="clean cohort authority"
        ),
        expected_cohort_authority_sha256=authority["cohort_authority"]["sha256"],
        source_state_manifest=_validated_record_path(
            authority["source_state_manifest"], label="source-state manifest"
        ),
        expected_source_state_manifest_sha256=authority["source_state_manifest"][
            "sha256"
        ],
        teacher_manifest=_validated_record_path(
            authority["teacher_manifest"], label="teacher manifest"
        ),
        expected_teacher_manifest_sha256=authority["teacher_manifest"]["sha256"],
        benchmark_exclusion_manifest=_validated_record_path(
            authority["benchmark_exclusion_manifest"],
            label="benchmark exclusion manifest",
        ),
        expected_benchmark_exclusion_manifest_sha256=authority[
            "benchmark_exclusion_manifest"
        ]["sha256"],
    )
    manifests = v1_trainer._load_manifests(manifest_args)

    def columns(split: str, key: str) -> list[str]:
        return [str(item[key]["path"]) for item in authority[split]]

    def shas(split: str, key: str) -> list[str]:
        return [str(item[key]["sha256"]) for item in authority[split]]

    train_base, validation_base, cohort = v1_trainer.preflight_bindings(
        train_shards=columns("source_train", "training_shard"),
        train_shas=shas("source_train", "training_shard"),
        train_contexts=columns("source_train", "adaptive_context"),
        train_context_shas=shas("source_train", "adaptive_context"),
        validation_shards=columns("source_validation", "training_shard"),
        validation_shas=shas("source_validation", "training_shard"),
        validation_contexts=columns("source_validation", "adaptive_context"),
        validation_context_shas=shas("source_validation", "adaptive_context"),
        cohort_and_manifests=manifests,
    )

    def bind_all(
        bases: Sequence[v1_trainer.SceneBinding], split: str
    ) -> list[ResponseSceneBinding]:
        by_scene = {str(item["scene_id"]): item for item in authority[split]}
        return [
            bind_response_scene(base, by_scene[base.scene_id], fit_text_bank=fit)
            for base in bases
        ]

    authority["verified_path"] = str(source)
    authority["verified_sha256"] = digest
    return (
        authority,
        bind_all(train_base, "source_train"),
        bind_all(validation_base, "source_validation"),
        fit,
        manifests,
        cohort,
    )


def _state_copy(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().contiguous().clone()
        for name, value in model.state_dict().items()
    }


def input_records(bindings: Sequence[ResponseSceneBinding]) -> list[dict[str, Any]]:
    return [
        {
            "scene_id": item.scene_id,
            "training_shard": dict(item.base.training_shard),
            "adaptive_context": dict(item.base.adaptive_context),
            "hard_negative_authority": dict(item.hard_negative),
            "hard_negative_content_authority_sha256": (
                item.hard_negative_content_authority_sha256
            ),
        }
        for item in bindings
    ]


def train(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    normalization_output = output.with_suffix(output.suffix + ".normalization.pt")
    certificate_output = output.with_suffix(output.suffix + ".certificate.json")
    report_output = output.with_suffix(output.suffix + ".json")
    existing = [
        item
        for item in (output, normalization_output, certificate_output, report_output)
        if item.exists() or item.is_symlink()
    ]
    if existing:
        raise FileExistsError(
            "response/listwise V2 first-writer outputs already exist: "
            + ", ".join(str(item) for item in existing)
        )
    (
        execution,
        train_bindings,
        validation_bindings,
        fit,
        manifests,
        cohort,
    ) = _prepare_training_authority(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
    )
    normalization = v1_trainer.fit_normalization(
        [item.base for item in train_bindings],
        source_state_cohort_authority_sha256=cohort[
            "source_state_cohort_authority_sha256"
        ],
    )
    normalization_content_sha = typed_context_normalization_authority_sha256(
        normalization
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
    fit_device = fit.embeddings.to(device)
    epoch_zero = evaluate(model, validation_bindings, normalization, fit_device, device)
    if not epoch_zero["selection_eligible"]:
        raise RuntimeError("response/listwise V2 epoch-zero identity did not pass")
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
    epochs_without_improvement = 0
    for epoch in range(1, v1_trainer.EPOCHS + 1):
        training = train_one_epoch(
            model,
            optimizer,
            train_bindings,
            normalization,
            fit_device,
            device,
        )
        validation = evaluate(
            model, validation_bindings, normalization, fit_device, device
        )
        record = {
            "epoch": epoch,
            "training": training,
            "validation": validation,
            "model_state_dict_sha256": v1_trainer._state_sha(model),
        }
        history.append(record)
        selected = select_best_epoch(history)
        if selected == epoch:
            best_epoch = epoch
            best_state = _state_copy(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(json.dumps(record, sort_keys=True), flush=True)
        if epochs_without_improvement >= v1_trainer.PATIENCE:
            break
    selected_epoch = select_best_epoch(history)
    if selected_epoch != best_epoch:
        raise RuntimeError("response/listwise V2 selected/best epoch differs")
    model.load_state_dict(best_state, strict=True)
    selected_validation = evaluate(
        model, validation_bindings, normalization, fit_device, device
    )
    if selected_validation != history[selected_epoch]["validation"]:
        raise RuntimeError("response/listwise V2 restored metrics differ")
    model.cpu()

    normalization_path = write_torch_noclobber(normalization_output, normalization)
    normalization_sha = sha256_file(normalization_path)
    (
        cohort_authority,
        cohort_file,
        _source,
        source_file,
        _teacher,
        teacher_file,
        _exclusion,
        exclusion_file,
    ) = manifests
    certificate = build_typed_context_training_certificate(
        training_contract=training_contract(),
        model=model,
        normalization_authority_file=file_record(normalization_path),
        cohort_authority={
            "file": cohort_file,
            "authority_sha256": cohort_authority["authority_sha256"],
        },
        external_manifests={
            "source_state": source_file,
            "teacher": teacher_file,
            "benchmark_exclusion": exclusion_file,
        },
        input_records_by_split={
            "source_train": v1_trainer.input_records(
                [item.base for item in train_bindings]
            ),
            "source_validation": v1_trainer.input_records(
                [item.base for item in validation_bindings]
            ),
        },
        selected_epoch=selected_epoch,
        selected_validation=selected_validation,
    )
    certificate_path = write_frozen_json(certificate_output, certificate)
    certificate_sha = sha256_file(certificate_path)
    checkpoint_path, checkpoint_sha = write_typed_context_checkpoint(
        output,
        model,
        normalization_authority=normalization,
        normalization_file_sha256=normalization_sha,
        certificate=certificate,
        certificate_file_sha256=certificate_sha,
    )
    report = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "training_contract": training_contract(),
        "training_contract_sha256": TRAINING_CONTRACT_SHA256,
        "execution_authority": {
            "path": execution["verified_path"],
            "sha256": execution["verified_sha256"],
        },
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha},
        "normalization_authority": file_record(normalization_path),
        "normalization_content_authority_sha256": normalization_content_sha,
        "training_certificate": file_record(certificate_path),
        "fit_text_bank": dict(fit.record),
        "selected_epoch": selected_epoch,
        "automatic_fallback_to_epoch_zero": selected_epoch == 0,
        "selected_validation": selected_validation,
        "history": history,
        "input_records_by_split": {
            "source_train": input_records(train_bindings),
            "source_validation": input_records(validation_bindings),
        },
        "cohort": cohort,
        "source_access": source_access(),
    }
    write_frozen_json(report_output, report)
    return report


@torch.inference_mode()
def dry_source_loss(args: argparse.Namespace) -> dict[str, Any]:
    prereg, prereg_sha, prereg_path = load_json_object(
        args.preregistration,
        expected_sha256=args.expected_preregistration_sha256,
        label="response/listwise V2 preregistration",
    )
    prereg = _validate_preregistration(prereg)
    permitted = prereg["permitted_preflight"]["scene0001_source_only_dry_loss"]
    declared = {
        "accepted_v2": {
            "path": str(Path(args.accepted_v2).expanduser().resolve()),
            "sha256": str(args.expected_accepted_v2_sha256),
        },
        "official_teacher": {
            "path": str(Path(args.official_teacher).expanduser().resolve()),
            "sha256": str(args.expected_official_teacher_sha256),
        },
        "hard_negative_authority": {
            "path": str(Path(args.hard_negative_authority).expanduser().resolve()),
            "sha256": str(args.expected_hard_negative_authority_sha256),
            "content_authority_sha256": str(
                args.expected_hard_negative_content_authority_sha256
            ),
        },
    }
    for name, record in declared.items():
        expected = permitted[name]
        for key, value in record.items():
            differs = (
                Path(value).resolve() != Path(str(expected[key])).resolve()
                if key == "path"
                else value != expected[key]
            )
            if differs:
                raise ValueError(f"dry-loss {name} {key} differs from preregistration")
    accepted, accepted_sha, accepted_path = load_torch_mapping(
        args.accepted_v2,
        expected_sha256=args.expected_accepted_v2_sha256,
        map_location="cpu",
        label="dry-loss AcceptedV2 authority",
    )
    teacher, teacher_sha, teacher_path = load_torch_mapping(
        args.official_teacher,
        expected_sha256=args.expected_official_teacher_sha256,
        map_location="cpu",
        label="dry-loss official multiview teacher",
    )
    if (
        accepted.get("scene_id") != "scene0001_00"
        or teacher.get("scene_id") != "scene0001_00"
    ):
        raise ValueError("dry-loss source scene differs")
    authority = _load_response_authority(
        args.hard_negative_authority,
        expected_file_sha256=args.expected_hard_negative_authority_sha256,
        expected_content_authority_sha256=(
            args.expected_hard_negative_content_authority_sha256
        ),
        expected_scene_id="scene0001_00",
    )
    fit = load_fit_text_bank(
        args.fit_text_bank, expected_sha256=args.expected_fit_text_bank_sha256
    )
    prereg_fit = prereg["generic_fit_text_bank"]
    if (
        fit.record["sha256"] != prereg_fit["sha256"]
        or Path(fit.record["path"]).resolve() != Path(prereg_fit["path"]).resolve()
    ):
        raise ValueError("dry-loss fit text bank differs from preregistration")
    accepted_rows = accepted["accepted_v2_e0"].to(args.device)
    base_loss = accepted_rows.sum() * 0.0
    total, metrics = source_global_response_listwise_loss(
        base_loss,
        accepted_rows,
        teacher["pair_descriptors"].to(args.device),
        teacher["pair_region_indices"],
        fit.embeddings.to(args.device),
        accepted["canonical_region_indices"],
        authority,
        accepted_v2_file_sha256=accepted_sha,
        teacher_file_sha256=teacher_sha,
        teacher_pair_descriptors_sha256=teacher["channel_sha256"]["pair_descriptors"],
        fit_text_bank_file_sha256=fit.record["sha256"],
        config=recommended_v2_config(),
    )
    report = {
        "schema": "radio_gs.surface_region_response_listwise_v2_source_dry_loss.v1",
        "schema_version": 1,
        "status": "source_only_dry_loss_complete_no_optimizer_no_parameter_update",
        "scene_id": "scene0001_00",
        "loss": float(total.cpu()),
        "metrics": _float_metrics(metrics),
        "complete_canonical_rows": int(accepted_rows.shape[0]),
        "authorities": {
            "implementation": file_record(Path(__file__).resolve()),
            "preregistration": {"path": str(prereg_path), "sha256": prereg_sha},
            "accepted_v2": {"path": str(accepted_path), "sha256": accepted_sha},
            "official_teacher": {"path": str(teacher_path), "sha256": teacher_sha},
            "hard_negative": {
                "path": str(Path(args.hard_negative_authority).resolve()),
                "sha256": authority.file_sha256,
                "content_authority_sha256": authority.content_authority_sha256,
            },
            "fit_text_bank": dict(fit.record),
        },
        "source_access": source_access(),
        "optimizer_constructed": False,
        "parameter_update_performed": False,
        "benchmark_opened": False,
    }
    write_frozen_json(args.output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    dry = commands.add_parser("dry-loss")
    dry.add_argument("--preregistration", required=True)
    dry.add_argument("--expected-preregistration-sha256", required=True)
    dry.add_argument("--accepted-v2", required=True)
    dry.add_argument("--expected-accepted-v2-sha256", required=True)
    dry.add_argument("--official-teacher", required=True)
    dry.add_argument("--expected-official-teacher-sha256", required=True)
    dry.add_argument("--hard-negative-authority", required=True)
    dry.add_argument("--expected-hard-negative-authority-sha256", required=True)
    dry.add_argument("--expected-hard-negative-content-authority-sha256", required=True)
    dry.add_argument("--fit-text-bank", required=True)
    dry.add_argument("--expected-fit-text-bank-sha256", required=True)
    dry.add_argument("--output", required=True)
    dry.add_argument("--device", default="cpu")

    full = commands.add_parser("train")
    full.add_argument("--execution-authority", required=True)
    full.add_argument("--expected-execution-authority-sha256", required=True)
    full.add_argument("--output", required=True)
    full.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = dry_source_loss(args) if args.command == "dry-loss" else train(args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
