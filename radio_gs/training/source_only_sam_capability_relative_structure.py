"""Official-SAM relative supervision in frozen capability-relation space.

This v3 objective changes no field architecture.  It decodes the same one
RADIO vector per primitive, removes radial gradients with the established
tangent-only reparameterization, applies the frozen official DINOv3 and SAM3
adaptors, and ranks the geometric mean of their cosine affinities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from radio_gs.training.source_only_multiteacher import source_direction_only_radio
from radio_gs.training.source_only_sam_relative_structure import (
    SAM_RELATIVE_GUARD_BATCH_SIZE,
    SAM_RELATIVE_MARGIN,
    SAM_RELATIVE_TRIPLET_BATCH_SIZE,
    SourceOnlySamRelativeBundle,
    SourceSamRelativeTeacher,
    load_source_only_sam_relative_bundle,
)
from radio_gs.training.source_only_sam_structure import SAM_STRUCTURE_WEIGHT
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_json_object,
    validate_file_record,
)


SOURCE_ONLY_SAM_CAPABILITY_RELATIVE_SCHEMA = (
    "radio_gs.source_only_sam_capability_relative_structure.v3"
)
SOURCE_ONLY_SAM_CAPABILITY_RELATIVE_SCHEMA_VERSION = 3


def source_only_sam_capability_relative_contract() -> dict[str, Any]:
    return {
        "schema": SOURCE_ONLY_SAM_CAPABILITY_RELATIVE_SCHEMA,
        "schema_version": SOURCE_ONLY_SAM_CAPABILITY_RELATIVE_SCHEMA_VERSION,
        "persistent_semantic_feature": "canonical_radio_only",
        "mapping_teacher": {
            "model": "official_sam",
            "input": "legal_source_rgb_only",
            "triplet_axis": "frozen_source_only_sam_relative_structure_v2",
            "teacher_payload_persisted": False,
        },
        "relation_space": {
            "student": "tangent_only_canonical_radio",
            "views": ["official_dino_v3_adaptor", "official_sam3_adaptor"],
            "adaptors_frozen": True,
            "per_view_affinity": "clamp((1+cosine)/2,0,1)",
            "fusion": "sqrt(dino_affinity*sam3_affinity)",
            "scene_or_query_parameters": False,
        },
        "loss": {
            "ranking": (
                "relu(margin+capability_relation_anchor_negative-"
                "capability_relation_anchor_positive)"
            ),
            "ranking_margin": SAM_RELATIVE_MARGIN,
            "already_ordered_relation": "zero_ranking_gradient",
            "same_guard": (
                "relu(control_same_capability_relation-current_same_capability_relation)"
            ),
            "separate_guard": (
                "relu(current_separate_capability_relation-control_separate_capability_relation)"
            ),
            "guard_class_balance": "equal_same_and_separate_mass",
            "ranking_guard_balance": "equal_half_mass",
            "edge_weight": "unchanged_v2_normalized_log1p_vote_mass_times_dominance",
            "weight": SAM_STRUCTURE_WEIGHT,
            "triplet_batch_size": SAM_RELATIVE_TRIPLET_BATCH_SIZE,
            "guard_batch_size": SAM_RELATIVE_GUARD_BATCH_SIZE,
            "student_gauge_gradient": (
                "zero_by_detached_norm_direction_reparameterization_v1"
            ),
        },
        "query_time": {
            "source_rgb": False,
            "target_rgb": False,
            "sam_decoder": False,
            "sam_cache_or_mask": False,
            "teacher_relation_or_triplet": False,
        },
        "forbidden_inputs": [
            "benchmark_query_or_text_bank",
            "benchmark_target_or_evaluation_rgb",
            "benchmark_ground_truth",
            "benchmark_label_or_mask",
            "benchmark_metric_or_prediction",
        ],
    }


SOURCE_ONLY_SAM_CAPABILITY_RELATIVE_CONTRACT_SHA256 = canonical_json_sha256(
    source_only_sam_capability_relative_contract()
)


def validate_source_only_sam_capability_relative_manifest(
    value: object,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("source-only SAM capability-relative manifest must be a mapping")
    payload = dict(value)
    required = {
        "schema",
        "schema_version",
        "contract",
        "contract_sha256",
        "status",
        "scene_id",
        "base_relative_manifest",
        "official_adaptor_checkpoint",
        "loss",
        "source_gates",
        "access_audit",
        "execution",
    }
    contract = source_only_sam_capability_relative_contract()
    if (
        set(payload) != required
        or payload.get("schema") != SOURCE_ONLY_SAM_CAPABILITY_RELATIVE_SCHEMA
        or payload.get("schema_version")
        != SOURCE_ONLY_SAM_CAPABILITY_RELATIVE_SCHEMA_VERSION
        or payload.get("contract") != contract
        or payload.get("contract_sha256")
        != SOURCE_ONLY_SAM_CAPABILITY_RELATIVE_CONTRACT_SHA256
        or payload.get("status") != "preregistered_training_not_started"
        or not str(payload.get("scene_id", ""))
        or payload.get("loss") != contract["loss"]
    ):
        raise ValueError("source-only SAM capability-relative contract differs")
    for name in ("base_relative_manifest", "official_adaptor_checkpoint"):
        record = payload.get(name)
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise ValueError(f"source-only SAM capability-relative {name} differs")
        validate_file_record(record, label=f"source-only SAM capability-relative {name}")
    expected_access = {
        "source_rgb_opened_during_mapping": True,
        "official_sam_opened_during_mapping": True,
        "official_dino_and_sam_adaptors_opened_during_training": True,
        "query_time_source_rgb_opened": False,
        "query_time_target_rgb_opened": False,
        "benchmark_query_or_text_opened": False,
        "benchmark_target_or_evaluation_rgb_opened": False,
        "benchmark_ground_truth_opened": False,
        "benchmark_labels_or_masks_opened": False,
        "benchmark_metrics_or_predictions_opened": False,
    }
    if payload.get("access_audit") != expected_access:
        raise ValueError("source-only SAM capability-relative access audit differs")
    expected_gates = {
        "radio_reconstruction_no_regression": {
            "mean_cosine_max_regression": 0.005,
            "p05_cosine_max_regression": 0.01,
        },
        "official_capability_no_regression": {
            "mean_cosine_max_regression": 0.005,
            "p05_cosine_max_regression": 0.01,
        },
        "global_capability_same_relation_non_decrease": True,
        "global_capability_separate_relation_non_increase": True,
        "global_capability_relation_gap_strict_improvement": True,
        "capability_triplet_gap_strict_improvement": True,
        "capability_triplet_violation_strict_decrease": True,
        "six_task_benchmark_gate": (
            "closed_after_source_gate_checkpoint_seal_no_benchmark_in_this_stage"
        ),
    }
    if payload.get("source_gates") != expected_gates:
        raise ValueError("source-only SAM capability-relative source gates differ")
    if payload.get("execution") != {
        "gpu_started": False,
        "per_scene_or_per_task_tuning": False,
        "output_no_clobber": True,
        "teacher_payload_saved_in_checkpoint": False,
        "v2_candidate_checkpoint_used": False,
    }:
        raise ValueError("source-only SAM capability-relative execution differs")
    return payload


@dataclass(frozen=True)
class SourceSamCapabilityRelativeTeacher:
    relative_teacher: SourceSamRelativeTeacher
    pair_control_relation: torch.Tensor
    positive_control_relation: torch.Tensor
    negative_control_relation: torch.Tensor

    @property
    def pair_teacher(self):
        return self.relative_teacher.pair_teacher

    @property
    def triplet_index(self) -> torch.Tensor:
        return self.relative_teacher.triplet_index

    @property
    def triplet_weight(self) -> torch.Tensor:
        return self.relative_teacher.triplet_weight

    @property
    def scale_bin(self) -> torch.Tensor:
        return self.relative_teacher.scale_bin

    @property
    def num_triplets(self) -> int:
        return self.relative_teacher.num_triplets

    def training_subset(
        self, *, pair_keep: torch.Tensor, triplet_keep: torch.Tensor
    ) -> "SourceSamCapabilityRelativeTeacher":
        pair_selected = torch.as_tensor(pair_keep).bool().cpu().reshape(-1)
        triplet_selected = torch.as_tensor(triplet_keep).bool().cpu().reshape(-1)
        subset = self.relative_teacher.training_subset(
            pair_keep=pair_selected, triplet_keep=triplet_selected
        )
        return SourceSamCapabilityRelativeTeacher(
            relative_teacher=subset,
            pair_control_relation=self.pair_control_relation[pair_selected].contiguous(),
            positive_control_relation=self.positive_control_relation[
                triplet_selected
            ].contiguous(),
            negative_control_relation=self.negative_control_relation[
                triplet_selected
            ].contiguous(),
        )


@dataclass(frozen=True)
class SourceOnlySamCapabilityRelativeBundle:
    manifest: dict[str, Any]
    manifest_source: Path
    manifest_sha256: str
    base: SourceOnlySamRelativeBundle
    teacher: SourceSamCapabilityRelativeTeacher


def _check_frozen_views(official_views: torch.nn.Module) -> None:
    parameters = list(official_views.parameters())
    if any(parameter.requires_grad for parameter in parameters):
        raise ValueError("official capability adaptors must be frozen")


def capability_geometric_mean_relation(
    radio: torch.Tensor,
    edge: torch.Tensor,
    *,
    official_views: torch.nn.Module,
) -> torch.Tensor:
    values = torch.as_tensor(radio)
    index = torch.as_tensor(edge).long().to(values.device)
    if (
        values.ndim != 2
        or values.shape[1] != 1280
        or index.ndim != 2
        or index.shape[0] != 2
        or bool((index < 0).any())
        or bool((index >= values.shape[0]).any())
    ):
        raise ValueError("capability relation RADIO/edge axes differ")
    _check_frozen_views(official_views)
    dino = official_views.project_dino_primitives(values)
    sam = official_views.project_sam3_primitives(values)
    dino_affinity = (
        0.5 * (1.0 + (dino[index[0]] * dino[index[1]]).sum(dim=-1))
    ).clamp(0.0, 1.0)
    sam_affinity = (
        0.5 * (1.0 + (sam[index[0]] * sam[index[1]]).sum(dim=-1))
    ).clamp(0.0, 1.0)
    return torch.sqrt((dino_affinity * sam_affinity).clamp_min(0.0))


@torch.no_grad()
def _edge_capability_relation(
    field: torch.nn.Module,
    official_views: torch.nn.Module,
    edge: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    values: list[torch.Tensor] = []
    device = field.local_codes.device
    for start in range(0, edge.shape[1], int(batch_size)):
        part = edge[:, start : start + int(batch_size)]
        unique, inverse = torch.unique(part.reshape(-1), sorted=True, return_inverse=True)
        radio = field.radio_features(unique.to(device)).float()
        local = inverse.reshape_as(part).to(device)
        values.append(
            capability_geometric_mean_relation(
                radio, local, official_views=official_views
            ).cpu()
        )
    return torch.cat(values)


def load_source_only_sam_capability_relative_bundle(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_xyz: torch.Tensor,
    expected_valid: torch.Tensor,
    expected_canonical_radio_cache_sha256: str,
    expected_field_checkpoint_sha256: str,
    control_field: torch.nn.Module,
    official_views: torch.nn.Module,
) -> SourceOnlySamCapabilityRelativeBundle:
    manifest, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="source-only SAM capability-relative manifest",
    )
    manifest = validate_source_only_sam_capability_relative_manifest(manifest)
    if (
        str(getattr(official_views, "radio_checkpoint_sha256", ""))
        != manifest["official_adaptor_checkpoint"]["sha256"]
    ):
        raise ValueError("official capability adaptor checkpoint differs")
    base_record = manifest["base_relative_manifest"]
    base = load_source_only_sam_relative_bundle(
        base_record["path"],
        expected_sha256=base_record["sha256"],
        expected_xyz=expected_xyz,
        expected_valid=expected_valid,
        expected_canonical_radio_cache_sha256=expected_canonical_radio_cache_sha256,
        expected_field_checkpoint_sha256=expected_field_checkpoint_sha256,
        control_field=control_field,
    )
    pair_control = _edge_capability_relation(
        control_field,
        official_views,
        base.teacher.pair_teacher.global_edge_index,
        SAM_RELATIVE_GUARD_BATCH_SIZE,
    )
    positive_control = _edge_capability_relation(
        control_field,
        official_views,
        base.teacher.triplet_index[[0, 1]],
        SAM_RELATIVE_TRIPLET_BATCH_SIZE,
    )
    negative_control = _edge_capability_relation(
        control_field,
        official_views,
        base.teacher.triplet_index[[0, 2]],
        SAM_RELATIVE_TRIPLET_BATCH_SIZE,
    )
    teacher = SourceSamCapabilityRelativeTeacher(
        relative_teacher=base.teacher,
        pair_control_relation=pair_control,
        positive_control_relation=positive_control,
        negative_control_relation=negative_control,
    )
    return SourceOnlySamCapabilityRelativeBundle(
        manifest=manifest,
        manifest_source=source,
        manifest_sha256=digest,
        base=base,
        teacher=teacher,
    )


def source_sam_capability_relative_batch_loss(
    field: torch.nn.Module,
    *,
    official_views: torch.nn.Module,
    teacher: SourceSamCapabilityRelativeTeacher,
    triplet_indices: torch.Tensor,
    guard_edge_indices: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    triplet_selected = torch.as_tensor(triplet_indices).long().cpu().reshape(-1)
    guard_selected = torch.as_tensor(guard_edge_indices).long().cpu().reshape(-1)
    if triplet_selected.numel() == 0 or guard_selected.numel() == 0:
        raise ValueError("capability-relative batches require both objectives")
    if (
        int(triplet_selected.min()) < 0
        or int(triplet_selected.max()) >= teacher.num_triplets
        or int(guard_selected.min()) < 0
        or int(guard_selected.max()) >= teacher.pair_teacher.num_edges
    ):
        raise ValueError("capability-relative batch is outside its teacher axis")
    triplet = teacher.triplet_index[:, triplet_selected]
    guard_edge = teacher.pair_teacher.global_edge_index[:, guard_selected]
    all_rows = torch.cat([triplet.reshape(-1), guard_edge.reshape(-1)])
    unique, inverse = torch.unique(all_rows, sorted=True, return_inverse=True)
    device = field.local_codes.device
    radio = source_direction_only_radio(field.radio_features(unique.to(device)))
    triplet_size = triplet.numel()
    local_triplet = inverse[:triplet_size].reshape_as(triplet).to(device)
    local_guard = inverse[triplet_size:].reshape_as(guard_edge).to(device)
    positive_relation = capability_geometric_mean_relation(
        radio,
        local_triplet[[0, 1]],
        official_views=official_views,
    )
    negative_relation = capability_geometric_mean_relation(
        radio,
        local_triplet[[0, 2]],
        official_views=official_views,
    )
    triplet_weight = teacher.triplet_weight[triplet_selected].to(device)
    ranking_values = F.relu(
        SAM_RELATIVE_MARGIN + negative_relation - positive_relation
    )
    ranking_loss = (ranking_values * triplet_weight).sum() / triplet_weight.sum()

    guard_relation = capability_geometric_mean_relation(
        radio, local_guard, official_views=official_views
    )
    control_guard = teacher.pair_control_relation[guard_selected].to(device)
    same = teacher.pair_teacher.same_relation[guard_selected].to(device)
    guard_weight = teacher.pair_teacher.edge_weight[guard_selected].to(device)
    same_values = F.relu(control_guard - guard_relation)
    separate_values = F.relu(guard_relation - control_guard)

    def weighted(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not bool(mask.any()):
            return values.sum() * 0.0
        local_weight = guard_weight[mask]
        return (values[mask] * local_weight).sum() / local_weight.sum()

    same_guard = weighted(same_values, same)
    separate_guard = weighted(separate_values, ~same)
    loss = 0.5 * (ranking_loss + 0.5 * (same_guard + separate_guard))
    return loss, {
        "capability_ranking_loss": ranking_loss.detach(),
        "capability_same_guard_loss": same_guard.detach(),
        "capability_separate_guard_loss": separate_guard.detach(),
        "triplets": torch.tensor(triplet_selected.numel(), device=device),
        "guard_edges": torch.tensor(guard_selected.numel(), device=device),
    }


@torch.no_grad()
def source_sam_capability_pair_metrics(
    field: torch.nn.Module,
    *,
    official_views: torch.nn.Module,
    teacher: SourceSamCapabilityRelativeTeacher,
    batch_size: int = SAM_RELATIVE_GUARD_BATCH_SIZE,
) -> dict[str, float]:
    relation = _edge_capability_relation(
        field, official_views, teacher.pair_teacher.global_edge_index, batch_size
    )
    same = teacher.pair_teacher.same_relation
    separate = ~same
    return {
        "sam_capability_structure_edges": teacher.pair_teacher.num_edges,
        "sam_capability_same_edges": int(same.sum()),
        "sam_capability_separate_edges": int(separate.sum()),
        "sam_capability_same_mean_relation": float(relation[same].mean()),
        "sam_capability_separate_mean_relation": float(relation[separate].mean()),
        "sam_capability_relation_gap": float(
            relation[same].mean() - relation[separate].mean()
        ),
    }


@torch.no_grad()
def source_sam_capability_relative_metrics(
    field: torch.nn.Module,
    *,
    official_views: torch.nn.Module,
    teacher: SourceSamCapabilityRelativeTeacher,
    batch_size: int = SAM_RELATIVE_TRIPLET_BATCH_SIZE,
) -> dict[str, float]:
    positive = _edge_capability_relation(
        field, official_views, teacher.triplet_index[[0, 1]], batch_size
    )
    negative = _edge_capability_relation(
        field, official_views, teacher.triplet_index[[0, 2]], batch_size
    )
    gap = positive - negative
    violation = gap < SAM_RELATIVE_MARGIN
    return {
        "sam_capability_relative_triplets": teacher.num_triplets,
        "sam_capability_relative_positive_mean_relation": float(positive.mean()),
        "sam_capability_relative_negative_mean_relation": float(negative.mean()),
        "sam_capability_relative_gap": float(gap.mean()),
        "sam_capability_relative_violation_rate": float(violation.float().mean()),
        "sam_capability_relative_hinge_mean": float(
            F.relu(SAM_RELATIVE_MARGIN - gap).mean()
        ),
    }


def evaluate_source_sam_capability_relative_gates(
    *,
    control_pair: Mapping[str, float],
    candidate_pair: Mapping[str, float],
    control_relative: Mapping[str, float],
    candidate_relative: Mapping[str, float],
) -> dict[str, Any]:
    same_axis = (
        int(control_pair.get("sam_capability_structure_edges", -1))
        == int(candidate_pair.get("sam_capability_structure_edges", -2))
        and int(control_relative.get("sam_capability_relative_triplets", -1))
        == int(candidate_relative.get("sam_capability_relative_triplets", -2))
        and int(control_relative.get("sam_capability_relative_triplets", 0)) > 0
    )
    same_pass = float(candidate_pair["sam_capability_same_mean_relation"]) >= float(
        control_pair["sam_capability_same_mean_relation"]
    )
    separate_pass = float(
        candidate_pair["sam_capability_separate_mean_relation"]
    ) <= float(control_pair["sam_capability_separate_mean_relation"])
    pair_gap_pass = float(candidate_pair["sam_capability_relation_gap"]) > float(
        control_pair["sam_capability_relation_gap"]
    )
    triplet_gap_pass = float(
        candidate_relative["sam_capability_relative_gap"]
    ) > float(control_relative["sam_capability_relative_gap"])
    violation_pass = float(
        candidate_relative["sam_capability_relative_violation_rate"]
    ) < float(control_relative["sam_capability_relative_violation_rate"])
    passed = all(
        (same_axis, same_pass, separate_pass, pair_gap_pass, triplet_gap_pass, violation_pass)
    )
    return {
        "same_relation_axis": same_axis,
        "global_capability_same_relation_non_decrease": same_pass,
        "global_capability_separate_relation_non_increase": separate_pass,
        "global_capability_relation_gap_strict_improvement": pair_gap_pass,
        "capability_triplet_gap_strict_improvement": triplet_gap_pass,
        "capability_triplet_violation_strict_decrease": violation_pass,
        "all_structure_gates_passed": passed,
        "benchmark_gate_opened": False,
    }


def evaluate_source_sam_capability_relative_full_gates(
    *,
    manifest: Mapping[str, Any],
    control_primary: Mapping[str, float],
    candidate_primary: Mapping[str, float],
    control_capability: Mapping[str, float],
    candidate_capability: Mapping[str, float],
    structure_decision: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_source_only_sam_capability_relative_manifest(manifest)
    primary_rule = validated["source_gates"]["radio_reconstruction_no_regression"]
    capability_rule = validated["source_gates"]["official_capability_no_regression"]
    if not {"mean_cosine", "p05_cosine"}.issubset(control_primary) or not {
        "mean_cosine",
        "p05_cosine",
    }.issubset(candidate_primary):
        raise ValueError("capability-relative primary metrics are incomplete")
    primary_pass = (
        float(candidate_primary["mean_cosine"])
        >= float(control_primary["mean_cosine"])
        - float(primary_rule["mean_cosine_max_regression"])
        and float(candidate_primary["p05_cosine"])
        >= float(control_primary["p05_cosine"])
        - float(primary_rule["p05_cosine_max_regression"])
    )
    if set(control_capability) != set(candidate_capability) or not control_capability:
        raise ValueError("capability-relative capability metric axis differs")
    checks = {}
    for name in sorted(control_capability):
        threshold = (
            "p05_cosine_max_regression" if "p05" in name else "mean_cosine_max_regression"
        )
        checks[name] = float(candidate_capability[name]) >= float(
            control_capability[name]
        ) - float(capability_rule[threshold])
    capability_pass = all(checks.values())
    structure_pass = bool(structure_decision.get("all_structure_gates_passed", False))
    passed = primary_pass and capability_pass and structure_pass
    return {
        **dict(structure_decision),
        "raw_radio_preservation": primary_pass,
        "official_capability_checks": checks,
        "official_capability_preservation": capability_pass,
        "all_source_gates_passed": passed,
        "benchmark_gate_opened": False,
        "checkpoint_seal_allowed": passed,
    }


__all__ = [
    "SOURCE_ONLY_SAM_CAPABILITY_RELATIVE_CONTRACT_SHA256",
    "SOURCE_ONLY_SAM_CAPABILITY_RELATIVE_SCHEMA",
    "SourceOnlySamCapabilityRelativeBundle",
    "SourceSamCapabilityRelativeTeacher",
    "capability_geometric_mean_relation",
    "evaluate_source_sam_capability_relative_full_gates",
    "evaluate_source_sam_capability_relative_gates",
    "load_source_only_sam_capability_relative_bundle",
    "source_only_sam_capability_relative_contract",
    "source_sam_capability_pair_metrics",
    "source_sam_capability_relative_batch_loss",
    "source_sam_capability_relative_metrics",
    "validate_source_only_sam_capability_relative_manifest",
]
