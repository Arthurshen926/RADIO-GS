"""Source-only official-SAM structure supervision for one RADIO field.

This module deliberately does *not* add a SAM feature branch.  Official SAM
masks are lifted through a frozen raster-adjoint responsibility operator at
mapping time and reduced to sparse same-region / cross-boundary primitive
relations.  Training changes only the direction of the existing canonical
RADIO descriptor; the teacher masks, mask identities, RGB frames, and SAM
features are absent from the saved field state and from every query-time path.

Nested SAM masks are allowed to disagree.  We retain only relation pairs with
a globally fixed dominance margin and abstain on unseen or ambiguous pairs.
The loss is class-balanced so a scene with many within-region edges cannot
erase the much rarer boundary signal.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from radio_gs.training.source_only_multiteacher import source_direction_only_radio
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
)


SOURCE_ONLY_SAM_STRUCTURE_SCHEMA = "radio_gs.source_only_sam_radio_structure.v1"
SOURCE_ONLY_SAM_STRUCTURE_SCHEMA_VERSION = 1
OFFICIAL_SAM_BUILD_AUTHORITY_SCHEMA = (
    "radio_gs.source_official_sam_build_authority.v1"
)
RELATION_CACHE_TEACHER = "official_sam3_multimask_scale_ordered_regions"

# Dataset-independent method constants.  They are intentionally absent from
# the CLI so target metrics cannot turn this into per-scene tuning.
SAM_STRUCTURE_WEIGHT = 0.05
SAM_STRUCTURE_EDGE_CAP = 65_536
SAM_STRUCTURE_BATCH_SIZE = 512
SAM_STRUCTURE_DOMINANCE_MINIMUM = 1.0 / 3.0
SAM_STRUCTURE_SEPARATION_MARGIN = 0.25


def source_only_sam_structure_contract() -> dict[str, Any]:
    return {
        "schema": SOURCE_ONLY_SAM_STRUCTURE_SCHEMA,
        "schema_version": SOURCE_ONLY_SAM_STRUCTURE_SCHEMA_VERSION,
        "persistent_semantic_feature": "canonical_radio_only",
        "mapping_teacher": {
            "model": "official_sam",
            "input": "legal_source_rgb_only",
            "lifting": "true_alpha_compositing_raster_adjoint",
            "output": "sparse_primitive_relation_votes",
            "mask_identity_persisted": False,
            "sam_feature_persisted": False,
        },
        "relation": {
            "same_evidence": "both_endpoints_inside_one_source_mask",
            "separate_evidence": "one_inside_one_outside_one_source_mask",
            "scale_votes_preserved_in_teacher_cache": True,
            "field_target": "dominant_same_or_separate_after_scale_sum",
            "ambiguous_or_unseen": "abstain",
            "dominance_minimum": SAM_STRUCTURE_DOMINANCE_MINIMUM,
            "edge_cap": SAM_STRUCTURE_EDGE_CAP,
            "edge_selection": "ascending_edge_axis_evenly_spaced_floor_v1",
        },
        "loss": {
            "same": "one_minus_radio_direction_cosine",
            "separate": "relu(cosine_minus_fixed_margin)",
            "separation_margin": SAM_STRUCTURE_SEPARATION_MARGIN,
            "class_balance": "equal_same_and_separate_mass",
            "edge_weight": "normalized_log1p_vote_mass_times_dominance",
            "weight": SAM_STRUCTURE_WEIGHT,
            "batch_size": SAM_STRUCTURE_BATCH_SIZE,
            "student_gauge_gradient": (
                "zero_by_detached_norm_direction_reparameterization_v1"
            ),
        },
        "query_time": {
            "source_rgb": False,
            "target_rgb": False,
            "sam_decoder": False,
            "sam_cache_or_mask": False,
            "teacher_relation_cache": False,
        },
        "forbidden_inputs": [
            "benchmark_query_or_text_bank",
            "benchmark_target_or_evaluation_rgb",
            "benchmark_ground_truth",
            "benchmark_label_or_mask",
            "benchmark_metric_or_prediction",
        ],
    }


SOURCE_ONLY_SAM_STRUCTURE_CONTRACT_SHA256 = canonical_json_sha256(
    source_only_sam_structure_contract()
)


def _file_record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    result = {"path": str(value["path"]), "sha256": str(value["sha256"])}
    validate_file_record(result, label=label)
    return result


def validate_source_only_sam_structure_manifest(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("source-only SAM structure manifest must be a mapping")
    payload = dict(value)
    required = {
        "schema",
        "schema_version",
        "contract",
        "contract_sha256",
        "status",
        "scene_id",
        "field_control",
        "canonical_radio_cache",
        "relation_cache",
        "relation_graph",
        "official_sam_build_authority",
        "loss",
        "source_gates",
        "access_audit",
        "execution",
    }
    contract = source_only_sam_structure_contract()
    if (
        set(payload) != required
        or payload.get("schema") != SOURCE_ONLY_SAM_STRUCTURE_SCHEMA
        or payload.get("schema_version") != SOURCE_ONLY_SAM_STRUCTURE_SCHEMA_VERSION
        or payload.get("contract") != contract
        or payload.get("contract_sha256")
        != SOURCE_ONLY_SAM_STRUCTURE_CONTRACT_SHA256
        or payload.get("status") != "preregistered_training_not_started"
        or not str(payload.get("scene_id", ""))
        or payload.get("loss") != contract["loss"]
    ):
        raise ValueError("source-only SAM structure manifest contract differs")
    for name in (
        "field_control",
        "canonical_radio_cache",
        "relation_cache",
        "relation_graph",
        "official_sam_build_authority",
    ):
        payload[name] = _file_record(payload[name], label=f"SAM structure {name}")
    expected_access = {
        "source_rgb_opened_during_mapping": True,
        "official_sam_opened_during_mapping": True,
        "query_time_source_rgb_opened": False,
        "query_time_target_rgb_opened": False,
        "benchmark_query_or_text_opened": False,
        "benchmark_target_or_evaluation_rgb_opened": False,
        "benchmark_ground_truth_opened": False,
        "benchmark_labels_or_masks_opened": False,
        "benchmark_metrics_or_predictions_opened": False,
    }
    if payload.get("access_audit") != expected_access:
        raise ValueError("source-only SAM structure access audit differs")
    expected_gates = {
        "radio_reconstruction_no_regression",
        "gauge_no_regression",
        "sam_same_cosine_non_decrease",
        "sam_separate_cosine_non_increase",
        "sam_relation_gap_strict_improvement",
        "six_task_benchmark_gate",
    }
    gates = payload.get("source_gates")
    if not isinstance(gates, Mapping) or set(gates) != expected_gates:
        raise ValueError("source-only SAM structure gate schema differs")
    if gates.get("six_task_benchmark_gate") != (
        "closed_until_all_source_gates_pass_then_frozen_one_shot"
    ):
        raise ValueError("source-only SAM structure benchmark gate differs")
    execution = payload.get("execution")
    if not isinstance(execution, Mapping) or execution != {
        "gpu_started": False,
        "per_scene_or_per_task_tuning": False,
        "output_no_clobber": True,
        "teacher_payload_saved_in_checkpoint": False,
    }:
        raise ValueError("source-only SAM structure execution contract differs")
    return payload


def _evenly_spaced_selection(count: int, cap: int) -> torch.Tensor:
    if count <= 0 or cap <= 0:
        raise ValueError("SAM structure relation count/cap must be positive")
    if count <= cap:
        return torch.arange(count, dtype=torch.long)
    if cap == 1:
        return torch.zeros(1, dtype=torch.long)
    position = torch.arange(cap, dtype=torch.int64)
    return torch.div(position * (count - 1), cap - 1, rounding_mode="floor")


@dataclass(frozen=True)
class SourceSamStructureTeacher:
    global_edge_index: torch.Tensor
    same_relation: torch.Tensor
    edge_weight: torch.Tensor
    relation_cache_source: Path
    relation_cache_sha256: str

    @property
    def num_edges(self) -> int:
        return int(self.global_edge_index.shape[1])

    def subset(self, keep: torch.Tensor) -> "SourceSamStructureTeacher":
        selected = torch.as_tensor(keep).bool().cpu().reshape(-1)
        if selected.shape != (self.num_edges,) or not bool(selected.any()):
            raise ValueError("SAM structure subset must retain at least one edge")
        return SourceSamStructureTeacher(
            global_edge_index=self.global_edge_index[:, selected].contiguous(),
            same_relation=self.same_relation[selected].contiguous(),
            edge_weight=self.edge_weight[selected].contiguous(),
            relation_cache_source=self.relation_cache_source,
            relation_cache_sha256=self.relation_cache_sha256,
        )


@dataclass(frozen=True)
class SourceOnlySamStructureBundle:
    manifest: dict[str, Any]
    manifest_source: Path
    manifest_sha256: str
    teacher: SourceSamStructureTeacher


def _validate_official_sam_authority(record: Mapping[str, str]) -> None:
    payload, _digest, _source = load_json_object(
        record["path"],
        expected_sha256=record["sha256"],
        label="official source SAM build authority",
    )
    expected = {
        "official_sam": True,
        "source_rgb_only": True,
        "query_free": True,
        "target_or_evaluation_rgb_opened": False,
        "benchmark_query_opened": False,
        "benchmark_gt_or_metric_opened": False,
        "teacher_artifacts_training_only": True,
    }
    if payload.get("schema") != OFFICIAL_SAM_BUILD_AUTHORITY_SCHEMA or any(
        payload.get(name) is not wanted for name, wanted in expected.items()
    ):
        raise ValueError("official source SAM build authority differs")
    mask_caches = payload.get("source_mask_caches")
    if not isinstance(mask_caches, list) or not mask_caches:
        raise ValueError("official source SAM build authority has no mask caches")
    for index, item in enumerate(mask_caches):
        _file_record(item, label=f"official source SAM mask cache {index}")


def _global_relation_edges(
    relation: Mapping[str, Any],
    graph: Mapping[str, Any],
    *,
    expected_xyz: torch.Tensor,
) -> torch.Tensor:
    edge = torch.as_tensor(relation.get("edge_index")).long().cpu()
    graph_xyz = torch.as_tensor(graph.get("xyz")).float().cpu()
    field_xyz = torch.as_tensor(expected_xyz).float().cpu()
    if (
        edge.ndim != 2
        or edge.shape[0] != 2
        or edge.shape[1] == 0
        or graph_xyz.ndim != 2
        or graph_xyz.shape[1] != 3
        or int(edge.min()) < 0
        or int(edge.max()) >= graph_xyz.shape[0]
    ):
        raise ValueError("official SAM relation graph layout differs")
    if graph_xyz.shape == field_xyz.shape and torch.equal(graph_xyz, field_xyz):
        rows = torch.arange(graph_xyz.shape[0], dtype=torch.long)
    else:
        rows = torch.as_tensor(graph.get("global_rows")).long().cpu().reshape(-1)
        if (
            rows.shape != (graph_xyz.shape[0],)
            or rows.numel() != torch.unique(rows).numel()
            or int(rows.min()) < 0
            or int(rows.max()) >= field_xyz.shape[0]
            or not torch.equal(graph_xyz, field_xyz[rows])
        ):
            raise ValueError("official SAM relation graph is not aligned to the RADIO field")
    return rows[edge].contiguous()


def load_source_only_sam_structure_bundle(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_xyz: torch.Tensor,
    expected_valid: torch.Tensor,
    expected_canonical_radio_cache_sha256: str,
    expected_field_checkpoint_sha256: str,
) -> SourceOnlySamStructureBundle:
    manifest, manifest_sha256, manifest_source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="source-only SAM structure manifest",
    )
    manifest = validate_source_only_sam_structure_manifest(manifest)
    if (
        manifest["canonical_radio_cache"]["sha256"]
        != str(expected_canonical_radio_cache_sha256)
        or manifest["field_control"]["sha256"]
        != str(expected_field_checkpoint_sha256)
    ):
        raise ValueError("source-only SAM structure caller lineage differs")
    _validate_official_sam_authority(manifest["official_sam_build_authority"])
    relation, relation_sha, relation_source = load_torch_mapping(
        manifest["relation_cache"]["path"],
        expected_sha256=manifest["relation_cache"]["sha256"],
        map_location="cpu",
        label="source-only official SAM relation cache",
    )
    graph, graph_sha, _graph_source = load_torch_mapping(
        manifest["relation_graph"]["path"],
        expected_sha256=manifest["relation_graph"]["sha256"],
        map_location="cpu",
        label="source-only official SAM relation graph",
    )
    metadata = relation.get("metadata")
    if (
        relation.get("schema_version") != 2
        or not isinstance(metadata, Mapping)
        or metadata.get("teacher") != RELATION_CACHE_TEACHER
        or metadata.get("query_free") is not True
        or metadata.get("labels_opened") is not False
        or metadata.get("instances_opened") is not False
        or metadata.get("text_opened") is not False
        or metadata.get("membership_lifting") != "raster_adjoint"
        or metadata.get("scene_graph_sha256") != graph_sha
    ):
        raise ValueError("source-only official SAM relation cache provenance differs")
    global_edge = _global_relation_edges(
        relation, graph, expected_xyz=expected_xyz
    )
    valid = torch.as_tensor(expected_valid).bool().cpu().reshape(-1)
    if valid.shape != (torch.as_tensor(expected_xyz).shape[0],):
        raise ValueError("source-only SAM field validity axis differs")
    same_votes = torch.as_tensor(relation.get("same_votes")).float().cpu()
    separate_votes = torch.as_tensor(relation.get("separate_votes")).float().cpu()
    if (
        same_votes.ndim != 2
        or separate_votes.shape != same_votes.shape
        or same_votes.shape[0] != global_edge.shape[1]
        or not bool(torch.isfinite(same_votes).all())
        or not bool(torch.isfinite(separate_votes).all())
        or bool((same_votes < 0).any())
        or bool((separate_votes < 0).any())
    ):
        raise ValueError("source-only official SAM relation votes differ")
    same_mass = same_votes.sum(dim=-1)
    separate_mass = separate_votes.sum(dim=-1)
    total = same_mass + separate_mass
    dominance = (same_mass - separate_mass).abs() / total.clamp_min(1e-12)
    keep = valid[global_edge].all(dim=0)
    keep &= total > 0
    keep &= dominance >= SAM_STRUCTURE_DOMINANCE_MINIMUM
    if not bool(keep.any()):
        raise ValueError("source-only official SAM relations have no unambiguous field edges")
    selected = torch.where(keep)[0]
    selected = selected[
        _evenly_spaced_selection(int(selected.numel()), SAM_STRUCTURE_EDGE_CAP)
    ]
    label = same_mass[selected] > separate_mass[selected]
    if not bool(label.any()) or not bool((~label).any()):
        raise ValueError("source-only official SAM relations require both classes")
    weight = torch.log1p(total[selected]) * dominance[selected]
    weight = weight / weight.mean().clamp_min(1e-6)
    teacher = SourceSamStructureTeacher(
        global_edge_index=global_edge[:, selected].contiguous(),
        same_relation=label.contiguous(),
        edge_weight=weight.float().contiguous(),
        relation_cache_source=relation_source,
        relation_cache_sha256=relation_sha,
    )
    return SourceOnlySamStructureBundle(
        manifest=manifest,
        manifest_source=manifest_source,
        manifest_sha256=manifest_sha256,
        teacher=teacher,
    )


def source_sam_structure_batch_loss(
    field: torch.nn.Module,
    *,
    teacher: SourceSamStructureTeacher,
    edge_indices: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    selected = torch.as_tensor(edge_indices).long().cpu().reshape(-1)
    if selected.numel() == 0:
        zero = field.local_codes.sum() * 0.0
        return zero, {
            "same_edges": torch.tensor(0, device=field.local_codes.device),
            "separate_edges": torch.tensor(0, device=field.local_codes.device),
        }
    if int(selected.min()) < 0 or int(selected.max()) >= teacher.num_edges:
        raise ValueError("source-only SAM relation batch is outside its edge axis")
    edge = teacher.global_edge_index[:, selected]
    unique, inverse = torch.unique(edge.reshape(-1), sorted=True, return_inverse=True)
    device = field.local_codes.device
    radio = source_direction_only_radio(field.radio_features(unique.to(device)))
    direction = F.normalize(radio.float(), dim=-1, eps=1e-8)
    local_edge = inverse.reshape_as(edge).to(device)
    cosine = (direction[local_edge[0]] * direction[local_edge[1]]).sum(dim=-1)
    same = teacher.same_relation[selected].to(device)
    weight = teacher.edge_weight[selected].to(device).float().clamp_min(1e-6)
    positive = 1.0 - cosine
    negative = F.relu(cosine - SAM_STRUCTURE_SEPARATION_MARGIN)

    def balanced_part(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not bool(mask.any()):
            return values.sum() * 0.0
        local_weight = weight[mask]
        return (values[mask] * local_weight).sum() / local_weight.sum()

    same_loss = balanced_part(positive, same)
    separate_loss = balanced_part(negative, ~same)
    loss = 0.5 * (same_loss + separate_loss)
    return loss, {
        "same_edges": same.sum().detach(),
        "separate_edges": (~same).sum().detach(),
        "same_loss": same_loss.detach(),
        "separate_loss": separate_loss.detach(),
    }


@torch.no_grad()
def source_sam_structure_metrics(
    field: torch.nn.Module,
    *,
    teacher: SourceSamStructureTeacher,
    batch_size: int = SAM_STRUCTURE_BATCH_SIZE,
) -> dict[str, float]:
    if batch_size <= 0:
        raise ValueError("source-only SAM metric batch size must be positive")
    same_cosine: list[torch.Tensor] = []
    separate_cosine: list[torch.Tensor] = []
    device = field.local_codes.device
    for start in range(0, teacher.num_edges, int(batch_size)):
        stop = min(start + int(batch_size), teacher.num_edges)
        edge = teacher.global_edge_index[:, start:stop]
        unique, inverse = torch.unique(edge.reshape(-1), sorted=True, return_inverse=True)
        direction = F.normalize(
            field.radio_features(unique.to(device)).float(), dim=-1, eps=1e-8
        )
        local_edge = inverse.reshape_as(edge).to(device)
        cosine = (
            direction[local_edge[0]] * direction[local_edge[1]]
        ).sum(dim=-1).cpu()
        same = teacher.same_relation[start:stop]
        same_cosine.append(cosine[same])
        separate_cosine.append(cosine[~same])
    positive = torch.cat([part for part in same_cosine if part.numel()])
    negative = torch.cat([part for part in separate_cosine if part.numel()])
    return {
        "sam_structure_edges": teacher.num_edges,
        "sam_same_edges": int(positive.numel()),
        "sam_separate_edges": int(negative.numel()),
        "sam_same_mean_cosine": float(positive.mean()),
        "sam_separate_mean_cosine": float(negative.mean()),
        "sam_relation_gap": float(positive.mean() - negative.mean()),
    }


def evaluate_source_sam_structure_gates(
    *, control: Mapping[str, float], candidate: Mapping[str, float]
) -> dict[str, Any]:
    required = {
        "sam_structure_edges",
        "sam_same_edges",
        "sam_separate_edges",
        "sam_same_mean_cosine",
        "sam_separate_mean_cosine",
        "sam_relation_gap",
    }
    if not required.issubset(control) or not required.issubset(candidate):
        raise ValueError("source-only SAM structure metrics are incomplete")
    same_axis = (
        int(control["sam_structure_edges"]),
        int(control["sam_same_edges"]),
        int(control["sam_separate_edges"]),
    ) == (
        int(candidate["sam_structure_edges"]),
        int(candidate["sam_same_edges"]),
        int(candidate["sam_separate_edges"]),
    )
    same_pass = float(candidate["sam_same_mean_cosine"]) >= float(
        control["sam_same_mean_cosine"]
    )
    separate_pass = float(candidate["sam_separate_mean_cosine"]) <= float(
        control["sam_separate_mean_cosine"]
    )
    gap_pass = float(candidate["sam_relation_gap"]) > float(
        control["sam_relation_gap"]
    )
    passed = same_axis and same_pass and separate_pass and gap_pass
    return {
        "same_relation_axis": same_axis,
        "sam_same_cosine_non_decrease": same_pass,
        "sam_separate_cosine_non_increase": separate_pass,
        "sam_relation_gap_strict_improvement": gap_pass,
        "all_structure_gates_passed": passed,
        "benchmark_gate_opened": passed,
    }


def validate_single_radio_checkpoint_payload(payload: object) -> None:
    """Prove that a canonical checkpoint did not persist teacher features."""

    if not isinstance(payload, Mapping) or not isinstance(payload.get("state_dict"), Mapping):
        raise ValueError("canonical RADIO checkpoint payload differs")
    forbidden = ("sam", "clip", "siglip", "dino", "teacher", "mask", "prototype")
    offending = [
        str(name)
        for name, tensor in payload["state_dict"].items()
        if torch.is_tensor(tensor) and any(token in str(name).lower() for token in forbidden)
    ]
    if offending:
        raise ValueError(f"teacher feature tensors persisted in RADIO checkpoint: {offending}")
    for metadata_name in (
        "source_only_sam_structure",
        "source_only_sam_relative_structure",
    ):
        metadata = payload.get(metadata_name, {})
        if metadata:
            if (
                not isinstance(metadata, Mapping)
                or metadata.get("teacher_payload_saved") is not False
            ):
                raise ValueError("source-only SAM teacher persistence declaration differs")
            if any(torch.is_tensor(value) for value in metadata.values()):
                raise ValueError("source-only SAM metadata contains a teacher tensor")


__all__ = [
    "OFFICIAL_SAM_BUILD_AUTHORITY_SCHEMA",
    "SAM_STRUCTURE_BATCH_SIZE",
    "SAM_STRUCTURE_DOMINANCE_MINIMUM",
    "SAM_STRUCTURE_EDGE_CAP",
    "SAM_STRUCTURE_SEPARATION_MARGIN",
    "SAM_STRUCTURE_WEIGHT",
    "SOURCE_ONLY_SAM_STRUCTURE_CONTRACT_SHA256",
    "SOURCE_ONLY_SAM_STRUCTURE_SCHEMA",
    "SourceOnlySamStructureBundle",
    "SourceSamStructureTeacher",
    "evaluate_source_sam_structure_gates",
    "load_source_only_sam_structure_bundle",
    "source_only_sam_structure_contract",
    "source_sam_structure_batch_loss",
    "source_sam_structure_metrics",
    "validate_single_radio_checkpoint_payload",
    "validate_source_only_sam_structure_manifest",
]
