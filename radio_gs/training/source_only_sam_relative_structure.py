"""Scale-matched, no-harm official-SAM supervision for one RADIO field.

Version 1 of the source-only SAM teacher treated every pair inside one mask as
an equivalence relation.  That is too strong for nested part/object/context
masks.  This module keeps the same immutable source masks and the same single
RADIO field, but compiles them into relative statements:

    sim(anchor, same-scale inside) >= sim(anchor, same-scale outside) + margin.

Only violated orderings receive a ranking gradient.  A control-relative guard
also forbids the selected same edges from becoming less similar and selected
boundary edges from becoming more similar.  Both terms operate on RADIO
direction through the tangent-only reparameterization; no SAM feature, mask ID,
RGB frame, or teacher tensor is persisted in the field checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from radio_gs.training.source_only_multiteacher import source_direction_only_radio
from radio_gs.training.source_only_sam_structure import (
    SAM_STRUCTURE_BATCH_SIZE,
    SAM_STRUCTURE_DOMINANCE_MINIMUM,
    SAM_STRUCTURE_EDGE_CAP,
    SAM_STRUCTURE_WEIGHT,
    SourceOnlySamStructureBundle,
    SourceSamStructureTeacher,
    _global_relation_edges,
    evaluate_source_sam_structure_gates,
    load_source_only_sam_structure_bundle,
    source_sam_structure_metrics,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
)


SOURCE_ONLY_SAM_RELATIVE_SCHEMA = "radio_gs.source_only_sam_radio_relative_structure.v2"
SOURCE_ONLY_SAM_RELATIVE_SCHEMA_VERSION = 2
SAM_RELATIVE_MARGIN = 0.05
SAM_RELATIVE_TRIPLET_CAP = 65_536
SAM_RELATIVE_TRIPLET_BATCH_SIZE = 512
SAM_RELATIVE_GUARD_BATCH_SIZE = 512


def source_only_sam_relative_contract() -> dict[str, Any]:
    return {
        "schema": SOURCE_ONLY_SAM_RELATIVE_SCHEMA,
        "schema_version": SOURCE_ONLY_SAM_RELATIVE_SCHEMA_VERSION,
        "persistent_semantic_feature": "canonical_radio_only",
        "mapping_teacher": {
            "model": "official_sam",
            "input": "legal_source_rgb_only",
            "base_relation_authority": "validated_source_only_sam_radio_structure_v1",
            "output": "scale_matched_relative_primitive_ordering",
            "teacher_payload_persisted": False,
        },
        "triplet": {
            "anchor_orientation": "both_endpoints_of_each_undirected_relation_edge",
            "scale_matching": "exact_vote_scale_bin",
            "positive": "strongest_dominant_same_neighbor_per_anchor_scale",
            "negative": "strongest_dominant_separate_neighbor_per_anchor_scale",
            "ambiguous_or_unseen": "abstain",
            "dominance_minimum": SAM_STRUCTURE_DOMINANCE_MINIMUM,
            "cap": SAM_RELATIVE_TRIPLET_CAP,
            "cap_selection": "ascending_anchor_scale_peer_evenly_spaced_floor_v1",
        },
        "loss": {
            "ranking": "relu(margin+cos_anchor_negative-cos_anchor_positive)",
            "ranking_margin": SAM_RELATIVE_MARGIN,
            "already_ordered_relation": "zero_ranking_gradient",
            "same_guard": "relu(control_same_cosine-current_same_cosine)",
            "separate_guard": "relu(current_separate_cosine-control_separate_cosine)",
            "guard_class_balance": "equal_same_and_separate_mass",
            "ranking_guard_balance": "equal_half_mass",
            "edge_weight": "normalized_log1p_vote_mass_times_dominance",
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


SOURCE_ONLY_SAM_RELATIVE_CONTRACT_SHA256 = canonical_json_sha256(
    source_only_sam_relative_contract()
)


def validate_source_only_sam_relative_manifest(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("source-only SAM relative manifest must be a mapping")
    payload = dict(value)
    required = {
        "schema",
        "schema_version",
        "contract",
        "contract_sha256",
        "status",
        "scene_id",
        "base_structure_manifest",
        "loss",
        "source_gates",
        "access_audit",
        "execution",
    }
    contract = source_only_sam_relative_contract()
    if (
        set(payload) != required
        or payload.get("schema") != SOURCE_ONLY_SAM_RELATIVE_SCHEMA
        or payload.get("schema_version") != SOURCE_ONLY_SAM_RELATIVE_SCHEMA_VERSION
        or payload.get("contract") != contract
        or payload.get("contract_sha256")
        != SOURCE_ONLY_SAM_RELATIVE_CONTRACT_SHA256
        or payload.get("status") != "preregistered_training_not_started"
        or not str(payload.get("scene_id", ""))
        or payload.get("loss") != contract["loss"]
    ):
        raise ValueError("source-only SAM relative manifest contract differs")
    base = payload.get("base_structure_manifest")
    if not isinstance(base, Mapping) or set(base) != {"path", "sha256"}:
        raise ValueError("source-only SAM relative base manifest differs")
    validate_file_record(base, label="source-only SAM relative base manifest")
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
        raise ValueError("source-only SAM relative access audit differs")
    gates = payload.get("source_gates")
    expected_gates = {
        "radio_reconstruction_no_regression",
        "official_capability_no_regression",
        "global_same_cosine_non_decrease",
        "global_separate_cosine_non_increase",
        "global_relation_gap_strict_improvement",
        "scale_triplet_gap_strict_improvement",
        "scale_triplet_violation_strict_decrease",
        "six_task_benchmark_gate",
    }
    if not isinstance(gates, Mapping) or set(gates) != expected_gates:
        raise ValueError("source-only SAM relative gate schema differs")
    expected_gate_values = {
        "radio_reconstruction_no_regression": {
            "mean_cosine_max_regression": 0.005,
            "p05_cosine_max_regression": 0.01,
        },
        "official_capability_no_regression": {
            "mean_cosine_max_regression": 0.005,
            "p05_cosine_max_regression": 0.01,
        },
        "global_same_cosine_non_decrease": True,
        "global_separate_cosine_non_increase": True,
        "global_relation_gap_strict_improvement": True,
        "scale_triplet_gap_strict_improvement": True,
        "scale_triplet_violation_strict_decrease": True,
        "six_task_benchmark_gate": (
            "closed_until_all_source_gates_pass_then_frozen_one_shot"
        ),
    }
    if dict(gates) != expected_gate_values:
        raise ValueError("source-only SAM relative benchmark gate differs")
    if payload.get("execution") != {
        "gpu_started": False,
        "per_scene_or_per_task_tuning": False,
        "output_no_clobber": True,
        "teacher_payload_saved_in_checkpoint": False,
        "v1_candidate_checkpoint_used": False,
    }:
        raise ValueError("source-only SAM relative execution contract differs")
    return payload


@dataclass(frozen=True)
class SourceSamRelativeTeacher:
    pair_teacher: SourceSamStructureTeacher
    triplet_index: torch.Tensor
    triplet_weight: torch.Tensor
    scale_bin: torch.Tensor
    pair_control_cosine: torch.Tensor
    positive_control_cosine: torch.Tensor
    negative_control_cosine: torch.Tensor

    @property
    def num_triplets(self) -> int:
        return int(self.triplet_index.shape[1])

    def training_subset(
        self, *, pair_keep: torch.Tensor, triplet_keep: torch.Tensor
    ) -> "SourceSamRelativeTeacher":
        pair_selected = torch.as_tensor(pair_keep).bool().cpu().reshape(-1)
        triplet_selected = torch.as_tensor(triplet_keep).bool().cpu().reshape(-1)
        if (
            pair_selected.shape != (self.pair_teacher.num_edges,)
            or triplet_selected.shape != (self.num_triplets,)
            or not bool(pair_selected.any())
            or not bool(triplet_selected.any())
        ):
            raise ValueError("source-only SAM relative training subset is empty")
        pair = self.pair_teacher.subset(pair_selected)
        return SourceSamRelativeTeacher(
            pair_teacher=pair,
            triplet_index=self.triplet_index[:, triplet_selected].contiguous(),
            triplet_weight=self.triplet_weight[triplet_selected].contiguous(),
            scale_bin=self.scale_bin[triplet_selected].contiguous(),
            pair_control_cosine=self.pair_control_cosine[pair_selected].contiguous(),
            positive_control_cosine=(
                self.positive_control_cosine[triplet_selected].contiguous()
            ),
            negative_control_cosine=(
                self.negative_control_cosine[triplet_selected].contiguous()
            ),
        )


@dataclass(frozen=True)
class SourceOnlySamRelativeBundle:
    manifest: dict[str, Any]
    manifest_source: Path
    manifest_sha256: str
    base: SourceOnlySamStructureBundle
    teacher: SourceSamRelativeTeacher


def _best_peer_by_anchor(
    edge: torch.Tensor, strength: torch.Tensor, mask: torch.Tensor, num_rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = torch.where(mask)[0]
    peer = torch.full((num_rows,), -1, dtype=torch.long)
    best_strength = torch.zeros(num_rows, dtype=torch.float32)
    if selected.numel() == 0:
        return peer, best_strength
    oriented_anchor = torch.cat([edge[0, selected], edge[1, selected]])
    oriented_peer = torch.cat([edge[1, selected], edge[0, selected]])
    oriented_strength = torch.cat([strength[selected], strength[selected]]).float()
    # Stable lexicographic order: anchor ascending, strength descending, peer
    # ascending.  The first row per anchor is therefore deterministic.
    order = torch.argsort(oriented_peer, stable=True)
    order = order[torch.argsort(oriented_strength[order], descending=True, stable=True)]
    order = order[torch.argsort(oriented_anchor[order], stable=True)]
    anchor_sorted = oriented_anchor[order]
    first = torch.ones(order.numel(), dtype=torch.bool)
    first[1:] = anchor_sorted[1:] != anchor_sorted[:-1]
    chosen = order[first]
    peer[oriented_anchor[chosen]] = oriented_peer[chosen]
    best_strength[oriented_anchor[chosen]] = oriented_strength[chosen]
    return peer, best_strength


def _evenly_spaced_triplets(count: int, cap: int) -> torch.Tensor:
    if count <= cap:
        return torch.arange(count, dtype=torch.long)
    position = torch.arange(cap, dtype=torch.int64)
    return torch.div(position * (count - 1), cap - 1, rounding_mode="floor")


def build_scale_matched_triplets(
    *,
    global_edge: torch.Tensor,
    same_votes: torch.Tensor,
    separate_votes: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    edge = torch.as_tensor(global_edge).long().cpu()
    same = torch.as_tensor(same_votes).float().cpu()
    separate = torch.as_tensor(separate_votes).float().cpu()
    valid_rows = torch.as_tensor(valid).bool().cpu().reshape(-1)
    if (
        edge.ndim != 2
        or edge.shape[0] != 2
        or same.ndim != 2
        or separate.shape != same.shape
        or same.shape[0] != edge.shape[1]
        or int(edge.min()) < 0
        or int(edge.max()) >= valid_rows.numel()
    ):
        raise ValueError("source-only SAM relative vote table differs")
    total = same + separate
    dominance = (same - separate).abs() / total.clamp_min(1e-12)
    valid_edge = valid_rows[edge].all(dim=0)
    triplet_parts: list[torch.Tensor] = []
    weight_parts: list[torch.Tensor] = []
    scale_parts: list[torch.Tensor] = []
    num_rows = int(valid_rows.numel())
    for scale in range(int(same.shape[1])):
        observed = valid_edge & (total[:, scale] > 0)
        observed &= dominance[:, scale] >= SAM_STRUCTURE_DOMINANCE_MINIMUM
        same_mask = observed & (same[:, scale] > separate[:, scale])
        separate_mask = observed & (separate[:, scale] > same[:, scale])
        strength = torch.log1p(total[:, scale]) * dominance[:, scale]
        positive_peer, positive_strength = _best_peer_by_anchor(
            edge, strength, same_mask, num_rows
        )
        negative_peer, negative_strength = _best_peer_by_anchor(
            edge, strength, separate_mask, num_rows
        )
        anchors = torch.where((positive_peer >= 0) & (negative_peer >= 0))[0]
        if anchors.numel() == 0:
            continue
        triplet_parts.append(
            torch.stack(
                [anchors, positive_peer[anchors], negative_peer[anchors]], dim=0
            )
        )
        weight_parts.append(
            torch.sqrt(
                positive_strength[anchors].clamp_min(1e-12)
                * negative_strength[anchors].clamp_min(1e-12)
            )
        )
        scale_parts.append(torch.full_like(anchors, scale))
    if not triplet_parts:
        raise ValueError("source-only SAM relative teacher has no matched triplets")
    triplet = torch.cat(triplet_parts, dim=1)
    weight = torch.cat(weight_parts).float()
    scale_bin = torch.cat(scale_parts).long()
    # Parts are already in scale-major/anchor-major order.  Include peers in a
    # deterministic final key so future relation builders cannot alter order.
    key = (
        scale_bin.to(torch.int64) * (num_rows**3)
        + triplet[0].to(torch.int64) * (num_rows**2)
        + triplet[1].to(torch.int64) * num_rows
        + triplet[2].to(torch.int64)
    )
    order = torch.argsort(key, stable=True)
    selected = order[_evenly_spaced_triplets(order.numel(), SAM_RELATIVE_TRIPLET_CAP)]
    triplet = triplet[:, selected].contiguous()
    weight = weight[selected]
    weight = (weight / weight.mean().clamp_min(1e-6)).contiguous()
    scale_bin = scale_bin[selected].contiguous()
    if bool((triplet[0] == triplet[1]).any()) or bool(
        (triplet[0] == triplet[2]).any()
    ) or bool((triplet[1] == triplet[2]).any()):
        raise ValueError("source-only SAM relative triplet is degenerate")
    return triplet, weight, scale_bin


@torch.no_grad()
def _edge_cosine(field: torch.nn.Module, edge: torch.Tensor, batch_size: int) -> torch.Tensor:
    values: list[torch.Tensor] = []
    device = field.local_codes.device
    for start in range(0, edge.shape[1], batch_size):
        part = edge[:, start : start + batch_size]
        unique, inverse = torch.unique(part.reshape(-1), sorted=True, return_inverse=True)
        direction = F.normalize(
            field.radio_features(unique.to(device)).float(), dim=-1, eps=1e-8
        )
        local = inverse.reshape_as(part).to(device)
        values.append((direction[local[0]] * direction[local[1]]).sum(-1).cpu())
    return torch.cat(values)


def load_source_only_sam_relative_bundle(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_xyz: torch.Tensor,
    expected_valid: torch.Tensor,
    expected_canonical_radio_cache_sha256: str,
    expected_field_checkpoint_sha256: str,
    control_field: torch.nn.Module,
) -> SourceOnlySamRelativeBundle:
    manifest, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="source-only SAM relative manifest",
    )
    manifest = validate_source_only_sam_relative_manifest(manifest)
    base_record = manifest["base_structure_manifest"]
    base = load_source_only_sam_structure_bundle(
        base_record["path"],
        expected_sha256=base_record["sha256"],
        expected_xyz=expected_xyz,
        expected_valid=expected_valid,
        expected_canonical_radio_cache_sha256=expected_canonical_radio_cache_sha256,
        expected_field_checkpoint_sha256=expected_field_checkpoint_sha256,
    )
    relation, _relation_sha, _relation_source = load_torch_mapping(
        base.manifest["relation_cache"]["path"],
        expected_sha256=base.manifest["relation_cache"]["sha256"],
        map_location="cpu",
        label="source-only SAM relative relation cache",
    )
    graph, _graph_sha, _graph_source = load_torch_mapping(
        base.manifest["relation_graph"]["path"],
        expected_sha256=base.manifest["relation_graph"]["sha256"],
        map_location="cpu",
        label="source-only SAM relative relation graph",
    )
    global_edge = _global_relation_edges(
        relation, graph, expected_xyz=torch.as_tensor(expected_xyz).float().cpu()
    )
    triplet, triplet_weight, scale_bin = build_scale_matched_triplets(
        global_edge=global_edge,
        same_votes=torch.as_tensor(relation["same_votes"]),
        separate_votes=torch.as_tensor(relation["separate_votes"]),
        valid=torch.as_tensor(expected_valid),
    )
    pair_control = _edge_cosine(
        control_field, base.teacher.global_edge_index, SAM_STRUCTURE_BATCH_SIZE
    )
    positive_control = _edge_cosine(
        control_field, triplet[[0, 1]], SAM_STRUCTURE_BATCH_SIZE
    )
    negative_control = _edge_cosine(
        control_field, triplet[[0, 2]], SAM_STRUCTURE_BATCH_SIZE
    )
    teacher = SourceSamRelativeTeacher(
        pair_teacher=base.teacher,
        triplet_index=triplet,
        triplet_weight=triplet_weight,
        scale_bin=scale_bin,
        pair_control_cosine=pair_control,
        positive_control_cosine=positive_control,
        negative_control_cosine=negative_control,
    )
    return SourceOnlySamRelativeBundle(
        manifest=manifest,
        manifest_source=source,
        manifest_sha256=digest,
        base=base,
        teacher=teacher,
    )


def source_sam_relative_batch_loss(
    field: torch.nn.Module,
    *,
    teacher: SourceSamRelativeTeacher,
    triplet_indices: torch.Tensor,
    guard_edge_indices: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    triplet_selected = torch.as_tensor(triplet_indices).long().cpu().reshape(-1)
    guard_selected = torch.as_tensor(guard_edge_indices).long().cpu().reshape(-1)
    if triplet_selected.numel() == 0 or guard_selected.numel() == 0:
        raise ValueError("source-only SAM relative batches must contain both objectives")
    if (
        int(triplet_selected.min()) < 0
        or int(triplet_selected.max()) >= teacher.num_triplets
        or int(guard_selected.min()) < 0
        or int(guard_selected.max()) >= teacher.pair_teacher.num_edges
    ):
        raise ValueError("source-only SAM relative batch is outside its teacher axis")
    triplet = teacher.triplet_index[:, triplet_selected]
    guard_edge = teacher.pair_teacher.global_edge_index[:, guard_selected]
    all_rows = torch.cat([triplet.reshape(-1), guard_edge.reshape(-1)])
    unique, inverse = torch.unique(all_rows, sorted=True, return_inverse=True)
    device = field.local_codes.device
    radio = source_direction_only_radio(field.radio_features(unique.to(device)))
    direction = F.normalize(radio.float(), dim=-1, eps=1e-8)
    local_triplet_size = triplet.numel()
    local_triplet = inverse[:local_triplet_size].reshape_as(triplet).to(device)
    local_guard = inverse[local_triplet_size:].reshape_as(guard_edge).to(device)
    positive_cosine = (
        direction[local_triplet[0]] * direction[local_triplet[1]]
    ).sum(-1)
    negative_cosine = (
        direction[local_triplet[0]] * direction[local_triplet[2]]
    ).sum(-1)
    triplet_weight = teacher.triplet_weight[triplet_selected].to(device)
    ranking_values = F.relu(
        SAM_RELATIVE_MARGIN + negative_cosine - positive_cosine
    )
    ranking_loss = (ranking_values * triplet_weight).sum() / triplet_weight.sum()

    guard_cosine = (
        direction[local_guard[0]] * direction[local_guard[1]]
    ).sum(-1)
    control_guard = teacher.pair_control_cosine[guard_selected].to(device)
    same = teacher.pair_teacher.same_relation[guard_selected].to(device)
    guard_weight = teacher.pair_teacher.edge_weight[guard_selected].to(device)
    same_values = F.relu(control_guard - guard_cosine)
    separate_values = F.relu(guard_cosine - control_guard)

    def weighted(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not bool(mask.any()):
            return values.sum() * 0.0
        local_weight = guard_weight[mask]
        return (values[mask] * local_weight).sum() / local_weight.sum()

    same_guard = weighted(same_values, same)
    separate_guard = weighted(separate_values, ~same)
    guard_loss = 0.5 * (same_guard + separate_guard)
    loss = 0.5 * (ranking_loss + guard_loss)
    return loss, {
        "ranking_loss": ranking_loss.detach(),
        "same_guard_loss": same_guard.detach(),
        "separate_guard_loss": separate_guard.detach(),
        "triplets": torch.tensor(triplet_selected.numel(), device=device),
        "guard_edges": torch.tensor(guard_selected.numel(), device=device),
    }


@torch.no_grad()
def source_sam_relative_metrics(
    field: torch.nn.Module,
    *,
    teacher: SourceSamRelativeTeacher,
    batch_size: int = SAM_RELATIVE_TRIPLET_BATCH_SIZE,
) -> dict[str, float]:
    positive = _edge_cosine(field, teacher.triplet_index[[0, 1]], batch_size)
    negative = _edge_cosine(field, teacher.triplet_index[[0, 2]], batch_size)
    gap = positive - negative
    violation = gap < SAM_RELATIVE_MARGIN
    return {
        "sam_relative_triplets": teacher.num_triplets,
        "sam_relative_positive_mean_cosine": float(positive.mean()),
        "sam_relative_negative_mean_cosine": float(negative.mean()),
        "sam_relative_gap": float(gap.mean()),
        "sam_relative_violation_rate": float(violation.float().mean()),
        "sam_relative_hinge_mean": float(F.relu(SAM_RELATIVE_MARGIN - gap).mean()),
    }


def evaluate_source_sam_relative_gates(
    *,
    control_pair: Mapping[str, float],
    candidate_pair: Mapping[str, float],
    control_relative: Mapping[str, float],
    candidate_relative: Mapping[str, float],
) -> dict[str, Any]:
    pair = evaluate_source_sam_structure_gates(
        control=control_pair, candidate=candidate_pair
    )
    same_axis = int(control_relative.get("sam_relative_triplets", -1)) == int(
        candidate_relative.get("sam_relative_triplets", -2)
    ) and int(control_relative.get("sam_relative_triplets", 0)) > 0
    gap_pass = float(candidate_relative["sam_relative_gap"]) > float(
        control_relative["sam_relative_gap"]
    )
    violation_pass = float(candidate_relative["sam_relative_violation_rate"]) < float(
        control_relative["sam_relative_violation_rate"]
    )
    passed = (
        bool(pair["all_structure_gates_passed"])
        and same_axis
        and gap_pass
        and violation_pass
    )
    return {
        "global_pair_gates": pair,
        "same_triplet_axis": same_axis,
        "scale_triplet_gap_strict_improvement": gap_pass,
        "scale_triplet_violation_strict_decrease": violation_pass,
        "all_structure_gates_passed": passed,
        "benchmark_gate_opened": passed,
    }


def evaluate_source_sam_relative_full_gates(
    *,
    manifest: Mapping[str, Any],
    control_primary: Mapping[str, float],
    candidate_primary: Mapping[str, float],
    control_capability: Mapping[str, float],
    candidate_capability: Mapping[str, float],
    structure_decision: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_source_only_sam_relative_manifest(manifest)
    primary_rule = validated["source_gates"]["radio_reconstruction_no_regression"]
    capability_rule = validated["source_gates"][
        "official_capability_no_regression"
    ]
    primary_required = {"mean_cosine", "p05_cosine"}
    if not primary_required.issubset(control_primary) or not primary_required.issubset(
        candidate_primary
    ):
        raise ValueError("source-only SAM relative primary metrics are incomplete")
    primary_pass = (
        float(candidate_primary["mean_cosine"])
        >= float(control_primary["mean_cosine"])
        - float(primary_rule["mean_cosine_max_regression"])
        and float(candidate_primary["p05_cosine"])
        >= float(control_primary["p05_cosine"])
        - float(primary_rule["p05_cosine_max_regression"])
    )
    if set(control_capability) != set(candidate_capability) or not control_capability:
        raise ValueError("source-only SAM relative capability metric axis differs")
    capability_checks: dict[str, bool] = {}
    for name in sorted(control_capability):
        threshold_name = (
            "p05_cosine_max_regression"
            if "p05" in name
            else "mean_cosine_max_regression"
        )
        capability_checks[name] = float(candidate_capability[name]) >= float(
            control_capability[name]
        ) - float(capability_rule[threshold_name])
    capability_pass = all(capability_checks.values())
    structure_pass = bool(structure_decision.get("all_structure_gates_passed", False))
    passed = primary_pass and capability_pass and structure_pass
    return {
        **dict(structure_decision),
        "raw_radio_preservation": primary_pass,
        "official_capability_checks": capability_checks,
        "official_capability_preservation": capability_pass,
        "all_source_gates_passed": passed,
        "benchmark_gate_opened": passed,
    }


__all__ = [
    "SAM_RELATIVE_GUARD_BATCH_SIZE",
    "SAM_RELATIVE_MARGIN",
    "SAM_RELATIVE_TRIPLET_BATCH_SIZE",
    "SAM_RELATIVE_TRIPLET_CAP",
    "SOURCE_ONLY_SAM_RELATIVE_CONTRACT_SHA256",
    "SOURCE_ONLY_SAM_RELATIVE_SCHEMA",
    "SourceOnlySamRelativeBundle",
    "SourceSamRelativeTeacher",
    "build_scale_matched_triplets",
    "evaluate_source_sam_relative_gates",
    "evaluate_source_sam_relative_full_gates",
    "load_source_only_sam_relative_bundle",
    "source_only_sam_relative_contract",
    "source_sam_relative_batch_loss",
    "source_sam_relative_metrics",
    "source_sam_structure_metrics",
    "validate_source_only_sam_relative_manifest",
]
