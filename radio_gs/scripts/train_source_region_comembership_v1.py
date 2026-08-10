#!/usr/bin/env python3
"""Train the fixed source-only RegionCoMembershipV1 four-plus-two pilot."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.models.region_comembership_v1 import (
    PAIR_FEATURE_NAMES,
    RegionCoMembershipV1,
)
from radio_gs.scripts.build_source_region_comembership_v1 import (
    SCHEMA as AUTHORITY_SCHEMA,
    source_access as builder_source_access,
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
    "radio_gs.source_region_comembership_v1_execution_authority.v1"
)
CHECKPOINT_SCHEMA = "radio_gs.region_comembership_v1_checkpoint.v1"
TRAIN_SCENES = (
    "scene0001_00",
    "scene0002_00",
    "scene0003_00",
    "scene0005_00",
)
VALIDATION_SCENES = ("scene0004_00", "scene0008_00")
TOPOLOGY_SELECTION_ADDENDUM = Path(
    "paper/artifacts/source_only_region_comembership_v1_topology_consistent_selection_addendum_20260807.json"
)
TOPOLOGY_CHECKPOINT_ADDENDUM = Path(
    "paper/artifacts/source_only_region_comembership_v1_topology_checkpoint_selection_addendum_20260807.json"
)
SOURCE_PROMOTION_ADDENDUM = Path(
    "paper/artifacts/source_only_region_comembership_v1_source_promotion_addendum_20260807.json"
)
SEED = 0
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 100
PATIENCE = 12
THRESHOLDS = tuple(index / 100.0 for index in range(5, 100, 5))


@dataclass(frozen=True)
class SceneAuthority:
    scene_id: str
    split: str
    record: dict[str, str]
    pair_features: torch.Tensor
    pair_indices: torch.Tensor
    targets: torch.Tensor
    evidence_weights: torch.Tensor
    region_count: int
    dominant_instance_ids: torch.Tensor
    instance_purity: torch.Tensor
    instance_label_coverage: torch.Tensor
    instance_observed: torch.Tensor


def source_access() -> dict[str, bool]:
    return {
        **builder_source_access(),
        "source_validation_used_for_selection": True,
        "target_pair_features_opened": False,
    }


def training_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model": {
            "class": "RegionCoMembershipV1",
            "pair_features": list(PAIR_FEATURE_NAMES),
            "shared_scene_independent_linear_logistic_head": True,
            "scene_embedding": False,
            "query_input": False,
            "epoch_zero_probability": 0.5,
        },
        "cohort": {
            "source_train": list(TRAIN_SCENES),
            "source_validation": list(VALIDATION_SCENES),
        },
        "normalization": {
            "fit": "positive-evidence pair rows from source_train only",
            "median": True,
            "robust_scale": "1.4826_times_mad_or_one",
            "source_validation_contribution": False,
        },
        "objective": {
            "loss": "evidence_weighted_binary_cross_entropy_with_logits",
            "class_balance": "equal positive and negative mean within each scene",
            "scene_balance": "equal mean across source-train scenes",
        },
        "optimizer": {
            "name": "AdamW",
            "seed": SEED,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "patience": PATIENCE,
        },
        "selection": {
            "checkpoint": "maximum_source_validation_topology_score_at_each_epochs_best_fixed_grid_threshold",
            "epoch_zero_candidate": True,
            "threshold_grid": list(THRESHOLDS),
            "threshold_protocol_addendum": file_record(
                Path(__file__).resolve().parents[2] / TOPOLOGY_SELECTION_ADDENDUM
            ),
            "checkpoint_protocol_addendum": file_record(
                Path(__file__).resolve().parents[2] / TOPOLOGY_CHECKPOINT_ADDENDUM
            ),
            "source_promotion_addendum": file_record(
                Path(__file__).resolve().parents[2] / SOURCE_PROMOTION_ADDENDUM
            ),
            "threshold_metric": "source_validation_topology_score_equal_scene_equal_instance",
            "threshold_topology_score": "instance_macro_iou_minus_contamination_minus_giant_excess",
            "threshold_tie_break": "iou_then_f1_then_low_contamination_then_low_giant_excess_then_low_threshold",
            "pair_edge_weighted_f1": "diagnostic_only",
            "validation_bce": "diagnostic_and_checkpoint_tie_break_only",
        },
        "target_access_before_gate": False,
        "source_access": source_access(),
    }


TRAINING_CONTRACT_SHA256 = canonical_json_sha256(training_contract())


def _sha(value: object, *, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return digest


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must be an exact file record")
    return {
        "path": str(value["path"]),
        "sha256": _sha(value["sha256"], label=f"{label} SHA-256"),
    }


def validate_execution_authority(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("co-membership execution authority must be a mapping")
    authority = dict(value)
    required = {
        "schema",
        "schema_version",
        "status",
        "implementation",
        "preregistration",
        "source_train",
        "source_validation",
        "training_authorized",
        "target_execution_authorized",
        "source_access",
    }
    if set(authority) != required:
        raise ValueError("co-membership execution authority fields differ")
    if (
        authority.get("schema") != EXECUTION_AUTHORITY_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("status")
        != "authorized_after_complete_4train_2validation_source_preflight"
        or authority.get("training_authorized") is not True
        or authority.get("target_execution_authorized") is not False
        or authority.get("source_access") != source_access()
    ):
        raise ValueError("co-membership execution authority header differs")
    authority["implementation"] = _record(
        authority["implementation"], label="implementation"
    )
    authority["preregistration"] = _record(
        authority["preregistration"], label="preregistration"
    )
    for split, expected in (
        ("source_train", TRAIN_SCENES),
        ("source_validation", VALIDATION_SCENES),
    ):
        records = authority[split]
        if not isinstance(records, list) or len(records) != len(expected):
            raise ValueError(f"{split} co-membership scene count differs")
        normalized = []
        for item in records:
            if not isinstance(item, Mapping) or set(item) != {"scene_id", "authority"}:
                raise ValueError(f"{split} co-membership scene record differs")
            normalized.append(
                {
                    "scene_id": str(item["scene_id"]),
                    "authority": _record(
                        item["authority"], label=f"{split} scene authority"
                    ),
                }
            )
        if tuple(item["scene_id"] for item in normalized) != expected:
            raise ValueError(f"{split} differs from the fixed co-membership cohort")
        authority[split] = normalized
    return authority


def _authority_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    keys = {
        "schema",
        "schema_version",
        "scene_id",
        "split",
        "producer",
        "preregistration",
        "input_authority",
        "candidate_policy",
        "feature_names",
        "feature_names_sha256",
        "target_semantics",
        "source_access",
    }
    return {key: payload[key] for key in keys}


def load_scene_authority(
    record: Mapping[str, str], *, expected_scene_id: str, expected_split: str
) -> SceneAuthority:
    path = validate_file_record(record, label="source co-membership scene authority")
    payload, digest, source = load_torch_mapping(
        path,
        expected_sha256=record["sha256"],
        map_location="cpu",
        label="source region co-membership authority",
    )
    required = {
        "schema",
        "schema_version",
        "scene_id",
        "split",
        "producer",
        "preregistration",
        "input_authority",
        "candidate_policy",
        "feature_names",
        "feature_names_sha256",
        "target_semantics",
        "source_access",
        "content_authority_sha256",
        "region_fingerprints",
        "canonical_region_indices",
        "dominant_instance_ids",
        "dominant_instance_mass",
        "positive_instance_mass",
        "all_visible_mass",
        "instance_purity",
        "instance_label_coverage",
        "instance_observed",
        "pair_indices",
        "pair_features",
        "same_instance_targets",
        "pair_evidence_weights",
        "channel_sha256",
        "audit",
    }
    if set(payload) != required:
        raise ValueError("source co-membership authority fields differ")
    if (
        payload.get("schema") != AUTHORITY_SCHEMA
        or payload.get("schema_version") != 1
        or payload.get("scene_id") != expected_scene_id
        or payload.get("split") != expected_split
        or payload.get("feature_names") != list(PAIR_FEATURE_NAMES)
        or payload.get("feature_names_sha256")
        != canonical_json_sha256(list(PAIR_FEATURE_NAMES))
        or payload.get("source_access") != builder_source_access()
        or payload.get("content_authority_sha256")
        != canonical_json_sha256(_authority_identity(payload))
    ):
        raise ValueError("source co-membership authority identity differs")
    validate_file_record(
        payload["producer"], label="source co-membership builder implementation"
    )
    canonical = torch.as_tensor(payload["canonical_region_indices"])
    pairs = torch.as_tensor(payload["pair_indices"])
    features = torch.as_tensor(payload["pair_features"])
    targets = torch.as_tensor(payload["same_instance_targets"])
    weights = torch.as_tensor(payload["pair_evidence_weights"])
    dominant = torch.as_tensor(payload["dominant_instance_ids"])
    purity = torch.as_tensor(payload["instance_purity"])
    coverage = torch.as_tensor(payload["instance_label_coverage"])
    observed = torch.as_tensor(payload["instance_observed"])
    region_count = int(canonical.numel())
    if (
        canonical.dtype != torch.int64
        or canonical.ndim != 1
        or pairs.dtype != torch.int64
        or pairs.ndim != 2
        or pairs.shape[0] != 2
        or features.dtype != torch.float32
        or features.shape != (pairs.shape[1], len(PAIR_FEATURE_NAMES))
        or targets.dtype != torch.bool
        or targets.shape != (pairs.shape[1],)
        or weights.dtype != torch.float32
        or weights.shape != targets.shape
        or not bool(torch.isfinite(features).all())
        or not bool(torch.isfinite(weights).all())
        or bool((weights < 0).any())
        or bool((weights > 1 + 1e-6).any())
        or bool((pairs < 0).any())
        or bool((pairs >= region_count).any())
        or bool((pairs[0] >= pairs[1]).any())
        or dominant.dtype != torch.int64
        or dominant.shape != (region_count,)
        or purity.dtype != torch.float32
        or purity.shape != (region_count,)
        or coverage.dtype != torch.float32
        or coverage.shape != (region_count,)
        or observed.dtype != torch.bool
        or observed.shape != (region_count,)
        or not bool(torch.isfinite(purity).all())
        or not bool(torch.isfinite(coverage).all())
        or bool((purity < 0).any())
        or bool((purity > 1 + 1e-6).any())
        or bool((coverage < 0).any())
        or bool((coverage > 1 + 1e-6).any())
        or not torch.equal(observed, dominant > 0)
    ):
        raise ValueError("source co-membership pair tensors differ")
    pair_keys = pairs[0] * region_count + pairs[1]
    if pair_keys.numel() <= 0 or (
        pair_keys.numel() > 1 and not bool((pair_keys[1:] > pair_keys[:-1]).all())
    ):
        raise ValueError("source co-membership pairs are not sorted unique")
    channels = payload["channel_sha256"]
    if not isinstance(channels, Mapping):
        raise ValueError("source co-membership channel authority differs")
    for name, declared in channels.items():
        if name not in payload or not torch.is_tensor(payload[name]):
            raise ValueError("source co-membership channel name differs")
        if tensor_sha256(payload[name]) != declared:
            raise ValueError(f"source co-membership channel changed: {name}")
    if not bool((weights > 0).any()):
        raise ValueError("source co-membership scene has no labeled pair evidence")
    return SceneAuthority(
        scene_id=expected_scene_id,
        split=expected_split,
        record={"path": str(source), "sha256": digest},
        pair_features=features.contiguous(),
        pair_indices=pairs.contiguous(),
        targets=targets.contiguous(),
        evidence_weights=weights.contiguous(),
        region_count=region_count,
        dominant_instance_ids=dominant.contiguous(),
        instance_purity=purity.contiguous(),
        instance_label_coverage=coverage.contiguous(),
        instance_observed=observed.contiguous(),
    )


def prepare(
    path: str | Path, *, expected_sha256: str
) -> tuple[dict[str, Any], tuple[SceneAuthority, ...], tuple[SceneAuthority, ...]]:
    raw, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="co-membership execution authority",
    )
    authority = validate_execution_authority(raw)
    implementation = validate_file_record(
        authority["implementation"], label="co-membership trainer implementation"
    )
    if implementation != Path(__file__).resolve():
        raise ValueError("execution authority binds another trainer")
    validate_file_record(
        authority["preregistration"], label="co-membership preregistration"
    )

    def load(split: str, scenes: Sequence[str]) -> tuple[SceneAuthority, ...]:
        return tuple(
            load_scene_authority(
                item["authority"],
                expected_scene_id=scene,
                expected_split=split,
            )
            for item, scene in zip(authority[split], scenes)
        )

    authority["verified_path"] = str(source)
    authority["verified_sha256"] = digest
    return (
        authority,
        load("source_train", TRAIN_SCENES),
        load("source_validation", VALIDATION_SCENES),
    )


def fit_normalization(scenes: Sequence[SceneAuthority]) -> dict[str, Any]:
    selected = torch.cat(
        [scene.pair_features[scene.evidence_weights > 0] for scene in scenes]
    )
    median = selected.median(dim=0).values
    mad = (selected - median).abs().median(dim=0).values
    scale = torch.where(mad > 0, mad * 1.4826, torch.ones_like(mad))
    return {
        "schema": "radio_gs.region_comembership_v1_train_normalization.v1",
        "schema_version": 1,
        "fit_scene_ids": [scene.scene_id for scene in scenes],
        "fit_pair_rows": int(selected.shape[0]),
        "feature_names": list(PAIR_FEATURE_NAMES),
        "median": median.contiguous(),
        "mad": mad.contiguous(),
        "robust_scale": scale.contiguous(),
        "validation_contribution": False,
    }


def balanced_scene_loss(
    logits: torch.Tensor, targets: torch.Tensor, evidence_weights: torch.Tensor
) -> torch.Tensor:
    values = torch.as_tensor(logits)
    labels = torch.as_tensor(targets, device=values.device).bool()
    weights = torch.as_tensor(evidence_weights, device=values.device).float()
    if (
        values.ndim != 1
        or labels.shape != values.shape
        or weights.shape != values.shape
        or not bool(torch.isfinite(values).all())
        or not bool(torch.isfinite(weights).all())
        or bool((weights < 0).any())
    ):
        raise ValueError("co-membership scene loss inputs differ")
    active = weights > 0
    positive = active & labels
    negative = active & ~labels
    if not bool(positive.any()) or not bool(negative.any()):
        raise ValueError("co-membership scene needs positive and negative evidence")
    units = F.binary_cross_entropy_with_logits(values, labels.float(), reduction="none")
    positive_loss = (units[positive] * weights[positive]).sum() / weights[
        positive
    ].sum()
    negative_loss = (units[negative] * weights[negative]).sum() / weights[
        negative
    ].sum()
    return 0.5 * (positive_loss + negative_loss)


def evaluate(
    model: RegionCoMembershipV1,
    scenes: Sequence[SceneAuthority],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    per_scene: dict[str, Any] = {}
    with torch.no_grad():
        for scene in scenes:
            logits = model(scene.pair_features.to(device))
            loss = balanced_scene_loss(
                logits, scene.targets.to(device), scene.evidence_weights.to(device)
            )
            per_scene[scene.scene_id] = {
                "balanced_weighted_bce": float(loss.cpu()),
                "positive_pairs": int(
                    ((scene.evidence_weights > 0) & scene.targets).sum()
                ),
                "negative_pairs": int(
                    ((scene.evidence_weights > 0) & ~scene.targets).sum()
                ),
            }
    return {
        "scene_macro_balanced_weighted_bce": sum(
            row["balanced_weighted_bce"] for row in per_scene.values()
        )
        / len(per_scene),
        "per_scene": per_scene,
        "validation_no_grad": True,
        "benchmark_opened": False,
    }


def select_best_epoch(history: Sequence[Mapping[str, Any]]) -> int:
    if not history or [int(row.get("epoch", -1)) for row in history] != list(
        range(len(history))
    ):
        raise ValueError("co-membership history must be contiguous from epoch zero")
    return int(
        max(
            history,
            key=lambda row: (
                float(
                    row["validation_topology"]["selected"][
                        "scene_macro_instance_macro"
                    ]["topology_score"]
                ),
                float(
                    row["validation_topology"]["selected"][
                        "scene_macro_instance_macro"
                    ]["iou"]
                ),
                float(
                    row["validation_topology"]["selected"][
                        "scene_macro_instance_macro"
                    ]["f1"]
                ),
                -float(
                    row["validation_topology"]["selected"][
                        "scene_macro_instance_macro"
                    ]["contamination"]
                ),
                -float(
                    row["validation_topology"]["selected"][
                        "scene_macro_instance_macro"
                    ]["giant_excess"]
                ),
                -float(row["validation"]["scene_macro_balanced_weighted_bce"]),
                -int(row["epoch"]),
            ),
        )["epoch"]
    )


def _weighted_f1(
    probability: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    threshold: float,
) -> float:
    active = weight > 0
    prediction = probability >= float(threshold)
    true_positive = weight[active & prediction & target].sum()
    false_positive = weight[active & prediction & ~target].sum()
    false_negative = weight[active & ~prediction & target].sum()
    denominator = 2 * true_positive + false_positive + false_negative
    return float((2 * true_positive / denominator.clamp_min(1e-12)).cpu())


def component_scene_metrics(
    *,
    pair_indices: torch.Tensor,
    pair_probabilities: torch.Tensor,
    dominant_instance_ids: torch.Tensor,
    instance_purity: torch.Tensor,
    instance_label_coverage: torch.Tensor,
    instance_observed: torch.Tensor,
    threshold: float,
) -> dict[str, Any]:
    """Evaluate the transitive graph closure from every labeled region seed."""

    pairs = torch.as_tensor(pair_indices).detach().long().cpu()
    probability = torch.as_tensor(pair_probabilities).detach().float().cpu()
    dominant = torch.as_tensor(dominant_instance_ids).detach().long().cpu()
    purity = torch.as_tensor(instance_purity).detach().float().cpu()
    coverage = torch.as_tensor(instance_label_coverage).detach().float().cpu()
    observed = torch.as_tensor(instance_observed).detach().bool().cpu()
    count = int(dominant.numel())
    cutoff = float(threshold)
    if (
        count <= 0
        or pairs.ndim != 2
        or pairs.shape[0] != 2
        or probability.shape != (pairs.shape[1],)
        or purity.shape != (count,)
        or coverage.shape != (count,)
        or observed.shape != (count,)
        or not bool(torch.isfinite(probability).all())
        or not bool(torch.isfinite(purity).all())
        or not bool(torch.isfinite(coverage).all())
        or bool((pairs < 0).any())
        or bool((pairs >= count).any())
        or bool((pairs[0] >= pairs[1]).any())
        or not 0 <= cutoff <= 1
    ):
        raise ValueError("co-membership component metric inputs differ")
    evidence = purity * coverage
    eligible = observed & (dominant > 0) & (evidence > 0)
    if not bool(eligible.any()):
        raise ValueError("co-membership component metric has no eligible regions")

    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return
        if root_left > root_right:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left

    for left, right in pairs[:, probability >= cutoff].T.tolist():
        union(int(left), int(right))

    component_total: dict[int, float] = {}
    component_instance: dict[tuple[int, int], float] = {}
    target_total: dict[int, float] = {}
    eligible_rows = torch.nonzero(eligible, as_tuple=False).flatten().tolist()
    for row in eligible_rows:
        root = find(row)
        instance = int(dominant[row])
        weight = float(evidence[row])
        component_total[root] = component_total.get(root, 0.0) + weight
        key = (root, instance)
        component_instance[key] = component_instance.get(key, 0.0) + weight
        target_total[instance] = target_total.get(instance, 0.0) + weight
    scene_total = sum(target_total.values())
    by_instance: dict[int, dict[str, float]] = {}
    for instance in sorted(target_total):
        seed_weight = 0.0
        sums = {"iou": 0.0, "f1": 0.0, "contamination": 0.0, "giant_excess": 0.0}
        target_weight = target_total[instance]
        for row in eligible_rows:
            if int(dominant[row]) != instance:
                continue
            root = find(row)
            weight = float(evidence[row])
            selected = component_total[root]
            correct = component_instance[(root, instance)]
            union_weight = target_weight + selected - correct
            values = {
                "iou": correct / max(union_weight, 1e-12),
                "f1": 2.0 * correct / max(target_weight + selected, 1e-12),
                "contamination": (selected - correct) / max(selected, 1e-12),
                "giant_excess": max(
                    0.0, selected / scene_total - target_weight / scene_total
                ),
            }
            seed_weight += weight
            for name, value in values.items():
                sums[name] += weight * value
        by_instance[instance] = {
            name: value / max(seed_weight, 1e-12) for name, value in sums.items()
        }
    macro = {
        name: sum(row[name] for row in by_instance.values()) / len(by_instance)
        for name in ("iou", "f1", "contamination", "giant_excess")
    }
    macro["topology_score"] = (
        macro["iou"] - macro["contamination"] - macro["giant_excess"]
    )
    return {
        "instance_macro": macro,
        "per_instance": {str(key): value for key, value in by_instance.items()},
        "eligible_regions": len(eligible_rows),
        "eligible_instances": len(by_instance),
        "selected_edges": int((probability >= cutoff).sum()),
        "largest_predicted_component_scene_evidence_fraction": max(
            component_total.values()
        )
        / scene_total,
    }


def select_validation_threshold_from_probabilities(
    scenes: Sequence[SceneAuthority],
    probabilities: Mapping[str, torch.Tensor],
    *,
    include_candidate_oracle: bool = True,
) -> dict[str, Any]:
    if not scenes or set(probabilities) != {scene.scene_id for scene in scenes}:
        raise ValueError("co-membership validation probabilities differ")
    rows = []
    for threshold in THRESHOLDS:
        per_scene = {}
        for scene in scenes:
            component = component_scene_metrics(
                pair_indices=scene.pair_indices,
                pair_probabilities=probabilities[scene.scene_id],
                dominant_instance_ids=scene.dominant_instance_ids,
                instance_purity=scene.instance_purity,
                instance_label_coverage=scene.instance_label_coverage,
                instance_observed=scene.instance_observed,
                threshold=threshold,
            )
            component["diagnostic_pair_edge_weighted_f1"] = _weighted_f1(
                probabilities[scene.scene_id],
                scene.targets,
                scene.evidence_weights,
                threshold,
            )
            per_scene[scene.scene_id] = component
        macro = {
            name: sum(
                scene_row["instance_macro"][name] for scene_row in per_scene.values()
            )
            / len(per_scene)
            for name in ("iou", "f1", "contamination", "giant_excess", "topology_score")
        }
        macro["diagnostic_pair_edge_weighted_f1"] = sum(
            scene_row["diagnostic_pair_edge_weighted_f1"]
            for scene_row in per_scene.values()
        ) / len(per_scene)
        rows.append(
            {
                "threshold": threshold,
                "scene_macro_instance_macro": macro,
                "per_scene": per_scene,
            }
        )
    selected = max(
        rows,
        key=lambda row: (
            float(row["scene_macro_instance_macro"]["topology_score"]),
            float(row["scene_macro_instance_macro"]["iou"]),
            float(row["scene_macro_instance_macro"]["f1"]),
            -float(row["scene_macro_instance_macro"]["contamination"]),
            -float(row["scene_macro_instance_macro"]["giant_excess"]),
            -float(row["threshold"]),
        ),
    )
    diagnostic_pair_selected = max(
        rows,
        key=lambda row: (
            float(
                row["scene_macro_instance_macro"]["diagnostic_pair_edge_weighted_f1"]
            ),
            -float(row["threshold"]),
        ),
    )
    result = {
        "selected": selected,
        "diagnostic_pair_edge_f1_selected_threshold": diagnostic_pair_selected[
            "threshold"
        ],
        "grid": rows,
        "selection_metric": "topology_score",
        "pair_edge_weighted_f1_selection_role": "diagnostic_only",
        "candidate_graph_oracle_ceiling": None,
        "benchmark_opened": False,
    }
    if include_candidate_oracle:
        oracle_per_scene = {
            scene.scene_id: component_scene_metrics(
                pair_indices=scene.pair_indices,
                pair_probabilities=scene.targets.float(),
                dominant_instance_ids=scene.dominant_instance_ids,
                instance_purity=scene.instance_purity,
                instance_label_coverage=scene.instance_label_coverage,
                instance_observed=scene.instance_observed,
                threshold=0.5,
            )
            for scene in scenes
        }
        oracle_macro = {
            name: sum(
                row["instance_macro"][name] for row in oracle_per_scene.values()
            )
            / len(oracle_per_scene)
            for name in (
                "iou",
                "f1",
                "contamination",
                "giant_excess",
                "topology_score",
            )
        }
        result["candidate_graph_oracle_ceiling"] = {
            "scene_macro_instance_macro": oracle_macro,
            "per_scene": oracle_per_scene,
        }
    return result


def select_validation_threshold(
    model: RegionCoMembershipV1,
    scenes: Sequence[SceneAuthority],
    device: torch.device,
    *,
    include_candidate_oracle: bool = True,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        probability = {
            scene.scene_id: model.probability(scene.pair_features.to(device)).cpu()
            for scene in scenes
        }
    return select_validation_threshold_from_probabilities(
        scenes,
        probability,
        include_candidate_oracle=include_candidate_oracle,
    )


def _state_copy(model: RegionCoMembershipV1) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().contiguous().clone()
        for name, value in model.state_dict().items()
    }


def _state_sha(state: Mapping[str, torch.Tensor]) -> str:
    return canonical_json_sha256(
        {name: tensor_sha256(value) for name, value in sorted(state.items())}
    )


def _checkpoint_topology_summary(selection: Mapping[str, Any]) -> dict[str, Any]:
    selected = selection["selected"]
    return {
        "selected": {
            "threshold": float(selected["threshold"]),
            "scene_macro_instance_macro": dict(selected["scene_macro_instance_macro"]),
        },
        "diagnostic_pair_edge_f1_selected_threshold": float(
            selection["diagnostic_pair_edge_f1_selected_threshold"]
        ),
        "selection_metric": "topology_score",
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    report_path = output.with_suffix(output.suffix + ".json")
    if (
        output.exists()
        or output.is_symlink()
        or report_path.exists()
        or report_path.is_symlink()
    ):
        raise FileExistsError("co-membership checkpoint/report output must be new")
    execution, train_scenes, validation_scenes = prepare(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
    )
    normalization = fit_normalization(train_scenes)
    device = torch.device(str(args.device))
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    model = RegionCoMembershipV1(
        normalization["median"], normalization["robust_scale"]
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    epoch_zero = evaluate(model, validation_scenes, device)
    epoch_zero_topology = _checkpoint_topology_summary(
        select_validation_threshold(
            model,
            validation_scenes,
            device,
            include_candidate_oracle=False,
        )
    )
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "training": None,
            "validation": epoch_zero,
            "validation_topology": epoch_zero_topology,
            "state_dict_sha256": _state_sha(_state_copy(model)),
        }
    ]
    best_epoch = 0
    best_state = _state_copy(model)
    stale = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        scene_losses = []
        for scene in train_scenes:
            logits = model(scene.pair_features.to(device))
            loss = balanced_scene_loss(
                logits, scene.targets.to(device), scene.evidence_weights.to(device)
            )
            (loss / len(train_scenes)).backward()
            scene_losses.append(float(loss.detach().cpu()))
        optimizer.step()
        validation = evaluate(model, validation_scenes, device)
        validation_topology = _checkpoint_topology_summary(
            select_validation_threshold(
                model,
                validation_scenes,
                device,
                include_candidate_oracle=False,
            )
        )
        row = {
            "epoch": epoch,
            "training": {
                "scene_macro_balanced_weighted_bce": sum(scene_losses)
                / len(scene_losses),
                "equal_scene_weight": 1.0 / len(train_scenes),
            },
            "validation": validation,
            "validation_topology": validation_topology,
            "state_dict_sha256": _state_sha(_state_copy(model)),
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
        if stale >= PATIENCE:
            break
    selected_epoch = select_best_epoch(history)
    if selected_epoch != best_epoch:
        raise RuntimeError("co-membership selected and retained epochs differ")
    model.load_state_dict(best_state, strict=True)
    selected_validation = evaluate(model, validation_scenes, device)
    threshold = select_validation_threshold(model, validation_scenes, device)
    epoch_zero_score = float(
        history[0]["validation_topology"]["selected"]["scene_macro_instance_macro"][
            "topology_score"
        ]
    )
    selected_score = float(
        threshold["selected"]["scene_macro_instance_macro"]["topology_score"]
    )
    promotion_gate = {
        "selected_epoch_positive": selected_epoch > 0,
        "selected_topology_score_strictly_exceeds_epoch_zero": selected_score
        > epoch_zero_score,
        "epoch_zero_topology_score": epoch_zero_score,
        "selected_topology_score": selected_score,
    }
    promotion_gate["passed"] = bool(
        promotion_gate["selected_epoch_positive"]
        and promotion_gate["selected_topology_score_strictly_exceeds_epoch_zero"]
    )
    model.cpu()
    state = _state_copy(model)
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": 1,
        "training_contract": training_contract(),
        "training_contract_sha256": TRAINING_CONTRACT_SHA256,
        "execution_authority": {
            "path": execution["verified_path"],
            "sha256": execution["verified_sha256"],
        },
        "feature_names": list(PAIR_FEATURE_NAMES),
        "model_state_dict": state,
        "model_state_dict_sha256": _state_sha(state),
        "selected_epoch": selected_epoch,
        "epoch_zero_validation_topology": history[0]["validation_topology"],
        "selected_validation": selected_validation,
        "selected_validation_topology": _checkpoint_topology_summary(threshold),
        "promotion_gate": promotion_gate,
        "selected_probability_threshold": threshold["selected"]["threshold"],
        "threshold_selection": threshold,
        "source_access": source_access(),
        "target_execution_performed": False,
    }
    checkpoint_path = write_torch_noclobber(output, checkpoint)
    report = {
        "schema": "radio_gs.region_comembership_v1_pilot_result.v1",
        "schema_version": 1,
        "status": "source_only_4train_2validation_pilot_complete",
        "checkpoint": file_record(checkpoint_path),
        "selected_epoch": selected_epoch,
        "automatic_epoch_zero_fallback": selected_epoch == 0,
        "epoch_zero_validation_topology": history[0]["validation_topology"],
        "selected_validation": selected_validation,
        "selected_validation_topology": _checkpoint_topology_summary(threshold),
        "promotion_gate": promotion_gate,
        "threshold_selection": threshold,
        "history": history,
        "source_access": source_access(),
        "target_execution_performed": False,
    }
    write_frozen_json(report_path, report)
    return report


def synthetic_dry_run() -> dict[str, Any]:
    features = torch.arange(90, dtype=torch.float32).reshape(6, 15) / 90
    target = torch.tensor([True, False, True, False, True, False])
    weight = torch.tensor([1.0, 1.0, 0.8, 0.9, 0.7, 0.6])
    model = RegionCoMembershipV1(torch.zeros(15), torch.ones(15))
    epoch_zero = model(features)
    loss = balanced_scene_loss(epoch_zero, target, weight)
    loss.backward()
    return {
        "schema": "radio_gs.region_comembership_v1_synthetic_train_dry_run.v1",
        "pair_rows": 6,
        "epoch_zero_probability": float(torch.sigmoid(epoch_zero).mean()),
        "balanced_weighted_bce": float(loss.detach()),
        "gradient_nonzero": bool(model.logit.weight.grad.abs().sum() > 0),
        "threshold_grid": list(THRESHOLDS),
        "benchmark_opened": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("synthetic-dry-run")
    run = commands.add_parser("train")
    run.add_argument("--execution-authority", required=True)
    run.add_argument("--expected-execution-authority-sha256", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = synthetic_dry_run() if args.command == "synthetic-dry-run" else train(args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
