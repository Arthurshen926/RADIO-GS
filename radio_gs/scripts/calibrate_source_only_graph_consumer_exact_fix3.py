#!/usr/bin/env python3
"""Recalibrate graph null gains with the exact residual-consumer recurrence.

FIX2 used the correct scalar gain units, but its second and third null order
statistics assumed a full 256-row core at every step.  This source-only FIX3
replays the consumer's covered-set update and marginal primitive de-duplication
exactly.  It also measures true-edge reach and mixed-edge precision without
opening target data, benchmark queries, masks, images, or metrics.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
import json
import math
from pathlib import Path
from typing import Any

import torch

from radio_gs.scripts import calibrate_source_only_graph_confidence_v1 as fix2
from radio_gs.scripts.train_source_region_comembership_v2 import SceneAuthorityV2
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    validate_file_record,
    write_frozen_json,
)


AUTHORITY_SCHEMA = (
    "radio_gs.source_only_graph_consumer_exact_fix3_execution_authority.v1"
)
RESULT_SCHEMA = "radio_gs.source_only_graph_consumer_exact_calibration_fix3.v1"
STEPS = 3
NULL_QUANTILE = 0.99
NOVEL_MASS_REFERENCE = 256.0


def source_access() -> dict[str, bool]:
    return dict(fix2.source_access())


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must be an exact file record")
    path = validate_file_record(value, label=label)
    return {"path": str(path), "sha256": str(value["sha256"])}


def validate_execution_authority(value: object) -> dict[str, Any]:
    required = {
        "schema",
        "schema_version",
        "status",
        "implementation",
        "fix2_execution_authority",
        "fix2_result",
        "calibration_change",
        "source_access",
        "benchmark_execution_authorized",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("FIX3 execution authority fields differ")
    authority = dict(value)
    if (
        authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("status") != "authorized_source_only_consumer_exact_fix3"
        or authority.get("source_access") != source_access()
        or authority.get("benchmark_execution_authorized") is not False
        or authority.get("calibration_change")
        != {
            "gain": "(2*calibrated_edge_lower-1)_plus*marginal_unique_primitives/256",
            "covered_update": "exact_union_after_each_selected_candidate",
            "null_population": (
                "different_instance_direct_edges_after_raw_p_reliability_and_"
                "pair_feature_OOD_gates"
            ),
            "aggregation": "per_scene_q99_then_max_across_six_scenes",
            "steps": STEPS,
            "unchanged": [
                "epsilon_logit",
                "raw_edge_probability_minimum",
                "O0_anchor_contract",
                "pair_reliability_gate",
                "pair_feature_OOD_gate",
            ],
        }
    ):
        raise ValueError("FIX3 execution authority header differs")
    for name in ("implementation", "fix2_execution_authority", "fix2_result"):
        authority[name] = _record(authority[name], label=name)
    return authority


def exact_direct_edge_trace(
    *,
    scene: SceneAuthorityV2,
    probability_lower: torch.Tensor,
    edge_eligible_mask: torch.Tensor,
    target_filter: bool | None,
    steps: int = STEPS,
    novel_mass_reference: float = NOVEL_MASS_REFERENCE,
) -> dict[str, torch.Tensor]:
    """Run the residual consumer's greedy marginal recurrence per source anchor.

    ``target_filter=False`` is the null replay, ``True`` is the true-only
    capability audit, and ``None`` is the deployable mixed candidate replay.
    The source anchor itself is not a residual candidate.
    """

    lower = torch.as_tensor(probability_lower).detach().float().cpu()
    eligible_edge = torch.as_tensor(edge_eligible_mask).detach().bool().cpu()
    if (
        lower.shape != scene.targets.shape
        or eligible_edge.shape != scene.targets.shape
        or int(steps) <= 0
        or float(novel_mass_reference) <= 0.0
        or not bool(torch.isfinite(lower).all())
        or bool((lower < 0.0).any())
        or bool((lower > 1.0).any())
    ):
        raise ValueError("consumer-exact trace inputs differ")
    adjacency: list[list[tuple[int, int]]] = [
        [] for _ in range(scene.region_count)
    ]
    for edge_index in range(int(scene.pair_indices.shape[1])):
        if not bool(eligible_edge[edge_index]):
            continue
        is_true = bool(scene.targets[edge_index])
        if target_filter is not None and is_true is not target_filter:
            continue
        left = int(scene.pair_indices[0, edge_index])
        right = int(scene.pair_indices[1, edge_index])
        adjacency[left].append((right, edge_index))
        adjacency[right].append((left, edge_index))
    gains = torch.zeros((scene.region_count, int(steps)), dtype=torch.float32)
    labels = torch.zeros((scene.region_count, int(steps)), dtype=torch.bool)
    candidate_rows = torch.full(
        (scene.region_count, int(steps)), -1, dtype=torch.int64
    )
    marginal_primitives = torch.zeros(
        (scene.region_count, int(steps)), dtype=torch.int64
    )
    has_candidate = torch.tensor([bool(row) for row in adjacency], dtype=torch.bool)
    confidence_excess = (2.0 * lower - 1.0).clamp_min(0.0)
    for anchor, edges in enumerate(adjacency):
        covered: set[int] = set()
        used: set[int] = set()
        for step in range(int(steps)):
            best: tuple[float, int, int, int] | None = None
            for candidate, edge_index in edges:
                if candidate in used:
                    continue
                core = scene.core_rows[candidate].tolist()
                marginal = sum(int(row) not in covered for row in core)
                gain = float(confidence_excess[edge_index]) * (
                    marginal / float(novel_mass_reference)
                )
                # Consumer tie break is the lower canonical region index.
                key = (gain, -candidate)
                if best is None or key > (best[0], -best[1]):
                    best = (gain, candidate, edge_index, marginal)
            if best is None or best[0] <= 0.0:
                break
            gain, candidate, edge_index, marginal = best
            gains[anchor, step] = gain
            labels[anchor, step] = bool(scene.targets[edge_index])
            candidate_rows[anchor, step] = candidate
            marginal_primitives[anchor, step] = marginal
            used.add(candidate)
            covered.update(int(row) for row in scene.core_rows[candidate].tolist())
    return {
        "gains": gains.contiguous(),
        "labels": labels.contiguous(),
        "candidate_rows": candidate_rows.contiguous(),
        "marginal_primitives": marginal_primitives.contiguous(),
        "has_candidate": has_candidate.contiguous(),
    }


def apply_strict_sequential_thresholds(
    gains: torch.Tensor, thresholds: Sequence[float]
) -> torch.Tensor:
    values = torch.as_tensor(gains).detach().float().cpu()
    limits = torch.tensor(tuple(float(v) for v in thresholds), dtype=torch.float32)
    if (
        values.ndim != 2
        or values.shape[1] != limits.numel()
        or limits.numel() <= 0
        or not bool(torch.isfinite(values).all())
        or not bool(torch.isfinite(limits).all())
        or bool((values < 0.0).any())
        or bool((limits < 0.0).any())
    ):
        raise ValueError("sequential threshold inputs differ")
    selected = torch.zeros_like(values, dtype=torch.bool)
    alive = torch.ones(values.shape[0], dtype=torch.bool)
    for step in range(values.shape[1]):
        selected[:, step] = alive & (values[:, step] > limits[step])
        alive &= selected[:, step]
    return selected.contiguous()


def _raw_probability_required(
    *, threshold: float, epsilon_logit: float, full_core_fraction: float = 1.0
) -> float:
    gain = float(threshold)
    fraction = float(full_core_fraction)
    if gain < 0.0 or not 0.0 < fraction <= 1.0:
        raise ValueError("required probability inputs differ")
    excess = gain / fraction
    if excess >= 1.0:
        return 1.0
    lower = (1.0 + excess) / 2.0
    return fix2.expit(fix2.logit(lower) + float(epsilon_logit))


def _selection_audit(
    *,
    trace: Mapping[str, torch.Tensor],
    selected: torch.Tensor,
) -> dict[str, Any]:
    has = trace["has_candidate"]
    labels = trace["labels"]
    eligible = int(has.sum())
    selected_count = int(selected.sum())
    selected_positive = int((selected & labels).sum())
    any_selected = selected.any(dim=1)
    return {
        "eligible_anchor_count": eligible,
        "anchor_with_any_selected_count": int((any_selected & has).sum()),
        "anchor_with_any_selected_rate": float(
            (any_selected[has]).float().mean() if eligible else 0.0
        ),
        "selected_edge_count": selected_count,
        "selected_positive_edge_count": selected_positive,
        "selected_edge_precision": (
            selected_positive / selected_count if selected_count else None
        ),
        "selected_count_by_step": [
            int(selected[:, step].sum()) for step in range(selected.shape[1])
        ],
        "mean_marginal_primitives_by_selected_step": [
            (
                float(
                    trace["marginal_primitives"][selected[:, step], step]
                    .float()
                    .mean()
                )
                if bool(selected[:, step].any())
                else None
            )
            for step in range(selected.shape[1])
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"FIX3 source result exists: {output}")
    raw, authority_sha, authority_path = load_json_object(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
        label="FIX3 source execution authority",
    )
    authority = validate_execution_authority(raw)
    if authority["implementation"] != file_record(Path(__file__).resolve()):
        raise ValueError("FIX3 implementation binding differs")
    fix2_authority_raw, _, _ = load_json_object(
        authority["fix2_execution_authority"]["path"],
        expected_sha256=authority["fix2_execution_authority"]["sha256"],
        label="FIX3-bound FIX2 execution authority",
    )
    fix2_authority = fix2.validate_execution_authority(fix2_authority_raw)
    fix2_result, _, _ = load_json_object(
        authority["fix2_result"]["path"],
        expected_sha256=authority["fix2_result"]["sha256"],
        label="FIX3-bound FIX2 result",
    )
    if (
        fix2_result.get("schema") != fix2.RESULT_SCHEMA
        or fix2_result.get("execution_authority")
        != authority["fix2_execution_authority"]
        or fix2_result.get("target_execution_performed") is not False
        or fix2_result.get("source_access") != source_access()
    ):
        raise ValueError("FIX3 parent result binding differs")
    checkpoint, loaded = fix2._load_inputs(fix2_authority)
    epsilon = float(
        fix2_result["deployment_config"]["residual_config"]["epsilon_logit"]
    )
    traces: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    per_scene_null_thresholds: dict[str, list[float]] = {}
    edge_gate_audit: dict[str, dict[str, int]] = {}
    reliability_minimum = float(
        fix2_result["deployment_config"]["residual_config"][
            "minimum_reliability"
        ]
    )
    ood_raw_limit = float(
        fix2_result["deployment_config"]["feature_ood"]["raw_score_limit"]
    )
    median = torch.as_tensor(checkpoint["normalization"]["median"]).float()
    robust_scale = torch.as_tensor(
        checkpoint["normalization"]["robust_scale"]
    ).float()
    for scene, probability in loaded:
        lower = fix2.lower_probability(probability, epsilon)
        reliability = torch.minimum(
            scene.pair_features[:, 17], scene.pair_features[:, 18]
        )
        ood_raw = (
            (scene.pair_features - median) / robust_scale
        ).abs().max(dim=1).values
        edge_eligible = (
            (probability >= fix2.RAW_EDGE_PROBABILITY_MINIMUM)
            & (reliability >= reliability_minimum)
            & (ood_raw <= ood_raw_limit)
        )
        edge_gate_audit[scene.scene_id] = {
            "all_edge_count": int(scene.targets.numel()),
            "all_true_edge_count": int(scene.targets.sum()),
            "all_false_edge_count": int((~scene.targets).sum()),
            "raw_p_ge_0.9_edge_count": int(
                (probability >= fix2.RAW_EDGE_PROBABILITY_MINIMUM).sum()
            ),
            "eligible_edge_count": int(edge_eligible.sum()),
            "eligible_true_edge_count": int((edge_eligible & scene.targets).sum()),
            "eligible_false_edge_count": int((edge_eligible & ~scene.targets).sum()),
        }
        false_trace = exact_direct_edge_trace(
            scene=scene,
            probability_lower=lower,
            edge_eligible_mask=edge_eligible,
            target_filter=False,
        )
        true_trace = exact_direct_edge_trace(
            scene=scene,
            probability_lower=lower,
            edge_eligible_mask=edge_eligible,
            target_filter=True,
        )
        mixed_trace = exact_direct_edge_trace(
            scene=scene,
            probability_lower=lower,
            edge_eligible_mask=edge_eligible,
            target_filter=None,
        )
        traces[scene.scene_id] = {
            "false": false_trace,
            "true": true_trace,
            "mixed": mixed_trace,
        }
        per_scene_null_thresholds[scene.scene_id] = [
            float(torch.quantile(false_trace["gains"][:, step], NULL_QUANTILE))
            for step in range(STEPS)
        ]
    thresholds = tuple(
        max(row[step] for row in per_scene_null_thresholds.values())
        for step in range(STEPS)
    )
    per_scene_audit: dict[str, Any] = {}
    pooled = {
        "true_eligible": 0,
        "true_anchor_selected": 0,
        "true_selected": 0,
        "mixed_selected": 0,
        "mixed_positive": 0,
    }
    pooled_by_split = {
        split: {key: 0 for key in pooled}
        for split in ("source_train", "source_validation")
    }
    for scene, _ in loaded:
        scene_trace = traces[scene.scene_id]
        true_selected = apply_strict_sequential_thresholds(
            scene_trace["true"]["gains"], thresholds
        )
        mixed_selected = apply_strict_sequential_thresholds(
            scene_trace["mixed"]["gains"], thresholds
        )
        false_selected = apply_strict_sequential_thresholds(
            scene_trace["false"]["gains"], thresholds
        )
        true_audit = _selection_audit(
            trace=scene_trace["true"], selected=true_selected
        )
        mixed_audit = _selection_audit(
            trace=scene_trace["mixed"], selected=mixed_selected
        )
        false_audit = _selection_audit(
            trace=scene_trace["false"], selected=false_selected
        )
        per_scene_audit[scene.scene_id] = {
            "split": scene.split,
            "true_edge_capability": true_audit,
            "mixed_edge_selection": mixed_audit,
            "false_edge_null_replay": false_audit,
        }
        pooled["true_eligible"] += true_audit["eligible_anchor_count"]
        pooled["true_anchor_selected"] += true_audit[
            "anchor_with_any_selected_count"
        ]
        pooled["true_selected"] += true_audit["selected_edge_count"]
        pooled["mixed_selected"] += mixed_audit["selected_edge_count"]
        pooled["mixed_positive"] += mixed_audit["selected_positive_edge_count"]
        split_pool = pooled_by_split[scene.split]
        split_pool["true_eligible"] += true_audit["eligible_anchor_count"]
        split_pool["true_anchor_selected"] += true_audit[
            "anchor_with_any_selected_count"
        ]
        split_pool["true_selected"] += true_audit["selected_edge_count"]
        split_pool["mixed_selected"] += mixed_audit["selected_edge_count"]
        split_pool["mixed_positive"] += mixed_audit[
            "selected_positive_edge_count"
        ]
    pooled_audit = {
        **pooled,
        "true_anchor_with_any_selected_rate": pooled["true_anchor_selected"]
        / max(pooled["true_eligible"], 1),
        "mixed_selected_edge_precision": pooled["mixed_positive"]
        / max(pooled["mixed_selected"], 1),
    }
    split_audit = {
        split: {
            **row,
            "true_anchor_with_any_selected_rate": row["true_anchor_selected"]
            / max(row["true_eligible"], 1),
            "mixed_selected_edge_precision": row["mixed_positive"]
            / max(row["mixed_selected"], 1),
        }
        for split, row in pooled_by_split.items()
    }
    deployment = copy.deepcopy(fix2_result["deployment_config"])
    deployment["residual_config"]["null_step_thresholds"] = list(thresholds)
    deployment["graph"]["maximum_selected_regions"] = STEPS
    deployment["graph"]["null_calibration"] = (
        "consumer_exact_direct_false_edge_greedy_with_covered_union_and_"
        "marginal_primitive_deduplication"
    )
    result = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "status": "source_only_consumer_exact_fix3_complete_target_unopened",
        "execution_authority": {
            "path": str(authority_path),
            "sha256": authority_sha,
        },
        "supersedes_fix2_result": authority["fix2_result"],
        "fix2_issue": {
            "gain_units_were_correct": True,
            "step_one_was_already_consumer_exact": True,
            "step_two_and_three_issue": (
                "full_core_order_statistics_did_not_replay_covered_union_and_"
                "marginal_primitive_deduplication"
            ),
            "feasibility_implication": (
                "raw_p_equal_0.9_full_core_gain_is_only_0.7123292753761552_"
                "and_cannot_pass_step_one;only_stronger_edges_can_modify_O0"
            ),
        },
        "consumer_exact_null_calibration": {
            "gain_formula": (
                "(2*sigmoid(logit(raw_p)-epsilon_logit)-1)_plus*"
                "marginal_unique_primitives/256"
            ),
            "strict_comparison": "candidate_gain_strictly_greater_than_threshold",
            "null_population": "different_instance_direct_edges_only",
            "scene_quantile": NULL_QUANTILE,
            "cross_scene_aggregation": "maximum",
            "edge_gate_order": [
                "raw_probability_greater_than_or_equal_to_0.9",
                "minimum_pair_appearance_boundary_concentration_greater_than_or_equal_to_source_q05",
                "pair_feature_robust_OOD_raw_score_less_than_or_equal_to_source_limit",
                "then_consumer_exact_sequential_gain",
            ],
            "edge_gate_audit": edge_gate_audit,
            "per_scene_thresholds": per_scene_null_thresholds,
            "global_thresholds": list(thresholds),
            "full_core_raw_probability_required_by_step": [
                _raw_probability_required(
                    threshold=value, epsilon_logit=epsilon
                )
                for value in thresholds
            ],
        },
        "source_true_edge_selection_audit": {
            "scope": (
                "conditional_graph_capability_with_each_source_region_as_a_"
                "pseudo_anchor;does_not_claim_text_query_anchor_coverage"
            ),
            "per_scene": per_scene_audit,
            "pooled": pooled_audit,
            "by_split": split_audit,
        },
        "deployment_config": deployment,
        "source_access": source_access(),
        "benchmark_execution_authorized": False,
        "target_execution_performed": False,
    }
    result["content_authority_sha256"] = canonical_json_sha256(
        {key: value for key, value in result.items() if key != "content_authority_sha256"}
    )
    write_frozen_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-authority", required=True)
    parser.add_argument("--expected-execution-authority-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
