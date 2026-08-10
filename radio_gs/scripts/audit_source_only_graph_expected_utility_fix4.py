#!/usr/bin/env python3
"""Audit a source-only expected-utility-lower graph residual (FIX4).

FIX4 keeps every precision and safety gate from FIX3, but removes the
familywise q99 null-gain hard stop.  A candidate must still have raw p>=0.9,
pass source-frozen pair reliability and OOD gates, be reached from the frozen
O0 anchor contract, and have strictly positive one-sided calibrated expected
utility.  The consumer adds at most the already frozen bounded logit residual
to at most three regions; it never replaces O0 or performs a hard graph union.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import copy
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.scripts import calibrate_source_only_graph_confidence_v1 as fix2
from radio_gs.scripts import calibrate_source_only_graph_consumer_exact_fix3 as fix3
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    validate_file_record,
    write_frozen_json,
)


AUTHORITY_SCHEMA = (
    "radio_gs.source_only_graph_expected_utility_fix4_execution_authority.v1"
)
RESULT_SCHEMA = "radio_gs.source_only_graph_expected_utility_audit_fix4.v1"
ZERO_THRESHOLDS = (0.0, 0.0, 0.0)
MINIMUM_VALIDATION_WILSON_LOWER = 0.95
MINIMUM_EVERY_VALIDATION_SCENE_WILSON_LOWER = 0.90
MINIMUM_VALIDATION_TRUE_PSEUDO_ANCHOR_COVERAGE = 0.90


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
        "fix3_execution_authority",
        "fix3_result",
        "fixed_intervention",
        "source_gate",
        "source_access",
        "benchmark_execution_authorized",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("FIX4 execution authority fields differ")
    authority = dict(value)
    if (
        authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("status") != "authorized_source_only_expected_utility_fix4"
        or authority.get("source_access") != source_access()
        or authority.get("benchmark_execution_authorized") is not False
        or authority.get("fixed_intervention")
        != {
            "consumer_null_step_thresholds": list(ZERO_THRESHOLDS),
            "comparison": "strict_gain_greater_than_zero",
            "maximum_selected_regions": 3,
            "unchanged": [
                "epsilon_logit",
                "raw_edge_probability_minimum",
                "pair_reliability_gate",
                "pair_feature_OOD_gate",
                "O0_supermajority_anchor_contract",
                "anchor_quorum",
                "nonnegative_bounded_residual",
                "bitwise_O0_fallback",
            ],
        }
        or authority.get("source_gate")
        != {
            "minimum_validation_mixed_precision_Wilson95_lower": (
                MINIMUM_VALIDATION_WILSON_LOWER
            ),
            "minimum_every_validation_scene_mixed_precision_Wilson95_lower": (
                MINIMUM_EVERY_VALIDATION_SCENE_WILSON_LOWER
            ),
            "minimum_validation_true_pseudo_anchor_coverage": (
                MINIMUM_VALIDATION_TRUE_PSEUDO_ANCHOR_COVERAGE
            ),
            "failure_action": "reject_fix4_do_not_open_target",
        }
    ):
        raise ValueError("FIX4 execution authority header differs")
    for name in ("implementation", "fix3_execution_authority", "fix3_result"):
        authority[name] = _record(authority[name], label=name)
    return authority


def edge_eligible_mask(
    *,
    pair_features: torch.Tensor,
    raw_probability: torch.Tensor,
    median: torch.Tensor,
    robust_scale: torch.Tensor,
    raw_probability_minimum: float,
    reliability_minimum: float,
    ood_raw_limit: float,
) -> tuple[torch.Tensor, dict[str, int]]:
    features = torch.as_tensor(pair_features).detach().float().cpu()
    probability = torch.as_tensor(raw_probability).detach().float().cpu()
    center = torch.as_tensor(median).detach().float().cpu()
    scale = torch.as_tensor(robust_scale).detach().float().cpu()
    if (
        features.ndim != 2
        or features.shape[1] != 21
        or probability.shape != (features.shape[0],)
        or center.shape != (21,)
        or scale.shape != (21,)
        or bool((scale <= 0).any())
    ):
        raise ValueError("FIX4 edge gate axes differ")
    raw_gate = probability >= float(raw_probability_minimum)
    reliability = torch.minimum(features[:, 17], features[:, 18])
    reliability_gate = reliability >= float(reliability_minimum)
    ood_raw = ((features - center) / scale).abs().max(dim=1).values
    ood_gate = ood_raw <= float(ood_raw_limit)
    eligible = raw_gate & reliability_gate & ood_gate
    return eligible.contiguous(), {
        "all_edge_count": int(features.shape[0]),
        "raw_probability_gate_count": int(raw_gate.sum()),
        "raw_and_reliability_gate_count": int((raw_gate & reliability_gate).sum()),
        "all_three_edge_gate_count": int(eligible.sum()),
    }


def _residual_amplitude_summary(
    *,
    trace: Mapping[str, torch.Tensor],
    selected: torch.Tensor,
    epsilon_logit: float,
) -> dict[str, float | int | None]:
    marginal = trace["marginal_primitives"].float()
    safe_marginal = marginal.clamp_min(1.0)
    excess = trace["gains"] * fix3.NOVEL_MASS_REFERENCE / safe_marginal
    amplitude = float(epsilon_logit) * excess
    labels = trace["labels"]

    def summarize(mask: torch.Tensor) -> dict[str, float | int | None]:
        values = amplitude[mask]
        return {
            "count": int(values.numel()),
            "mean": float(values.mean()) if values.numel() else None,
            "maximum": float(values.max()) if values.numel() else None,
        }

    return {
        "selected": summarize(selected),
        "selected_true": summarize(selected & labels),
        "selected_false": summarize(selected & ~labels),
    }


def _scene_audit(
    *,
    scene: Any,
    mixed_trace: Mapping[str, torch.Tensor],
    true_trace: Mapping[str, torch.Tensor],
    edge_gate: Mapping[str, int],
    eligible_edges: torch.Tensor,
    epsilon_logit: float,
) -> dict[str, Any]:
    mixed_selected = fix3.apply_strict_sequential_thresholds(
        mixed_trace["gains"], ZERO_THRESHOLDS
    )
    true_selected = fix3.apply_strict_sequential_thresholds(
        true_trace["gains"], ZERO_THRESHOLDS
    )
    mixed = fix3._selection_audit(trace=mixed_trace, selected=mixed_selected)
    true = fix3._selection_audit(trace=true_trace, selected=true_selected)
    selected_count = int(mixed_selected.sum())
    selected_positive = int((mixed_selected & mixed_trace["labels"]).sum())
    precision = selected_positive / selected_count
    return {
        "split": scene.split,
        "edge_gate": {
            **edge_gate,
            "eligible_true_edge_count": int((eligible_edges & scene.targets).sum()),
            "eligible_false_edge_count": int((eligible_edges & ~scene.targets).sum()),
        },
        "mixed": {
            **mixed,
            "selected_false_edge_count": selected_count - selected_positive,
            "selected_false_rate": 1.0 - precision,
            "selected_precision_Wilson95_lower": fix2.one_sided_wilson_lower(
                selected_positive, selected_count
            ),
            "bounded_residual_amplitude": _residual_amplitude_summary(
                trace=mixed_trace,
                selected=mixed_selected,
                epsilon_logit=epsilon_logit,
            ),
        },
        "true_only_capability": true,
    }


def _aggregate(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    selected = sum(int(row["mixed"]["selected_edge_count"]) for row in rows)
    positive = sum(
        int(row["mixed"]["selected_positive_edge_count"]) for row in rows
    )
    eligible_anchor = sum(
        int(row["true_only_capability"]["eligible_anchor_count"]) for row in rows
    )
    reached_anchor = sum(
        int(row["true_only_capability"]["anchor_with_any_selected_count"])
        for row in rows
    )
    return {
        "scene_count": len(rows),
        "selected_edge_count": selected,
        "selected_positive_edge_count": positive,
        "selected_false_edge_count": selected - positive,
        "selected_precision": positive / selected,
        "selected_false_rate": (selected - positive) / selected,
        "selected_precision_Wilson95_lower": fix2.one_sided_wilson_lower(
            positive, selected
        ),
        "selected_count_by_step": [
            sum(int(row["mixed"]["selected_count_by_step"][step]) for row in rows)
            for step in range(3)
        ],
        "true_pseudo_anchor_eligible_count": eligible_anchor,
        "true_pseudo_anchor_with_any_selected_count": reached_anchor,
        "true_pseudo_anchor_coverage": reached_anchor / eligible_anchor,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"FIX4 result exists: {output}")
    raw, authority_sha, authority_path = load_json_object(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
        label="FIX4 execution authority",
    )
    authority = validate_execution_authority(raw)
    if authority["implementation"] != file_record(Path(__file__).resolve()):
        raise ValueError("FIX4 implementation binding differs")
    fix3_authority_raw, _, _ = load_json_object(
        authority["fix3_execution_authority"]["path"],
        expected_sha256=authority["fix3_execution_authority"]["sha256"],
        label="FIX4-bound FIX3 execution authority",
    )
    fix3_authority = fix3.validate_execution_authority(fix3_authority_raw)
    fix3_result, _, _ = load_json_object(
        authority["fix3_result"]["path"],
        expected_sha256=authority["fix3_result"]["sha256"],
        label="FIX4-bound FIX3 result",
    )
    if (
        fix3_result.get("schema") != fix3.RESULT_SCHEMA
        or fix3_result.get("execution_authority")
        != authority["fix3_execution_authority"]
        or fix3_result.get("source_access") != source_access()
        or fix3_result.get("target_execution_performed") is not False
    ):
        raise ValueError("FIX4 parent binding differs")
    fix2_authority_raw, _, _ = load_json_object(
        fix3_authority["fix2_execution_authority"]["path"],
        expected_sha256=fix3_authority["fix2_execution_authority"]["sha256"],
        label="FIX4-bound FIX2 execution authority",
    )
    fix2_authority = fix2.validate_execution_authority(fix2_authority_raw)
    checkpoint, loaded = fix2._load_inputs(fix2_authority)
    config = fix3_result["deployment_config"]
    epsilon = float(config["residual_config"]["epsilon_logit"])
    reliability_minimum = float(
        config["residual_config"]["minimum_reliability"]
    )
    ood_raw_limit = float(config["feature_ood"]["raw_score_limit"])
    median = checkpoint["normalization"]["median"]
    robust_scale = checkpoint["normalization"]["robust_scale"]
    per_scene: dict[str, Any] = {}
    for scene, probability in loaded:
        eligible, edge_gate = edge_eligible_mask(
            pair_features=scene.pair_features,
            raw_probability=probability,
            median=median,
            robust_scale=robust_scale,
            raw_probability_minimum=fix2.RAW_EDGE_PROBABILITY_MINIMUM,
            reliability_minimum=reliability_minimum,
            ood_raw_limit=ood_raw_limit,
        )
        lower = fix2.lower_probability(probability, epsilon)
        mixed_trace = fix3.exact_direct_edge_trace(
            scene=scene,
            probability_lower=lower,
            edge_eligible_mask=eligible,
            target_filter=None,
        )
        true_trace = fix3.exact_direct_edge_trace(
            scene=scene,
            probability_lower=lower,
            edge_eligible_mask=eligible,
            target_filter=True,
        )
        per_scene[scene.scene_id] = _scene_audit(
            scene=scene,
            mixed_trace=mixed_trace,
            true_trace=true_trace,
            edge_gate=edge_gate,
            eligible_edges=eligible,
            epsilon_logit=epsilon,
        )
    by_split = {
        split: _aggregate(
            [row for row in per_scene.values() if row["split"] == split]
        )
        for split in ("source_train", "source_validation")
    }
    validation = by_split["source_validation"]
    gate = {
        "validation_mixed_precision_Wilson95_lower": validation[
            "selected_precision_Wilson95_lower"
        ]
        >= MINIMUM_VALIDATION_WILSON_LOWER,
        "every_validation_scene_mixed_precision_Wilson95_lower": all(
            per_scene[scene]["mixed"]["selected_precision_Wilson95_lower"]
            >= MINIMUM_EVERY_VALIDATION_SCENE_WILSON_LOWER
            for scene in fix2.VALIDATION_SCENES
        ),
        "validation_true_pseudo_anchor_coverage": validation[
            "true_pseudo_anchor_coverage"
        ]
        >= MINIMUM_VALIDATION_TRUE_PSEUDO_ANCHOR_COVERAGE,
    }
    gate["passed"] = all(gate.values())
    deployment = copy.deepcopy(config)
    deployment["residual_config"]["null_step_thresholds"] = list(ZERO_THRESHOLDS)
    deployment["graph"]["expected_utility_mode"] = (
        "strict_positive_one_sided_calibrated_expected_utility_lower"
    )
    deployment["graph"]["familywise_q99_hard_null_used"] = False
    result = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "status": (
            "source_only_fix4_promoted_target_still_unopened"
            if gate["passed"]
            else "source_only_fix4_rejected_target_must_remain_unopened"
        ),
        "execution_authority": {
            "path": str(authority_path),
            "sha256": authority_sha,
        },
        "parent_fix3_result": authority["fix3_result"],
        "method_interpretation": {
            "name": "risk_bounded_soft_residual",
            "not_uncontrolled_graph_reasons": [
                "raw_edge_probability_must_be_at_least_0.9",
                "one_sided_logit_correction_is_applied_before_utility",
                "pair_reliability_and_pair_feature_OOD_gates_are_unchanged",
                "O0_supermajority_anchor_contract_and_quorum_are_unchanged",
                "strictly_positive_expected_utility_lower_is_required",
                "at_most_three_regions_receive_a_nonnegative_residual",
                "per_primitive_residual_is_bounded_by_epsilon_logit",
                "O0_logits_remain_the_canonical_capability_and_failed_gates_are_bitwise_O0",
                "no_hard_connected_component_or_binary_region_union_is_used",
            ],
        },
        "source_exact_consumer_audit": {
            "per_scene": per_scene,
            "by_split": by_split,
        },
        "source_gate": {
            "thresholds": authority["source_gate"],
            "outcomes": gate,
            "decision": "promote_source_only" if gate["passed"] else "reject",
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

