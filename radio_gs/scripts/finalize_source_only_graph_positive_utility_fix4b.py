#!/usr/bin/env python3
"""Finalize the honest source-only positive-utility graph contract (FIX4B)."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.interfaces.lerf_o0_anchored_positive_utility_residual import (
    CONTRACT_SHA256 as POSITIVE_UTILITY_CONTRACT_SHA256,
    INTERNAL_GAIN_THRESHOLDS,
    MAXIMUM_REGIONS,
    SCHEMA as POSITIVE_UTILITY_INTERFACE_SCHEMA,
    SourceFixedPositiveUtilityConfig,
)
from radio_gs.scripts import audit_source_only_graph_expected_utility_fix4 as fix4
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
    "radio_gs.source_only_graph_positive_utility_fix4b_execution_authority.v1"
)
RESULT_SCHEMA = "radio_gs.source_only_graph_positive_utility_contract_fix4b.v1"
MINIMUM_EVERY_VALIDATION_SCENE_WILSON_LOWER = 0.95


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
        "positive_utility_interface",
        "positive_utility_interface_contract_sha256",
        "fix4_execution_authority",
        "fix4_result",
        "fix3_execution_authority",
        "fix3_result",
        "fix2_execution_authority",
        "fix2_result",
        "promotion_gate",
        "source_access",
        "benchmark_execution_authorized",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("FIX4B execution authority fields differ")
    authority = dict(value)
    if (
        authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("status") != "authorized_source_only_positive_utility_fix4b"
        or authority.get("positive_utility_interface_contract_sha256")
        != POSITIVE_UTILITY_CONTRACT_SHA256
        or authority.get("source_access") != source_access()
        or authority.get("benchmark_execution_authorized") is not False
        or authority.get("promotion_gate")
        != {
            "minimum_every_validation_scene_Wilson95_lower": (
                MINIMUM_EVERY_VALIDATION_SCENE_WILSON_LOWER
            ),
            "validation_pooled_marginal_weighted_signed_utility": (
                "strictly_greater_than_zero"
            ),
            "validation_true_pseudo_anchor_reach": (
                "strictly_greater_than_fix3"
            ),
            "failure_action": "reject_fix4b_do_not_open_target",
        }
    ):
        raise ValueError("FIX4B execution authority header differs")
    for name in (
        "implementation",
        "positive_utility_interface",
        "fix4_execution_authority",
        "fix4_result",
        "fix3_execution_authority",
        "fix3_result",
        "fix2_execution_authority",
        "fix2_result",
    ):
        authority[name] = _record(authority[name], label=name)
    return authority


def marginal_weighted_signed_utility(
    *,
    gains: torch.Tensor,
    marginal_primitives: torch.Tensor,
    labels: torch.Tensor,
    selected: torch.Tensor,
    novel_mass_reference: float,
) -> dict[str, float | int]:
    gain = torch.as_tensor(gains).detach().double().cpu()
    marginal = torch.as_tensor(marginal_primitives).detach().double().cpu()
    truth = torch.as_tensor(labels).detach().bool().cpu()
    keep = torch.as_tensor(selected).detach().bool().cpu()
    if (
        gain.shape != marginal.shape
        or gain.shape != truth.shape
        or gain.shape != keep.shape
        or gain.ndim != 2
        or float(novel_mass_reference) <= 0.0
        or not bool(torch.isfinite(gain).all())
        or bool((gain < 0.0).any())
        or bool((marginal < 0.0).any())
        or bool((keep & (marginal <= 0.0)).any())
    ):
        raise ValueError("signed utility inputs differ")
    selected_gain = gain[keep]
    selected_marginal_weight = marginal[keep] / float(novel_mass_reference)
    sign = torch.where(
        truth[keep],
        torch.ones_like(selected_gain),
        -torch.ones_like(selected_gain),
    )
    signed_sum = float((sign * selected_gain).sum())
    marginal_weight_sum = float(selected_marginal_weight.sum())
    return {
        "selected_count": int(keep.sum()),
        "positive_count": int((keep & truth).sum()),
        "negative_count": int((keep & ~truth).sum()),
        "signed_gain_sum": signed_sum,
        "marginal_weight_sum": marginal_weight_sum,
        "marginal_weighted_signed_utility": signed_sum
        / max(marginal_weight_sum, 1e-12),
    }


def _combine_signed(rows: Sequence[Mapping[str, float | int]]) -> dict[str, float | int]:
    signed_sum = sum(float(row["signed_gain_sum"]) for row in rows)
    marginal_weight = sum(float(row["marginal_weight_sum"]) for row in rows)
    return {
        "selected_count": sum(int(row["selected_count"]) for row in rows),
        "positive_count": sum(int(row["positive_count"]) for row in rows),
        "negative_count": sum(int(row["negative_count"]) for row in rows),
        "signed_gain_sum": signed_sum,
        "marginal_weight_sum": marginal_weight,
        "marginal_weighted_signed_utility": signed_sum
        / max(marginal_weight, 1e-12),
    }


def _validate_parent_chain(authority: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    fix4_authority_raw, _, _ = load_json_object(
        authority["fix4_execution_authority"]["path"],
        expected_sha256=authority["fix4_execution_authority"]["sha256"],
        label="FIX4B-bound FIX4 execution authority",
    )
    fix4_authority = fix4.validate_execution_authority(fix4_authority_raw)
    fix4_result, _, _ = load_json_object(
        authority["fix4_result"]["path"],
        expected_sha256=authority["fix4_result"]["sha256"],
        label="FIX4B-bound FIX4 result",
    )
    fix3_authority_raw, _, _ = load_json_object(
        authority["fix3_execution_authority"]["path"],
        expected_sha256=authority["fix3_execution_authority"]["sha256"],
        label="FIX4B-bound FIX3 execution authority",
    )
    fix3_authority = fix3.validate_execution_authority(fix3_authority_raw)
    fix3_result, _, _ = load_json_object(
        authority["fix3_result"]["path"],
        expected_sha256=authority["fix3_result"]["sha256"],
        label="FIX4B-bound FIX3 result",
    )
    fix2_authority_raw, _, _ = load_json_object(
        authority["fix2_execution_authority"]["path"],
        expected_sha256=authority["fix2_execution_authority"]["sha256"],
        label="FIX4B-bound FIX2 execution authority",
    )
    fix2_authority = fix2.validate_execution_authority(fix2_authority_raw)
    fix2_result, _, _ = load_json_object(
        authority["fix2_result"]["path"],
        expected_sha256=authority["fix2_result"]["sha256"],
        label="FIX4B-bound FIX2 result",
    )
    if (
        fix4_result.get("schema") != fix4.RESULT_SCHEMA
        or fix4_result.get("execution_authority") != authority["fix4_execution_authority"]
        or fix4_authority["fix3_execution_authority"]
        != authority["fix3_execution_authority"]
        or fix4_authority["fix3_result"] != authority["fix3_result"]
        or fix3_result.get("execution_authority") != authority["fix3_execution_authority"]
        or fix3_authority["fix2_execution_authority"]
        != authority["fix2_execution_authority"]
        or fix3_authority["fix2_result"] != authority["fix2_result"]
        or fix2_result.get("execution_authority") != authority["fix2_execution_authority"]
        or any(
            row.get("source_access") != source_access()
            or row.get("target_execution_performed") is not False
            for row in (fix4_result, fix3_result, fix2_result)
        )
    ):
        raise ValueError("FIX4B parent chain differs")
    return fix2_authority, fix3_result, fix4_result


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"FIX4B result exists: {output}")
    raw, authority_sha, authority_path = load_json_object(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
        label="FIX4B execution authority",
    )
    authority = validate_execution_authority(raw)
    if authority["implementation"] != file_record(Path(__file__).resolve()):
        raise ValueError("FIX4B implementation binding differs")
    fix2_authority, fix3_result, fix4_result = _validate_parent_chain(authority)
    checkpoint, loaded = fix2._load_inputs(fix2_authority)
    fix4_config = fix4_result["deployment_config"]
    epsilon = float(fix4_config["residual_config"]["epsilon_logit"])
    reliability_minimum = float(
        fix4_config["residual_config"]["minimum_reliability"]
    )
    maximum_feature_ood_score = float(
        fix4_config["residual_config"]["maximum_feature_ood_score"]
    )
    minimum_anchor_agreement = float(
        fix4_config["residual_config"]["minimum_anchor_agreement"]
    )
    minimum_stability = float(
        fix4_config["residual_config"]["minimum_stability"]
    )
    novel_reference = float(
        fix4_config["residual_config"]["novel_mass_reference"]
    )
    clean_config = SourceFixedPositiveUtilityConfig(
        epsilon_logit=epsilon,
        novel_mass_reference=novel_reference,
        minimum_reliability=reliability_minimum,
        maximum_feature_ood_score=maximum_feature_ood_score,
        minimum_anchor_agreement=minimum_anchor_agreement,
        minimum_stability=minimum_stability,
    )
    ood_raw_limit = float(fix4_config["feature_ood"]["raw_score_limit"])
    per_scene: dict[str, Any] = {}
    for scene, probability in loaded:
        eligible, edge_gate = fix4.edge_eligible_mask(
            pair_features=scene.pair_features,
            raw_probability=probability,
            median=checkpoint["normalization"]["median"],
            robust_scale=checkpoint["normalization"]["robust_scale"],
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
        selected = fix3.apply_strict_sequential_thresholds(
            mixed_trace["gains"], INTERNAL_GAIN_THRESHOLDS
        )
        true_selected = fix3.apply_strict_sequential_thresholds(
            true_trace["gains"], INTERNAL_GAIN_THRESHOLDS
        )
        selected_count = int(selected.sum())
        positive = int((selected & mixed_trace["labels"]).sum())
        signed = marginal_weighted_signed_utility(
            gains=mixed_trace["gains"],
            marginal_primitives=mixed_trace["marginal_primitives"],
            labels=mixed_trace["labels"],
            selected=selected,
            novel_mass_reference=novel_reference,
        )
        true_audit = fix3._selection_audit(
            trace=true_trace, selected=true_selected
        )
        parity = fix4_result["source_exact_consumer_audit"]["per_scene"][
            scene.scene_id
        ]
        if (
            selected_count != parity["mixed"]["selected_edge_count"]
            or positive != parity["mixed"]["selected_positive_edge_count"]
            or true_audit["anchor_with_any_selected_count"]
            != parity["true_only_capability"]["anchor_with_any_selected_count"]
        ):
            raise RuntimeError("FIX4B exact replay differs from frozen FIX4 audit")
        per_scene[scene.scene_id] = {
            "split": scene.split,
            "edge_gate": edge_gate,
            "selected_edge_count": selected_count,
            "selected_positive_edge_count": positive,
            "selected_negative_edge_count": selected_count - positive,
            "selected_precision": positive / selected_count,
            "selected_precision_Wilson95_lower": fix2.one_sided_wilson_lower(
                positive, selected_count
            ),
            "selected_count_by_step": [
                int(selected[:, step].sum()) for step in range(MAXIMUM_REGIONS)
            ],
            "true_pseudo_anchor_eligible_count": true_audit[
                "eligible_anchor_count"
            ],
            "true_pseudo_anchor_with_any_selected_count": true_audit[
                "anchor_with_any_selected_count"
            ],
            "true_pseudo_anchor_reach": true_audit[
                "anchor_with_any_selected_rate"
            ],
            "marginal_weighted_signed_utility": signed,
        }
    by_split = {}
    for split in ("source_train", "source_validation"):
        rows = [row for row in per_scene.values() if row["split"] == split]
        signed = _combine_signed(
            [row["marginal_weighted_signed_utility"] for row in rows]
        )
        selected = sum(int(row["selected_edge_count"]) for row in rows)
        positive = sum(int(row["selected_positive_edge_count"]) for row in rows)
        eligible_anchor = sum(
            int(row["true_pseudo_anchor_eligible_count"]) for row in rows
        )
        reached_anchor = sum(
            int(row["true_pseudo_anchor_with_any_selected_count"])
            for row in rows
        )
        by_split[split] = {
            "scene_count": len(rows),
            "selected_edge_count": selected,
            "selected_positive_edge_count": positive,
            "selected_negative_edge_count": selected - positive,
            "selected_precision": positive / selected,
            "selected_precision_Wilson95_lower": fix2.one_sided_wilson_lower(
                positive, selected
            ),
            "selected_count_by_step": [
                sum(int(row["selected_count_by_step"][step]) for row in rows)
                for step in range(MAXIMUM_REGIONS)
            ],
            "true_pseudo_anchor_reach": reached_anchor / eligible_anchor,
            "marginal_weighted_signed_utility": signed,
        }
    validation = by_split["source_validation"]
    fix3_reach = float(
        fix3_result["source_true_edge_selection_audit"]["by_split"]
        ["source_validation"]["true_anchor_with_any_selected_rate"]
    )
    outcomes = {
        "both_validation_scene_Wilson95_lower_at_least_0.95": all(
            per_scene[scene]["selected_precision_Wilson95_lower"]
            >= MINIMUM_EVERY_VALIDATION_SCENE_WILSON_LOWER
            for scene in fix2.VALIDATION_SCENES
        ),
        "validation_pooled_marginal_weighted_signed_utility_positive": float(
            validation["marginal_weighted_signed_utility"]
            ["marginal_weighted_signed_utility"]
        )
        > 0.0,
        "validation_true_pseudo_anchor_reach_strictly_exceeds_fix3": float(
            validation["true_pseudo_anchor_reach"]
        )
        > fix3_reach,
    }
    outcomes["passed"] = all(outcomes.values())
    deployment_config = {
        "interface": {
            "schema": POSITIVE_UTILITY_INTERFACE_SCHEMA,
            "contract_sha256": POSITIVE_UTILITY_CONTRACT_SHA256,
        },
        "edge_confidence_surrogate": {
            "raw_probability_minimum": fix2.RAW_EDGE_PROBABILITY_MINIMUM,
            "epsilon_logit": epsilon,
            "lower_score_formula": (
                "sigmoid(logit(clamp(raw_edge_score,1e-7,1-1e-7))-epsilon_logit)"
            ),
            "semantics": (
                "source_frozen_conservative_edge_confidence_surrogate_not_a_"
                "posterior_probability_or_conformal_or_FWER_guarantee"
            ),
        },
        "positive_utility_config": {
            "epsilon_logit": clean_config.epsilon_logit,
            "novel_mass_reference": clean_config.novel_mass_reference,
            "minimum_reliability": clean_config.minimum_reliability,
            "maximum_feature_ood_score": clean_config.maximum_feature_ood_score,
            "minimum_anchor_agreement": clean_config.minimum_anchor_agreement,
            "minimum_stability": clean_config.minimum_stability,
        },
        "selection": {
            "minimum_positive_gain": 0.0,
            "comparison": "strictly_greater_than",
            "maximum_selected_regions": MAXIMUM_REGIONS,
            "gain_formula": (
                "positive_part(2*lower_score-1)*unique_valid_marginal_primitives/256"
            ),
            "tie_break": "lower_canonical_region_index",
        },
        "query_gate": {
            "inputs": [
                "pair_feature_reliability",
                "pair_feature_OOD_score",
                "O0_anchor_agreement",
                "deterministic_replay_stability",
            ],
            "conjunction": True,
            "failed_gate": "bitwise_O0_only",
        },
        "anchor": {
            "O0_final_score_minimum": 0.6,
            "valid_core_supermajority": minimum_anchor_agreement,
            "quorum": 2,
        },
        "feature_OOD": {
            "raw_score_limit": ood_raw_limit,
            "unit_score_maximum": maximum_feature_ood_score,
            "input": "used_direct_edge_pair_features_exact_21_channel_order",
        },
        "residual": {
            "per_primitive_logit_maximum": epsilon,
            "sign": "nonnegative_only",
            "aggregation": "pointwise_max",
            "canonical_capability": "exact_frozen_O0_primitive_logits",
        },
    }
    result = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "status": (
            "source_only_positive_utility_fix4b_promoted_target_unopened"
            if outcomes["passed"]
            else "source_only_positive_utility_fix4b_rejected_target_must_remain_unopened"
        ),
        "execution_authority": {
            "path": str(authority_path),
            "sha256": authority_sha,
        },
        "parent_chain": {
            "fix4_execution_authority": authority["fix4_execution_authority"],
            "fix4_result": authority["fix4_result"],
            "fix3_execution_authority": authority["fix3_execution_authority"],
            "fix3_result": authority["fix3_result"],
            "fix2_execution_authority": authority["fix2_execution_authority"],
            "fix2_result": authority["fix2_result"],
        },
        "method_claim": (
            "risk_bounded_positive_utility_soft_residual_using_a_source_frozen_"
            "conservative_edge_confidence_surrogate;not_posterior_not_"
            "conformal_not_FWER_and_not_hard_graph_completion"
        ),
        "source_exact_consumer_audit": {
            "per_scene": per_scene,
            "by_split": by_split,
        },
        "promotion_gate": {
            "thresholds": authority["promotion_gate"],
            "fix3_validation_true_pseudo_anchor_reach": fix3_reach,
            "outcomes": outcomes,
            "decision": "promote_source_only" if outcomes["passed"] else "reject",
        },
        "deployment_config": deployment_config,
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
