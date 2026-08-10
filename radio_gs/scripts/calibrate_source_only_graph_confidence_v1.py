#!/usr/bin/env python3
"""Calibrate a conservative graph residual using source authorities only.

This program has no benchmark, target, query, renderer, or metric input.  It
audits the promoted RegionCoMembershipV2 checkpoint on its exact four source
training and two source validation authorities and emits one immutable JSON
configuration for a capability-preserving target consumer.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import torch

from radio_gs.interfaces.lerf_o0_anchored_conformal_residual import (
    CONTRACT_SHA256 as RESIDUAL_CONTRACT_SHA256,
    SourceFixedResidualConfig,
)
from radio_gs.interfaces.region_comembership_v2_formal import validate_checkpoint
from radio_gs.models.region_comembership_v2 import RegionCoMembershipV2
from radio_gs.scripts.train_source_region_comembership_v2 import (
    TRAIN_SCENES,
    VALIDATION_SCENES,
    SceneAuthorityV2,
    load_scene_authority_v2,
    validate_execution_authority as validate_training_execution_authority,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
    write_frozen_json,
)


AUTHORITY_SCHEMA = "radio_gs.source_only_graph_calibration_execution_authority.v1"
RESULT_SCHEMA = "radio_gs.source_only_graph_conformal_calibration.v1"
RAW_EDGE_PROBABILITY_MINIMUM = 0.9
ONE_SIDED_CONFIDENCE = 0.95
TAIL_QUANTILE = 0.99
NOVEL_MASS_REFERENCE = 256.0
NULL_STEPS = 3
ANCHOR_QUORUM = 2
O0_CORE_SUPERMAJORITY = 0.75
MAXIMUM_NULL_ACTIVATION = 0.0
MINIMUM_STABILITY = 1.0


def source_access() -> dict[str, bool]:
    return {
        "source_instance_labels_opened": True,
        "source_train_pair_features_opened": True,
        "source_validation_pair_features_opened": True,
        "source_query_evidence_opened": False,
        "benchmark_images_opened": False,
        "benchmark_labels_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_queries_opened": False,
        "benchmark_metrics_opened": False,
        "target_heldout_opened": False,
        "target_pair_features_opened": False,
        "target_metrics_computed": False,
    }


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
        "residual_interface",
        "residual_interface_contract_sha256",
        "source_training_execution_authority",
        "checkpoint",
        "source_train",
        "source_validation",
        "fixed_method",
        "source_access",
        "benchmark_execution_authorized",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("source graph calibration authority fields differ")
    authority = dict(value)
    if (
        authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("status") != "authorized_source_only_calibration"
        or authority.get("source_access") != source_access()
        or authority.get("benchmark_execution_authorized") is not False
        or authority.get("residual_interface_contract_sha256")
        != RESIDUAL_CONTRACT_SHA256
    ):
        raise ValueError("source graph calibration authority header differs")
    for name in (
        "implementation",
        "residual_interface",
        "source_training_execution_authority",
        "checkpoint",
    ):
        authority[name] = _record(authority[name], label=name)
    method = authority.get("fixed_method")
    expected_method = {
        "raw_edge_probability_minimum": RAW_EDGE_PROBABILITY_MINIMUM,
        "one_sided_confidence": ONE_SIDED_CONFIDENCE,
        "tail_quantile": TAIL_QUANTILE,
        "novel_mass_reference": NOVEL_MASS_REFERENCE,
        "null_steps": NULL_STEPS,
        "anchor_quorum": ANCHOR_QUORUM,
        "o0_core_supermajority": O0_CORE_SUPERMAJORITY,
        "maximum_null_activation": MAXIMUM_NULL_ACTIVATION,
        "minimum_stability": MINIMUM_STABILITY,
    }
    if method != expected_method:
        raise ValueError("source graph calibration fixed method differs")
    for split, expected in (
        ("source_train", TRAIN_SCENES),
        ("source_validation", VALIDATION_SCENES),
    ):
        rows = authority[split]
        if not isinstance(rows, list) or len(rows) != len(expected):
            raise ValueError(f"source graph calibration {split} differs")
        normalized = []
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {"scene_id", "authority"}:
                raise ValueError(f"source graph calibration {split} row differs")
            normalized.append(
                {
                    "scene_id": str(row["scene_id"]),
                    "authority": _record(
                        row["authority"], label=f"{split} source authority"
                    ),
                }
            )
        if tuple(row["scene_id"] for row in normalized) != tuple(expected):
            raise ValueError(f"source graph calibration {split} scene axis differs")
        authority[split] = normalized
    return authority


def one_sided_wilson_lower(
    positive_count: int,
    total_count: int,
    *,
    confidence: float = ONE_SIDED_CONFIDENCE,
) -> float:
    """Return the finite one-sided Wilson lower confidence bound."""

    positive = int(positive_count)
    total = int(total_count)
    if total <= 0 or positive < 0 or positive > total:
        raise ValueError("Wilson counts differ")
    if not 0.5 < float(confidence) < 1.0:
        raise ValueError("Wilson confidence differs")
    proportion = positive / total
    z = NormalDist().inv_cdf(float(confidence))
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = proportion + z2 / (2.0 * total)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z2 / (4.0 * total * total)
    )
    return max(0.0, min(1.0, (center - radius) / denominator))


def logit(value: float) -> float:
    probability = float(value)
    if not 0.0 < probability < 1.0:
        raise ValueError("logit probability must be open-unit")
    return math.log(probability) - math.log1p(-probability)


def expit(value: float) -> float:
    scalar = float(value)
    if scalar >= 0:
        inverse = math.exp(-scalar)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(scalar)
    return exponential / (1.0 + exponential)


def lower_probability(probability: torch.Tensor, epsilon_logit: float) -> torch.Tensor:
    values = torch.as_tensor(probability).detach().float().cpu()
    if (
        values.ndim != 1
        or not bool(torch.isfinite(values).all())
        or bool((values < 0.0).any())
        or bool((values > 1.0).any())
        or float(epsilon_logit) < 0.0
    ):
        raise ValueError("lower probability inputs differ")
    clamped = values.clamp(min=1e-7, max=1.0 - 1e-7)
    return torch.sigmoid(torch.logit(clamped) - float(epsilon_logit)).contiguous()


def scene_tail_audit(
    *,
    scene: SceneAuthorityV2,
    probability: torch.Tensor,
    threshold: float = RAW_EDGE_PROBABILITY_MINIMUM,
) -> dict[str, Any]:
    values = torch.as_tensor(probability).detach().float().cpu()
    if values.shape != scene.targets.shape:
        raise ValueError("scene tail probability axis differs")
    selected = values >= float(threshold)
    count = int(selected.sum())
    positive = int(scene.targets[selected].sum())
    if count <= 0:
        raise ValueError("scene tail has no support")
    weights = scene.evidence_weights[selected].double()
    weighted_positive = float(weights[scene.targets[selected]].sum())
    weighted_total = float(weights.sum())
    precision = positive / count
    wilson = one_sided_wilson_lower(positive, count)
    epsilon = max(0.0, logit(float(threshold)) - logit(wilson))
    return {
        "candidate_pair_count": int(values.numel()),
        "accepted_edge_count": count,
        "accepted_positive_count": positive,
        "precision": precision,
        "evidence_weighted_precision": weighted_positive / weighted_total,
        "one_sided_wilson_lower": wilson,
        "tail_logit_nonconformity": epsilon,
    }


def leave_one_scene_out_max_audit(
    per_scene: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Audit the max scene score as a finite cross-scene rank construction."""

    rows = []
    for heldout in sorted(per_scene):
        calibration = [
            float(row["tail_logit_nonconformity"])
            for scene, row in per_scene.items()
            if scene != heldout
        ]
        epsilon = max(calibration)
        corrected_floor = expit(logit(RAW_EDGE_PROBABILITY_MINIMUM) - epsilon)
        heldout_lower = float(per_scene[heldout]["one_sided_wilson_lower"])
        rows.append(
            {
                "heldout_scene_id": heldout,
                "fit_scene_count": len(calibration),
                "fit_max_epsilon_logit": epsilon,
                "corrected_tail_floor": corrected_floor,
                "heldout_one_sided_wilson_lower": heldout_lower,
                "covered": corrected_floor <= heldout_lower + 1e-15,
            }
        )
    return rows


def false_edge_null_thresholds(
    *,
    scene: SceneAuthorityV2,
    probability_lower: torch.Tensor,
    steps: int = NULL_STEPS,
    quantile: float = TAIL_QUANTILE,
    novel_mass_reference: float = NOVEL_MASS_REFERENCE,
) -> tuple[float, ...]:
    """Upper-tail source null gains for the first three sequential steps.

    Every different-instance edge is treated as a null candidate from both
    endpoints.  Full candidate-core size is used at every step, which is an
    upper bound on later marginal novelty and is therefore conservative.
    """

    lower = torch.as_tensor(probability_lower).detach().float().cpu()
    if lower.shape != scene.targets.shape or int(steps) <= 0:
        raise ValueError("null threshold inputs differ")
    per_anchor: list[list[float]] = [[] for _ in range(scene.region_count)]
    core_fraction = scene.token_mask.sum(dim=1).float() / float(
        novel_mass_reference
    )
    null_indices = torch.nonzero(~scene.targets, as_tuple=False).flatten()
    gain = (2.0 * lower[null_indices] - 1.0).clamp_min(0.0)
    for offset, edge_index in enumerate(null_indices.tolist()):
        left = int(scene.pair_indices[0, edge_index])
        right = int(scene.pair_indices[1, edge_index])
        base = float(gain[offset])
        per_anchor[left].append(base * float(core_fraction[right]))
        per_anchor[right].append(base * float(core_fraction[left]))
    order = torch.zeros((scene.region_count, int(steps)), dtype=torch.float32)
    for anchor, values in enumerate(per_anchor):
        selected = sorted(values, reverse=True)[: int(steps)]
        if selected:
            order[anchor, : len(selected)] = torch.tensor(selected)
    return tuple(
        float(torch.quantile(order[:, step], float(quantile)))
        for step in range(int(steps))
    )


def _macro_and_micro(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float | int]:
    count = sum(int(row["accepted_edge_count"]) for row in rows)
    positive = sum(int(row["accepted_positive_count"]) for row in rows)
    return {
        "scene_count": len(rows),
        "accepted_edge_count": count,
        "accepted_positive_count": positive,
        "micro_precision": positive / count,
        "scene_macro_precision": sum(float(row["precision"]) for row in rows)
        / len(rows),
        "scene_macro_evidence_weighted_precision": sum(
            float(row["evidence_weighted_precision"]) for row in rows
        )
        / len(rows),
        "minimum_scene_precision": min(float(row["precision"]) for row in rows),
        "minimum_scene_one_sided_wilson_lower": min(
            float(row["one_sided_wilson_lower"]) for row in rows
        ),
    }


def _load_inputs(
    authority: Mapping[str, Any],
) -> tuple[dict[str, Any], list[tuple[SceneAuthorityV2, torch.Tensor]]]:
    training_raw, _, _ = load_json_object(
        authority["source_training_execution_authority"]["path"],
        expected_sha256=authority["source_training_execution_authority"]["sha256"],
        label="source graph calibration training execution",
    )
    training = validate_training_execution_authority(training_raw)
    checkpoint_raw, _, _ = load_torch_mapping(
        authority["checkpoint"]["path"],
        expected_sha256=authority["checkpoint"]["sha256"],
        map_location="cpu",
        label="source graph calibration checkpoint",
    )
    checkpoint = validate_checkpoint(checkpoint_raw)
    if checkpoint["execution_authority"] != authority[
        "source_training_execution_authority"
    ]:
        raise ValueError("checkpoint training execution binding differs")
    if training["source_train"] != authority["source_train"] or training[
        "source_validation"
    ] != authority["source_validation"]:
        raise ValueError("source graph calibration six-scene binding differs")
    model = RegionCoMembershipV2(
        checkpoint["normalization"]["median"],
        checkpoint["normalization"]["robust_scale"],
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    loaded: list[tuple[SceneAuthorityV2, torch.Tensor]] = []
    with torch.no_grad():
        for split in ("source_train", "source_validation"):
            for row in authority[split]:
                scene = load_scene_authority_v2(
                    row["authority"],
                    expected_scene_id=row["scene_id"],
                    expected_split=split,
                )
                loaded.append((scene, model.probability(scene.pair_features)))
    return checkpoint, loaded


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"source graph calibration result exists: {output}")
    authority_raw, authority_sha, authority_path = load_json_object(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
        label="source graph calibration execution authority",
    )
    authority = validate_execution_authority(authority_raw)
    if authority["implementation"] != file_record(Path(__file__).resolve()):
        raise ValueError("source graph calibration implementation binding differs")
    checkpoint, loaded = _load_inputs(authority)
    per_scene: dict[str, dict[str, Any]] = {}
    for scene, probability in loaded:
        row = scene_tail_audit(scene=scene, probability=probability)
        row["split"] = scene.split
        per_scene[scene.scene_id] = row
    loo = leave_one_scene_out_max_audit(per_scene)
    epsilon = max(
        float(row["tail_logit_nonconformity"]) for row in per_scene.values()
    )
    calibrated_floor = expit(logit(RAW_EDGE_PROBABILITY_MINIMUM) - epsilon)

    null_by_scene: dict[str, list[float]] = {}
    reliability_values = []
    ood_scene_quantiles: dict[str, float] = {}
    median = torch.as_tensor(checkpoint["normalization"]["median"]).float()
    scale = torch.as_tensor(checkpoint["normalization"]["robust_scale"]).float()
    for scene, probability in loaded:
        lower = lower_probability(probability, epsilon)
        null_by_scene[scene.scene_id] = list(
            false_edge_null_thresholds(scene=scene, probability_lower=lower)
        )
        accepted_true = (probability >= RAW_EDGE_PROBABILITY_MINIMUM) & scene.targets
        reliability_values.append(
            torch.minimum(
                scene.pair_features[accepted_true, 17],
                scene.pair_features[accepted_true, 18],
            )
        )
        robust_max = ((scene.pair_features - median) / scale).abs().max(dim=1).values
        ood_scene_quantiles[scene.scene_id] = float(
            torch.quantile(robust_max, TAIL_QUANTILE)
        )
    null_thresholds = tuple(
        max(row[step] for row in null_by_scene.values())
        for step in range(NULL_STEPS)
    )
    reliability_threshold = float(
        torch.quantile(torch.cat(reliability_values), 0.05)
    )
    raw_ood_limit = max(ood_scene_quantiles.values())
    config = SourceFixedResidualConfig(
        epsilon_logit=epsilon,
        novel_mass_reference=NOVEL_MASS_REFERENCE,
        null_step_thresholds=null_thresholds,
        minimum_reliability=reliability_threshold,
        maximum_feature_ood_score=0.5,
        minimum_anchor_agreement=O0_CORE_SUPERMAJORITY,
        maximum_null_activation=MAXIMUM_NULL_ACTIVATION,
        minimum_stability=MINIMUM_STABILITY,
    )
    train_rows = [per_scene[scene] for scene in TRAIN_SCENES]
    validation_rows = [per_scene[scene] for scene in VALIDATION_SCENES]
    deployment_config = {
        "residual_config": {
            "epsilon_logit": config.epsilon_logit,
            "novel_mass_reference": config.novel_mass_reference,
            "null_step_thresholds": list(config.null_step_thresholds),
            "minimum_reliability": config.minimum_reliability,
            "maximum_feature_ood_score": config.maximum_feature_ood_score,
            "minimum_anchor_agreement": config.minimum_anchor_agreement,
            "maximum_null_activation": config.maximum_null_activation,
            "minimum_stability": config.minimum_stability,
        },
        "graph": {
            "method": "direct_O0_anchor_edge_residual",
            "raw_edge_probability_minimum": RAW_EDGE_PROBABILITY_MINIMUM,
            "edge_lower_formula": "sigmoid(logit(clamp(p,1e-7,1-1e-7))-epsilon_logit)",
            "calibrated_edge_lower_minimum": calibrated_floor,
            "anchor_quorum": ANCHOR_QUORUM,
            "anchor_quorum_definition": (
                "at_least_two_query_qualified_O0_supermajority_anchors;"
                "candidate_requires_at_least_one_direct_raw_p_ge_0.9_edge_"
                "from_any_anchor_not_two_direct_neighbors_or_two_paths"
            ),
            "maximum_selected_regions": NULL_STEPS,
            "formal_checkpoint_safety_cap": int(
                checkpoint["selected_rule"]["maximum_regions"]
            ),
            "candidate_region_lower": (
                "maximum_calibrated_lower_of_direct_p_ge_0.9_edges_from_any_"
                "query_qualified_O0_anchor;O0_anchor_region_lower_is_1"
            ),
        },
        "anchor": {
            "o0_core_supermajority": O0_CORE_SUPERMAJORITY,
            "o0_final_score_minimum": 0.6,
            "definition": (
                "mean(valid_core_exact_O0_final_score_strictly_greater_than_0.6)"
            ),
            "rank256_query_top_tail": "diagnostic_not_used_by_primary",
            "top_tail_size": int(checkpoint["selected_rule"]["maximum_regions"]),
        },
        "feature_ood": {
            "input": (
                "target_used_direct_edge_pair_features_in_the_exact_21_channel_"
                "source_order_not_rank256_descriptor_reliability"
            ),
            "raw_score": "max_abs((pair_feature-source_median)/source_robust_scale)",
            "raw_score_limit": raw_ood_limit,
            "unit_score": "raw_score/(raw_score+raw_score_limit)",
            "unit_score_maximum": 0.5,
            "scene_quantile": TAIL_QUANTILE,
            "maximum_target_ood_fraction": 0.01,
        },
        "query_gate_diagnostics": {
            "reliability": (
                "minimum_over_used_direct_edge_pair_features_of_minimum_"
                "appearance_concentration_channel17_and_boundary_"
                "concentration_channel18;no_edges_is_0;never_use_rank256_"
                "descriptor_reliability"
            ),
            "feature_ood_score": (
                "maximum_over_used_graph_edges_of_raw_score/(raw_score+limit);"
                "missing_feature_evidence_is_1"
            ),
            "anchor_agreement": (
                "minimum_over_quorum_O0_anchors_of_fraction_valid_core_exact_"
                "O0_final_scores_strictly_greater_than_0.6;"
                "missing_anchor_evidence_is_0"
            ),
            "null_activation": (
                "fraction_of_selected_steps_whose_candidate_gain_is_not_"
                "strictly_above_the_corresponding_source_false_edge_null_"
                "threshold;valid_sequential_readout_is_exactly_0"
            ),
            "stability": (
                "fraction_of_selected_regions_unchanged_under_exact_repeat_and_"
                "canonical_tie_replay;missing_replay_is_0"
            ),
        },
        "source_query_bank_requirement": (
            "not_applicable_primary_uses_only_runtime_O0_supermajority_anchors_"
            "and_source_calibrated_query_independent_direct_edges"
        ),
        "fallback_on_any_failed_gate": "bitwise_O0_only_no_graph_residual",
    }
    result = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "status": "source_only_graph_calibration_complete_target_unopened",
        "execution_authority": {
            "path": str(authority_path),
            "sha256": authority_sha,
        },
        "checkpoint": authority["checkpoint"],
        "residual_interface_contract_sha256": RESIDUAL_CONTRACT_SHA256,
        "tail_audit": {
            "raw_edge_probability_minimum": RAW_EDGE_PROBABILITY_MINIMUM,
            "one_sided_confidence": ONE_SIDED_CONFIDENCE,
            "one_sided_bound": "Wilson_score_lower_not_split_conformal",
            "epsilon_formula": (
                "max_scene max(0,logit(0.9)-logit(scene_Wilson95_lower))"
            ),
            "per_scene": per_scene,
            "source_train": _macro_and_micro(train_rows),
            "source_validation": _macro_and_micro(validation_rows),
        },
        "cross_scene_rank_audit": {
            "method": "leave_one_scene_out_max_nonconformity",
            "rows": loo,
            "covered_scene_count": sum(bool(row["covered"]) for row in loo),
            "scene_count": len(loo),
            "empirical_coverage": sum(bool(row["covered"]) for row in loo)
            / len(loo),
            "full_six_scene_epsilon_logit": epsilon,
            "finite_scene_rank_miscoverage_upper_resolution": 1.0
            / (len(loo) + 1.0),
            "claim": (
                "cross_fit_rank_audit_plus_one_sided_confidence_lower_bound;"
                "not_claimed_as_split_conformal_probability_calibration"
            ),
        },
        "source_fixed_calibration": {
            "calibrated_edge_lower_minimum": calibrated_floor,
            "null_gain_quantile": TAIL_QUANTILE,
            "per_scene_null_step_thresholds": null_by_scene,
            "global_null_step_thresholds": list(null_thresholds),
            "minimum_reliability_quantile": 0.05,
            "minimum_reliability": reliability_threshold,
            "per_scene_feature_ood_q99": ood_scene_quantiles,
            "feature_ood_raw_limit": raw_ood_limit,
        },
        "deployment_config": deployment_config,
        "source_proven_vs_interface_constants": {
            "source_proven": [
                "raw_edge_probability_minimum",
                "epsilon_logit",
                "calibrated_edge_lower_minimum",
                "novel_mass_reference",
                "null_step_thresholds",
                "minimum_reliability",
                "feature_ood_raw_limit",
            ],
            "conservative_interface_constants_not_source_query_calibrated": [
                "o0_core_supermajority",
                "anchor_quorum",
                "maximum_null_activation",
                "minimum_stability",
            ],
            "query_top_tail": "diagnostic_not_used_by_primary",
        },
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
