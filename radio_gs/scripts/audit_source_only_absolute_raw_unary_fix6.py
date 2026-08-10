#!/usr/bin/env python3
"""Audit and, only if safe, promote a parameter-free absolute raw unary.

The candidate is cardinality-controlled: a region/query raw canonical score
must exceed the semantic neutral point 0.5 and the query must dominate the same
frozen 806-query target-blind distractor bank.  Official source multiview
teacher responses to that exact query provide the source-only absolute-unary
truth.  Target scenes, queries, masks, images, and metrics remain unopened.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from radio_gs.scripts import audit_source_only_graph_expected_utility_fix4 as fix4
from radio_gs.scripts import calibrate_source_only_graph_confidence_v1 as fix2
from radio_gs.scripts import calibrate_source_only_graph_consumer_exact_fix3 as fix3
from radio_gs.scripts import finalize_source_only_graph_positive_utility_fix4b as fix4b
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
    write_frozen_json,
)


AUTHORITY_SCHEMA = "radio_gs.source_only_absolute_raw_unary_fix6_execution_authority.v1"
RESULT_SCHEMA = "radio_gs.source_only_absolute_raw_unary_fix6_audit.v1"
NEUTRAL_PROBABILITY = 0.5
LOGIT_SCALE = 10.0


def source_access() -> dict[str, bool]:
    return {
        "source_aggregate_prototypes_opened": True,
        "source_official_multiview_teacher_opened": True,
        "source_instance_labels_opened": True,
        "source_train_pair_features_opened": True,
        "source_validation_pair_features_opened": True,
        "generic_target_blind_806_query_bank_opened": True,
        "canonical_generic_negative_bank_opened": True,
        "benchmark_images_opened": False,
        "benchmark_labels_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_queries_opened": False,
        "benchmark_metrics_opened": False,
        "target_heldout_opened": False,
        "target_metrics_computed": False,
        "per_scene_hyperparameters": False,
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
        "graph_calibration_authority",
        "fix4b_result",
        "fix5_result",
        "source_v21b_authority",
        "fixed_candidate",
        "promotion_gate",
        "source_access",
        "benchmark_execution_authorized",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("FIX6 source authority fields differ")
    authority = dict(value)
    if (
        authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("status") != "authorized_source_only_absolute_raw_unary_FIX6"
        or authority.get("source_access") != source_access()
        or authority.get("benchmark_execution_authorized") is not False
        or authority.get("fixed_candidate")
        != {
            "raw_probability": "sigmoid(10*(query_cosine-hardest_canonical_negative_cosine))",
            "absolute_threshold": NEUTRAL_PROBABILITY,
            "threshold_source": "semantic_neutral_point_not_fitted",
            "dominance_reference": "same_frozen_target_blind_806_query_bank",
            "dominance": "argmax_all_exact_ties_retained",
            "target_requirement": "runtime_query_must_beat_the_same_806_distractors",
            "scene_query_bank_argmax_used": False,
            "per_scene_parameters": False,
        }
        or authority.get("promotion_gate")
        != {
            "minimum_every_validation_scene_unary_Wilson95_lower": 0.95,
            "validation_pooled_signed_absolute_utility": "strictly_greater_than_zero",
            "minimum_every_validation_scene_graph_Wilson95_lower": 0.95,
            "every_validation_scene_confirmed_anchor_coverage": "at_least_FIX5_retained_reach",
            "failure_action": "reject_FIX6_keep_target_unopened",
        }
    ):
        raise ValueError("FIX6 source authority header differs")
    for name in (
        "implementation",
        "graph_calibration_authority",
        "fix4b_result",
        "fix5_result",
        "source_v21b_authority",
    ):
        authority[name] = _record(authority[name], label=name)
    return authority


def _teacher_probability_for_dominant_query(
    *,
    teacher_descriptors: torch.Tensor,
    teacher_region_indices: torch.Tensor,
    dominant_query_indices: torch.Tensor,
    positive_text: torch.Tensor,
    negative_text: torch.Tensor,
    region_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    descriptor = torch.as_tensor(teacher_descriptors).detach().float().cpu()
    region = torch.as_tensor(teacher_region_indices).detach().long().cpu()
    query = torch.as_tensor(dominant_query_indices).detach().long().cpu()
    if (
        descriptor.ndim != 2
        or region.shape != (descriptor.shape[0],)
        or query.shape != (region_count,)
        or descriptor.shape[1] != positive_text.shape[1]
        or descriptor.shape[1] != negative_text.shape[1]
        or bool((region < 0).any())
        or bool((region >= region_count).any())
    ):
        raise ValueError("FIX6 source teacher response axes differ")
    positive = (descriptor * positive_text[query[region]]).sum(dim=1)
    negative = (descriptor @ negative_text.T).amax(dim=1)
    probability = torch.sigmoid((positive - negative) * LOGIT_SCALE)
    total = torch.zeros(region_count, dtype=torch.float32)
    count = torch.zeros(region_count, dtype=torch.float32)
    total.index_add_(0, region, probability)
    count.index_add_(0, region, torch.ones_like(probability))
    mean = total / count.clamp_min(1.0)
    return mean.contiguous(), count.long().contiguous()


def _unary_audit(
    *,
    maximum_probability: torch.Tensor,
    candidate: torch.Tensor,
    teacher_true: torch.Tensor,
) -> dict[str, Any]:
    probability = torch.as_tensor(maximum_probability).detach().float().cpu()
    keep = torch.as_tensor(candidate).detach().bool().cpu()
    truth = torch.as_tensor(teacher_true).detach().bool().cpu()
    if probability.shape != keep.shape or keep.shape != truth.shape:
        raise ValueError("FIX6 unary audit axes differ")
    count = int(keep.sum())
    positive = int((keep & truth).sum())
    strength = (2.0 * probability[keep] - 1.0).clamp_min(0.0)
    sign = torch.where(
        truth[keep], torch.ones_like(strength), -torch.ones_like(strength)
    )
    signed_sum = float((sign * strength).sum())
    strength_sum = float(strength.sum())
    teacher_count = int(truth.sum())
    return {
        "candidate_count": count,
        "teacher_positive_count": positive,
        "teacher_negative_count": count - positive,
        "precision": positive / count,
        "precision_Wilson95_lower": fix2.one_sided_wilson_lower(positive, count),
        "teacher_positive_reference_count": teacher_count,
        "teacher_positive_recall": positive / teacher_count,
        "signed_strength_sum": signed_sum,
        "strength_sum": strength_sum,
        "signed_absolute_utility_per_candidate": signed_sum / count,
        "signed_strength_ratio": signed_sum / max(strength_sum, 1e-12),
    }


def _graph_audit(trace: Mapping[str, torch.Tensor], selected: torch.Tensor) -> dict[str, Any]:
    keep = torch.as_tensor(selected).detach().bool().cpu()
    count = int(keep.sum())
    positive = int((keep & trace["labels"]).sum())
    signed = fix4b.marginal_weighted_signed_utility(
        gains=trace["gains"],
        marginal_primitives=trace["marginal_primitives"],
        labels=trace["labels"],
        selected=keep,
        novel_mass_reference=fix3.NOVEL_MASS_REFERENCE,
    )
    return {
        "selected_count": count,
        "selected_positive_count": positive,
        "selected_negative_count": count - positive,
        "selected_precision": positive / count,
        "selected_precision_Wilson95_lower": fix2.one_sided_wilson_lower(
            positive, count
        ),
        "marginal_weighted_signed_utility": signed,
    }


def _pool(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    count = sum(int(row["unary"]["candidate_count"]) for row in rows)
    positive = sum(int(row["unary"]["teacher_positive_count"]) for row in rows)
    teacher_count = sum(
        int(row["unary"]["teacher_positive_reference_count"]) for row in rows
    )
    signed = sum(float(row["unary"]["signed_strength_sum"]) for row in rows)
    strength = sum(float(row["unary"]["strength_sum"]) for row in rows)
    graph_count = sum(int(row["graph_conditioned"]["selected_count"]) for row in rows)
    graph_positive = sum(
        int(row["graph_conditioned"]["selected_positive_count"]) for row in rows
    )
    return {
        "scene_count": len(rows),
        "unary": {
            "candidate_count": count,
            "teacher_positive_count": positive,
            "teacher_negative_count": count - positive,
            "precision": positive / count,
            "precision_Wilson95_lower": fix2.one_sided_wilson_lower(
                positive, count
            ),
            "teacher_positive_reference_count": teacher_count,
            "teacher_positive_recall": positive / teacher_count,
            "signed_strength_sum": signed,
            "strength_sum": strength,
            "signed_absolute_utility_per_candidate": signed / count,
            "signed_strength_ratio": signed / strength,
        },
        "graph_conditioned": {
            "selected_count": graph_count,
            "selected_positive_count": graph_positive,
            "selected_negative_count": graph_count - graph_positive,
            "selected_precision": graph_positive / graph_count,
            "selected_precision_Wilson95_lower": fix2.one_sided_wilson_lower(
                graph_positive, graph_count
            ),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"FIX6 source result exists: {output}")
    raw_authority, authority_sha, authority_path = load_json_object(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
        label="FIX6 source execution authority",
    )
    authority = validate_execution_authority(raw_authority)
    if authority["implementation"] != file_record(Path(__file__).resolve()):
        raise ValueError("FIX6 source implementation binding differs")
    graph_raw, _, _ = load_json_object(
        authority["graph_calibration_authority"]["path"],
        expected_sha256=authority["graph_calibration_authority"]["sha256"],
        label="FIX6 graph calibration authority",
    )
    graph_authority = fix2.validate_execution_authority(graph_raw)
    checkpoint, loaded = fix2._load_inputs(graph_authority)
    fix4b_result, _, _ = load_json_object(
        authority["fix4b_result"]["path"],
        expected_sha256=authority["fix4b_result"]["sha256"],
        label="FIX6 FIX4B result",
    )
    fix5_result, _, _ = load_json_object(
        authority["fix5_result"]["path"],
        expected_sha256=authority["fix5_result"]["sha256"],
        label="FIX6 FIX5 result",
    )
    if (
        fix4b_result.get("status")
        != "source_only_positive_utility_fix4b_promoted_target_unopened"
        or fix5_result.get("status")
        != "source_only_raw_dominant_FIX5_promoted_target_unopened"
    ):
        raise ValueError("FIX6 parent source promotion differs")
    deployment = fix4b_result["deployment_config"]
    config = deployment["positive_utility_config"]
    v21b, _, _ = load_json_object(
        authority["source_v21b_authority"]["path"],
        expected_sha256=authority["source_v21b_authority"]["sha256"],
        label="FIX6 source V2.1B authority",
    )
    fit_record = _record(v21b["fit_text_bank"], label="FIX6 fit text bank")
    negative_record = _record(
        v21b["canonical_negative_bank"], label="FIX6 canonical negative bank"
    )
    fit_raw, _, _ = load_torch_mapping(
        fit_record["path"],
        expected_sha256=fit_record["sha256"],
        map_location="cpu",
        label="FIX6 fixed 806-query bank",
    )
    negative_raw, _, _ = load_torch_mapping(
        negative_record["path"],
        expected_sha256=negative_record["sha256"],
        map_location="cpu",
        label="FIX6 canonical negative bank",
    )
    fit = torch.as_tensor(fit_raw["embeddings"]).detach().float().cpu()
    negative = torch.as_tensor(negative_raw["embeddings"]).detach().float().cpu()
    if fit.shape != (806, 1536) or negative.ndim != 2 or negative.shape[1] != 1536:
        raise ValueError("FIX6 fixed bank dimensions differ")
    shards = {
        str(item["scene_id"]): _record(
            item["training_shard"], label=f"FIX6 {item['scene_id']} shard"
        )
        for split in ("source_train", "source_validation")
        for item in v21b[split]
    }
    if set(shards) != {scene.scene_id for scene, _ in loaded}:
        raise ValueError("FIX6 source scene bindings differ")

    per_scene: dict[str, Any] = {}
    for scene, raw_edge_probability in loaded:
        shard, _, _ = load_torch_mapping(
            shards[scene.scene_id]["path"],
            expected_sha256=shards[scene.scene_id]["sha256"],
            map_location="cpu",
            label=f"FIX6 {scene.scene_id} source shard",
        )
        prototype = torch.as_tensor(shard["accepted_v2_e0"]).detach().float().cpu()
        positive_score = prototype @ fit.T
        negative_score = (prototype @ negative.T).amax(dim=1, keepdim=True)
        probability = torch.sigmoid(
            (positive_score - negative_score) * LOGIT_SCALE
        )
        dominant = probability == probability.amax(dim=1, keepdim=True)
        absolute = probability > NEUTRAL_PROBABILITY
        candidate_pairs = dominant & absolute
        if bool((candidate_pairs.sum(dim=1) > 1).any()):
            raise RuntimeError("FIX6 source observed an exact dominant-query tie")
        maximum_probability, query = probability.max(dim=1)
        candidate = candidate_pairs.any(dim=1)
        teacher_probability, teacher_count = _teacher_probability_for_dominant_query(
            teacher_descriptors=shard[
                "official_multiview_siglip2_teacher_pair_descriptors"
            ],
            teacher_region_indices=shard[
                "official_multiview_siglip2_teacher_pair_region_indices"
            ],
            dominant_query_indices=query,
            positive_text=fit,
            negative_text=negative,
            region_count=scene.region_count,
        )
        teacher_true = (teacher_probability > NEUTRAL_PROBABILITY) & (
            teacher_count > 0
        )
        unary_audit = _unary_audit(
            maximum_probability=maximum_probability,
            candidate=candidate,
            teacher_true=teacher_true,
        )

        eligible, edge_gate = fix4.edge_eligible_mask(
            pair_features=scene.pair_features,
            raw_probability=raw_edge_probability,
            median=checkpoint["normalization"]["median"],
            robust_scale=checkpoint["normalization"]["robust_scale"],
            raw_probability_minimum=fix2.RAW_EDGE_PROBABILITY_MINIMUM,
            reliability_minimum=float(config["minimum_reliability"]),
            ood_raw_limit=float(deployment["feature_OOD"]["raw_score_limit"]),
        )
        lower = fix2.lower_probability(
            raw_edge_probability, float(config["epsilon_logit"])
        )
        left, right = scene.pair_indices[0].long(), scene.pair_indices[1].long()
        same_dominant = query[left] == query[right]
        absolute_edge = candidate[left] & candidate[right]
        graph_trace = fix3.exact_direct_edge_trace(
            scene=scene,
            probability_lower=lower,
            edge_eligible_mask=eligible & same_dominant & absolute_edge,
            target_filter=None,
        )
        graph_selected = fix3.apply_strict_sequential_thresholds(
            graph_trace["gains"], (0.0, 0.0, 0.0)
        )
        original_true = fix3.exact_direct_edge_trace(
            scene=scene,
            probability_lower=lower,
            edge_eligible_mask=eligible,
            target_filter=True,
        )
        original_anchor = original_true["has_candidate"]
        confirmed_absolute = candidate & teacher_true
        confirmed_coverage = float(
            (confirmed_absolute & original_anchor).sum() / original_anchor.sum()
        )
        fix5_reach = float(
            fix5_result["per_scene"][scene.scene_id]["true_anchor_retained_reach"]
        )
        per_scene[scene.scene_id] = {
            "split": scene.split,
            "fixed_distractor_query_count": int(fit.shape[0]),
            "absolute_raw_region_query_pair_count_before_dominance": int(
                absolute.sum()
            ),
            "dominant_absolute_region_count": int(candidate.sum()),
            "dominant_exact_tie_region_count": int(
                (candidate_pairs.sum(dim=1) > 1).sum()
            ),
            "unary": unary_audit,
            "graph_conditioned": _graph_audit(graph_trace, graph_selected),
            "edge_gate": edge_gate,
            "original_true_eligible_anchor_count": int(original_anchor.sum()),
            "teacher_confirmed_absolute_anchor_count": int(
                (confirmed_absolute & original_anchor).sum()
            ),
            "teacher_confirmed_absolute_anchor_coverage": confirmed_coverage,
            "FIX5_retained_true_anchor_reach": fix5_reach,
            "coverage_minus_FIX5": confirmed_coverage - fix5_reach,
        }
    by_split = {
        split: _pool(
            [row for row in per_scene.values() if row["split"] == split]
        )
        for split in ("source_train", "source_validation")
    }
    validation = by_split["source_validation"]
    outcomes = {
        "both_validation_scene_unary_Wilson95_lower_at_least_0.95": all(
            per_scene[scene_id]["unary"]["precision_Wilson95_lower"] >= 0.95
            for scene_id in fix2.VALIDATION_SCENES
        ),
        "validation_pooled_signed_absolute_utility_positive": float(
            validation["unary"]["signed_absolute_utility_per_candidate"]
        )
        > 0.0,
        "both_validation_scene_graph_Wilson95_lower_at_least_0.95": all(
            per_scene[scene_id]["graph_conditioned"][
                "selected_precision_Wilson95_lower"
            ]
            >= 0.95
            for scene_id in fix2.VALIDATION_SCENES
        ),
        "both_validation_scene_confirmed_anchor_coverage_at_least_FIX5": all(
            per_scene[scene_id]["coverage_minus_FIX5"] >= 0.0
            for scene_id in fix2.VALIDATION_SCENES
        ),
    }
    outcomes["passed"] = all(outcomes.values())
    result = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "status": (
            "source_only_absolute_raw_unary_FIX6_promoted_target_unopened"
            if outcomes["passed"]
            else "source_only_absolute_raw_unary_FIX6_rejected_target_remained_unopened"
        ),
        "execution_authority": {"path": str(authority_path), "sha256": authority_sha},
        "candidate_contract": authority["fixed_candidate"],
        "cardinality_control": {
            "source": "candidate query competes with the frozen 806-query bank",
            "target_if_promoted": "runtime query must compete with the same frozen 806 distractors",
            "scene_query_count_used": False,
        },
        "teacher_truth": (
            "official_multiview_teacher_mean_raw_canonical_probability_for_the_"
            "exact_candidate_query_above_semantic_neutral_point"
        ),
        "per_scene": per_scene,
        "by_split": by_split,
        "promotion_gate": {
            "thresholds": authority["promotion_gate"],
            "outcomes": outcomes,
        },
        "decision": (
            "reject_do_not_implement_or_open_target"
            if not outcomes["passed"]
            else "promote_interface_then_target_noGT_only"
        ),
        "source_access": source_access(),
        "benchmark_execution_authorized": False,
        "target_execution_performed": False,
    }
    result["content_authority_sha256"] = canonical_json_sha256(result)
    write_frozen_json(output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-authority", required=True)
    parser.add_argument("--expected-execution-authority-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(result["status"])
    print(result["promotion_gate"]["outcomes"])


if __name__ == "__main__":
    main()
