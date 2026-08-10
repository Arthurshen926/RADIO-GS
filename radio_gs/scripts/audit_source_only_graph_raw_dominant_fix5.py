#!/usr/bin/env python3
"""Audit aggregate raw-unary dominant-query graph gating on source scenes.

This source-only audit uses each region's immutable accepted-v2 aggregate
prototype and the frozen generic 806-query/canonical-negative banks.  The
prototype's raw canonical dominant query is a source analogue of the target
region-core mean raw canonical dominant query.  It is not a primitive-majority
calibration and does not open benchmark data.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from radio_gs.interfaces import lerf_raw_unary_region_specificity as unary
from radio_gs.scripts import audit_source_only_graph_expected_utility_fix4 as fix4
from radio_gs.scripts import calibrate_source_only_graph_confidence_v1 as fix2
from radio_gs.scripts import calibrate_source_only_graph_consumer_exact_fix3 as fix3
from radio_gs.scripts import finalize_source_only_graph_positive_utility_fix4b as fix4b
from radio_gs.scripts import eval_lerf_direct_3d_selection as frozen
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
    write_frozen_json,
)


AUTHORITY_SCHEMA = "radio_gs.source_only_graph_raw_dominant_fix5_execution_authority.v1"
RESULT_SCHEMA = "radio_gs.source_only_graph_raw_dominant_fix5_audit.v1"


def source_access() -> dict[str, bool]:
    return {
        "source_instance_labels_opened": True,
        "source_train_pair_features_opened": True,
        "source_validation_pair_features_opened": True,
        "source_aggregate_prototypes_opened": True,
        "generic_target_blind_text_bank_opened": True,
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
        "raw_unary_interface",
        "raw_unary_contract_sha256",
        "graph_calibration_authority",
        "fix4b_result",
        "source_v21b_authority",
        "method",
        "promotion_gate",
        "source_access",
        "benchmark_execution_authorized",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("FIX5 source execution authority fields differ")
    authority = dict(value)
    if (
        authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("status") != "authorized_source_only_raw_dominant_fix5"
        or authority.get("raw_unary_contract_sha256") != unary.CONTRACT_SHA256
        or authority.get("source_access") != source_access()
        or authority.get("benchmark_execution_authorized") is not False
        or authority.get("method")
        != {
            "source_unary": "accepted_v2_aggregate_prototype_raw_canonical_probability",
            "dominance": "region_aggregate_argmax_all_exact_ties_retained",
            "edge_gate": "both_endpoint_dominant_query_intersection_nonempty",
            "consumer": "exact_FIX4B_three_step_unique_marginal_recurrence",
            "primitive_majority_used": False,
            "target_threshold_fitted": False,
        }
        or authority.get("promotion_gate")
        != {
            "minimum_every_validation_scene_Wilson95_lower": 0.95,
            "validation_pooled_marginal_weighted_signed_utility": "strictly_greater_than_zero",
            "validation_true_anchor_retained_reach": "strictly_greater_than_FIX3",
            "failure_action": "reject_FIX5_do_not_open_target",
        }
    ):
        raise ValueError("FIX5 source execution authority header differs")
    for name in (
        "implementation",
        "raw_unary_interface",
        "graph_calibration_authority",
        "fix4b_result",
        "source_v21b_authority",
    ):
        authority[name] = _record(authority[name], label=name)
    return authority


def _load_bank(record: Mapping[str, str], *, label: str) -> dict[str, Any]:
    raw, _, _ = load_torch_mapping(
        record["path"],
        expected_sha256=record["sha256"],
        map_location="cpu",
        label=label,
    )
    if "embeddings" not in raw:
        raise ValueError(f"{label} lacks embeddings")
    embeddings = torch.as_tensor(raw["embeddings"]).detach().float().cpu().contiguous()
    if (
        embeddings.ndim != 2
        or min(embeddings.shape) <= 0
        or not bool(torch.isfinite(embeddings).all())
    ):
        raise ValueError(f"{label} embedding axis differs")
    result = dict(raw)
    result["embeddings"] = embeddings
    return result


def _dominant_query_mask(
    *,
    prototypes: torch.Tensor,
    positive_text: torch.Tensor,
    negative_text: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    descriptor = torch.as_tensor(prototypes).detach().float().cpu().contiguous()
    if (
        descriptor.ndim != 2
        or descriptor.shape[1] != positive_text.shape[1]
        or descriptor.shape[1] != negative_text.shape[1]
        or not bool(torch.isfinite(descriptor).all())
    ):
        raise ValueError("source aggregate prototype axis differs")
    descriptor_gpu = descriptor.to(device)
    positive_gpu = positive_text.to(device)
    negative_gpu = negative_text.to(device)
    positive_score = descriptor_gpu @ positive_gpu.T
    negative_score = (descriptor_gpu @ negative_gpu.T).amax(dim=1, keepdim=True)
    raw = torch.sigmoid((positive_score - negative_score) * 10.0).cpu()
    dominant = raw == raw.amax(dim=1, keepdim=True)
    if not bool(dominant.any(dim=1).all()):
        raise RuntimeError("source aggregate dominant query is empty")
    return dominant.bool().contiguous()


def _selected_rows(
    trace: Mapping[str, torch.Tensor], selected: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    keep = torch.as_tensor(selected).detach().bool().cpu()
    return (
        trace["gains"][keep].float().cpu(),
        trace["marginal_primitives"][keep].long().cpu(),
        trace["labels"][keep].bool().cpu(),
        keep,
    )


def _audit_trace(
    trace: Mapping[str, torch.Tensor], selected: torch.Tensor
) -> dict[str, Any]:
    gain, marginal, labels, keep = _selected_rows(trace, selected)
    count = int(keep.sum())
    positive = int(labels.sum())
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
        "selected_count_by_step": [
            int(keep[:, step].sum()) for step in range(keep.shape[1])
        ],
        "mean_selected_gain": float(gain.mean()),
        "mean_selected_marginal_primitives": float(marginal.float().mean()),
        "marginal_weighted_signed_utility": signed,
    }


def _combine(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    count = sum(int(row["selected_count"]) for row in rows)
    positive = sum(int(row["selected_positive_count"]) for row in rows)
    signed_sum = sum(
        float(row["marginal_weighted_signed_utility"]["signed_gain_sum"])
        for row in rows
    )
    marginal = sum(
        float(row["marginal_weighted_signed_utility"]["marginal_weight_sum"])
        for row in rows
    )
    original_true_eligible = sum(int(row["original_true_eligible_anchor_count"]) for row in rows)
    retained_true_reached = sum(int(row["retained_true_reached_anchor_count"]) for row in rows)
    return {
        "scene_count": len(rows),
        "selected_count": count,
        "selected_positive_count": positive,
        "selected_negative_count": count - positive,
        "selected_precision": positive / count,
        "selected_precision_Wilson95_lower": fix2.one_sided_wilson_lower(
            positive, count
        ),
        "selected_count_by_step": [
            sum(int(row["selected_count_by_step"][step]) for row in rows)
            for step in range(fix3.STEPS)
        ],
        "validation_true_anchor_retained_reach": retained_true_reached
        / original_true_eligible,
        "marginal_weighted_signed_utility": {
            "signed_gain_sum": signed_sum,
            "marginal_weight_sum": marginal,
            "marginal_weighted_signed_utility": signed_sum / marginal,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"FIX5 source result exists: {output}")
    raw_authority, authority_sha, authority_path = load_json_object(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
        label="FIX5 source execution authority",
    )
    authority = validate_execution_authority(raw_authority)
    if authority["implementation"] != file_record(Path(__file__).resolve()):
        raise ValueError("FIX5 source implementation binding differs")
    if authority["raw_unary_interface"] != file_record(Path(unary.__file__).resolve()):
        raise ValueError("FIX5 raw-unary interface binding differs")

    graph_raw, _, _ = load_json_object(
        authority["graph_calibration_authority"]["path"],
        expected_sha256=authority["graph_calibration_authority"]["sha256"],
        label="FIX5 graph calibration authority",
    )
    graph_authority = fix2.validate_execution_authority(graph_raw)
    checkpoint, loaded = fix2._load_inputs(graph_authority)
    fix4b_result, _, _ = load_json_object(
        authority["fix4b_result"]["path"],
        expected_sha256=authority["fix4b_result"]["sha256"],
        label="FIX5 parent FIX4B result",
    )
    if fix4b_result.get("status") != "source_only_positive_utility_fix4b_promoted_target_unopened":
        raise ValueError("FIX5 requires promoted FIX4B")
    deployment = fix4b_result["deployment_config"]
    config = deployment["positive_utility_config"]

    v21b, _, _ = load_json_object(
        authority["source_v21b_authority"]["path"],
        expected_sha256=authority["source_v21b_authority"]["sha256"],
        label="FIX5 source V2.1B authority",
    )
    if v21b.get("schema") != "radio_gs.surface_region_v21b_conditioned_rank256_exact4x2_execution_authority.v1":
        raise ValueError("FIX5 source V2.1B authority schema differs")
    fit_record = _record(v21b["fit_text_bank"], label="FIX5 fit text bank")
    negative_record = _record(v21b["canonical_negative_bank"], label="FIX5 negative bank")
    fit = _load_bank(fit_record, label="FIX5 generic fit text bank")
    negative = _load_bank(negative_record, label="FIX5 canonical negative bank")
    if fit["embeddings"].shape[1] != negative["embeddings"].shape[1]:
        raise ValueError("FIX5 positive/negative text dimensions differ")
    shards = {
        str(item["scene_id"]): _record(item["training_shard"], label=f"FIX5 {item['scene_id']} shard")
        for split in ("source_train", "source_validation")
        for item in v21b[split]
    }
    if set(shards) != {scene.scene_id for scene, _ in loaded}:
        raise ValueError("FIX5 source graph and aggregate prototype scenes differ")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("FIX5 requested CUDA but CUDA is unavailable")
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("highest")
    per_scene: dict[str, Any] = {}
    for scene, raw_probability in loaded:
        shard, _, _ = load_torch_mapping(
            shards[scene.scene_id]["path"],
            expected_sha256=shards[scene.scene_id]["sha256"],
            map_location="cpu",
            label=f"FIX5 {scene.scene_id} aggregate prototype shard",
        )
        prototypes = torch.as_tensor(shard["accepted_v2_e0"]).detach().float().cpu()
        if prototypes.shape[0] != scene.region_count:
            raise ValueError("FIX5 source prototype/graph region axis differs")
        dominant = _dominant_query_mask(
            prototypes=prototypes,
            positive_text=fit["embeddings"],
            negative_text=negative["embeddings"],
            device=device,
        )
        pair_left = scene.pair_indices[0].long()
        pair_right = scene.pair_indices[1].long()
        shared_dominant = (dominant[pair_left] & dominant[pair_right]).any(dim=1)
        eligible, edge_gate = fix4.edge_eligible_mask(
            pair_features=scene.pair_features,
            raw_probability=raw_probability,
            median=checkpoint["normalization"]["median"],
            robust_scale=checkpoint["normalization"]["robust_scale"],
            raw_probability_minimum=fix2.RAW_EDGE_PROBABILITY_MINIMUM,
            reliability_minimum=float(config["minimum_reliability"]),
            ood_raw_limit=float(deployment["feature_OOD"]["raw_score_limit"]),
        )
        lower = fix2.lower_probability(raw_probability, float(config["epsilon_logit"]))
        base_trace = fix3.exact_direct_edge_trace(
            scene=scene,
            probability_lower=lower,
            edge_eligible_mask=eligible,
            target_filter=None,
        )
        base_selected = fix3.apply_strict_sequential_thresholds(
            base_trace["gains"], unary_thresholds := (0.0, 0.0, 0.0)
        )
        gated_eligible = eligible & shared_dominant
        gated_trace = fix3.exact_direct_edge_trace(
            scene=scene,
            probability_lower=lower,
            edge_eligible_mask=gated_eligible,
            target_filter=None,
        )
        gated_selected = fix3.apply_strict_sequential_thresholds(
            gated_trace["gains"], unary_thresholds
        )
        original_true = fix3.exact_direct_edge_trace(
            scene=scene,
            probability_lower=lower,
            edge_eligible_mask=eligible,
            target_filter=True,
        )
        gated_true = fix3.exact_direct_edge_trace(
            scene=scene,
            probability_lower=lower,
            edge_eligible_mask=gated_eligible,
            target_filter=True,
        )
        gated_true_selected = fix3.apply_strict_sequential_thresholds(
            gated_true["gains"], unary_thresholds
        )
        base_audit = _audit_trace(base_trace, base_selected)
        frozen_parent = fix4b_result["source_exact_consumer_audit"]["per_scene"][scene.scene_id]
        if (
            base_audit["selected_count"] != frozen_parent["selected_edge_count"]
            or base_audit["selected_positive_count"]
            != frozen_parent["selected_positive_edge_count"]
        ):
            raise RuntimeError("FIX5 base replay differs from frozen FIX4B")
        audit = _audit_trace(gated_trace, gated_selected)
        original_eligible = int(original_true["has_candidate"].sum())
        retained_reached = int(
            (gated_true_selected.any(dim=1) & original_true["has_candidate"]).sum()
        )
        audit.update(
            {
                "split": scene.split,
                "edge_gate": edge_gate,
                "base_FIX4B": base_audit,
                "raw_dominant_query_count": int(dominant.shape[1]),
                "shared_dominant_eligible_edge_count": int(gated_eligible.sum()),
                "original_true_eligible_anchor_count": original_eligible,
                "retained_true_reached_anchor_count": retained_reached,
                "true_anchor_retained_reach": retained_reached / original_eligible,
            }
        )
        per_scene[scene.scene_id] = audit

    by_split = {
        split: _combine(
            [row for row in per_scene.values() if row["split"] == split]
        )
        for split in ("source_train", "source_validation")
    }
    validation = by_split["source_validation"]
    fix3_reach = float(
        fix4b_result["promotion_gate"]["fix3_validation_true_pseudo_anchor_reach"]
    )
    outcomes = {
        "both_validation_scene_Wilson95_lower_at_least_0.95": all(
            per_scene[scene_id]["selected_precision_Wilson95_lower"] >= 0.95
            for scene_id in fix2.VALIDATION_SCENES
        ),
        "validation_pooled_marginal_weighted_signed_utility_positive": float(
            validation["marginal_weighted_signed_utility"]["marginal_weighted_signed_utility"]
        )
        > 0.0,
        "validation_true_anchor_retained_reach_strictly_exceeds_FIX3": float(
            validation["validation_true_anchor_retained_reach"]
        )
        > fix3_reach,
    }
    outcomes["passed"] = all(outcomes.values())
    result = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "status": (
            "source_only_raw_dominant_FIX5_promoted_target_unopened"
            if outcomes["passed"]
            else "source_only_raw_dominant_FIX5_rejected_target_must_remain_unopened"
        ),
        "execution_authority": {"path": str(authority_path), "sha256": authority_sha},
        "method_claim": (
            "source_aggregate_prototype_raw_canonical_dominant_query_gate_is_"
            "consumer_exact_safe;source_does_not_calibrate_primitive_majority"
        ),
        "raw_unary_interface": file_record(Path(unary.__file__).resolve()),
        "raw_unary_contract_sha256": unary.CONTRACT_SHA256,
        "source_banks": {"fit_text_bank": fit_record, "canonical_negative_bank": negative_record},
        "per_scene": per_scene,
        "by_split": by_split,
        "promotion_gate": {
            "thresholds": authority["promotion_gate"],
            "fix3_validation_true_anchor_reach": fix3_reach,
            "outcomes": outcomes,
        },
        "source_target_analogy_boundary": {
            "source": "accepted_v2_aggregate_prototype_raw_canonical_dominant_query",
            "target": "region_valid_core_mean_raw_canonical_dominant_query",
            "primitive_majority_calibrated": False,
            "fraction_threshold": None,
        },
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
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(result["status"])
    print(result["promotion_gate"]["outcomes"])


if __name__ == "__main__":
    main()
