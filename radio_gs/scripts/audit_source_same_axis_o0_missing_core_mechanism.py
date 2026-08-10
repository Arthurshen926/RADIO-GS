#!/usr/bin/env python3
"""Audit scene0001 same-axis O0 missing-core utility before heldout access."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from radio_gs.interfaces import source_missing_core_conditional_utility as utility_api
from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces.source_missing_core_conditional_utility import (
    MissingCoreConditionalUtility,
    source_missing_core_conditional_utility,
)
from radio_gs.scripts.build_lerf_o0_anchored_graph_residual_cache import (
    exact_o0_readout,
)
from radio_gs.scripts import build_lerf_o0_anchored_graph_residual_cache as exact_o0_api
from radio_gs.scripts.calibrate_source_only_graph_confidence_v1 import (
    one_sided_wilson_lower,
)
from radio_gs.scripts.train_source_region_comembership_v2 import (
    load_scene_authority_v2,
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


AUTHORITY_SCHEMA = "radio_gs.source_same_axis_o0_mechanism_audit_authority.v1"
RESULT_SCHEMA = "radio_gs.source_same_axis_o0_mechanism_audit.v1"
KNN_CHUNK_SIZE = 65536
FEATURE_NAMES = (
    "unit_o0_score",
    "unit_margin_below_0p6",
    "missing_min_seed_distance_over_seed_nn_median",
    "missing_mean_seed_distance_over_seed_nn_median",
    "missing_core_knn10_seed_density",
    "anchor_positive_fraction",
    "log1p_anchor_missing_count",
    "log1p_valid_core_count",
    "region_scale_index",
    "query_selected_scale_index",
    "missing_score_q25",
    "missing_score_q50",
    "missing_score_q75",
    "missing_score_maximum",
    "appearance_concentration",
    "boundary_concentration",
    "boundary_dispersion_one_minus_concentration",
    "core_spatial_rms_radius",
    "full_scalar_source_robust_ood_linf",
) + tuple(f"full_scalar_source_robust_{index:02d}" for index in range(18))


def source_access() -> dict[str, bool]:
    return {
        "source_train_instance_labels_opened_after_O0_frozen": True,
        "source_validation_instance_labels_opened": False,
        "source_accepted_v2_opened": True,
        "source_query_independent_capability_reliability_opened": True,
        "source_train_full_scalar_shard_opened": True,
        "benchmark_images_opened": False,
        "benchmark_labels_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_queries_opened": False,
        "benchmark_metrics_opened": False,
        "target_heldout_opened": False,
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
        "scene_id",
        "split",
        "implementation",
        "utility_interface",
        "exact_o0_implementation",
        "parent_authority",
        "dense_stream_execution_authority",
        "combined_text_subset",
        "raw_combined_multiscale_scores",
        "accepted_v2",
        "source_membership_authority",
        "source_training_shard",
        "source_capability_descriptor",
        "fixed_audit",
        "outputs",
        "source_access",
        "benchmark_execution_authorized",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("same-axis O0 mechanism authority fields differ")
    authority = dict(value)
    expected_audit = {
        "positive_query_count": 64,
        "canonical_negative_count": 4,
        "canonical_negative_probability_logit_scale": 10.0,
        "knn": 10,
        "knn_chunk_size": KNN_CHUNK_SIZE,
        "scale_selection": "highest_raw_smoothed_peak_lower_scale_tie_break",
        "per_scale_query_minmax": True,
        "o0_positive": "strictly_greater_than_0p6",
        "qualified_anchor": "valid_core_positive_fraction_at_least_0p75",
        "missing_core": "valid_core_O0_score_less_than_or_equal_to_0p6",
        "hard_label": "primitive_exact_mass_argmax_equals_anchor_dominant_instance",
        "soft_utility": "2_times_target_instance_mass_fraction_minus_1",
        "feature_names": list(FEATURE_NAMES),
        "scene0001_gate": {
            "minimum_qualified_anchor_query_pairs": 32,
            "minimum_missing_core_units": 256,
            "minimum_hard_positive_missing_units": 32,
            "minimum_hard_negative_missing_units": 32,
        },
        "heldout_scene_opened_before_gate": False,
    }
    if (
        authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("status")
        != "sealed_after_raw_O0_scores_before_source_instance_utility_open"
        or authority.get("scene_id") != "scene0001_00"
        or authority.get("split") != "source_train"
        or authority.get("fixed_audit") != expected_audit
        or authority.get("source_access") != source_access()
        or authority.get("benchmark_execution_authorized") is not False
    ):
        raise ValueError("same-axis O0 mechanism authority header differs")
    for name in (
        "implementation",
        "utility_interface",
        "exact_o0_implementation",
        "parent_authority",
        "dense_stream_execution_authority",
        "combined_text_subset",
        "raw_combined_multiscale_scores",
        "accepted_v2",
        "source_membership_authority",
        "source_training_shard",
        "source_capability_descriptor",
    ):
        authority[name] = _record(authority[name], label=name)
    outputs = authority.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {
        "unit_table",
        "report",
    }:
        raise ValueError("same-axis O0 mechanism outputs differ")
    authority["outputs"] = {
        name: str(Path(path).expanduser().resolve()) for name, path in outputs.items()
    }
    return authority


def _source_robust_normalize(
    values: torch.Tensor, eligible: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    scalar = torch.as_tensor(values).detach().float().cpu()
    keep = torch.as_tensor(eligible).detach().bool().cpu()
    if scalar.ndim != 2 or keep.shape != (scalar.shape[0],) or not bool(keep.any()):
        raise ValueError("source full-scalar normalization inputs differ")
    selected = scalar[keep]
    location = selected.median(dim=0).values
    mad = (selected - location).abs().median(dim=0).values
    scale = 1.4826 * mad
    scale[(scale <= 0.0) & (selected.amax(0) > selected.amin(0))] = 1.0
    scale[scale <= 0.0] = 1.0
    normalized = (scalar - location) / scale
    return normalized.contiguous(), normalized.abs().amax(dim=1).contiguous()


def _group_statistics(
    utility: MissingCoreConditionalUtility,
    *,
    region_count: int,
) -> torch.Tensor:
    result = torch.zeros(region_count, 4, dtype=torch.float32)
    for region in torch.unique(utility.unit_region_indices, sorted=True).tolist():
        scores = utility.unit_o0_scores[utility.unit_region_indices == region]
        result[region] = torch.stack(
            (
                torch.quantile(scores, 0.25),
                torch.quantile(scores, 0.50),
                torch.quantile(scores, 0.75),
                scores.max(),
            )
        )
    return result


def build_unit_feature_table(
    *,
    utility: MissingCoreConditionalUtility,
    o0_scores: torch.Tensor,
    primitive_valid_mask: torch.Tensor,
    region_query_indices: torch.Tensor,
    region_rows: torch.Tensor,
    token_mask: torch.Tensor,
    xyz: torch.Tensor,
    region_scale_indices: torch.Tensor,
    selected_query_scale_indices: torch.Tensor,
    appearance_concentration: torch.Tensor,
    boundary_concentration: torch.Tensor,
    raw_full_scalar_summary: torch.Tensor,
    full_scalar_eligible: torch.Tensor,
) -> torch.Tensor:
    """Build fixed proposal features without consulting proposal labels."""

    rows = torch.as_tensor(region_rows).long().cpu()
    mask = torch.as_tensor(token_mask).bool().cpu()
    points = torch.as_tensor(xyz).float().cpu()
    scores = torch.as_tensor(o0_scores).float().cpu()
    valid = torch.as_tensor(primitive_valid_mask).bool().cpu()
    region_query = torch.as_tensor(region_query_indices).long().cpu()
    region_count = rows.shape[0]
    if (
        scores.ndim != 2
        or valid.shape != (scores.shape[0],)
        or region_query.shape != (region_count,)
        or points.shape != (scores.shape[0], 3)
    ):
        raise ValueError("same-axis O0 geometric feature axes differ")
    normalized_scalar, scalar_ood = _source_robust_normalize(
        raw_full_scalar_summary, full_scalar_eligible
    )
    spatial = torch.zeros(region_count, dtype=torch.float32)
    for region in range(region_count):
        active = rows[region, mask[region]]
        cloud = points[active]
        center = cloud.mean(dim=0, keepdim=True)
        spatial[region] = ((cloud - center).square().sum(dim=1).mean()).sqrt()
    missing_stats = _group_statistics(utility, region_count=region_count)
    unit_region = utility.unit_region_indices
    unit_query = utility.unit_query_indices
    if unit_region.numel() == 0:
        return torch.empty(0, len(FEATURE_NAMES), dtype=torch.float32)
    minimum_seed_distance = torch.zeros(unit_region.numel(), dtype=torch.float32)
    mean_seed_distance = torch.zeros_like(minimum_seed_distance)
    knn_seed_density = torch.zeros_like(minimum_seed_distance)
    for region in torch.unique(unit_region, sorted=True).tolist():
        query = int(region_query[region])
        active = rows[region, mask[region]]
        active = active[valid[active]]
        seed = scores[active, query] > 0.6
        seed_rows = active[seed]
        if seed_rows.numel() < 2:
            raise RuntimeError("qualified same-axis O0 region lacks two seeds")
        seed_points = points[seed_rows]
        seed_pairwise = torch.cdist(seed_points, seed_points)
        seed_pairwise.fill_diagonal_(float("inf"))
        seed_scale = seed_pairwise.amin(dim=1).median().clamp_min(1e-6)
        unit_offsets = torch.where(unit_region == region)[0]
        missing_points = points[utility.unit_primitive_rows[unit_offsets]]
        distance_to_seed = torch.cdist(missing_points, seed_points)
        minimum_seed_distance[unit_offsets] = (
            distance_to_seed.amin(dim=1) / seed_scale
        )
        mean_seed_distance[unit_offsets] = (
            distance_to_seed.mean(dim=1) / seed_scale
        )
        distance_to_core = torch.cdist(missing_points, points[active])
        neighbors = distance_to_core.topk(
            k=min(10, int(active.numel())), largest=False, dim=1
        ).indices
        knn_seed_density[unit_offsets] = seed[neighbors].float().mean(dim=1)
    region_scale = torch.as_tensor(region_scale_indices).float().cpu()
    query_scale = torch.as_tensor(selected_query_scale_indices).float().cpu()
    appearance = torch.as_tensor(appearance_concentration).float().cpu()
    boundary = torch.as_tensor(boundary_concentration).float().cpu()
    feature = torch.cat(
        (
            utility.unit_o0_scores[:, None],
            (0.6 - utility.unit_o0_scores)[:, None],
            minimum_seed_distance[:, None],
            mean_seed_distance[:, None],
            knn_seed_density[:, None],
            utility.positive_fraction[unit_region, None],
            torch.log1p(utility.missing_counts[unit_region].float())[:, None],
            torch.log1p(utility.valid_core_counts[unit_region].float())[:, None],
            region_scale[unit_region, None],
            query_scale[unit_query, None],
            missing_stats[unit_region],
            appearance[unit_region, None],
            boundary[unit_region, None],
            (1.0 - boundary[unit_region])[:, None],
            spatial[unit_region, None],
            scalar_ood[unit_region, None],
            normalized_scalar[unit_region],
        ),
        dim=1,
    ).float().contiguous()
    if feature.shape != (unit_region.numel(), len(FEATURE_NAMES)) or not bool(
        torch.isfinite(feature).all()
    ):
        raise RuntimeError("same-axis O0 missing-core feature table differs")
    return feature


def _auc(values: torch.Tensor, labels: torch.Tensor) -> float:
    score = torch.as_tensor(values).float().cpu()
    truth = torch.as_tensor(labels).bool().cpu()
    positives = int(truth.sum())
    negatives = int((~truth).sum())
    if positives == 0 or negatives == 0:
        return 0.5
    order = torch.argsort(score, stable=True)
    sorted_score = score[order]
    ranks = torch.empty_like(score)
    _, counts = torch.unique_consecutive(sorted_score, return_counts=True)
    stops = counts.cumsum(0)
    starts = stops - counts
    average_rank = 0.5 * (starts.float() + 1.0 + stops.float())
    ranks[order] = torch.repeat_interleave(average_rank, counts)
    rank_sum = float(ranks[truth].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def _average_precision(values: torch.Tensor, labels: torch.Tensor) -> float:
    score = torch.as_tensor(values).float().cpu()
    truth = torch.as_tensor(labels).bool().cpu()
    positives = int(truth.sum())
    if positives == 0:
        return 0.0
    order = torch.argsort(score, descending=True, stable=True)
    sorted_score = score[order]
    ranked = truth[order].float()
    _, counts = torch.unique_consecutive(sorted_score, return_counts=True)
    stops = counts.cumsum(0)
    cumulative_true = ranked.cumsum(0)[stops - 1]
    precision = cumulative_true / stops.float()
    recall = cumulative_true / positives
    recall_previous = torch.cat((torch.zeros(1), recall[:-1]))
    # Group-end integration makes AP invariant to stored order within ties.
    return float(((recall - recall_previous) * precision).sum())


def _feature_audit(features: torch.Tensor, labels: torch.Tensor) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(FEATURE_NAMES):
        values = features[:, index]
        auc = _auc(values, labels)
        ap_positive = _average_precision(values, labels)
        ap_negative = _average_precision(-values, labels)
        rows.append(
            {
                "name": name,
                "positive_mean": float(values[labels].mean()),
                "negative_mean": float(values[~labels].mean()),
                "auc_positive_orientation": auc,
                "best_orientation_auc": max(auc, 1.0 - auc),
                "average_precision_positive_orientation": ap_positive,
                "average_precision_negative_orientation": ap_negative,
                "best_orientation_average_precision": max(
                    ap_positive, ap_negative
                ),
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    raw_authority, authority_sha, authority_path = load_json_object(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
        label="same-axis O0 mechanism execution authority",
    )
    authority = validate_execution_authority(raw_authority)
    if authority["implementation"] != file_record(Path(__file__).resolve()):
        raise ValueError("same-axis O0 mechanism implementation changed")
    if authority["utility_interface"] != file_record(
        Path(utility_api.__file__).resolve()
    ) or authority["exact_o0_implementation"] != file_record(
        Path(exact_o0_api.__file__).resolve()
    ):
        raise ValueError("same-axis O0 mechanism dependency changed")
    unit_output = Path(authority["outputs"]["unit_table"])
    report_output = Path(authority["outputs"]["report"])
    if any(path.exists() or path.is_symlink() for path in (unit_output, report_output)):
        raise FileExistsError("same-axis O0 mechanism outputs must both be new")
    subset, _, _ = load_torch_mapping(
        authority["combined_text_subset"]["path"],
        expected_sha256=authority["combined_text_subset"]["sha256"],
        map_location="cpu",
        label="same-axis O0 combined text subset",
    )
    raw_scores, _, _ = load_torch_mapping(
        authority["raw_combined_multiscale_scores"]["path"],
        expected_sha256=authority["raw_combined_multiscale_scores"]["sha256"],
        map_location="cpu",
        label="same-axis O0 raw combined scores",
    )
    positive_count = int(subset.get("positive_query_count", -1))
    query_names = [str(value) for value in subset.get("queries", [])]
    raw = torch.as_tensor(raw_scores.get("features")).detach().float().cpu()
    valid = torch.as_tensor(raw_scores.get("valid")).detach().bool().cpu()
    xyz = torch.as_tensor(raw_scores.get("xyz")).detach().float().cpu()
    metadata = raw_scores.get("metadata", {})
    if (
        subset.get("scene_id") != "scene0001_00"
        or positive_count != 64
        or len(query_names) != 68
        or raw.ndim != 3
        or raw.shape[1:] != (3, 68)
        or valid.shape != (raw.shape[0],)
        or xyz.shape != (raw.shape[0], 3)
        or metadata.get("query_names") != query_names
        or metadata.get("feature_space")
        != "primitive_text_query_scores_multiscale_unreduced"
        or metadata.get("scale_aggregation") != "none_frozen_downstream_only"
        or metadata.get("completion")
        != {
            "applied": False,
            "reason": "frozen_direct3d_requires_raw_unreduced_scale_scores",
        }
    ):
        raise ValueError("same-axis O0 raw score contract differs")
    o0 = exact_o0_readout(
        positive_scores=raw[:, :, :positive_count],
        negative_scores=raw[:, :, positive_count:],
        xyz=xyz,
        valid=valid,
        chunk_size=KNN_CHUNK_SIZE,
    )
    accepted, _, _ = load_torch_mapping(
        authority["accepted_v2"]["path"],
        expected_sha256=authority["accepted_v2"]["sha256"],
        map_location="cpu",
        label="same-axis O0 source AcceptedV2",
    )
    scene = load_scene_authority_v2(
        authority["source_membership_authority"],
        expected_scene_id="scene0001_00",
        expected_split="source_train",
    )
    if (
        not torch.equal(scene.region_rows, accepted["region_rows"])
        or not torch.equal(scene.token_mask, accepted["token_mask"])
        or torch.as_tensor(
            subset["region_dominant_positive_subset_index"]
        ).shape
        != (scene.region_count,)
        or torch.as_tensor(
            subset["region_dominant_positive_subset_index"]
        ).dtype
        != torch.long
    ):
        raise ValueError("same-axis O0 source region axes differ")
    utility = source_missing_core_conditional_utility(
        o0_scores=o0.final_scores,
        region_rows=scene.region_rows,
        core_mask=scene.token_mask,
        primitive_valid_mask=valid,
        region_query_indices=subset["region_dominant_positive_subset_index"],
        region_dominant_instance_ids=scene.dominant_instance_ids,
        primitive_instance_mass=scene.primitive_instance_mass,
    )
    shard, _, _ = load_torch_mapping(
        authority["source_training_shard"]["path"],
        expected_sha256=authority["source_training_shard"]["sha256"],
        map_location="cpu",
        label="same-axis O0 source training shard",
    )
    capability, _, _ = load_torch_mapping(
        authority["source_capability_descriptor"]["path"],
        expected_sha256=authority["source_capability_descriptor"]["sha256"],
        map_location="cpu",
        label="same-axis O0 source capability descriptor",
    )
    if (
        torch.as_tensor(shard.get("accepted_v2_e0")).shape != (4096, 1536)
        or not torch.equal(
            torch.as_tensor(shard["accepted_v2_e0"]),
            torch.as_tensor(accepted["accepted_v2_e0"]),
        )
        or torch.as_tensor(shard.get("raw_full_scalar_summary")).shape
        != (4096, 18)
        or torch.as_tensor(shard.get("eligible")).shape != (4096,)
        or not torch.equal(
            torch.as_tensor(capability.get("region_rows")), scene.region_rows
        )
        or not torch.equal(
            torch.as_tensor(capability.get("token_mask")), scene.token_mask
        )
        or torch.as_tensor(capability.get("appearance_concentration")).shape
        != (4096,)
        or torch.as_tensor(capability.get("boundary_concentration")).shape
        != (4096,)
    ):
        raise ValueError("same-axis O0 source feature axes differ")
    features = build_unit_feature_table(
        utility=utility,
        o0_scores=o0.final_scores,
        primitive_valid_mask=valid,
        region_query_indices=subset[
            "region_dominant_positive_subset_index"
        ],
        region_rows=scene.region_rows,
        token_mask=scene.token_mask,
        xyz=xyz,
        region_scale_indices=accepted["scale_indices"],
        selected_query_scale_indices=o0.selected_scale_indices,
        appearance_concentration=capability["appearance_concentration"],
        boundary_concentration=capability["boundary_concentration"],
        raw_full_scalar_summary=shard["raw_full_scalar_summary"],
        full_scalar_eligible=shard["eligible"],
    )
    labels = utility.unit_hard_labels
    positive = int(labels.sum())
    total = int(labels.numel())
    negative = total - positive
    gate = authority["fixed_audit"]["scene0001_gate"]
    outcomes = {
        "qualified_anchor_query_pairs": int(utility.qualified_region_mask.sum()),
        "missing_core_units": total,
        "hard_positive_missing_units": positive,
        "hard_negative_missing_units": negative,
    }
    outcomes["passed"] = (
        outcomes["qualified_anchor_query_pairs"]
        >= gate["minimum_qualified_anchor_query_pairs"]
        and total >= gate["minimum_missing_core_units"]
        and positive >= gate["minimum_hard_positive_missing_units"]
        and negative >= gate["minimum_hard_negative_missing_units"]
    )
    feature_rows = _feature_audit(features, labels) if total else []
    unit_payload = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "scene_id": "scene0001_00",
        "feature_names": list(FEATURE_NAMES),
        "features": features,
        "hard_labels": labels,
        "soft_target_mass_fraction": utility.unit_soft_target_mass_fraction,
        "signed_utility": utility.unit_signed_utility,
        "unit_region_indices": utility.unit_region_indices,
        "unit_query_indices": utility.unit_query_indices,
        "unit_primitive_rows": utility.unit_primitive_rows,
        "qualified_region_mask": utility.qualified_region_mask,
        "positive_fraction": utility.positive_fraction,
        "missing_counts": utility.missing_counts,
        "selected_query_scale_indices": o0.selected_scale_indices,
        "execution_authority": {
            "path": str(authority_path),
            "sha256": authority_sha,
        },
    }
    unit_payload["channel_sha256"] = {
        name: tensor_sha256(unit_payload[name])
        for name in (
            "features",
            "hard_labels",
            "soft_target_mass_fraction",
            "signed_utility",
            "unit_region_indices",
            "unit_query_indices",
            "unit_primitive_rows",
            "qualified_region_mask",
            "positive_fraction",
            "missing_counts",
            "selected_query_scale_indices",
        )
    }
    write_torch_noclobber(unit_output, unit_payload)
    report = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "status": (
            "scene0001_same_axis_O0_mechanism_gate_passed"
            if outcomes["passed"]
            else "scene0001_same_axis_O0_mechanism_gate_failed"
        ),
        "execution_authority": {
            "path": str(authority_path),
            "sha256": authority_sha,
        },
        "unit_table": file_record(unit_output),
        "same_axis_O0": {
            "valid_primitives": int(valid.sum()),
            "positive_queries": positive_count,
            "selected_scale_counts": {
                str(index): int((o0.selected_scale_indices == index).sum())
                for index in range(3)
            },
        },
        "sample_gate": {"thresholds": gate, "outcomes": outcomes},
        "unconditional_completion_reference": {
            "hard_precision": positive / max(total, 1),
            "hard_precision_Wilson95_lower": one_sided_wilson_lower(
                positive, total
            ),
            "soft_target_mass_fraction_mean": float(
                utility.unit_soft_target_mass_fraction.mean()
            )
            if total
            else 0.0,
            "signed_utility_mean": float(utility.unit_signed_utility.mean())
            if total
            else 0.0,
            "positive_fraction_quantiles": torch.quantile(
                utility.positive_fraction[utility.qualified_region_mask],
                torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]),
            ).tolist()
            if bool(utility.qualified_region_mask.any())
            else [],
            "missing_score_quantiles": torch.quantile(
                utility.unit_o0_scores,
                torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]),
            ).tolist()
            if total
            else [],
        },
        "feature_learnability_audit": {
            "features": feature_rows,
            "best_single_feature": max(
                feature_rows,
                key=lambda row: row[
                    "best_orientation_average_precision"
                ],
                default=None,
            ),
        },
        "heldout_scene0004_membership_opened": False,
        "source_access": source_access(),
        "benchmark_execution_authorized": False,
        "target_execution_performed": False,
    }
    report["content_authority_sha256"] = canonical_json_sha256(report)
    write_frozen_json(report_output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-authority", required=True)
    parser.add_argument("--expected-execution-authority-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    result = run(build_parser().parse_args(argv))
    print(result["status"])
    print(result["sample_gate"])


if __name__ == "__main__":
    main()
