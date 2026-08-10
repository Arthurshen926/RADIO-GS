#!/usr/bin/env python3
"""Audit graph completion with exact source-heldout instance support.

This is a source-only diagnostic.  It reuses the frozen O0 query field and the
native-V3 candidate graph, but constructs every completion label from a
deterministic held-out third of the per-view exact responsibility stream.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.querying.absolute_relevance_relation_readout import (
    absolute_relevance_relation_readout,
)
from radio_gs.querying.source_heldout_missing_support import (
    FEATURE_NAMES,
    SEMANTIC_BOUNDARY,
    calibration_free_maximin_selection,
    heldout_missing_support_label_for_seed_instance,
    infer_seed_instance_from_training_views,
    proposal_feature_vector,
)
from radio_gs.models.region_comembership_native_v3 import RegionCoMembershipNativeV3
from radio_gs.scripts.build_lerf_o0_anchored_graph_residual_cache import (
    exact_o0_readout,
)
from radio_gs.scripts.build_source_region_comembership_v1 import (
    INSTANCE_KEY_STRIDE,
    _exact_hit_instance_mass,
    _instance_raster,
    _load_instance_members,
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


AUTHORITY_SCHEMA = "radio_gs.source_heldout_missing_support_audit_authority.v1"
TABLE_SCHEMA = "radio_gs.source_heldout_missing_support_table.v1"
REPORT_SCHEMA = "radio_gs.source_heldout_missing_support_audit.v1"
FOLD_COUNT = 3
RELATION_THRESHOLD = 0.85
MAXIMUM_REGIONS = 8
KNN_CHUNK_SIZE = 65536


def source_access() -> dict[str, bool]:
    return {
        "source_query_scores_opened": True,
        "source_exact_responsibility_views_opened": True,
        "source_instance_labels_opened": True,
        "source_native_v3_pair_features_opened": True,
        "source_factorized_state_opened": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "benchmark_queries_opened": False,
        "benchmark_metrics_opened": False,
        "target_metrics_computed": False,
        "gpu_used": False,
    }


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} record differs")
    path = validate_file_record(value, label=label)
    return {"path": str(path), "sha256": str(value["sha256"])}


def validate_execution_authority(
    path: str | Path, *, expected_sha256: str
) -> dict[str, Any]:
    raw, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="source-heldout missing-support authority",
    )
    required = {
        "schema",
        "schema_version",
        "status",
        "scene_id",
        "producer",
        "inputs",
        "fixed_protocol",
        "outputs",
        "source_access",
        "benchmark_execution_authorized",
    }
    authority = dict(raw)
    protocol = {
        "fold_count": FOLD_COUNT,
        "fold_assignment": "view_index_modulo_3",
        "training_evidence": "exact_view_mass_from_other_two_folds",
        "label_evidence": "exact_view_mass_from_heldout_fold_only",
        "semantic_boundary": SEMANTIC_BOUNDARY,
        "relation_threshold": RELATION_THRESHOLD,
        "maximum_regions": MAXIMUM_REGIONS,
        "path_method": "widest_path",
        "candidate": "nonseed_selected_native_v3_region_with_anchor_above_boundary_and_missing_core",
        "hard_label": "heldout_missing_core_seed_instance_mass_strictly_exceeds_every_other_instance",
        "signed_utility": "2_times_heldout_seed_instance_mass_fraction_minus_1",
        "ranking": "within_query_fold_maximin_empirical_rank_top1_no_label_threshold",
        "feature_names": list(FEATURE_NAMES),
    }
    if (
        set(authority) != required
        or authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("status")
        != "authorized_source_only_heldout_support_audit"
        or not isinstance(authority.get("scene_id"), str)
        or authority.get("fixed_protocol") != protocol
        or authority.get("source_access") != source_access()
        or authority.get("benchmark_execution_authorized") is not False
    ):
        raise ValueError("source-heldout missing-support authority differs")
    producer = _record(authority["producer"], label="source-heldout producer")
    if Path(producer["path"]).resolve() != Path(__file__).resolve():
        raise ValueError("source-heldout producer path differs")
    inputs = authority.get("inputs")
    expected_inputs = {
        "native_v3_source",
        "native_v3_checkpoint",
        "combined_text_subset",
        "raw_combined_multiscale_scores",
    }
    if not isinstance(inputs, Mapping) or set(inputs) != expected_inputs:
        raise ValueError("source-heldout inputs differ")
    authority["inputs"] = {
        name: _record(record, label=f"source-heldout {name}")
        for name, record in inputs.items()
    }
    outputs = authority.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {"table", "report"}:
        raise ValueError("source-heldout outputs differ")
    authority["outputs"] = {
        name: str(Path(value).expanduser().resolve()) for name, value in outputs.items()
    }
    if any(authority["outputs"][name] != value for name, value in outputs.items()):
        raise ValueError("source-heldout outputs must be canonical")
    authority["producer"] = producer
    authority["verified_record"] = {"path": str(source), "sha256": digest}
    return authority


def _load_view_fold_mass(
    *,
    manifest_record: Mapping[str, str],
    instance_zip_record: Mapping[str, str],
    primitive_count: int,
    instance_columns: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    manifest, _, manifest_path = load_json_object(
        manifest_record["path"],
        expected_sha256=manifest_record["sha256"],
        label="source-heldout exact responsibility manifest",
    )
    metadata = manifest.get("metadata", {})
    views = manifest.get("views")
    height = int(metadata.get("feature_height", -1))
    width = int(metadata.get("feature_width", -1))
    if (
        manifest.get("formula_contract", {}).get("name")
        != "sparse_exact_marginal_responsibility_authority_v1"
        or not isinstance(views, list)
        or len(views) < FOLD_COUNT
        or int(manifest.get("num_gaussians", -1)) != int(primitive_count)
        or height <= 0
        or width <= 0
    ):
        raise ValueError("source-heldout exact responsibility contract differs")
    instance_zip = validate_file_record(
        instance_zip_record, label="source-heldout instance ZIP"
    )
    members = _load_instance_members(
        instance_zip, str(instance_zip_record["sha256"])
    )
    fold_mass = torch.zeros(
        FOLD_COUNT,
        int(primitive_count),
        int(instance_columns),
        dtype=torch.float64,
    )
    counts = [0] * FOLD_COUNT
    with ZipFile(instance_zip) as archive:
        for expected_view, record in enumerate(views):
            frame = int(record["frame_index"])
            if frame not in members:
                raise ValueError("source-heldout instance view is absent")
            payload, digest, _ = load_torch_mapping(
                manifest_path.parent / str(record["relative_path"]),
                expected_sha256=str(record["sha256"]),
                map_location="cpu",
                label="source-heldout exact responsibility view",
            )
            if (
                payload.get("schema")
                != "radio_gs.sparse_exact_marginal_responsibility_view.v1"
                or int(payload.get("view_index", -1)) != expected_view
                or int(payload.get("frame_index", -1)) != frame
                or digest != str(record["sha256"])
            ):
                raise ValueError("source-heldout exact view identity differs")
            raster = _instance_raster(
                archive, members[frame], height=height, width=width
            )
            keys, mass = _exact_hit_instance_mass(
                gaussian_ids=payload["gaussian_ids"],
                pixel_ids=payload["pixel_ids"],
                base_weights=payload["base_weights"],
                pixel_instance_ids=raster,
                num_gaussians=int(primitive_count),
                num_pixels=height * width,
            )
            primitive = keys // INSTANCE_KEY_STRIDE
            instance = keys % INSTANCE_KEY_STRIDE
            if bool((instance >= int(instance_columns)).any()):
                raise ValueError("source-heldout instance axis exceeds source authority")
            fold = expected_view % FOLD_COUNT
            flat = primitive * int(instance_columns) + instance
            fold_mass[fold].view(-1).index_add_(0, flat, mass)
            counts[fold] += 1
    return fold_mass.float().contiguous(), {
        "views": len(views),
        "views_per_fold": counts,
        "fold_visible_mass": [float(fold_mass[index].sum()) for index in range(FOLD_COUNT)],
        "label_fold_overlap": 0,
    }


def _source_region_reliability(
    native: Mapping[str, Any], factorized: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = torch.as_tensor(native["region_rows"]).long().cpu()
    mask = torch.as_tensor(native["token_mask"]).bool().cpu()
    global_rows = torch.as_tensor(factorized["global_rows"]).long().cpu()
    primitive_count = int(native["primitive_count"])
    lookup = torch.full((primitive_count,), -1, dtype=torch.long)
    lookup[global_rows] = torch.arange(global_rows.numel())
    safe = rows.clamp_min(0)
    compact = lookup[safe]
    exact = mask & (compact >= 0)
    safe_compact = compact.clamp_min(0)
    observation = torch.as_tensor(factorized["observation_evidence"]).float().cpu()
    agreement = 1.0 - torch.as_tensor(factorized["directional_dispersion"]).float().cpu()
    visibility_value = torch.as_tensor(factorized["visibility_purity_value"]).float().cpu()
    visibility_known = torch.as_tensor(factorized["visibility_purity_known"]).bool().cpu()
    count = exact.sum(dim=1).clamp_min(1)
    mean_observation = (observation[safe_compact] * exact).sum(dim=1) / count
    mean_agreement = (agreement[safe_compact] * exact).sum(dim=1) / count
    visible_mask = exact & visibility_known[safe_compact]
    visible_count = visible_mask.sum(dim=1).clamp_min(1)
    visibility = (
        visibility_value[safe_compact] * visible_mask
    ).sum(dim=1) / visible_count
    visibility[~visible_mask.any(dim=1)] = 0.0
    inferred_count = mean_observation / (1.0 - mean_observation).clamp_min(1e-6)
    return (
        inferred_count.clamp(0.0, 1e6).contiguous(),
        mean_agreement.clamp(0.0, 1.0).contiguous(),
        visibility.clamp(0.0, 1.0).contiguous(),
    )


def _native_probability(
    native: Mapping[str, Any], checkpoint: Mapping[str, Any]
) -> torch.Tensor:
    normalization = checkpoint["normalization"]
    model = RegionCoMembershipNativeV3(
        normalization["median"], normalization["robust_scale"]
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    features = torch.as_tensor(native["pair_features"]).float().cpu()
    with torch.inference_mode():
        return model.probability(features).float().cpu().contiguous()


def _summary(labels: torch.Tensor, utility: torch.Tensor, selected: torch.Tensor) -> dict[str, Any]:
    truth = torch.as_tensor(labels).bool().cpu()
    signed = torch.as_tensor(utility).float().cpu()
    keep = torch.as_tensor(selected).bool().cpu()
    total = int(truth.numel())
    chosen = int(keep.sum())
    return {
        "candidate_units": total,
        "positive_units": int(truth.sum()),
        "negative_units": int((~truth).sum()),
        "positive_fraction": float(truth.float().mean()) if total else 0.0,
        "signed_utility_mean": float(signed.mean()) if total else 0.0,
        "rank_selected_units": chosen,
        "rank_selected_positive_units": int(truth[keep].sum()),
        "rank_selected_positive_fraction": float(truth[keep].float().mean()) if chosen else 0.0,
        "rank_selected_signed_utility_mean": float(signed[keep].mean()) if chosen else 0.0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    authority = validate_execution_authority(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
    )
    output = Path(authority["outputs"]["table"])
    report_path = Path(authority["outputs"]["report"])
    if output.exists() or output.is_symlink() or report_path.exists() or report_path.is_symlink():
        raise FileExistsError("source-heldout audit output already exists")
    records = authority["inputs"]
    native, _, _ = load_torch_mapping(
        records["native_v3_source"]["path"],
        expected_sha256=records["native_v3_source"]["sha256"],
        map_location="cpu",
        label="source-heldout native V3 source",
    )
    checkpoint, _, _ = load_torch_mapping(
        records["native_v3_checkpoint"]["path"],
        expected_sha256=records["native_v3_checkpoint"]["sha256"],
        map_location="cpu",
        label="source-heldout native V3 checkpoint",
    )
    subset, _, _ = load_torch_mapping(
        records["combined_text_subset"]["path"],
        expected_sha256=records["combined_text_subset"]["sha256"],
        map_location="cpu",
        label="source-heldout text subset",
    )
    raw, _, _ = load_torch_mapping(
        records["raw_combined_multiscale_scores"]["path"],
        expected_sha256=records["raw_combined_multiscale_scores"]["sha256"],
        map_location="cpu",
        label="source-heldout raw scores",
    )
    scene_id = str(authority["scene_id"])
    positive_count = int(subset["positive_query_count"])
    raw_scores = torch.as_tensor(raw["features"]).float().cpu()
    valid = torch.as_tensor(raw["valid"]).bool().cpu()
    xyz = torch.as_tensor(raw["xyz"]).float().cpu()
    rows = torch.as_tensor(native["region_rows"]).long().cpu()
    core = torch.as_tensor(native["token_mask"]).bool().cpu()
    if (
        native.get("scene_id") != scene_id
        or subset.get("scene_id") != scene_id
        or raw_scores.ndim != 3
        or raw_scores.shape[1] != 3
        or raw_scores.shape[2] <= positive_count
        or valid.shape != (raw_scores.shape[0],)
        or xyz.shape != (raw_scores.shape[0], 3)
        or rows.shape[0] != 4096
        or core.shape != rows.shape
        or int(native["primitive_count"]) != raw_scores.shape[0]
    ):
        raise ValueError("source-heldout scene axes differ")
    o0 = exact_o0_readout(
        positive_scores=raw_scores[:, :, :positive_count],
        negative_scores=raw_scores[:, :, positive_count:],
        xyz=xyz,
        valid=valid,
        chunk_size=int(args.knn_chunk_size),
    )
    accepted_record = native["input_authority"]["accepted_v2"]
    accepted, _, _ = load_torch_mapping(
        accepted_record["path"],
        expected_sha256=accepted_record["sha256"],
        map_location="cpu",
        label="source-heldout AcceptedV2",
    )
    anchor = torch.as_tensor(accepted["anchor_index"]).long().cpu()
    region_scale = torch.as_tensor(accepted["scale_indices"]).long().cpu()
    anchor_rows = rows[torch.arange(rows.shape[0]), anchor]
    region_anchor = o0.scores_by_scale[anchor_rows, region_scale, :positive_count]
    selected_scale = o0.selected_scale_indices[:positive_count]
    seed_unary = region_anchor.clone()
    seed_unary[region_scale[:, None] != selected_scale[None, :]] = 0.0
    probability = _native_probability(native, checkpoint)
    relation = absolute_relevance_relation_readout(
        region_absolute_relevance=seed_unary,
        pair_indices=native["pair_indices"],
        pair_probabilities=probability,
        absolute_boundary=SEMANTIC_BOUNDARY,
        relation_threshold=RELATION_THRESHOLD,
        maximum_regions=MAXIMUM_REGIONS,
        path_method="widest_path",
    )
    factorized_record = native["input_authority"]["factorized_state"]
    factorized, _, _ = load_torch_mapping(
        factorized_record["path"],
        expected_sha256=factorized_record["sha256"],
        map_location="cpu",
        label="source-heldout factorized state",
    )
    observation_count, agreement, visibility = _source_region_reliability(
        native, factorized
    )
    parent_record = native["input_authority"]["parent_v2_source_authority"]
    parent, _, _ = load_torch_mapping(
        parent_record["path"],
        expected_sha256=parent_record["sha256"],
        map_location="cpu",
        label="source-heldout parent V2 source authority",
    )
    fold_mass, fold_audit = _load_view_fold_mass(
        manifest_record=parent["input_authority"]["exact_marginal"],
        instance_zip_record=parent["input_authority"]["instance_zip"],
        primitive_count=int(native["primitive_count"]),
        instance_columns=int(native["instance_columns_including_zero"]),
    )
    total_mass = fold_mass.sum(dim=0)
    feature_rows: list[torch.Tensor] = []
    label_rows: list[bool] = []
    utility_rows: list[float] = []
    fraction_rows: list[float] = []
    fold_rows: list[int] = []
    query_rows: list[int] = []
    seed_rows_out: list[int] = []
    target_rows_out: list[int] = []
    instance_rows: list[int] = []
    visible_mass_rows: list[float] = []
    missing_count_rows: list[int] = []
    query_axis = torch.arange(positive_count)
    for query in query_axis.tolist():
        if not bool(relation.query_gate[query]):
            continue
        seed = int(relation.seed_region_indices[query])
        seed_members = rows[seed, core[seed]]
        seed_members = seed_members[valid[seed_members]]
        if seed_members.numel() <= 0:
            continue
        seed_scores = o0.scores_by_scale[
            seed_members, int(selected_scale[query]), query
        ]
        seed_median = float(seed_scores.median())
        seed_instance_by_fold = []
        local_seed_rows = torch.arange(seed_members.numel(), dtype=torch.long)
        for fold in range(FOLD_COUNT):
            seed_instance_by_fold.append(
                infer_seed_instance_from_training_views(
                    seed_rows=local_seed_rows,
                    training_primitive_instance_mass=(
                        total_mass[seed_members] - fold_mass[fold, seed_members]
                    ),
                )
            )
        for target in torch.where(relation.selected_region_masks[:, query])[0].tolist():
            if target == seed:
                continue
            members = rows[target, core[target]]
            members = members[valid[members]]
            if members.numel() <= 0:
                continue
            target_scores = o0.scores_by_scale[
                members, int(selected_scale[query]), query
            ]
            target_anchor = float(
                o0.scores_by_scale[
                    int(anchor_rows[target]), int(selected_scale[query]), query
                ]
            )
            if target_anchor <= SEMANTIC_BOUNDARY or not bool(
                (target_scores <= SEMANTIC_BOUNDARY).any()
            ):
                continue
            feature = proposal_feature_vector(
                edge_comembership_reliability=float(
                    relation.path_support[target, query]
                ),
                source_observation_count=float(observation_count[target]),
                source_observation_agreement=float(agreement[target]),
                target_selected_scale_scores=target_scores,
                target_anchor_score=target_anchor,
                seed_median_score=seed_median,
                target_visibility=float(visibility[target]),
            )
            for fold in range(FOLD_COUNT):
                label = heldout_missing_support_label_for_seed_instance(
                    target_selected_scale_scores=target_scores,
                    target_rows=members,
                    seed_instance_id=seed_instance_by_fold[fold],
                    heldout_primitive_instance_mass=fold_mass[fold],
                )
                if not label.evaluable:
                    continue
                feature_rows.append(feature)
                label_rows.append(label.hard_positive)
                utility_rows.append(label.signed_utility)
                fraction_rows.append(label.heldout_target_mass_fraction)
                fold_rows.append(fold)
                query_rows.append(query)
                seed_rows_out.append(seed)
                target_rows_out.append(target)
                instance_rows.append(label.seed_instance_id)
                visible_mass_rows.append(label.heldout_visible_mass)
                missing_count_rows.append(label.missing_primitive_count)
    if not feature_rows:
        raise RuntimeError("source-heldout audit produced no evaluable proposals")
    features = torch.stack(feature_rows).float().contiguous()
    labels = torch.tensor(label_rows, dtype=torch.bool)
    utility = torch.tensor(utility_rows, dtype=torch.float32)
    fractions = torch.tensor(fraction_rows, dtype=torch.float32)
    folds = torch.tensor(fold_rows, dtype=torch.long)
    queries = torch.tensor(query_rows, dtype=torch.long)
    seeds = torch.tensor(seed_rows_out, dtype=torch.long)
    targets = torch.tensor(target_rows_out, dtype=torch.long)
    instances = torch.tensor(instance_rows, dtype=torch.long)
    visible_mass_tensor = torch.tensor(visible_mass_rows, dtype=torch.float32)
    missing_counts = torch.tensor(missing_count_rows, dtype=torch.long)
    groups = folds * positive_count + queries
    rank_selected, rank_score = calibration_free_maximin_selection(
        features, groups, targets
    )
    table = {
        "schema": TABLE_SCHEMA,
        "schema_version": 1,
        "scene_id": scene_id,
        "feature_names": list(FEATURE_NAMES),
        "features": features,
        "hard_labels": labels,
        "signed_utility": utility,
        "heldout_target_mass_fraction": fractions,
        "fold_indices": folds,
        "query_indices": queries,
        "seed_region_indices": seeds,
        "target_region_indices": targets,
        "seed_instance_ids_from_training_views": instances,
        "heldout_visible_mass": visible_mass_tensor,
        "missing_primitive_counts": missing_counts,
        "calibration_free_rank_score": rank_score,
        "calibration_free_rank_selected": rank_selected,
        "execution_authority": authority["verified_record"],
        "channel_sha256": {},
    }
    tensor_names = tuple(
        name for name, value in table.items() if torch.is_tensor(value)
    )
    table["channel_sha256"] = {
        name: tensor_sha256(table[name]) for name in tensor_names
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_torch_noclobber(output, table)
    per_fold = []
    for fold in range(FOLD_COUNT):
        local = folds == fold
        per_fold.append({"fold": fold, **_summary(labels[local], utility[local], rank_selected[local])})
    report = {
        "schema": REPORT_SCHEMA,
        "schema_version": 1,
        "status": "source_heldout_missing_support_audit_complete",
        "scene_id": scene_id,
        "execution_authority": authority["verified_record"],
        "table": file_record(output),
        "fixed_protocol": authority["fixed_protocol"],
        "fold_audit": fold_audit,
        "graph_audit": {
            "queries": positive_count,
            "query_gate_passed": int(relation.query_gate.sum()),
            "selected_nonseed_region_query_pairs": int(
                relation.selected_region_masks.sum() - relation.query_gate.sum()
            ),
            "evaluable_query_fold_groups": int(torch.unique(groups).numel()),
        },
        "overall": _summary(labels, utility, rank_selected),
        "per_fold": per_fold,
        "label_isolation": {
            "seed_instance_uses_heldout_fold": False,
            "target_label_uses_training_folds": False,
            "view_fold_overlap": 0,
            "query_field_is_query_free_transductive_source_field": True,
            "native_v3_probability_has_aggregate_source_label_pretraining_overlap": True,
            "probability_overlap_action": "feature_only_not_an_independent_generalization_claim",
        },
        "source_access": source_access(),
        "benchmark_execution_authorized": False,
    }
    report["content_authority_sha256"] = canonical_json_sha256(report)
    write_frozen_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-authority", required=True)
    parser.add_argument("--expected-execution-authority-sha256", required=True)
    parser.add_argument("--knn-chunk-size", type=int, default=KNN_CHUNK_SIZE)
    print(json.dumps(run(parser.parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
