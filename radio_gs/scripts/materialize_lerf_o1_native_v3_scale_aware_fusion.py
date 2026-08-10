#!/usr/bin/env python3
"""Materialize a premetric O1/native-V3 scale-aware LERF score cache."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from radio_gs.interfaces.surface_region_target_accepted_v2 import (
    validate_target_accepted_v2_authority,
)
from radio_gs.querying import scale_aware_native_v3_support_fusion as fusion
from radio_gs.scripts import eval_lerf_direct_3d_selection as frozen
from radio_gs.scripts.infer_region_comembership_native_v3 import (
    validate_inference_authority,
)
from radio_gs.scripts.materialize_region_comembership_features_native_v3 import (
    validate_feature_authority,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    load_torch_mapping,
    sha256_file,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


EXECUTION_SCHEMA = "radio_gs.lerf_o1_native_v3_scale_aware_execution.v1"
EXECUTION_STATUS = "authorized_figurines_development_premetric_materialization"
OUTPUT_SCHEMA = "radio_gs.lerf_o1_native_v3_scale_aware_external_scores.v1"
LOGIT_SCALE = 10.0
KNN_K = 10


def access_audit() -> dict[str, bool]:
    return {
        "benchmark_query_score_caches_opened": True,
        "native_v3_query_independent_authorities_opened": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "target_metrics_computed": False,
        "gpu_used": False,
        "result_dependent_parameters": False,
    }


def frozen_multiscale_relevance(
    *,
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    xyz: torch.Tensor,
    valid: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return exact frozen per-scale remap, query scale, and raw peaks."""

    probability = frozen.canonical_negative_relevancy_query_scores(
        positive_scores,
        negative_scores,
        logit_scale=LOGIT_SCALE,
    )
    if probability.ndim != 3 or probability.shape[1] != 3:
        raise ValueError("O1 probability must align as [N,3,Q]")
    count, scales, queries = probability.shape
    smoothed = frozen.vala_knn_smoothed_scores(
        probability.reshape(count, scales * queries),
        xyz,
        k=KNN_K,
        chunk_size=int(chunk_size),
        valid_mask=valid,
    ).reshape(count, scales, queries)
    raw_peaks = smoothed[torch.as_tensor(valid).bool()].amax(dim=0)
    selected_scale = raw_peaks.argmax(dim=0).long().contiguous()
    remapped = frozen.vala_minmax_remap_scores(
        smoothed.reshape(count, scales * queries),
        valid_mask=valid,
    ).reshape(count, scales, queries)
    return remapped.contiguous(), selected_scale, raw_peaks.contiguous()


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
        label="O1/native-V3 scale-aware execution authority",
    )
    authority = dict(raw)
    required = {
        "schema",
        "schema_version",
        "status",
        "scene_id",
        "method_contract_sha256",
        "producer",
        "inputs",
        "outputs",
        "constants",
        "access_audit",
        "metric_execution_authorized",
    }
    if (
        set(authority) != required
        or authority.get("schema") != EXECUTION_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("status") != EXECUTION_STATUS
        or authority.get("scene_id") != "figurines"
        or authority.get("method_contract_sha256")
        != fusion.READOUT_CONTRACT_SHA256
        or authority.get("constants")
        != {
            "canonical_negative_logit_scale": LOGIT_SCALE,
            "knn_k": KNN_K,
            "semantic_boundary": fusion.SEMANTIC_BOUNDARY,
            "native_v3_relation_threshold": fusion.RELATION_THRESHOLD,
            "maximum_regions": fusion.MAXIMUM_REGIONS,
            "path_method": fusion.PATH_METHOD,
        }
        or authority.get("access_audit") != access_audit()
        or authority.get("metric_execution_authorized") is not False
    ):
        raise ValueError("O1/native-V3 scale-aware execution header differs")
    producer = _record(authority["producer"], label="scale-aware producer")
    if Path(producer["path"]).resolve() != Path(__file__).resolve():
        raise ValueError("scale-aware producer path differs")
    inputs = authority.get("inputs")
    expected_inputs = {
        "o1_manifest",
        "o1_positive",
        "o1_negative",
        "native_v3_feature",
        "native_v3_inference",
        "frozen_evaluator",
    }
    if not isinstance(inputs, Mapping) or set(inputs) != expected_inputs:
        raise ValueError("scale-aware input authority differs")
    outputs = authority.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {"cache", "report"}:
        raise ValueError("scale-aware output authority differs")
    canonical_outputs = {
        name: str(Path(value).expanduser().resolve())
        for name, value in outputs.items()
    }
    if any(value != outputs[name] for name, value in canonical_outputs.items()):
        raise ValueError("scale-aware output path must be absolute and canonical")
    if len(set(canonical_outputs.values())) != 2:
        raise ValueError("scale-aware output paths must differ")
    authority["producer"] = producer
    authority["inputs"] = {
        name: _record(inputs[name], label=f"scale-aware input {name}")
        for name in sorted(expected_inputs)
    }
    authority["outputs"] = canonical_outputs
    authority["verified_record"] = {"path": str(source), "sha256": digest}
    return authority


def _paired_o1_caches(
    authority: Mapping[str, Any]
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    records = authority["inputs"]
    positive_raw, _, _ = load_torch_mapping(
        records["o1_positive"]["path"],
        expected_sha256=records["o1_positive"]["sha256"],
        map_location="cpu",
        label="Figurines O1 positive cache",
    )
    negative_raw, _, _ = load_torch_mapping(
        records["o1_negative"]["path"],
        expected_sha256=records["o1_negative"]["sha256"],
        map_location="cpu",
        label="Figurines O1 negative cache",
    )
    query_ids = tuple(str(value) for value in positive_raw["query_ids"])
    positive = frozen.validate_ours_multiscale_query_score_cache(
        positive_raw,
        expected_xyz=torch.as_tensor(positive_raw["xyz"]),
        expected_query_ids=query_ids,
        expected_renderer_geometry_checkpoint_sha256=(
            positive_raw["renderer_geometry_checkpoint_sha256"]
        ),
    )
    negative = frozen.validate_ours_multiscale_query_score_cache(
        negative_raw,
        expected_xyz=positive_raw["xyz"],
        expected_query_ids=frozen.NEGATIVE_PROMPTS,
        expected_renderer_geometry_checkpoint_sha256=(
            positive.renderer_geometry_checkpoint_sha256
        ),
    )
    for name in (
        "valid",
        "scale_ids",
        "scale_radii_m",
        "xyz_sha256",
        "field_checkpoint_sha256",
        "readout_checkpoint_sha256",
        "renderer_geometry_checkpoint_sha256",
    ):
        left, right = getattr(positive, name), getattr(negative, name)
        equal = torch.equal(left, right) if torch.is_tensor(left) else left == right
        if not bool(equal):
            raise ValueError(f"Figurines O1 positive/negative {name} differs")
    if (
        positive.score_semantics != "raw_independent_normalized_cosine"
        or negative.score_semantics != "raw_independent_normalized_cosine"
        or tuple(float(value) for value in positive.scale_radii_m)
        != (0.25, 0.45, 0.7)
    ):
        raise ValueError("Figurines O1 frozen score semantics differ")
    return positive, negative, positive_raw, negative_raw


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    authority = validate_execution_authority(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
    )
    output = Path(authority["outputs"]["cache"])
    report_path = Path(authority["outputs"]["report"])
    if output.exists() or output.is_symlink() or report_path.exists() or report_path.is_symlink():
        raise FileExistsError("scale-aware output already exists")
    positive, negative, positive_raw, _negative_raw = _paired_o1_caches(authority)

    records = authority["inputs"]
    manifest, _, _ = load_json_object(
        records["o1_manifest"]["path"],
        expected_sha256=records["o1_manifest"]["sha256"],
        label="Figurines O1 manifest",
    )
    if (
        manifest.get("status") != "complete_source_only_oracle_matrix_materialization"
        or manifest.get("scene") != "figurines"
        or manifest.get("o1", {}).get("positive") != records["o1_positive"]
        or manifest.get("o1", {}).get("negative") != records["o1_negative"]
        or manifest.get("benchmark_metrics_opened") is not False
    ):
        raise ValueError("Figurines O1 manifest binding differs")

    feature_raw, _, _ = load_torch_mapping(
        records["native_v3_feature"]["path"],
        expected_sha256=records["native_v3_feature"]["sha256"],
        map_location="cpu",
        label="Figurines native-V3 feature",
    )
    feature = validate_feature_authority(feature_raw)
    inference_raw, _, _ = load_torch_mapping(
        records["native_v3_inference"]["path"],
        expected_sha256=records["native_v3_inference"]["sha256"],
        map_location="cpu",
        label="Figurines native-V3 inference",
    )
    inference = validate_inference_authority(inference_raw)
    accepted_record = feature["input_authority"]["accepted_v2"]
    accepted_raw, _, _ = load_torch_mapping(
        accepted_record["path"],
        expected_sha256=accepted_record["sha256"],
        map_location="cpu",
        label="Figurines native-V3 AcceptedV2",
    )
    accepted = validate_target_accepted_v2_authority(accepted_raw)
    if (
        feature["scene_id"] != "figurines"
        or inference["scene_id"] != "figurines"
        or inference["feature_authority"] != records["native_v3_feature"]
        or inference["selected_rule"]
        != {
            "method": "dual_path_widest",
            "maximum_regions": fusion.MAXIMUM_REGIONS,
            "threshold": fusion.RELATION_THRESHOLD,
        }
        or accepted["physical_space_authority"]["geometry_checkpoint_sha256"]
        != positive.renderer_geometry_checkpoint_sha256
        or not torch.equal(feature["region_rows"], accepted["region_rows"])
        or not torch.equal(feature["token_mask"], accepted["token_mask"])
        or not torch.equal(
            feature["canonical_region_indices"],
            accepted["canonical_region_indices"],
        )
    ):
        raise ValueError("Figurines O1/native-V3 physical or region binding differs")

    per_scale, selected_scale, raw_peaks = frozen_multiscale_relevance(
        positive_scores=positive.query_scores,
        negative_scores=negative.query_scores,
        xyz=positive_raw["xyz"],
        valid=positive.valid,
        chunk_size=int(args.knn_chunk_size),
    )
    result = fusion.scale_aware_native_v3_support_fusion(
        primitive_relevance_by_scale=per_scale,
        selected_scale_indices=selected_scale,
        primitive_valid=positive.valid,
        region_rows=feature["region_rows"],
        semantic_core_mask=feature["token_mask"],
        region_anchor_positions=accepted["anchor_index"],
        region_scale_indices=accepted["scale_indices"],
        pair_indices=inference["pair_indices"],
        pair_probabilities=inference["pair_probabilities"],
    )
    base_selected = result.primitive_unary >= fusion.SEMANTIC_BOUNDARY
    final_selected = result.final_primitive_relevance >= fusion.SEMANTIC_BOUNDARY
    if bool((base_selected & ~final_selected).any()):
        raise RuntimeError("scale-aware fusion removed frozen O1 support")

    output.parent.mkdir(parents=True, exist_ok=True)
    cache = {
        "schema": OUTPUT_SCHEMA,
        "query_scores": result.final_primitive_relevance,
        "valid": positive.valid.bool().cpu().contiguous(),
        "xyz": torch.as_tensor(positive_raw["xyz"]).float().cpu().contiguous(),
        "metadata": {
            "query_names": list(positive.query_ids),
            "score_semantics": "continuous_monotone_O1_native_v3_scale_aware_support",
            "score_postprocess": "none_already_frozen_knn10_per_scale_minmax",
            "producer": authority["producer"],
            "execution_authority": authority["verified_record"],
        },
        "selection": {
            "selected_scale_indices": selected_scale,
            "raw_smoothed_peaks": raw_peaks,
            "seed_region_indices": result.seed_region_indices,
            "query_gate": result.query_gate,
            "relation_selected_region_masks": result.relation_selected_region_masks,
            "relation_path_support": result.relation_path_support,
        },
    }
    write_torch_noclobber(output, cache)
    changed_per_query = (result.final_primitive_relevance != result.primitive_unary).sum(
        dim=0
    )
    crossings_per_query = (~base_selected & final_selected).sum(dim=0)
    report = {
        "schema": OUTPUT_SCHEMA,
        "status": "figurines_O1_native_v3_scale_aware_premetric_complete",
        "scene": "figurines",
        "method_contract_sha256": fusion.READOUT_CONTRACT_SHA256,
        "execution_authority": authority["verified_record"],
        "inputs": authority["inputs"],
        "output_cache": file_record(output),
        "query_count": int(result.primitive_unary.shape[1]),
        "valid_primitives": int(positive.valid.sum()),
        "selected_scale_indices": selected_scale.tolist(),
        "query_gate": result.query_gate.tolist(),
        "selected_relation_region_count": result.relation_selected_region_masks.sum(
            dim=0
        ).tolist(),
        "changed_primitive_query_cells": int(result.changed_primitive_query_cells),
        "changed_primitives_per_query": changed_per_query.tolist(),
        "new_threshold_crossings_per_query": crossings_per_query.tolist(),
        "total_new_threshold_crossings": int(crossings_per_query.sum()),
        "base_threshold_support": int(base_selected.sum()),
        "final_threshold_support": int(final_selected.sum()),
        "monotone_support_preserved": True,
        "seed_only_queries_exact_O1": [
            int(index)
            for index in torch.where(
                result.relation_selected_region_masks.sum(dim=0) <= 1
            )[0].tolist()
        ],
        "access_audit": access_audit(),
        "metric_execution_authorized": False,
    }
    write_frozen_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-authority", required=True)
    parser.add_argument("--expected-execution-authority-sha256", required=True)
    parser.add_argument("--knn-chunk-size", type=int, default=65536)
    args = parser.parse_args()
    report = materialize(args)
    print(report)


if __name__ == "__main__":
    main()
