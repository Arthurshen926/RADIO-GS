#!/usr/bin/env python3
"""Audit native-V3 corroboration in the frozen deployed query scale."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from radio_gs.interfaces.surface_region_target_accepted_v2 import (
    validate_target_accepted_v2_authority,
)
from radio_gs.querying import (
    deployed_scale_corroborated_native_v3_support_fusion as deployed,
)
from radio_gs.querying import scale_aware_native_v3_support_fusion as legacy_fusion
from radio_gs.scripts import (
    materialize_lerf_o1_native_v3_corroborated_fusion as parent,
)
from radio_gs.scripts import (
    materialize_lerf_o1_native_v3_scale_aware_fusion as frozen_materializer,
)
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
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


EXECUTION_SCHEMA = "radio_gs.lerf_o1_native_v3_deployed_corroborated_execution.v1"
EXECUTION_STATUS = "authorized_post_result_deployed_scale_premetric_audit"
OUTPUT_SCHEMA = "radio_gs.lerf_o1_native_v3_deployed_corroborated_scores.v1"


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
        label="deployed-scale corroboration execution authority",
    )
    authority = dict(raw)
    required = {
        "schema",
        "schema_version",
        "status",
        "scene_id",
        "method_contract_sha256",
        "producer",
        "parent_execution_authority",
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
        or not isinstance(authority.get("scene_id"), str)
        or not authority.get("scene_id")
        or authority.get("method_contract_sha256")
        != deployed.READOUT_CONTRACT_SHA256
        or authority.get("constants")
        != {
            "semantic_boundary": deployed.SEMANTIC_BOUNDARY,
            "native_v3_relation_threshold": deployed.RELATION_THRESHOLD,
            "maximum_regions": deployed.MAXIMUM_REGIONS,
            "path_method": deployed.PATH_METHOD,
            "target_corroboration": (
                "any_valid_target_core_primitive_at_frozen_selected_scale_ge_0p6"
            ),
        }
        or authority.get("access_audit") != frozen_materializer.access_audit()
        or authority.get("metric_execution_authorized") is not False
    ):
        raise ValueError("deployed-scale corroboration header differs")
    producer = _record(authority["producer"], label="deployed producer")
    if Path(producer["path"]).resolve() != Path(__file__).resolve():
        raise ValueError("deployed producer path differs")
    parent_record = _record(
        authority["parent_execution_authority"], label="deployed parent authority"
    )
    parent_authority = parent.validate_execution_authority(
        parent_record["path"], expected_sha256=parent_record["sha256"]
    )
    if parent_authority["scene_id"] != authority["scene_id"]:
        raise ValueError("deployed parent scene differs")
    outputs = authority.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {"cache", "report"}:
        raise ValueError("deployed outputs differ")
    canonical_outputs = {
        name: str(Path(value).expanduser().resolve())
        for name, value in outputs.items()
    }
    if any(value != outputs[name] for name, value in canonical_outputs.items()):
        raise ValueError("deployed outputs must be canonical absolute paths")
    authority["producer"] = producer
    authority["parent_execution_authority"] = parent_record
    authority["parent"] = parent_authority
    authority["outputs"] = canonical_outputs
    authority["verified_record"] = {"path": str(source), "sha256": digest}
    return authority


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    authority = validate_execution_authority(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
    )
    parent_authority = authority["parent"]
    scene_id = str(authority["scene_id"])
    records = parent_authority["inputs"]
    output = Path(authority["outputs"]["cache"])
    report_path = Path(authority["outputs"]["report"])
    if output.exists() or output.is_symlink() or report_path.exists() or report_path.is_symlink():
        raise FileExistsError("deployed-scale output already exists")
    manifest, _, _ = load_json_object(
        records["o1_manifest"]["path"],
        expected_sha256=records["o1_manifest"]["sha256"],
        label=f"{scene_id} O1 result",
    )
    parent._validate_o1_result(manifest, authority=parent_authority)
    positive, negative, positive_raw, _negative_raw = (
        frozen_materializer._paired_o1_caches(parent_authority)
    )
    feature_raw, _, _ = load_torch_mapping(
        records["native_v3_feature"]["path"],
        expected_sha256=records["native_v3_feature"]["sha256"],
        map_location="cpu",
        label=f"{scene_id} native-V3 feature",
    )
    feature = validate_feature_authority(feature_raw)
    inference_raw, _, _ = load_torch_mapping(
        records["native_v3_inference"]["path"],
        expected_sha256=records["native_v3_inference"]["sha256"],
        map_location="cpu",
        label=f"{scene_id} native-V3 inference",
    )
    inference = validate_inference_authority(inference_raw)
    accepted_record = feature["input_authority"]["accepted_v2"]
    accepted_raw, _, _ = load_torch_mapping(
        accepted_record["path"],
        expected_sha256=accepted_record["sha256"],
        map_location="cpu",
        label=f"{scene_id} AcceptedV2",
    )
    accepted = validate_target_accepted_v2_authority(accepted_raw)
    if (
        feature["scene_id"] != scene_id
        or inference["scene_id"] != scene_id
        or inference["feature_authority"] != records["native_v3_feature"]
        or accepted["physical_space_authority"]["geometry_checkpoint_sha256"]
        != positive.renderer_geometry_checkpoint_sha256
        or not torch.equal(feature["region_rows"], accepted["region_rows"])
        or not torch.equal(feature["token_mask"], accepted["token_mask"])
    ):
        raise ValueError("deployed-scale physical binding differs")
    per_scale, selected_scale, raw_peaks = (
        frozen_materializer.frozen_multiscale_relevance(
            positive_scores=positive.query_scores,
            negative_scores=negative.query_scores,
            xyz=positive_raw["xyz"],
            valid=positive.valid,
            chunk_size=int(args.knn_chunk_size),
        )
    )
    result = deployed.deployed_scale_corroborated_native_v3_support_fusion(
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
    prior_raw, _, _ = load_torch_mapping(
        records["prior_candidate_cache"]["path"],
        expected_sha256=records["prior_candidate_cache"]["sha256"],
        map_location="cpu",
        label=f"{scene_id} prior candidate",
    )
    prior_report, _, _ = load_json_object(
        records["prior_premetric_report"]["path"],
        expected_sha256=records["prior_premetric_report"]["sha256"],
        label=f"{scene_id} prior premetric",
    )
    prior_final = torch.as_tensor(prior_raw["query_scores"]).float().cpu()
    base = result.primitive_unary
    final = result.final_primitive_relevance
    base_selected = base >= deployed.SEMANTIC_BOUNDARY
    prior_crossings = ~base_selected & (prior_final >= deployed.SEMANTIC_BOUNDARY)
    final_crossings = ~base_selected & (final >= deployed.SEMANTIC_BOUNDARY)
    if (
        prior_raw.get("metadata", {}).get("query_names") != list(positive.query_ids)
        or prior_report.get("scene") != scene_id
        or int(prior_crossings.sum())
        != int(prior_report["total_new_threshold_crossings"])
        or bool((final > prior_final).any())
        or bool((final_crossings & ~prior_crossings).any())
        or bool((base_selected & ~(final >= deployed.SEMANTIC_BOUNDARY)).any())
    ):
        raise ValueError("deployed-scale crossing intersection differs")
    output.parent.mkdir(parents=True, exist_ok=True)
    cache = {
        "schema": OUTPUT_SCHEMA,
        "query_scores": final,
        "valid": positive.valid.bool().cpu().contiguous(),
        "xyz": torch.as_tensor(positive_raw["xyz"]).float().cpu().contiguous(),
        "metadata": {
            "query_names": list(positive.query_ids),
            "score_semantics": "continuous_monotone_native_v3_deployed_scale_corroborated",
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
            "deployed_corroborated_region_masks": (
                result.deployed_corroborated_region_masks
            ),
            "relation_path_support": result.relation_path_support,
        },
    }
    write_torch_noclobber(output, cache)
    report = {
        "schema": OUTPUT_SCHEMA,
        "status": f"{scene_id}_deployed_scale_corroboration_premetric_complete",
        "scene": scene_id,
        "method_contract_sha256": deployed.READOUT_CONTRACT_SHA256,
        "parent_execution_authority": authority["parent_execution_authority"],
        "execution_authority": authority["verified_record"],
        "output_cache": file_record(output),
        "query_names": list(positive.query_ids),
        "relation_selected_region_count": result.relation_selected_region_masks.sum(
            dim=0
        ).tolist(),
        "deployed_corroborated_region_count": (
            result.deployed_corroborated_region_masks.sum(dim=0).tolist()
        ),
        "changed_primitive_query_cells": int(result.changed_primitive_query_cells),
        "prior_new_threshold_crossings_per_query": prior_crossings.sum(dim=0).tolist(),
        "retained_new_threshold_crossings_per_query": final_crossings.sum(dim=0).tolist(),
        "prior_total_new_threshold_crossings": int(prior_crossings.sum()),
        "retained_total_new_threshold_crossings": int(final_crossings.sum()),
        "removed_total_new_threshold_crossings": int(
            prior_crossings.sum() - final_crossings.sum()
        ),
        "retained_crossings_are_exact_subset_of_prior": True,
        "base_threshold_support": int(base_selected.sum()),
        "final_threshold_support": int((final >= deployed.SEMANTIC_BOUNDARY).sum()),
        "monotone_support_preserved": True,
        "access_audit": frozen_materializer.access_audit(),
        "metric_execution_authorized": False,
    }
    write_frozen_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-authority", required=True)
    parser.add_argument("--expected-execution-authority-sha256", required=True)
    parser.add_argument("--knn-chunk-size", type=int, default=65536)
    print(materialize(parser.parse_args()))


if __name__ == "__main__":
    main()
