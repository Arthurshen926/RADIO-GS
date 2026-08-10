#!/usr/bin/env python3
"""Materialize the parameter-free target-corroborated native-V3 readout."""

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
    corroborated_scale_aware_native_v3_support_fusion as corroborated,
)
from radio_gs.querying import scale_aware_native_v3_support_fusion as legacy_fusion
from radio_gs.scripts import (
    materialize_lerf_o1_native_v3_scale_aware_fusion as frozen_materializer,
)
from radio_gs.scripts import (
    materialize_lerf_o1_native_v3_scale_aware_fusion_cross_scene as cross_scene,
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


EXECUTION_SCHEMA = "radio_gs.lerf_o1_native_v3_corroborated_execution.v1"
EXECUTION_STATUS = "authorized_post_result_premetric_corroboration_audit"
OUTPUT_SCHEMA = "radio_gs.lerf_o1_native_v3_corroborated_external_scores.v1"
O1_RESULT_CONTRACTS = {"figurines_oracle_matrix", "streaming_source_only"}


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
        label="corroborated O1/native-V3 execution authority",
    )
    authority = dict(raw)
    required = {
        "schema",
        "schema_version",
        "status",
        "scene_id",
        "o1_result_contract",
        "method_contract_sha256",
        "producer",
        "inputs",
        "outputs",
        "constants",
        "access_audit",
        "metric_execution_authorized",
    }
    scene_id = authority.get("scene_id")
    if (
        set(authority) != required
        or authority.get("schema") != EXECUTION_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("status") != EXECUTION_STATUS
        or not isinstance(scene_id, str)
        or not scene_id
        or authority.get("o1_result_contract") not in O1_RESULT_CONTRACTS
        or authority.get("method_contract_sha256")
        != corroborated.READOUT_CONTRACT_SHA256
        or authority.get("constants")
        != {
            "canonical_negative_logit_scale": frozen_materializer.LOGIT_SCALE,
            "knn_k": frozen_materializer.KNN_K,
            "semantic_boundary": corroborated.SEMANTIC_BOUNDARY,
            "native_v3_relation_threshold": corroborated.RELATION_THRESHOLD,
            "maximum_regions": corroborated.MAXIMUM_REGIONS,
            "path_method": corroborated.PATH_METHOD,
            "target_corroboration": (
                "any_valid_semantic_core_primitive_at_registered_scale_ge_0p6"
            ),
        }
        or authority.get("access_audit") != frozen_materializer.access_audit()
        or authority.get("metric_execution_authorized") is not False
    ):
        raise ValueError("corroborated execution header differs")
    producer = _record(authority["producer"], label="corroborated producer")
    if Path(producer["path"]).resolve() != Path(__file__).resolve():
        raise ValueError("corroborated producer path differs")
    inputs = authority.get("inputs")
    expected_inputs = {
        "o1_manifest",
        "o1_positive",
        "o1_negative",
        "native_v3_feature",
        "native_v3_inference",
        "prior_candidate_cache",
        "prior_premetric_report",
        "frozen_evaluator",
        "frozen_figurines_materializer",
        "frozen_cross_scene_materializer",
    }
    if not isinstance(inputs, Mapping) or set(inputs) != expected_inputs:
        raise ValueError("corroborated input set differs")
    verified_inputs = {
        name: _record(inputs[name], label=f"corroborated input {name}")
        for name in sorted(expected_inputs)
    }
    dependencies = {
        "frozen_figurines_materializer": Path(frozen_materializer.__file__).resolve(),
        "frozen_cross_scene_materializer": Path(cross_scene.__file__).resolve(),
    }
    for name, expected in dependencies.items():
        if Path(verified_inputs[name]["path"]).resolve() != expected:
            raise ValueError(f"corroborated dependency {name} differs")
    outputs = authority.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {"cache", "report"}:
        raise ValueError("corroborated output set differs")
    canonical_outputs = {
        name: str(Path(value).expanduser().resolve())
        for name, value in outputs.items()
    }
    if any(value != outputs[name] for name, value in canonical_outputs.items()):
        raise ValueError("corroborated outputs must be canonical absolute paths")
    if len(set(canonical_outputs.values())) != 2:
        raise ValueError("corroborated output paths must differ")
    authority["producer"] = producer
    authority["inputs"] = verified_inputs
    authority["outputs"] = canonical_outputs
    authority["verified_record"] = {"path": str(source), "sha256": digest}
    return authority


def _validate_o1_result(
    manifest: Mapping[str, Any],
    *,
    authority: Mapping[str, Any],
) -> None:
    scene_id = str(authority["scene_id"])
    records = authority["inputs"]
    if authority["o1_result_contract"] == "streaming_source_only":
        cross_scene.validate_o1_materialization_result(
            manifest,
            scene_id=scene_id,
            positive_record=records["o1_positive"],
            negative_record=records["o1_negative"],
        )
        return
    if (
        scene_id != "figurines"
        or manifest.get("status")
        != "complete_source_only_oracle_matrix_materialization"
        or manifest.get("scene") != scene_id
        or manifest.get("o1", {}).get("positive") != records["o1_positive"]
        or manifest.get("o1", {}).get("negative") != records["o1_negative"]
        or manifest.get("benchmark_metrics_opened") is not False
    ):
        raise ValueError("corroborated Figurines O1 manifest differs")


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    authority = validate_execution_authority(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
    )
    scene_id = str(authority["scene_id"])
    output = Path(authority["outputs"]["cache"])
    report_path = Path(authority["outputs"]["report"])
    if output.exists() or output.is_symlink() or report_path.exists() or report_path.is_symlink():
        raise FileExistsError("corroborated output already exists")
    records = authority["inputs"]
    manifest, _, _ = load_json_object(
        records["o1_manifest"]["path"],
        expected_sha256=records["o1_manifest"]["sha256"],
        label=f"{scene_id} O1 materialization result",
    )
    _validate_o1_result(manifest, authority=authority)
    positive, negative, positive_raw, _negative_raw = (
        frozen_materializer._paired_o1_caches(authority)
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
        label=f"{scene_id} native-V3 AcceptedV2",
    )
    accepted = validate_target_accepted_v2_authority(accepted_raw)
    if (
        feature["scene_id"] != scene_id
        or inference["scene_id"] != scene_id
        or inference["feature_authority"] != records["native_v3_feature"]
        or inference["selected_rule"]
        != {
            "method": "dual_path_widest",
            "maximum_regions": corroborated.MAXIMUM_REGIONS,
            "threshold": corroborated.RELATION_THRESHOLD,
        }
        or accepted["physical_space_authority"]["geometry_checkpoint_sha256"]
        != positive.renderer_geometry_checkpoint_sha256
        or not torch.equal(feature["region_rows"], accepted["region_rows"])
        or not torch.equal(feature["token_mask"], accepted["token_mask"])
    ):
        raise ValueError(f"{scene_id} corroborated physical binding differs")

    per_scale, selected_scale, raw_peaks = (
        frozen_materializer.frozen_multiscale_relevance(
            positive_scores=positive.query_scores,
            negative_scores=negative.query_scores,
            xyz=positive_raw["xyz"],
            valid=positive.valid,
            chunk_size=int(args.knn_chunk_size),
        )
    )
    result = corroborated.corroborated_scale_aware_native_v3_support_fusion(
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
        label=f"{scene_id} prior scale-aware cache",
    )
    prior_report, _, _ = load_json_object(
        records["prior_premetric_report"]["path"],
        expected_sha256=records["prior_premetric_report"]["sha256"],
        label=f"{scene_id} prior scale-aware report",
    )
    prior_final = torch.as_tensor(prior_raw["query_scores"]).float().cpu()
    base = result.primitive_unary
    final = result.final_primitive_relevance
    base_selected = base >= corroborated.SEMANTIC_BOUNDARY
    prior_selected = prior_final >= corroborated.SEMANTIC_BOUNDARY
    final_selected = final >= corroborated.SEMANTIC_BOUNDARY
    prior_crossings = ~base_selected & prior_selected
    final_crossings = ~base_selected & final_selected
    if (
        prior_raw.get("schema") != frozen_materializer.OUTPUT_SCHEMA
        or prior_raw.get("metadata", {}).get("query_names")
        != list(positive.query_ids)
        or prior_final.shape != base.shape
        or prior_report.get("scene") != scene_id
        or prior_report.get("output_cache") != records["prior_candidate_cache"]
        or prior_report.get("method_contract_sha256")
        != legacy_fusion.READOUT_CONTRACT_SHA256
        or int(prior_crossings.sum())
        != int(prior_report["total_new_threshold_crossings"])
        or bool((final > prior_final).any())
        or bool((final_crossings & ~prior_crossings).any())
        or bool((base_selected & ~final_selected).any())
    ):
        raise ValueError("corroborated prior-candidate intersection differs")

    output.parent.mkdir(parents=True, exist_ok=True)
    cache = {
        "schema": OUTPUT_SCHEMA,
        "query_scores": final,
        "valid": positive.valid.bool().cpu().contiguous(),
        "xyz": torch.as_tensor(positive_raw["xyz"]).float().cpu().contiguous(),
        "metadata": {
            "query_names": list(positive.query_ids),
            "score_semantics": "continuous_monotone_O1_native_v3_target_corroborated",
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
            "corroborated_relation_region_masks": (
                result.corroborated_relation_region_masks
            ),
            "relation_path_support": result.relation_path_support,
        },
    }
    write_torch_noclobber(output, cache)
    prior_per_query = prior_crossings.sum(dim=0)
    retained_per_query = final_crossings.sum(dim=0)
    report = {
        "schema": OUTPUT_SCHEMA,
        "status": f"{scene_id}_O1_native_v3_corroborated_premetric_complete",
        "scene": scene_id,
        "method_contract_sha256": corroborated.READOUT_CONTRACT_SHA256,
        "parent_method_contract_sha256": legacy_fusion.READOUT_CONTRACT_SHA256,
        "execution_authority": authority["verified_record"],
        "inputs": authority["inputs"],
        "output_cache": file_record(output),
        "query_names": list(positive.query_ids),
        "query_count": int(base.shape[1]),
        "valid_primitives": int(positive.valid.sum()),
        "selected_scale_indices": selected_scale.tolist(),
        "query_gate": result.query_gate.tolist(),
        "relation_selected_region_count": result.relation_selected_region_masks.sum(
            dim=0
        ).tolist(),
        "corroborated_relation_region_count": (
            result.corroborated_relation_region_masks.sum(dim=0).tolist()
        ),
        "changed_primitive_query_cells": int(result.changed_primitive_query_cells),
        "prior_new_threshold_crossings_per_query": prior_per_query.tolist(),
        "retained_new_threshold_crossings_per_query": retained_per_query.tolist(),
        "prior_total_new_threshold_crossings": int(prior_crossings.sum()),
        "retained_total_new_threshold_crossings": int(final_crossings.sum()),
        "removed_total_new_threshold_crossings": int(
            prior_crossings.sum() - final_crossings.sum()
        ),
        "retained_crossings_are_exact_subset_of_prior": True,
        "base_threshold_support": int(base_selected.sum()),
        "final_threshold_support": int(final_selected.sum()),
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
