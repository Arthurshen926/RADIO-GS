#!/usr/bin/env python3
"""Freeze source-only text subsets for a same-axis O0 missing-core sentinel.

The subset is selected exclusively from query-independent AcceptedV2 region
descriptors and frozen target-blind SigLIP2 text banks.  Exact source instance
membership is deliberately not opened here.  The emitted positive rows and the
four canonical negatives can be evaluated together by the existing cold-stream
SurfaceRegion materializer, avoiding a second dense field traversal.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
    write_torch_noclobber,
)


AUTHORITY_SCHEMA = "radio_gs.source_same_axis_o0_sentinel_execution_authority.v1"
OUTPUT_SCHEMA = "radio_gs.source_same_axis_o0_query_subset.v1"
LOGIT_SCALE = 10.0
NEUTRAL_PROBABILITY = 0.5


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must be an exact file record")
    path = validate_file_record(value, label=label)
    return {"path": str(path), "sha256": str(value["sha256"])}


def _deferred_record(value: object, *, label: str) -> dict[str, str]:
    """Validate a later-stage binding without opening or hashing its content."""

    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must be an exact deferred file record")
    digest = value.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{label} SHA-256 differs")
    path = Path(str(value.get("path", ""))).expanduser().resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must resolve to a regular non-symlink file")
    return {"path": str(path), "sha256": digest}


def validate_execution_authority(value: object) -> dict[str, Any]:
    required = {
        "schema",
        "schema_version",
        "status",
        "implementation",
        "dense_stream_implementation",
        "frozen_method",
        "same_axis_o0_contract",
        "scene0001_mechanism_gate",
        "thermal_policy",
        "text_banks",
        "canonical_negative_bank",
        "scenes",
        "output_root",
        "source_access",
        "benchmark_execution_authorized",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("same-axis source O0 authority fields differ")
    authority = dict(value)
    expected_method = {
        "query_bank_order": [
            "imagenet1k_primary_fit",
            "counterfactual_attributes_fit",
            "high_precision_part_of_fit",
            "lexical_sibling_relation_fit",
            "synonym_relation_fit",
        ],
        "region_query_selection": (
            "lower_index_argmax_raw_canonical_negative_probability"
        ),
        "raw_probability": (
            "sigmoid(10*(region_query_cosine-hardest_canonical_negative_cosine))"
        ),
        "absolute_threshold": NEUTRAL_PROBABILITY,
        "absolute_comparator": "strictly_greater_than",
        "selected_positive_order": "ascending_global_bank_row",
        "combined_stream_order": "selected_positive_then_four_canonical_negative",
        "query_strings_forwarded_to_streamer": False,
        "source_instance_labels_used_for_query_generation": False,
        "per_scene_hyperparameters": False,
    }
    expected_access = {
        "source_accepted_v2_region_descriptors_opened": True,
        "target_blind_text_banks_opened": True,
        "canonical_negative_bank_opened": True,
        "source_instance_labels_opened": False,
        "source_validation_instance_labels_opened": False,
        "benchmark_images_opened": False,
        "benchmark_labels_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_queries_opened": False,
        "benchmark_metrics_opened": False,
        "target_heldout_opened": False,
        "target_metrics_computed": False,
    }
    expected_o0 = {
        "raw_score_source": "combined_cold_stream_float16_N_by_3_by_Q",
        "positive_negative_split": "query_record_role",
        "canonical_negative_probability": (
            "sigmoid(10*(positive_cosine-hardest_of_four_negative_cosines))"
        ),
        "knn": 10,
        "knn_chunk_size": 65536,
        "scale_selection": "highest_raw_smoothed_peak_lower_scale_tie_break",
        "normalization": "independent_per_scale_query_minmax_then_clip_0_1",
        "o0_positive": "strictly_greater_than_0p6",
        "qualified_anchor": "valid_core_positive_fraction_at_least_0p75",
        "missing_core": "valid_core_O0_score_less_than_or_equal_to_0p6",
        "source_instance_membership_used_only_after_O0_is_frozen": True,
    }
    expected_gate = {
        "minimum_qualified_anchor_query_pairs": 32,
        "minimum_missing_core_units": 256,
        "minimum_hard_positive_missing_units": 32,
        "minimum_hard_negative_missing_units": 32,
        "failure_action": "do_not_open_scene0004_membership_labels",
    }
    expected_thermal = {
        "physical_gpu": 1,
        "thermal_pacing_seconds_per_batch": 0.25,
        "telemetry_poll_seconds": 300,
        "no_pause_below_temperature_c": 81,
        "maximum_temperature_c": 88,
        "maximum_consecutive_overheat_polls": 3,
        "maximum_power_limit_w": 300.5,
        "peer_GPU_monitoring": False,
    }
    if (
        authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("status")
        != "authorized_source_only_same_axis_O0_1train_1heldout_sentinel"
        or authority.get("frozen_method") != expected_method
        or authority.get("same_axis_o0_contract") != expected_o0
        or authority.get("scene0001_mechanism_gate") != expected_gate
        or authority.get("thermal_policy") != expected_thermal
        or authority.get("source_access") != expected_access
        or authority.get("benchmark_execution_authorized") is not False
    ):
        raise ValueError("same-axis source O0 authority header differs")
    authority["implementation"] = _record(
        authority["implementation"], label="subset implementation"
    )
    authority["dense_stream_implementation"] = _record(
        authority["dense_stream_implementation"],
        label="dense stream implementation",
    )
    banks = authority.get("text_banks")
    if not isinstance(banks, list) or len(banks) != 5:
        raise ValueError("same-axis source O0 text-bank list differs")
    for index, bank in enumerate(banks):
        if not isinstance(bank, Mapping) or set(bank) != {
            "component_id",
            "path",
            "sha256",
        }:
            raise ValueError("same-axis source O0 text-bank record differs")
        record = _record(
            {"path": bank["path"], "sha256": bank["sha256"]},
            label=f"text bank {index}",
        )
        banks[index] = {"component_id": str(bank["component_id"]), **record}
    authority["text_banks"] = banks
    authority["canonical_negative_bank"] = _record(
        authority["canonical_negative_bank"], label="canonical negative bank"
    )
    scenes = authority.get("scenes")
    if not isinstance(scenes, list) or [item.get("scene_id") for item in scenes] != [
        "scene0001_00",
        "scene0004_00",
    ]:
        raise ValueError("same-axis source O0 sentinel scene order differs")
    expected_splits = ("source_train", "source_validation")
    for item, split in zip(scenes, expected_splits):
        if not isinstance(item, Mapping) or set(item) != {
            "scene_id",
            "split",
            "accepted_v2",
            "source_membership_authority",
            "dense_stream_inputs",
            "execution",
        }:
            raise ValueError("same-axis source O0 scene fields differ")
        if item["split"] != split:
            raise ValueError("same-axis source O0 scene split differs")
        item["accepted_v2"] = _record(
            item["accepted_v2"], label=f"{item['scene_id']} AcceptedV2"
        )
        # Membership is SHA-bound now but remains unopened by this producer.
        membership = item["source_membership_authority"]
        if not isinstance(membership, Mapping) or set(membership) != {
            "path",
            "sha256",
        }:
            raise ValueError("source membership record differs")
        stream = item["dense_stream_inputs"]
        if not isinstance(stream, Mapping) or set(stream) != {
            "field_checkpoint",
            "factorized_primitive_state",
            "factorized_radio_cache_sha256",
            "support_graph",
            "readout_checkpoint",
            "readout_legacy_radio_authority",
            "official_radio_checkpoint",
        }:
            raise ValueError("dense stream input fields differ")
        for name in (
            "field_checkpoint",
            "factorized_primitive_state",
            "support_graph",
            "readout_checkpoint",
            "readout_legacy_radio_authority",
            "official_radio_checkpoint",
        ):
            stream[name] = _deferred_record(
                stream[name], label=f"{item['scene_id']} {name}"
            )
        digest = stream["factorized_radio_cache_sha256"]
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("factorized RADIO cache SHA-256 differs")
        execution = item["execution"]
        if not isinstance(execution, Mapping) or set(execution) != {
            "query_subset_authorized",
            "dense_stream_authorized",
            "dense_stream_condition",
        }:
            raise ValueError("same-axis source O0 execution policy differs")
    root = Path(str(authority["output_root"])).expanduser().resolve()
    if not root.is_absolute():
        raise ValueError("same-axis source O0 output root must be absolute")
    authority["output_root"] = str(root)
    return authority


def select_dominant_absolute_queries(
    *,
    region_descriptors: torch.Tensor,
    positive_embeddings: torch.Tensor,
    negative_embeddings: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Select a fixed text row per absolute region without source labels."""

    region = torch.as_tensor(region_descriptors).detach().float().cpu()
    positive = torch.as_tensor(positive_embeddings).detach().float().cpu()
    negative = torch.as_tensor(negative_embeddings).detach().float().cpu()
    if (
        region.ndim != 2
        or positive.ndim != 2
        or negative.ndim != 2
        or region.shape[1] != positive.shape[1]
        or region.shape[1] != negative.shape[1]
        or min(region.shape[0], positive.shape[0], negative.shape[0]) <= 0
        or not bool(torch.isfinite(region).all())
        or not bool(torch.isfinite(positive).all())
        or not bool(torch.isfinite(negative).all())
    ):
        raise ValueError("same-axis source O0 query-selection inputs differ")
    region = F.normalize(region, dim=1, eps=1e-8)
    positive = F.normalize(positive, dim=1, eps=1e-8)
    negative = F.normalize(negative, dim=1, eps=1e-8)
    positive_cosine = region @ positive.T
    hardest_negative = (region @ negative.T).amax(dim=1)
    probability = torch.sigmoid(
        LOGIT_SCALE * (positive_cosine - hardest_negative[:, None])
    )
    maximum_probability, dominant_global = probability.max(dim=1)
    absolute = maximum_probability > NEUTRAL_PROBABILITY
    selected_global = torch.unique(dominant_global[absolute], sorted=True)
    global_to_subset = torch.full(
        (positive.shape[0],), -1, dtype=torch.long
    )
    global_to_subset[selected_global] = torch.arange(
        selected_global.numel(), dtype=torch.long
    )
    dominant_subset = torch.full_like(dominant_global, -1)
    dominant_subset[absolute] = global_to_subset[dominant_global[absolute]]
    return {
        "maximum_probability": maximum_probability.contiguous(),
        "dominant_global_index": dominant_global.long().contiguous(),
        "absolute_region_mask": absolute.contiguous(),
        "selected_global_indices": selected_global.long().contiguous(),
        "dominant_positive_subset_index": dominant_subset.long().contiguous(),
    }


def _load_bank(record: Mapping[str, str], *, label: str) -> dict[str, Any]:
    raw, _, _ = load_torch_mapping(
        record["path"],
        expected_sha256=record["sha256"],
        map_location="cpu",
        label=label,
    )
    embeddings = torch.as_tensor(raw.get("embeddings")).detach().float().cpu()
    queries = raw.get("queries")
    if (
        embeddings.ndim != 2
        or embeddings.shape[1] != 1536
        or not isinstance(queries, list)
        or len(queries) != embeddings.shape[0]
        or any(not isinstance(value, str) or not value for value in queries)
        or raw.get("benchmark_vocabulary_opened", False) is not False
        or raw.get("uses_benchmark_vocabulary_for_construction", False)
        is not False
    ):
        raise ValueError(f"{label} contract differs")
    return {"raw": raw, "embeddings": embeddings, "queries": queries}


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"source O0 subset output exists: {output}")
    raw, authority_sha, authority_path = load_json_object(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
        label="same-axis source O0 execution authority",
    )
    authority = validate_execution_authority(raw)
    if authority["implementation"] != file_record(Path(__file__).resolve()):
        raise ValueError("same-axis source O0 subset implementation changed")
    scene_rows = [
        item for item in authority["scenes"] if item["scene_id"] == args.scene_id
    ]
    if len(scene_rows) != 1:
        raise ValueError("same-axis source O0 scene identity differs")
    scene = scene_rows[0]
    if scene["execution"]["query_subset_authorized"] is not True:
        raise ValueError("same-axis source O0 query subset is not authorized")
    expected_output = Path(authority["output_root"]) / args.scene_id / (
        "combined_text_subset.pt"
    )
    if output != expected_output:
        raise ValueError("same-axis source O0 subset output path differs")
    accepted, _, _ = load_torch_mapping(
        scene["accepted_v2"]["path"],
        expected_sha256=scene["accepted_v2"]["sha256"],
        map_location="cpu",
        label=f"{args.scene_id} AcceptedV2",
    )
    if (
        accepted.get("scene_id") != args.scene_id
        or torch.as_tensor(accepted.get("accepted_v2_e0")).shape != (4096, 1536)
    ):
        raise ValueError("same-axis source O0 AcceptedV2 axes differ")
    positive_parts: list[torch.Tensor] = []
    positive_records: list[dict[str, Any]] = []
    offset = 0
    for bank_record in authority["text_banks"]:
        bank = _load_bank(bank_record, label=bank_record["component_id"])
        positive_parts.append(bank["embeddings"])
        for row, query in enumerate(bank["queries"]):
            positive_records.append(
                {
                    "global_index": offset + row,
                    "component_id": bank_record["component_id"],
                    "component_row": row,
                    "query": query,
                }
            )
        offset += len(bank["queries"])
    positive = torch.cat(positive_parts, dim=0).contiguous()
    negative_raw, _, _ = load_torch_mapping(
        authority["canonical_negative_bank"]["path"],
        expected_sha256=authority["canonical_negative_bank"]["sha256"],
        map_location="cpu",
        label="same-axis source O0 canonical negative bank",
    )
    negative = torch.as_tensor(negative_raw.get("embeddings")).detach().float().cpu()
    negative_queries = negative_raw.get("queries")
    if negative.shape != (4, 1536) or negative_queries != [
        "object",
        "things",
        "stuff",
        "texture",
    ]:
        raise ValueError("same-axis source O0 canonical negatives differ")
    selected = select_dominant_absolute_queries(
        region_descriptors=accepted["accepted_v2_e0"],
        positive_embeddings=positive,
        negative_embeddings=negative,
    )
    selected_global = selected["selected_global_indices"]
    positive_ids = [f"positive_{int(index):05d}" for index in selected_global]
    negative_ids = [f"canonical_negative_{index}" for index in range(4)]
    combined = torch.cat([positive[selected_global], negative], dim=0).contiguous()
    query_records = [
        {"query_id": query_id, "role": "positive", **positive_records[int(index)]}
        for query_id, index in zip(positive_ids, selected_global.tolist())
    ] + [
        {
            "query_id": query_id,
            "role": "canonical_negative",
            "canonical_negative_row": index,
            "query": negative_queries[index],
        }
        for index, query_id in enumerate(negative_ids)
    ]
    payload = {
        "schema": OUTPUT_SCHEMA,
        "schema_version": 1,
        "scene_id": args.scene_id,
        "split": scene["split"],
        "queries": positive_ids + negative_ids,
        "embeddings": combined,
        "query_records": query_records,
        "positive_query_count": len(positive_ids),
        "canonical_negative_query_count": len(negative_ids),
        "region_maximum_probability": selected["maximum_probability"],
        "region_dominant_global_index": selected["dominant_global_index"],
        "region_absolute_mask": selected["absolute_region_mask"],
        "region_dominant_positive_subset_index": selected[
            "dominant_positive_subset_index"
        ],
        "selected_positive_global_indices": selected_global,
        "method": authority["frozen_method"],
        "execution_authority": {
            "path": str(authority_path),
            "sha256": authority_sha,
        },
        "input_authority": {
            "accepted_v2": scene["accepted_v2"],
            "text_banks": authority["text_banks"],
            "canonical_negative_bank": authority["canonical_negative_bank"],
        },
        "source_access": authority["source_access"],
    }
    payload["channel_sha256"] = {
        name: tensor_sha256(payload[name])
        for name in (
            "embeddings",
            "region_maximum_probability",
            "region_dominant_global_index",
            "region_absolute_mask",
            "region_dominant_positive_subset_index",
            "selected_positive_global_indices",
        )
    }
    payload["content_authority_sha256"] = canonical_json_sha256(
        {
            "scene_id": payload["scene_id"],
            "split": payload["split"],
            "queries": payload["queries"],
            "query_records": payload["query_records"],
            "method": payload["method"],
            "execution_authority": payload["execution_authority"],
            "input_authority": payload["input_authority"],
            "source_access": payload["source_access"],
            "channel_sha256": payload["channel_sha256"],
        }
    )
    write_torch_noclobber(output, payload)
    return {
        "status": "source_same_axis_O0_query_subset_complete",
        "scene_id": args.scene_id,
        "output": file_record(output),
        "positive_queries": len(positive_ids),
        "canonical_negative_queries": len(negative_ids),
        "absolute_regions": int(selected["absolute_region_mask"].sum()),
        "source_instance_labels_opened": False,
        "benchmark_execution_authorized": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-authority", required=True)
    parser.add_argument("--expected-execution-authority-sha256", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    result = materialize(build_parser().parse_args(argv))
    print(result)


if __name__ == "__main__":
    main()
