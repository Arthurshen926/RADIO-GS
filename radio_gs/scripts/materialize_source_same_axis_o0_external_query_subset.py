#!/usr/bin/env python3
"""Materialize a label-blind same-axis O0 query subset for an external source scene.

This is a deliberately thin external-scene adapter around the frozen query
selection implementation used for scene0001/scene0004.  It exists so an
additional source scene can be audited without changing the SHA-bound original
authority or pretending that the scene belongs to the original validation split.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.scripts import materialize_source_same_axis_o0_query_subset as frozen_api
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
    write_torch_noclobber,
)


AUTHORITY_SCHEMA = "radio_gs.source_same_axis_o0_external_subset_authority.v1"
OUTPUT_SCHEMA = "radio_gs.source_same_axis_o0_query_subset.v1"
SCENE_ID = "scene0002_00"
SPLIT = "source_external_validation_selector_only"


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
        "frozen_selection_implementation",
        "frozen_method",
        "accepted_v2",
        "text_banks",
        "canonical_negative_bank",
        "source_membership_authority_deferred",
        "output",
        "source_access",
        "benchmark_execution_authorized",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("external same-axis O0 subset authority fields differ")
    authority = dict(value)
    expected_access = {
        "source_accepted_v2_region_descriptors_opened": True,
        "target_blind_text_banks_opened": True,
        "canonical_negative_bank_opened": True,
        "source_instance_membership_payload_opened": False,
        "benchmark_images_opened": False,
        "benchmark_labels_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_queries_opened": False,
        "benchmark_metrics_opened": False,
        "target_heldout_opened": False,
        "target_metrics_computed": False,
    }
    if (
        authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("status")
        != "sealed_before_scene0002_label_blind_query_subset"
        or authority.get("scene_id") != SCENE_ID
        or authority.get("split") != SPLIT
        or authority.get("frozen_method")
        != {
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
            "absolute_threshold": 0.5,
            "absolute_comparator": "strictly_greater_than",
            "selected_positive_order": "ascending_global_bank_row",
            "combined_stream_order": (
                "selected_positive_then_four_canonical_negative"
            ),
            "query_strings_forwarded_to_streamer": False,
            "source_instance_labels_used_for_query_generation": False,
            "per_scene_hyperparameters": False,
        }
        or authority.get("source_access") != expected_access
        or authority.get("benchmark_execution_authorized") is not False
    ):
        raise ValueError("external same-axis O0 subset authority header differs")
    for name in (
        "implementation",
        "frozen_selection_implementation",
        "accepted_v2",
        "canonical_negative_bank",
        "source_membership_authority_deferred",
    ):
        authority[name] = _record(authority[name], label=name)
    banks = authority.get("text_banks")
    if not isinstance(banks, list) or len(banks) != 5:
        raise ValueError("external same-axis O0 text banks differ")
    clean_banks = []
    for row in banks:
        if not isinstance(row, Mapping) or set(row) != {
            "component_id",
            "path",
            "sha256",
        }:
            raise ValueError("external same-axis O0 text-bank record differs")
        clean_banks.append(
            {
                "component_id": str(row["component_id"]),
                **_record(
                    {"path": row["path"], "sha256": row["sha256"]},
                    label=f"text bank {row['component_id']}",
                ),
            }
        )
    authority["text_banks"] = clean_banks
    output = Path(str(authority["output"])).expanduser().resolve()
    if not output.is_absolute():
        raise ValueError("external same-axis O0 subset output must be absolute")
    authority["output"] = str(output)
    return authority


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    raw, authority_sha, authority_path = load_json_object(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
        label="external same-axis O0 subset authority",
    )
    authority = validate_execution_authority(raw)
    if authority["implementation"] != file_record(Path(__file__).resolve()):
        raise ValueError("external same-axis O0 subset implementation changed")
    if authority["frozen_selection_implementation"] != file_record(
        Path(frozen_api.__file__).resolve()
    ):
        raise ValueError("frozen same-axis O0 selection implementation changed")
    output = Path(authority["output"])
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"external same-axis O0 subset output exists: {output}")
    accepted, _, _ = load_torch_mapping(
        authority["accepted_v2"]["path"],
        expected_sha256=authority["accepted_v2"]["sha256"],
        map_location="cpu",
        label="scene0002 AcceptedV2",
    )
    if (
        accepted.get("scene_id") != SCENE_ID
        or torch.as_tensor(accepted.get("accepted_v2_e0")).shape != (4096, 1536)
    ):
        raise ValueError("scene0002 AcceptedV2 axes differ")
    positive_parts: list[torch.Tensor] = []
    positive_records: list[dict[str, Any]] = []
    offset = 0
    for bank_record in authority["text_banks"]:
        bank = frozen_api._load_bank(
            bank_record, label=bank_record["component_id"]
        )
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
        label="canonical negative bank",
    )
    negative = torch.as_tensor(negative_raw.get("embeddings")).detach().float().cpu()
    negative_queries = negative_raw.get("queries")
    if negative.shape != (4, 1536) or negative_queries != [
        "object",
        "things",
        "stuff",
        "texture",
    ]:
        raise ValueError("canonical negatives differ")
    selected = frozen_api.select_dominant_absolute_queries(
        region_descriptors=accepted["accepted_v2_e0"],
        positive_embeddings=positive,
        negative_embeddings=negative,
    )
    selected_global = selected["selected_global_indices"]
    positive_ids = [f"positive_{int(index):05d}" for index in selected_global]
    negative_ids = [f"canonical_negative_{index}" for index in range(4)]
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
        "scene_id": SCENE_ID,
        "split": SPLIT,
        "queries": positive_ids + negative_ids,
        "embeddings": torch.cat([positive[selected_global], negative], dim=0).contiguous(),
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
            "accepted_v2": authority["accepted_v2"],
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
        "status": "source_same_axis_O0_external_query_subset_complete",
        "scene_id": SCENE_ID,
        "output": file_record(output),
        "positive_queries": len(positive_ids),
        "canonical_negative_queries": len(negative_ids),
        "absolute_regions": int(selected["absolute_region_mask"].sum()),
        "source_instance_membership_payload_opened": False,
        "benchmark_execution_authorized": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-authority", required=True)
    parser.add_argument("--expected-execution-authority-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    print(materialize(build_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
