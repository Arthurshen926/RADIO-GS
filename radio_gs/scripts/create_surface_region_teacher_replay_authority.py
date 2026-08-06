#!/usr/bin/env python3
"""Bind immutable fixed-core teacher caches to a registered replay run."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_torch_mapping,
    write_frozen_json,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_TYPE = "surface-region-teacher-replay-authority-v1"
_TEACHER_SEMANTICS = "fixed_core_geodesic_support_without_input_context_v1"


def create_authority(
    cache_path: str | Path,
    *,
    run_manifest: str | Path,
    output: str | Path,
) -> dict:
    cache, cache_sha256, cache_source = load_torch_mapping(
        cache_path,
        map_location="cpu",
        label="fixed-core SurfaceRegion teacher cache",
    )
    metadata = cache.get("metadata", {})
    records = metadata.get("region_records", [])
    required_tensors = (
        "official_summary_tokens",
        "official_crop_summaries",
        "teacher_mask",
    )
    if (
        metadata.get("schema_version") not in {3, 4}
        or metadata.get("split_role") not in {"train", "validation"}
        or metadata.get("teacher_region_semantics") != _TEACHER_SEMANTICS
        or metadata.get("teacher_target_source") != "fresh_official_runtime"
        or metadata.get("teacher_regions_saturated") != 0
        or metadata.get("complete_scene_regions") is not True
        or metadata.get("failed_scenes")
        or not isinstance(records, list)
        or not records
        or any(key not in cache for key in required_tensors)
        or any(len(torch.as_tensor(cache[key])) != len(records) for key in required_tensors)
    ):
        raise ValueError("cache is not a complete fresh fixed-core teacher authority")
    if metadata.get("schema_version") == 4:
        completion = metadata.get("eligibility_completion")
        roles = [record.get("row_role") for record in records]
        full_ids = {
            str(record.get("region_id", ""))
            for record in records
            if record.get("row_role") == "full_support"
        }
        if (
            not isinstance(completion, dict)
            or completion.get("schema_version") != 1
            or completion.get("validation_checkpoint_selection")
            != "full_support_rows_only"
            or not full_ids
            or any(role not in {"full_support", "eligibility_completion"} for role in roles)
            or any(
                record.get("row_role") == "eligibility_completion"
                and str(record.get("paired_full_region_id", "")) not in full_ids
                for record in records
            )
        ):
            raise ValueError(
                "paired cache is not a complete fixed-core teacher authority"
            )
    source_builder = str(metadata.get("builder_script_sha256", ""))
    teacher_contract = str(metadata.get("teacher_region_contract_sha256", ""))
    teacher_protocol = str(metadata.get("teacher_target_protocol_sha256", ""))
    radio_checkpoint = str(metadata.get("radio_checkpoint_sha256", ""))
    if any(
        _SHA256.fullmatch(value) is None
        for value in (
            source_builder,
            teacher_contract,
            teacher_protocol,
            radio_checkpoint,
        )
    ):
        raise ValueError("teacher cache lacks complete SHA-256 provenance")
    scene_names = metadata.get("scene_names")
    if (
        not isinstance(scene_names, list)
        or not scene_names
        or scene_names != sorted(set(str(value) for value in scene_names))
    ):
        raise ValueError("teacher cache scene authority is invalid")
    manifest_record = file_record(run_manifest)
    payload = {
        "artifact_type": _ARTIFACT_TYPE,
        "schema_version": 1,
        "authorization_scope": "exact_historical_cache_fixed_teacher_replay_only",
        "run_manifest": manifest_record,
        "cache": {"path": str(cache_source), "sha256": cache_sha256},
        "split_role": str(metadata["split_role"]),
        "split_file_sha256": str(metadata["split_file_sha256"]),
        "scene_names": scene_names,
        "teacher_region_contract_sha256": teacher_contract,
        "teacher_target_protocol_sha256": teacher_protocol,
        "radio_checkpoint_sha256": radio_checkpoint,
        "source_builder_script_sha256": source_builder,
    }
    write_frozen_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    create_authority(
        args.cache,
        run_manifest=args.run_manifest,
        output=args.output,
    )


if __name__ == "__main__":
    main()
