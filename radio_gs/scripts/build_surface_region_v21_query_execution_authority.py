#!/usr/bin/env python3
"""Build an exact-query V2.1 relevance authority after source promotion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from radio_gs.interfaces import surface_region_v21_target as target_formal
from radio_gs.interfaces.surface_region_v21_query_relevance import (
    IMPLEMENTATION_DEPENDENCIES,
    IMPLEMENTATION_PATH,
    PREREGISTRATION_PATH,
    QUERY_EXECUTION_SCHEMA,
    query_relevance_access_audit,
)
from radio_gs.interfaces.surface_region_v21_source_gate import (
    validate_source_pilot_chain,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    load_torch_mapping,
    write_frozen_json,
)


def _canonical_existing(value: object, *, label: str) -> Path:
    raw = str(value)
    path = Path(raw).expanduser().resolve()
    if raw != str(path) or not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be an existing canonical regular file")
    return path


def _canonical_new(value: object, *, label: str) -> Path:
    raw = str(value)
    path = Path(raw).expanduser().resolve()
    if raw != str(path) or path.exists() or path.is_symlink():
        raise ValueError(f"{label} must be a new canonical path")
    return path


def _promoted_negative(source_gate: Mapping[str, Any]) -> dict[str, str]:
    raw, _, _ = load_json_object(
        source_gate["execution_authority"]["path"],
        expected_sha256=source_gate["execution_authority"]["sha256"],
        label="promoted V2.1 source execution authority",
    )
    value = raw.get("canonical_negative_bank")
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError("promoted V2.1 canonical-negative record differs")
    return {"path": str(value["path"]), "sha256": str(value["sha256"])}


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_gate = validate_source_pilot_chain(
        args.source_pilot_result,
        expected_sha256=args.expected_source_pilot_result_sha256,
        require_promotion=True,
    )
    if source_gate.get("source_promotion_authorized") is not True:
        raise ValueError("V2.1 source promotion is not authorized")

    authority_output = _canonical_new(
        args.output_authority, label="query execution authority output"
    )
    relevance_output = _canonical_new(
        args.query_relevance_output, label="query relevance output"
    )
    if authority_output == relevance_output:
        raise ValueError("query execution authority and relevance outputs must differ")
    descriptor_path = _canonical_existing(
        args.target_descriptor, label="V2.1 target descriptor"
    )
    positive_path = _canonical_existing(
        args.positive_text_cache, label="official positive text cache"
    )
    descriptor_record = file_record(descriptor_path)
    descriptor_raw, _, _ = load_torch_mapping(
        descriptor_path,
        expected_sha256=descriptor_record["sha256"],
        map_location="cpu",
        label="V2.1 target descriptor",
    )
    descriptor = target_formal.validate_target_descriptor_authority(descriptor_raw)
    target_execution = target_formal.validate_target_execution_authority(
        descriptor["target_execution_authority"]["path"],
        expected_sha256=descriptor["target_execution_authority"]["sha256"],
        expected_scene_id=descriptor["scene_id"],
        expected_output=descriptor_path,
    )
    if (
        target_execution["source_pilot_result"] != source_gate["source_result"]
        or target_execution["target_inputs"]["v21_checkpoint"]
        != source_gate["checkpoint"]
        or target_execution["target_inputs"]["v21_normalization"]
        != source_gate["normalization_authority"]
    ):
        raise ValueError("V2.1 target descriptor uses another promoted source")
    authority = {
        "schema": QUERY_EXECUTION_SCHEMA,
        "schema_version": 1,
        "status": "authorized_after_v21_source_promotion_for_calibrated_query_relevance",
        "scene_id": descriptor["scene_id"],
        "physical_space_id": descriptor["physical_space_id"],
        "source_pilot_result": dict(source_gate["source_result"]),
        "implementation": file_record(IMPLEMENTATION_PATH),
        "implementation_dependencies": {
            name: file_record(path)
            for name, path in IMPLEMENTATION_DEPENDENCIES.items()
        },
        "preregistration": file_record(PREREGISTRATION_PATH),
        "target_descriptor": descriptor_record,
        "positive_text_cache": file_record(positive_path),
        "canonical_negative_bank": _promoted_negative(source_gate),
        "query_relevance_output": str(relevance_output),
        "query_execution_authorized": True,
        "metric_execution_authorized": False,
        "access_audit": query_relevance_access_audit(),
    }
    written = write_frozen_json(authority_output, authority)
    return {
        "status": "v21_query_execution_authority_built_after_source_pass",
        "authority": file_record(written),
        "query_relevance_output": str(relevance_output),
        "scene_id": descriptor["scene_id"],
        "physical_space_id": descriptor["physical_space_id"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pilot-result", required=True)
    parser.add_argument("--expected-source-pilot-result-sha256", required=True)
    parser.add_argument("--target-descriptor", required=True)
    parser.add_argument("--positive-text-cache", required=True)
    parser.add_argument("--query-relevance-output", required=True)
    parser.add_argument("--output-authority", required=True)
    return parser


def main() -> None:
    print(json.dumps(build(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
