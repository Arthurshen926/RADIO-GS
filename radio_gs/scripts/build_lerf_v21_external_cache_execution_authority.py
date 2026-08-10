#!/usr/bin/env python3
"""Build the single preregistered LERF V2.1 external-cache authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from radio_gs.interfaces.surface_region_v21_query_relevance import (
    validate_query_execution_authority,
    validate_query_relevance_authority,
)
from radio_gs.interfaces.surface_region_v21_source_gate import (
    validate_source_pilot_chain,
)
from radio_gs.scripts import (
    build_lerf_region_comembership_external_cache_v21 as external,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
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


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_gate = validate_source_pilot_chain(
        args.source_pilot_result,
        expected_sha256=args.expected_source_pilot_result_sha256,
        require_promotion=True,
    )
    if source_gate.get("source_promotion_authorized") is not True:
        raise ValueError("V2.1 source promotion is not authorized")

    authority_output = _canonical_new(
        args.output_authority, label="external execution authority output"
    )
    cache_output = _canonical_new(args.output_cache, label="external cache output")
    report_output = _canonical_new(
        args.output_report, label="external cache report output"
    )
    if len({authority_output, cache_output, report_output}) != 3:
        raise ValueError("external authority/cache/report outputs must differ")
    query_execution_path = _canonical_existing(
        args.query_relevance_execution_authority,
        label="V2.1 query relevance execution authority",
    )
    relevance_path = _canonical_existing(
        args.query_relevance_authority, label="V2.1 query relevance authority"
    )
    feature_path = _canonical_existing(
        args.comembership_feature_authority,
        label="formal co-membership feature authority",
    )
    inference_path = _canonical_existing(
        args.comembership_inference_authority,
        label="formal co-membership inference authority",
    )
    renderer_path = _canonical_existing(
        args.renderer_geometry_checkpoint, label="renderer geometry checkpoint"
    )
    query_execution_record = file_record(query_execution_path)
    relevance_record = file_record(relevance_path)
    query_execution = validate_query_execution_authority(
        query_execution_path,
        expected_sha256=query_execution_record["sha256"],
        expected_output=relevance_path,
    )
    if (
        query_execution["source_pilot_result"] != source_gate["source_result"]
        or query_execution["verified_source_gate"]["checkpoint"]
        != source_gate["checkpoint"]
        or query_execution["verified_source_gate"]["normalization_authority"]
        != source_gate["normalization_authority"]
    ):
        raise ValueError("V2.1 query execution uses another promoted source")
    relevance_raw, _, _ = load_torch_mapping(
        relevance_path,
        expected_sha256=relevance_record["sha256"],
        map_location="cpu",
        label="V2.1 query relevance authority",
    )
    relevance = validate_query_relevance_authority(relevance_raw)
    if relevance["query_execution_authority"] != query_execution_record:
        raise ValueError("V2.1 relevance binds another query execution authority")
    descriptor = query_execution["verified_descriptor"]
    authority = {
        "schema": external.EXECUTION_AUTHORITY_SCHEMA,
        "schema_version": 1,
        "status": external.EXECUTION_AUTHORITY_STATUS,
        "scene_id": descriptor["scene_id"],
        "physical_space_id": descriptor["physical_space_id"],
        "source_pilot_result": dict(source_gate["source_result"]),
        "v21_checkpoint": dict(source_gate["checkpoint"]),
        "v21_normalization": dict(source_gate["normalization_authority"]),
        "canonical_negative_bank": dict(query_execution["canonical_negative_bank"]),
        "target_descriptor": dict(query_execution["target_descriptor"]),
        "positive_text_cache": dict(query_execution["positive_text_cache"]),
        "query_relevance_execution_authority": query_execution_record,
        "query_relevance_authority": relevance_record,
        "comembership_feature_authority": file_record(feature_path),
        "comembership_inference_authority": file_record(inference_path),
        "renderer_geometry_checkpoint": file_record(renderer_path),
        "preregistration": file_record(external.PREREGISTRATION),
        "implementation": file_record(Path(external.__file__).resolve()),
        "implementation_dependencies": {
            name: file_record(path)
            for name, path in external.IMPLEMENTATION_DEPENDENCIES.items()
        },
        "output_cache": str(cache_output),
        "output_report": str(report_output),
        "query_readout_authorized": True,
        "target_metric_authorized": False,
        "access_audit": external.external_cache_access_audit(),
    }
    written = write_frozen_json(authority_output, authority)
    return {
        "status": "lerf_v21_external_execution_authority_built_after_source_pass",
        "authority": file_record(written),
        "output_cache": str(cache_output),
        "output_report": str(report_output),
        "scene_id": descriptor["scene_id"],
        "physical_space_id": descriptor["physical_space_id"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pilot-result", required=True)
    parser.add_argument("--expected-source-pilot-result-sha256", required=True)
    parser.add_argument("--query-relevance-execution-authority", required=True)
    parser.add_argument("--query-relevance-authority", required=True)
    parser.add_argument("--comembership-feature-authority", required=True)
    parser.add_argument("--comembership-inference-authority", required=True)
    parser.add_argument("--renderer-geometry-checkpoint", required=True)
    parser.add_argument("--output-cache", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--output-authority", required=True)
    return parser


def main() -> None:
    print(json.dumps(build(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
