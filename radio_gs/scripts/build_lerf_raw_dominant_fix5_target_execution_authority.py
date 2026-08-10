#!/usr/bin/env python3
"""Build a scene-generic FIX5 target score-cache execution authority.

The builder binds an existing no-quality FIX4C execution/cache/report to the
promoted source-only FIX5 raw-dominant audit and chooses new no-clobber cache,
report, and authority paths.  It opens no target image, GT, mask, or metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from radio_gs.interfaces import lerf_raw_unary_region_specificity as unary
from radio_gs.scripts import audit_source_only_graph_raw_dominant_fix5 as source_fix5
from radio_gs.scripts import build_lerf_o0_anchored_positive_utility_residual_cache as fix4b
from radio_gs.scripts import build_lerf_o0_anchored_positive_utility_residual_cache_fix4c as fix4c
from radio_gs.scripts import build_lerf_o0_anchored_raw_dominant_positive_utility_cache_fix5 as fix5
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    write_frozen_json,
)


IMPLEMENTATION = Path(__file__).resolve()


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


def _expected_record(
    path_value: object, expected_sha256: object, *, label: str
) -> dict[str, str]:
    path = _canonical_existing(path_value, label=label)
    record = file_record(path)
    if record["sha256"] != str(expected_sha256):
        raise ValueError(f"{label} SHA256 differs")
    return record


def _compose_authority(
    *,
    parent_execution: Mapping[str, str],
    parent_cache: Mapping[str, str],
    parent_report: Mapping[str, str],
    source_execution: Mapping[str, str],
    source_result: Mapping[str, str],
    output_cache: Path,
    output_report: Path,
) -> dict[str, Any]:
    return {
        "schema": fix5.EXECUTION_SCHEMA,
        "schema_version": 1,
        "status": fix5.EXECUTION_STATUS,
        "implementation": file_record(fix5.IMPLEMENTATION),
        "dependencies": {
            name: file_record(path) for name, path in fix5.DEPENDENCIES.items()
        },
        "source_fix5_execution_authority": dict(source_execution),
        "source_fix5_result": dict(source_result),
        "parent_fix4c_execution_authority": dict(parent_execution),
        "parent_fix4c_cache": dict(parent_cache),
        "parent_fix4c_report": dict(parent_report),
        "fixed_intervention": {
            "raw_input": "canonical_negative_probability_before_VALA_minmax_at_each_query_frozen_O0_scale",
            "region_statistic": "valid_core_mean_raw_probability_argmax_all_exact_ties_retained",
            "anchor_order": "filter_O0_anchor_before_direct_graph_support_propagation",
            "candidate_gate": "existing_graph_candidate_and_raw_dominant_query",
            "primitive_majority_threshold": None,
            "residual_and_selection": "bitwise_frozen_FIX4B",
            "probability_fusion": "endpoint_safe_monotone_FIX4C",
        },
        "output_cache": str(output_cache),
        "output_report": str(output_report),
        "target_score_cache_authorized": True,
        "target_quality_execution_authorized": False,
        "access_audit": {
            "query_names_opened": True,
            "raw_query_score_cache_opened": True,
            "target_images_opened": False,
            "target_quality_data_opened": False,
            "target_quality_readout_executed": False,
        },
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    authority_output = _canonical_new(
        args.output_authority, label="FIX5 execution authority output"
    )
    output_cache = _canonical_new(args.output_cache, label="FIX5 score cache output")
    output_report = _canonical_new(args.output_report, label="FIX5 report output")
    if len({authority_output, output_cache, output_report}) != 3:
        raise ValueError("FIX5 authority, cache, and report outputs must differ")

    parent_execution = _expected_record(
        args.parent_fix4c_execution_authority,
        args.expected_parent_fix4c_execution_authority_sha256,
        label="parent FIX4C execution authority",
    )
    parent_cache = _expected_record(
        args.parent_fix4c_cache,
        args.expected_parent_fix4c_cache_sha256,
        label="parent FIX4C cache",
    )
    parent_report = _expected_record(
        args.parent_fix4c_report,
        args.expected_parent_fix4c_report_sha256,
        label="parent FIX4C report",
    )
    parent = fix4c._load_and_validate_execution(
        parent_execution["path"], expected_sha256=parent_execution["sha256"]
    )
    if (
        parent["output_cache"] != parent_cache["path"]
        or parent["output_report"] != parent_report["path"]
    ):
        raise ValueError("provided FIX4C cache/report do not belong to parent authority")

    source_execution = _expected_record(
        args.source_fix5_execution_authority,
        args.expected_source_fix5_execution_authority_sha256,
        label="source FIX5 execution authority",
    )
    source_raw, _, _ = load_json_object(
        source_execution["path"],
        expected_sha256=source_execution["sha256"],
        label="source FIX5 execution authority",
    )
    source_fix5.validate_execution_authority(source_raw)
    source_result = _expected_record(
        args.source_fix5_result,
        args.expected_source_fix5_result_sha256,
        label="source FIX5 result",
    )
    result_raw, _, _ = load_json_object(
        source_result["path"],
        expected_sha256=source_result["sha256"],
        label="source FIX5 result",
    )
    if (
        result_raw.get("status")
        != "source_only_raw_dominant_FIX5_promoted_target_unopened"
        or result_raw.get("execution_authority") != source_execution
        or result_raw.get("raw_unary_contract_sha256") != unary.CONTRACT_SHA256
        or result_raw.get("promotion_gate", {}).get("outcomes", {}).get("passed")
        is not True
        or result_raw.get("target_execution_performed") is not False
    ):
        raise ValueError("source FIX5 result is not the promoted target-unopened chain")

    parent_paths = {
        Path(parent_execution["path"]),
        Path(parent_cache["path"]),
        Path(parent_report["path"]),
        Path(source_execution["path"]),
        Path(source_result["path"]),
    }
    if any(path in parent_paths for path in (authority_output, output_cache, output_report)):
        raise ValueError("FIX5 output collides with a frozen parent/source input")
    authority = _compose_authority(
        parent_execution=parent_execution,
        parent_cache=parent_cache,
        parent_report=parent_report,
        source_execution=source_execution,
        source_result=source_result,
        output_cache=output_cache,
        output_report=output_report,
    )
    written = write_frozen_json(authority_output, authority)
    authority_record = file_record(written)
    # Full consumer-side validation catches schema/dependency drift immediately.
    validated = fix5._load_and_validate_execution(
        authority_record["path"], expected_sha256=authority_record["sha256"]
    )
    if (
        validated["output_cache"] != str(output_cache)
        or validated["output_report"] != str(output_report)
        or validated["source_fix5_result"] != source_result
    ):
        raise RuntimeError("generated FIX5 target authority replay differs")
    return {
        "status": "scene_generic_FIX5_target_execution_authority_complete",
        "authority": authority_record,
        "output_cache": str(output_cache),
        "output_report": str(output_report),
        "target_quality_execution_authorized": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-fix4c-execution-authority", required=True)
    parser.add_argument(
        "--expected-parent-fix4c-execution-authority-sha256", required=True
    )
    parser.add_argument("--parent-fix4c-cache", required=True)
    parser.add_argument("--expected-parent-fix4c-cache-sha256", required=True)
    parser.add_argument("--parent-fix4c-report", required=True)
    parser.add_argument("--expected-parent-fix4c-report-sha256", required=True)
    parser.add_argument("--source-fix5-execution-authority", required=True)
    parser.add_argument(
        "--expected-source-fix5-execution-authority-sha256", required=True
    )
    parser.add_argument("--source-fix5-result", required=True)
    parser.add_argument("--expected-source-fix5-result-sha256", required=True)
    parser.add_argument("--output-cache", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--output-authority", required=True)
    return parser


def main() -> None:
    print(json.dumps(build(build_parser().parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
