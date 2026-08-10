#!/usr/bin/env python3
"""Run one preregistered frozen FIX6 O0 anchor self-completion metric."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from radio_gs.interfaces import lerf_o0_anchor_self_completion as formal
from radio_gs.scripts import build_lerf_o0_anchor_self_completion_cache as builder
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
)


def validate_authority(path: str | Path, digest: str) -> dict[str, Any]:
    raw, actual, source = load_json_object(
        path, expected_sha256=digest, label="FIX6 metric authority"
    )
    value = dict(raw)
    required = {
        "schema",
        "schema_version",
        "status",
        "scene_id",
        "implementation",
        "interface",
        "launcher",
        "cache_execution_authority",
        "external_query_score_cache",
        "cache_report",
        "frozen_evaluator",
        "frozen_summary_head",
        "config",
        "renderer_geometry_checkpoint",
        "all_query_text_cache",
        "canonical_negative_text_cache",
        "label_root",
        "output_dir",
        "protocol",
        "single_candidate_no_sweep",
        "metric_execution_authorized",
        "access_audit",
    }
    if (
        set(value) != required
        or value["schema"] != builder.METRIC_AUTHORITY_SCHEMA
        or value["schema_version"] != 2
        or value["status"]
        != "preregistered_single_FIX6_anchor_self_completion_metric"
        or value["protocol"] != builder.METRIC_PROTOCOL
        or value["single_candidate_no_sweep"] is not True
        or value["metric_execution_authorized"] is not True
        or value["implementation"] != file_record(builder.IMPLEMENTATION)
        or value["interface"] != file_record(builder.INTERFACE)
        or value["launcher"] != file_record(builder.LAUNCHER)
    ):
        raise ValueError("FIX6 metric authority differs")
    for name in (
        "implementation",
        "interface",
        "launcher",
        "cache_execution_authority",
        "external_query_score_cache",
        "cache_report",
        "frozen_evaluator",
        "frozen_summary_head",
        "config",
        "renderer_geometry_checkpoint",
        "all_query_text_cache",
        "canonical_negative_text_cache",
    ):
        validate_file_record(value[name], label=f"FIX6 metric {name}")
    cache, _, _ = load_torch_mapping(
        value["external_query_score_cache"]["path"],
        expected_sha256=value["external_query_score_cache"]["sha256"],
        map_location="cpu",
        label="FIX6 external cache",
    )
    formal.validate_external_query_score_cache(cache)
    value["verified_record"] = {"path": str(source), "sha256": actual}
    return value


def build_command(authority: dict[str, Any], *, gpu: int) -> list[str]:
    if int(gpu) < 0:
        raise ValueError("gpu must be non-negative")
    return [
        sys.executable,
        authority["frozen_evaluator"]["path"],
        "--config",
        authority["config"]["path"],
        "--checkpoint",
        authority["renderer_geometry_checkpoint"]["path"],
        "--scene",
        authority["scene_id"],
        "--protocol_preset",
        "vala_paper_3d",
        "--label_dir",
        authority["label_root"],
        "--output_dir",
        authority["output_dir"],
        "--summary_head_weights",
        authority["frozen_summary_head"]["path"],
        "--text_embedding_cache",
        authority["all_query_text_cache"]["path"],
        "--canonical_embedding_cache",
        authority["canonical_negative_text_cache"]["path"],
        "--external_query_score_cache",
        authority["external_query_score_cache"]["path"],
        "--gpu",
        str(int(gpu)),
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    authority = validate_authority(
        args.execution_authority, args.expected_execution_authority_sha256
    )
    command = build_command(authority, gpu=args.gpu)
    if not args.execute:
        return {
            "status": "FIX6_metric_dry_run",
            "execution_authority": authority["verified_record"],
            "command": command,
        }
    output = Path(authority["output_dir"])
    if output.exists() or output.is_symlink():
        raise FileExistsError("FIX6 metric output must be new")
    subprocess.run(command, check=True)
    result = output / authority["scene_id"] / "lerf_direct_3d_selection_results.json"
    if not result.is_file() or result.is_symlink():
        raise RuntimeError("FIX6 metric result is unavailable")
    return {
        "status": "FIX6_metric_complete",
        "execution_authority": authority["verified_record"],
        "result": file_record(result),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-authority", required=True)
    parser.add_argument("--expected-execution-authority-sha256", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), indent=2, allow_nan=False))
