#!/usr/bin/env python3
"""Launch one frozen native-V3 LERF metric candidate, or print its dry run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from radio_gs.scripts import (
    build_lerf_v21_native_v3_frozen_metric_bridge as bridge,
)
from radio_gs.utils.immutable_artifacts import file_record


def build_command(
    authority: dict[str, Any], *, gpu: int
) -> list[str]:
    if gpu < 0:
        raise ValueError("gpu must be non-negative")
    protocol = authority["protocol"]
    if (
        protocol["protocol_preset"] != "vala_paper_3d"
        or protocol["score_threshold"] != 0.6
        or protocol["projection_mode"] != "selected_only_alpha"
    ):
        raise ValueError("launcher protocol differs")
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
        str(gpu),
    ]


def launch(args: argparse.Namespace) -> dict[str, Any]:
    authority = bridge.validate_metric_authority(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
    )
    command = build_command(authority, gpu=args.gpu)
    if not args.execute:
        return {
            "status": "native_v3_frozen_metric_synthetic_dry_run",
            "execution_authority": authority["verified_record"],
            "command": command,
            "protocol": authority["protocol"],
            "access_audit": {
                "label_root_opened": False,
                "benchmark_labels_opened": False,
                "target_metrics_computed": False,
                "subprocess_started": False,
            },
        }
    output = Path(authority["output_dir"])
    if output.exists() or output.is_symlink():
        raise FileExistsError("frozen metric output directory must be new")
    subprocess.run(command, check=True)
    result = (
        output
        / authority["scene_id"]
        / "lerf_direct_3d_selection_results.json"
    )
    if not result.is_file() or result.is_symlink():
        raise RuntimeError("frozen LERF evaluator did not produce its exact result")
    return {
        "status": "native_v3_frozen_metric_complete",
        "execution_authority": authority["verified_record"],
        "result": file_record(result),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-authority", required=True)
    parser.add_argument("--expected-execution-authority-sha256", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the one frozen candidate; omitted means synthetic dry-run",
    )
    return parser


def main() -> None:
    print(json.dumps(launch(build_parser().parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = ["build_command", "build_parser", "launch"]
