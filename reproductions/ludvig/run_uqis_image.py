#!/usr/bin/env python3
"""Run the benchmark-local LUDVIG adapter for UQIS image queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from radio_gs.benchmarks.scannet_uqis.ludvig_image_adapter import (
    DEFAULT_SIGMOID_CALIBRATION,
    ExactLudvigImageAdapterConfig,
    load_frozen_sigmoid_calibration,
    run_exact_ludvig_image_adapter,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--query-manifest",
        type=Path,
        required=True,
        help=(
            "One-query workspace query_manifest.json "
            "(the only query/target input)"
        ),
    )
    result.add_argument(
        "--workspace-receipt",
        type=Path,
        required=True,
        help="Receipt binding the fresh one-query method workspace",
    )
    result.add_argument("--phase-b-dir", type=Path, required=True)
    result.add_argument("--phase-b-manifest-sha256", required=True)
    result.add_argument("--phase-c-dir", type=Path, required=True)
    result.add_argument("--phase-c-manifest-sha256", required=True)
    result.add_argument(
        "--ludvig-upstream",
        type=Path,
        default=Path("/root/baselines/LUDVIG"),
    )
    result.add_argument(
        "--driver-library-dir",
        type=Path,
        default=Path("/root/baselines/LUDVIG/.driver535"),
    )
    result.add_argument("--device", default="cuda:0")
    result.add_argument(
        "--calibration",
        type=Path,
        help=(
            "Optional immutable global sigmoid artifact fitted on dev; "
            "omitting it uses frozen scale=1,bias=0"
        ),
    )
    result.add_argument(
        "--chunk-size",
        type=int,
        default=65_536,
        help="Implementation-only mesh chunk size; scoring constants stay frozen",
    )
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def run(args: argparse.Namespace, *, argv: list[str] | None = None):
    calibration = (
        load_frozen_sigmoid_calibration(args.calibration)
        if args.calibration is not None
        else DEFAULT_SIGMOID_CALIBRATION
    )
    return run_exact_ludvig_image_adapter(
        ExactLudvigImageAdapterConfig(
            query_manifest_path=args.query_manifest,
            workspace_receipt_path=args.workspace_receipt,
            phase_b_dir=args.phase_b_dir,
            expected_phase_b_manifest_sha256=args.phase_b_manifest_sha256,
            phase_c_dir=args.phase_c_dir,
            expected_phase_c_manifest_sha256=args.phase_c_manifest_sha256,
            ludvig_upstream=args.ludvig_upstream,
            output_dir=args.output_dir,
            driver_library_dir=args.driver_library_dir,
            device=args.device,
            calibration=calibration,
            chunk_size=args.chunk_size,
        ),
        argv=tuple(argv or ()),
    )


def main() -> None:
    args = parser().parse_args()
    payload = run(args, argv=sys.argv)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
