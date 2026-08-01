#!/usr/bin/env python3
"""Run public-query PFPR Phase D over exact LUDVIG uplifted features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_d import PhaseDConfig, run_phase_d


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-c-dir", type=Path, required=True)
    parser.add_argument("--phase-c-manifest-sha256", required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--scene", default="scene0050_02")
    parser.add_argument("--ludvig-upstream", type=Path, default=Path("/root/baselines/LUDVIG"))
    parser.add_argument("--driver-library-dir", type=Path, default=Path("/root/baselines/LUDVIG/.driver535"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_phase_d(
        PhaseDConfig(
            phase_c_dir=args.phase_c_dir,
            expected_phase_c_manifest_sha256=args.phase_c_manifest_sha256,
            benchmark_dir=args.benchmark_dir,
            ludvig_upstream=args.ludvig_upstream,
            output_dir=args.output_dir,
            driver_library_dir=args.driver_library_dir,
            device=args.device,
            scene_id=args.scene,
        ),
        argv=sys.argv,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
