#!/usr/bin/env python3
"""Run the isolated evaluator-only Phase E for LUDVIG-on-PFPR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_e import PhaseEConfig, run_phase_e


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-d-dir", type=Path, required=True)
    parser.add_argument("--phase-d-manifest-sha256", required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--scene", default="scene0050_02")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_phase_e(
        PhaseEConfig(
            phase_d_dir=args.phase_d_dir,
            expected_phase_d_manifest_sha256=args.phase_d_manifest_sha256,
            benchmark_dir=args.benchmark_dir,
            output_dir=args.output_dir,
            scene_id=args.scene,
        ),
        argv=sys.argv,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
