#!/usr/bin/env python3
"""Run hash-bound PFPR Phase C with the released LUDVIG rasterizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_c import PhaseCConfig, run_phase_c


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-b-dir", type=Path, required=True)
    parser.add_argument("--phase-b-manifest-sha256", required=True)
    parser.add_argument("--ludvig-upstream", type=Path, default=Path("/root/baselines/LUDVIG"))
    parser.add_argument("--driver-library-dir", type=Path, default=Path("/root/baselines/LUDVIG/.driver535"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_phase_c(
        PhaseCConfig(
            phase_b_dir=args.phase_b_dir,
            expected_phase_b_manifest_sha256=args.phase_b_manifest_sha256,
            ludvig_upstream=args.ludvig_upstream,
            output_dir=args.output_dir,
            driver_library_dir=args.driver_library_dir,
            device=args.device,
        ),
        argv=sys.argv,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
