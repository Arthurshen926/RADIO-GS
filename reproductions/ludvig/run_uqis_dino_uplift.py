#!/usr/bin/env python3
"""Build one authority-bound UQIS LUDVIG DINO field."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from radio_gs.benchmarks.scannet_uqis.ludvig_dino_uplift import UpliftConfig, run_uplift


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--phase-b-dir", type=Path, required=True)
    result.add_argument("--phase-b-manifest-sha256", required=True)
    result.add_argument("--ludvig-upstream", type=Path, default=Path("/root/baselines/LUDVIG"))
    result.add_argument("--driver-library-dir", type=Path, default=Path("/root/baselines/LUDVIG/.driver535"))
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    payload = run_uplift(
        UpliftConfig(
            phase_b_dir=args.phase_b_dir,
            expected_phase_b_manifest_sha256=args.phase_b_manifest_sha256,
            ludvig_upstream=args.ludvig_upstream,
            driver_library_dir=args.driver_library_dir,
            device=args.device,
            output_dir=args.output_dir,
        ),
        argv=sys.argv,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
