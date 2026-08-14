#!/usr/bin/env python3
"""Run the audited LUDVIG DINO/PCA encoder on one UQIS field bridge.

Unlike the legacy PFPR wrapper, this entry point takes the hash-bound UQIS
source-adapter ledger explicitly.  It does not consult the single-scene PFPR
ledger registry and cannot silently substitute that older authority.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_b import PhaseBConfig, run_phase_b


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--phase-a-dir", type=Path, required=True)
    result.add_argument("--phase-a-manifest-sha256", required=True)
    result.add_argument("--source-adapter-ledger", type=Path, required=True)
    result.add_argument("--source-adapter-ledger-sha256", required=True)
    result.add_argument("--dino-checkpoint", type=Path, required=True)
    result.add_argument(
        "--ludvig-upstream", type=Path, default=Path("/root/baselines/LUDVIG")
    )
    result.add_argument(
        "--dinov2-source", type=Path, default=Path("/root/baselines/LUDVIG")
    )
    result.add_argument(
        "--driver-library-dir",
        type=Path,
        default=Path("/root/baselines/LUDVIG/.driver535"),
    )
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--views", type=int, default=120)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def run(args: argparse.Namespace, *, argv=None):
    return run_phase_b(
        PhaseBConfig(
            phase_a_dir=args.phase_a_dir,
            expected_phase_a_manifest_sha256=args.phase_a_manifest_sha256,
            dino_checkpoint=args.dino_checkpoint,
            ludvig_upstream=args.ludvig_upstream,
            source_adapter_ledger=args.source_adapter_ledger,
            dinov2_source=args.dinov2_source,
            output_dir=args.output_dir,
            driver_library_dir=args.driver_library_dir,
            device=args.device,
            view_count=args.views,
            expected_source_adapter_ledger_sha256=(
                args.source_adapter_ledger_sha256
            ),
        ),
        argv=argv,
    )


def main() -> None:
    args = parser().parse_args()
    print(json.dumps(run(args, argv=sys.argv), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
