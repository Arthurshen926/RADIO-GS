#!/usr/bin/env python3
"""Run one UQIS LUDVIG world-point query in its isolated workspace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from radio_gs.benchmarks.scannet_uqis.ludvig_point3d_adapter import Point3DConfig, run_point3d


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--workspace-receipt", type=Path, required=True)
    parser.add_argument("--field-dir", type=Path, required=True)
    parser.add_argument("--field-manifest-sha256", required=True)
    parser.add_argument("--ludvig-upstream", type=Path, default=Path("/root/baselines/LUDVIG"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = run_point3d(Point3DConfig(
        query_manifest_path=args.query_manifest,
        workspace_receipt_path=args.workspace_receipt,
        field_dir=args.field_dir,
        expected_field_manifest_sha256=args.field_manifest_sha256,
        ludvig_upstream=args.ludvig_upstream,
        output_dir=args.output_dir,
    ), argv=sys.argv)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
