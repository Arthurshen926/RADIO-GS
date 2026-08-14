#!/usr/bin/env python3
"""Stage one completed UQIS geometry for audited LUDVIG DINO Phase B/C."""

from __future__ import annotations

import argparse
import json

from radio_gs.benchmarks.scannet_uqis.ludvig_dino_field import (
    stage_uqis_ludvig_dino_bridge,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation-receipt", required=True)
    parser.add_argument("--expected-observation-receipt-sha256", required=True)
    parser.add_argument("--construction-authority", required=True)
    parser.add_argument("--expected-construction-authority-sha256", required=True)
    parser.add_argument("--geometry-run-receipt", required=True)
    parser.add_argument("--expected-geometry-run-receipt-sha256", required=True)
    parser.add_argument("--dino-checkpoint", required=True)
    parser.add_argument("--ludvig-upstream", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = stage_uqis_ludvig_dino_bridge(
        args.observation_receipt,
        expected_observation_receipt_sha256=args.expected_observation_receipt_sha256,
        construction_authority_path=args.construction_authority,
        expected_construction_authority_sha256=args.expected_construction_authority_sha256,
        geometry_run_receipt_path=args.geometry_run_receipt,
        expected_geometry_run_receipt_sha256=args.expected_geometry_run_receipt_sha256,
        dino_checkpoint=args.dino_checkpoint,
        ludvig_upstream=args.ludvig_upstream,
        output_dir=args.output_dir,
    )
    print(json.dumps({
        "scene_id": result["scene_id"],
        "gaussians": result["geometry"]["gaussians"],
        "view_count": result["view_selection"]["count"],
        "run_manifest_sha256": result["run_manifest_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
