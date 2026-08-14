#!/usr/bin/env python3
"""Stage or train authority-bound LUDVIG Gaussian geometry for UQIS."""

from __future__ import annotations

import argparse
import json

from radio_gs.benchmarks.scannet_uqis.ludvig_geometry import (
    run_ludvig_geometry_training,
    stage_ludvig_geometry_scene,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    stage = sub.add_parser("stage")
    stage.add_argument("--observation-receipt", required=True)
    stage.add_argument("--expected-observation-receipt-sha256", required=True)
    stage.add_argument("--construction-authority", required=True)
    stage.add_argument("--expected-construction-authority-sha256", required=True)
    stage.add_argument("--scene-id", required=True)
    stage.add_argument("--output-dir", required=True)
    train = sub.add_parser("train")
    train.add_argument("--staging-dir", required=True)
    train.add_argument("--expected-staging-receipt-sha256", required=True)
    train.add_argument("--ludvig-upstream", required=True)
    train.add_argument("--python", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--iterations", type=int, required=True)
    train.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()
    if args.command == "stage":
        result = stage_ludvig_geometry_scene(
            args.observation_receipt,
            expected_observation_receipt_sha256=args.expected_observation_receipt_sha256,
            construction_authority_path=args.construction_authority,
            expected_construction_authority_sha256=args.expected_construction_authority_sha256,
            scene_id=args.scene_id,
            output_dir=args.output_dir,
        )
    else:
        result = run_ludvig_geometry_training(
            args.staging_dir,
            expected_staging_receipt_sha256=args.expected_staging_receipt_sha256,
            ludvig_upstream=args.ludvig_upstream,
            python=args.python,
            output_dir=args.output_dir,
            iterations=args.iterations,
            device_index=args.device_index,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
