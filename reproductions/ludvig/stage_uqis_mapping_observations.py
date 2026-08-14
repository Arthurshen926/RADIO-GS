#!/usr/bin/env python3
"""Stage authority-legal UQIS mapping RGB/poses for LUDVIG fields."""

from __future__ import annotations

import argparse
import json

from radio_gs.benchmarks.scannet_uqis.ludvig_mapping_stage import (
    stage_ludvig_mapping_observations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-plan", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = stage_ludvig_mapping_observations(
        args.mapping_plan,
        args.output_dir,
        expected_plan_sha256=args.expected_plan_sha256,
    )
    print(json.dumps({
        "scene_count": result["scene_count"],
        "frame_count": result["frame_count"],
        "receipt_sha256": result["receipt_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
