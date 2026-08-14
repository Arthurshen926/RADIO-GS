#!/usr/bin/env python3
"""Freeze authority-bound CLIP and DINO mapping jobs for UQIS-9."""

from __future__ import annotations

import argparse
import json

from radio_gs.benchmarks.scannet_uqis.ludvig_mapping_plan import (
    build_ludvig_mapping_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--construction-authority", required=True)
    parser.add_argument("--ludvig-upstream", required=True)
    parser.add_argument("--dino-checkpoint", required=True)
    parser.add_argument("--openclip-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build_ludvig_mapping_plan(
        args.construction_authority,
        ludvig_upstream=args.ludvig_upstream,
        dino_checkpoint=args.dino_checkpoint,
        openclip_checkpoint=args.openclip_checkpoint,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "scene_count": result["scene_count"],
                "field_count": result["field_count"],
                "method_identity_sha256": result["method_identity_sha256"],
                "plan_sha256": result["plan_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

