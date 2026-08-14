#!/usr/bin/env python3
"""Build one authority-bound unpruned LUDVIG OpenCLIP text field."""

from __future__ import annotations

import argparse
import json
import sys

from radio_gs.benchmarks.scannet_uqis.ludvig_clip_field import run_clip_field


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-a-dir", required=True)
    parser.add_argument("--phase-a-manifest-sha256", required=True)
    parser.add_argument("--ludvig-upstream", required=True)
    parser.add_argument("--open-clip-site-packages", required=True)
    parser.add_argument("--open-clip-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    payload = run_clip_field(
        args.phase_a_dir, expected_phase_a_manifest_sha256=args.phase_a_manifest_sha256,
        ludvig_upstream=args.ludvig_upstream,
        open_clip_site_packages=args.open_clip_site_packages,
        open_clip_checkpoint=args.open_clip_checkpoint, output_dir=args.output_dir,
        device=args.device, argv=sys.argv,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
