#!/usr/bin/env python3
"""Run one isolated UQIS LUDVIG/LERF text query."""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from radio_gs.benchmarks.scannet_uqis.ludvig_text_adapter import TextConfig, run_text
from radio_gs.benchmarks.scannet_uqis.ludvig_text_diffusion import TextDiffusionConfig

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--workspace-receipt", type=Path, required=True)
    parser.add_argument("--field-dir", type=Path, required=True)
    parser.add_argument("--field-manifest-sha256", required=True)
    parser.add_argument("--dino-field-dir", type=Path)
    parser.add_argument("--dino-field-manifest-sha256")
    parser.add_argument("--diffusion-neighbors", type=int, default=64)
    parser.add_argument("--diffusion-iterations", type=int, default=20)
    parser.add_argument("--diffusion-feature-bandwidth", type=float, default=0.5)
    parser.add_argument("--diffusion-regularizer-bandwidth", type=float, default=2.0)
    parser.add_argument("--diffusion-seed-quantile", type=float, default=0.999)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ludvig-upstream", type=Path, default=Path("/root/baselines/LUDVIG"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    diffusion = TextDiffusionConfig(
        neighbors=args.diffusion_neighbors,
        iterations=args.diffusion_iterations,
        feature_bandwidth=args.diffusion_feature_bandwidth,
        regularizer_bandwidth=args.diffusion_regularizer_bandwidth,
        seed_quantile=args.diffusion_seed_quantile,
    )
    result = run_text(
        TextConfig(
            query_manifest_path=args.query_manifest,
            workspace_receipt_path=args.workspace_receipt,
            field_dir=args.field_dir,
            expected_field_manifest_sha256=args.field_manifest_sha256,
            ludvig_upstream=args.ludvig_upstream,
            output_dir=args.output_dir,
            dino_field_dir=args.dino_field_dir,
            expected_dino_field_manifest_sha256=args.dino_field_manifest_sha256,
            diffusion=diffusion,
            device=args.device,
        ),
        argv=sys.argv,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
if __name__ == "__main__": main()
