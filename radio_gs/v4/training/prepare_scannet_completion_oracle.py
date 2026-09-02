"""Prepare hash-bound ScanNet scenes for a supervised completion oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from radio_gs.v4.completion.scannet import (
    MASK_DROPOUT_KEEP_PROBABILITY,
    RADIO_PROJECTION_SHA256,
    prepare_scene,
)


def run(args: argparse.Namespace) -> dict:
    if not args.allow_instance_oracle_training:
        raise PermissionError("preparation requires explicit supervised instance-oracle authorization")
    root = Path(args.scene_root).resolve(strict=True)
    radio_feature_root = Path(args.radio_feature_root).resolve(strict=True)
    scene_ids = list(dict.fromkeys(args.scene_id or []))
    if not scene_ids:
        scene_ids = sorted(
            path.parents[1].name
            for path in root.glob("*/instance_annotations/*.aggregation.json")
        )
    if not scene_ids:
        raise FileNotFoundError("no ScanNet scenes with complete instance annotations were found")
    output_root = Path(args.output_root).resolve()
    records = []
    for scene_id in scene_ids:
        records.append(prepare_scene(
            root, radio_feature_root, scene_id, output=output_root / f"{scene_id}.pt",
            voxel_size=args.voxel_size,
            maximum_splat_radius=args.maximum_splat_radius,
            surface_band_voxels=args.surface_band_voxels,
            maximum_contributors_per_pixel=args.maximum_contributors_per_pixel,
            observation_view_count=args.observation_view_count,
            heldout_view_count=args.heldout_view_count,
            feature_height=args.feature_height,
            feature_width=args.feature_width,
            minimum_observed_elements=args.minimum_observed_elements,
            minimum_total_elements=args.minimum_total_elements,
            minimum_voxel_instance_purity=args.minimum_voxel_instance_purity,
        ))
    manifest = {
        "schema": "radio_gs.surface_object_memory_v4.scannet_completion_cache_manifest.v4",
        "supervision": "complete_3d_instance_membership",
        "oracle_identity_diagnostic_only": True,
        "source_features": "rgb_plus_fixed_jl64_radio",
        "radio_projection_sha256": RADIO_PROJECTION_SHA256,
        "source_object_view_mask_keep_probability": MASK_DROPOUT_KEEP_PROBABILITY,
        "scene_count": len(records),
        "records": records,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-root", required=True)
    parser.add_argument("--radio-feature-root", required=True)
    parser.add_argument("--scene-id", action="append", default=[])
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--voxel-size", type=float, default=0.04)
    parser.add_argument("--maximum-splat-radius", type=int, required=True)
    parser.add_argument("--surface-band-voxels", type=float, required=True)
    parser.add_argument("--maximum-contributors-per-pixel", type=int, required=True)
    parser.add_argument("--observation-view-count", type=int, default=16)
    parser.add_argument("--heldout-view-count", type=int, default=16)
    parser.add_argument("--feature-height", type=int, default=60)
    parser.add_argument("--feature-width", type=int, default=81)
    parser.add_argument("--minimum-observed-elements", type=int, default=8)
    parser.add_argument("--minimum-total-elements", type=int, default=24)
    parser.add_argument("--minimum-voxel-instance-purity", type=float, default=0.8)
    parser.add_argument("--allow-instance-oracle-training", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"scene_count": report["scene_count"], "records": report["records"]}, indent=2))


if __name__ == "__main__":
    main()
