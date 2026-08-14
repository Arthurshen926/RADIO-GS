#!/usr/bin/env python3
"""Build a non-evaluable UQIS image workspace for exact-runtime smoke.

This utility exists only to exercise the LUDVIG image adapter while the final
UQIS targets and frame-exclusion receipts are unavailable.  It uses an
official ScanNet mesh domain and a real crop, but it deliberately emits no GT,
pairing, metric, or formal-release claim.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image

from radio_gs.benchmarks.scannet_pfir.protocol import load_mesh_instances
from radio_gs.benchmarks.scannet_uqis.protocol import (
    BENCHMARK_VERSION,
    PREDICTION_DOMAIN,
    UQISProtocolConfig,
    canonical_json_sha256,
    sha256_file,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build(args: argparse.Namespace) -> dict:
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite smoke fixture: {output}")
    scene_id = str(args.scene)
    if not scene_id.startswith("scene"):
        raise ValueError("--scene must be a ScanNet scene ID")
    source_crop = Path(args.crop_rgb).resolve()
    if not source_crop.is_file():
        raise FileNotFoundError(source_crop)
    query_material = {
        "purpose": "ludvig_uqis_exact_runtime_image_smoke",
        "scene_id": scene_id,
        "crop_sha256": sha256_file(source_crop),
    }
    query_id = "uq_" + canonical_json_sha256(query_material)[:32]
    config = UQISProtocolConfig()

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp_", dir=output.parent)
    )
    try:
        assets = temporary / "assets"
        assets.mkdir()
        xyz, _instance_ids, _metadata = load_mesh_instances(
            args.mesh_ply, args.aggregation, args.segmentation
        )
        mesh_path = assets / "official_mesh_xyz.npy"
        np.save(mesh_path, np.asarray(xyz, dtype=np.float32), allow_pickle=False)
        crop_path = assets / "query.png"
        with Image.open(source_crop) as image:
            rgb = image.convert("RGB")
            resampling = getattr(Image, "Resampling", Image).BICUBIC
            rgb.resize((config.crop_size_px, config.crop_size_px), resampling).save(
                crop_path,
                format="PNG",
                optimize=False,
                compress_level=9,
            )
        common = {
            "benchmark_version": BENCHMARK_VERSION,
            "split_role": "pilot",
            "release_tier": "pilot_harness",
            "formal_benchmark_eligible": False,
            "protocol_config": asdict(config),
            "protocol_config_sha256": canonical_json_sha256(asdict(config)),
            "query_id_salt_sha256": hashlib.sha256(
                b"ludvig-uqis-exact-runtime-smoke-only"
            ).hexdigest(),
        }
        image_manifest = {
            **common,
            "visibility": "method_input",
            "modality": "image",
            "prediction_domain": PREDICTION_DOMAIN,
            "scene_domains": [
                {
                    "scene_id": scene_id,
                    "mesh_xyz_path": str(output / "assets" / mesh_path.name),
                    "mesh_xyz_sha256": sha256_file(mesh_path),
                    "mesh_vertices": int(len(xyz)),
                }
            ],
            "queries": [
                {
                    "query_id": query_id,
                    "scene_id": scene_id,
                    "modality": "image",
                    "crop_rgb_path": str(output / "assets" / crop_path.name),
                    "crop_rgb_sha256": sha256_file(crop_path),
                    "available_method_inputs": ["scene_id", "crop_rgb"],
                }
            ],
        }
        query_manifest_path = temporary / "query_manifest.image.json"
        _write_json(query_manifest_path, image_manifest)
        fixture = {
            "schema_version": "ludvig_uqis_image_smoke_fixture_v1",
            "status": "method_workspace_ready",
            "benchmark_version": BENCHMARK_VERSION,
            "formal_benchmark_eligible": False,
            "evaluator_metrics_allowed": False,
            "uqis_construction_compliant": False,
            "reason": (
                "final UQIS target/query-frame exclusion receipts are unavailable; "
                "the real legacy crop is resized only to test exact runtime plumbing"
            ),
            "scene_id": scene_id,
            "query_id": query_id,
            "query_manifest": {
                "path": "query_manifest.image.json",
                "sha256": sha256_file(query_manifest_path),
            },
            "source_assets": {
                "mesh_ply_sha256": sha256_file(args.mesh_ply),
                "aggregation_sha256": sha256_file(args.aggregation),
                "segmentation_sha256": sha256_file(args.segmentation),
                "real_crop_sha256": sha256_file(source_crop),
            },
            "method_visible_inputs_only": True,
            "evaluator_private_manifest_emitted": False,
        }
        _write_json(temporary / "smoke_fixture_manifest.json", fixture)
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return fixture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--mesh-ply", type=Path, required=True)
    parser.add_argument("--aggregation", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--crop-rgb", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
