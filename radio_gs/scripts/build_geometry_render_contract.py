#!/usr/bin/env python3
"""Build a query-free renderer contract from a trained RGB 3DGS PLY.

The resulting config/checkpoint exists only so observation lifting can reuse
the repository's audited Gaussian renderer.  Random compact feature rows and
the codec are never read as teacher targets or method descriptors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from radio_gs.models.explicit_gaussian import ExplicitFeatureGaussian
from radio_gs.models.featsharp_3d import FeatSharp3D
from radio_gs.models.hcd_codec import build_feature_codec


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_contract(
    *,
    ply_path: str | Path,
    scene_root: str | Path,
    feature_dir: str | Path,
    output_config: str | Path,
    output_checkpoint: str | Path,
    latent_dim: int = 8,
) -> dict:
    ply_path = Path(ply_path).resolve()
    scene_root = Path(scene_root).resolve()
    feature_dir = Path(feature_dir).resolve()
    output_config = Path(output_config).resolve()
    output_checkpoint = Path(output_checkpoint).resolve()
    intrinsic_path = scene_root / "intrinsic" / "intrinsic_depth.txt"
    if not intrinsic_path.is_file():
        intrinsic_path = scene_root / "intrinsics_depth.txt"
    intrinsic = np.loadtxt(intrinsic_path, dtype=np.float32).reshape(4, 4)

    # PFIR reconstruction inputs are normalized to the depth-camera raster.
    from PIL import Image

    depth_path = next(iter(sorted((scene_root / "depth").glob("*.png"))))
    with Image.open(depth_path) as image:
        image_width, image_height = image.size
    feature_height = image_height // 8
    feature_width = image_width // 8

    torch.manual_seed(0)
    model = ExplicitFeatureGaussian(latent_dim=int(latent_dim))
    model.load_from_ply(str(ply_path))
    codec = build_feature_codec(
        input_dim=1280,
        bottleneck_dim=int(latent_dim),
        codec_type="direct",
        dual_stream=False,
        symmetric_decoder=False,
    )
    sharpener = FeatSharp3D(
        mode="analytical", feature_dim=int(latent_dim), strength=0.0
    )
    provenance = {
        "schema_version": 1,
        "purpose": "query_free_geometry_render_contract",
        "scene_root": str(scene_root),
        "ply_path": str(ply_path),
        "ply_sha256": _sha256(ply_path),
        "feature_dir": str(feature_dir),
        "random_feature_rows_used_by_method": False,
        "codec_used_by_method": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "codec_state_dict": codec.state_dict(),
            "sharpener_state_dict": sharpener.state_dict(),
            "geometry_render_contract": provenance,
        },
        output_checkpoint,
    )
    config = {
        "architecture": "explicit",
        "latent_dim": int(latent_dim),
        "radio_feature_dim": 1280,
        "bottleneck_dim": int(latent_dim),
        "codec_type": "direct",
        "dual_stream": False,
        "symmetric_decoder": False,
        "ply_path": str(ply_path),
        "dataset_type": "scannet",
        "scene": scene_root.name,
        "scene_root": str(scene_root),
        "feature_dir": str(feature_dir),
        "pose_dir": str(scene_root / "pose"),
        "pose_file": "",
        "image_height": int(image_height),
        "image_width": int(image_width),
        "fx": float(intrinsic[0, 0]),
        "fy": float(intrinsic[1, 1]),
        "cx": float(intrinsic[0, 2]),
        "cy": float(intrinsic[1, 2]),
        "feature_height": int(feature_height),
        "feature_width": int(feature_width),
        "max_channels_per_chunk": 32,
        "use_2dgs": False,
        "featsharp_mode": "analytical",
        "featsharp_strength": 0.0,
        "use_refiner": False,
    }
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    report = {
        **provenance,
        "config": str(output_config),
        "checkpoint": str(output_checkpoint),
        "checkpoint_sha256": _sha256(output_checkpoint),
        "num_gaussians": int(model.num_gaussians),
        "image_size_hw": [int(image_height), int(image_width)],
        "feature_size_hw": [int(feature_height), int(feature_width)],
    }
    output_checkpoint.with_suffix(output_checkpoint.suffix + ".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply-path", required=True)
    parser.add_argument("--scene-root", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--latent-dim", type=int, default=8)
    args = parser.parse_args()
    print(
        json.dumps(
            build_contract(
                ply_path=args.ply_path,
                scene_root=args.scene_root,
                feature_dir=args.feature_dir,
                output_config=args.output_config,
                output_checkpoint=args.output_checkpoint,
                latent_dim=args.latent_dim,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
