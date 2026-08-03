#!/usr/bin/env python3
"""Build a query-free Gaussian renderer carrier for promptable-NVS fields.

The carrier imports only RGB 3DGS geometry from a PLY.  Its random latent rows
and direct codec are never used as method descriptors; canonical MPR training
later supplies the sole feature field.  A base promptable config provides the
already-audited camera and raw RADIO feature mapping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

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
    base_config: str | Path,
    ply_path: str | Path,
    output_config: str | Path,
    output_checkpoint: str | Path,
    latent_dim: int = 8,
) -> dict[str, object]:
    base = Path(base_config).resolve()
    ply = Path(ply_path).resolve()
    config = Path(output_config).resolve()
    checkpoint = Path(output_checkpoint).resolve()
    for label, path in (("base config", base), ("geometry PLY", ply)):
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
    if int(latent_dim) <= 0:
        raise ValueError("latent_dim must be positive")

    torch.manual_seed(0)
    model = ExplicitFeatureGaussian(latent_dim=int(latent_dim))
    model.load_from_ply(str(ply))
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
        "purpose": "promptable_nvs_query_free_geometry_render_contract",
        "base_config": str(base),
        "base_config_sha256": _sha256(base),
        "ply_path": str(ply),
        "ply_sha256": _sha256(ply),
        "random_feature_rows_used_by_method": False,
        "codec_used_by_method": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "codec_state_dict": codec.state_dict(),
            "sharpener_state_dict": sharpener.state_dict(),
            "geometry_render_contract": provenance,
        },
        checkpoint,
    )
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        yaml.safe_dump(
            {
                "base_config": str(base),
                "architecture": "explicit",
                "latent_dim": int(latent_dim),
                "bottleneck_dim": int(latent_dim),
                "codec_type": "direct",
                "dual_stream": False,
                "symmetric_decoder": False,
                "ply_path": str(ply),
                "use_refiner": False,
                "featsharp_mode": "analytical",
                "featsharp_strength": 0.0,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    report = {
        **provenance,
        "config": str(config),
        "config_sha256": _sha256(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "num_gaussians": int(model.num_gaussians),
        "latent_dim": int(latent_dim),
    }
    checkpoint.with_suffix(checkpoint.suffix + ".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--ply-path", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--latent-dim", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(build_contract(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
