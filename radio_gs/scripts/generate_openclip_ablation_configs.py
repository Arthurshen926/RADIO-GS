#!/usr/bin/env python3
"""Generate RADIO-GS configs for dense OpenCLIP-token ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import yaml

from radio_gs.scripts.generate_samclip_ablation_configs import (
    RADIO_HELPER_EMPTY_KEYS,
    RADIO_HELPER_EMPTY_NAME_KEYS,
    RADIO_HELPER_FALSE_KEYS,
    RADIO_HELPER_ZERO_KEYS,
)


SAMCLIP_ZERO_KEYS = (
    "samclip_mask_loss_weight",
    "samclip_contrastive_loss_weight",
    "samclip_background_loss_weight",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_manifest(feature_dir: Path) -> dict[str, Any]:
    path = feature_dir / "openclip_dense_manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _scene_slug(scene: str, dataset_type: str) -> str:
    if dataset_type == "lerf" and not scene.startswith("lerf_"):
        return f"lerf_{scene}"
    return scene


def generate_config(
    template_path: str | Path,
    *,
    scene: str,
    feature_dir: str | Path,
    output_root: str | Path,
    variant: str,
    scene_root: str | Path | None = None,
    repo_root: str | Path | None = None,
    epochs: int | None = None,
) -> Path:
    """Clone one RADIO-GS config into a dense OpenCLIP-token ablation config."""
    template_path = Path(template_path)
    feature_dir = Path(feature_dir)
    output_root = Path(output_root)
    repo_root = Path.cwd().resolve() if repo_root is None else Path(repo_root)
    variant = str(variant).strip()
    if not variant:
        raise ValueError("variant must be non-empty")

    cfg = _load_yaml(template_path)
    dataset_type = str(cfg.get("dataset_type", "scannet" if scene.startswith("scene") else "lerf"))
    slug = _scene_slug(scene, dataset_type)
    exp_name = f"{slug}_{variant}"
    manifest = _load_manifest(feature_dir)

    cfg.update(
        {
            "exp_name": exp_name,
            "output_dir": str(repo_root / "output" / "radio_gs" / exp_name),
            "scene": scene,
            "radio_feature_dim": 512,
            "codec_type": "identity",
            "bottleneck_dim": 512,
            "hybrid_output_dim": 512,
            "latent_dim": 512,
            "dual_stream": False,
            "symmetric_decoder": False,
            "feature_dir": str(feature_dir),
            "val_feature_dir": str(feature_dir),
            "samclip_language_feature_dir": "",
            "samclip_feature_level": 0,
            "samclip_mask_loss_weight": 0.0,
            "samclip_contrastive_loss_weight": 0.0,
            "samclip_background_loss_weight": 0.0,
        }
    )
    if scene_root is not None:
        scene_root_path = Path(scene_root)
        cfg["scene_root"] = str(scene_root_path)
        if dataset_type == "lerf":
            cfg["rgb_dir"] = str(scene_root_path / "images")
        elif dataset_type == "scannet":
            cfg["rgb_dir"] = str(scene_root_path / "color")
            cfg["val_rgb_dir"] = str(scene_root_path / "color")
    if epochs is not None:
        cfg["epochs"] = int(epochs)

    output_size = manifest.get("output_size")
    if isinstance(output_size, list) and len(output_size) == 2:
        cfg["feature_height"] = int(output_size[0])
        cfg["feature_width"] = int(output_size[1])

    for key in RADIO_HELPER_ZERO_KEYS:
        cfg[key] = 0.0
    for key in RADIO_HELPER_EMPTY_KEYS:
        cfg[key] = ""
    for key in RADIO_HELPER_EMPTY_NAME_KEYS:
        cfg[key] = ""
    for key in RADIO_HELPER_FALSE_KEYS:
        cfg[key] = False
    for key in SAMCLIP_ZERO_KEYS:
        cfg[key] = 0.0
    cfg["samclip_language_feature_dir"] = ""
    cfg["grounding_use_adaptor"] = False
    cfg["featsharp_mode"] = "none"
    cfg["featsharp_strength"] = 0.0

    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{exp_name}.yaml"
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--templates", type=Path, nargs="+", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--scene-root", type=Path, default=None)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd().resolve())
    parser.add_argument("--epochs", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    paths = [
        generate_config(
            template,
            scene=args.scene,
            scene_root=args.scene_root,
            feature_dir=args.feature_dir,
            output_root=args.output_root,
            variant=args.variant,
            repo_root=args.repo_root,
            epochs=args.epochs,
        )
        for template in args.templates
    ]
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
