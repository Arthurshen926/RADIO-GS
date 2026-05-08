#!/usr/bin/env python3
"""Generate controlled LERF component-ablation configs from seed-7 mainline configs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


SCENE_BASE_CONFIGS = {
    "figurines": "radio_gs/configs/generated/seeds/lerf_hybrid_v14_figurines_fdh_ws240_240ep_seed7.yaml",
    "ramen": "radio_gs/configs/generated/seeds/lerf_hybrid_v14_ramen_fdh_ws240_240ep_seed7.yaml",
    "teatime": "radio_gs/configs/generated/seeds/lerf_hybrid_v14_teatime_fdh_ws240_240ep_seed7.yaml",
    "waldo_kitchen": "radio_gs/configs/generated/seeds/lerf_hybrid_v14_waldo_kitchen_fdh_ws240_240ep_seed7.yaml",
}

DEFAULT_VARIANTS = ("no_refiner", "no_hybrid", "direct_codec")


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return dict(payload)


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def build_variant_payload(
    base: dict[str, Any],
    *,
    scene: str,
    variant: str,
    repo_root: Path,
    epochs: int,
) -> dict[str, Any]:
    payload = dict(base)
    tag = f"lerf_{scene}_component_{variant}_seed7"
    payload["exp_name"] = f"radio_gs_{tag}"
    payload["output_dir"] = str(repo_root / "output" / "radio_gs" / tag)
    payload["seed"] = 7
    payload["mixed_seed"] = 7
    payload["epochs"] = int(epochs)
    payload["codec_type"] = "hcd"

    if variant == "no_refiner":
        payload["use_refiner"] = False
        payload["refiner_rgb_guide"] = False
        payload["refiner_depth_guide"] = False
        payload["refiner_depth_grad"] = False
        payload["refiner_alpha_guide"] = False
        payload["refiner_boundary_guide"] = False
    elif variant == "no_hybrid":
        bottleneck_dim = int(payload.get("bottleneck_dim", payload.get("hybrid_output_dim", 192)))
        payload["architecture"] = "explicit"
        payload["latent_dim"] = bottleneck_dim
    elif variant == "direct_codec":
        payload["codec_type"] = "direct"
        payload["dual_stream"] = False
        payload["symmetric_decoder"] = False
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return payload


def generate_configs(
    *,
    repo_root: Path,
    output_dir: Path,
    scenes: list[str],
    variants: list[str],
    epochs: int,
) -> list[Path]:
    written: list[Path] = []
    for scene in scenes:
        if scene not in SCENE_BASE_CONFIGS:
            raise ValueError(f"Unsupported scene: {scene}")
        base_path = repo_root / SCENE_BASE_CONFIGS[scene]
        base = read_yaml(base_path)
        for variant in variants:
            payload = build_variant_payload(
                base,
                scene=scene,
                variant=variant,
                repo_root=repo_root,
                epochs=epochs,
            )
            out_path = output_dir / f"lerf_{scene}_component_{variant}_seed7.yaml"
            write_yaml(out_path, payload)
            written.append(out_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo_root",
        default=str(Path(__file__).resolve().parents[2]),
    )
    parser.add_argument(
        "--output_dir",
        default="radio_gs/configs/generated/ablation",
    )
    parser.add_argument(
        "--scenes",
        default=",".join(SCENE_BASE_CONFIGS),
        help="Comma-separated LERF scenes",
    )
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated variants: no_refiner,no_hybrid,direct_codec",
    )
    parser.add_argument("--epochs", type=int, default=240)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    scenes = [part.strip() for part in args.scenes.split(",") if part.strip()]
    variants = [part.strip() for part in args.variants.split(",") if part.strip()]

    for path in generate_configs(
        repo_root=repo_root,
        output_dir=output_dir,
        scenes=scenes,
        variants=variants,
        epochs=args.epochs,
    ):
        print(path)


if __name__ == "__main__":
    main()
