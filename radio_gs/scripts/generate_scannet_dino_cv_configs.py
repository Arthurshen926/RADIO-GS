#!/usr/bin/env python3
"""Generate ScanNet VALA8 DINO cross-view fine-tuning configs.

The base checkpoints may still come from historical v67-named runs, but this
generator's default scene set is paper-facing VALA8 only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import yaml


BASE_VARIANT = "v67fair_teacherbalanced_gidx_labelpoint_dp080_pce10_tdist05_s32768_b4_long20_fromv63"
DEFAULT_VARIANT = "v67_dino_cv001_b2_s32768_ft20"
SCENES = [
    "scene0000_00",
    "scene0062_00",
    "scene0070_00",
    "scene0097_00",
    "scene0140_00",
    "scene0347_00",
    "scene0400_00",
    "scene0590_00",
]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def generate_config(
    scene: str,
    base_config_dir: Path,
    output_config_dir: Path,
    repo_root: Path,
    variant: str = DEFAULT_VARIANT,
    cross_view_weight: float = 0.001,
    batch_size: int = 2,
    epochs: int = 20,
    cross_view_downsample: int = 2,
    cross_view_max_tokens: int = 128,
    direct_render_consistency_weight: float = 0.05,
    direct_text_contrast_weight: float = 0.05,
    direct_cached_visible_fraction: float = 0.5,
    direct_cached_visible_candidate_multiplier: int = 1,
    direct_cached_visible_balance: bool = False,
    direct_text_contrast_center_logits: bool = False,
) -> Path:
    base_path = base_config_dir / f"scannet_og_hybrid_{BASE_VARIANT}_{scene}.yaml"
    cfg = _load_yaml(base_path)
    output_dir = repo_root / "output" / "radio_gs" / f"scannet_og_{scene}_{variant}"
    warmstart = (
        repo_root
        / "output"
        / "radio_gs"
        / f"scannet_og_{scene}_{BASE_VARIANT}"
        / "checkpoints"
        / "best.pth"
    )
    cfg.update(
        {
            "exp_name": f"radio_gs_scannet_og_{scene}_{variant}",
            "output_dir": str(output_dir),
            "batch_size": int(batch_size),
            "train_shuffle": False,
            "epochs": int(epochs),
            "warmup_epochs": min(int(cfg.get("warmup_epochs", 4)), 2),
            "lr_features": min(float(cfg.get("lr_features", 1e-4)), 2e-5),
            "lr_hash": min(float(cfg.get("lr_hash", 1.5e-4)), 3e-5),
            "lr_decoder": min(float(cfg.get("lr_decoder", 6e-5)), 1.5e-5),
            "lr_heads": min(float(cfg.get("lr_heads", 6e-5)), 1.5e-5),
            "lr_refiner": min(float(cfg.get("lr_refiner", 2e-4)), 4e-5),
            "warmstart_from": str(warmstart),
            "radio_adaptor_alignment_checkpoint": "/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar",
            "radio_adaptor_cross_view_names": "dino_v3",
            "radio_adaptor_cross_view_weight": float(cross_view_weight),
            "radio_adaptor_cross_view_downsample": int(cross_view_downsample),
            "radio_adaptor_cross_view_max_tokens": int(cross_view_max_tokens),
            "radio_adaptor_cross_view_temperature": 1.0,
            "direct_point_view_count_weighting": "clipped_log",
            "direct_point_view_count_min_weight": 0.25,
            "direct_point_view_count_percentile_low": 5.0,
            "direct_point_view_count_percentile_high": 95.0,
            "direct_point_text_contrast_weight": float(direct_text_contrast_weight),
            "direct_point_text_contrast_temperature": 0.1,
            "direct_point_text_contrast_confidence_threshold": 0.05,
            "direct_point_text_contrast_pair_weighting": "visibility",
            "direct_point_text_contrast_max_points": 4096,
            "direct_point_text_contrast_center_logits": bool(direct_text_contrast_center_logits),
            "direct_point_render_consistency_weight": float(direct_render_consistency_weight),
            "direct_point_render_consistency_mode": "cosine",
            "direct_point_cached_visible_fraction": float(direct_cached_visible_fraction),
            "direct_point_cached_visible_candidate_multiplier": int(
                direct_cached_visible_candidate_multiplier
            ),
            "direct_point_cached_visible_balance": bool(direct_cached_visible_balance),
        }
    )
    output_path = output_config_dir / f"scannet_og_hybrid_{variant}_{scene}.yaml"
    _write_yaml(output_path, cfg)
    return output_path


def generate_configs(
    scenes: Iterable[str],
    base_config_dir: Path,
    output_config_dir: Path,
    repo_root: Path,
    variant: str = DEFAULT_VARIANT,
    cross_view_weight: float = 0.001,
    batch_size: int = 2,
    epochs: int = 20,
    direct_render_consistency_weight: float = 0.05,
    direct_text_contrast_weight: float = 0.05,
    direct_cached_visible_fraction: float = 0.5,
    direct_cached_visible_candidate_multiplier: int = 1,
    direct_cached_visible_balance: bool = False,
    direct_text_contrast_center_logits: bool = False,
) -> list[Path]:
    return [
        generate_config(
            scene=scene,
            base_config_dir=base_config_dir,
            output_config_dir=output_config_dir,
            repo_root=repo_root,
            variant=variant,
            cross_view_weight=cross_view_weight,
            batch_size=batch_size,
            epochs=epochs,
            direct_render_consistency_weight=direct_render_consistency_weight,
            direct_text_contrast_weight=direct_text_contrast_weight,
            direct_cached_visible_fraction=direct_cached_visible_fraction,
            direct_cached_visible_candidate_multiplier=direct_cached_visible_candidate_multiplier,
            direct_cached_visible_balance=direct_cached_visible_balance,
            direct_text_contrast_center_logits=direct_text_contrast_center_logits,
        )
        for scene in scenes
    ]


def scene_from_config_path(path: Path, variant: str) -> str:
    prefix = f"scannet_og_hybrid_{variant}_"
    name = path.name
    if not name.startswith(prefix) or not name.endswith(".yaml"):
        raise ValueError(f"Config name does not match variant {variant}: {path}")
    return name[len(prefix) : -len(".yaml")]


def write_launch_plan(
    paths: list[Path],
    variant: str,
    path: Path,
    *,
    batch_size: int,
    cross_view_weight: float,
    direct_render_consistency_weight: float,
    direct_text_contrast_weight: float,
    direct_cached_visible_fraction: float,
    direct_cached_visible_candidate_multiplier: int,
    direct_cached_visible_balance: bool,
    direct_text_contrast_center_logits: bool,
) -> None:
    gpu4 = [scene_from_config_path(p, variant) for p in paths[::2]]
    gpu5 = [scene_from_config_path(p, variant) for p in paths[1::2]]
    lines = [
        "# ScanNet DINO Cross-View Launch Plan",
        "",
        f"- Variant: `{variant}`",
        "- Protocol: paper-facing VALA8, teacher-balanced, gaussian_index, label_point, label_index",
        f"- Batch size: {batch_size}",
        "- Direct point samples: inherited from v67 configs (`32768`)",
        f"- Cross-view adaptor: `dino_v3`, weight `{cross_view_weight:g}`",
        f"- Direct render consistency: weight `{direct_render_consistency_weight:g}`, mode `cosine`",
        f"- Cached direct-point visible sampling: `{direct_cached_visible_fraction:g}` of samples",
        f"- Cached visible candidate multiplier: `{direct_cached_visible_candidate_multiplier}`",
        f"- Cached visible teacher-balanced replay: `{direct_cached_visible_balance}`",
        f"- Visibility-weighted text contrast: weight `{direct_text_contrast_weight:g}`, pair weighting `visibility`",
        "- Direct text contrast cap: `4096` points per batch",
        "- Direct text contrast pseudo-labels: "
        + (
            "scene-centered teacher text logits"
            if direct_text_contrast_center_logits
            else "raw teacher text logits"
        ),
        "",
        "## Suggested GPU Split",
        "",
        f"- GPU4 scenes: `{', '.join(gpu4)}`",
        f"- GPU5 scenes: `{', '.join(gpu5)}`",
        "",
        "## Configs",
        "",
    ]
    lines.extend(f"- `{path}`" for path in paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="+", default=SCENES)
    parser.add_argument("--base_config_dir", default="radio_gs/configs/generated/scannet_og")
    parser.add_argument("--output_config_dir", default="radio_gs/configs/generated/scannet_dino_cv")
    parser.add_argument("--repo_root", default=str(Path.cwd().resolve()))
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--cross_view_weight", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--direct_render_consistency_weight", type=float, default=0.05)
    parser.add_argument("--direct_text_contrast_weight", type=float, default=0.05)
    parser.add_argument("--direct_cached_visible_fraction", type=float, default=0.5)
    parser.add_argument("--direct_cached_visible_candidate_multiplier", type=int, default=1)
    parser.add_argument("--direct_cached_visible_balance", action="store_true")
    parser.add_argument(
        "--direct_text_contrast_center_logits",
        action="store_true",
        help="Center teacher text logits for direct-point text contrast.",
    )
    args = parser.parse_args()

    paths = generate_configs(
        args.scenes,
        base_config_dir=Path(args.base_config_dir),
        output_config_dir=Path(args.output_config_dir),
        repo_root=Path(args.repo_root),
        variant=args.variant,
        cross_view_weight=args.cross_view_weight,
        batch_size=args.batch_size,
        epochs=args.epochs,
        direct_render_consistency_weight=args.direct_render_consistency_weight,
        direct_text_contrast_weight=args.direct_text_contrast_weight,
        direct_cached_visible_fraction=args.direct_cached_visible_fraction,
        direct_cached_visible_candidate_multiplier=args.direct_cached_visible_candidate_multiplier,
        direct_cached_visible_balance=args.direct_cached_visible_balance,
        direct_text_contrast_center_logits=args.direct_text_contrast_center_logits,
    )
    write_launch_plan(
        paths,
        args.variant,
        Path(args.repo_root) / "output/radio_gs/reports/scannet_dino_cv_launch_plan.md",
        batch_size=args.batch_size,
        cross_view_weight=args.cross_view_weight,
        direct_render_consistency_weight=args.direct_render_consistency_weight,
        direct_text_contrast_weight=args.direct_text_contrast_weight,
        direct_cached_visible_fraction=args.direct_cached_visible_fraction,
        direct_cached_visible_candidate_multiplier=args.direct_cached_visible_candidate_multiplier,
        direct_cached_visible_balance=args.direct_cached_visible_balance,
        direct_text_contrast_center_logits=args.direct_text_contrast_center_logits,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
