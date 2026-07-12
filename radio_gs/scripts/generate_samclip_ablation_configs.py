#!/usr/bin/env python3
"""Generate RADIO-GS configs for SAM-CLIP language-feature ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import yaml


RADIO_HELPER_ZERO_KEYS = (
    "adaptor_weight",
    "boundary_aware_loss_weight",
    "depth_guided_feature_weight",
    "depth_loss_weight",
    "geometric_edge_loss_weight",
    "geom_depth_loss_weight",
    "gradient_loss_weight",
    "hybrid_semantic_adaptor_reg_weight",
    "hybrid_semantic_aux_weight",
    "seg_loss_weight",
    "siglip_alignment_weight",
    "siglip_summary_alignment_weight",
    "text_heatmap_distill_weight",
    "tv_weight",
    "radio_adaptor_alignment_weight",
    "radio_adaptor_relation_weight",
    "radio_adaptor_local_affinity_weight",
    "radio_adaptor_token_contrast_weight",
    "radio_adaptor_peak_background_weight",
    "radio_adaptor_region_weight",
    "radio_adaptor_mask_logit_weight",
    "radio_adaptor_cross_view_weight",
    "radio_adaptor_cross_view_propagation_weight",
    "radio_adaptor_cross_view_mask_propagation_weight",
    "direct_point_loss_weight",
    "direct_point_summary_alignment_weight",
    "direct_point_summary_adapter_weight",
    "direct_point_text_loss_weight",
    "direct_point_adapter_text_loss_weight",
    "direct_point_adapter_text_distill_weight",
    "direct_point_text_pseudo_ce_weight",
    "direct_point_adapter_text_pseudo_ce_weight",
    "direct_point_adapter_decoder_anchor_weight",
    "direct_point_text_distill_weight",
    "grounding_query_loss_weight",
    "frozen_depth_head_weight",
    "frozen_seg_head_weight",
)

RADIO_HELPER_EMPTY_KEYS = (
    "siglip_projection_weights",
    "siglip_summary_head_weights",
    "direct_point_teacher_cache",
    "direct_point_text_embeddings",
    "text_heatmap_distill_embeddings",
    "grounding_text_embeddings",
    "frozen_depth_head_path",
    "frozen_seg_head_path",
    "warmstart_from",
    "resume_from",
)

RADIO_HELPER_EMPTY_NAME_KEYS = (
    "radio_adaptor_alignment_names",
    "radio_adaptor_relation_names",
    "radio_adaptor_local_affinity_names",
    "radio_adaptor_token_contrast_names",
    "radio_adaptor_peak_background_names",
    "radio_adaptor_region_names",
    "radio_adaptor_mask_logit_names",
    "radio_adaptor_cross_view_names",
    "radio_adaptor_cross_view_propagation_names",
    "radio_adaptor_cross_view_mask_propagation_names",
)

RADIO_HELPER_FALSE_KEYS = (
    "hybrid_semantic_adaptor",
    "hybrid_semantic_adaptor_use_depth_guidance",
    "hybrid_semantic_adaptor_use_geometry_guidance",
    "refiner_alpha_guide",
    "refiner_boundary_guide",
    "refiner_depth_grad",
    "refiner_depth_guide",
    "refiner_rgb_guide",
    "self_guided",
    "use_refiner",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_manifest(level_root: Path) -> dict[str, Any]:
    path = level_root / "samclip_manifest.json"
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
    scene_root: str | Path | None = None,
    samclip_root: str | Path,
    level: int,
    output_root: str | Path,
    variant: str,
    repo_root: str | Path | None = None,
    epochs: int | None = None,
    samclip_mask_loss_weight: float = 0.0,
    samclip_contrastive_loss_weight: float = 0.0,
    samclip_background_loss_weight: float = 0.0,
    samclip_contrastive_temperature: float = 0.07,
    samclip_mask_min_pixels: int = 16,
    samclip_mask_max_regions: int = 64,
) -> Path:
    """Clone one RADIO-GS config into a SAM-CLIP ablation config."""
    template_path = Path(template_path)
    samclip_root = Path(samclip_root)
    output_root = Path(output_root)
    repo_root = Path.cwd().resolve() if repo_root is None else Path(repo_root)
    level = int(level)
    variant = str(variant).strip()
    if not variant:
        raise ValueError("variant must be non-empty")

    cfg = _load_yaml(template_path)
    dataset_type = str(cfg.get("dataset_type", "scannet" if scene.startswith("scene") else "lerf"))
    slug = _scene_slug(scene, dataset_type)
    exp_name = f"{slug}_{variant}"
    level_root = samclip_root / scene / f"l{level}"
    manifest = _load_manifest(level_root)

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
            "feature_dir": str(level_root),
            "val_feature_dir": str(level_root),
            "samclip_feature_level": level,
            "samclip_language_feature_dir": str(level_root),
            "samclip_mask_loss_weight": float(samclip_mask_loss_weight),
            "samclip_contrastive_loss_weight": float(samclip_contrastive_loss_weight),
            "samclip_background_loss_weight": float(samclip_background_loss_weight),
            "samclip_contrastive_temperature": float(samclip_contrastive_temperature),
            "samclip_mask_min_pixels": int(samclip_mask_min_pixels),
            "samclip_mask_max_regions": int(samclip_mask_max_regions),
            "samclip_source_config": str(template_path),
        }
    )
    if scene_root is not None:
        cfg["scene_root"] = str(Path(scene_root))
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
    parser.add_argument("--samclip-root", type=Path, required=True)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd().resolve())
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--samclip-mask-loss-weight", type=float, default=0.0)
    parser.add_argument("--samclip-contrastive-loss-weight", type=float, default=0.0)
    parser.add_argument("--samclip-background-loss-weight", type=float, default=0.0)
    parser.add_argument("--samclip-contrastive-temperature", type=float, default=0.07)
    parser.add_argument("--samclip-mask-min-pixels", type=int, default=16)
    parser.add_argument("--samclip-mask-max-regions", type=int, default=64)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    paths = [
        generate_config(
            template,
            scene=args.scene,
            scene_root=args.scene_root,
            samclip_root=args.samclip_root,
            level=args.level,
            output_root=args.output_root,
            variant=args.variant,
            repo_root=args.repo_root,
            epochs=args.epochs,
            samclip_mask_loss_weight=args.samclip_mask_loss_weight,
            samclip_contrastive_loss_weight=args.samclip_contrastive_loss_weight,
            samclip_background_loss_weight=args.samclip_background_loss_weight,
            samclip_contrastive_temperature=args.samclip_contrastive_temperature,
            samclip_mask_min_pixels=args.samclip_mask_min_pixels,
            samclip_mask_max_regions=args.samclip_mask_max_regions,
        )
        for template in args.templates
    ]
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
