#!/usr/bin/env python3
"""Generate RADIO-GS configs for prepared OpenGaussian ScanNet scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_TEMPLATE = Path("radio_gs/configs/scannet_hybrid_v14_template.yaml")
DEFAULT_PREPARED_ROOT = Path("dataset/scannet_og")
DEFAULT_OUTPUT_ROOT = Path("radio_gs/configs/generated/scannet_og")
DEFAULT_GEOM_TAG = "og_rgb_3dgs"
DEFAULT_ITERS = 30000


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_transforms(scene_root: Path) -> dict[str, Any]:
    for name in ("transforms.json", "transforms_train.json"):
        path = scene_root / name
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"No transforms.json or transforms_train.json found in {scene_root}")


def _feature_size(image_height: int, image_width: int, patch_size: int = 8) -> tuple[int, int]:
    return image_height // patch_size, image_width // patch_size


def generate_config(
    scene: str,
    prepared_root: str | Path = DEFAULT_PREPARED_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    template_path: str | Path = DEFAULT_TEMPLATE,
    repo_root: str | Path | None = None,
    geom_tag: str = DEFAULT_GEOM_TAG,
    iters: int = DEFAULT_ITERS,
    batch_size: int = 3,
    num_workers: int = 4,
    variant: str = "v14",
    epochs: int | None = None,
    siglip_summary_alignment_weight: float | None = None,
    direct_point_loss_weight: float = 0.0,
    direct_point_sample_count: int = 2048,
    direct_point_sample_strategy: str = "uniform",
    direct_point_query_mode: str = "gaussian_index",
    direct_point_gaussian_position_mode: str = "gaussian_center",
    direct_point_source: str = "gaussian",
    direct_point_teacher_cache: str = "",
    direct_point_feature_key: str = "features",
    direct_point_k: int = 8,
    direct_point_candidate_k: int = 0,
    direct_point_summary_alignment_weight: float = 0.0,
    direct_point_summary_adapter_weight: float = 0.0,
    direct_point_text_loss_weight: float = 0.0,
    direct_point_adapter_text_loss_weight: float = 0.0,
    direct_point_adapter_text_distill_weight: float = 0.0,
    direct_point_text_pseudo_ce_weight: float = 0.0,
    direct_point_text_pseudo_ce_confidence_threshold: float = 0.0,
    direct_point_text_pseudo_ce_logit_scale: float = 1.0,
    direct_point_text_pseudo_ce_center_logits: bool = False,
    direct_point_text_pseudo_ce_splits: str = "",
    direct_point_adapter_text_pseudo_ce_weight: float = 0.0,
    direct_point_adapter_text_pseudo_ce_confidence_threshold: float = 0.0,
    direct_point_adapter_text_pseudo_ce_logit_scale: float = 1.0,
    direct_point_adapter_text_pseudo_ce_center_logits: bool = False,
    direct_point_adapter_text_pseudo_ce_splits: str = "",
    direct_point_adapter_decoder_anchor_weight: float = 0.0,
    direct_point_text_embeddings: str = "checkpoints/siglip2_scannet_og_text_embeddings.pt",
    direct_point_text_split: str = "19",
    direct_point_text_temperature: float = 0.07,
    direct_point_text_ce_weighting: str = "none",
    direct_point_text_ce_min_weight: float = 0.5,
    direct_point_text_ce_max_weight: float = 3.0,
    direct_point_text_distill_weight: float = 0.0,
    direct_point_text_distill_temperature: float = 1.0,
    direct_point_text_distill_confidence_threshold: float = 0.0,
) -> Path:
    """Generate one ScanNet OpenGaussian RADIO-GS config and return its path."""
    repo_root = Path.cwd().resolve() if repo_root is None else Path(repo_root)
    prepared_root = Path(prepared_root)
    output_root = Path(output_root)
    template_path = Path(template_path)
    scene_root = prepared_root / scene

    if not scene_root.exists():
        raise FileNotFoundError(f"Prepared OpenGaussian scene not found: {scene_root}")

    required = [
        scene_root / "color",
        scene_root / "traj_w_c.txt",
        scene_root / "splits" / "train_frames.txt",
        scene_root / "splits" / "val_frames.txt",
        scene_root / "points3d.ply",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required prepared-scene path missing: {path}")

    transforms = _load_transforms(scene_root)
    image_width = int(transforms.get("w", 640))
    image_height = int(transforms.get("h", 480))
    feature_height, feature_width = _feature_size(image_height, image_width)
    variant = str(variant).strip() or "v14"

    cfg = _load_yaml(template_path)
    cfg.update(
        {
            "exp_name": f"radio_gs_scannet_og_{scene}_{variant}",
            "output_dir": str(repo_root / "output" / "radio_gs" / f"scannet_og_{scene}_{variant}"),
            "dataset_type": "scannet",
            "scene": scene,
            "scene_root": str(scene_root),
            "ply_path": str(
                repo_root
                / "output"
                / "3dgs_models"
                / "scannet_og"
                / scene
                / geom_tag
                / "point_cloud"
                / f"iteration_{int(iters)}"
                / "point_cloud.ply"
            ),
            "image_height": image_height,
            "image_width": image_width,
            "fx": float(transforms.get("fl_x", cfg.get("fx", 577.0))),
            "fy": float(transforms.get("fl_y", transforms.get("fl_x", cfg.get("fy", 577.0)))),
            "cx": float(transforms.get("cx", cfg.get("cx", image_width / 2.0 - 0.5))),
            "cy": float(transforms.get("cy", cfg.get("cy", image_height / 2.0 - 0.5))),
            "feature_height": feature_height,
            "feature_width": feature_width,
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "feature_dir": str(repo_root / "output" / "radio_features_scannet_og" / scene),
            "val_feature_dir": str(repo_root / "output" / "radio_features_scannet_og" / scene),
            "pose_file": str(scene_root / "traj_w_c.txt"),
            "val_pose_file": str(scene_root / "traj_w_c.txt"),
            "pose_dir": "",
            "val_pose_dir": "",
            "train_frame_ids_path": str(scene_root / "splits" / "train_frames.txt"),
            "val_frame_ids_path": str(scene_root / "splits" / "val_frames.txt"),
            "rgb_dir": str(scene_root / "color"),
            "val_rgb_dir": str(scene_root / "color"),
            "depth_dir": "",
            "val_depth_dir": "",
            "semantics_dir": "",
            "val_semantics_dir": "",
            "instance_dir": "",
            "val_instance_dir": "",
            "depth_loss_weight": 0.0,
            "geom_depth_loss_weight": 0.0,
            "seg_loss_weight": 0.0,
            "frozen_depth_head_weight": 0.0,
            "frozen_seg_head_weight": 0.0,
            "direct_point_loss_weight": float(direct_point_loss_weight),
            "direct_point_sample_count": int(direct_point_sample_count),
            "direct_point_sample_strategy": str(direct_point_sample_strategy),
            "direct_point_query_mode": str(direct_point_query_mode),
            "direct_point_gaussian_position_mode": str(direct_point_gaussian_position_mode),
            "direct_point_source": str(direct_point_source),
            "direct_point_teacher_cache": str(direct_point_teacher_cache),
            "direct_point_feature_key": str(direct_point_feature_key),
            "direct_point_k": int(direct_point_k),
            "direct_point_candidate_k": int(direct_point_candidate_k),
            "direct_point_summary_alignment_weight": float(
                direct_point_summary_alignment_weight
            ),
            "direct_point_summary_adapter_weight": float(
                direct_point_summary_adapter_weight
            ),
            "direct_point_text_loss_weight": float(direct_point_text_loss_weight),
            "direct_point_adapter_text_loss_weight": float(
                direct_point_adapter_text_loss_weight
            ),
            "direct_point_adapter_text_distill_weight": float(
                direct_point_adapter_text_distill_weight
            ),
            "direct_point_text_pseudo_ce_weight": float(
                direct_point_text_pseudo_ce_weight
            ),
            "direct_point_text_pseudo_ce_confidence_threshold": float(
                direct_point_text_pseudo_ce_confidence_threshold
            ),
            "direct_point_text_pseudo_ce_logit_scale": float(
                direct_point_text_pseudo_ce_logit_scale
            ),
            "direct_point_text_pseudo_ce_center_logits": bool(
                direct_point_text_pseudo_ce_center_logits
            ),
            "direct_point_text_pseudo_ce_splits": str(
                direct_point_text_pseudo_ce_splits
            ),
            "direct_point_adapter_text_pseudo_ce_weight": float(
                direct_point_adapter_text_pseudo_ce_weight
            ),
            "direct_point_adapter_text_pseudo_ce_confidence_threshold": float(
                direct_point_adapter_text_pseudo_ce_confidence_threshold
            ),
            "direct_point_adapter_text_pseudo_ce_logit_scale": float(
                direct_point_adapter_text_pseudo_ce_logit_scale
            ),
            "direct_point_adapter_text_pseudo_ce_center_logits": bool(
                direct_point_adapter_text_pseudo_ce_center_logits
            ),
            "direct_point_adapter_text_pseudo_ce_splits": str(
                direct_point_adapter_text_pseudo_ce_splits
            ),
            "direct_point_adapter_decoder_anchor_weight": float(
                direct_point_adapter_decoder_anchor_weight
            ),
            "direct_point_text_embeddings": str(direct_point_text_embeddings),
            "direct_point_text_split": str(direct_point_text_split),
            "direct_point_text_temperature": float(direct_point_text_temperature),
            "direct_point_text_ce_weighting": str(direct_point_text_ce_weighting),
            "direct_point_text_ce_min_weight": float(direct_point_text_ce_min_weight),
            "direct_point_text_ce_max_weight": float(direct_point_text_ce_max_weight),
            "direct_point_text_distill_weight": float(direct_point_text_distill_weight),
            "direct_point_text_distill_temperature": float(
                direct_point_text_distill_temperature
            ),
            "direct_point_text_distill_confidence_threshold": float(
                direct_point_text_distill_confidence_threshold
            ),
        }
    )
    if epochs is not None:
        cfg["epochs"] = int(epochs)
    if siglip_summary_alignment_weight is not None:
        cfg["siglip_summary_alignment_weight"] = float(siglip_summary_alignment_weight)

    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"scannet_og_hybrid_{variant}_{scene}.yaml"
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="+", required=True, help="Prepared ScanNet scene ids")
    parser.add_argument("--prepared_root", default=str(DEFAULT_PREPARED_ROOT))
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--repo_root", default=str(Path.cwd().resolve()))
    parser.add_argument("--geom_tag", default=DEFAULT_GEOM_TAG)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--variant", default="v14")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--siglip_summary_alignment_weight", type=float, default=None)
    parser.add_argument("--direct_point_loss_weight", type=float, default=0.0)
    parser.add_argument("--direct_point_sample_count", type=int, default=2048)
    parser.add_argument(
        "--direct_point_sample_strategy",
        choices=["uniform", "class_balanced", "teacher_balanced"],
        default="uniform",
    )
    parser.add_argument(
        "--direct_point_query_mode",
        choices=["gaussian_index", "knn"],
        default="gaussian_index",
    )
    parser.add_argument(
        "--direct_point_gaussian_position_mode",
        choices=["gaussian_center", "label_point"],
        default="gaussian_center",
    )
    parser.add_argument(
        "--direct_point_source",
        choices=["gaussian", "label_ply", "points3d"],
        default="gaussian",
    )
    parser.add_argument("--direct_point_teacher_cache", default="")
    parser.add_argument(
        "--direct_point_feature_key",
        choices=["features", "fused", "semantic", "geometry"],
        default="features",
    )
    parser.add_argument("--direct_point_k", type=int, default=8)
    parser.add_argument("--direct_point_candidate_k", type=int, default=0)
    parser.add_argument("--direct_point_summary_alignment_weight", type=float, default=0.0)
    parser.add_argument("--direct_point_summary_adapter_weight", type=float, default=0.0)
    parser.add_argument("--direct_point_text_loss_weight", type=float, default=0.0)
    parser.add_argument("--direct_point_adapter_text_loss_weight", type=float, default=0.0)
    parser.add_argument("--direct_point_adapter_text_distill_weight", type=float, default=0.0)
    parser.add_argument("--direct_point_text_pseudo_ce_weight", type=float, default=0.0)
    parser.add_argument(
        "--direct_point_text_pseudo_ce_confidence_threshold",
        type=float,
        default=0.0,
    )
    parser.add_argument("--direct_point_text_pseudo_ce_logit_scale", type=float, default=1.0)
    parser.add_argument("--direct_point_text_pseudo_ce_center_logits", action="store_true")
    parser.add_argument("--direct_point_text_pseudo_ce_splits", default="")
    parser.add_argument("--direct_point_adapter_text_pseudo_ce_weight", type=float, default=0.0)
    parser.add_argument(
        "--direct_point_adapter_text_pseudo_ce_confidence_threshold",
        type=float,
        default=0.0,
    )
    parser.add_argument("--direct_point_adapter_text_pseudo_ce_logit_scale", type=float, default=1.0)
    parser.add_argument("--direct_point_adapter_text_pseudo_ce_center_logits", action="store_true")
    parser.add_argument("--direct_point_adapter_text_pseudo_ce_splits", default="")
    parser.add_argument("--direct_point_adapter_decoder_anchor_weight", type=float, default=0.0)
    parser.add_argument(
        "--direct_point_text_embeddings",
        default="checkpoints/siglip2_scannet_og_text_embeddings.pt",
    )
    parser.add_argument("--direct_point_text_split", choices=["19", "15", "10"], default="19")
    parser.add_argument("--direct_point_text_temperature", type=float, default=0.07)
    parser.add_argument(
        "--direct_point_text_ce_weighting",
        choices=["none", "inverse_batch", "inverse_pool", "sqrt_inverse_pool_capped"],
        default="none",
    )
    parser.add_argument("--direct_point_text_ce_min_weight", type=float, default=0.5)
    parser.add_argument("--direct_point_text_ce_max_weight", type=float, default=3.0)
    parser.add_argument("--direct_point_text_distill_weight", type=float, default=0.0)
    parser.add_argument("--direct_point_text_distill_temperature", type=float, default=1.0)
    parser.add_argument(
        "--direct_point_text_distill_confidence_threshold",
        type=float,
        default=0.0,
    )
    args = parser.parse_args()

    for scene in args.scenes:
        path = generate_config(
            scene=scene,
            prepared_root=args.prepared_root,
            output_root=args.output_root,
            template_path=args.template,
            repo_root=args.repo_root,
            geom_tag=args.geom_tag,
            iters=args.iters,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            variant=args.variant,
            epochs=args.epochs,
            siglip_summary_alignment_weight=args.siglip_summary_alignment_weight,
            direct_point_loss_weight=args.direct_point_loss_weight,
            direct_point_sample_count=args.direct_point_sample_count,
            direct_point_sample_strategy=args.direct_point_sample_strategy,
            direct_point_query_mode=args.direct_point_query_mode,
            direct_point_gaussian_position_mode=args.direct_point_gaussian_position_mode,
            direct_point_source=args.direct_point_source,
            direct_point_teacher_cache=args.direct_point_teacher_cache,
            direct_point_feature_key=args.direct_point_feature_key,
            direct_point_k=args.direct_point_k,
            direct_point_candidate_k=args.direct_point_candidate_k,
            direct_point_summary_alignment_weight=args.direct_point_summary_alignment_weight,
            direct_point_summary_adapter_weight=args.direct_point_summary_adapter_weight,
            direct_point_text_loss_weight=args.direct_point_text_loss_weight,
            direct_point_adapter_text_loss_weight=args.direct_point_adapter_text_loss_weight,
            direct_point_adapter_text_distill_weight=(
                args.direct_point_adapter_text_distill_weight
            ),
            direct_point_text_pseudo_ce_weight=args.direct_point_text_pseudo_ce_weight,
            direct_point_text_pseudo_ce_confidence_threshold=(
                args.direct_point_text_pseudo_ce_confidence_threshold
            ),
            direct_point_text_pseudo_ce_logit_scale=(
                args.direct_point_text_pseudo_ce_logit_scale
            ),
            direct_point_text_pseudo_ce_center_logits=(
                args.direct_point_text_pseudo_ce_center_logits
            ),
            direct_point_text_pseudo_ce_splits=(
                args.direct_point_text_pseudo_ce_splits
            ),
            direct_point_adapter_text_pseudo_ce_weight=(
                args.direct_point_adapter_text_pseudo_ce_weight
            ),
            direct_point_adapter_text_pseudo_ce_confidence_threshold=(
                args.direct_point_adapter_text_pseudo_ce_confidence_threshold
            ),
            direct_point_adapter_text_pseudo_ce_logit_scale=(
                args.direct_point_adapter_text_pseudo_ce_logit_scale
            ),
            direct_point_adapter_text_pseudo_ce_center_logits=(
                args.direct_point_adapter_text_pseudo_ce_center_logits
            ),
            direct_point_adapter_text_pseudo_ce_splits=(
                args.direct_point_adapter_text_pseudo_ce_splits
            ),
            direct_point_adapter_decoder_anchor_weight=(
                args.direct_point_adapter_decoder_anchor_weight
            ),
            direct_point_text_embeddings=args.direct_point_text_embeddings,
            direct_point_text_split=args.direct_point_text_split,
            direct_point_text_temperature=args.direct_point_text_temperature,
            direct_point_text_ce_weighting=args.direct_point_text_ce_weighting,
            direct_point_text_ce_min_weight=args.direct_point_text_ce_min_weight,
            direct_point_text_ce_max_weight=args.direct_point_text_ce_max_weight,
            direct_point_text_distill_weight=args.direct_point_text_distill_weight,
            direct_point_text_distill_temperature=args.direct_point_text_distill_temperature,
            direct_point_text_distill_confidence_threshold=(
                args.direct_point_text_distill_confidence_threshold
            ),
        )
        print(path)


if __name__ == "__main__":
    main()
