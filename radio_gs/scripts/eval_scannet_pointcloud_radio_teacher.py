#!/usr/bin/env python3
"""Evaluate multiview frame-wise RADIO features on ScanNet label points.

This is a diagnostic baseline for Open-Vocabulary Point Cloud Understanding:
it bypasses RADIO-GS parameters entirely, samples extracted 2-D RADIO feature
maps at 3-D label-PLY vertices through known camera poses, projects the
aggregated RADIO reference feature into SigLIP text space, then runs the same
OpenGaussian NYU40 19/15/10 metrics as the direct RADIO-GS evaluator.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from tqdm import tqdm

from radio_gs.artifact_paths import DEFAULT_SIGLIP2_PROJECTION_WEIGHTS
from radio_gs.config import load_config
from radio_gs.scannet_constants import (
    NYU40_ID_TO_NAME,
    OPENGAUSSIAN_NYU40_CLASS_SPLITS,
    compute_split_metrics,
)
from radio_gs.scripts.diagnose_scannet_point_logits import (
    _load_teacher_batch,
    _select_feature_paths,
)
from radio_gs.scripts.eval_lerf_grounding import parse_prompt_templates
from radio_gs.scripts.eval_scannet_pointcloud_radio_gs import (
    CLASS_ALIAS_MODES,
    DEFAULT_PREPARED_ROOT,
    FEATURE_RGB_PROJECTION_SEED,
    LOGIT_CALIBRATION_MODES,
    _apply_logit_calibration,
    _default_label_ply,
    _empty_split_diagnostics,
    _finalize_split_diagnostics,
    _fixed_rgb_projection_matrix,
    _format_scene_path,
    _load_or_generate_class_text_embeddings,
    _load_projection,
    _normalize_rgb_values,
    _parse_splits,
    _project_features_to_rgb_values,
    _project_points,
    _read_label_ply,
    _save_feature_rgb_ply,
    _save_language_features_npz,
    _save_prediction_ply,
    _save_split_logits_npz,
    _subsample_points,
    _update_split_diagnostics,
    _write_csv,
)
from radio_gs.scripts.train_feature_field import sample_multiview_radio_targets


def _raw_ids_from_pred_indices(
    pred_indices: np.ndarray | torch.Tensor,
    split_ids: Iterable[int],
) -> np.ndarray:
    """Map contiguous class predictions back to raw NYU40 ids."""
    pred_np = (
        pred_indices.detach().cpu().numpy()
        if isinstance(pred_indices, torch.Tensor)
        else np.asarray(pred_indices)
    ).astype(np.int64, copy=False)
    raw_ids = np.asarray([int(class_id) for class_id in split_ids], dtype=np.int32)
    if pred_np.size == 0:
        return np.empty(pred_np.shape, dtype=np.int32)
    if pred_np.min() < 0 or pred_np.max() >= raw_ids.shape[0]:
        raise ValueError(
            f"Predicted class index out of range for split of size {raw_ids.shape[0]}"
        )
    return raw_ids[pred_np]


def _compute_teacher_split_metrics(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    teacher_valid: np.ndarray,
    split_ids: Iterable[int],
) -> dict:
    """Compute metrics after masking points without any valid teacher view."""
    pred_labels = np.asarray(pred_labels, dtype=np.int32).reshape(-1)
    gt_labels = np.asarray(gt_labels, dtype=np.int32).reshape(-1)
    teacher_valid = np.asarray(teacher_valid, dtype=bool).reshape(-1)
    if pred_labels.shape != gt_labels.shape or pred_labels.shape != teacher_valid.shape:
        raise ValueError(
            "pred_labels, gt_labels, and teacher_valid must have matching shapes; "
            f"got {pred_labels.shape}, {gt_labels.shape}, {teacher_valid.shape}"
        )
    eval_gt = gt_labels.copy()
    eval_gt[~teacher_valid] = 0
    metrics = compute_split_metrics(pred_labels, eval_gt, list(split_ids))
    metrics.update(
        {
            "teacher_valid_points": int(teacher_valid.sum()),
            "teacher_valid_ratio": (
                float(teacher_valid.mean()) if teacher_valid.shape[0] else 0.0
            ),
        }
    )
    return metrics


def _accumulate_multiview_targets(
    config,
    feature_paths: list[Path],
    points_xyz: torch.Tensor,
    *,
    device: torch.device,
    split: str,
    view_chunk_size: int,
    normalize_features: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate frame-wise RADIO features over view chunks for one point chunk."""
    if not feature_paths:
        raise ValueError("feature_paths must not be empty")
    if view_chunk_size <= 0:
        view_chunk_size = len(feature_paths)

    points = points_xyz.to(device=device, dtype=torch.float32)
    target_sum: Optional[torch.Tensor] = None
    view_counts_total = torch.zeros(points.shape[0], device=device, dtype=torch.long)

    for start in range(0, len(feature_paths), view_chunk_size):
        chunk_paths = feature_paths[start : start + view_chunk_size]
        feature_batch, poses_w2c, K = _load_teacher_batch(
            config,
            chunk_paths,
            device,
            split,
        )
        with torch.no_grad():
            targets, valid, view_counts = sample_multiview_radio_targets(
                points,
                feature_batch,
                poses_w2c,
                K,
                normalize_sampled_features=normalize_features,
            )
        if target_sum is None:
            target_sum = torch.zeros(
                points.shape[0],
                targets.shape[1],
                device=device,
                dtype=torch.float32,
            )
        counts_f = view_counts.to(device=device, dtype=torch.float32).unsqueeze(1)
        target_sum += targets.float() * counts_f
        view_counts_total += view_counts.to(device=device, dtype=torch.long)
        del feature_batch, poses_w2c, K, targets, valid

    if target_sum is None:
        target_sum = torch.empty(points.shape[0], 0, device=device, dtype=torch.float32)
    denom = view_counts_total.clamp_min(1).to(dtype=torch.float32).unsqueeze(1)
    targets = target_sum / denom
    valid = view_counts_total > 0
    if targets.numel() > 0:
        targets = targets.clone()
        targets[~valid] = 0.0
    return targets, valid, view_counts_total


def _save_teacher_feature_cache(
    path: str | Path,
    *,
    xyz: np.ndarray,
    labels: np.ndarray,
    features: torch.Tensor,
    valid: torch.Tensor,
    view_counts: torch.Tensor,
    sample_indices: Optional[np.ndarray],
    metadata: dict,
) -> str:
    """Save point-aligned multiview frame-wise RADIO features for training."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xyz_t = torch.as_tensor(np.asarray(xyz, dtype=np.float32), dtype=torch.float32).cpu()
    labels_t = torch.as_tensor(np.asarray(labels, dtype=np.int64), dtype=torch.long).cpu()
    features_t = torch.as_tensor(features).detach().cpu().half()
    valid_t = torch.as_tensor(valid, dtype=torch.bool).detach().cpu()
    counts_t = torch.as_tensor(view_counts, dtype=torch.long).detach().cpu()
    payload = {
        "xyz": xyz_t,
        "labels": labels_t,
        "features": features_t,
        "valid": valid_t,
        "view_counts": counts_t,
        "sample_indices": (
            torch.as_tensor(sample_indices, dtype=torch.long).cpu()
            if sample_indices is not None
            else None
        ),
        "metadata": dict(metadata),
    }
    torch.save(payload, out_path)
    return str(out_path)


def _save_teacher_language_features_npz(
    path: str | Path,
    *,
    xyz: np.ndarray,
    labels: np.ndarray,
    features: torch.Tensor,
    valid: torch.Tensor,
) -> str:
    """Save normalized teacher language features, zeroing points without teacher views."""
    out_path = Path(path)
    visual = features.detach().float().cpu().clone()
    valid_cpu = valid.detach().bool().cpu()
    if visual.shape[0] != valid_cpu.shape[0]:
        raise ValueError(
            f"features/valid length mismatch: {visual.shape[0]} vs {valid_cpu.shape[0]}"
        )
    if visual.numel() > 0:
        visual[~valid_cpu] = 0.0
    _save_language_features_npz(out_path, xyz, labels, visual)
    with np.load(out_path) as data:
        payload = {key: data[key] for key in data.files}
    payload["valid"] = valid_cpu.numpy().astype(bool)
    np.savez_compressed(out_path, **payload)
    return str(out_path)


def _compute_scene_mean_logit_bias(
    *,
    config,
    feature_paths: list[Path],
    xyz: torch.Tensor,
    projection: torch.nn.Module,
    split_text_embeddings: Dict[str, torch.Tensor],
    split_ids: dict[str, list[int]],
    device: torch.device,
    teacher_split: str,
    view_chunk_size: int,
    chunk_size: int,
    normalize_features: bool,
) -> dict[str, Optional[torch.Tensor]]:
    logit_sum_by_split = {
        split: torch.zeros(len(ids), dtype=torch.float64)
        for split, ids in split_ids.items()
    }
    logit_count = 0
    for start in tqdm(range(0, xyz.shape[0], chunk_size), desc="teacher logit calibration"):
        end = min(start + chunk_size, xyz.shape[0])
        with torch.no_grad():
            targets, valid, _ = _accumulate_multiview_targets(
                config,
                feature_paths,
                xyz[start:end],
                device=device,
                split=teacher_split,
                view_chunk_size=view_chunk_size,
                normalize_features=normalize_features,
            )
            if not bool(valid.any()):
                continue
            visual = _project_points(targets[valid].float(), projection)
            for split in split_ids:
                logits = visual @ split_text_embeddings[split].to(device).T
                logit_sum_by_split[split] += logits.detach().double().cpu().sum(dim=0)
            logit_count += int(valid.sum().item())
    if logit_count == 0:
        return {split: None for split in split_ids}
    return {
        split: (values / float(logit_count)).float()
        for split, values in logit_sum_by_split.items()
    }


def evaluate_teacher_scene(
    scene: str,
    config_path: str,
    label_ply: str,
    projection: torch.nn.Module,
    split_text_embeddings: Dict[str, torch.Tensor],
    device: torch.device,
    *,
    split_names: Iterable[str],
    max_views: int = 32,
    teacher_split: str = "val",
    view_chunk_size: int = 8,
    chunk_size: int = 4096,
    normalize_features: bool = False,
    max_points: Optional[int] = None,
    sample_seed: int = 42,
    output_dir: Optional[Path] = None,
    save_ply: bool = False,
    save_logits_npz: bool = False,
    save_feature_rgb_ply: bool = False,
    save_language_features_npz: bool = False,
    teacher_cache_path: Optional[Path] = None,
    feature_rgb_seed: int = FEATURE_RGB_PROJECTION_SEED,
    logit_calibration: str = "none",
    logit_calibration_alpha: float = 1.0,
) -> dict:
    if logit_calibration not in LOGIT_CALIBRATION_MODES:
        raise ValueError(
            f"logit_calibration must be one of: {', '.join(LOGIT_CALIBRATION_MODES)}"
        )
    if save_logits_npz and output_dir is None:
        raise ValueError("output_dir is required when save_logits_npz=True")
    if save_feature_rgb_ply and output_dir is None:
        raise ValueError("output_dir is required when save_feature_rgb_ply=True")
    if save_language_features_npz and output_dir is None:
        raise ValueError("output_dir is required when save_language_features_npz=True")

    config = load_config(config_path)
    feature_paths = _select_feature_paths(config, max_views=max_views, split=teacher_split)
    xyz_np, labels_np = _read_label_ply(label_ply)
    xyz_np, labels_np, sample_indices = _subsample_points(
        xyz_np,
        labels_np,
        max_points=max_points,
        seed=sample_seed,
    )
    xyz = torch.from_numpy(xyz_np).to(device=device, dtype=torch.float32)

    split_ids = {
        split: OPENGAUSSIAN_NYU40_CLASS_SPLITS[split]
        for split in split_names
    }
    pred_by_split = {
        split: np.full(labels_np.shape, -1, dtype=np.int32)
        for split in split_ids
    }
    diagnostics_by_split = {
        split: _empty_split_diagnostics(ids)
        for split, ids in split_ids.items()
    }
    teacher_valid_all = np.zeros(labels_np.shape[0], dtype=bool)
    view_counts_all = np.zeros(labels_np.shape[0], dtype=np.int32)
    feature_rgb_projection: Optional[np.ndarray] = None
    feature_rgb_value_parts: list[np.ndarray] = []
    teacher_feature_parts: list[torch.Tensor] = []
    teacher_valid_parts: list[torch.Tensor] = []
    teacher_view_count_parts: list[torch.Tensor] = []
    teacher_language_feature_parts: list[torch.Tensor] = []

    if logit_calibration == "scene_mean":
        logit_bias_by_split = _compute_scene_mean_logit_bias(
            config=config,
            feature_paths=feature_paths,
            xyz=xyz,
            projection=projection,
            split_text_embeddings=split_text_embeddings,
            split_ids=split_ids,
            device=device,
            teacher_split=teacher_split,
            view_chunk_size=view_chunk_size,
            chunk_size=chunk_size,
            normalize_features=normalize_features,
        )
    else:
        logit_bias_by_split = {split: None for split in split_ids}

    for start in tqdm(range(0, xyz.shape[0], chunk_size), desc=f"{scene} teacher query"):
        end = min(start + chunk_size, xyz.shape[0])
        with torch.no_grad():
            targets, valid, view_counts = _accumulate_multiview_targets(
                config,
                feature_paths,
                xyz[start:end],
                device=device,
                split=teacher_split,
                view_chunk_size=view_chunk_size,
                normalize_features=normalize_features,
            )
            visual = _project_points(targets.float(), projection)
            visual = visual.clone()
            visual[~valid] = 0.0

        if teacher_cache_path is not None:
            teacher_feature_parts.append(targets.detach().cpu())
            teacher_valid_parts.append(valid.detach().cpu())
            teacher_view_count_parts.append(view_counts.detach().cpu())

        valid_np = valid.detach().cpu().numpy().astype(bool)
        teacher_valid_all[start:end] = valid_np
        view_counts_all[start:end] = view_counts.detach().cpu().numpy().astype(np.int32)

        if save_feature_rgb_ply:
            if feature_rgb_projection is None:
                feature_rgb_projection = _fixed_rgb_projection_matrix(
                    int(visual.shape[1]),
                    seed=feature_rgb_seed,
                )
            feature_rgb_value_parts.append(
                _project_features_to_rgb_values(visual, feature_rgb_projection)
            )
        if save_language_features_npz:
            teacher_language_feature_parts.append(visual.detach().float().cpu())

        for split, ids in split_ids.items():
            logits = _apply_logit_calibration(
                visual @ split_text_embeddings[split].to(device).T,
                logit_bias_by_split.get(split),
                alpha=logit_calibration_alpha,
            )
            pred_idx = logits.argmax(dim=-1).detach().cpu().numpy()
            pred_ids = _raw_ids_from_pred_indices(pred_idx, ids)
            pred_by_split[split][start:end] = pred_ids
            if bool(valid.any()):
                _update_split_diagnostics(
                    diagnostics_by_split[split],
                    logits[valid],
                    labels_np[start:end][valid_np],
                    pred_ids[valid_np],
                    save_logits_npz=save_logits_npz,
                )

    scene_results: dict[str, dict] = {}
    for split, pred_labels in pred_by_split.items():
        metrics = _compute_teacher_split_metrics(
            pred_labels=pred_labels,
            gt_labels=labels_np,
            teacher_valid=teacher_valid_all,
            split_ids=split_ids[split],
        )
        metrics.update(_finalize_split_diagnostics(diagnostics_by_split[split], split_ids[split]))
        if save_logits_npz and output_dir is not None:
            metrics["logits_npz"] = _save_split_logits_npz(
                output_dir,
                scene,
                split,
                diagnostics_by_split[split],
            )
        scene_results[split] = metrics
        if save_ply and output_dir is not None:
            _save_prediction_ply(
                output_dir / "visualizations" / scene / f"teacher_pred_split_{split}.ply",
                xyz_np,
                labels_np,
                pred_labels,
            )

    feature_rgb_ply = None
    if save_feature_rgb_ply and output_dir is not None:
        feature_rgb_ply = output_dir / "visualizations" / scene / "teacher_language_feature_rgb.ply"
        feature_rgb_values = (
            np.concatenate(feature_rgb_value_parts, axis=0)
            if feature_rgb_value_parts
            else np.empty((0, 3), dtype=np.float32)
        )
        _save_feature_rgb_ply(
            feature_rgb_ply,
            xyz_np,
            labels_np,
            _normalize_rgb_values(feature_rgb_values),
        )

    language_features_npz = None
    if save_language_features_npz and output_dir is not None:
        language_features_npz = output_dir / "visualizations" / scene / "teacher_language_features.npz"
        _save_teacher_language_features_npz(
            language_features_npz,
            xyz=xyz_np,
            labels=labels_np,
            features=(
                torch.cat(teacher_language_feature_parts, dim=0)
                if teacher_language_feature_parts
                else torch.empty((0, 0), dtype=torch.float32)
            ),
            valid=torch.as_tensor(teacher_valid_all, dtype=torch.bool),
        )

    teacher_cache = None
    if teacher_cache_path is not None:
        teacher_cache = _save_teacher_feature_cache(
            teacher_cache_path,
            xyz=xyz_np,
            labels=labels_np,
            features=(
                torch.cat(teacher_feature_parts, dim=0)
                if teacher_feature_parts
                else torch.empty((0, 0), dtype=torch.float32)
            ),
            valid=(
                torch.cat(teacher_valid_parts, dim=0)
                if teacher_valid_parts
                else torch.empty((0,), dtype=torch.bool)
            ),
            view_counts=(
                torch.cat(teacher_view_count_parts, dim=0)
                if teacher_view_count_parts
                else torch.empty((0,), dtype=torch.long)
            ),
            sample_indices=sample_indices,
            metadata={
                "scene": scene,
                "teacher_split": teacher_split,
                "teacher_max_views": int(max_views),
                "teacher_view_chunk_size": int(view_chunk_size),
                "normalize_teacher_features": bool(normalize_features),
                "feature_projection": "multiview_radio_teacher_1280d",
                "label_ply": str(label_ply),
            },
        )

    return {
        "scene": scene,
        "label_ply": str(label_ply),
        "num_points": int(labels_np.shape[0]),
        "sample_indices": sample_indices.tolist() if sample_indices is not None else None,
        "teacher_split": teacher_split,
        "teacher_max_views": int(max_views),
        "teacher_view_chunk_size": int(view_chunk_size),
        "normalize_teacher_features": bool(normalize_features),
        "teacher_feature_frames": [
            int(path.stem.split("_")[-1]) for path in feature_paths
        ],
        "teacher_valid_points": int(teacher_valid_all.sum()),
        "teacher_valid_ratio": (
            float(teacher_valid_all.mean()) if teacher_valid_all.shape[0] else 0.0
        ),
        "mean_teacher_view_count": (
            float(view_counts_all.mean()) if view_counts_all.shape[0] else 0.0
        ),
        "feature_projection": "multiview_radio_teacher_plus_siglip_summary_head",
        "logit_calibration": {
            "mode": logit_calibration,
            "alpha": float(logit_calibration_alpha),
            "bias_by_split": {
                split: (
                    bias.detach().cpu().tolist()
                    if isinstance(bias, torch.Tensor)
                    else None
                )
                for split, bias in logit_bias_by_split.items()
            },
        },
        "language_feature_rgb_ply": str(feature_rgb_ply) if feature_rgb_ply is not None else None,
        "language_features_npz": (
            str(language_features_npz) if language_features_npz is not None else None
        ),
        "teacher_feature_cache": teacher_cache,
        "splits": scene_results,
    }


def _discover_scenes(prepared_root: Path) -> list[str]:
    return sorted(path.name for path in prepared_root.glob("scene*") if path.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate multiview RADIO teacher ScanNet point-cloud understanding"
    )
    parser.add_argument("--scene", default="scene0000_00", help="Scene id or 'all'")
    parser.add_argument("--prepared_root", default=str(DEFAULT_PREPARED_ROOT))
    parser.add_argument("--config", required=True, help="Config path; may contain {scene}")
    parser.add_argument("--label_ply", default=None, help="Optional label PLY path; may contain {scene}")
    parser.add_argument("--output_dir", default="output/scannet_pointcloud_teacher_eval")
    parser.add_argument("--class_splits", default="19,15,10")
    parser.add_argument("--max_views", type=int, default=32)
    parser.add_argument("--teacher_split", choices=["all", "train", "val"], default="val")
    parser.add_argument("--view_chunk_size", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument(
        "--normalize_teacher_features",
        action="store_true",
        help="L2-normalize per-view 1280d RADIO point features before multiview averaging.",
    )
    parser.add_argument("--max_points", type=int, default=None)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--save_ply", action="store_true")
    parser.add_argument("--save_logits_npz", action="store_true")
    parser.add_argument("--save_feature_rgb_ply", action="store_true")
    parser.add_argument("--save_language_features_npz", action="store_true")
    parser.add_argument(
        "--save_teacher_cache",
        action="store_true",
        help="Save point-aligned multiview frame-wise RADIO features as a training cache.",
    )
    parser.add_argument(
        "--teacher_cache_path",
        default=None,
        help="Optional cache path; may contain {scene}. Defaults under output_dir/teacher_feature_cache/.",
    )
    parser.add_argument("--feature_rgb_seed", type=int, default=FEATURE_RGB_PROJECTION_SEED)
    parser.add_argument("--prompt_templates", default="{query}")
    parser.add_argument("--text_embedding_cache", default=None)
    parser.add_argument("--class_aliases", choices=CLASS_ALIAS_MODES, default="none")
    parser.add_argument("--projection_weights", default=DEFAULT_SIGLIP2_PROJECTION_WEIGHTS)
    parser.add_argument("--summary_head_weights", default="checkpoints/siglip2_summary_head.pth")
    parser.add_argument("--radio_checkpoint", default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar")
    parser.add_argument("--use_summary_head", action="store_true", default=True)
    parser.add_argument("--no_summary_head", dest="use_summary_head", action="store_false")
    parser.add_argument("--logit_calibration", choices=LOGIT_CALIBRATION_MODES, default="none")
    parser.add_argument("--logit_calibration_alpha", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    prepared_root = Path(args.prepared_root)
    scenes = _discover_scenes(prepared_root) if args.scene == "all" else [args.scene]
    if not scenes:
        raise FileNotFoundError(f"No prepared scenes found under {prepared_root}")
    split_names = _parse_splits(args.class_splits)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    projection = _load_projection(args, device)
    prompt_templates = parse_prompt_templates(args.prompt_templates)
    split_text_embeddings: Dict[str, torch.Tensor] = {}
    for split in split_names:
        class_names = [
            NYU40_ID_TO_NAME[class_id]
            for class_id in OPENGAUSSIAN_NYU40_CLASS_SPLITS[split]
        ]
        cache_path = None
        if args.text_embedding_cache:
            base = Path(args.text_embedding_cache)
            alias_suffix = (
                f"_aliases_{args.class_aliases}" if args.class_aliases != "none" else ""
            )
            cache_path = str(base.with_name(f"{base.stem}_split{split}{alias_suffix}.pt"))
        split_text_embeddings[split] = _load_or_generate_class_text_embeddings(
            class_names,
            device,
            cache_path=cache_path,
            prompt_templates=prompt_templates,
            class_aliases=args.class_aliases,
        )

    print("=" * 72)
    print("  ScanNet Multiview RADIO Teacher Point-Cloud Evaluation")
    print("=" * 72)
    print(f"  Scenes:      {', '.join(scenes)}")
    print(f"  Splits:      {', '.join(split_names)}")
    print(f"  Teacher:     split={args.teacher_split} max_views={args.max_views}")
    print(f"  View chunks: {args.view_chunk_size}")
    print(f"  Normalize:   {args.normalize_teacher_features}")
    print(f"  Aliases:     {args.class_aliases}")
    print(
        f"  Calibration: {args.logit_calibration}"
        + (
            f" (alpha={args.logit_calibration_alpha:g})"
            if args.logit_calibration != "none"
            else ""
        )
    )
    print(f"  Chunk size:  {args.chunk_size}")
    print(f"  Max points:  {args.max_points or 'all'}")
    print()

    all_results: dict[str, dict] = {}
    for scene in scenes:
        label_ply = _format_scene_path(args.label_ply, scene) or _default_label_ply(prepared_root, scene)
        teacher_cache_path = None
        if args.save_teacher_cache:
            formatted_cache = _format_scene_path(args.teacher_cache_path, scene)
            teacher_cache_path = (
                Path(formatted_cache)
                if formatted_cache
                else output_dir / "teacher_feature_cache" / f"{scene}_radio_teacher_features.pt"
            )
        result = evaluate_teacher_scene(
            scene=scene,
            config_path=_format_scene_path(args.config, scene),
            label_ply=label_ply,
            projection=projection,
            split_text_embeddings=split_text_embeddings,
            device=device,
            split_names=split_names,
            max_views=args.max_views,
            teacher_split=args.teacher_split,
            view_chunk_size=args.view_chunk_size,
            chunk_size=args.chunk_size,
            normalize_features=args.normalize_teacher_features,
            max_points=args.max_points,
            sample_seed=args.sample_seed,
            output_dir=output_dir,
            save_ply=args.save_ply,
            save_logits_npz=args.save_logits_npz,
            save_feature_rgb_ply=args.save_feature_rgb_ply,
            save_language_features_npz=args.save_language_features_npz,
            teacher_cache_path=teacher_cache_path,
            feature_rgb_seed=args.feature_rgb_seed,
            logit_calibration=args.logit_calibration,
            logit_calibration_alpha=args.logit_calibration_alpha,
        )
        all_results[scene] = result
        for split in split_names:
            metrics = result["splits"][split]
            print(
                f"{scene} split{split}: "
                f"mIoU={metrics['miou']:.4f} mAcc={metrics['macc']:.4f} "
                f"valid={metrics['num_valid']} "
                f"teacher_valid={result['teacher_valid_ratio']:.3f}"
            )

    macro = {}
    for split in split_names:
        macro[split] = {
            "miou": float(np.mean([res["splits"][split]["miou"] for res in all_results.values()])),
            "macc": float(np.mean([res["splits"][split]["macc"] for res in all_results.values()])),
        }

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": {key: str(value) for key, value in vars(args).items()},
        "prompt_templates": prompt_templates,
        "class_aliases": args.class_aliases,
        "macro": macro,
        "scenes": all_results,
    }
    json_path = output_dir / "scannet_pointcloud_radio_teacher_results.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_csv(output_dir / "scannet_pointcloud_radio_teacher_results.csv", all_results)

    print("\nMacro:")
    for split in split_names:
        print(
            f"  split{split}: "
            f"mIoU={macro[split]['miou']:.4f} mAcc={macro[split]['macc']:.4f}"
        )
    print(f"\nSaved JSON: {json_path}")


if __name__ == "__main__":
    main()
