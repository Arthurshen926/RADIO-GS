#!/usr/bin/env python3
"""Evaluate RADIO-GS on ScanNet in the Gaussian domain used by VALA.

This is intentionally separate from ``eval_scannet_pointcloud_radio_gs``.
The latter queries the learned field at annotated mesh vertices and optionally
calibrates/propagates logits on that mesh.  VALA instead:

1. predicts one open-vocabulary class per optimized Gaussian center;
2. propagates mesh labels to those centers with anisotropic Gaussian-density
   voting; and
3. computes per-scene mIoU/mAcc with opacity-times-volume weights.

The released VALA evaluator forms a radius-culled Euclidean top-k candidate
set and then performs Mahalanobis-density voting.  The implementation below is
an equivalent KD-tree acceleration of that operation; it also records the
row-aligned OpenGaussian metric as an audit-only comparator because the local
ScanNet RGB Gaussians happen to preserve the input mesh vertex count/order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from tqdm import tqdm

from radio_gs.artifact_paths import DEFAULT_SIGLIP2_PROJECTION_WEIGHTS
from radio_gs.config import load_config
from radio_gs.scannet_constants import (
    NYU40_ID_TO_NAME,
    OPENGAUSSIAN_NYU40_CLASS_SPLITS,
)
from radio_gs.scripts.eval_lerf_grounding import parse_prompt_templates
from radio_gs.querying.unified_query import cosine_bank_torch
from radio_gs.scripts.eval_scannet_pointcloud_radio_gs import (
    _build_hybrid_model,
    _decode_gaussian_indices_1280,
    _default_label_ply,
    _format_scene_path,
    _load_or_generate_class_text_embeddings,
    _load_projection,
    _parse_scene_list,
    _parse_splits,
    _read_label_ply,
)


DEFAULT_SCENES = (
    "scene0000_00",
    "scene0062_00",
    "scene0070_00",
    "scene0097_00",
    "scene0140_00",
    "scene0347_00",
    "scene0400_00",
    "scene0590_00",
)


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert GraphDECO/VALA ``(w,x,y,z)`` quaternions to rotation matrices."""
    q = np.asarray(quaternion, dtype=np.float32)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = (q[:, idx] for idx in range(4))
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(-1, 3, 3)


def assign_vala_pseudo_labels(
    gaussian_xyz: np.ndarray,
    gaussian_scales: np.ndarray,
    gaussian_rotations: np.ndarray,
    point_xyz: np.ndarray,
    point_labels: np.ndarray,
    *,
    radius_factor: float = 5.0,
    candidate_k: int = 1000,
    fallback_k: int = 1,
    class_balance: bool = True,
    chunk_size: int = 512,
) -> tuple[np.ndarray, dict[str, object]]:
    """Assign raw NYU40 labels with VALA's anisotropic density vote.

    KD-tree top-k is equivalent to the released evaluator's dense ``cdist`` +
    radius mask + ``topk`` operation.  ``fallback_k`` implements the fallback
    described in the paper for an empty radius set.  In the prepared eight
    scenes the optimized Gaussian closest point is always inside the radius,
    so this setting does not affect the result.
    """
    gaussian_xyz = np.asarray(gaussian_xyz, dtype=np.float32)
    gaussian_scales = np.asarray(gaussian_scales, dtype=np.float32)
    gaussian_rotations = np.asarray(gaussian_rotations, dtype=np.float32)
    point_xyz = np.asarray(point_xyz, dtype=np.float32)
    point_labels = np.asarray(point_labels, dtype=np.int32)
    if gaussian_xyz.ndim != 2 or gaussian_xyz.shape[1] != 3:
        raise ValueError(f"gaussian_xyz must be [N,3], got {gaussian_xyz.shape}")
    if gaussian_scales.shape != gaussian_xyz.shape:
        raise ValueError("gaussian_scales must match gaussian_xyz")
    if gaussian_rotations.shape != (gaussian_xyz.shape[0], 4):
        raise ValueError("gaussian_rotations must be [N,4]")
    if point_xyz.ndim != 2 or point_xyz.shape[1] != 3:
        raise ValueError(f"point_xyz must be [Q,3], got {point_xyz.shape}")
    if point_labels.shape != (point_xyz.shape[0],):
        raise ValueError("point_labels must align with point_xyz")
    if candidate_k <= 0 or chunk_size <= 0:
        raise ValueError("candidate_k and chunk_size must be positive")

    num_gaussians = gaussian_xyz.shape[0]
    query_k = min(int(candidate_k), point_xyz.shape[0])
    num_labels = int(max(int(point_labels.max(initial=0)) + 1, 1))
    radii = float(radius_factor) * gaussian_scales.max(axis=1)
    rotations = _quaternion_to_rotation_matrix(gaussian_rotations)
    inv_scale = 1.0 / np.maximum(gaussian_scales, 1e-6)
    inv_sqrt_cov = np.einsum(
        "nij,njk,nlk->nil",
        rotations,
        np.eye(3, dtype=np.float32)[None, ...] * inv_scale[:, :, None],
        rotations,
        optimize=True,
    )
    tree = cKDTree(point_xyz)
    labels_out = np.zeros(num_gaussians, dtype=np.int32)
    candidate_counts = np.zeros(num_gaussians, dtype=np.int32)
    empty_before_fallback = np.zeros(num_gaussians, dtype=bool)

    for start in tqdm(
        range(0, num_gaussians, int(chunk_size)),
        desc="VALA pseudo-GT",
    ):
        end = min(start + int(chunk_size), num_gaussians)
        centers = gaussian_xyz[start:end]
        try:
            distances, indices = tree.query(centers, k=query_k, workers=-1)
        except TypeError:  # scipy<1.6
            distances, indices = tree.query(centers, k=query_k)
        distances = np.asarray(distances)
        indices = np.asarray(indices, dtype=np.int64)
        if distances.ndim == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        valid = distances <= radii[start:end, None]
        empty = ~valid.any(axis=1)
        empty_before_fallback[start:end] = empty
        if fallback_k > 0 and empty.any():
            valid[empty, : min(int(fallback_k), query_k)] = True
        candidate_counts[start:end] = valid.sum(axis=1)

        selected_points = point_xyz[indices]
        selected_labels = point_labels[indices]
        delta = selected_points - centers[:, None, :]
        whitened = np.einsum(
            "gij,gkj->gki", inv_sqrt_cov[start:end], delta, optimize=True
        )
        distance_sq = np.einsum("gki,gki->gk", whitened, whitened, optimize=True)
        # Subtracting one per-Gaussian constant preserves the class argmax but
        # prevents every fallback density from underflowing to zero.
        min_valid_distance_sq = np.where(valid, distance_sq, np.inf).min(axis=1)
        min_valid_distance_sq[~valid.any(axis=1)] = 0.0
        stable_distance_sq = distance_sq - min_valid_distance_sq[:, None]
        weights = np.zeros_like(distance_sq, dtype=np.float32)
        weights[valid] = np.exp(-0.5 * stable_distance_sq[valid]).astype(np.float32)

        rows = np.broadcast_to(
            np.arange(end - start, dtype=np.int64)[:, None], selected_labels.shape
        )
        scores = np.zeros((end - start, num_labels), dtype=np.float32)
        np.add.at(scores, (rows.ravel(), selected_labels.ravel()), weights.ravel())
        if class_balance:
            counts = np.zeros_like(scores)
            np.add.at(
                counts,
                (rows.ravel(), selected_labels.ravel()),
                valid.astype(np.float32).ravel(),
            )
            scores /= np.maximum(counts, 1.0)
        labels_out[start:end] = scores.argmax(axis=1).astype(np.int32)

    quantiles = np.quantile(candidate_counts, [0.0, 0.1, 0.5, 0.9, 0.99, 1.0])
    stats: dict[str, object] = {
        "num_gaussians": int(num_gaussians),
        "num_label_points": int(point_xyz.shape[0]),
        "radius_factor": float(radius_factor),
        "candidate_k": int(candidate_k),
        "fallback_k": int(fallback_k),
        "class_balance": bool(class_balance),
        "empty_before_fallback": int(empty_before_fallback.sum()),
        "candidate_count_quantiles": {
            key: float(value)
            for key, value in zip(("min", "p10", "p50", "p90", "p99", "max"), quantiles)
        },
        "num_at_candidate_cap": int((candidate_counts == query_k).sum()),
    }
    return labels_out, stats


def volume_weighted_split_metrics(
    pseudo_gt_raw: np.ndarray,
    pred_raw: np.ndarray,
    significance: np.ndarray,
    split_ids: Iterable[int],
) -> dict[str, object]:
    """VALA mIoU/mAcc for one scene and one OpenGaussian class split."""
    gt = np.asarray(pseudo_gt_raw, dtype=np.int32)
    pred = np.asarray(pred_raw, dtype=np.int32).copy()
    weights = np.asarray(significance, dtype=np.float64).reshape(-1)
    if gt.shape != pred.shape or gt.shape != weights.shape:
        raise ValueError("gt, pred, and significance must have the same length")
    split_ids = [int(value) for value in split_ids]
    valid = np.isin(gt, split_ids)
    pred[~valid] = 0
    per_class: dict[str, object] = {}
    ious: list[float] = []
    accuracies: list[float] = []
    for class_id in split_ids:
        gt_mask = gt == class_id
        if not gt_mask.any():
            continue
        pred_mask = pred == class_id
        intersection = float(weights[gt_mask & pred_mask].sum())
        union = float(weights[gt_mask | pred_mask].sum())
        total = float(weights[gt_mask].sum())
        iou = intersection / union if union > 0 else 0.0
        accuracy = intersection / total if total > 0 else 0.0
        per_class[str(class_id)] = {
            "name": NYU40_ID_TO_NAME.get(class_id, str(class_id)),
            "iou": iou,
            "acc": accuracy,
            "weighted_gt": total,
        }
        ious.append(iou)
        accuracies.append(accuracy)
    return {
        "miou": float(np.mean(ious)) if ious else 0.0,
        "macc": float(np.mean(accuracies)) if accuracies else 0.0,
        "num_valid_gaussians": int(valid.sum()),
        "num_present_classes": len(ious),
        "per_class": per_class,
    }


def _load_or_build_pseudo_gt(
    cache_path: Path,
    gaussian_xyz: np.ndarray,
    gaussian_scales: np.ndarray,
    gaussian_rotations: np.ndarray,
    point_xyz: np.ndarray,
    point_labels: np.ndarray,
    *,
    radius_factor: float,
    candidate_k: int,
    fallback_k: int,
    class_balance: bool,
    chunk_size: int,
    force: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    geometry_hash = _array_sha256(
        np.concatenate([gaussian_xyz, gaussian_scales, gaussian_rotations], axis=1)
    )
    label_hash = _array_sha256(np.concatenate([point_xyz, point_labels[:, None]], axis=1))
    settings = {
        "radius_factor": float(radius_factor),
        "candidate_k": int(candidate_k),
        "fallback_k": int(fallback_k),
        "class_balance": bool(class_balance),
        "geometry_sha256": geometry_hash,
        "label_cloud_sha256": label_hash,
    }
    if cache_path.exists() and not force:
        cached = np.load(cache_path, allow_pickle=False)
        cached_settings = json.loads(str(cached["settings_json"].item()))
        if cached_settings == settings:
            return np.asarray(cached["pseudo_labels"], dtype=np.int32), json.loads(
                str(cached["stats_json"].item())
            )
    pseudo_labels, stats = assign_vala_pseudo_labels(
        gaussian_xyz,
        gaussian_scales,
        gaussian_rotations,
        point_xyz,
        point_labels,
        radius_factor=radius_factor,
        candidate_k=candidate_k,
        fallback_k=fallback_k,
        class_balance=class_balance,
        chunk_size=chunk_size,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        pseudo_labels=pseudo_labels,
        settings_json=np.asarray(json.dumps(settings, sort_keys=True)),
        stats_json=np.asarray(json.dumps(stats, sort_keys=True)),
    )
    return pseudo_labels, stats


def _predict_gaussian_labels(
    model: torch.nn.Module,
    codec: torch.nn.Module,
    projection: torch.nn.Module,
    split_text_embeddings: dict[str, torch.Tensor],
    split_names: list[str],
    *,
    device: torch.device,
    chunk_size: int,
    compact_feature_key: str,
) -> dict[str, np.ndarray]:
    predictions = {
        split: np.empty(model.num_gaussians, dtype=np.int32) for split in split_names
    }
    for start in tqdm(
        range(0, model.num_gaussians, int(chunk_size)),
        desc="Gaussian text classification",
    ):
        end = min(start + int(chunk_size), model.num_gaussians)
        indices = torch.arange(start, end, device=device, dtype=torch.long)
        with torch.no_grad():
            decoded = _decode_gaussian_indices_1280(
                model,
                codec,
                indices,
                points_xyz=None,
                compact_feature_key=compact_feature_key,
            )
            visual = F.normalize(projection(decoded.unsqueeze(0)).squeeze(0).float(), dim=-1)
            for split in split_names:
                class_ids = np.asarray(
                    OPENGAUSSIAN_NYU40_CLASS_SPLITS[split], dtype=np.int32
                )
                logits = cosine_bank_torch(
                    visual, split_text_embeddings[split].to(device)
                )
                predictions[split][start:end] = class_ids[
                    logits.argmax(dim=-1).cpu().numpy()
                ]
    return predictions


def _scene_macro(scene_results: dict[str, dict[str, object]], protocol: str) -> dict[str, dict[str, float]]:
    macro: dict[str, dict[str, float]] = {}
    for split in ("19", "15", "10"):
        rows = [result[protocol][split] for result in scene_results.values()]
        macro[split] = {
            "miou": float(np.mean([float(row["miou"]) for row in rows])),
            "macc": float(np.mean([float(row["macc"]) for row in rows])),
        }
    return macro


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--prepared_root", default="dataset/scannet_og")
    parser.add_argument("--config", required=True, help="May contain {scene}")
    parser.add_argument("--checkpoint", required=True, help="May contain {scene}")
    parser.add_argument("--label_ply", default=None, help="May contain {scene}")
    parser.add_argument(
        "--output_dir", default="output/scannet_pointcloud_eval/vala_gaussian_protocol"
    )
    parser.add_argument("--class_splits", default="19,15,10")
    parser.add_argument("--feature_chunk_size", type=int, default=8192)
    parser.add_argument("--pseudo_chunk_size", type=int, default=512)
    parser.add_argument("--radius_factor", type=float, default=5.0)
    parser.add_argument("--candidate_k", type=int, default=1000)
    parser.add_argument("--fallback_k", type=int, default=1)
    parser.add_argument("--no_class_balance", action="store_true")
    parser.add_argument("--force_pseudo_gt", action="store_true")
    parser.add_argument("--row_opacity_threshold", type=float, default=0.1)
    parser.add_argument("--compact_feature_key", default="features")
    parser.add_argument("--prompt_templates", default="{query}")
    parser.add_argument("--text_embedding_cache", default=None)
    parser.add_argument("--text_encoder", choices=("siglip2", "openclip"), default="siglip2")
    parser.add_argument("--openclip_model", default="ViT-B-16")
    parser.add_argument("--openclip_pretrained", default="laion2b_s34b_b88k")
    parser.add_argument("--class_aliases", default="none")
    parser.add_argument("--projection_weights", default=DEFAULT_SIGLIP2_PROJECTION_WEIGHTS)
    parser.add_argument("--summary_head_weights", default="checkpoints/siglip2_summary_head.pth")
    parser.add_argument(
        "--radio_checkpoint",
        default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar",
    )
    parser.add_argument("--use_summary_head", action="store_true", default=True)
    parser.add_argument("--no_summary_head", dest="use_summary_head", action="store_false")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    scenes = _parse_scene_list(args.scene_list) or []
    split_names = _parse_splits(args.class_splits)
    if split_names != ["19", "15", "10"]:
        raise ValueError("This audit currently requires class_splits=19,15,10")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    projection = _load_projection(args, device)
    prompt_templates = parse_prompt_templates(args.prompt_templates)
    split_text_embeddings: dict[str, torch.Tensor] = {}
    for split in split_names:
        class_names = [
            NYU40_ID_TO_NAME[class_id]
            for class_id in OPENGAUSSIAN_NYU40_CLASS_SPLITS[split]
        ]
        cache_path = None
        if args.text_embedding_cache:
            base = Path(args.text_embedding_cache)
            cache_path = str(base.with_name(f"{base.stem}_split{split}.pt"))
        split_text_embeddings[split] = _load_or_generate_class_text_embeddings(
            class_names,
            device,
            cache_path=cache_path,
            prompt_templates=prompt_templates,
            class_aliases=args.class_aliases,
            text_encoder=args.text_encoder,
            openclip_model=args.openclip_model,
            openclip_pretrained=args.openclip_pretrained,
        )

    scene_results: dict[str, dict[str, object]] = {}
    prepared_root = Path(args.prepared_root)
    for scene in scenes:
        print(f"\n=== {scene} ===", flush=True)
        config_path = _format_scene_path(args.config, scene)
        checkpoint_path = _format_scene_path(args.checkpoint, scene)
        config = load_config(config_path)
        model, codec = _build_hybrid_model(config, checkpoint_path, device)
        gaussian_xyz = model.get_xyz().detach().float().cpu().numpy()
        gaussian_scales = model.get_scaling().detach().float().cpu().numpy()
        gaussian_rotations = model.get_rotation().detach().float().cpu().numpy()
        gaussian_opacity = model.get_opacity().detach().float().cpu().numpy().reshape(-1)
        label_ply = _format_scene_path(args.label_ply, scene) or _default_label_ply(
            prepared_root, scene
        )
        point_xyz, point_labels = _read_label_ply(label_ply)
        pseudo_labels, pseudo_stats = _load_or_build_pseudo_gt(
            output_dir / "pseudo_gt" / f"{scene}.npz",
            gaussian_xyz,
            gaussian_scales,
            gaussian_rotations,
            point_xyz,
            point_labels,
            radius_factor=args.radius_factor,
            candidate_k=args.candidate_k,
            fallback_k=args.fallback_k,
            class_balance=not args.no_class_balance,
            chunk_size=args.pseudo_chunk_size,
            force=args.force_pseudo_gt,
        )
        predictions = _predict_gaussian_labels(
            model,
            codec,
            projection,
            split_text_embeddings,
            split_names,
            device=device,
            chunk_size=args.feature_chunk_size,
            compact_feature_key=args.compact_feature_key,
        )
        significance = gaussian_scales.prod(axis=1) * gaussian_opacity
        if point_labels.shape[0] != gaussian_xyz.shape[0]:
            row_labels = np.zeros_like(pseudo_labels)
            row_available = False
            row_distance = None
        else:
            row_labels = point_labels
            row_available = True
            row_distance = np.linalg.norm(gaussian_xyz - point_xyz, axis=1)
        row_weights = np.ones_like(significance)
        row_labels_filtered = row_labels.copy()
        row_labels_filtered[gaussian_opacity < float(args.row_opacity_threshold)] = 0
        vala_metrics = {}
        row_metrics = {}
        for split in split_names:
            split_ids = OPENGAUSSIAN_NYU40_CLASS_SPLITS[split]
            vala_metrics[split] = volume_weighted_split_metrics(
                pseudo_labels, predictions[split], significance, split_ids
            )
            row_metrics[split] = volume_weighted_split_metrics(
                row_labels_filtered, predictions[split], row_weights, split_ids
            )
            print(
                f"split{split}: VALA-pseudo {vala_metrics[split]['miou']:.4f}/"
                f"{vala_metrics[split]['macc']:.4f}; row {row_metrics[split]['miou']:.4f}/"
                f"{row_metrics[split]['macc']:.4f}",
                flush=True,
            )
        prediction_path = output_dir / "predictions" / f"{scene}.npz"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            prediction_path,
            gaussian_xyz=gaussian_xyz,
            pseudo_labels=pseudo_labels,
            row_labels=row_labels,
            significance=significance,
            **{f"pred_split_{split}": predictions[split] for split in split_names},
        )
        scene_results[scene] = {
            "num_gaussians": int(gaussian_xyz.shape[0]),
            "num_label_points": int(point_xyz.shape[0]),
            "label_ply": str(label_ply),
            "pseudo_gt": pseudo_stats,
            "row_alignment": {
                "available": row_available,
                "mean_center_displacement": (
                    float(row_distance.mean()) if row_distance is not None else None
                ),
                "p95_center_displacement": (
                    float(np.quantile(row_distance, 0.95)) if row_distance is not None else None
                ),
                "pseudo_row_label_agreement": (
                    float((pseudo_labels == row_labels).mean()) if row_available else None
                ),
            },
            "vala_pseudo_volume": vala_metrics,
            "opengaussian_row_unweighted": row_metrics,
            "prediction_npz": str(prediction_path),
        }
        del model, codec
        torch.cuda.empty_cache()

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": {
            "prediction_domain": "optimized Gaussian centers",
            "prediction_postprocess": "none",
            "test_scene_calibration": "none",
            "pseudo_gt": "VALA Mahalanobis-density voting",
            "metric_weights": "opacity * sx * sy * sz",
            "scene_aggregation": "unweighted scene macro",
            "text_encoder": args.text_encoder,
            "prompt_templates": prompt_templates,
        },
        "args": {key: str(value) for key, value in vars(args).items()},
        "macro": {
            "vala_pseudo_volume": _scene_macro(scene_results, "vala_pseudo_volume"),
            "opengaussian_row_unweighted": _scene_macro(
                scene_results, "opengaussian_row_unweighted"
            ),
        },
        "scenes": scene_results,
    }
    report_path = output_dir / "scannet_vala_gaussian_protocol_results.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\nMacro:")
    print(json.dumps(report["macro"], indent=2))
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
