#!/usr/bin/env python3
"""Diagnose ScanNet point-level text logits for RADIO-GS direct queries.

The script samples labeled ScanNet points, compares direct RADIO-GS point-query
classification against a multi-view teacher RADIO target sampled from extracted
2-D features, and writes per-point evidence for failure analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from radio_gs.config import load_config
from radio_gs.data.benchmark_paths import (
    extract_feature_frame_index,
    list_feature_paths,
    load_frame_id_list,
    load_w2c_from_pose_file,
)
from radio_gs.scannet_constants import (
    NYU40_ID_TO_NAME,
    OPENGAUSSIAN_NYU40_CLASS_SPLITS,
    compute_split_metrics,
)
from radio_gs.scripts.eval_lerf_grounding import (
    load_or_generate_prompt_ensemble_embeddings,
    parse_prompt_templates,
)
from radio_gs.scripts.eval_scannet_pointcloud_radio_gs import (
    _build_hybrid_model,
    _default_label_ply,
    _load_projection,
    _project_points,
    _read_label_ply,
)
from radio_gs.scripts.train_feature_field import sample_multiview_radio_targets


def _sample_label_indices(
    labels: np.ndarray,
    split_ids: Iterable[int],
    max_points: Optional[int],
    seed: int,
) -> np.ndarray:
    """Sample roughly class-balanced indices for labels inside ``split_ids``."""
    split_set = {int(class_id) for class_id in split_ids}
    valid_indices = np.flatnonzero(np.isin(labels, list(split_set)))
    if max_points is None or max_points <= 0 or valid_indices.shape[0] <= max_points:
        return valid_indices.astype(np.int64, copy=False)

    rng = np.random.default_rng(seed)
    classes = [class_id for class_id in sorted(split_set) if np.any(labels == class_id)]
    if not classes:
        return np.empty(0, dtype=np.int64)

    per_class = max(1, int(np.ceil(max_points / len(classes))))
    chosen: list[np.ndarray] = []
    for class_id in classes:
        class_indices = np.flatnonzero(labels == class_id)
        take = min(per_class, class_indices.shape[0])
        chosen.append(rng.choice(class_indices, size=take, replace=False))
    indices = np.concatenate(chosen, axis=0)
    if indices.shape[0] > max_points:
        indices = rng.choice(indices, size=max_points, replace=False)
    return np.sort(indices.astype(np.int64, copy=False))


def _topk_names(scores: np.ndarray, class_names: list[str], k: int = 3) -> list[dict[str, float | str]]:
    order = np.argsort(scores)[::-1][: max(1, int(k))]
    return [
        {"name": class_names[int(idx)], "score": round(float(scores[int(idx)]), 6)}
        for idx in order
    ]


def _select_feature_paths(config, max_views: int, split: str) -> list[Path]:
    frame_ids: Optional[list[int]] = None
    if split == "train":
        frame_ids = load_frame_id_list(getattr(config, "train_frame_ids_path", ""))
    elif split == "val":
        frame_ids = load_frame_id_list(getattr(config, "val_frame_ids_path", ""))
    paths = list_feature_paths(getattr(config, "feature_dir", ""), frame_ids=frame_ids)
    if not paths:
        raise FileNotFoundError(f"No RADIO feature maps found in {getattr(config, 'feature_dir', '')}")
    if max_views > 0 and len(paths) > max_views:
        positions = np.linspace(0, len(paths) - 1, num=max_views)
        paths = [paths[int(round(pos))] for pos in positions]
    return paths


def _load_w2c_for_feature_paths(config, feature_paths: list[Path], split: str) -> np.ndarray:
    frame_ids = [extract_feature_frame_index(path) for path in feature_paths]
    pose_file = getattr(config, "pose_file", "") or getattr(config, "val_pose_file", "")
    if not pose_file:
        raise ValueError("Config must define pose_file or val_pose_file for teacher diagnostics")
    try:
        return load_w2c_from_pose_file(pose_file, frame_ids)
    except IndexError:
        raw = np.loadtxt(str(pose_file)).reshape(-1, 4, 4).astype(np.float32)
        all_paths = _select_feature_paths(config, max_views=0, split=split)
        if len(all_paths) != len(raw):
            raise
        frame_to_order = {
            extract_feature_frame_index(path): idx
            for idx, path in enumerate(all_paths)
        }
        missing = [frame_id for frame_id in frame_ids if frame_id not in frame_to_order]
        if missing:
            raise IndexError(f"Pose-order fallback missing frame ids: {missing}") from None
        w2c = np.linalg.inv(raw)
        return np.stack([w2c[frame_to_order[frame_id]] for frame_id in frame_ids], axis=0)


def _load_teacher_batch(
    config,
    feature_paths: list[Path],
    device: torch.device,
    split: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    poses = torch.from_numpy(_load_w2c_for_feature_paths(config, feature_paths, split)).to(device=device)

    features = []
    for path in feature_paths:
        feat = torch.load(path, map_location="cpu")
        if feat.dim() == 3:
            feat = feat.unsqueeze(0)
        if feat.dim() != 4 or feat.shape[0] != 1:
            raise ValueError(f"Expected feature map [C,H,W] or [1,C,H,W], got {tuple(feat.shape)} at {path}")
        features.append(feat[0])
    feature_batch = torch.stack(features, dim=0).to(device=device)

    _, _, feature_height, feature_width = feature_batch.shape
    image_height = float(getattr(config, "image_height", feature_height))
    image_width = float(getattr(config, "image_width", feature_width))
    scale_x = float(feature_width) / max(image_width, 1.0)
    scale_y = float(feature_height) / max(image_height, 1.0)
    K = torch.tensor(
        [
            [float(getattr(config, "fx", 1.0)) * scale_x, 0.0, float(getattr(config, "cx", 0.0)) * scale_x],
            [0.0, float(getattr(config, "fy", 1.0)) * scale_y, float(getattr(config, "cy", 0.0)) * scale_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    return feature_batch, poses.float(), K


def _decode_compact_1280(codec, compact: torch.Tensor) -> torch.Tensor:
    if hasattr(codec, "decode_points"):
        return codec.decode_points(compact.float())
    compact_map = compact.T.reshape(1, compact.shape[1], compact.shape[0], 1)
    decoded = codec.decode(compact_map.float())
    return decoded.squeeze(0).squeeze(-1).T.contiguous()


def _query_model_visuals(
    model,
    codec,
    projection: torch.nn.Module,
    points: torch.Tensor,
    *,
    k: int,
    chunk_size: int,
    query_mode: str = "knn",
    gaussian_indices: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if query_mode not in {"knn", "gaussian_index"}:
        raise ValueError("query_mode must be one of: knn, gaussian_index")
    if query_mode == "gaussian_index":
        if gaussian_indices is None:
            raise ValueError("gaussian_indices is required when query_mode='gaussian_index'")
        gaussian_indices = gaussian_indices.to(device=points.device, dtype=torch.long).reshape(-1)
        if gaussian_indices.shape[0] != points.shape[0]:
            raise ValueError(
                "gaussian_indices must have the same row count as points, got "
                f"{gaussian_indices.shape[0]} vs {points.shape[0]}"
            )
    visual_parts: list[torch.Tensor] = []
    nearest_index_parts: list[torch.Tensor] = []
    nearest_distance_parts: list[torch.Tensor] = []
    xyz = model.get_xyz().to(device=points.device, dtype=torch.float32)
    for start in range(0, points.shape[0], chunk_size):
        end = min(start + chunk_size, points.shape[0])
        chunk = points[start:end]
        with torch.no_grad():
            if query_mode == "gaussian_index":
                assert gaussian_indices is not None
                aux = model.query_gaussian_points(
                    gaussian_indices[start:end],
                    return_aux=True,
                )
            else:
                aux = model.query_compact_points(chunk, k=k, return_aux=True)
            compact = aux["features"]
            decoded = _decode_compact_1280(codec, compact)
            visual_parts.append(_project_points(decoded, projection))
            nn_idx = aux["gaussian_indices"].detach()
            if nn_idx.dim() > 1:
                nn_idx = nn_idx[:, 0]
            nearest_index_parts.append(nn_idx)
            nearest_distance_parts.append(torch.linalg.norm(chunk - xyz[nn_idx], dim=-1).detach())
    return (
        torch.cat(visual_parts, dim=0),
        torch.cat(nearest_index_parts, dim=0),
        torch.cat(nearest_distance_parts, dim=0),
    )


def _classify_logits(visual: torch.Tensor, text_embeddings: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    logits = (visual @ text_embeddings.T).detach().float().cpu().numpy()
    pred_idx = logits.argmax(axis=1)
    return logits, pred_idx


def run_diagnostics(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    split_ids = OPENGAUSSIAN_NYU40_CLASS_SPLITS[args.class_split]
    class_names = [NYU40_ID_TO_NAME[class_id] for class_id in split_ids]
    config = load_config(args.config)
    label_ply = args.label_ply or _default_label_ply(Path(args.prepared_root), args.scene)

    xyz_np, labels_np = _read_label_ply(label_ply)
    sample_indices = _sample_label_indices(labels_np, split_ids, args.max_points, args.sample_seed)
    sampled_xyz_np = xyz_np[sample_indices]
    sampled_labels_np = labels_np[sample_indices]
    points = torch.from_numpy(sampled_xyz_np).to(device=device, dtype=torch.float32)

    model, codec = _build_hybrid_model(config, args.checkpoint, device)
    projection = _load_projection(args, device)
    prompt_templates = parse_prompt_templates(args.prompt_templates)
    cache_path = None
    if args.text_embedding_cache:
        base = Path(args.text_embedding_cache)
        cache_path = str(base.with_name(f"{base.stem}_split{args.class_split}.pt"))
    text_embeddings = load_or_generate_prompt_ensemble_embeddings(
        class_names,
        device,
        cache_path=cache_path,
        prompt_templates=prompt_templates,
    )

    model_visual, nearest_indices, nearest_distances = _query_model_visuals(
        model,
        codec,
        projection,
        points,
        k=args.k,
        chunk_size=args.chunk_size,
        query_mode=args.query_mode,
        gaussian_indices=torch.from_numpy(sample_indices).to(device=device, dtype=torch.long)
        if args.query_mode == "gaussian_index"
        else None,
    )
    model_logits, model_pred_idx = _classify_logits(model_visual, text_embeddings)
    raw_ids = np.asarray(split_ids, dtype=np.int32)
    model_pred_labels = raw_ids[model_pred_idx]

    feature_paths = _select_feature_paths(config, args.max_views, args.teacher_split)
    teacher_features, teacher_poses, K = _load_teacher_batch(
        config,
        feature_paths,
        device,
        args.teacher_split,
    )
    with torch.no_grad():
        teacher_targets, teacher_valid, view_counts = sample_multiview_radio_targets(
            points,
            teacher_features,
            teacher_poses,
            K,
        )
        teacher_visual = _project_points(teacher_targets.float(), projection)
    teacher_logits, teacher_pred_idx = _classify_logits(teacher_visual, text_embeddings)
    teacher_pred_labels = raw_ids[teacher_pred_idx]
    teacher_valid_np = teacher_valid.detach().cpu().numpy().astype(bool)

    rows: list[dict] = []
    for i, point_index in enumerate(sample_indices.tolist()):
        label_id = int(sampled_labels_np[i])
        row = {
            "point_index": int(point_index),
            "label_id": label_id,
            "label_name": NYU40_ID_TO_NAME.get(label_id, f"class_{label_id}"),
            "model_pred_id": int(model_pred_labels[i]),
            "model_pred_name": NYU40_ID_TO_NAME.get(int(model_pred_labels[i]), str(int(model_pred_labels[i]))),
            "teacher_pred_id": int(teacher_pred_labels[i]) if teacher_valid_np[i] else -1,
            "teacher_pred_name": (
                NYU40_ID_TO_NAME.get(int(teacher_pred_labels[i]), str(int(teacher_pred_labels[i])))
                if teacher_valid_np[i]
                else "invalid"
            ),
            "model_correct": bool(int(model_pred_labels[i]) == label_id),
            "teacher_correct": bool(teacher_valid_np[i] and int(teacher_pred_labels[i]) == label_id),
            "teacher_valid": bool(teacher_valid_np[i]),
            "teacher_view_count": int(view_counts[i].item()),
            "nearest_gaussian": int(nearest_indices[i].item()),
            "nearest_gaussian_distance": float(nearest_distances[i].item()),
            "model_topk": _topk_names(model_logits[i], class_names, k=args.topk),
            "teacher_topk": _topk_names(teacher_logits[i], class_names, k=args.topk) if teacher_valid_np[i] else [],
        }
        rows.append(row)

    model_metrics = compute_split_metrics(model_pred_labels, sampled_labels_np, split_ids)
    teacher_metrics = compute_split_metrics(
        teacher_pred_labels[teacher_valid_np],
        sampled_labels_np[teacher_valid_np],
        split_ids,
    )
    summary = {
        "scene": args.scene,
        "class_split": args.class_split,
        "num_points": int(len(rows)),
        "teacher_valid_points": int(teacher_valid_np.sum()),
        "teacher_valid_ratio": float(teacher_valid_np.mean()) if len(teacher_valid_np) else 0.0,
        "mean_teacher_view_count": float(view_counts.float().mean().item()) if len(rows) else 0.0,
        "mean_nearest_gaussian_distance": float(nearest_distances.float().mean().item()) if len(rows) else 0.0,
        "model": {"miou": model_metrics["miou"], "macc": model_metrics["macc"]},
        "teacher": {"miou": teacher_metrics["miou"], "macc": teacher_metrics["macc"]},
        "teacher_feature_frames": [extract_feature_frame_index(path) for path in feature_paths],
    }
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": {key: str(value) for key, value in vars(args).items()},
        "summary": summary,
        "rows": rows,
    }


def _write_outputs(payload: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "point_logit_diagnostics.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = output_dir / "point_logit_diagnostics.csv"
    fieldnames = [
        "point_index",
        "label_id",
        "label_name",
        "model_pred_id",
        "model_pred_name",
        "teacher_pred_id",
        "teacher_pred_name",
        "model_correct",
        "teacher_correct",
        "teacher_valid",
        "teacher_view_count",
        "nearest_gaussian",
        "nearest_gaussian_distance",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload["rows"]:
            writer.writerow({key: row[key] for key in fieldnames})
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--prepared_root", default="dataset/scannet_og")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label_ply", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--class_split", choices=sorted(OPENGAUSSIAN_NYU40_CLASS_SPLITS), default="10")
    parser.add_argument("--max_points", type=int, default=512)
    parser.add_argument("--max_views", type=int, default=8)
    parser.add_argument("--teacher_split", choices=["all", "train", "val"], default="val")
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--query_mode", choices=["knn", "gaussian_index"], default="knn")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--prompt_templates", default="{query}")
    parser.add_argument("--text_embedding_cache", default="checkpoints/siglip2_scannet_og_text_embeddings.pt")
    parser.add_argument("--projection_weights", default="checkpoints/siglip2_feat_projection.pth")
    parser.add_argument("--summary_head_weights", default="checkpoints/siglip2_summary_head.pth")
    parser.add_argument("--radio_checkpoint", default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar")
    parser.add_argument("--use_summary_head", action="store_true", default=True)
    parser.add_argument("--no_summary_head", dest="use_summary_head", action="store_false")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    payload = run_diagnostics(args)
    _write_outputs(payload, Path(args.output_dir))
    s = payload["summary"]
    print(
        f"Summary: model mIoU={s['model']['miou']:.4f}, "
        f"teacher mIoU={s['teacher']['miou']:.4f}, "
        f"teacher_valid={s['teacher_valid_ratio']:.3f}"
    )


if __name__ == "__main__":
    main()
