#!/usr/bin/env python3
"""Render direct-3D coarse masks on arbitrary LERF views for mask-head training."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, ".")

from radio_gs.scripts.eval_lerf_direct_3d_selection import (
    GaussianSelectionProxy,
    SelectionSpec,
    apply_selection_ratio_bounds,
    build_lerf_dataset_for_scene,
    build_mask_renderer,
    compute_selection_ranking_scores,
    select_gaussians_from_scores,
)
from radio_gs.scripts.eval_lerf_grounding import (
    DEFAULT_LABEL_DIR,
    load_lerf_ovs_labels,
    load_render_pipeline,
    resolve_lerf_label_dir,
)


def choose_training_frame_ids(
    available_frames: Iterable[int],
    *,
    excluded: Iterable[int],
    max_frames: int,
) -> list[int]:
    """Choose evenly spaced unlabelled frames for pseudo-mask training."""

    excluded_set = {int(frame_id) for frame_id in excluded}
    candidates = sorted({int(frame_id) for frame_id in available_frames} - excluded_set)
    if not candidates:
        return []
    limit = int(max_frames)
    if limit <= 0 or limit >= len(candidates):
        return candidates
    if limit == 1:
        return [candidates[len(candidates) // 2]]
    raw = np.linspace(0, len(candidates) - 1, num=limit)
    selected: list[int] = []
    for idx in np.rint(raw).astype(int).tolist():
        frame_id = candidates[int(np.clip(idx, 0, len(candidates) - 1))]
        if frame_id not in selected:
            selected.append(frame_id)
    cursor = 0
    while len(selected) < limit and cursor < len(candidates):
        if candidates[cursor] not in selected:
            selected.append(candidates[cursor])
        cursor += 1
    return sorted(selected)


def coarse_mask_path(root: str | Path, *, frame_id: int, category: str) -> Path:
    safe_category = str(category).replace("/", "_")
    return Path(root) / f"frame_{int(frame_id):05d}_{safe_category}.png"


def _parse_frame_ids(raw: str) -> list[int]:
    if not str(raw or "").strip():
        return []
    return [int(part) for part in re.split(r"[,| ]+", str(raw).strip()) if part]


def _load_score_cache_unchecked(path: str | Path) -> tuple[torch.Tensor, list[str]]:
    payload = torch.load(Path(path), map_location="cpu")
    scores = payload.get("scores")
    if not torch.is_tensor(scores) or scores.ndim != 2:
        raise ValueError(f"score cache must contain scores [N,K]: {path}")
    metadata = payload.get("metadata", {})
    categories = [str(value) for value in metadata.get("categories", [])]
    if not categories:
        raise ValueError(f"score cache metadata must contain categories: {path}")
    if scores.shape[1] != len(categories):
        raise ValueError(
            f"score cache category count mismatch: scores={scores.shape[1]} categories={len(categories)}"
        )
    return scores.float().cpu(), categories


def save_pred_mask(path: str | Path, mask: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (np.asarray(mask).astype(np.uint8) * 255))


def render_coarse_masks(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(f"cuda:{int(args.gpu)}" if torch.cuda.is_available() else "cpu")
    label_dir = resolve_lerf_label_dir(args.label_dir)
    frame_annotations, scene_categories, img_h, img_w = load_lerf_ovs_labels(label_dir, args.scene)
    scores, cache_categories = _load_score_cache_unchecked(args.score_cache)
    if cache_categories != scene_categories:
        raise ValueError("score cache categories do not match LERF label category order")

    model, _codec, _renderer, _sharpener, _refiner, config, _is_hybrid = load_render_pipeline(
        args.config,
        args.checkpoint,
        device,
    )
    dataset = build_lerf_dataset_for_scene(
        args.scene,
        config,
        str(label_dir),
        feature_height=img_h,
        feature_width=img_w,
    )
    frame_ids = _parse_frame_ids(args.frame_ids)
    if not frame_ids:
        frame_ids = choose_training_frame_ids(
            dataset.pose_by_frame_idx.keys(),
            excluded=frame_annotations.keys(),
            max_frames=int(args.max_frames),
        )
    if not frame_ids:
        raise RuntimeError(f"No frame ids selected for {args.scene}")

    spec = SelectionSpec(args.selection_mode, float(args.selection_value))
    ranking_scores = compute_selection_ranking_scores(scores, mode=spec.mode)
    selected = select_gaussians_from_scores(scores, spec, min_select=int(args.min_select))
    selected = apply_selection_ratio_bounds(
        selected,
        ranking_scores,
        min_ratio=float(args.selection_min_ratio),
        max_ratio=float(args.selection_max_ratio),
        min_select=int(args.min_select),
    )
    selected = selected.to(device=device, dtype=torch.float32)
    proxy = GaussianSelectionProxy(model, selected)
    renderer = build_mask_renderer(config, height=img_h, width=img_w, device=device)
    output_dir = Path(args.output_dir)

    n_saved = 0
    for frame_id in tqdm(frame_ids, desc=f"render coarse masks {args.scene}", leave=False):
        pose_w2c = dataset.pose_by_frame_idx.get(int(frame_id))
        if pose_w2c is None:
            continue
        viewmat = torch.from_numpy(pose_w2c.copy()).float().to(device)
        with torch.no_grad():
            rendered = renderer.render_features(proxy, viewmat)
            silhouette = rendered["feature_map"].detach().float().cpu().numpy()
        for category_idx, category in enumerate(scene_categories):
            pred = silhouette[category_idx] > float(args.silhouette_threshold)
            save_pred_mask(coarse_mask_path(output_dir, frame_id=int(frame_id), category=category), pred)
            n_saved += 1

    summary = {
        "scene": args.scene,
        "frames": [int(frame_id) for frame_id in frame_ids],
        "categories": scene_categories,
        "output_dir": str(output_dir),
        "n_saved": int(n_saved),
        "selection_mode": args.selection_mode,
        "selection_value": float(args.selection_value),
        "silhouette_threshold": float(args.silhouette_threshold),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--score_cache", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--label_dir", default=DEFAULT_LABEL_DIR)
    parser.add_argument("--frame_ids", default="")
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument(
        "--selection_mode",
        choices=["score_threshold", "top_ratio", "mean_std", "score_margin", "score_ratio", "entropy_score"],
        default="score_threshold",
    )
    parser.add_argument("--selection_value", type=float, default=0.35)
    parser.add_argument("--selection_min_ratio", type=float, default=0.0)
    parser.add_argument("--selection_max_ratio", type=float, default=0.0)
    parser.add_argument("--silhouette_threshold", type=float, default=0.7)
    parser.add_argument("--min_select", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    return parser


def main() -> None:
    summary = render_coarse_masks(build_argparser().parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
