#!/usr/bin/env python3
"""Threshold-free one-point instance-affinity oracle on 2-D teacher features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from radio_gs.scripts.eval_lerf_grounding import build_gt_masks, load_lerf_ovs_labels


def _load_map(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor) or value.ndim != 3:
        raise ValueError(f"Expected one [C,H,W] tensor: {path}")
    return F.normalize(value.float(), dim=0, eps=1e-8)


def _average_precision(scores: np.ndarray, target: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    positives = target[order].astype(np.float64)
    total = float(positives.sum())
    if total <= 0:
        return 0.0
    precision = np.cumsum(positives) / np.arange(1, positives.size + 1)
    return float((precision * positives).sum() / total)


def _ranking_metrics(scores: np.ndarray, target: np.ndarray) -> dict[str, float]:
    order = np.argsort(-scores, kind="stable")
    ranked = target[order].astype(np.int64)
    cumulative = np.cumsum(ranked)
    count = np.arange(1, ranked.size + 1)
    positives = int(target.sum())
    union = positives + count - cumulative
    iou = cumulative / np.maximum(union, 1)
    equal_area_k = max(positives, 1)
    return {
        "average_precision": _average_precision(scores, target),
        "oracle_best_iou": float(iou.max(initial=0.0)),
        "equal_area_recall": float(cumulative[equal_area_k - 1] / max(positives, 1)),
        "positive_negative_margin": float(scores[target].mean() - scores[~target].mean()),
        "prevalence": float(target.mean()),
    }


def _centroid_seed(mask: np.ndarray) -> int:
    rows, cols = np.nonzero(mask)
    center = np.array([rows.mean(), cols.mean()])
    best = np.argmin((rows - center[0]) ** 2 + (cols - center[1]) ** 2)
    return int(rows[best] * mask.shape[1] + cols[best])


def run(args: argparse.Namespace) -> dict:
    annotations, categories, image_height, image_width = load_lerf_ovs_labels(
        args.label_dir, args.scene
    )
    root = Path(args.feature_root)
    sources = {
        "radio_raw_1280": "backbone",
        "radio_sam3_1024": "sam3",
        "radio_dinov3_4096": "dino_v3_7b",
    }
    rows: list[dict] = []
    rng = np.random.default_rng(args.seed)
    for frame_id, objects in sorted(annotations.items()):
        maps = {
            name: _load_map(root / folder / f"rgb_{int(frame_id)}.pt")
            for name, folder in sources.items()
        }
        feature_height, feature_width = next(iter(maps.values())).shape[-2:]
        for object_index, obj in enumerate(objects):
            category = str(obj["category"])
            full_mask = build_gt_masks(
                [obj], [category], image_height, image_width
            )[category]
            mask = F.interpolate(
                torch.from_numpy(full_mask).float()[None, None],
                size=(feature_height, feature_width),
                mode="nearest",
            )[0, 0].numpy().astype(bool)
            indices = np.flatnonzero(mask.reshape(-1))
            if indices.size == 0 or indices.size == mask.size:
                continue
            seeds = [("centroid", _centroid_seed(mask))]
            random_count = min(int(args.random_seeds), indices.size)
            random_indices = rng.choice(indices, size=random_count, replace=False)
            seeds.extend((f"random_{index}", int(seed)) for index, seed in enumerate(random_indices))
            flattened = {
                name: value.permute(1, 2, 0).reshape(-1, value.shape[0])
                for name, value in maps.items()
            }
            for seed_role, seed_index in seeds:
                scores = {
                    name: (values @ values[seed_index]).numpy()
                    for name, values in flattened.items()
                }
                scores["sam3_dino_equal_fusion"] = 0.5 * (
                    scores["radio_sam3_1024"] + scores["radio_dinov3_4096"]
                )
                for source, source_scores in scores.items():
                    rows.append(
                        {
                            "frame_id": int(frame_id),
                            "object_index": int(object_index),
                            "category": category,
                            "seed_role": seed_role,
                            "seed_index": seed_index,
                            "source": source,
                            "num_positive_tokens": int(mask.sum()),
                            **_ranking_metrics(source_scores, mask.reshape(-1)),
                        }
                    )
    aggregate = {}
    for source in [*sources, "sam3_dino_equal_fusion"]:
        selected = [row for row in rows if row["source"] == source]
        aggregate[source] = {
            key: float(np.mean([row[key] for row in selected]))
            for key in (
                "average_precision",
                "oracle_best_iou",
                "equal_area_recall",
                "positive_negative_margin",
                "prevalence",
            )
        }
        aggregate[source]["num_queries"] = len(selected)
    return {
        "scene": args.scene,
        "protocol": {
            "query": "one GT-sampled 2D teacher-grid point",
            "metrics": "threshold-free AP/equal-area recall plus oracle-best-IoU diagnostic",
            "test_calibration": "none; oracle best IoU is labelled upper-bound only",
            "random_seed": args.seed,
            "random_seeds_per_instance": args.random_seeds,
        },
        "aggregate": aggregate,
        "queries": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="ramen")
    parser.add_argument("--label_dir", default="/mnt/pool/sqy/3d_understanding/lerf_ovs/label")
    parser.add_argument("--feature_root", required=True)
    parser.add_argument("--random_seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
