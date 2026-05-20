#!/usr/bin/env python3
"""Evaluate OpenGaussian LeRF mask renders with the official frame protocol.

Ground truth may be either OpenGaussian's `label/<scene>/gt/<frame>/*.jpg`
mask layout or the local LERF polygon JSON layout at `label/<scene>/<frame>.json`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw


SCENE_GT_FRAMES: dict[str, list[str]] = {
    "waldo_kitchen": ["frame_00053", "frame_00066", "frame_00089", "frame_00140", "frame_00154"],
    "ramen": ["frame_00006", "frame_00024", "frame_00060", "frame_00065", "frame_00081", "frame_00119", "frame_00128"],
    "figurines": ["frame_00041", "frame_00105", "frame_00152", "frame_00195"],
    "teatime": ["frame_00002", "frame_00025", "frame_00043", "frame_00107", "frame_00129", "frame_00140"],
}
DEFAULT_SCENES = ("figurines", "ramen", "teatime", "waldo_kitchen")


@dataclass(frozen=True)
class ObjectResult:
    frame: str
    query: str
    gt_path: str
    pred_path: str
    iou: float
    missing: bool = False


@dataclass(frozen=True)
class SceneResult:
    scene: str
    miou: float
    acc025: float
    acc05: float
    count: int
    missing: int
    objects: list[ObjectResult]


@dataclass(frozen=True)
class GroundTruthMask:
    query: str
    path: str
    mask: np.ndarray


def load_binary_mask(path: Path, *, threshold: int = 10, grayscale: bool = False) -> np.ndarray:
    image = Image.open(path)
    if grayscale:
        image = image.convert("L")
    return (np.asarray(image) > threshold).astype(bool)


def calculate_iou(mask_gt: np.ndarray, mask_pred: np.ndarray) -> float:
    if mask_gt.shape != mask_pred.shape:
        raise ValueError(f"Mask shapes differ: gt={mask_gt.shape}, pred={mask_pred.shape}")
    intersection = np.logical_and(mask_gt, mask_pred).sum()
    union = np.logical_or(mask_gt, mask_pred).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def _rate_above(values: list[float], threshold: float) -> float:
    if not values:
        return 0.0
    return float((np.asarray(values) > threshold).sum() / len(values))


def _coerce_polygons(segmentation: object) -> list[np.ndarray]:
    if not isinstance(segmentation, list) or not segmentation:
        return []
    first = segmentation[0]
    if isinstance(first, (int, float)):
        arr = np.asarray(segmentation, dtype=np.float32)
        if arr.ndim == 1 and arr.size >= 6 and arr.size % 2 == 0:
            return [arr.reshape(-1, 2)]
        return []
    arr = np.asarray(segmentation, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] >= 2:
        return [arr[:, :2]]
    polygons: list[np.ndarray] = []
    for polygon in segmentation:
        arr = np.asarray(polygon, dtype=np.float32)
        if arr.ndim == 1 and arr.size >= 6 and arr.size % 2 == 0:
            arr = arr.reshape(-1, 2)
        elif arr.ndim == 2 and arr.shape[1] >= 2:
            arr = arr[:, :2]
        else:
            continue
        if arr.shape[0] >= 3:
            polygons.append(arr)
    return polygons


def _rasterize_polygons(polygons: list[np.ndarray], height: int, width: int) -> np.ndarray:
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        points = [tuple(map(float, point[:2])) for point in polygon]
        draw.polygon(points, outline=1, fill=1)
    return np.asarray(image, dtype=np.uint8).astype(bool)


def _load_json_gt_masks(label_root: Path, frame: str) -> list[GroundTruthMask]:
    scene_label_root = label_root.parent if label_root.name == "gt" else label_root
    json_path = scene_label_root / f"{frame}.json"
    if not json_path.exists():
        return []
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    info = payload.get("info", {})
    width = int(info["width"])
    height = int(info["height"])
    by_query: dict[str, np.ndarray] = {}
    for obj in payload.get("objects", []):
        query = str(obj.get("category", "")).strip()
        polygons = _coerce_polygons(obj.get("segmentation"))
        if not query or not polygons:
            continue
        mask = _rasterize_polygons(polygons, height, width)
        if query in by_query:
            by_query[query] = np.logical_or(by_query[query], mask)
        else:
            by_query[query] = mask
    return [
        GroundTruthMask(query=query, path=str(json_path), mask=mask)
        for query, mask in sorted(by_query.items())
    ]


def _load_frame_gt_masks(label_root: Path, frame: str, *, threshold: int) -> list[GroundTruthMask]:
    frame_gt_root = label_root / frame
    if frame_gt_root.exists():
        return [
            GroundTruthMask(
                query=gt_path.stem,
                path=str(gt_path),
                mask=load_binary_mask(gt_path, threshold=threshold),
            )
            for gt_path in sorted(frame_gt_root.glob("*.jpg"))
        ]
    return _load_json_gt_masks(label_root, frame)


def evaluate_scene(gt_root: Path, pred_root: Path, scene: str, *, threshold: int = 10) -> SceneResult:
    if scene not in SCENE_GT_FRAMES:
        choices = ", ".join(sorted(SCENE_GT_FRAMES))
        raise ValueError(f"Unknown scene {scene!r}; expected one of: {choices}")

    objects: list[ObjectResult] = []
    ious: list[float] = []
    missing = 0
    for frame in SCENE_GT_FRAMES[scene]:
        for gt in _load_frame_gt_masks(gt_root, frame, threshold=threshold):
            pred_path = pred_root / f"{frame}_{gt.query}.png"
            if not pred_path.exists():
                missing += 1
                ious.append(0.0)
                objects.append(
                    ObjectResult(
                        frame=frame,
                        query=gt.query,
                        gt_path=gt.path,
                        pred_path=str(pred_path),
                        iou=0.0,
                        missing=True,
                    )
                )
                continue
            mask_pred = load_binary_mask(pred_path, threshold=threshold, grayscale=True)
            iou = calculate_iou(gt.mask, mask_pred)
            ious.append(iou)
            objects.append(
                ObjectResult(
                    frame=frame,
                    query=gt.query,
                    gt_path=gt.path,
                    pred_path=str(pred_path),
                    iou=iou,
                )
            )

    return SceneResult(
        scene=scene,
        miou=_mean(ious),
        acc025=_rate_above(ious, 0.25),
        acc05=_rate_above(ious, 0.5),
        count=len(ious),
        missing=missing,
        objects=objects,
    )


def _pred_root(model_root: Path, scene: str, iteration: int) -> Path:
    return model_root / scene / "text2obj" / f"ours_{iteration}" / "renders_cluster_silhouette"


def evaluate_run(
    lerf_root: Path,
    model_root: Path,
    scenes: Sequence[str],
    *,
    iteration: int = 70000,
    threshold: int = 10,
) -> dict[str, object]:
    scene_results: dict[str, SceneResult] = {}
    for scene in scenes:
        gt_root = lerf_root / "label" / scene / "gt"
        scene_results[scene] = evaluate_scene(
            gt_root,
            _pred_root(model_root, scene, iteration),
            scene,
            threshold=threshold,
        )

    macro_source = list(scene_results.values())
    report = {
        "protocol": "OpenGaussian LERF object-selection mask IoU",
        "iteration": iteration,
        "threshold": threshold,
        "scenes": {scene: asdict(result) for scene, result in scene_results.items()},
        "macro": {
            "miou": _mean([result.miou for result in macro_source]),
            "acc025": _mean([result.acc025 for result in macro_source]),
            "acc05": _mean([result.acc05 for result in macro_source]),
            "count": int(sum(result.count for result in macro_source)),
            "missing": int(sum(result.missing for result in macro_source)),
        },
    }
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lerf-root", type=Path, required=True, help="LERF-OVS root containing label/<scene> annotations.")
    parser.add_argument("--model-root", type=Path, required=True, help="Root containing per-scene OpenGaussian model outputs.")
    parser.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES), choices=DEFAULT_SCENES)
    parser.add_argument("--iteration", type=int, default=70000)
    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = evaluate_run(
        args.lerf_root,
        args.model_root,
        args.scenes,
        iteration=args.iteration,
        threshold=args.threshold,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    macro = report["macro"]
    print(
        "Macro "
        f"mIoU={macro['miou']:.4f} "
        f"Acc@0.25={macro['acc025']:.4f} "
        f"Acc@0.5={macro['acc05']:.4f} "
        f"objects={macro['count']} missing={macro['missing']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
