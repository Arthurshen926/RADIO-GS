#!/usr/bin/env python3
"""Evaluate reproduced OpenGaussian ScanNet point-cloud outputs.

This is a thin, configurable wrapper around OpenGaussian's official
``scripts/eval_scannet.py`` logic.  It reads the reproduced
``cluster_lang.npz`` and final Gaussian PLY, evaluates the OpenGaussian
NYU40 19/15/10 splits, and writes JSON plus qualitative PLY/PNG artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData, PlyElement

from radio_gs.scannet_constants import (
    NYU40_ID_TO_NAME,
    OPENGAUSSIAN_NYU40_CLASS_SPLITS,
    compute_split_metrics,
)


DEFAULT_SCENES = (
    "scene0000_00",
    "scene0062_00",
    "scene0070_00",
    "scene0097_00",
    "scene0140_00",
    "scene0200_00",
    "scene0347_00",
    "scene0400_00",
    "scene0590_00",
    "scene0645_00",
)

TEXT_FEATURE_ALIASES = {
    "shower curtain": "showercurtain",
    "floor mat": "floormat",
    "night stand": "nightstand",
}


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _normalise(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, eps)


def _npz_get(data: np.lib.npyio.NpzFile, *names: str) -> np.ndarray:
    for name in names:
        if name in data:
            return np.asarray(data[name])
    raise KeyError(f"None of {names!r} found in {data.files!r}")


def _read_label_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    names = set(vertex.data.dtype.names or ())
    required = {"x", "y", "z", "label"}
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"{path} is missing fields: {missing}")
    xyz = np.stack(
        [
            np.asarray(vertex["x"], dtype=np.float32),
            np.asarray(vertex["y"], dtype=np.float32),
            np.asarray(vertex["z"], dtype=np.float32),
        ],
        axis=1,
    )
    labels = np.asarray(vertex["label"], dtype=np.int64)
    return xyz, labels


def _read_opacity(path: Path) -> np.ndarray:
    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    names = set(vertex.data.dtype.names or ())
    if "opacity" not in names:
        raise ValueError(f"{path} is missing field: opacity")
    return np.asarray(vertex["opacity"], dtype=np.float32)


def _load_text_features(path: Path) -> dict[str, np.ndarray]:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return {str(key): np.asarray(value, dtype=np.float32) for key, value in raw.items()}


def _text_feature_for_name(features: dict[str, np.ndarray], class_name: str) -> np.ndarray:
    names = [class_name, TEXT_FEATURE_ALIASES.get(class_name, class_name)]
    compact = class_name.replace(" ", "")
    if compact not in names:
        names.append(compact)
    for name in names:
        if name in features:
            return features[name]
    raise KeyError(f"Missing OpenGaussian text feature for class name: {class_name}")


def _predict_raw_labels_for_split(
    leaf_lang_feat: np.ndarray,
    leaf_occu_count: np.ndarray,
    leaf_ind: np.ndarray,
    text_features: dict[str, np.ndarray],
    split_ids: list[int],
    min_occu_count: int,
) -> np.ndarray:
    class_names = [NYU40_ID_TO_NAME[int(class_id)] for class_id in split_ids]
    query = np.stack([_text_feature_for_name(text_features, name) for name in class_names], axis=0)

    lang = np.array(leaf_lang_feat, dtype=np.float32, copy=True)
    lang[np.asarray(leaf_occu_count).reshape(-1) < min_occu_count] = 0.0
    query = _normalise(query.astype(np.float32))
    lang = _normalise(lang.astype(np.float32))

    cluster_to_class = np.argmax(query @ lang.T, axis=0)
    safe_leaf_ind = np.clip(np.asarray(leaf_ind, dtype=np.int64).reshape(-1), 0, lang.shape[0] - 1)
    return np.asarray(split_ids, dtype=np.int64)[cluster_to_class[safe_leaf_ind]]


def _label_palette(label_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(label_ids, dtype=np.int64).reshape(-1)
    colors = np.zeros((ids.shape[0], 3), dtype=np.uint8)
    for raw_id in np.unique(ids):
        raw_id = int(raw_id)
        if raw_id <= 0:
            colors[ids == raw_id] = np.array([35, 35, 35], dtype=np.uint8)
            continue
        rng = np.random.RandomState(raw_id * 104729 + 17)
        colors[ids == raw_id] = rng.randint(55, 240, size=3, dtype=np.uint8)
    return colors


def _write_prediction_ply(
    path: Path,
    xyz: np.ndarray,
    labels: np.ndarray,
    pred_labels: np.ndarray,
    color_labels: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = _label_palette(color_labels)
    arr = np.empty(
        xyz.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("label", "i4"),
            ("pred_label", "i4"),
        ],
    )
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["red"], arr["green"], arr["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    arr["label"] = labels.astype(np.int32)
    arr["pred_label"] = pred_labels.astype(np.int32)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def _project_points(
    xyz: np.ndarray,
    labels: np.ndarray,
    split_ids: list[int],
    path: Path,
    title: str,
    image_size: int = 900,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels, dtype=np.int64)
    target = np.isin(labels, np.asarray(split_ids, dtype=np.int64))
    colors = _label_palette(labels)
    colors[~target] = np.array([28, 28, 28], dtype=np.uint8)

    xy = xyz[:, [0, 1]].astype(np.float32)
    finite = np.isfinite(xy).all(axis=1)
    xy = xy[finite]
    colors = colors[finite]
    if xy.size == 0:
        Image.new("RGB", (image_size, image_size), (255, 255, 255)).save(path)
        return

    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    scale = (image_size - 70) / float(max(span))
    pix = (xy - min_xy) * scale + 35.0
    pix[:, 1] = image_size - pix[:, 1]
    pix = np.clip(np.rint(pix), 0, image_size - 1).astype(np.int32)

    image = np.full((image_size, image_size, 3), 250, dtype=np.uint8)
    order = np.argsort(xyz[finite, 2])
    image[pix[order, 1], pix[order, 0]] = colors[order]
    pil = Image.fromarray(image, mode="RGB")
    draw = ImageDraw.Draw(pil)
    draw.rectangle([0, 0, image_size, 30], fill=(255, 255, 255))
    draw.text((12, 8), title, fill=(20, 20, 20))
    pil.save(path)


def _evaluate_scene(
    scene: str,
    model_root: Path,
    data_root: Path,
    text_features: dict[str, np.ndarray],
    output_root: Path,
    iteration: int,
    opacity_threshold: float,
    min_occu_count: int,
    save_ply: bool,
    save_png: bool,
) -> dict[str, Any]:
    scene_data = data_root / scene
    scene_model = model_root / scene
    label_ply = scene_data / f"{scene}_vh_clean_2.labels.ply"
    gaussian_ply = scene_model / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    mapping_file = scene_model / "cluster_lang.npz"
    if not label_ply.exists():
        raise FileNotFoundError(label_ply)
    if not gaussian_ply.exists():
        raise FileNotFoundError(gaussian_ply)
    if not mapping_file.exists():
        raise FileNotFoundError(mapping_file)

    xyz, labels = _read_label_ply(label_ply)
    opacity = _read_opacity(gaussian_ply)
    with np.load(mapping_file) as mapping:
        leaf_lang_feat = _npz_get(mapping, "leaf_feat", "leaf_feat.npy")
        leaf_occu_count = _npz_get(mapping, "occu_count", "occu_count.npy")
        leaf_ind = _npz_get(mapping, "leaf_ind", "leaf_ind.npy")

    if labels.shape[0] != leaf_ind.reshape(-1).shape[0]:
        raise ValueError(
            f"{scene}: label count {labels.shape[0]} != leaf_ind count {leaf_ind.reshape(-1).shape[0]}"
        )
    if labels.shape[0] != opacity.shape[0]:
        raise ValueError(f"{scene}: label count {labels.shape[0]} != opacity count {opacity.shape[0]}")

    kept_by_opacity = _sigmoid(opacity) >= opacity_threshold
    eval_labels = labels.copy()
    eval_labels[~kept_by_opacity] = 0

    scene_out = output_root / "visualizations" / scene
    split_results: dict[str, Any] = {}
    for split, split_ids in OPENGAUSSIAN_NYU40_CLASS_SPLITS.items():
        pred_labels = _predict_raw_labels_for_split(
            leaf_lang_feat,
            leaf_occu_count,
            leaf_ind,
            text_features,
            [int(v) for v in split_ids],
            min_occu_count=min_occu_count,
        )
        metrics = compute_split_metrics(pred_labels, eval_labels, split_ids)
        split_results[split] = metrics

        if save_ply:
            _write_prediction_ply(
                scene_out / f"pred_split_{split}.ply",
                xyz,
                eval_labels,
                pred_labels,
                pred_labels,
            )
            _write_prediction_ply(
                scene_out / f"gt_split_{split}.ply",
                xyz,
                eval_labels,
                pred_labels,
                eval_labels,
            )
        if save_png:
            _project_points(
                xyz,
                eval_labels,
                [int(v) for v in split_ids],
                scene_out / f"gt_split_{split}.png",
                f"{scene} GT split{split}",
            )
            _project_points(
                xyz,
                pred_labels,
                [int(v) for v in split_ids],
                scene_out / f"pred_split_{split}.png",
                f"{scene} OpenGaussian pred split{split}",
            )

    return {
        "scene": scene,
        "label_ply": str(label_ply),
        "model_path": str(scene_model),
        "mapping_file": str(mapping_file),
        "gaussian_ply": str(gaussian_ply),
        "num_points": int(labels.shape[0]),
        "opacity_filter": {
            "enabled": True,
            "threshold": float(opacity_threshold),
            "num_filtered": int((~kept_by_opacity).sum()),
            "num_points": int(labels.shape[0]),
        },
        "splits": split_results,
        "visualization_dir": str(scene_out) if save_ply or save_png else None,
    }


def _macro(scenes: dict[str, Any]) -> dict[str, dict[str, float]]:
    macro: dict[str, dict[str, float]] = {}
    for split in OPENGAUSSIAN_NYU40_CLASS_SPLITS:
        macro[split] = {}
        for key in ("miou", "macc"):
            values = [float(entry["splits"][split][key]) for entry in scenes.values()]
            macro[split][key] = float(np.mean(values)) if values else math.nan
    return macro


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# OpenGaussian ScanNet Baseline",
        "",
        f"- Model root: `{payload['args']['model_root']}`",
        f"- Data root: `{payload['args']['data_root']}`",
        f"- Iteration: `{payload['args']['iteration']}`",
        "",
        "| method | split19 mIoU | split19 mAcc | split15 mIoU | split15 mAcc | split10 mIoU | split10 mAcc |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    macro = payload["macro"]
    lines.append(
        "| OpenGaussian reproduced | "
        f"{macro['19']['miou']:.4f} | {macro['19']['macc']:.4f} | "
        f"{macro['15']['miou']:.4f} | {macro['15']['macc']:.4f} | "
        f"{macro['10']['miou']:.4f} | {macro['10']['macc']:.4f} |"
    )
    lines.extend(
        [
            "",
            "| scene | split19 mIoU | split19 mAcc | split15 mIoU | split15 mAcc | split10 mIoU | split10 mAcc |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scene, entry in sorted(payload["scenes"].items()):
        splits = entry["splits"]
        lines.append(
            f"| {scene} | "
            f"{splits['19']['miou']:.4f} | {splits['19']['macc']:.4f} | "
            f"{splits['15']['miou']:.4f} | {splits['15']['macc']:.4f} | "
            f"{splits['10']['miou']:.4f} | {splits['10']['macc']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", default="output/baselines/opengaussian/scannet")
    parser.add_argument("--data-root", default="dataset/scannet_og")
    parser.add_argument("--text-features", default="/root/baselines/OpenGaussian/assets/text_features.json")
    parser.add_argument("--output-dir", default="output/baselines/opengaussian/scannet_eval")
    parser.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES))
    parser.add_argument("--iteration", type=int, default=90000)
    parser.add_argument("--opacity-threshold", type=float, default=0.1)
    parser.add_argument("--min-occu-count", type=int, default=2)
    parser.add_argument("--no-ply", action="store_true")
    parser.add_argument("--no-png", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    text_features = _load_text_features(Path(args.text_features))
    scenes: dict[str, Any] = {}
    for scene in args.scenes:
        print(f"Evaluating OpenGaussian ScanNet scene: {scene}", flush=True)
        scenes[scene] = _evaluate_scene(
            scene,
            model_root=Path(args.model_root),
            data_root=Path(args.data_root),
            text_features=text_features,
            output_root=output_dir,
            iteration=args.iteration,
            opacity_threshold=args.opacity_threshold,
            min_occu_count=args.min_occu_count,
            save_ply=not args.no_ply,
            save_png=not args.no_png,
        )

    payload: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": "OpenGaussian reproduced",
        "source": "OpenGaussian official ScanNet point-cloud protocol wrapper",
        "args": {
            "model_root": args.model_root,
            "data_root": args.data_root,
            "text_features": args.text_features,
            "output_dir": args.output_dir,
            "scenes": args.scenes,
            "iteration": args.iteration,
            "opacity_threshold": args.opacity_threshold,
            "min_occu_count": args.min_occu_count,
        },
        "macro": _macro(scenes),
        "scenes": scenes,
    }
    json_path = output_dir / "opengaussian_scannet_results.json"
    md_path = output_dir / "opengaussian_scannet_results.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(md_path, payload)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
