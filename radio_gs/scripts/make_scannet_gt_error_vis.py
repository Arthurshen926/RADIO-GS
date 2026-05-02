#!/usr/bin/env python3
"""Create GT and error-coloured ScanNet PLYs from evaluator prediction PLYs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from radio_gs.scannet_constants import OPENGAUSSIAN_NYU40_CLASS_SPLITS
from radio_gs.scripts.eval_scannet_pointcloud_radio_gs import _label_colors


def _read_prediction_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    names = set(vertex.data.dtype.names or ())
    required = {"x", "y", "z", "label", "pred_label"}
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"{path} is missing required fields: {missing}")
    xyz = np.stack(
        [
            np.asarray(vertex["x"], dtype=np.float32),
            np.asarray(vertex["y"], dtype=np.float32),
            np.asarray(vertex["z"], dtype=np.float32),
        ],
        axis=1,
    )
    labels = np.asarray(vertex["label"], dtype=np.int32)
    pred_labels = np.asarray(vertex["pred_label"], dtype=np.int32)
    return xyz, labels, pred_labels


def _write_colored_ply(
    path: Path,
    xyz: np.ndarray,
    colors: np.ndarray,
    labels: np.ndarray,
    pred_labels: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _error_colors(labels: np.ndarray, pred_labels: np.ndarray, split: str) -> np.ndarray:
    target_ids = set(int(v) for v in OPENGAUSSIAN_NYU40_CLASS_SPLITS[split])
    target = np.array([int(v) in target_ids for v in labels], dtype=bool)
    correct = target & (labels == pred_labels)
    wrong = target & (labels != pred_labels)

    colors = np.zeros((labels.shape[0], 3), dtype=np.uint8)
    colors[~target] = np.array([35, 35, 35], dtype=np.uint8)
    colors[correct] = np.array([185, 185, 185], dtype=np.uint8)
    colors[wrong] = np.array([230, 45, 45], dtype=np.uint8)
    return colors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred_ply", required=True, help="Evaluator pred_split_*.ply")
    parser.add_argument("--split", required=True, choices=sorted(OPENGAUSSIAN_NYU40_CLASS_SPLITS))
    parser.add_argument("--output_dir", default=None, help="Defaults to the prediction PLY directory")
    args = parser.parse_args()

    pred_ply = Path(args.pred_ply)
    output_dir = Path(args.output_dir) if args.output_dir else pred_ply.parent
    xyz, labels, pred_labels = _read_prediction_ply(pred_ply)

    _write_colored_ply(
        output_dir / f"gt_split_{args.split}.ply",
        xyz,
        _label_colors(labels),
        labels,
        pred_labels,
    )
    _write_colored_ply(
        output_dir / f"error_split_{args.split}.ply",
        xyz,
        _error_colors(labels, pred_labels, args.split),
        labels,
        pred_labels,
    )
    print(f"Wrote GT/error PLYs to {output_dir}")


if __name__ == "__main__":
    main()
