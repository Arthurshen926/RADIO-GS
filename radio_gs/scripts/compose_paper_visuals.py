"""Compose paper-style comparison figures from two visualization runs.

The script expects two output directories produced by
`generate_visualizations_v2.py` and creates side-by-side comparison figures
for the main modalities:

  - feature PCA
  - depth
  - segmentation
  - grounding
  - a compact overview grid

Example:
  python radio_gs/scripts/compose_paper_visuals.py \
    --baseline_dir output/paper_figures/room0_nofdh_240ep \
    --ours_dir output/paper_figures/room0_fdh_ws240 \
    --output_dir output/paper_figures/room0_comparison
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _load_rgb(path: Path) -> np.ndarray:
  img = cv2.imread(str(path), cv2.IMREAD_COLOR)
  if img is None:
    raise FileNotFoundError(f"Could not read image: {path}")
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _resize_to_height(img: np.ndarray, target_height: int) -> np.ndarray:
  if img.shape[0] == target_height:
    return img
  scale = target_height / float(img.shape[0])
  target_width = max(1, int(round(img.shape[1] * scale)))
  return cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def _add_banner(img: np.ndarray, text: str, height: int = 44) -> np.ndarray:
  banner = np.full((height, img.shape[1], 3), 18, dtype=np.uint8)
  cv2.putText(
    banner,
    text,
    (16, int(height * 0.68)),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 255, 255),
    2,
    cv2.LINE_AA,
  )
  return np.concatenate([banner, img], axis=0)


def _stack_horizontal(left: np.ndarray, right: np.ndarray, gap: int = 12) -> np.ndarray:
  target_height = max(left.shape[0], right.shape[0])
  left = _resize_to_height(left, target_height)
  right = _resize_to_height(right, target_height)
  spacer = np.full((target_height, gap, 3), 255, dtype=np.uint8)
  return np.concatenate([left, spacer, right], axis=1)


def _pad_to_width(img: np.ndarray, target_width: int) -> np.ndarray:
  if img.shape[1] >= target_width:
    return img
  pad = np.full((img.shape[0], target_width - img.shape[1], 3), 255, dtype=np.uint8)
  return np.concatenate([img, pad], axis=1)


def _save_pair(
  baseline_path: Path,
  ours_path: Path,
  output_path: Path,
  title: str,
  baseline_label: str,
  ours_label: str,
) -> None:
  if not baseline_path.exists():
    raise FileNotFoundError(f"Missing baseline image: {baseline_path}")
  if not ours_path.exists():
    raise FileNotFoundError(f"Missing ours image: {ours_path}")

  baseline = _add_banner(_load_rgb(baseline_path), baseline_label)
  ours = _add_banner(_load_rgb(ours_path), ours_label)
  pair = _stack_horizontal(baseline, ours)
  pair = _add_banner(pair, title, height=48)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  cv2.imwrite(str(output_path), cv2.cvtColor(pair, cv2.COLOR_RGB2BGR))


def _first_match(root: Path, relative_glob: str) -> Path:
  matches = sorted(root.glob(relative_glob))
  if not matches:
    raise FileNotFoundError(f"No files matched {relative_glob} under {root}")
  return matches[0]


def main() -> None:
  parser = argparse.ArgumentParser(description="Compose RADIO-GS paper figures")
  parser.add_argument("--baseline_dir", required=True)
  parser.add_argument("--ours_dir", required=True)
  parser.add_argument("--output_dir", required=True)
  parser.add_argument("--baseline_label", default="Baseline")
  parser.add_argument("--ours_label", default="Ours")
  args = parser.parse_args()

  baseline_dir = Path(args.baseline_dir)
  ours_dir = Path(args.ours_dir)
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  modality_specs = [
    ("feature_pca", "pca_grid.png", "Feature PCA Comparison", "feature_pca_comparison.png"),
    ("depth", "depth_grid.png", "Depth Comparison", "depth_comparison.png"),
    ("segmentation", "seg_grid.png", "Segmentation Comparison", "segmentation_comparison.png"),
    ("grounding_seg", "grounding_seg_grid.png", "Grounding Comparison", "grounding_comparison.png"),
  ]

  pair_images = []
  for subdir, filename, title, out_name in modality_specs:
    baseline_path = baseline_dir / subdir / filename
    ours_path = ours_dir / subdir / filename
    output_path = output_dir / out_name
    _save_pair(
      baseline_path,
      ours_path,
      output_path,
      title,
      args.baseline_label,
      args.ours_label,
    )
    pair_images.append(_load_rgb(output_path))

  # Compact overview grid: 4 rows x 1 column pairs, stacked vertically.
  max_width = max(img.shape[1] for img in pair_images)
  overview = np.concatenate(
    [
      _pad_to_width(_add_banner(img, f"{idx + 1}. {label}"), max_width)
      for idx, (img, label) in enumerate(
        zip(
          pair_images,
          [spec[2] for spec in modality_specs],
        )
      )
    ],
    axis=0,
  )
  cv2.imwrite(str(output_dir / "overview_comparison.png"), cv2.cvtColor(overview, cv2.COLOR_RGB2BGR))

  # Representative composite frame comparison.
  baseline_frame = _first_match(baseline_dir / "composite", "composite_frame_*.png")
  ours_frame = _first_match(ours_dir / "composite", "composite_frame_*.png")
  _save_pair(
    baseline_frame,
    ours_frame,
    output_dir / "composite_frame_comparison.png",
    "Representative Multi-Task Composite",
    args.baseline_label,
    args.ours_label,
  )

  print(f"Saved comparison figures to {output_dir}")


if __name__ == "__main__":
  main()