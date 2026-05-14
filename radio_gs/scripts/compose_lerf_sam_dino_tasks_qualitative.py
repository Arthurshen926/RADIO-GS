#!/usr/bin/env python3
"""Compose the paper-facing LERF SAM/DINO downstream qualitative figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def rel_or_str(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def resize_width(image: np.ndarray, width: int) -> np.ndarray:
    scale = float(width) / float(image.shape[1])
    height = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def add_band(title: str, width: int) -> np.ndarray:
    band = np.full((46, width, 3), 18, dtype=np.uint8)
    cv2.putText(
        band,
        title,
        (16, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return band


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sam_visual",
        default=(
            "output/lerf_sam_dino_tasks/mainline_fixed_vis/waldo_kitchen/visualizations/"
            "waldo_kitchen/waldo_kitchen_sam3_mask_prompt_propagation_00154_plate.png"
        ),
    )
    parser.add_argument(
        "--dino_visual",
        default=(
            "output/lerf_sam_dino_tasks/formal_v8_mutual_homography_ransac_vis/ramen/visualizations/"
            "ramen/ramen_dino_v3_mask_propagation_00024_bowl.png"
        ),
    )
    parser.add_argument("--output", default="paper/figures/lerf_sam_dino_tasks_qualitative.png")
    parser.add_argument("--manifest", default="output/radio_gs/reports/lerf_sam_dino_tasks_qualitative_manifest.json")
    parser.add_argument("--width", type=int, default=2400)
    args = parser.parse_args()

    sam_path = Path(args.sam_visual)
    dino_path = Path(args.dino_visual)
    width = int(args.width)
    sam = resize_width(read_image(sam_path), width)
    dino = resize_width(read_image(dino_path), width)
    spacer = np.full((16, width, 3), 255, dtype=np.uint8)
    figure = np.vstack(
        [
            add_band("SAM3-adaptor mask propagation from reconstructed features", width),
            sam,
            spacer,
            add_band("DINOv3-adaptor robust mask propagation, with dense matching kept as diagnostic", width),
            dino,
        ]
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), figure)

    manifest = {
        "figure": str(output),
        "width": width,
        "sam_visual": rel_or_str(sam_path),
        "dino_visual": rel_or_str(dino_path),
        "notes": (
            "DINO panel uses the formal robust mask-propagation readout rather than raw "
            "nearest-neighbor match lines; raw dense matching remains a diagnostic."
        ),
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
