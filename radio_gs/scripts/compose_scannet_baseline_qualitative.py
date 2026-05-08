#!/usr/bin/env python3
"""Compose ScanNet qualitative comparisons for RADIO-GS and OpenGaussian."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData


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


def _read_colored_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    names = set(vertex.data.dtype.names or ())
    required = {"x", "y", "z", "red", "green", "blue"}
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
    rgb = np.stack(
        [
            np.asarray(vertex["red"], dtype=np.uint8),
            np.asarray(vertex["green"], dtype=np.uint8),
            np.asarray(vertex["blue"], dtype=np.uint8),
        ],
        axis=1,
    )
    return xyz, rgb


def _project_colored_points(
    xyz: np.ndarray,
    rgb: np.ndarray,
    *,
    image_size: int,
    title: str,
) -> Image.Image:
    xy = xyz[:, [0, 1]].astype(np.float32)
    finite = np.isfinite(xy).all(axis=1)
    xy = xy[finite]
    z = xyz[finite, 2]
    rgb = rgb[finite]
    image = np.full((image_size, image_size, 3), 250, dtype=np.uint8)
    if xy.size:
        min_xy = xy.min(axis=0)
        max_xy = xy.max(axis=0)
        span = np.maximum(max_xy - min_xy, 1e-6)
        scale = (image_size - 70) / float(max(span))
        pix = (xy - min_xy) * scale + 35.0
        pix[:, 1] = image_size - pix[:, 1]
        pix = np.clip(np.rint(pix), 0, image_size - 1).astype(np.int32)
        order = np.argsort(z)
        image[pix[order, 1], pix[order, 0]] = rgb[order]
    pil = Image.fromarray(image, mode="RGB")
    draw = ImageDraw.Draw(pil)
    draw.rectangle([0, 0, image_size, 32], fill=(255, 255, 255))
    draw.text((12, 9), title, fill=(20, 20, 20))
    return pil


def _find_radio_scene_dir(radio_root: Path, scene: str) -> Path | None:
    candidates = sorted(
        radio_root.glob(
            f"{scene}_*v67_teacherbalanced_fromv63_best_gidx_labelpoint/visualizations/{scene}"
        )
    )
    if candidates:
        return candidates[-1]
    candidates = sorted(radio_root.glob(f"{scene}_*v67*/visualizations/{scene}"))
    return candidates[-1] if candidates else None


def _load_available_scenes(opengaussian_eval: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    json_path = opengaussian_eval / "opengaussian_scannet_results.json"
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        return sorted(payload.get("scenes", {}).keys())
    return list(DEFAULT_SCENES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opengaussian-eval", default="output/baselines/opengaussian/scannet_eval")
    parser.add_argument("--radio-root", default="output/scannet_pointcloud_eval")
    parser.add_argument("--output", default="output/baselines/opengaussian/scannet_qualitative_comparison.png")
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--split", default="10", choices=("10", "15", "19"))
    parser.add_argument("--image-size", type=int, default=620)
    args = parser.parse_args()

    og_eval = Path(args.opengaussian_eval)
    radio_root = Path(args.radio_root)
    scenes = _load_available_scenes(og_eval, args.scenes)

    rows: list[Image.Image] = []
    used_scenes: list[str] = []
    for scene in scenes:
        og_dir = og_eval / "visualizations" / scene
        radio_dir = _find_radio_scene_dir(radio_root, scene)
        if radio_dir is None:
            print(f"Skipping {scene}: missing RADIO-GS v67 visualization")
            continue
        gt_ply = og_dir / f"gt_split_{args.split}.ply"
        radio_ply = radio_dir / f"pred_split_{args.split}.ply"
        og_ply = og_dir / f"pred_split_{args.split}.ply"
        missing = [path for path in (gt_ply, radio_ply, og_ply) if not path.exists()]
        if missing:
            print(f"Skipping {scene}: missing {', '.join(str(path) for path in missing)}")
            continue

        panels: list[Image.Image] = []
        for title, ply_path in (
            (f"{scene} GT", gt_ply),
            (f"{scene} RADIO-GS", radio_ply),
            (f"{scene} OpenGaussian", og_ply),
        ):
            xyz, rgb = _read_colored_points(ply_path)
            panels.append(
                _project_colored_points(
                    xyz,
                    rgb,
                    image_size=args.image_size,
                    title=title,
                )
            )
        row = Image.new("RGB", (args.image_size * 3, args.image_size), (255, 255, 255))
        for idx, panel in enumerate(panels):
            row.paste(panel, (idx * args.image_size, 0))
        rows.append(row)
        used_scenes.append(scene)

    if not rows:
        raise RuntimeError("No qualitative rows could be composed")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    montage = Image.new("RGB", (args.image_size * 3, args.image_size * len(rows)), (255, 255, 255))
    for idx, row in enumerate(rows):
        montage.paste(row, (0, idx * args.image_size))
    montage.save(output)
    meta = {
        "output": str(output),
        "opengaussian_eval": str(og_eval),
        "radio_root": str(radio_root),
        "split": args.split,
        "scenes": used_scenes,
    }
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {output}")
    print(f"Wrote {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
