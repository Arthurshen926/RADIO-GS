#!/usr/bin/env python3
"""Compose ScanNet open-vocabulary 3D query qualitative panels.

The figure is intentionally binary-query focused: each row highlights one
NYU40 class in a ScanNet scene, matching the direct point-query protocol more
closely than a full 19-class semantic coloring.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData

from radio_gs.scannet_constants import NYU40_ID_TO_NAME


REPO_ROOT = Path(__file__).resolve().parents[2]
NYU40_NAME_TO_ID = {name: class_id for class_id, name in NYU40_ID_TO_NAME.items()}


@dataclass(frozen=True)
class QueryCase:
    scene: str
    query: str


DEFAULT_CASES = [
    QueryCase("scene0097_00", "cabinet"),
    QueryCase("scene0062_00", "door"),
    QueryCase("scene0590_00", "picture"),
]


def rel_or_str(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_cases(items: Iterable[str]) -> list[QueryCase]:
    cases: list[QueryCase] = []
    for item in items:
        parts = item.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Expected scene:query, got {item!r}")
        query = parts[1].strip().lower()
        if query not in NYU40_NAME_TO_ID:
            raise ValueError(f"Unknown NYU40 query {query!r}")
        cases.append(QueryCase(parts[0].strip(), query))
    return cases


def read_ply_fields(path: Path) -> dict[str, np.ndarray]:
    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    names = set(vertex.data.dtype.names or ())
    required = {"x", "y", "z"}
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"{path} is missing fields: {missing}")
    fields: dict[str, np.ndarray] = {
        "xyz": np.stack(
            [
                np.asarray(vertex["x"], dtype=np.float32),
                np.asarray(vertex["y"], dtype=np.float32),
                np.asarray(vertex["z"], dtype=np.float32),
            ],
            axis=1,
        )
    }
    if {"red", "green", "blue"}.issubset(names):
        fields["rgb"] = np.stack(
            [
                np.asarray(vertex["red"], dtype=np.uint8),
                np.asarray(vertex["green"], dtype=np.uint8),
                np.asarray(vertex["blue"], dtype=np.uint8),
            ],
            axis=1,
        )
    if "label" in names:
        fields["label"] = np.asarray(vertex["label"], dtype=np.int64)
    if "pred_label" in names:
        fields["pred_label"] = np.asarray(vertex["pred_label"], dtype=np.int64)
    return fields


def find_ours_ply(root: Path, scene: str, split: str, pattern: str) -> Path:
    explicit = root / pattern.format(scene=scene, split=split)
    if explicit.exists():
        return explicit
    candidates = sorted(
        root.glob(f"{scene}_v67_dino_cv001_b2_s32768_ft20_gidx_labelpoint/visualizations/{scene}/pred_split_{split}.ply")
    )
    if candidates:
        return candidates[-1]
    candidates = sorted(root.glob(f"{scene}_*v67*din*cv*/visualizations/{scene}/pred_split_{split}.ply"))
    if candidates:
        return candidates[-1]
    candidates = sorted(root.glob(f"{scene}_*v67*/visualizations/{scene}/pred_split_{split}.ply"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"No GaussFM pred_split_{split}.ply found for {scene} under {root}")


def rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    rz = np.asarray(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rx = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(pitch), -math.sin(pitch)],
            [0.0, math.sin(pitch), math.cos(pitch)],
        ],
        dtype=np.float32,
    )
    return rx @ rz


def projection_params(xyz: np.ndarray, image_size: int, yaw: float, pitch: float) -> dict[str, np.ndarray | float]:
    finite = np.isfinite(xyz).all(axis=1)
    pts = xyz[finite].astype(np.float32)
    center = pts.mean(axis=0)
    rot = rotation_matrix(yaw, pitch)
    rotated = (pts - center) @ rot.T
    xy = rotated[:, :2]
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    scale = (image_size - 54) / float(max(span))
    offset = np.asarray([image_size * 0.5, image_size * 0.53], dtype=np.float32)
    return {
        "center": center,
        "rot": rot,
        "min_xy": min_xy,
        "max_xy": max_xy,
        "scale": float(scale),
        "offset": offset,
    }


def project_points(xyz: np.ndarray, params: dict[str, np.ndarray | float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.asarray(params["center"], dtype=np.float32)
    rot = np.asarray(params["rot"], dtype=np.float32)
    min_xy = np.asarray(params["min_xy"], dtype=np.float32)
    max_xy = np.asarray(params["max_xy"], dtype=np.float32)
    scale = float(params["scale"])
    offset = np.asarray(params["offset"], dtype=np.float32)
    rotated = (xyz.astype(np.float32) - center) @ rot.T
    scene_mid = (min_xy + max_xy) * 0.5
    screen = (rotated[:, :2] - scene_mid) * scale + offset
    screen[:, 1] = 2 * offset[1] - screen[:, 1]
    return screen, rotated[:, 2], np.isfinite(screen).all(axis=1)


def draw_points(
    xyz: np.ndarray,
    colors: np.ndarray,
    params: dict[str, np.ndarray | float],
    *,
    image_size: int,
    radius: int,
    background: tuple[int, int, int] = (252, 252, 252),
) -> Image.Image:
    canvas = np.full((image_size, image_size, 3), background, dtype=np.uint8)
    screen, depth, finite = project_points(xyz, params)
    pix = np.rint(screen[finite]).astype(np.int32)
    valid = (
        (pix[:, 0] >= 0)
        & (pix[:, 0] < image_size)
        & (pix[:, 1] >= 0)
        & (pix[:, 1] < image_size)
    )
    pix = pix[valid]
    cols = colors[finite][valid].astype(np.uint8)
    z = depth[finite][valid]
    order = np.argsort(z)
    if radius <= 1:
        canvas[pix[order, 1], pix[order, 0]] = cols[order]
    else:
        for x, y, color in zip(pix[order, 0], pix[order, 1], cols[order]):
            x0 = max(0, int(x) - radius)
            x1 = min(image_size, int(x) + radius + 1)
            y0 = max(0, int(y) - radius)
            y1 = min(image_size, int(y) + radius + 1)
            canvas[y0:y1, x0:x1] = color
    return Image.fromarray(canvas, mode="RGB")


def add_panel_header(image: Image.Image, title: str, subtitle: str = "") -> Image.Image:
    header_h = 34 if not subtitle else 52
    out = Image.new("RGB", (image.width, image.height + header_h), (255, 255, 255))
    out.paste(image, (0, header_h))
    draw = ImageDraw.Draw(out)
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
        subtitle_font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
    draw.rectangle([0, 0, image.width - 1, header_h - 1], fill=(255, 255, 255), outline=(222, 222, 222))
    draw.text((10, 8), title, fill=(20, 20, 20), font=title_font)
    if subtitle:
        draw.text((10, 29), subtitle, fill=(70, 70, 70), font=subtitle_font)
    return out


def binary_colors(labels: np.ndarray, class_id: int, color: tuple[int, int, int]) -> np.ndarray:
    colors = np.full((labels.shape[0], 3), (222, 222, 222), dtype=np.uint8)
    colors[labels == class_id] = np.asarray(color, dtype=np.uint8)
    return colors


def overview_colors(rgb: np.ndarray | None, n_points: int) -> np.ndarray:
    if rgb is None:
        return np.full((n_points, 3), (180, 180, 180), dtype=np.uint8)
    colors = rgb.astype(np.float32)
    colors = np.clip(colors * 0.92 + 18.0, 0, 255).astype(np.uint8)
    return colors


def iou_for_class(gt: np.ndarray, pred: np.ndarray, class_id: int) -> float:
    gt_mask = gt == class_id
    pred_mask = pred == class_id
    union = np.logical_or(gt_mask, pred_mask).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(gt_mask, pred_mask).sum() / union)


def paste_grid(rows: list[list[Image.Image]], gutter: int = 10) -> Image.Image:
    if not rows:
        raise ValueError("No rows to paste")
    col_widths = [max(row[col].width for row in rows) for col in range(len(rows[0]))]
    row_heights = [max(panel.height for panel in row) for row in rows]
    width = sum(col_widths) + gutter * (len(col_widths) - 1)
    height = sum(row_heights) + gutter * (len(row_heights) - 1)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    y = 0
    for row, row_h in zip(rows, row_heights):
        x = 0
        for col, panel in enumerate(row):
            canvas.paste(panel, (x, y))
            x += col_widths[col] + gutter
        y += row_h + gutter
    return canvas


def make_case_row(
    case: QueryCase,
    *,
    scannet_root: Path,
    baseline_eval: Path,
    baseline_name: str,
    ours_root: Path,
    ours_pattern: str,
    split: str,
    image_size: int,
    yaw: float,
    pitch: float,
) -> tuple[list[Image.Image], dict[str, object]]:
    class_id = NYU40_NAME_TO_ID[case.query]
    scene_dir = baseline_eval / "visualizations" / case.scene
    gt_ply = scene_dir / f"gt_split_{split}.ply"
    baseline_ply = scene_dir / f"pred_split_{split}.ply"
    ours_ply = find_ours_ply(ours_root, case.scene, split, ours_pattern)
    overview_ply = scannet_root / case.scene / f"{case.scene}_vh_clean_2.labels.ply"
    if not overview_ply.exists():
        overview_ply = gt_ply

    overview = read_ply_fields(overview_ply)
    gt = read_ply_fields(gt_ply)
    baseline = read_ply_fields(baseline_ply)
    ours = read_ply_fields(ours_ply)
    for name, payload in {"gt": gt, "baseline": baseline, "ours": ours}.items():
        if "label" not in payload:
            raise ValueError(f"{name} PLY for {case.scene} has no label field")
        if name != "gt" and "pred_label" not in payload:
            raise ValueError(f"{name} PLY for {case.scene} has no pred_label field")
        if payload["label"].shape[0] != gt["label"].shape[0]:
            raise ValueError(f"{name} point count differs from GT for {case.scene}")

    params = projection_params(gt["xyz"], image_size=image_size, yaw=yaw, pitch=pitch)
    overview_panel = draw_points(
        overview["xyz"],
        overview_colors(overview.get("rgb"), overview["xyz"].shape[0]),
        projection_params(overview["xyz"], image_size=image_size, yaw=yaw, pitch=pitch),
        image_size=image_size,
        radius=1,
    )
    gt_panel = draw_points(
        gt["xyz"],
        binary_colors(gt["label"], class_id, (55, 175, 85)),
        params,
        image_size=image_size,
        radius=1,
    )
    baseline_iou = iou_for_class(gt["label"], baseline["pred_label"], class_id)
    ours_iou = iou_for_class(gt["label"], ours["pred_label"], class_id)
    baseline_panel = draw_points(
        gt["xyz"],
        binary_colors(baseline["pred_label"], class_id, (65, 120, 220)),
        params,
        image_size=image_size,
        radius=1,
    )
    ours_panel = draw_points(
        gt["xyz"],
        binary_colors(ours["pred_label"], class_id, (225, 115, 55)),
        params,
        image_size=image_size,
        radius=1,
    )

    scene_title = case.scene.replace("_", " ")
    panels = [
        add_panel_header(overview_panel, scene_title, f'query: "{case.query}"'),
        add_panel_header(gt_panel, "GT binary mask", f"NYU40 id {class_id}"),
        add_panel_header(baseline_panel, baseline_name, f"IoU {baseline_iou:.3f}"),
        add_panel_header(ours_panel, "GaussFM", f"IoU {ours_iou:.3f}"),
    ]
    manifest = {
        "scene": case.scene,
        "query": case.query,
        "nyu40_id": class_id,
        "split": split,
        "baseline": baseline_name,
        "baseline_iou": round(baseline_iou, 4),
        "ctf_gs_iou": round(ours_iou, 4),
        "overview_ply": rel_or_str(overview_ply),
        "gt_ply": rel_or_str(gt_ply),
        "baseline_ply": rel_or_str(baseline_ply),
        "ctf_gs_ply": rel_or_str(ours_ply),
    }
    return panels, manifest


def write_markdown(path: Path, manifest: dict[str, object]) -> None:
    lines = [
        "# ScanNet Open-Vocabulary 3D Query Qualitative",
        "",
        "Binary query point-cloud visualization for the VALA-aligned direct point-query protocol.",
        f"The baseline is `{manifest['baseline_name']}`; GaussFM panels use saved ScanNet direct point-query predictions.",
        "",
        f"Figure: `{manifest['output']}`",
        "",
        f"| Scene | Query | {manifest['baseline_name']} IoU | GaussFM IoU | GaussFM source |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for case in manifest["cases"]:
        lines.append(
            "| {scene} | {query} | {baseline_iou:.4f} | {ctf_gs_iou:.4f} | `{ctf_gs_ply}` |".format(
                **case
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scannet-root", default="dataset/scannet_og")
    parser.add_argument("--baseline-eval", default="output/baselines/vala/scannet_vala8_compat_20260611_res2")
    parser.add_argument("--baseline-name", default="VALA")
    parser.add_argument(
        "--opengaussian-eval",
        default=None,
        help="Deprecated alias for --baseline-eval, kept for old reproduction commands.",
    )
    parser.add_argument("--ours-root", default="output/scannet_pointcloud_eval")
    parser.add_argument(
        "--ours-pattern",
        default="{scene}_v67_dino_cv001_b2_s32768_ft20_gidx_labelpoint/visualizations/{scene}/pred_split_{split}.ply",
        help="Path pattern below --ours-root; may use {scene} and {split}",
    )
    parser.add_argument("--output", default="paper/figures/scannet_openvocab_3d_query_qualitative.png")
    parser.add_argument("--manifest", default="paper/artifacts/scannet_openvocab_3d_query_qualitative_manifest.json")
    parser.add_argument("--report", default="paper/artifacts/scannet_openvocab_3d_query_qualitative.md")
    parser.add_argument("--cases", nargs="*", default=None, help="scene:query entries")
    parser.add_argument("--split", default="19", choices=("10", "15", "19"))
    parser.add_argument("--image-size", type=int, default=360)
    parser.add_argument("--yaw", type=float, default=-38.0)
    parser.add_argument("--pitch", type=float, default=58.0)
    args = parser.parse_args()

    baseline_eval = Path(args.opengaussian_eval) if args.opengaussian_eval else Path(args.baseline_eval)
    cases = parse_cases(args.cases) if args.cases else DEFAULT_CASES
    rows: list[list[Image.Image]] = []
    case_manifests: list[dict[str, object]] = []
    for case in cases:
        row, case_manifest = make_case_row(
            case,
            scannet_root=Path(args.scannet_root),
            baseline_eval=baseline_eval,
            baseline_name=args.baseline_name,
            ours_root=Path(args.ours_root),
            ours_pattern=args.ours_pattern,
            split=args.split,
            image_size=args.image_size,
            yaw=args.yaw,
            pitch=args.pitch,
        )
        rows.append(row)
        case_manifests.append(case_manifest)

    figure = paste_grid(rows, gutter=10)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.save(output)

    manifest = {
        "output": rel_or_str(output),
        "baseline_name": args.baseline_name,
        "baseline_eval": rel_or_str(baseline_eval),
        "ours_root": rel_or_str(Path(args.ours_root)),
        "split": args.split,
        "layout": f"overview | GT binary query | {args.baseline_name} prediction | GaussFM prediction",
        "cases": case_manifests,
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(report_path, manifest)
    print(f"Wrote {output}")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
