#!/usr/bin/env python3
"""Compose submission-style LERF 2D/3D OVS qualitative panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from radio_gs.scripts.compose_lerf_main_qualitative import (
    DEFAULT_OVS_2D3D_GROUPS,
    DR_SPLAT_SCENE_DIR,
    REPO_ROOT,
    QualCase,
    build_gt_mask,
    compute_iou,
    ctf_gs_2d_visual_path,
    dr_splat_3d_render_path,
    fit_panel,
    load_binary_mask,
    put_text_box,
    read_image,
    rel_or_str,
    resize_mask,
)


GREEN = (70, 185, 80)
ORANGE = (224, 122, 58)
BLUE = (54, 118, 224)
RED = (38, 38, 220)


def _slug(query: str) -> str:
    return query.replace(" ", "_").replace("/", "_")


def _scene_eval_name(scene: str) -> str:
    return f"{scene}_0"


def read_wide_visual_panel(path: Path, *, index: int = 5, columns: int = 6) -> np.ndarray:
    """Read one column from a wide PNG visual without OpenCV decoding stalls."""
    image = Image.open(path).convert("RGB")
    if image.width < columns:
        return np.asarray(image)[:, :, ::-1].copy()
    panel_w = image.width // columns
    x0 = min(index, columns - 1) * panel_w
    x1 = image.width if index == columns - 1 else x0 + panel_w
    crop = image.crop((x0, 0, x1, image.height))
    return np.asarray(crop)[:, :, ::-1].copy()


def _mask_candidates(root: Path, scene: str, frame: str, query: str) -> list[Path]:
    return [
        root / "eval" / scene / frame / f"chosen_{query}.png",
        root / "eval" / scene / frame / f"{query}.png",
        root / "eval" / _scene_eval_name(scene) / frame / f"chosen_{query}.png",
        root / "eval" / _scene_eval_name(scene) / frame / f"{query}.png",
        root / "eval" / _scene_eval_name(scene) / "pred" / f"frame_{frame}_{query}.png",
        root / "eval" / _scene_eval_name(scene) / "masks" / f"frame_{frame}_{query}.png",
    ]


def _overlay_candidates(root: Path, scene: str, frame: str, query: str) -> list[Path]:
    return [
        root / "eval" / scene / frame / f"overlay_{query}.png",
        root / "eval" / _scene_eval_name(scene) / frame / f"overlay_{query}.png",
        root / "eval" / scene / frame / f"heatmap_{query}.png",
        root / "eval" / _scene_eval_name(scene) / frame / f"heatmap_{query}.png",
    ]


def resolve_2d_baseline_mask(
    *,
    case: QualCase,
    preferred_root: Path,
    fallback_root: Path,
) -> tuple[Path, str, bool]:
    for path in _mask_candidates(preferred_root, case.scene, case.frame_id, case.query):
        if path.exists():
            return path, "LangSplatV2 (repro.)", False
    for path in _mask_candidates(fallback_root, case.scene, case.frame_id, case.query):
        if path.exists():
            return path, "LangSplat classic (repro.; V2 masks not exported)", True
    candidates = _mask_candidates(preferred_root, case.scene, case.frame_id, case.query)
    raise FileNotFoundError(f"No 2D baseline mask for {case}: first tried {candidates[0]}")


def resolve_2d_baseline_visual(
    *,
    case: QualCase,
    preferred_root: Path,
    fallback_root: Path,
) -> Path | None:
    for path in _overlay_candidates(preferred_root, case.scene, case.frame_id, case.query):
        if path.exists():
            return path
    for path in _overlay_candidates(fallback_root, case.scene, case.frame_id, case.query):
        if path.exists():
            return path
    return None


def dr_splat_mask_path(root: Path, case: QualCase) -> Path:
    scene_dir = DR_SPLAT_SCENE_DIR[case.scene]
    return (
        root
        / scene_dir
        / "predictions_mask_0.4"
        / "renders_silhouette"
        / f"frame_{case.frame_id}"
        / f"{case.query}.png"
    )


def ours_3d_mask_path(root: Path, case: QualCase, selection: str) -> Path:
    return root / "pred_masks" / selection / case.scene / f"frame_{case.frame_id}_{case.query}.png"


def dimmed_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    *,
    alpha: float = 0.72,
) -> np.ndarray:
    out = (0.34 * image.astype(np.float32) + 0.66 * 255.0).astype(np.uint8)
    color_arr = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    out[mask] = ((1.0 - alpha) * out[mask].astype(np.float32) + alpha * color_arr).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 3, lineType=cv2.LINE_AA)
    return out


def blank_support_panel(
    image: np.ndarray,
    pred: np.ndarray,
    *,
    color: tuple[int, int, int],
    width: int,
    height: int,
) -> np.ndarray:
    """Show direct-3D selected support on a blank canvas.

    Unlike the 2D rendered-view panels, this does not retain the RGB
    background. The selected primitive/mask support is projected to the same
    view for evaluation, then rendered alone on white.
    """
    support = np.full_like(image, 255)
    if np.any(pred):
        tinted = image.copy()
        color_arr = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
        tinted[pred] = (0.70 * tinted[pred].astype(np.float32) + 0.30 * color_arr).astype(np.uint8)
        support[pred] = tinted[pred]
        contours, _ = cv2.findContours(pred.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(support, contours, -1, color, 3, lineType=cv2.LINE_AA)
    return fit_panel(support, width=width, height=height)


def tagged_panel(panel: np.ndarray, text: str, *, color: tuple[int, int, int]) -> np.ndarray:
    return put_text_box(panel, [text], origin=(8, 24), font_scale=0.42, color=(24, 24, 24), bg=(255, 255, 255))


def text_header(text: str, width: int, height: int = 34) -> np.ndarray:
    panel = np.full((height, width, 3), 255, dtype=np.uint8)
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 1)[0]
    x = max(6, (width - size[0]) // 2)
    cv2.putText(panel, text, (x, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.line(panel, (0, height - 1), (width, height - 1), (225, 225, 225), 1)
    return panel


def row_label(text: str, width: int, height: int) -> np.ndarray:
    panel = np.full((height, width, 3), 255, dtype=np.uint8)
    font_scale = 0.58
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
    if size[0] > width - 8:
        font_scale = max(0.42, (width - 10) / max(size[0], 1) * font_scale)
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
    cv2.putText(
        panel,
        text,
        (max(4, (width - size[0]) // 2), height // 2 + 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (28, 28, 28),
        1,
        cv2.LINE_AA,
    )
    return panel


def make_case_panels(
    case: QualCase,
    *,
    label_root: Path,
    baseline_2d_root: Path,
    baseline_2d_fallback_root: Path,
    baseline_3d_root: Path,
    ours_2d_root: Path,
    ours_3d_root: Path,
    ours_3d_selection: str,
    panel_width: int,
    panel_height: int,
) -> tuple[list[list[np.ndarray]], dict[str, object]]:
    rgb_path = label_root / case.scene / f"frame_{case.frame_id}.jpg"
    label_json = label_root / case.scene / f"frame_{case.frame_id}.json"
    rgb = read_image(rgb_path)
    gt = build_gt_mask(label_json, case.query)

    baseline_2d_path, baseline_2d_label, fallback_used = resolve_2d_baseline_mask(
        case=case,
        preferred_root=baseline_2d_root,
        fallback_root=baseline_2d_fallback_root,
    )
    baseline_2d = resize_mask(load_binary_mask(baseline_2d_path), gt.shape)
    baseline_2d_visual_path = resolve_2d_baseline_visual(
        case=case,
        preferred_root=baseline_2d_root,
        fallback_root=baseline_2d_fallback_root,
    )

    baseline_3d_path = dr_splat_mask_path(baseline_3d_root, case)
    baseline_3d = resize_mask(load_binary_mask(baseline_3d_path), gt.shape)
    baseline_3d_render_path = dr_splat_3d_render_path(baseline_3d_root, case)

    ours_3d_path = ours_3d_mask_path(ours_3d_root, case, ours_3d_selection)
    ours_3d = resize_mask(load_binary_mask(ours_3d_path), gt.shape)
    ours_2d_path = ctf_gs_2d_visual_path(ours_2d_root, case)
    ours_2d_visual = read_wide_visual_panel(ours_2d_path, index=5, columns=6)

    gt_2d = fit_panel(dimmed_overlay(rgb, gt, GREEN), width=panel_width, height=panel_height)
    gt_3d = blank_support_panel(rgb, gt, color=GREEN, width=panel_width, height=panel_height)
    if baseline_2d_visual_path is not None:
        prior_2d = fit_panel(read_image(baseline_2d_visual_path), width=panel_width, height=panel_height)
    else:
        prior_2d = fit_panel(dimmed_overlay(rgb, baseline_2d, ORANGE), width=panel_width, height=panel_height)
    prior_2d = tagged_panel(prior_2d, "LangSplatV2", color=ORANGE)
    prior_3d = blank_support_panel(
        rgb,
        baseline_3d,
        color=ORANGE,
        width=panel_width,
        height=panel_height,
    )
    prior_3d = tagged_panel(prior_3d, "Dr. Splat", color=ORANGE)
    ours_2d = fit_panel(ours_2d_visual, width=panel_width, height=panel_height)
    ours_3d_panel = blank_support_panel(
        rgb,
        ours_3d,
        color=BLUE,
        width=panel_width,
        height=panel_height,
    )

    manifest = {
        "scene": case.scene,
        "frame_id": case.frame_id,
        "query": case.query,
        "baseline_2d": baseline_2d_label,
        "baseline_3d": "Dr. Splat (repro.)",
        "baseline_2d_fallback_used": fallback_used,
        "baseline_2d_iou": round(compute_iou(gt, baseline_2d), 4),
        "baseline_3d_iou": round(compute_iou(gt, baseline_3d), 4),
        "ours_3d_iou": round(compute_iou(gt, ours_3d), 4),
        "rgb": rel_or_str(rgb_path),
        "label_json": rel_or_str(label_json),
        "baseline_2d_mask": rel_or_str(baseline_2d_path),
        "baseline_2d_visual": rel_or_str(baseline_2d_visual_path) if baseline_2d_visual_path else "",
        "baseline_3d_mask": rel_or_str(baseline_3d_path),
        "baseline_3d_render": rel_or_str(baseline_3d_render_path),
        "ours_2d_visual": rel_or_str(ours_2d_path),
        "ours_3d_mask": rel_or_str(ours_3d_path),
    }
    return [[gt_2d, gt_3d], [prior_2d, prior_3d], [ours_2d, ours_3d_panel]], manifest


def make_scene_block(
    cases: list[QualCase],
    *,
    label_root: Path,
    baseline_2d_root: Path,
    baseline_2d_fallback_root: Path,
    baseline_3d_root: Path,
    ours_2d_root: Path,
    ours_3d_root: Path,
    ours_3d_selection: str,
    panel_width: int,
    panel_height: int,
    reference_width: int,
    row_label_width: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    scene = cases[0].scene
    header_h = 36
    query_h = 34
    row_count = 3
    margin = 12
    block_w = reference_width + row_label_width + len(cases) * 2 * panel_width + 2 * margin
    block_h = header_h + row_count * panel_height + query_h + 2 * margin
    canvas = np.full((block_h, block_w, 3), 255, dtype=np.uint8)
    cv2.rectangle(canvas, (2, 2), (block_w - 3, block_h - 3), (225, 232, 242), 2, cv2.LINE_AA)

    first_rgb = read_image(label_root / scene / f"frame_{cases[0].frame_id}.jpg")
    ref_panel = fit_panel(first_rgb, width=reference_width - 18, height=panel_height + 20)
    cv2.putText(canvas, scene.replace("_", " ").title(), (margin + 4, margin + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (35, 80, 180), 1, cv2.LINE_AA)
    ref_y = margin + header_h + (row_count * panel_height - ref_panel.shape[0]) // 2
    ref_x = margin + 5
    canvas[ref_y : ref_y + ref_panel.shape[0], ref_x : ref_x + ref_panel.shape[1]] = ref_panel
    cv2.putText(canvas, "RGB", (ref_x + 8, ref_y + ref_panel.shape[0] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (45, 45, 45), 1, cv2.LINE_AA)

    x0 = margin + reference_width
    y0 = margin + header_h
    for row_idx, label in enumerate(["GT", "Prior", "Ours"]):
        label_panel = row_label(label, row_label_width, panel_height)
        y = y0 + row_idx * panel_height
        canvas[y : y + panel_height, x0 : x0 + row_label_width] = label_panel

    manifests: list[dict[str, object]] = []
    for idx, case in enumerate(cases):
        panels, manifest = make_case_panels(
            case,
            label_root=label_root,
            baseline_2d_root=baseline_2d_root,
            baseline_2d_fallback_root=baseline_2d_fallback_root,
            baseline_3d_root=baseline_3d_root,
            ours_2d_root=ours_2d_root,
            ours_3d_root=ours_3d_root,
            ours_3d_selection=ours_3d_selection,
            panel_width=panel_width,
            panel_height=panel_height,
        )
        base_x = x0 + row_label_width + idx * 2 * panel_width
        canvas[margin : margin + header_h, base_x : base_x + panel_width] = text_header("2D OVS", panel_width, header_h)
        canvas[margin : margin + header_h, base_x + panel_width : base_x + 2 * panel_width] = text_header("3D OVS", panel_width, header_h)
        for row_idx in range(3):
            y = y0 + row_idx * panel_height
            canvas[y : y + panel_height, base_x : base_x + panel_width] = panels[row_idx][0]
            canvas[y : y + panel_height, base_x + panel_width : base_x + 2 * panel_width] = panels[row_idx][1]
        cv2.rectangle(canvas, (base_x, y0), (base_x + 2 * panel_width, y0 + row_count * panel_height), (226, 226, 226), 1)
        query = f'"{case.query.title()}"'
        text_size = cv2.getTextSize(query, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)[0]
        qx = base_x + panel_width - text_size[0] // 2
        qy = y0 + row_count * panel_height + 24
        cv2.putText(canvas, query, (qx, qy), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (24, 24, 24), 1, cv2.LINE_AA)
        manifests.append(manifest)
    return canvas, manifests


def write_report(path: Path, payload: dict[str, object]) -> None:
    lines = [
        "# LERF 2D and 3D OVS Qualitative",
        "",
        f"Figure: `{payload['figure']}`",
        "",
        "Layout: each query has GT, Prior, and Ours rows with separate 2D rendered-view OVS and 3D direct-selection OVS panels. The Prior row uses LangSplatV2 for 2D OVS and Dr. Splat for 3D OVS.",
        "",
        "| Scene | Frame | Query | 2D Prior | 3D Prior | Fallback? | Prior 2D IoU | Prior 3D IoU | Ours 3D IoU |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for case in payload["cases"]:
        lines.append(
            "| {scene} | {frame_id} | `{query}` | {baseline_2d} | {baseline_3d} | {fallback} | {baseline_2d_iou:.4f} | {baseline_3d_iou:.4f} | {ours_3d_iou:.4f} |".format(
                fallback="yes" if case["baseline_2d_fallback_used"] else "no",
                **case,
            )
        )
    lines.extend(
        [
            "",
            "Note: 2D prior panels show the locally reproduced LangSplatV2 heatmap/RGB visualizations to match the rendered-view style of the Ours 2D panels; IoU is still computed from the corresponding chosen masks. 3D prior panels show Dr. Splat direct-selection masks rendered alone on a blank canvas, matching the 3D OVS display style of our selected primitive masks.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-root", default="/mnt/pool/sqy/3d_understanding/lerf_ovs/label")
    parser.add_argument("--baseline-2d-root", default="output/baselines/langsplatv2/lerf_compat_20260518")
    parser.add_argument("--baseline-2d-fallback-root", default="output/baselines/langsplat/lerf_compat_20260518")
    parser.add_argument("--baseline-3d-root", default="output/baselines/dr_splat/lerf_compat_20260519")
    parser.add_argument("--ours-2d-root", default="output/radio_gs/freeze_eval")
    parser.add_argument("--ours-3d-root", default="output/radio_gs/lerf_direct3d_prompt_ensemble_policy_masks_20260528")
    parser.add_argument("--ours-3d-selection", default="thr0p65")
    parser.add_argument("--panel-width", type=int, default=230)
    parser.add_argument("--panel-height", type=int, default=140)
    parser.add_argument("--output", default="paper/figures/lerf_2d3d_ovs_qualitative.png")
    parser.add_argument("--manifest", default="paper/artifacts/lerf_2d3d_ovs_qualitative_manifest.json")
    parser.add_argument("--report", default="paper/artifacts/lerf_2d3d_ovs_qualitative.md")
    args = parser.parse_args()

    blocks: list[np.ndarray] = []
    cases_payload: list[dict[str, object]] = []
    for group in DEFAULT_OVS_2D3D_GROUPS:
        block, manifests = make_scene_block(
            group,
            label_root=Path(args.label_root),
            baseline_2d_root=Path(args.baseline_2d_root),
            baseline_2d_fallback_root=Path(args.baseline_2d_fallback_root),
            baseline_3d_root=Path(args.baseline_3d_root),
            ours_2d_root=Path(args.ours_2d_root),
            ours_3d_root=Path(args.ours_3d_root),
            ours_3d_selection=args.ours_3d_selection,
            panel_width=args.panel_width,
            panel_height=args.panel_height,
            reference_width=168,
            row_label_width=118,
        )
        blocks.append(block)
        cases_payload.extend(manifests)

    sep = np.full((12, blocks[0].shape[1], 3), 255, dtype=np.uint8)
    figure = np.vstack([item for block in blocks for item in (block, sep)][:-1])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), figure)
    payload = {
        "figure": rel_or_str(output),
        "script": "radio_gs/scripts/compose_lerf_ovs_submission_qualitative.py",
        "baseline_2d_root": rel_or_str(Path(args.baseline_2d_root)),
        "baseline_2d_fallback_root": rel_or_str(Path(args.baseline_2d_fallback_root)),
        "baseline_3d_root": rel_or_str(Path(args.baseline_3d_root)),
        "ours_2d_root": rel_or_str(Path(args.ours_2d_root)),
        "ours_3d_root": rel_or_str(Path(args.ours_3d_root)),
        "ours_3d_selection": args.ours_3d_selection,
        "cases": cases_payload,
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_report(Path(args.report), payload)
    print(f"Wrote {output}")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
