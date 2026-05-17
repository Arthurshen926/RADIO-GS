"""Build a compact alpha/depth boundary-case figure from diagnostic overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "output" / "radio_gs" / "reports"
DEFAULT_REPORT_JSON = REPORT_DIR / "alpha_depth_boundary_alignment_report.json"
DEFAULT_SWEEP_JSON = REPORT_DIR / "lerf_sam3_box_global_threshold_sweep_20260517_geometry.json"
DEFAULT_OUTPUT_PNG = REPO_ROOT / "paper" / "figures" / "alpha_depth_boundary_cases.png"
DEFAULT_OUTPUT_JSON = REPORT_DIR / "alpha_depth_boundary_case_figure_manifest.json"
DEFAULT_OUTPUT_MD = REPORT_DIR / "alpha_depth_boundary_case_figure_manifest.md"


def _case_score(case: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(case.get("boundary_error", 0.0)),
        float(case.get("discontinuity_error_boundary_mean", 0.0)),
        -float(case.get("iou", 0.0)),
    )


def select_cases(cases: list[dict[str, Any]], *, max_cases: int = 6) -> list[dict[str, Any]]:
    """Select compact worst-case examples while preserving scene coverage first."""
    ranked = sorted(cases, key=_case_score, reverse=True)
    selected: list[dict[str, Any]] = []
    seen_scenes: set[str] = set()
    for case in ranked:
        scene = str(case.get("scene", ""))
        if scene in seen_scenes:
            continue
        selected.append(case)
        seen_scenes.add(scene)
        if len(selected) >= max_cases:
            return selected
    selected_ids = {id(case) for case in selected}
    for case in ranked:
        if id(case) in selected_ids:
            continue
        selected.append(case)
        if len(selected) >= max_cases:
            break
    return selected


def _source_root_from_sweep(sweep_json: str | Path, *, run_label: str = "pad16_geometry") -> Path:
    payload = json.loads(Path(sweep_json).read_text(encoding="utf-8"))
    for run in payload.get("runs", []):
        if str(run.get("label", "")) == run_label:
            return Path(str(run["source_root"]))
    raise KeyError(f"Could not find run {run_label!r} in {sweep_json}")


def build_manifest(
    report_json: str | Path = DEFAULT_REPORT_JSON,
    *,
    source_root: str | Path | None = None,
    sweep_json: str | Path = DEFAULT_SWEEP_JSON,
    max_cases: int = 6,
) -> dict[str, Any]:
    report_path = Path(report_json)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    root = Path(source_root) if source_root is not None else _source_root_from_sweep(sweep_json)
    selected = select_cases(list(payload.get("worst_geometry_cases", [])), max_cases=max_cases)
    cases: list[dict[str, Any]] = []
    for rank, case in enumerate(selected, start=1):
        overlay_rel = Path(str(case["geometry_overlay_path"]))
        overlay_path = overlay_rel if overlay_rel.is_absolute() else root / overlay_rel
        cases.append(
            {
                "rank": rank,
                "scene": str(case.get("scene", "")),
                "frame": str(case.get("frame", "")),
                "category": str(case.get("category", "")),
                "iou": round(float(case.get("iou", 0.0)), 4),
                "boundary_error": round(float(case.get("boundary_error", 0.0)), 4),
                "discontinuity_error_boundary_mean": round(
                    float(case.get("discontinuity_error_boundary_mean", 0.0)), 4
                ),
                "overlay_path": str(overlay_path),
            }
        )
    return {
        "source_report": str(report_path),
        "source_root": str(root),
        "selection": "worst boundary-error cases with one-pass scene coverage",
        "max_cases": int(max_cases),
        "cases": cases,
    }


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _fit_image(image: Image.Image, *, width: int) -> Image.Image:
    scale = float(width) / max(float(image.width), 1.0)
    height = max(1, int(round(image.height * scale)))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _caption(case: dict[str, Any]) -> str:
    rank = int(case.get("rank", 1))
    return (
        f"{rank}. {case['scene']} / {case['category']} | "
        f"IoU {float(case['iou']):.2f}, BE {float(case['boundary_error']):.2f}"
    )


def build_case_figure(manifest: dict[str, Any], *, tile_width: int = 360, columns: int = 3) -> Image.Image:
    cases = list(manifest.get("cases", []))
    if not cases:
        raise ValueError("No cases selected for alpha/depth case figure")
    font = _font(15)
    caption_h = 42
    padding = 12
    fitted: list[tuple[dict[str, Any], Image.Image]] = []
    tile_h = 0
    for case in cases:
        image = Image.open(case["overlay_path"]).convert("RGB")
        resized = _fit_image(image, width=tile_width)
        fitted.append((case, resized))
        tile_h = max(tile_h, resized.height + caption_h)
    rows = (len(fitted) + columns - 1) // columns
    canvas_w = columns * tile_width + (columns + 1) * padding
    canvas_h = rows * tile_h + (rows + 1) * padding
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (case, image) in enumerate(fitted):
        row = idx // columns
        col = idx % columns
        x = padding + col * (tile_width + padding)
        y = padding + row * (tile_h + padding)
        draw.text((x, y), _caption(case), fill=(20, 20, 20), font=font)
        canvas.paste(image, (x, y + caption_h))
    return canvas


def build_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# Alpha/Depth Boundary Case Figure Manifest",
        "",
        f"- Source report: `{manifest.get('source_report', '')}`",
        f"- Source root: `{manifest.get('source_root', '')}`",
        f"- Selection: {manifest.get('selection', '')}",
        "",
        "| Rank | Scene | Frame | Category | IoU | Boundary error | Disc. error boundary | Overlay |",
        "|---:|---|---|---|---:|---:|---:|---|",
    ]
    for case in manifest.get("cases", []):
        lines.append(
            "| {rank} | {scene} | `{frame}` | {category} | {iou:.4f} | {be:.4f} | {disc:.4f} | `{overlay}` |".format(
                rank=int(case.get("rank", 1)),
                scene=case["scene"],
                frame=case["frame"],
                category=case["category"],
                iou=float(case["iou"]),
                be=float(case["boundary_error"]),
                disc=float(case.get("discontinuity_error_boundary_mean", 0.0)),
                overlay=case["overlay_path"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_case_figure(
    manifest: dict[str, Any],
    *,
    output_png: str | Path = DEFAULT_OUTPUT_PNG,
    output_json: str | Path = DEFAULT_OUTPUT_JSON,
    output_md: str | Path = DEFAULT_OUTPUT_MD,
    tile_width: int = 360,
    columns: int = 3,
) -> dict[str, Path]:
    png_path = Path(output_png)
    json_path = Path(output_json)
    md_path = Path(output_md)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    image = build_case_figure(manifest, tile_width=tile_width, columns=columns)
    image.save(png_path)
    json_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(manifest), encoding="utf-8")
    return {"png": png_path, "json": json_path, "markdown": md_path}


def main(argv: list[str] | None = None) -> dict[str, Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report_json", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--sweep_json", default=str(DEFAULT_SWEEP_JSON))
    parser.add_argument("--source_root", default="")
    parser.add_argument("--max_cases", type=int, default=6)
    parser.add_argument("--tile_width", type=int, default=360)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--output_png", default=str(DEFAULT_OUTPUT_PNG))
    parser.add_argument("--output_json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output_md", default=str(DEFAULT_OUTPUT_MD))
    args = parser.parse_args(argv)

    manifest = build_manifest(
        args.report_json,
        source_root=args.source_root or None,
        sweep_json=args.sweep_json,
        max_cases=args.max_cases,
    )
    paths = write_case_figure(
        manifest,
        output_png=args.output_png,
        output_json=args.output_json,
        output_md=args.output_md,
        tile_width=args.tile_width,
        columns=args.columns,
    )
    print(f"Wrote {paths['png']}")
    print(f"Wrote {paths['json']}")
    print(f"Wrote {paths['markdown']}")
    return paths


if __name__ == "__main__":
    main()
