"""Evaluate an unwarped nearest-view cached-teacher LERF baseline."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from radio_gs.scripts.eval_lerf_grounding import (
    DEFAULT_GT_FEATURE_ROOT,
    DEFAULT_LABEL_DIR,
    LERF_OVS_SCENES,
    SigLIP2SummaryHead,
    evaluate_scene,
    load_lerf_ovs_labels,
    load_or_generate_prompt_ensemble_embeddings,
    parse_prompt_templates,
    resolve_lerf_label_dir,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "radio_gs" / "lerf_nearest_view_cache_baseline_20260517"
DEFAULT_REPORT_MD = REPO_ROOT / "output" / "radio_gs" / "reports" / "lerf_nearest_view_cache_baseline.md"
DEFAULT_REPORT_JSON = REPO_ROOT / "output" / "radio_gs" / "reports" / "lerf_nearest_view_cache_baseline.json"
DEFAULT_REPORT_TEX = REPO_ROOT / "paper" / "lerf_nearest_view_cache_baseline_table.tex"
DEFAULT_SUMMARY_HEAD = REPO_ROOT / "checkpoints" / "siglip2_summary_head.pth"


@dataclass(frozen=True)
class FeatureFrame:
    frame_id: int
    feature_path: Path
    center: np.ndarray


@dataclass(frozen=True)
class FeatureIndex:
    scene: str
    root: Path
    frames: dict[int, FeatureFrame]


@dataclass(frozen=True)
class NearestMapping:
    target_frame: int
    source_frame: int
    distance: float
    target_feature_path: Path
    source_feature_path: Path


@dataclass(frozen=True)
class SceneResult:
    scene: str
    loc_acc: float
    miou: float
    n: int
    mean_nearest_distance: float


def _round4(value: float) -> float:
    return round(float(value), 4)


def _display_scene(scene: str) -> str:
    return {
        "figurines": "Figurines",
        "ramen": "Ramen",
        "teatime": "Teatime",
        "waldo_kitchen": "Waldo Kitchen",
    }.get(scene, scene)


def _latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


def _camera_center_from_c2w(c2w: np.ndarray) -> np.ndarray:
    return np.asarray(c2w[:3, 3], dtype=np.float32)


def _load_c2w_poses(path: Path) -> np.ndarray:
    raw = np.loadtxt(path, dtype=np.float32)
    return raw.reshape(-1, 4, 4)


def load_feature_index(feature_root: str | Path, scene: str) -> FeatureIndex:
    scene_root = Path(feature_root) / scene
    manifest = json.loads((scene_root / "frame_manifest.json").read_text(encoding="utf-8"))
    poses_c2w = _load_c2w_poses(scene_root / "traj_w_c.txt")
    frames: dict[int, FeatureFrame] = {}
    for row in manifest.get("frames", []):
        frame_id = int(row.get("frame_idx"))
        if frame_id < 1 or frame_id > len(poses_c2w):
            continue
        stem = str(row.get("saved_stem", f"rgb_{frame_id}"))
        path = scene_root / "backbone" / f"{stem}.pt"
        if not path.exists():
            continue
        frames[frame_id] = FeatureFrame(
            frame_id=frame_id,
            feature_path=path,
            center=_camera_center_from_c2w(poses_c2w[frame_id - 1]),
        )
    if not frames:
        raise ValueError(f"No cached feature frames found for {scene} under {scene_root}")
    return FeatureIndex(scene=scene, root=scene_root, frames=frames)


def build_nearest_mapping(index: FeatureIndex, target_frame_ids: Iterable[int]) -> dict[int, NearestMapping]:
    mapping: dict[int, NearestMapping] = {}
    for target_id in sorted({int(fid) for fid in target_frame_ids}):
        if target_id not in index.frames:
            continue
        target = index.frames[target_id]
        candidates = [frame for fid, frame in index.frames.items() if fid != target_id]
        if not candidates:
            continue
        source = min(candidates, key=lambda frame: float(np.linalg.norm(frame.center - target.center)))
        distance = float(np.linalg.norm(source.center - target.center))
        mapping[target_id] = NearestMapping(
            target_frame=target_id,
            source_frame=source.frame_id,
            distance=_round4(distance),
            target_feature_path=target.feature_path,
            source_feature_path=source.feature_path,
        )
    return mapping


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        os.symlink(source.resolve(), target)
    except OSError:
        shutil.copy2(source, target)


def materialize_nearest_cache(
    output_root: str | Path,
    scene: str,
    mapping: dict[int, NearestMapping],
) -> Path:
    scene_cache = Path(output_root) / "nearest_features" / scene
    backbone = scene_cache / "backbone"
    backbone.mkdir(parents=True, exist_ok=True)
    for row in mapping.values():
        _link_or_copy(row.source_feature_path, backbone / f"rgb_{row.target_frame}.pt")
    (scene_cache / "nearest_mapping.json").write_text(
        json.dumps(
            [
                {
                    "target_frame": row.target_frame,
                    "source_frame": row.source_frame,
                    "distance": row.distance,
                    "source_feature_path": str(row.source_feature_path),
                }
                for row in mapping.values()
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return scene_cache


def summarize_rows(rows: list[SceneResult], *, protocol: dict[str, Any]) -> dict[str, Any]:
    total_n = sum(row.n for row in rows)
    weighted_loc = sum(row.loc_acc * row.n for row in rows) / total_n if total_n else 0.0
    weighted_miou = sum(row.miou * row.n for row in rows) / total_n if total_n else 0.0
    return {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": protocol,
        "rows": [
            {
                "scene": row.scene,
                "loc_acc": _round4(row.loc_acc),
                "miou": _round4(row.miou),
                "n": int(row.n),
                "mean_nearest_distance": _round4(row.mean_nearest_distance),
            }
            for row in rows
        ],
        "macro": {
            "loc_acc": _round4(_mean(row.loc_acc for row in rows)),
            "miou": _round4(_mean(row.miou for row in rows)),
        },
        "weighted": {
            "loc_acc": _round4(weighted_loc),
            "miou": _round4(weighted_miou),
        },
        "mean_nearest_distance": _round4(_mean(row.mean_nearest_distance for row in rows)),
    }


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# LERF Nearest-View Cache Baseline",
        "",
        "Protocol: unwarped nearest-view cached teacher features. For each annotated target frame, the baseline substitutes the closest cached RADIO frame by camera-center distance, excluding the target frame itself, then runs the same LERF text scoring and thresholded-mask evaluator.",
        "",
        "| Scene | LocAcc | mIoU | N | Mean nearest distance |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary.get("rows", []):
        lines.append(
            "| {scene} | {loc:.4f} | {miou:.4f} | {n} | {dist:.4f} |".format(
                scene=row["scene"],
                loc=float(row["loc_acc"]),
                miou=float(row["miou"]),
                n=int(row["n"]),
                dist=float(row["mean_nearest_distance"]),
            )
        )
    macro = summary.get("macro", {})
    weighted = summary.get("weighted", {})
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "| Aggregate | LocAcc | mIoU |",
            "|---|---:|---:|",
            f"| Macro | {float(macro.get('loc_acc', 0.0)):.4f} | {float(macro.get('miou', 0.0)):.4f} |",
            f"| Query-weighted | {float(weighted.get('loc_acc', 0.0)):.4f} | {float(weighted.get('miou', 0.0)):.4f} |",
            "",
            "## Interpretation",
            "",
            "- This is a cache-only baseline, not a 3D scene representation.",
            "- The source feature map is not warped into the target camera, so the result measures how far a simple nearest-view cache can go without RADIO-GS reconstruction.",
            "- It should be reported separately from the same-frame RADIO teacher row and the rendered 3D feature-field row.",
            "",
        ]
    )
    return "\n".join(lines)


def build_latex_table(summary: dict[str, Any]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Nearest-view cached-teacher baseline on LERF-OVS. Each annotated target frame uses the closest cached RADIO feature map by camera-center distance, excluding the target frame itself; the feature map is not warped into the target camera.}",
        "\\label{tab:nearest_view_cache_baseline}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Scene & LocAcc & mIoU & N & Dist. \\\\",
        "\\midrule",
    ]
    for row in summary.get("rows", []):
        lines.append(
            "{scene} & {loc:.4f} & {miou:.4f} & {n} & {dist:.4f} \\\\".format(
                scene=_latex_escape(str(row["scene"])),
                loc=float(row["loc_acc"]),
                miou=float(row["miou"]),
                n=int(row["n"]),
                dist=float(row["mean_nearest_distance"]),
            )
        )
    macro = summary.get("macro", {})
    lines.extend(
        [
            "\\midrule",
            "Macro & {loc:.4f} & {miou:.4f} & -- & {dist:.4f} \\\\".format(
                loc=float(macro.get("loc_acc", 0.0)),
                miou=float(macro.get("miou", 0.0)),
                dist=float(summary.get("mean_nearest_distance", 0.0)),
            ),
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    summary: dict[str, Any],
    markdown_path: str | Path = DEFAULT_REPORT_MD,
    json_path: str | Path = DEFAULT_REPORT_JSON,
    latex_path: str | Path = DEFAULT_REPORT_TEX,
) -> dict[str, Path]:
    markdown_out = Path(markdown_path)
    json_out = Path(json_path)
    latex_out = Path(latex_path)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    latex_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.write_text(build_markdown(summary), encoding="utf-8")
    json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    latex_out.write_text(build_latex_table(summary), encoding="utf-8")
    return {"markdown": markdown_out, "json": json_out, "latex": latex_out}


def _load_projection(device: torch.device, summary_head_path: str | Path) -> torch.nn.Module:
    proj = SigLIP2SummaryHead.from_extracted_weights(str(summary_head_path))
    proj = proj.to(device)
    return (proj.half() if device.type == "cuda" else proj.float()).eval()


def _scene_categories(label_dir: str, scene: str) -> list[str]:
    _, categories, _, _ = load_lerf_ovs_labels(label_dir, scene)
    return categories


def evaluate_nearest_view_baseline(
    *,
    scenes: Iterable[str],
    label_dir: str,
    feature_root: str | Path,
    output_root: str | Path,
    summary_head_path: str | Path,
    text_cache_root: str | Path,
    device: torch.device,
    iou_threshold: float = 0.60,
    scoring: str = "softmax_scene",
    heatmap_upsample: int = 4,
    temperatures: dict[str, float] | None = None,
    prompt_templates: list[str] | None = None,
) -> dict[str, Any]:
    temps = temperatures or {
        "figurines": 50.0,
        "ramen": 40.0,
        "teatime": 25.0,
        "waldo_kitchen": 25.0,
    }
    proj = _load_projection(device, summary_head_path)
    rows: list[SceneResult] = []
    scene_payloads: dict[str, Any] = {}
    for scene in scenes:
        frame_annotations, _, _, _ = load_lerf_ovs_labels(label_dir, scene)
        index = load_feature_index(feature_root, scene)
        mapping = build_nearest_mapping(index, frame_annotations.keys())
        nearest_cache = materialize_nearest_cache(output_root, scene, mapping)
        categories = _scene_categories(label_dir, scene)
        text_cache = Path(text_cache_root) / f"{scene}_siglip2_text_embeddings.pt"
        text_embeddings = load_or_generate_prompt_ensemble_embeddings(
            categories,
            device,
            cache_path=str(text_cache),
            prompt_templates=prompt_templates or ["{query}"],
        )
        text_embeddings = text_embeddings.half() if device.type == "cuda" else text_embeddings.float()
        result = evaluate_scene(
            scene=scene,
            label_dir=label_dir,
            proj=proj,
            text_embeddings=text_embeddings,
            categories=categories,
            device=device,
            gt_feature_dir=str(nearest_cache),
            render_pipeline=None,
            iou_threshold=iou_threshold,
            scoring=scoring,
            heatmap_upsample=heatmap_upsample,
            temperature=float(temps.get(scene, 25.0)),
        )
        teacher = result.get("teacher") or result.get("gt") or {}
        distances = [row.distance for row in mapping.values()]
        rows.append(
            SceneResult(
                scene=_display_scene(scene),
                loc_acc=float(teacher.get("loc_acc", 0.0)),
                miou=float(teacher.get("miou", 0.0)),
                n=int(teacher.get("loc_total", 0)),
                mean_nearest_distance=_mean(distances),
            )
        )
        scene_payloads[scene] = {
            "nearest_cache": str(nearest_cache),
            "mapping_count": len(mapping),
            "mean_nearest_distance": _round4(_mean(distances)),
            "metrics": teacher,
            "temperature": float(temps.get(scene, 25.0)),
        }

    protocol = {
        "selection": "nearest_by_camera_center",
        "feature_source": "cached RADIO 1280-D teacher features",
        "target_frame_excluded": True,
        "warp": "none",
        "iou_threshold": iou_threshold,
        "scoring": scoring,
        "heatmap_upsample": heatmap_upsample,
        "prompt_templates": prompt_templates or ["{query}"],
    }
    summary = summarize_rows(rows, protocol=protocol)
    summary["scene_payloads"] = scene_payloads
    return summary


def _parse_scenes(raw: str) -> tuple[str, ...]:
    if raw == "all":
        return LERF_OVS_SCENES
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _parse_temperature_map(raw: str) -> dict[str, float]:
    if not raw:
        return {}
    values: dict[str, float] = {}
    for part in raw.split(","):
        if not part.strip():
            continue
        scene, value = part.split(":", 1)
        values[scene.strip()] = float(value)
    return values


def main(argv: list[str] | None = None) -> dict[str, Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", default="all")
    parser.add_argument("--label_dir", default=DEFAULT_LABEL_DIR)
    parser.add_argument("--feature_root", default=DEFAULT_GT_FEATURE_ROOT)
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--summary_head_weights", default=str(DEFAULT_SUMMARY_HEAD))
    parser.add_argument("--text_cache_root", default=str(DEFAULT_OUTPUT_ROOT / "text_cache"))
    parser.add_argument("--iou_threshold", type=float, default=0.60)
    parser.add_argument("--scoring", default="softmax_scene")
    parser.add_argument("--heatmap_upsample", type=int, default=4)
    parser.add_argument("--temperatures", default="figurines:50,ramen:40,teatime:25,waldo_kitchen:25")
    parser.add_argument("--prompt_templates", default="{query}")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_md", default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--output_json", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--output_tex", default=str(DEFAULT_REPORT_TEX))
    args = parser.parse_args(argv)

    label_dir = resolve_lerf_label_dir(args.label_dir)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    summary = evaluate_nearest_view_baseline(
        scenes=_parse_scenes(args.scenes),
        label_dir=label_dir,
        feature_root=args.feature_root,
        output_root=args.output_root,
        summary_head_path=args.summary_head_weights,
        text_cache_root=args.text_cache_root,
        device=device,
        iou_threshold=args.iou_threshold,
        scoring=args.scoring,
        heatmap_upsample=args.heatmap_upsample,
        temperatures=_parse_temperature_map(args.temperatures),
        prompt_templates=parse_prompt_templates(args.prompt_templates),
    )
    paths = write_outputs(summary, args.output_md, args.output_json, args.output_tex)
    print(f"Wrote {paths['markdown']}")
    print(f"Wrote {paths['json']}")
    print(f"Wrote {paths['latex']}")
    return paths


if __name__ == "__main__":
    main()
