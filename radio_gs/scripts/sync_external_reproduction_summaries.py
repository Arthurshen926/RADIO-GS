#!/usr/bin/env python3
"""Sync completed external baseline summaries into final_rows.yaml."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_FINAL_ROWS = Path("paper/artifacts/final_rows.yaml")
DEFAULT_GAGS_SUMMARY = Path("paper/artifacts/gags_lerf_summary.json")
DEFAULT_DRSPLAT_SUMMARY = Path("paper/artifacts/drsplat_lerf_summary.json")
DEFAULT_LEGAUSSIANS_SUMMARY = Path("paper/artifacts/legaussians_lerf_summary.json")
DEFAULT_SEMANTIC_GAUSSIANS_SUMMARY = Path(
    "output/baselines/semantic_gaussians/scannet_compat_20260520/semantic_gaussians_eval_metrics.json"
)
DEFAULT_LAGA_SUMMARY = Path("paper/artifacts/laga_lerf_summary.json")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: object) -> str:
    return f"{float(value):.4f}"


def _completion_phrase(count: int) -> str:
    if count == 4:
        return "all four LERF compatibility scenes completed"
    return f"{count}/4 LERF compatibility scenes completed"


def build_gags_status(summary: dict[str, Any]) -> str:
    rows = summary.get("completed_rows", [])
    scene_mean = summary["scene_mean"]
    weighted = summary["object_weighted"]
    return (
        f"{_completion_phrase(len(rows))} in the local GAGS compatibility rerun "
        "from Occam 30k RGB starts. Current summary: "
        f"scene-mean LocAcc {_fmt(scene_mean['locacc'])} / mIoU {_fmt(scene_mean['miou'])} "
        f"and object-weighted LocAcc {_fmt(weighted['locacc'])} / "
        f"mIoU {_fmt(weighted['miou'])} over {int(weighted['query_count'])} queries. "
        "This remains a compatibility rerun, not a strict released-checkpoint macro."
    )


def build_drsplat_status(summary: dict[str, Any]) -> str:
    scenes = summary.get("scenes", {})
    macro = summary["macro"]
    return (
        f"{_completion_phrase(len(scenes))} in the local Dr. Splat/VALA nested-mask "
        "compatibility rerun from Occam 30k RGB starts and PQ majority voting. "
        "Current summary: "
        f"mIoU {_fmt(macro['miou'])} / Acc@0.25 {_fmt(macro['acc025'])} / "
        f"Acc@0.5 {_fmt(macro['acc05'])} over {int(macro['count'])} objects; "
        f"missing rendered masks counted: {int(macro['missing'])}. "
        "This uses a local evaluator because upstream evaluation remains TBA."
    )


def build_legaussians_status(summary: dict[str, Any]) -> str:
    scenes = summary.get("scenes", {})
    scene_mean = summary["scene_mean"]
    weighted = summary["object_weighted"]
    return (
        f"{_completion_phrase(len(scenes))} in the local LEGaussians compatibility rerun "
        "using official quantize_features.py/train.py/render_mask.py outputs. Current summary: "
        f"scene-mean mIoU {_fmt(scene_mean['miou'])} / "
        f"Acc@0.25 {_fmt(scene_mean['acc025'])} / "
        f"Acc@0.5 {_fmt(scene_mean['acc05'])} and object-weighted "
        f"mIoU {_fmt(weighted['miou'])} over {int(weighted['count'])} objects; "
        f"missing rendered masks counted: {int(weighted['missing'])}. "
        "This remains a local LERF compatibility rerun."
    )


def build_semantic_gaussians_status(summary: dict[str, Any]) -> str:
    scenes = summary.get("scenes", {})
    mean_iou = summary["metrics"]["mean_iou"]
    return (
        f"all four ScanNet compatibility scenes completed in the local Semantic Gaussians "
        "RGB-GS/fusion/distill/label-PLY reproduction. Current summary: "
        f"ScanNet-20 label-PLY mean IoU {_fmt(mean_iou)} over {len(scenes)} scenes. "
        "This uses label-PLY evaluation because the extracted scenes provide *_vh_clean_2.labels.ply."
    )


def build_laga_status(summary: dict[str, Any]) -> str:
    scenes = summary.get("scenes", {})
    macro = summary["macro"]
    return (
        f"{_completion_phrase(len(scenes))} in the local LaGa compatibility rerun "
        "using train_scene.py, train_affinity_features.py, batch descriptor export, and nested-mask evaluation. "
        "Current summary: "
        f"mIoU {_fmt(macro['miou'])} / Acc@0.25 {_fmt(macro['acc025'])} / "
        f"Acc@0.5 {_fmt(macro['acc05'])} over {int(macro['count'])} objects; "
        f"missing rendered masks counted: {int(macro['missing'])}. "
        "This remains a local LERF compatibility rerun."
    )


def _update_queue_status(payload: dict[str, Any], method: str, status: str) -> bool:
    queue = payload.setdefault("external_reproduction_queue", {})
    for bucket in ("p0", "p1", "p2"):
        for row in queue.get(bucket, []):
            if row.get("method") == method:
                if row.get("status") == status:
                    return False
                row["status"] = status
                return True
    raise ValueError(f"method not found in external_reproduction_queue: {method}")


def sync_payload(
    payload: dict[str, Any],
    *,
    gags_summary_path: Path = DEFAULT_GAGS_SUMMARY,
    drsplat_summary_path: Path = DEFAULT_DRSPLAT_SUMMARY,
    legaussians_summary_path: Path = DEFAULT_LEGAUSSIANS_SUMMARY,
    semantic_gaussians_summary_path: Path = DEFAULT_SEMANTIC_GAUSSIANS_SUMMARY,
    laga_summary_path: Path = DEFAULT_LAGA_SUMMARY,
) -> bool:
    changed = False
    if gags_summary_path.exists():
        changed |= _update_queue_status(payload, "GAGS", build_gags_status(_read_json(gags_summary_path)))
    if drsplat_summary_path.exists():
        changed |= _update_queue_status(
            payload,
            "Dr. Splat",
            build_drsplat_status(_read_json(drsplat_summary_path)),
        )
    if legaussians_summary_path.exists():
        changed |= _update_queue_status(
            payload,
            "LEGaussians",
            build_legaussians_status(_read_json(legaussians_summary_path)),
        )
    if semantic_gaussians_summary_path.exists():
        changed |= _update_queue_status(
            payload,
            "Semantic Gaussians",
            build_semantic_gaussians_status(_read_json(semantic_gaussians_summary_path)),
        )
    if laga_summary_path.exists():
        changed |= _update_queue_status(payload, "LaGa", build_laga_status(_read_json(laga_summary_path)))
    return changed


def sync_final_rows(
    final_rows_path: str | Path = DEFAULT_FINAL_ROWS,
    *,
    gags_summary_path: str | Path = DEFAULT_GAGS_SUMMARY,
    drsplat_summary_path: str | Path = DEFAULT_DRSPLAT_SUMMARY,
    legaussians_summary_path: str | Path = DEFAULT_LEGAUSSIANS_SUMMARY,
    semantic_gaussians_summary_path: str | Path = DEFAULT_SEMANTIC_GAUSSIANS_SUMMARY,
    laga_summary_path: str | Path = DEFAULT_LAGA_SUMMARY,
) -> bool:
    path = Path(final_rows_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    changed = sync_payload(
        payload,
        gags_summary_path=Path(gags_summary_path),
        drsplat_summary_path=Path(drsplat_summary_path),
        legaussians_summary_path=Path(legaussians_summary_path),
        semantic_gaussians_summary_path=Path(semantic_gaussians_summary_path),
        laga_summary_path=Path(laga_summary_path),
    )
    if changed:
        path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return changed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-rows", type=Path, default=DEFAULT_FINAL_ROWS)
    parser.add_argument("--gags-summary", type=Path, default=DEFAULT_GAGS_SUMMARY)
    parser.add_argument("--drsplat-summary", type=Path, default=DEFAULT_DRSPLAT_SUMMARY)
    parser.add_argument("--legaussians-summary", type=Path, default=DEFAULT_LEGAUSSIANS_SUMMARY)
    parser.add_argument("--semantic-gaussians-summary", type=Path, default=DEFAULT_SEMANTIC_GAUSSIANS_SUMMARY)
    parser.add_argument("--laga-summary", type=Path, default=DEFAULT_LAGA_SUMMARY)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    changed = sync_final_rows(
        args.final_rows,
        gags_summary_path=args.gags_summary,
        drsplat_summary_path=args.drsplat_summary,
        legaussians_summary_path=args.legaussians_summary,
        semantic_gaussians_summary_path=args.semantic_gaussians_summary,
        laga_summary_path=args.laga_summary,
    )
    print("final_rows updated" if changed else "final_rows unchanged")


if __name__ == "__main__":
    main()
