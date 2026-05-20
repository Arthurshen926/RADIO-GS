#!/usr/bin/env python3
"""Summarize GAGS LERF compatibility eval logs."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("output/baselines/gags/lerf_compat_20260519")


@dataclass(frozen=True)
class EvalRow:
    scene: str
    checkpoint: int
    mask_thresh: float
    miou: float
    locacc: float
    query_count: int
    full_camera_list: bool
    log_path: str


def _require_match(pattern: str, text: str, label: str) -> re.Match[str]:
    match = re.search(pattern, text, re.MULTILINE)
    if match is None:
        raise ValueError(f"missing {label}")
    return match


def _scene_from_log(path: Path, text: str) -> str:
    match = re.search(r" - ([A-Za-z0-9_]+) - INFO - ", text)
    if match is not None:
        return match.group(1)
    for parent in path.parents:
        if parent.name in {"figurines", "ramen", "teatime", "waldo_kitchen"}:
            return parent.name
    raise ValueError(f"cannot infer scene from {path}")


def _checkpoint_from_path(path: Path) -> int:
    for parent in path.parents:
        match = re.fullmatch(r"ours_([0-9]+)", parent.name)
        if match is not None:
            return int(match.group(1))
    raise ValueError(f"cannot infer checkpoint from {path}")


def _query_count_from_eval_lines(text: str) -> int:
    totals = [
        int(match.group(1))
        for match in re.finditer(r"eval:\s*[0-9]+\s+acc_num:\s*[0-9]+/([0-9]+)\s+mean_iou:", text)
    ]
    if not totals:
        raise ValueError("missing per-frame query counts")
    return sum(totals)


def parse_eval_log(path: str | Path) -> EvalRow:
    log_path = Path(path)
    text = log_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"empty eval log: {log_path}")

    scene = _scene_from_log(log_path, text)
    return EvalRow(
        scene=scene,
        checkpoint=_checkpoint_from_path(log_path),
        mask_thresh=float(_require_match(r"trunc thresh:\s*([0-9.]+)", text, "mask threshold").group(1)),
        miou=float(_require_match(r"iou chosen:\s*([0-9.]+)", text, "iou chosen").group(1)),
        locacc=float(
            _require_match(r"Localization accuracy:\s*([0-9.]+)", text, "Localization accuracy").group(1)
        ),
        query_count=_query_count_from_eval_lines(text),
        full_camera_list="Using the full camera list for label-frame evaluation" in text,
        log_path=str(log_path),
    )


def _latest_completed_logs(root: Path) -> list[Path]:
    candidates: dict[str, Path] = {}
    for log_path in sorted(root.glob("*/train/ours_*/eval/*.log")):
        if log_path.stat().st_size == 0:
            continue
        try:
            row = parse_eval_log(log_path)
        except ValueError:
            continue
        previous = candidates.get(row.scene)
        if previous is None or log_path.name > previous.name:
            candidates[row.scene] = log_path
    return [candidates[scene] for scene in sorted(candidates)]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _weighted(rows: list[EvalRow], metric: str) -> float | None:
    total = sum(row.query_count for row in rows)
    if total == 0:
        return None
    return sum(float(getattr(row, metric)) * row.query_count for row in rows) / total


def build_summary(root: str | Path = DEFAULT_ROOT) -> dict[str, Any]:
    root_path = Path(root)
    rows = [parse_eval_log(path) for path in _latest_completed_logs(root_path)]
    rows = sorted(rows, key=lambda row: row.scene)
    query_total = sum(row.query_count for row in rows)
    return {
        "root": str(root_path),
        "completed_rows": [asdict(row) for row in rows],
        "scene_mean": {
            "locacc": _mean([row.locacc for row in rows]),
            "miou": _mean([row.miou for row in rows]),
        },
        "object_weighted": {
            "query_count": query_total,
            "locacc": _weighted(rows, "locacc"),
            "miou": _weighted(rows, "miou"),
        },
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# GAGS LERF Compatibility Summary",
        "",
        f"- root: `{summary['root']}`",
        f"- completed scenes: {len(summary['completed_rows'])}",
        (
            "- scene mean: "
            f"LocAcc {summary['scene_mean']['locacc']}, "
            f"mIoU {summary['scene_mean']['miou']}"
        ),
        (
            "- object weighted: "
            f"LocAcc {summary['object_weighted']['locacc']}, "
            f"mIoU {summary['object_weighted']['miou']}, "
            f"queries {summary['object_weighted']['query_count']}"
        ),
        "",
        "| Scene | Checkpoint | Mask Thresh | Queries | LocAcc | mIoU | Full Camera List | Log |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["completed_rows"]:
        lines.append(
            "| {scene} | {checkpoint} | {mask_thresh:.4f} | {queries} | {locacc:.4f} | "
            "{miou:.4f} | {full_camera_list} | `{log}` |".format(
                scene=row["scene"],
                checkpoint=row["checkpoint"],
                mask_thresh=row["mask_thresh"],
                queries=row["query_count"],
                locacc=row["locacc"],
                miou=row["miou"],
                full_camera_list=row["full_camera_list"],
                log=row["log_path"],
            )
        )
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="GAGS compatibility root")
    parser.add_argument("--out-json", default=None, help="Optional JSON summary output path")
    parser.add_argument("--out-md", default=None, help="Optional Markdown summary output path")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = build_summary(args.root)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    markdown = render_markdown(summary)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(markdown, encoding="utf-8")
    print(markdown, end="")


if __name__ == "__main__":
    main()
