#!/usr/bin/env python3
"""Summarize LangSplatV2 LERF compatibility eval logs."""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("output/baselines/langsplatv2/lerf_compat_20260518")


@dataclass(frozen=True)
class EvalRow:
    scene: str
    index: int
    checkpoint: int
    mask_thresh: float
    miou: float
    locacc: float
    query_count: int | None
    log_path: str


def _require_match(pattern: str, text: str, label: str) -> re.Match[str]:
    match = re.search(pattern, text, re.MULTILINE)
    if match is None:
        raise ValueError(f"missing {label}")
    return match


def _scene_index_from_path(path: Path) -> tuple[str, int]:
    parent = path.parent.name
    if "_" not in parent:
        return parent, 0
    scene, maybe_index = parent.rsplit("_", 1)
    if maybe_index.isdigit():
        return scene, int(maybe_index)
    return parent, 0


def _parse_chosen_levels(text: str) -> list[int] | None:
    match = re.search(r"chosen_lvl:\s*\n(\[[^\]]*\])", text, re.MULTILINE)
    if match is None:
        return None
    try:
        value = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return None
    if not isinstance(value, list):
        return None
    return [int(item) for item in value]


def parse_eval_log(path: str | Path) -> EvalRow:
    log_path = Path(path)
    text = log_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"empty eval log: {log_path}")

    path_scene, index = _scene_index_from_path(log_path)
    scene_match = re.search(r" - ([A-Za-z0-9_]+) - INFO - checkpoint:", text)
    scene = scene_match.group(1) if scene_match else path_scene
    checkpoint = int(_require_match(r"checkpoint:\s*([0-9]+)", text, "checkpoint").group(1))
    mask_thresh = float(_require_match(r"trunc thresh:\s*([0-9.]+)", text, "mask threshold").group(1))
    miou = float(_require_match(r"iou chosen:\s*([0-9.]+)", text, "iou chosen").group(1))
    locacc = float(
        _require_match(r"Localization accuracy:\s*([0-9.]+)", text, "Localization accuracy").group(1)
    )
    levels = _parse_chosen_levels(text)
    query_count = len(levels) if levels is not None else None
    return EvalRow(
        scene=scene,
        index=index,
        checkpoint=checkpoint,
        mask_thresh=mask_thresh,
        miou=miou,
        locacc=locacc,
        query_count=query_count,
        log_path=str(log_path),
    )


def _latest_completed_logs(root: Path) -> list[Path]:
    candidates: dict[tuple[str, int], Path] = {}
    for log_path in sorted((root / "eval").glob("*/*.log")):
        if log_path.stat().st_size == 0:
            continue
        try:
            row = parse_eval_log(log_path)
        except ValueError:
            continue
        key = (row.scene, row.index)
        previous = candidates.get(key)
        if previous is None or log_path.name > previous.name:
            candidates[key] = log_path
    return [candidates[key] for key in sorted(candidates)]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _weighted(rows: list[EvalRow], metric: str) -> float | None:
    usable = [row for row in rows if row.query_count is not None]
    total = sum(row.query_count or 0 for row in usable)
    if total == 0:
        return None
    return sum(float(getattr(row, metric)) * float(row.query_count or 0) for row in usable) / total


def build_summary(root: str | Path = DEFAULT_ROOT) -> dict[str, Any]:
    root_path = Path(root)
    rows = [parse_eval_log(path) for path in _latest_completed_logs(root_path)]
    rows = sorted(rows, key=lambda row: (row.scene, row.index))
    query_total = sum(row.query_count or 0 for row in rows)
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
        "# LangSplatV2 LERF Compatibility Summary",
        "",
        f"- root: `{summary['root']}`",
        f"- completed scene/index rows: {len(summary['completed_rows'])}",
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
        "| Scene | Index | Checkpoint | Mask Thresh | Queries | LocAcc | mIoU | Log |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["completed_rows"]:
        lines.append(
            "| {scene} | {index} | {checkpoint} | {mask_thresh:.4f} | {queries} | "
            "{locacc:.4f} | {miou:.4f} | `{log}` |".format(
                scene=row["scene"],
                index=row["index"],
                checkpoint=row["checkpoint"],
                mask_thresh=row["mask_thresh"],
                queries=row["query_count"] if row["query_count"] is not None else "-",
                locacc=row["locacc"],
                miou=row["miou"],
                log=row["log_path"],
            )
        )
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="LangSplatV2 compatibility root")
    parser.add_argument("--out-json", default=None, help="Optional JSON summary output path")
    parser.add_argument("--out-md", default=None, help="Optional Markdown summary output path")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = build_summary(args.root)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    markdown = render_markdown(summary)
    if args.out_md:
        Path(args.out_md).write_text(markdown, encoding="utf-8")
    print(markdown, end="")


if __name__ == "__main__":
    main()
