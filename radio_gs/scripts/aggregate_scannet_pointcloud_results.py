#!/usr/bin/env python3
"""Aggregate ScanNet pointcloud evaluation JSON files into report tables."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RESULT_JSON = "scannet_pointcloud_radio_gs_results.json"
DEFAULT_PATTERN = "*_v43fair_best_hybrid_pointwise_gidx_official"
SPLITS = ("19", "15", "10")
DEFAULT_TARGETS = {
    "19": 0.3052,
    "15": 0.3150,
    "10": 0.4000,
}
DEFAULT_PROTOCOL_NOTE = (
    "v43fair uses label CE and is a label-supervised diagnostic / upper-bound; "
    "v44/no-label should be used for the fair protocol."
)


@dataclass(frozen=True)
class SplitMetric:
    miou: float | None
    macc: float | None
    num_valid: int | None
    target: float | None
    passes_target: bool | None


@dataclass(frozen=True)
class SceneRow:
    scene: str
    result_dir: str
    query_mode: str
    opacity_mode: str
    official_ok: bool | None
    official_issues: list[str]
    split_metrics: dict[str, SplitMetric]


@dataclass(frozen=True)
class AggregateSummary:
    rows: list[SceneRow]
    macro: dict[str, dict[str, float | None]]
    missing_scenes: list[str]
    patterns: list[str]
    targets: dict[str, float]
    require_official: bool
    eval_root: Path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _fmt_float(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _fmt_int(value: int | float | None) -> str:
    if value is None:
        return "-"
    return str(int(round(value)))


def _target_label(value: bool | None) -> str:
    if value is None:
        return "-"
    return "pass" if value else "fail"


def _bool_csv(value: bool | None) -> str:
    if value is None:
        return ""
    return "true" if value else "false"


def _split_expected_scenes(values: list[str] | None) -> list[str]:
    if not values:
        return []
    scenes: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                scenes.append(item)
    return sorted(set(scenes))


def _resolve_result_paths(eval_root: Path, patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for match in sorted(eval_root.glob(pattern)):
            result_path = match if match.name == RESULT_JSON else match / RESULT_JSON
            if not result_path.exists() or result_path in seen:
                continue
            seen.add(result_path)
            paths.append(result_path)
    return paths


def _scene_rows_from_json(
    path: Path,
    payload: dict[str, Any],
    targets: dict[str, float],
    require_official: bool,
) -> list[SceneRow]:
    scenes = payload.get("scenes")
    if not isinstance(scenes, dict):
        return []

    rows: list[SceneRow] = []
    for scene_key in sorted(scenes):
        entry = scenes[scene_key]
        if not isinstance(entry, dict):
            continue
        scene = str(entry.get("scene") or scene_key)
        query_mode = str(entry.get("query_mode") or payload.get("args", {}).get("query_mode") or "unknown")
        opacity_filter = entry.get("opacity_filter")
        opacity_mode = "unknown"
        if isinstance(opacity_filter, dict):
            opacity_mode = str(opacity_filter.get("mode") or "unknown")

        official_issues: list[str] = []
        official_ok: bool | None = None
        if require_official:
            if query_mode != "gaussian_index":
                official_issues.append(f"query_mode={query_mode}")
            if opacity_mode != "label_index":
                official_issues.append(f"opacity_filter.mode={opacity_mode}")
            official_ok = not official_issues

        split_metrics: dict[str, SplitMetric] = {}
        splits = entry.get("splits") if isinstance(entry.get("splits"), dict) else {}
        for split in SPLITS:
            split_entry = splits.get(split) if isinstance(splits, dict) else None
            split_entry = split_entry if isinstance(split_entry, dict) else {}
            miou = _as_float(split_entry.get("miou"))
            macc = _as_float(split_entry.get("macc"))
            num_valid = _as_int(split_entry.get("num_valid"))
            target = targets.get(split)
            passes_target = None if miou is None or target is None else miou >= target
            split_metrics[split] = SplitMetric(
                miou=miou,
                macc=macc,
                num_valid=num_valid,
                target=target,
                passes_target=passes_target,
            )

        rows.append(
            SceneRow(
                scene=scene,
                result_dir=str(path.parent),
                query_mode=query_mode,
                opacity_mode=opacity_mode,
                official_ok=official_ok,
                official_issues=official_issues,
                split_metrics=split_metrics,
            )
        )
    return rows


def _compute_macro(rows: list[SceneRow]) -> dict[str, dict[str, float | None]]:
    macro: dict[str, dict[str, float | None]] = {}
    for split in SPLITS:
        macro[split] = {}
        for field in ("miou", "macc", "num_valid"):
            values = [
                getattr(row.split_metrics[split], field)
                for row in rows
                if getattr(row.split_metrics[split], field) is not None
            ]
            macro[split][field] = sum(values) / len(values) if values else None
    return macro


def aggregate_results(
    eval_root: Path,
    patterns: list[str],
    targets: dict[str, float],
    *,
    require_official: bool = False,
    expected_scenes: list[str] | None = None,
) -> AggregateSummary:
    rows: list[SceneRow] = []
    for path in _resolve_result_paths(eval_root, patterns):
        rows.extend(_scene_rows_from_json(path, _read_json(path), targets, require_official))

    rows = sorted(rows, key=lambda row: (row.scene, row.result_dir))
    found_scenes = {row.scene for row in rows}
    expected = sorted(set(expected_scenes or []))
    missing_scenes = [scene for scene in expected if scene not in found_scenes]
    return AggregateSummary(
        rows=rows,
        macro=_compute_macro(rows),
        missing_scenes=missing_scenes,
        patterns=patterns,
        targets=targets,
        require_official=require_official,
        eval_root=eval_root,
    )


def render_markdown(summary: AggregateSummary, protocol_note: str | None = None) -> str:
    lines = [
        "# ScanNet Pointcloud Results",
        "",
        f"- Eval root: `{summary.eval_root}`",
        f"- Patterns: {', '.join(f'`{pattern}`' for pattern in summary.patterns)}",
        "- Split targets: "
        + ", ".join(f"split{split}={target:.4f}" for split, target in summary.targets.items())
        + " (user/local target placeholders, not official ProFuse numbers unless separately verified).",
    ]
    if protocol_note:
        lines.append(f"- Protocol note: {protocol_note}")
    if summary.require_official:
        lines.append("- Official check: requires query_mode=gaussian_index and opacity_filter.mode=label_index.")
    lines.extend(
        [
            "",
            "| scene | split19 mIoU | split19 mAcc | split19 num_valid | split19 target | "
            "split15 mIoU | split15 mAcc | split15 num_valid | split15 target | "
            "split10 mIoU | split10 mAcc | split10 num_valid | split10 target | "
            "query_mode | opacity mode | official | official issues |",
            "|---|---:|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---|---|---|---|---|",
        ]
    )

    for row in summary.rows:
        cells = [row.scene]
        for split in SPLITS:
            metric = row.split_metrics[split]
            cells.extend(
                [
                    _fmt_float(metric.miou),
                    _fmt_float(metric.macc),
                    _fmt_int(metric.num_valid),
                    _target_label(metric.passes_target),
                ]
            )
        cells.extend(
            [
                row.query_mode,
                row.opacity_mode,
                _bool_csv(row.official_ok) if row.official_ok is not None else "-",
                "; ".join(row.official_issues) if row.official_issues else "-",
            ]
        )
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(["", "## Macro Average", ""])
    lines.append("| split | mIoU | mAcc | num_valid | target |")
    lines.append("|---|---:|---:|---:|---:|")
    for split in SPLITS:
        metric = summary.macro[split]
        lines.append(
            f"| {split} | {_fmt_float(metric['miou'])} | {_fmt_float(metric['macc'])} | "
            f"{_fmt_int(metric['num_valid'])} | {summary.targets[split]:.4f} |"
        )

    missing = ", ".join(summary.missing_scenes) if summary.missing_scenes else "none"
    lines.extend(["", f"Missing expected scenes: {missing}", ""])
    return "\n".join(lines)


def write_csv(summary: AggregateSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scene", "result_dir", "query_mode", "opacity_mode", "official_ok", "official_issues"]
    for split in SPLITS:
        fieldnames.extend(
            [
                f"split{split}_miou",
                f"split{split}_macc",
                f"split{split}_num_valid",
                f"split{split}_target",
                f"split{split}_passes_target",
            ]
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary.rows:
            out = {
                "scene": row.scene,
                "result_dir": row.result_dir,
                "query_mode": row.query_mode,
                "opacity_mode": row.opacity_mode,
                "official_ok": _bool_csv(row.official_ok),
                "official_issues": "; ".join(row.official_issues),
            }
            for split in SPLITS:
                metric = row.split_metrics[split]
                out[f"split{split}_miou"] = _fmt_float(metric.miou)
                out[f"split{split}_macc"] = _fmt_float(metric.macc)
                out[f"split{split}_num_valid"] = _fmt_int(metric.num_valid)
                out[f"split{split}_target"] = _fmt_float(metric.target)
                out[f"split{split}_passes_target"] = _bool_csv(metric.passes_target)
            writer.writerow(out)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate ScanNet pointcloud RADIO-GS result JSON files."
    )
    parser.add_argument("--eval-root", default="output/scannet_pointcloud_eval")
    parser.add_argument(
        "--pattern",
        action="append",
        default=None,
        help=f"Directory or JSON glob under --eval-root. May be repeated. Default: {DEFAULT_PATTERN}",
    )
    parser.add_argument("--output_md", default=None)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--protocol-note", default=DEFAULT_PROTOCOL_NOTE)
    parser.add_argument("--require-official", action="store_true")
    parser.add_argument("--expected-scenes", nargs="*", default=None)
    parser.add_argument("--profuse-split19", type=float, default=DEFAULT_TARGETS["19"])
    parser.add_argument("--profuse-split15", type=float, default=DEFAULT_TARGETS["15"])
    parser.add_argument("--profuse-split10", type=float, default=DEFAULT_TARGETS["10"])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    targets = {
        "19": args.profuse_split19,
        "15": args.profuse_split15,
        "10": args.profuse_split10,
    }
    summary = aggregate_results(
        eval_root=Path(args.eval_root),
        patterns=args.pattern or [DEFAULT_PATTERN],
        targets=targets,
        require_official=args.require_official,
        expected_scenes=_split_expected_scenes(args.expected_scenes),
    )
    markdown = render_markdown(summary, protocol_note=args.protocol_note)
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)
    if args.output_csv:
        write_csv(summary, Path(args.output_csv))


if __name__ == "__main__":
    main()
