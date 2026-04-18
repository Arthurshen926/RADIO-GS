#!/usr/bin/env python3
"""Summarize repeated eval logs into mean/std tables."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path


TABLE_PATTERN = re.compile(
    r"^(Oracle \(GT feat\)|Rendered \(adapted\)|Cross \(GT→render\))\s+"
    r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)",
    re.MULTILINE,
)


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def parse_rendered_row(log_path: Path) -> dict[str, float]:
    text = log_path.read_text(encoding="utf-8")
    for match in TABLE_PATTERN.finditer(text):
        mode = match.group(1)
        if mode != "Rendered (adapted)":
            continue
        return {
            "AbsRel": float(match.group(2)),
            "RMSE": float(match.group(3)),
            "Delta1": float(match.group(4)),
            "mIoU": float(match.group(5)),
            "PixAcc": float(match.group(6)),
            "Grnd_mAP": float(match.group(7)),
            "Grnd_IoU": float(match.group(8)),
            "Grnd_Cor": float(match.group(9)),
        }
    raise ValueError(f"Could not find rendered summary row in {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize repeated eval logs")
    parser.add_argument("logs", nargs="+", help="eval_rendered.py log files")
    parser.add_argument("--output", default=None, help="Optional markdown output path")
    args = parser.parse_args()

    rows = [(Path(log), parse_rendered_row(Path(log))) for log in args.logs]
    metrics = list(rows[0][1].keys())

    lines = [
        "# Seed Summary",
        "",
        "| Log | " + " | ".join(metrics) + " |",
        "|---|" + "|".join(["---:"] * len(metrics)) + "|",
    ]
    for log_path, row in rows:
        values = " | ".join(f"{row[key]:.4f}" for key in metrics)
        lines.append(f"| `{log_path}` | {values} |")

    lines.extend(["", "## Mean ± Std", "", "| Metric | Mean | Std |", "|---|---:|---:|"])
    for key in metrics:
        values = [row[key] for _, row in rows]
        lines.append(f"| {key} | {mean(values):.4f} | {std(values):.4f} |")

    output = "\n".join(lines) + "\n"
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
