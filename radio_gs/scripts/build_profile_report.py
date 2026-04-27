#!/usr/bin/env python3
"""Aggregate command profiles into a compact markdown table."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_time_log(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    fields = {}
    patterns = {
        "wall": r"Elapsed \(wall clock\) time .*: (.+)",
        "cpu": r"Percent of CPU this job got: (.+)",
        "max_rss_kb": r"Maximum resident set size \(kbytes\): (.+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        fields[key] = match.group(1).strip() if match else "-"
    if fields["wall"] == "-":
        match = re.search(r"^real\s+(.+)$", text, flags=re.MULTILINE)
        if match:
            fields["wall"] = match.group(1).strip() + " s"
    if fields["cpu"] == "-":
        fields["cpu"] = "shell-time"
    return fields


def parse_gpu_log(path: Path) -> tuple[str, str, str]:
    peak_mem = 0.0
    peak_util = 0.0
    total_util = 0.0
    samples = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        util = float(parts[2])
        mem = float(parts[3])
        peak_util = max(peak_util, util)
        peak_mem = max(peak_mem, mem)
        total_util += util
        samples += 1
    mean_util = total_util / samples if samples else 0.0
    return f"{peak_mem:.0f}", f"{peak_util:.0f}", f"{mean_util:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build markdown summary from profile directories")
    parser.add_argument("profile_dirs", nargs="+", help="Directories created by profile_command.sh")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    lines = [
        "# Efficiency Profile Summary",
        "",
        "| Profile | Wall Time | Peak GPU Mem (MiB) | Peak GPU Util (%) | Mean GPU Util (%) | Max RSS (KB) | CPU % |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for profile_dir in [Path(p) for p in args.profile_dirs]:
        time_info = parse_time_log(profile_dir / "time.log")
        peak_mem, peak_util, mean_util = parse_gpu_log(profile_dir / "gpu_metrics.csv")
        lines.append(
            f"| `{profile_dir}` | {time_info['wall']} | {peak_mem} | {peak_util} | {mean_util} | "
            f"{time_info['max_rss_kb']} | {time_info['cpu']} |"
        )

    output = "\n".join(lines) + "\n"
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
