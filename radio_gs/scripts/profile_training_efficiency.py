#!/usr/bin/env python3
"""
profile_training_efficiency.py
--------------------------------
Reads gpu_metrics.csv and training logs to produce an efficiency summary:
- Training time (wall clock)
- Peak VRAM usage
- Mean GPU utilization
- Throughput (epochs/hour)

Usage:
    python radio_gs/scripts/profile_training_efficiency.py \
        --exp_dirs output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep \
                   output/radio_gs/lerf_ramen_v14_fdh_ws240_240ep
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "output" / "radio_gs"

def parse_hms_seconds(value: str) -> int:
    hours, minutes, seconds = (int(part) for part in value.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def parse_train_log(log_path: Path):
    """Extract wall time and epoch throughput from a training log."""
    if not log_path.exists():
        return None

    text = log_path.read_text(errors="replace")
    full_timestamps = re.findall(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]', text)
    clock_timestamps = re.findall(r'\[(\d{2}:\d{2}:\d{2})\]', text)
    epochs = [int(epoch) for epoch in re.findall(r'E(\d+)', text)]

    wall_hours = None
    t_start = None
    t_end = None
    timestamp_style = None

    if full_timestamps:
        timestamp_style = "datetime"
        t_start = full_timestamps[0]
        t_end = full_timestamps[-1]
        dt_start = datetime.strptime(t_start, "%Y-%m-%d %H:%M:%S")
        dt_end = datetime.strptime(t_end, "%Y-%m-%d %H:%M:%S")
        wall_hours = (dt_end - dt_start).total_seconds() / 3600
    elif clock_timestamps:
        timestamp_style = "clock"
        t_start = clock_timestamps[0]
        t_end = clock_timestamps[-1]
        start_seconds = parse_hms_seconds(t_start)
        end_seconds = parse_hms_seconds(t_end)
        if end_seconds < start_seconds:
            end_seconds += 24 * 3600
        wall_hours = (end_seconds - start_seconds) / 3600

    if wall_hours is None:
        return None

    max_epoch = max(epochs) if epochs else 0
    return {
        "path": str(log_path),
        "t_start": t_start,
        "t_end": t_end,
        "timestamp_style": timestamp_style,
        "wall_hours": wall_hours,
        "max_epoch": max_epoch,
        "epochs_per_hour": max_epoch / wall_hours if wall_hours > 0 and max_epoch > 0 else None,
    }

def parse_gpu_metrics(csv_path: Path):
    """Parse plain nvidia-smi sampling logs -> mean util, peak vram."""
    if not csv_path.exists():
        return None
    utils = []
    vrams = []
    for line in csv_path.read_text(errors="replace").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            utils.append(float(parts[2]))
            vrams.append(float(parts[3]))
        except ValueError:
            continue
    if not utils:
        return None
    return {
        "mean_util_pct": sum(utils) / len(utils),
        "peak_vram_mib": max(vrams),
        "samples": len(utils),
    }


def parse_time_log(time_log: Path):
    """Extract wall-clock seconds from profile time logs."""
    if not time_log.exists():
        return None

    text = time_log.read_text(errors="replace")
    match = re.search(r"Elapsed \(wall clock\) time .*: (.+)", text)
    if match:
        wall = match.group(1).strip()
        return {"wall": wall, "wall_seconds": None, "source": "gnu_time"}

    match = re.search(r"^real\s+([0-9.]+)$", text, flags=re.MULTILINE)
    if match:
        wall_seconds = float(match.group(1))
        return {
            "wall": f"{wall_seconds:.3f} s",
            "wall_seconds": wall_seconds,
            "source": "shell_time",
        }

    return None


def find_train_log(exp_dir: Path) -> Path | None:
    candidates = [
        exp_dir / "logs" / "training.log",
        OUTPUT_ROOT / f"{exp_dir.name}.train.log",
        exp_dir / "train.log",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def profile_eval_workloads(profile_root: Path):
    rows = []
    if not profile_root.exists():
        return rows
    for profile_dir in sorted(path for path in profile_root.iterdir() if path.is_dir()):
        time_info = parse_time_log(profile_dir / "time.log")
        gpu_info = parse_gpu_metrics(profile_dir / "gpu_metrics.csv")
        if not time_info and not gpu_info:
            continue
        rows.append(
            {
                "name": profile_dir.name,
                "time": time_info,
                "gpu": gpu_info,
            }
        )
    return rows

def profile_exp(exp_dir: Path):
    name = exp_dir.name
    log_path = find_train_log(exp_dir)
    log_info = parse_train_log(log_path) if log_path else None

    # Check if eval is done
    eval_done = (exp_dir / "lerf_eval_best" / "summary.json").exists()

    return {
        "name": name,
        "log": log_info,
        "eval_done": eval_done,
    }

def build_report(exp_dirs: list[Path]) -> str:
    profile_rows = profile_eval_workloads(OUTPUT_ROOT / "profiles")
    lines = []
    lines.append("# RADIO-GS Training Efficiency Profile")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    lines.append("")
    lines.append(
        "This report separates two evidence types: experiment-log training throughput and explicit profiled workloads with GPU telemetry."
    )
    lines.append("")
    lines.append("## Training Throughput")
    lines.append("")
    lines.append("| Experiment | Wall Time (h) | Epochs | Ep/hr | Eval Done |")
    lines.append("|---|---:|---:|---:|---|")

    for exp_dir in exp_dirs:
        p = profile_exp(exp_dir)
        name = p["name"]
        if p["log"]:
            wall = f"{p['log']['wall_hours']:.1f}"
            ep = str(p["log"]["max_epoch"])
            eph = f"{p['log']['epochs_per_hour']:.1f}" if p["log"]["epochs_per_hour"] else "—"
        else:
            wall = ep = eph = "—"
        eval_str = "✅" if p["eval_done"] else "⏳"
        lines.append(f"| `{name}` | {wall} | {ep} | {eph} | {eval_str} |")

    lines.append("")
    lines.append("## Profiled Workloads")
    lines.append("")
    lines.append("| Profile | Wall Time | Peak VRAM (MiB) | Mean GPU% | Samples |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in profile_rows:
        wall = row["time"]["wall"] if row["time"] else "—"
        peak_vram = f"{row['gpu']['peak_vram_mib']:.0f}" if row["gpu"] else "—"
        mean_util = f"{row['gpu']['mean_util_pct']:.2f}" if row["gpu"] else "—"
        samples = str(row["gpu"]["samples"]) if row["gpu"] else "—"
        lines.append(f"| `{row['name']}` | {wall} | {peak_vram} | {mean_util} | {samples} |")

    lines.append("")
    lines.append("## Notes")
    lines.append("- Training wall time is measured from the first to last timestamp in `logs/training.log` when present.")
    lines.append("- Time-only training logs are handled as same-day or overnight runs; this is sufficient for the current sub-24h LERF jobs.")
    lines.append("- Epochs/hr = max observed epoch / wall hours from the training log.")
    lines.append("- GPU telemetry is only available for explicitly profiled workloads under `output/radio_gs/profiles`; most training runs do not have per-run GPU metrics.")
    lines.append("")
    return "\n".join(lines) + "\n"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dirs", nargs="+",
        default=[str(p) for p in sorted(OUTPUT_ROOT.glob("lerf_*")) if p.is_dir()])
    parser.add_argument("--output", default=str(OUTPUT_ROOT / "reports" / "efficiency_profile.md"))
    args = parser.parse_args()

    exp_dirs = [Path(d) for d in args.exp_dirs]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = build_report(exp_dirs)
    out.write_text(text)
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
