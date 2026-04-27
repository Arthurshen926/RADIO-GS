#!/usr/bin/env python3
"""
build_experiment_summary.py
----------------------------
Aggregate training + eval + profile artifacts for a RADIO-GS experiment run
into a human-readable Markdown report and a machine-readable JSON summary.

Usage:
    python radio_gs/scripts/build_experiment_summary.py --exp_dir output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep
    python radio_gs/scripts/build_experiment_summary.py --exp_dir output/radio_gs/lerf_ramen_v14_fdh_ws240_240ep_seed7 --out /tmp/ramen_seed7_summary.md
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path):
    """Load JSON file, return None if missing or malformed."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _read_last_line(path: Path):
    """Return last non-empty line of a text file, or None."""
    try:
        lines = path.read_text().strip().splitlines()
        for line in reversed(lines):
            if line.strip():
                return line.strip()
    except Exception:
        pass
    return None


def _parse_gpu_metrics(path: Path):
    """Parse gpu_metrics.csv → (mean_util_pct, peak_vram_gb) or (None, None)."""
    try:
        utils, vrams = [], []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    utils.append(float(row["gpu_util_%"]))
                except (KeyError, ValueError):
                    pass
                try:
                    vrams.append(float(row["vram_used_mb"]))
                except (KeyError, ValueError):
                    pass
        mean_util = sum(utils) / len(utils) if utils else None
        peak_vram_gb = max(vrams) / 1024.0 if vrams else None
        return mean_util, peak_vram_gb
    except Exception:
        return None, None


def _best_eval_row(eval_results: list):
    """Pick row with highest loc_acc (break ties by miou)."""
    if not eval_results:
        return None
    return max(eval_results, key=lambda r: (r.get("loc_acc", 0.0), r.get("miou", 0.0)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_summary(exp_dir: Path, out_md: Path = None, out_json: Path = None):
    exp_dir = exp_dir.resolve()

    # ---- Load artifacts ----
    train_report = _load_json(exp_dir / "reports" / "experiment_report.json") or {}
    run_manifest = _load_json(exp_dir / "reports" / "run_manifest.json") or {}
    eval_summary_raw = _load_json(exp_dir / "lerf_eval_best" / "summary.json")
    if isinstance(eval_summary_raw, dict):
        eval_summary = eval_summary_raw.get("results", [])
    elif isinstance(eval_summary_raw, list):
        eval_summary = eval_summary_raw
    else:
        eval_summary = None
    gpu_csv = exp_dir / "gpu_metrics.csv"
    time_log = exp_dir / "time.log"

    # ---- Extract config fields ----
    config = run_manifest.get("config", {})
    exp_name = train_report.get("exp_name") or run_manifest.get("exp_name") or exp_dir.name
    scene = config.get("scene", "unknown")
    seed = config.get("seed", "unknown")
    architecture = config.get("architecture", "unknown")
    epochs = config.get("epochs") or config.get("num_epochs", "unknown")
    warmstart_from = config.get("warmstart_from", None)
    feature_dir = config.get("feature_dir", None)
    start_time = run_manifest.get("start_time", "unknown")
    command = run_manifest.get("command", "unknown")
    cuda_devices = run_manifest.get("cuda_visible_devices", "unknown")

    # ---- Training metrics ----
    status = train_report.get("status", "unknown")
    best_epoch = train_report.get("best_epoch", None)
    best_val_cosine = None
    last_val = train_report.get("last_val_metrics", {})
    best_val_cosine = train_report.get("best_selection_score", last_val.get("cosine", None))
    last_train = train_report.get("last_train_metrics", {})
    final_train_loss = last_train.get("total", None)
    best_checkpoint = train_report.get("best_checkpoint", str(exp_dir / "checkpoints" / "best.pth"))

    # ---- Eval metrics ----
    best_eval = _best_eval_row(eval_summary) if eval_summary else None
    loc_acc = best_eval.get("loc_acc") if best_eval else None
    miou = best_eval.get("miou") if best_eval else None
    best_temp = best_eval.get("temp") if best_eval else None
    loc_total = best_eval.get("loc_total") if best_eval else None

    # ---- Efficiency ----
    mean_gpu_util, peak_vram_gb = _parse_gpu_metrics(gpu_csv) if gpu_csv.exists() else (None, None)
    wall_time_raw = _read_last_line(time_log) if time_log.exists() else None
    wall_time_sec = None
    if wall_time_raw is not None:
        try:
            wall_time_sec = float(wall_time_raw)
        except ValueError:
            wall_time_sec = None

    # ---- Build Markdown ----
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    lines.append(f"# Experiment Summary: `{exp_name}`")
    lines.append(f"_Generated: {now}_\n")

    lines.append("## 1. Header")
    lines.append(f"| Field | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| exp_name | `{exp_name}` |")
    lines.append(f"| scene | {scene} |")
    lines.append(f"| seed | {seed} |")
    lines.append(f"| status | {status} |")
    lines.append(f"| start_time | {start_time} |")
    lines.append(f"| cuda_devices | {cuda_devices} |")
    lines.append("")

    lines.append("## 2. Config")
    lines.append(f"| Field | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| architecture | {architecture} |")
    lines.append(f"| epochs | {epochs} |")
    lines.append(f"| warmstart_from | `{warmstart_from or 'none'}` |")
    lines.append(f"| feature_dir | `{feature_dir or 'none'}` |")
    lines.append(f"| command | `{command}` |")
    lines.append("")

    lines.append("## 3. Training")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| best_epoch | {best_epoch} |")
    bvc = f"{best_val_cosine:.4f}" if best_val_cosine is not None else "N/A"
    ftl = f"{final_train_loss:.4f}" if final_train_loss is not None else "N/A"
    lines.append(f"| best_val_cosine | {bvc} |")
    lines.append(f"| final_train_loss | {ftl} |")
    lines.append(f"| best_checkpoint | `{best_checkpoint}` |")
    lines.append("")

    lines.append("## 4. Eval (LERF-OVS)")
    if best_eval:
        lines.append(f"| Metric | Value |")
        lines.append(f"|---|---|")
        lines.append(f"| **LocAcc** | **{loc_acc:.4f}** |")
        lines.append(f"| mIoU | {miou:.4f} |")
        lines.append(f"| best_temp | {best_temp} |")
        lines.append(f"| loc_total | {loc_total} |")
        lines.append("")
        # Full sweep table
        lines.append("### Full Temperature Sweep")
        lines.append("| Temp | LocAcc | mIoU |")
        lines.append("|---|---|---|")
        for row in sorted(eval_summary, key=lambda r: r.get("temp", 0)):
            marker = " ← best" if row is best_eval else ""
            lines.append(f"| {row.get('temp')} | {row.get('loc_acc', 0):.4f} | {row.get('miou', 0):.4f} |{marker}")
    else:
        lines.append("_Eval results not found._")
    lines.append("")

    lines.append("## 5. Efficiency")
    if mean_gpu_util is not None or wall_time_sec is not None:
        lines.append(f"| Metric | Value |")
        lines.append(f"|---|---|")
        if mean_gpu_util is not None:
            lines.append(f"| mean_gpu_util | {mean_gpu_util:.1f}% |")
        if peak_vram_gb is not None:
            lines.append(f"| peak_vram_gb | {peak_vram_gb:.2f} GB |")
        if wall_time_sec is not None:
            h, rem = divmod(int(wall_time_sec), 3600)
            m, s = divmod(rem, 60)
            lines.append(f"| wall_time | {wall_time_sec:.0f}s ({h}h{m:02d}m{s:02d}s) |")
        elif wall_time_raw:
            lines.append(f"| wall_time | {wall_time_raw} |")
    else:
        lines.append("_No profiling data found._")
    lines.append("")

    lines.append("## 6. Artifacts")
    lines.append(f"| Artifact | Path |")
    lines.append(f"|---|---|")
    lines.append(f"| exp_dir | `{exp_dir}` |")
    lines.append(f"| best_checkpoint | `{best_checkpoint}` |")
    eval_dir = exp_dir / "lerf_eval_best"
    lines.append(f"| eval_dir | `{eval_dir}` |")
    lines.append(f"| reports_dir | `{exp_dir / 'reports'}` |")

    md_content = "\n".join(lines) + "\n"

    # ---- Build JSON ----
    summary_json = {
        "exp_name": exp_name,
        "scene": scene,
        "seed": seed,
        "architecture": architecture,
        "epochs": epochs,
        "warmstart_from": warmstart_from,
        "feature_dir": feature_dir,
        "status": status,
        "start_time": start_time,
        "best_epoch": best_epoch,
        "best_val_cosine": best_val_cosine,
        "final_train_loss": final_train_loss,
        "loc_acc": loc_acc,
        "miou": miou,
        "best_temp": best_temp,
        "loc_total": loc_total,
        "mean_gpu_util": mean_gpu_util,
        "peak_vram_gb": peak_vram_gb,
        "wall_time_sec": wall_time_sec,
        "best_checkpoint": best_checkpoint,
        "exp_dir": str(exp_dir),
    }

    # ---- Write outputs ----
    reports_dir = exp_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if out_md is None:
        out_md = reports_dir / "experiment_summary.md"
    if out_json is None:
        out_json = reports_dir / "experiment_summary.json"

    out_md.write_text(md_content)
    with open(out_json, "w") as f:
        json.dump(summary_json, f, indent=2)

    print(f"Wrote summary to {out_md}")
    print(f"Wrote JSON to   {out_json}")
    return summary_json


def main():
    parser = argparse.ArgumentParser(description="Build experiment summary from RADIO-GS artifacts.")
    parser.add_argument("--exp_dir", required=True, help="Path to experiment output directory")
    parser.add_argument("--out", default=None, help="Optional output path for Markdown report")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        print(f"ERROR: exp_dir does not exist: {exp_dir}")
        raise SystemExit(1)

    out_md = Path(args.out) if args.out else None
    build_summary(exp_dir, out_md=out_md)


if __name__ == "__main__":
    main()
