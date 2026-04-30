#!/usr/bin/env python3
"""Sweep LERF-OVS grounding evaluator settings and summarize the best runs."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_LABEL_DIR = "/mnt/pool/sqy/3d_understanding/lerf_ovs/label"
DEFAULT_PROMPT_TEMPLATES = [
    "{query}",
    "a photo of a {query}",
    "a close-up photo of a {query}",
]


@dataclass(frozen=True)
class SweepCase:
    scoring: str
    temperature: float
    iou_threshold: float
    prompt_templates: tuple[str, ...]

    @property
    def label(self) -> str:
        temp = _number_label(self.temperature)
        iou = _number_label(self.iou_threshold)
        prompt = f"p{len(self.prompt_templates)}"
        return f"{self.scoring}_T{temp}_iou{iou}_{prompt}"


def _split_csv(raw: str | None, default: Iterable[str]) -> list[str]:
    if raw is None:
        return list(default)
    return [part.strip() for part in raw.split(",") if part.strip()]


def _split_floats(raw: str, name: str) -> list[float]:
    values: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    return values


def _number_label(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def _parse_prompt_sets(values: list[str] | None) -> list[tuple[str, ...]]:
    if not values:
        return [tuple(DEFAULT_PROMPT_TEMPLATES)]
    if len(values) > 1 and all("|" not in value for value in values):
        return [tuple(value.strip() for value in values if value.strip())]

    prompt_sets: list[tuple[str, ...]] = []
    for value in values:
        templates = tuple(part.strip() for part in value.split("|") if part.strip())
        if not templates:
            raise ValueError("--prompt_templates entries cannot be empty")
        prompt_sets.append(templates)
    return prompt_sets


def iter_sweep_cases(args: argparse.Namespace) -> Iterable[SweepCase]:
    scorings = _split_csv(args.scoring, ["softmax_scene"])
    temps = _split_floats(args.temps, "--temps")
    iou_thresholds = _split_floats(args.iou_thresholds, "--iou_thresholds")
    prompt_sets = _parse_prompt_sets(args.prompt_templates)

    for scoring, temp, iou_threshold, prompt_templates in itertools.product(
        scorings,
        temps,
        iou_thresholds,
        prompt_sets,
    ):
        yield SweepCase(
            scoring=scoring,
            temperature=temp,
            iou_threshold=iou_threshold,
            prompt_templates=prompt_templates,
        )


def build_eval_command(
    args: argparse.Namespace,
    case: SweepCase,
    output_dir: Path,
) -> tuple[list[str], dict[str, str]]:
    cmd = [
        "bash",
        args.python_wrapper,
        "-m",
        "radio_gs.scripts.eval_lerf_grounding",
        "--scene",
        args.scene,
        "--label_dir",
        args.label_dir,
        "--output_dir",
        str(output_dir),
        "--scoring",
        case.scoring,
        "--relevancy_temp",
        str(case.temperature),
        "--iou_threshold",
        str(case.iou_threshold),
        "--prompt_templates",
        "|".join(case.prompt_templates),
        "--heatmap_upsample",
        str(args.heatmap_upsample),
        "--gpu",
        str(args.gpu),
    ]

    if args.config:
        cmd.extend(["--config", args.config])
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])
    if args.gt_feature_dir:
        cmd.extend(["--gt_feature_dir", args.gt_feature_dir])
    if args.text_embedding_cache:
        cmd.extend(["--text_embedding_cache", args.text_embedding_cache])
    if args.projection_weights:
        cmd.extend(["--projection_weights", args.projection_weights])
    if args.summary_head_weights:
        cmd.extend(["--summary_head_weights", args.summary_head_weights])
    if args.use_spatial_projection:
        cmd.append("--no_summary_head")
    if args.gt_only:
        cmd.append("--gt_only")
    if args.save_vis:
        cmd.append("--save_vis")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = getattr(args, "cuda_visible_devices", "5")
    return cmd, env


def read_metrics(result_path: Path, scene: str) -> dict[str, object]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if scene == "all":
        scene_items = payload["scenes"].items()
    else:
        scene_items = [(scene, payload["scenes"][scene])]

    row: dict[str, object] = {}
    for mode in ("rendered", "gt"):
        loc_correct = 0
        loc_total = 0
        mious: list[float] = []
        for _, scene_result in scene_items:
            metrics = scene_result.get(mode)
            if metrics is None and mode == "gt":
                metrics = scene_result.get("teacher")
            if not metrics:
                continue
            scene_loc_total = int(metrics.get("loc_total", 0))
            scene_loc_correct = metrics.get("loc_correct")
            if scene_loc_correct is None and scene_loc_total:
                scene_loc_correct = float(metrics.get("loc_acc", 0.0)) * scene_loc_total
            loc_correct += int(round(float(scene_loc_correct or 0)))
            loc_total += scene_loc_total
            mious.append(float(metrics.get("miou", 0.0)))
        row[f"{mode}_loc_acc"] = loc_correct / max(loc_total, 1) if loc_total else None
        row[f"{mode}_miou"] = sum(mious) / len(mious) if mious else None
        row[f"{mode}_loc_total"] = loc_total
    return row


def sort_results(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    def value(row: dict[str, object], key: str) -> float:
        raw = row.get(key)
        return float(raw) if raw is not None else -1.0

    return sorted(
        rows,
        key=lambda row: (
            value(row, "rendered_loc_acc"),
            value(row, "gt_loc_acc"),
            value(row, "rendered_miou"),
            value(row, "gt_miou"),
        ),
        reverse=True,
    )


def _run_case(args: argparse.Namespace, case: SweepCase, output_root: Path) -> dict[str, object]:
    run_dir = output_root / case.label
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd, env = build_eval_command(args, case, run_dir)
    command_path = run_dir / "command.json"
    command_path.write_text(
        json.dumps(
            {
                "cmd": cmd,
                "CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"],
                "case": case.__dict__,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    started = time.time()
    print(f"[SWEEP] Running {case.label}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, env=env, check=True)
    result_path = run_dir / "lerf_ovs_results.json"
    metrics = read_metrics(result_path, args.scene)
    return {
        "label": case.label,
        "scoring": case.scoring,
        "temperature": case.temperature,
        "iou_threshold": case.iou_threshold,
        "prompt_templates": "|".join(case.prompt_templates),
        "output_dir": str(run_dir),
        "elapsed_sec": round(time.time() - started, 3),
        **metrics,
    }


def write_summary(output_root: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "scene": args.scene,
        "label_dir": args.label_dir,
        "cuda_visible_devices": args.cuda_visible_devices,
        "sort_order": ["rendered_loc_acc", "gt_loc_acc", "rendered_miou", "gt_miou"],
        "results": rows,
        "best": rows[0] if rows else None,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_path = output_root / "summary.csv"
    fieldnames = [
        "label",
        "scoring",
        "temperature",
        "iou_threshold",
        "prompt_templates",
        "rendered_loc_acc",
        "gt_loc_acc",
        "rendered_miou",
        "gt_miou",
        "rendered_loc_total",
        "gt_loc_total",
        "elapsed_sec",
        "output_dir",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run eval_lerf_grounding.py over scoring/temp/IoU/prompt grids."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--gt_feature_dir", default=None)
    parser.add_argument("--gt_only", action="store_true")
    parser.add_argument("--scene", default="all")
    parser.add_argument("--label_dir", default=DEFAULT_LABEL_DIR)
    parser.add_argument("--output_root", required=True)
    parser.add_argument(
        "--scoring",
        default="softmax_scene",
        help="Comma-separated scoring modes: softmax_scene,cosine,relevancy",
    )
    parser.add_argument("--temps", default="20,30,40,50")
    parser.add_argument("--iou_thresholds", default="0.4,0.5,0.6")
    parser.add_argument(
        "--prompt_templates",
        action="append",
        help="Prompt template set separated by '|'. Repeat for multiple prompt-set combos.",
    )
    parser.add_argument("--text_embedding_cache", default=None)
    parser.add_argument("--projection_weights", default=None)
    parser.add_argument("--summary_head_weights", default=None)
    parser.add_argument(
        "--python_wrapper",
        default="radio_gs/scripts/run_repo_python.sh",
        help="Python wrapper used for evaluator launches",
    )
    parser.add_argument("--use_spatial_projection", action="store_true")
    parser.add_argument("--heatmap_upsample", type=int, default=4)
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument(
        "--cuda_visible_devices",
        default="5",
        help="CUDA_VISIBLE_DEVICES value for launched evaluator commands (default: 5)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Evaluator-local GPU id after CUDA_VISIBLE_DEVICES masking (default: 0)",
    )
    parser.add_argument("--jobs", type=int, default=1, help="Parallel evaluator processes")
    args = parser.parse_args()

    if not args.gt_only and (not args.config or not args.checkpoint) and not args.gt_feature_dir:
        parser.error("Provide --config + --checkpoint, or --gt_feature_dir/--gt_only")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cases = list(iter_sweep_cases(args))
    rows: list[dict[str, object]] = []

    if args.jobs == 1:
        for case in cases:
            rows.append(_run_case(args, case, output_root))
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(_run_case, args, case, output_root) for case in cases]
            for future in as_completed(futures):
                rows.append(future.result())

    rows = sort_results(rows)
    write_summary(output_root, rows, args)
    print(f"[SWEEP] Wrote {output_root / 'summary.json'}")
    print(f"[SWEEP] Wrote {output_root / 'summary.csv'}")
    if rows:
        best = rows[0]
        print(
            "[SWEEP] Best: "
            f"{best['label']} rendered_loc_acc={best.get('rendered_loc_acc')} "
            f"gt_loc_acc={best.get('gt_loc_acc')} rendered_miou={best.get('rendered_miou')} "
            f"gt_miou={best.get('gt_miou')}"
        )


if __name__ == "__main__":
    main()
