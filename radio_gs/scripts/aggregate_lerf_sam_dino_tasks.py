"""Aggregate per-scene LERF SAM3/DINOv3 task reports.

The evaluator is commonly launched once per LERF scene because each scene uses a
different RADIO-GS checkpoint.  This helper combines those scene-level JSON
files into the paper-facing aggregate JSON and Markdown report.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping


def _empty_seg() -> Dict[str, float]:
    return {"correct": 0.0, "total": 0.0, "iou_sum": 0.0}


def _empty_match() -> Dict[str, float]:
    return {"hits": 0.0, "total": 0.0, "score_sum": 0.0}


def _finalize_seg(acc: Mapping[str, float]) -> Dict[str, float]:
    total = int(acc["total"])
    if total == 0:
        return {"loc_acc": 0.0, "miou": 0.0, "n_samples": 0}
    return {
        "loc_acc": float(acc["correct"] / total),
        "miou": float(acc["iou_sum"] / total),
        "n_samples": total,
    }


def _finalize_match(acc: Mapping[str, float]) -> Dict[str, float]:
    total = int(acc["total"])
    if total == 0:
        return {"hit_rate": 0.0, "mean_score": 0.0, "n_matches": 0}
    return {
        "hit_rate": float(acc["hits"] / total),
        "mean_score": float(acc["score_sum"] / total),
        "n_matches": total,
    }


def _merge_seg(
    macro: MutableMapping[str, MutableMapping[str, MutableMapping[str, float]]],
    task: str,
    mode: str,
    metrics: Mapping[str, float],
) -> None:
    acc = macro[task][mode]
    n_samples = int(metrics["n_samples"])
    acc["correct"] += float(metrics["loc_acc"]) * n_samples
    acc["total"] += n_samples
    acc["iou_sum"] += float(metrics["miou"]) * n_samples


def _merge_match(
    macro: MutableMapping[str, MutableMapping[str, float]],
    mode: str,
    metrics: Mapping[str, float],
) -> None:
    acc = macro[mode]
    n_matches = int(metrics["n_matches"])
    acc["hits"] += float(metrics["hit_rate"]) * n_matches
    acc["total"] += n_matches
    acc["score_sum"] += float(metrics["mean_score"]) * n_matches


def load_scene_results(paths: Iterable[Path]) -> Dict[str, object]:
    scenes: Dict[str, object] = {}
    sam_macro = defaultdict(lambda: defaultdict(_empty_seg))
    dino_dense_macro = defaultdict(_empty_match)
    dino_mask_macro = defaultdict(lambda: defaultdict(_empty_seg))
    protocols = []

    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        protocols.append(report.get("protocol", {}))
        for scene, scene_report in report["scenes"].items():
            scenes[scene] = scene_report
            for task, modes in scene_report["sam3"].items():
                for mode, metrics in modes.items():
                    _merge_seg(sam_macro, task, mode, metrics)
            for mode, metrics in scene_report["dino_v3"]["dense_matching"].items():
                _merge_match(dino_dense_macro, mode, metrics)
            for mode, metrics in scene_report["dino_v3"]["mask_propagation"].items():
                _merge_seg(dino_mask_macro, "mask_propagation", mode, metrics)

    macro = {
        "sam3": {
            task: {mode: _finalize_seg(acc) for mode, acc in sorted(modes.items())}
            for task, modes in sorted(sam_macro.items())
        },
        "dino_v3": {
            "dense_matching": {
                mode: _finalize_match(acc) for mode, acc in sorted(dino_dense_macro.items())
            },
            "mask_propagation": {
                mode: _finalize_seg(acc)
                for mode, acc in sorted(dino_mask_macro["mask_propagation"].items())
            },
        },
    }
    return {
        "protocol": protocols[-1] if protocols else {},
        "scenes": dict(sorted(scenes.items())),
        "macro": macro,
    }


def _fmt(value: float) -> str:
    return f"{float(value):.4f}"


def write_markdown_report(report: Mapping[str, object], output_path: Path, *, title: str, note: str) -> None:
    macro = report["macro"]
    rows = [
        (
            "SAM3 point prompt segmentation",
            macro["sam3"]["point_prompt_segmentation"],
            "loc_acc",
            "miou",
            "n_samples",
        ),
        (
            "SAM3 box prompt segmentation",
            macro["sam3"]["box_prompt_segmentation"],
            "loc_acc",
            "miou",
            "n_samples",
        ),
        (
            "SAM3 mask prompt propagation",
            macro["sam3"]["mask_prompt_propagation"],
            "loc_acc",
            "miou",
            "n_samples",
        ),
        (
            "DINOv3 dense matching",
            macro["dino_v3"]["dense_matching"],
            "hit_rate",
            "mean_score",
            "n_matches",
        ),
        (
            "DINOv3 mask propagation + bg-suppressed readout",
            macro["dino_v3"]["mask_propagation"],
            "loc_acc",
            "miou",
            "n_samples",
        ),
    ]
    lines = [
        f"# {title}",
        "",
        note,
        "",
        "| Task | Teacher Loc/Hit | Frame-wise RADIO mIoU/Score | Rendered Loc/Hit | Rendered mIoU/Score | N |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task, metrics, loc_key, score_key, n_key in rows:
        teacher = metrics["teacher"]
        rendered = metrics["rendered"]
        lines.append(
            "| "
            f"{task} | {_fmt(teacher[loc_key])} | {_fmt(teacher[score_key])} | "
            f"{_fmt(rendered[loc_key])} | {_fmt(rendered[score_key])} | {int(rendered[n_key])} |"
        )
    lines.append("")
    dino = macro["dino_v3"]["mask_propagation"]
    rendered_miou = dino["rendered"]["miou"]
    teacher_miou = dino["teacher"]["miou"]
    rendered_loc = dino["rendered"]["loc_acc"]
    teacher_loc = dino["teacher"]["loc_acc"]
    lines.append(
        "Interpretation: background suppression improves rendered DINOv3 mask "
        f"propagation to {_fmt(rendered_loc)} LocAcc / {_fmt(rendered_miou)} mIoU. "
        f"The same readout gives the teacher {_fmt(teacher_loc)} LocAcc / "
        f"{_fmt(teacher_miou)} mIoU, so this row supports a narrowed DINO gap "
        "and a rendered LocAcc advantage, not same-protocol DINO mIoU superiority."
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path, help="Per-scene lerf_sam_dino_task_results.json files")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument(
        "--title",
        default="LERF SAM3/DINOv3 Formal Downstream Tasks",
    )
    parser.add_argument(
        "--note",
        default=(
            "Protocol: aggregate over per-scene RADIO-GS checkpoints. GT masks are "
            "used for prompts/support masks and final metrics only."
        ),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = load_scene_results(args.results)
    aggregate_path = args.output_dir / "lerf_sam_dino_task_aggregate.json"
    aggregate_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(
        report,
        args.output_dir / "lerf_sam_dino_task_report.md",
        title=args.title,
        note=args.note,
    )
    print(f"Wrote {aggregate_path}")
    print(f"Wrote {args.output_dir / 'lerf_sam_dino_task_report.md'}")


if __name__ == "__main__":
    main()
