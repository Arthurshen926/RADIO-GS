#!/usr/bin/env python3
"""Rebuild the room0 variant comparison report."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "output" / "radio_gs" / "reports"
COMPARE_PRECISION = 4

VARIANTS = {
    "nofdh_240ep": "room0_hybrid_v14_nofdh_240ep",
    "pure_frozen": "room0_hybrid_v14_pure_frozen",
    "pure_frozen_depth_only": "room0_hybrid_v14_pure_frozen_depth_only",
}

DEPTH_METRICS = [
    ("Depth AbsRel", "rendered_depth", "depth_abs_rel", False),
    ("Depth RMSE", "rendered_depth", "depth_rmse", False),
    ("Depth delta1", "rendered_depth", "depth_delta1", True),
]
SEG_METRICS = [
    ("mIoU", "rendered_seg", "seg_mIoU", True),
    ("acc", "rendered_seg", "seg_pixel_acc", True),
]
GROUND_METRICS = [
    ("mIoU@0.5", "rendered_grounding", "grnd_mIoU@0.5", True),
    ("mAP", "rendered_grounding", "grnd_mAP", True),
    ("corr", "rendered_grounding", "grnd_corr", True),
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float) -> str:
    return f"{value:.4f}"


def metric_value(payload: dict, section: str, key: str) -> float:
    return float(payload[section][key])


def compare_metric(best_value: float, latest_value: float, higher_is_better: bool) -> str:
    best_rounded = round(best_value, COMPARE_PRECISION)
    latest_rounded = round(latest_value, COMPARE_PRECISION)
    if best_rounded == latest_rounded:
        return "tie"
    if higher_is_better:
        return "best" if best_rounded > latest_rounded else "latest"
    return "best" if best_rounded < latest_rounded else "latest"


def read_best_training_cosine(path: Path) -> float:
    best = None
    pattern = re.compile(r"New best! cosine=([0-9.]+)")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match:
            best = float(match.group(1))
    if best is None:
        raise ValueError(f"failed to parse best training cosine from {path}")
    return best


def category_winner(best_payload: dict, latest_payload: dict, metrics: list[tuple[str, str, str, bool]]) -> str:
    winners = []
    for label, section, key, higher_is_better in metrics:
        best_value = metric_value(best_payload, section, key)
        latest_value = metric_value(latest_payload, section, key)
        winner = compare_metric(best_value, latest_value, higher_is_better)
        winners.append((label, winner))
    unique_winners = {winner for _, winner in winners}
    if len(unique_winners) == 1:
        return f"`{winners[0][1]}`"
    joined = ", ".join(f"`{winner}` {label}" for label, winner in winners)
    return f"mixed ({joined})"


def best_vs_latest_note(depth_winner: str, seg_winner: str, ground_winner: str) -> str:
    if depth_winner == "`tie`" and seg_winner == "`tie`" and ground_winner == "`tie`":
        return "Best and latest are tied at report precision across rendered tasks."
    if depth_winner == "`best`" and seg_winner == "`best`" and ground_winner == "`best`":
        return "Best and latest are very close; `best` is slightly better overall on rendered tasks."
    if (
        depth_winner == "`best`"
        and seg_winner == "mixed (`best` mIoU, `latest` acc)"
        and ground_winner == "`best`"
    ):
        return "Best and latest are nearly identical; the ranking does not change."
    if depth_winner == "`latest`" and seg_winner == "`latest`" and ground_winner == "`latest`":
        return "Best and latest are very close; `latest` is slightly better overall on rendered tasks."
    return "Best and latest remain close with only minor per-metric differences."


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def latex_note_text(text: str) -> str:
    return (
        text.replace("`best`", r"\texttt{best}")
        .replace("`latest`", r"\texttt{latest}")
        .replace("`tie`", r"\texttt{tie}")
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path}")


def resolve_eval_json(run_dir: Path, label: str) -> Path:
    preferred = run_dir / "auto_eval" / label / "eval_rendered_results.json"
    if preferred.exists():
        return preferred
    fallback = run_dir / "auto_eval_split" / label / "eval_rendered_results.json"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"missing eval JSON for {run_dir.name} label={label}: checked {preferred} and {fallback}"
    )


def build_markdown(records: list[dict]) -> str:
    lines = [
        "## Room0 Variant Comparison",
        "",
        "Variants compared from rendered evaluation outputs on `room_0`:",
        "",
    ]
    for record in records:
        lines.append(f"- `{record['variant']}`: `{record['run_name']}`")
    lines.extend(
        [
            "",
            "### Best Rendered Metrics Per Variant",
            "",
            "| Variant | Depth AbsRel ↓ | Depth RMSE ↓ | Depth delta1 ↑ | Seg mIoU ↑ | Seg Acc ↑ | Ground mIoU@0.5 ↑ | Ground mAP ↑ | Ground Corr ↑ |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for record in records:
        best = record["best"]
        lines.append(
            "| `{variant}` | {abs_rel} | {rmse} | {delta1} | {seg_miou} | {seg_acc} | {grnd_miou} | {grnd_map} | {grnd_corr} |".format(
                variant=record["variant"],
                abs_rel=fmt(metric_value(best, "rendered_depth", "depth_abs_rel")),
                rmse=fmt(metric_value(best, "rendered_depth", "depth_rmse")),
                delta1=fmt(metric_value(best, "rendered_depth", "depth_delta1")),
                seg_miou=fmt(metric_value(best, "rendered_seg", "seg_mIoU")),
                seg_acc=fmt(metric_value(best, "rendered_seg", "seg_pixel_acc")),
                grnd_miou=fmt(metric_value(best, "rendered_grounding", "grnd_mIoU@0.5")),
                grnd_map=fmt(metric_value(best, "rendered_grounding", "grnd_mAP")),
                grnd_corr=fmt(metric_value(best, "rendered_grounding", "grnd_corr")),
            )
        )
    lines.extend(
        [
            "",
            "### Best vs Latest",
            "",
            "| Variant | Checkpoint with better rendered depth | Checkpoint with better rendered segmentation | Checkpoint with better rendered grounding | Note |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for record in records:
        lines.append(
            f"| `{record['variant']}` | {record['depth_winner']} | {record['seg_winner']} | {record['ground_winner']} | {record['note']} |"
        )
    lines.extend(
        [
            "",
            "### Training Cosine Comparison",
            "",
            "Training best cosine values from `logs/training.log`:",
            "",
            "| Variant | Best training cosine ↑ |",
            "| --- | ---: |",
        ]
    )
    for record in records:
        lines.append(f"| `{record['variant']}` | {fmt(record['best_training_cosine'])} |")
    lines.extend(
        [
            "",
            "### Concise Takeaways",
            "",
            "- `pure_frozen` is the strongest semantic variant on room0, leading rendered segmentation and grounding by clear margins, but it gives up rendered depth accuracy.",
            "- `pure_frozen_depth_only` gives the best rendered depth metrics and the highest training cosine, but semantic transfer is noticeably weaker.",
            "- `nofdh_240ep` is the most balanced setting: depth remains competitive with `pure_frozen_depth_only` while segmentation and grounding stay substantially above the depth-only variant.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_latex(records: list[dict]) -> str:
    best_vs_latest_lines = [
        rf"For \texttt{{{latex_escape(record['variant'])}}}, {latex_note_text(record['note'])}"
        for record in records
    ]
    lines = [
        r"\section{Room0 Variant Comparison}",
        "",
        (
            r"We compare three room0 variants using the rendered evaluation outputs: "
            r"\texttt{room0\_hybrid\_v14\_nofdh\_240ep}, "
            r"\texttt{room0\_hybrid\_v14\_pure\_frozen}, and "
            r"\texttt{room0\_hybrid\_v14\_pure\_frozen\_depth\_only}."
        ),
        "",
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccccccc}",
        r"\hline",
        r"Variant & AbsRel $\downarrow$ & RMSE $\downarrow$ & $\delta_1$ $\uparrow$ & mIoU $\uparrow$ & Acc $\uparrow$ & Grnd mIoU@0.5 $\uparrow$ & Grnd mAP $\uparrow$ & Grnd Corr $\uparrow$ \\",
        r"\hline",
    ]
    for record in records:
        best = record["best"]
        lines.append(
            r"\texttt{{{variant}}} & {abs_rel} & {rmse} & {delta1} & {seg_miou} & {seg_acc} & {grnd_miou} & {grnd_map} & {grnd_corr} \\".format(
                variant=latex_escape(record["variant"]),
                abs_rel=fmt(metric_value(best, "rendered_depth", "depth_abs_rel")),
                rmse=fmt(metric_value(best, "rendered_depth", "depth_rmse")),
                delta1=fmt(metric_value(best, "rendered_depth", "depth_delta1")),
                seg_miou=fmt(metric_value(best, "rendered_seg", "seg_mIoU")),
                seg_acc=fmt(metric_value(best, "rendered_seg", "seg_pixel_acc")),
                grnd_miou=fmt(metric_value(best, "rendered_grounding", "grnd_mIoU@0.5")),
                grnd_map=fmt(metric_value(best, "rendered_grounding", "grnd_mAP")),
                grnd_corr=fmt(metric_value(best, "rendered_grounding", "grnd_corr")),
            )
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\caption{Best rendered metrics per room0 variant.}",
            r"\label{tab:room0_variant_comparison}",
            r"\end{table}",
            "",
            r"\paragraph{Best vs. latest.}",
            " ".join(best_vs_latest_lines),
            "",
            r"\paragraph{Training cosine comparison.}",
            (
                r"Best training cosine values from the training logs are: "
                rf"\texttt{{nofdh\_240ep}} = {fmt(records[0]['best_training_cosine'])}, "
                rf"\texttt{{pure\_frozen}} = {fmt(records[1]['best_training_cosine'])}, and "
                rf"\texttt{{pure\_frozen\_depth\_only}} = {fmt(records[2]['best_training_cosine'])}."
            ),
            "",
            r"\paragraph{Takeaways.}",
            (
                r"\texttt{pure\_frozen} is the strongest semantic variant on room0, with the best rendered segmentation and grounding, but it substantially underperforms on rendered depth. "
                r"\texttt{pure\_frozen\_depth\_only} gives the best rendered depth and the highest training cosine, but its semantic quality is clearly weaker. "
                r"\texttt{nofdh\_240ep} provides the most balanced trade-off, staying close to the best depth-only model on rendered depth while preserving much stronger semantic performance."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def collect_records() -> list[dict]:
    records = []
    output_root = REPO_ROOT / "output" / "radio_gs"
    for variant, run_name in VARIANTS.items():
        run_dir = output_root / run_name
        best_payload = load_json(resolve_eval_json(run_dir, "best"))
        latest_payload = load_json(resolve_eval_json(run_dir, "latest"))
        best_training_cosine = read_best_training_cosine(run_dir / "logs" / "training.log")
        depth_winner = category_winner(best_payload, latest_payload, DEPTH_METRICS)
        seg_winner = category_winner(best_payload, latest_payload, SEG_METRICS)
        ground_winner = category_winner(best_payload, latest_payload, GROUND_METRICS)
        records.append(
            {
                "variant": variant,
                "run_name": run_name,
                "best": best_payload,
                "latest": latest_payload,
                "best_training_cosine": best_training_cosine,
                "depth_winner": depth_winner,
                "seg_winner": seg_winner,
                "ground_winner": ground_winner,
                "note": best_vs_latest_note(depth_winner, seg_winner, ground_winner),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the room0 variant comparison report.")
    parser.add_argument(
        "--output_dir",
        default=str(OUTPUT_DIR),
        help="Directory where the markdown and LaTeX reports will be written.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    records = collect_records()
    write_text(output_dir / "room0_variant_comparison.md", build_markdown(records))
    write_text(output_dir / "room0_variant_comparison.tex", build_latex(records))


if __name__ == "__main__":
    main()
