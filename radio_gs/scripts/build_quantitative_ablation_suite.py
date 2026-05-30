#!/usr/bin/env python3
"""Build a unified quantitative ablation suite for the paper.

The project has many historical experiments. This script intentionally keeps a
small, audited set of paper-facing ablations and marks protocol-mixed rows as
diagnostic instead of merging them into the main method ranking.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class MetricDelta:
    name: str
    before: float
    after: float
    higher_is_better: bool = True

    @property
    def delta(self) -> float:
        raw = self.after - self.before
        return raw if self.higher_is_better else -raw

    @property
    def signed_delta(self) -> float:
        return self.after - self.before


@dataclass(frozen=True)
class ContributionRow:
    contribution: str
    task: str
    reference: str
    variant: str
    status: str
    metrics: tuple[MetricDelta, ...]
    source: str
    interpretation: str

    @property
    def positive_score(self) -> float:
        positive = [max(metric.delta, 0.0) for metric in self.metrics]
        return float(sum(positive) / max(len(positive), 1))

    @property
    def primary_delta(self) -> MetricDelta:
        return self.metrics[0]


@dataclass(frozen=True)
class Direct3DReadoutRow:
    row_id: str
    method: str
    threshold: str
    vpr_cache: str
    official_sam3: str
    rgb_postprocess: str
    prompt_ensemble: str
    miou: float
    acc025: float
    boundary_f: float | None
    trimap_iou: float | None
    note: str


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def require_sources(paths: list[Path]) -> list[str]:
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing ablation source artifact(s): " + ", ".join(str(path) for path in missing))
    return [rel(path) for path in paths]


def md(row: ContributionRow) -> dict[str, object]:
    payload = asdict(row)
    payload["score"] = round(row.positive_score, 6)
    payload["primary_metric"] = row.primary_delta.name
    payload["primary_delta"] = round(row.primary_delta.signed_delta, 6)
    return payload


def metric_pair(metric: MetricDelta) -> str:
    sign = "+" if metric.signed_delta >= 0 else ""
    return f"{metric.name}: {metric.before:.4f}->{metric.after:.4f} ({sign}{metric.signed_delta:.4f})"


def bool_label(value: object) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return str(value)


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def build_contributions() -> list[ContributionRow]:
    return [
        ContributionRow(
            contribution="VPR-to-field primitive feature registration",
            task="LERF Direct3D",
            reference="raw Gaussian-center score",
            variant="VPR-to-field compact primitive score",
            status="core direct-field evidence",
            metrics=(
                MetricDelta("mIoU", 0.046, 0.412),
                MetricDelta("Acc@0.25", 0.071, 0.588),
            ),
            source="paper/lerf_vpr_field_consistency_table.tex",
            interpretation="Largest direct-3D jump; primitive features need multiview registered support before selection is meaningful.",
        ),
        ContributionRow(
            contribution="CTR/HCD compact-to-teacher codec",
            task="LERF rendered architecture",
            reference="w/o CTR",
            variant="full CTF-GS seed-7 model",
            status="core architecture",
            metrics=(
                MetricDelta("LocAcc", 0.531, 0.858),
                MetricDelta("mIoU", 0.260, 0.485),
            ),
            source="paper/lerf_component_ablation_table.tex",
            interpretation="Dominant architectural dependency; direct 1x1 projection cannot recover teacher-compatible features.",
        ),
        ContributionRow(
            contribution="Final rendered-view readout vs frame-wise RADIO",
            task="2D teacher-vs-student feature usability",
            reference="frame-wise RADIO teacher",
            variant="CTF-GS rendered field + feature-only boundary readout",
            status="main claim evidence",
            metrics=(
                MetricDelta("LocAcc", 0.7985, 0.8598),
                MetricDelta("mIoU", 0.4634, 0.5889),
            ),
            source="paper/artifacts/final_rows.yaml",
            interpretation="Shows the student field improves text-grounding usability over frame-wise teacher features under the same evaluator.",
        ),
        ContributionRow(
            contribution="FGC/FDH geometry-aware warm-start",
            task="LERF rendered architecture",
            reference="w/o FGC warm-start",
            variant="full CTF-GS seed-7 model",
            status="core architecture",
            metrics=(
                MetricDelta("LocAcc", 0.802, 0.858),
                MetricDelta("mIoU", 0.424, 0.485),
            ),
            source="paper/lerf_component_ablation_table.tex",
            interpretation="Second largest controlled architecture contribution after CTR/HCD.",
        ),
        ContributionRow(
            contribution="Peak-component rendered mask readout",
            task="LERF rendered-view grounding",
            reference="threshold 0.60 mask",
            variant="threshold 0.60 + peak component",
            status="readout policy",
            metrics=(
                MetricDelta("mIoU", 0.5243, 0.5707),
                MetricDelta("LocAcc", 0.8712, 0.8598),
            ),
            source="paper/artifacts/final_rows.yaml",
            interpretation="Large boundary/support gain; it trades a small heatmap-peak drop for much better connected mask support.",
        ),
        ContributionRow(
            contribution="Direct3D prompt ensemble + component support policy",
            task="LERF Direct3D",
            reference="strict single-prompt pure one-map",
            variant="score-component guarded compact readout",
            status="main compact direct readout",
            metrics=(
                MetricDelta("mIoU", 0.44889714776724577, 0.501373864710331),
                MetricDelta("Acc@0.25", 0.6723696398925512, 0.7044309383271872),
                MetricDelta("Boundary-F", 0.6123990630730987, 0.63053347915411),
            ),
            source="paper/artifacts/lerf_direct3d_compact_readout_ablation_20260528.md",
            interpretation="Primary Waldo/small-object recovery mechanism for the compact direct row.",
        ),
        ContributionRow(
            contribution="SAM3 point-prompt feature readout",
            task="2D frozen-head downstream",
            reference="frame-wise RADIO teacher",
            variant="CTF-GS rendered field",
            status="downstream readout evidence",
            metrics=(
                MetricDelta("mIoU", 0.3700, 0.4173),
                MetricDelta("LocAcc", 1.0000, 1.0000),
            ),
            source="paper/artifacts/teacher_vs_ctfgs_2d_usability_20260525.md",
            interpretation="Strongest SAM3-adaptor primary-metric improvement with no point-prompt localization loss.",
        ),
        ContributionRow(
            contribution="DINOv3 dense matching feature readout",
            task="2D frozen-head downstream",
            reference="frame-wise RADIO teacher",
            variant="CTF-GS rendered field",
            status="downstream readout evidence with caveat",
            metrics=(
                MetricDelta("Mean score", 0.8547, 0.9048),
                MetricDelta("HitRate", 0.5723, 0.5396),
            ),
            source="paper/artifacts/teacher_vs_ctfgs_2d_usability_20260525.md",
            interpretation="Feature similarity improves, but hit-rate caveat shows DINO topology remains a separate concern.",
        ),
        ContributionRow(
            contribution="Feature-only SAM3 boundary readout",
            task="LERF rendered-view grounding",
            reference="peak-component core",
            variant="feature-only SAM3 boundary readout",
            status="readout policy",
            metrics=(
                MetricDelta("mIoU", 0.5707, 0.5889),
                MetricDelta("LocAcc", 0.8598, 0.8598),
            ),
            source="paper/artifacts/final_rows.yaml",
            interpretation="Boundary refinement improves region overlap without changing heatmap localization.",
        ),
        ContributionRow(
            contribution="ScanNet contextual kNN + spatial logit propagation",
            task="ScanNet VALA8 direct point-query",
            reference="DINO-CV contextual kNN alpha=0.5",
            variant="k16/cand80 + scene alpha 0.45 + spatial k12/a1",
            status="promoted ScanNet readout",
            metrics=(
                MetricDelta("split19 mIoU", 0.3704, 0.3806),
                MetricDelta("split19 mAcc", 0.6017, 0.6129),
                MetricDelta("split15 mIoU", 0.3771, 0.3871),
                MetricDelta("split15 mAcc", 0.6198, 0.6315),
                MetricDelta("split10 mIoU", 0.4585, 0.4711),
                MetricDelta("split10 mAcc", 0.7032, 0.7200),
            ),
            source="docs/experiments/2026-05-24-direct-field-joint2d3d-optimization.md",
            interpretation="Small but consistent VALA8 gains across all reported ScanNet splits.",
        ),
        ContributionRow(
            contribution="VFA view-space aligner",
            task="LERF rendered architecture",
            reference="w/o VFA",
            variant="full CTF-GS seed-7 model",
            status="core architecture",
            metrics=(
                MetricDelta("LocAcc", 0.840, 0.858),
                MetricDelta("mIoU", 0.480, 0.485),
            ),
            source="paper/lerf_component_ablation_table.tex",
            interpretation="Moderate localization gain; smaller region-overlap effect than CTR or FGC.",
        ),
        ContributionRow(
            contribution="Hybrid Gaussian code field",
            task="LERF rendered architecture",
            reference="w/o HGCF",
            variant="full CTF-GS seed-7 model",
            status="core architecture with tradeoff",
            metrics=(
                MetricDelta("LocAcc", 0.839, 0.858),
                MetricDelta("mIoU", 0.507, 0.485),
            ),
            source="paper/lerf_component_ablation_table.tex",
            interpretation="Improves peak stability but not raw mIoU in this controlled table; keep as a tradeoff rather than a universal gain.",
        ),
        ContributionRow(
            contribution="Official SAM3 box boundary diagnostic",
            task="LERF Direct3D",
            reference="score-component guarded compact readout",
            variant="frozen official SAM3 box readout",
            status="diagnostic, not core method",
            metrics=(
                MetricDelta("mIoU", 0.501373864710331, 0.5705),
                MetricDelta("Acc@0.25", 0.7044309383271872, 0.6835),
            ),
            source="paper/lerf_direct_3d_selection_table.tex",
            interpretation="Shows remaining boundary headroom, but it uses an external RGB SAM3 decoder and should not be counted as compact-field evidence.",
        ),
    ]


def build_missing_followups() -> list[dict[str, str]]:
    return [
        {
            "priority": "P0",
            "item": "optional full 2x2 Direct3D factorial for no-prompt + RGB/score guard",
            "reason": "the strict no-prompt/no-RGB cell is now filled; the remaining optional cell would isolate whether RGB/score support still helps without the prompt ensemble.",
        },
        {
            "priority": "P1",
            "item": "ScanNet module-removal training ablation on all VALA8 scenes",
            "reason": "readout ablations are complete enough for the paper, but full training-time removals for DINO-CV/context losses are not all positive or promoted.",
        },
        {
            "priority": "P1",
            "item": "multi-head DINO/SAM/SigLIP2 removal under the 2D frozen-head benchmark",
            "reason": "teacher-vs-student downstream wins are recorded, but per-head removal deltas are not yet a clean single-table factorial.",
        },
        {
            "priority": "P2",
            "item": "LERF final-row architecture ablation after feature-only SAM3 boundary readout",
            "reason": "core architecture table uses controlled seed-7 rendered features; final boundary readout is measured separately.",
        },
    ]


def write_json(path: Path, rows: list[ContributionRow], sources: list[str]) -> None:
    payload = {
        "generated": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "purpose": "Unified quantitative ablation and contribution ranking for CTF-GS paper writing.",
        "ranking_note": "score is the mean positive delta over listed metrics; negative secondary deltas are retained in each metric record.",
        "sources": sources,
        "contributions": [md(row) for row in sorted(rows, key=lambda item: item.positive_score, reverse=True)],
        "missing_followups": build_missing_followups(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, rows: list[ContributionRow]) -> None:
    ordered = sorted(rows, key=lambda item: item.positive_score, reverse=True)
    lines = [
        "# Unified Quantitative Ablation Suite",
        "",
        "This report consolidates the paper-facing ablations into one contribution ranking. Rows are grouped by protocol status so diagnostic readouts are not mistaken for core compact-field evidence.",
        "",
        "Score = mean positive delta over the listed metrics. Negative secondary deltas remain visible in the metric details.",
        "",
        "## Contribution Ranking",
        "",
        "| Rank | Contribution | Task | Reference -> Variant | Score | Metric deltas | Status |",
        "| ---: | --- | --- | --- | ---: | --- | --- |",
    ]
    for rank, row in enumerate(ordered, start=1):
        metric_text = "<br>".join(metric_pair(metric) for metric in row.metrics)
        lines.append(
            f"| {rank} | {row.contribution} | {row.task} | {row.reference} -> {row.variant} | "
            f"{row.positive_score:.4f} | {metric_text} | {row.status} |"
        )
    lines.extend(["", "## Interpretation", ""])
    for row in ordered:
        lines.append(f"- **{row.contribution}:** {row.interpretation} Source: `{row.source}`.")
    lines.extend(["", "## Missing Same-Protocol Follow-Ups", ""])
    lines.extend(
        f"- **{item['priority']}** {item['item']}: {item['reason']}"
        for item in build_missing_followups()
    )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex(path: Path, rows: list[ContributionRow], *, limit: int = 8) -> None:
    paper_rows = [row for row in rows if "diagnostic" not in row.status]
    ordered = sorted(paper_rows, key=lambda item: item.positive_score, reverse=True)[:limit]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Unified quantitative contribution ranking. Each row compares a reference and an enabled variant under the same local protocol unless marked diagnostic. Score is the mean positive delta over the listed metrics; secondary regressions are kept in the metric-delta column.}",
        r"\label{tab:quantitative_ablation_summary}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{p{0.20\linewidth}p{0.16\linewidth}p{0.20\linewidth}p{0.20\linewidth}cc}",
        r"\toprule",
        r"Contribution & Task & Reference $\rightarrow$ Variant & Metric deltas & Score & Status \\",
        r"\midrule",
    ]
    for row in ordered:
        metric_text = "; ".join(metric_pair(metric) for metric in row.metrics)
        ref_variant = f"{row.reference} -> {row.variant}"
        lines.append(
            f"{latex_escape(row.contribution)} & {latex_escape(row.task)} & "
            f"{latex_escape(ref_variant)} & {latex_escape(metric_text)} & "
            f"{row.positive_score:.3f} & {latex_escape(row.status)} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_direct3d_readout_rows(path: Path) -> list[Direct3DReadoutRow]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    rows = artifact["rows"]

    single = rows["compact_single_prompt_pure_onemap_thr0p70"]
    previous = rows["compact_direct_previous_promoted_baseline"]
    pure = rows["compact_prompt_ensemble_pure_onemap_thr0p70"]
    area = rows["compact_prompt_ensemble_rgb_component_guard_thr0p65"]
    score = rows["compact_prompt_ensemble_rgb_score_component_guard_thr0p55"]

    return [
        Direct3DReadoutRow(
            row_id="single_prompt_pure_one_map",
            method="Compact single prompt, pure one-map",
            threshold=str(single["global_threshold"]),
            vpr_cache=bool_label(single["uses_vpr_cache"]),
            official_sam3=bool_label(single["uses_official_rgb_sam_decoder"]),
            rgb_postprocess=bool_label(single["uses_rgb_postprocess"]),
            prompt_ensemble=bool_label(single["uses_prompt_ensemble"]),
            miou=float(single["miou"]),
            acc025=float(single["acc025"]),
            boundary_f=float(single["boundary_f"]),
            trimap_iou=float(single["trimap_iou"]),
            note="strict no-prompt/no-RGB compact-score readout",
        ),
        Direct3DReadoutRow(
            row_id="previous",
            method="Compact direct, previous promoted baseline",
            threshold="fixed",
            vpr_cache=bool_label(previous["uses_vpr_cache"]),
            official_sam3=bool_label(previous["uses_official_rgb_sam_decoder"]),
            rgb_postprocess=bool_label(previous["uses_rgb_postprocess"]),
            prompt_ensemble=bool_label(previous["uses_prompt_ensemble"]),
            miou=float(previous["miou"]),
            acc025=float(previous["acc025"]),
            boundary_f=float(previous["boundary_f"]),
            trimap_iou=None,
            note="older guarded baseline; not strict no-RGB",
        ),
        Direct3DReadoutRow(
            row_id="pure_one_map",
            method="Compact prompt ensemble, pure one-map",
            threshold=str(pure["global_threshold"]),
            vpr_cache=bool_label(pure["uses_vpr_cache"]),
            official_sam3=bool_label(pure["uses_official_rgb_sam_decoder"]),
            rgb_postprocess=bool_label(pure["uses_rgb_postprocess"]),
            prompt_ensemble=bool_label(pure["uses_prompt_ensemble"]),
            miou=float(pure["miou"]),
            acc025=float(pure["acc025"]),
            boundary_f=float(pure["boundary_f"]),
            trimap_iou=float(pure["trimap_iou"]),
            note="strict no-RGB compact-score readout",
        ),
        Direct3DReadoutRow(
            row_id="area_guard",
            method="Compact prompt ensemble + RGB component guard",
            threshold=str(area["global_threshold"]),
            vpr_cache=bool_label(area["uses_vpr_cache"]),
            official_sam3=bool_label(area["uses_official_rgb_sam_decoder"]),
            rgb_postprocess=bool_label(area["uses_rgb_postprocess"]),
            prompt_ensemble=bool_label(area["uses_prompt_ensemble"]),
            miou=float(area["miou"]),
            acc025=float(area["acc025"]),
            boundary_f=float(area["boundary_f"]),
            trimap_iou=float(area["trimap_iou"]),
            note="Acc-best compact guarded row",
        ),
        Direct3DReadoutRow(
            row_id="score_guard",
            method="Compact prompt ensemble + RGB/score-component guard",
            threshold=str(score["global_threshold"]),
            vpr_cache=bool_label(score["uses_vpr_cache"]),
            official_sam3=bool_label(score["uses_official_rgb_sam_decoder"]),
            rgb_postprocess=bool_label(score["uses_rgb_postprocess"]),
            prompt_ensemble=bool_label(score["uses_prompt_ensemble"]),
            miou=float(score["miou"]),
            acc025=float(score["acc025"]),
            boundary_f=float(score["boundary_f"]),
            trimap_iou=float(score["trimap_iou"]),
            note="overlap-balanced promoted compact row",
        ),
    ]


def write_direct3d_readout_latex(path: Path, rows: list[Direct3DReadoutRow]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Compact Direct3D readout ablation. All rows use compact Gaussian-center primitive scores and the OpenGaussian-style query-select-render evaluation. The pure one-map row is the strict no-VPR/no-RGB/no-SAM readout; guarded rows add GT-free component support policies but still avoid a VPR cache and official RGB SAM decoder.}",
        r"\label{tab:direct3d_compact_readout_ablation}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Readout & VPR & SAM3 & RGB & Prompt & mIoU & Acc@0.25 \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{latex_escape(row.method)} & {row.vpr_cache} & {row.official_sam3} & "
            f"{row.rgb_postprocess} & {row.prompt_ensemble} & {row.miou:.3f} & {row.acc025:.3f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_direct3d_readout_summary(path: Path, rows: list[Direct3DReadoutRow]) -> None:
    lines = [
        "# Direct3D Compact Readout Factorial Summary",
        "",
        "This summary is generated from `paper/artifacts/lerf_direct3d_compact_readout_ablation_20260528.json`.",
        "It separates compact-map evidence from RGB/component support policies and official SAM3 diagnostics.",
        "",
        "| Row | VPR cache | Official SAM3 | RGB postprocess | Prompt ensemble | Threshold | mIoU | Acc@0.25 | Boundary-F | Trimap IoU | Note |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        boundary = "-" if row.boundary_f is None else f"{row.boundary_f:.6f}"
        trimap = "-" if row.trimap_iou is None else f"{row.trimap_iou:.6f}"
        lines.append(
            f"| {row.method} | {row.vpr_cache} | {row.official_sam3} | {row.rgb_postprocess} | "
            f"{row.prompt_ensemble} | {row.threshold} | {row.miou:.6f} | {row.acc025:.6f} | "
            f"{boundary} | {trimap} | {row.note} |"
        )
    single = next(row for row in rows if row.row_id == "single_prompt_pure_one_map")
    pure = next(row for row in rows if row.row_id == "pure_one_map")
    score = next(row for row in rows if row.row_id == "score_guard")
    lines.extend(
        [
            "",
            "## Main Delta",
            "",
            (
                f"- Prompt ensemble over strict single-prompt pure one-map: "
                f"mIoU {single.miou:.6f}->{pure.miou:.6f} ({pure.miou - single.miou:+.6f}), "
                f"Acc@0.25 {single.acc025:.6f}->{pure.acc025:.6f} ({pure.acc025 - single.acc025:+.6f}), "
                f"Boundary-F {single.boundary_f:.6f}->{pure.boundary_f:.6f} ({pure.boundary_f - single.boundary_f:+.6f})."
            ),
            (
                f"- Guarded compact support over prompt-ensemble pure one-map: "
                f"mIoU {pure.miou:.6f}->{score.miou:.6f} ({score.miou - pure.miou:+.6f}), "
                f"Acc@0.25 {pure.acc025:.6f}->{score.acc025:.6f} ({score.acc025 - pure.acc025:+.6f}), "
                f"Boundary-F {pure.boundary_f:.6f}->{score.boundary_f:.6f} ({score.boundary_f - pure.boundary_f:+.6f})."
            ),
            "- Remaining optional cell: no-prompt plus RGB/score-component guard, which would isolate support-policy effects without prompt ensembling.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default="paper/artifacts/quantitative_ablation_suite.json")
    parser.add_argument("--markdown", default="paper/artifacts/quantitative_ablation_suite.md")
    parser.add_argument("--tex", default="paper/quantitative_ablation_summary_table.tex")
    parser.add_argument("--direct3d-tex", default="paper/lerf_direct3d_compact_readout_ablation_table.tex")
    parser.add_argument("--direct3d-summary", default="paper/artifacts/direct3d_compact_readout_factorial_summary.md")
    args = parser.parse_args()

    source_paths = [
        REPO_ROOT / "paper" / "artifacts" / "final_rows.yaml",
        REPO_ROOT / "paper" / "lerf_component_ablation_table.tex",
        REPO_ROOT / "paper" / "lerf_vpr_field_consistency_table.tex",
        REPO_ROOT / "paper" / "lerf_direct_3d_selection_table.tex",
        REPO_ROOT / "paper" / "artifacts" / "lerf_direct3d_compact_readout_ablation_20260528.md",
        REPO_ROOT / "paper" / "artifacts" / "lerf_direct3d_compact_readout_ablation_20260528.json",
        REPO_ROOT / "paper" / "artifacts" / "teacher_vs_ctfgs_2d_usability_20260525.md",
        REPO_ROOT / "docs" / "experiments" / "2026-05-24-direct-field-joint2d3d-optimization.md",
    ]
    sources = require_sources(source_paths)
    rows = build_contributions()
    direct3d_rows = build_direct3d_readout_rows(
        REPO_ROOT / "paper" / "artifacts" / "lerf_direct3d_compact_readout_ablation_20260528.json"
    )
    write_json(REPO_ROOT / args.json, rows, sources)
    write_markdown(REPO_ROOT / args.markdown, rows)
    write_latex(REPO_ROOT / args.tex, rows)
    write_direct3d_readout_latex(REPO_ROOT / args.direct3d_tex, direct3d_rows)
    write_direct3d_readout_summary(REPO_ROOT / args.direct3d_summary, direct3d_rows)
    print(f"Wrote {args.json}")
    print(f"Wrote {args.markdown}")
    print(f"Wrote {args.tex}")
    print(f"Wrote {args.direct3d_tex}")
    print(f"Wrote {args.direct3d_summary}")


if __name__ == "__main__":
    main()
