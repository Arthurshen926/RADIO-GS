#!/usr/bin/env python3
"""Build paper-oriented result tables and benchmark sheets for RADIO-GS.

This script packages the current strongest internal results together with
published open-vocabulary 3D baselines into paper-ready markdown and LaTeX
tables. The goal is to keep the submission narrative reproducible instead of
manually editing numbers across multiple notes.

Outputs:
  - output/radio_gs/reports/paper_submission_main_table.md
  - output/radio_gs/reports/paper_submission_main_table.tex
  - output/radio_gs/reports/paper_benchmark_targets.md

Notes:
  - RADIO-GS scores reflect the current best per-scene settings collected in
    the internal reports under output/radio_gs/reports/.
  - Published baseline numbers are the values currently used by this project for
    LERF-OVS comparison and should be re-checked against the exact paper table
    before a final submission freeze.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


SCENES = ["figurines", "ramen", "teatime", "waldo_kitchen"]
SCENE_LABELS = {
    "figurines": "Figurines",
    "ramen": "Ramen",
    "teatime": "Teatime",
    "waldo_kitchen": "Waldo Kitchen",
}
SCENE_ALIASES = {
    "waldo": "waldo_kitchen",
}
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_PATH = REPO_ROOT / "output" / "radio_gs" / "reports" / "sota_comparison_table.md"
DEFAULT_EVAL_ROOT = REPO_ROOT / "output" / "lerf_ovs_eval"


@dataclass(frozen=True)
class MethodRecord:
    name: str
    venue: str
    paper_title: str
    source_url: str
    scores: dict[str, float]
    notes: str = ""

    @property
    def macro(self) -> float:
        return sum(self.scores[scene] for scene in SCENES) / len(SCENES)


@dataclass(frozen=True)
class ResultEntry:
    scene: str
    section: str
    score: float
    path: str
    config: str
    checkpoint: str
    temp: str
    heatmap: str


PUBLISHED_BASELINES = [
    MethodRecord(
        name="LERF",
        venue="ICCV 2023",
        paper_title="LERF: Language Embedded Radiance Fields",
        source_url=(
            "https://openaccess.thecvf.com/content/ICCV2023/html/"
            "Kerr_LERF_Language_Embedded_Radiance_Fields_ICCV_2023_paper.html"
        ),
        scores={
            "figurines": 0.520,
            "ramen": 0.503,
            "teatime": 0.653,
            "waldo_kitchen": 0.456,
        },
        notes="Foundational NeRF-based open-vocabulary 3D querying baseline.",
    ),
    MethodRecord(
        name="LangSplat",
        venue="CVPR 2024",
        paper_title="LangSplat: 3D Language Gaussian Splatting",
        source_url=(
            "https://openaccess.thecvf.com/content/CVPR2024/html/"
            "Qin_LangSplat_3D_Language_Gaussian_Splatting_CVPR_2024_paper.html"
        ),
        scores={
            "figurines": 0.592,
            "ramen": 0.659,
            "teatime": 0.693,
            "waldo_kitchen": 0.600,
        },
        notes="Fast 3DGS-based language field with strong boundary quality.",
    ),
    MethodRecord(
        name="LEGaussians",
        venue="CVPR 2024",
        paper_title=(
            "Language Embedded 3D Gaussians for Open-Vocabulary Scene "
            "Understanding"
        ),
        source_url=(
            "https://openaccess.thecvf.com/content/CVPR2024/html/"
            "Shi_Language_Embedded_3D_Gaussians_for_Open-Vocabulary_"
            "Scene_Understanding_CVPR_2024_paper.html"
        ),
        scores={
            "figurines": 0.631,
            "ramen": 0.695,
            "teatime": 0.745,
            "waldo_kitchen": 0.593,
        },
        notes="Current strongest directly aligned published baseline in our setting.",
    ),
]

AUXILIARY_PUBLISHED_METHODS = [
    {
        "name": "Gaussian Grouping",
        "venue": "ECCV 2024",
        "paper_title": "Gaussian Grouping: Segment and Edit Anything in 3D Scenes",
        "source_url": "https://ymq2017.github.io/gaussian-grouping/",
        "fit": "Supplementary comparison for open-world 3D segmentation / editing",
        "notes": (
            "Published and relevant, but not directly aligned with the LERF-OVS "
            "LocAcc table. Best used in related work or a supplementary section."
        ),
    },
    {
        "name": "3D Gaussian Splatting",
        "venue": "SIGGRAPH 2023",
        "paper_title": "3D Gaussian Splatting for Real-Time Radiance Field Rendering",
        "source_url": "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/",
        "fit": "Geometry / rendering efficiency upper-bound, not a grounding baseline",
        "notes": (
            "Useful for efficiency and geometry discussions, but not a direct open-"
            "vocabulary grounding baseline."
        ),
    },
]


READINESS_AREAS = [
    ("Problem framing", 0.80, "Main task definition is already coherent."),
    ("Method implementation", 0.85, "Training, evaluation, and visualization all exist."),
    ("Main grounding results", 0.80, "LERF-OVS evidence is already strong and now provenance-backed."),
    ("Published baseline coverage", 0.45, "Main grounding baselines exist, auxiliary baselines remain thin."),
    ("Statistical confidence", 0.35, "Multiple seeds and confidence intervals are still missing."),
    ("Cross-domain generalization", 0.35, "Replica + LERF-OVS is not enough for a top-conference claim."),
    ("Submission packaging", 0.60, "Main table and audit are frozen; statistics and efficiency are still missing."),
]


def fmt(value: float) -> str:
    return f"{value:.3f}"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path}")


def normalize_scene_name(name: str) -> str:
    return SCENE_ALIASES.get(name.strip(), name.strip())


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def strip_md(text: str) -> str:
    return text.replace("**", "").strip()


def load_reported_ours(report_path: Path) -> MethodRecord:
    scores: dict[str, float] = {}
    in_main_result = False
    for line in report_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## Main Result:"):
            in_main_result = True
            continue
        if in_main_result and line.startswith("## "):
            break
        if not in_main_result:
            continue
        if not line.startswith("|"):
            continue
        columns = [strip_md(part) for part in line.strip().strip("|").split("|")]
        if len(columns) < 2:
            continue
        scene = normalize_scene_name(columns[0])
        if scene not in SCENES:
            continue
        try:
            scores[scene] = float(columns[1])
        except ValueError:
            continue
    missing = [scene for scene in SCENES if scene not in scores]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            f"failed to parse reported RADIO-GS scores for scenes: {missing_text}"
        )
    return MethodRecord(
        name="RADIO-GS",
        venue="This repository",
        paper_title=(
            "Foundation feature reconstruction in 3D Gaussian scenes for "
            "open-vocabulary scene understanding"
        ),
        source_url=relpath(report_path),
        scores=scores,
        notes=(
            "Current best per-scene settings as currently reported by the internal "
            "LERF-OVS SOTA comparison sheet."
        ),
    )


def collect_result_entries(eval_root: Path) -> dict[str, list[ResultEntry]]:
    entries: dict[str, list[ResultEntry]] = {scene: [] for scene in SCENES}
    for path in sorted(eval_root.rglob("lerf_ovs_results.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        args = payload.get("args", {})
        for raw_scene, scene_payload in payload.get("scenes", {}).items():
            scene = normalize_scene_name(raw_scene)
            if scene not in SCENES:
                continue
            for section in ("rendered", "gt"):
                metrics = scene_payload.get(section)
                if not isinstance(metrics, dict) or metrics.get("loc_acc") is None:
                    continue
                entries[scene].append(
                    ResultEntry(
                        scene=scene,
                        section=section,
                        score=float(metrics["loc_acc"]),
                        path=relpath(path),
                        config=str(args.get("config", "")),
                        checkpoint=str(args.get("checkpoint", "")),
                        temp=str(args.get("relevancy_temp", "")),
                        heatmap=str(args.get("heatmap_upsample", "")),
                    )
                )
    return entries


def find_exact_entry(
    entries: list[ResultEntry], target: float, section: str
) -> ResultEntry | None:
    candidates = [
        entry
        for entry in entries
        if entry.section == section and abs(entry.score - target) < 5e-4
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda entry: (len(entry.path), entry.path))[0]


def find_best_entry(entries: list[ResultEntry], section: str) -> ResultEntry | None:
    candidates = [entry for entry in entries if entry.section == section]
    if not candidates:
        return None
    return max(candidates, key=lambda entry: entry.score)


def summarise_entry(entry: ResultEntry | None) -> str:
    if entry is None:
        return "-"
    temp = entry.temp or "?"
    heatmap = entry.heatmap or "?"
    return f"{fmt(entry.score)} @ {entry.path} (T={temp}, hm={heatmap})"


def build_result_audit_markdown(
    report_path: Path,
    ours: MethodRecord,
    entries_by_scene: dict[str, list[ResultEntry]],
) -> str:
    lines = []
    lines.append("# RADIO-GS Paper Submission Result Audit")
    lines.append("")
    lines.append(
        "This audit cross-checks the current reported RADIO-GS main-table numbers "
        "against the JSON files available under output/lerf_ovs_eval/."
    )
    lines.append("")
    lines.append(f"- Report source: {relpath(report_path)}")
    lines.append(
        "- Goal: separate numbers that are directly backed by rendered JSON files from "
        "numbers that still need provenance cleanup before paper freeze."
    )
    lines.append("")
    lines.append("| Scene | Reported | Best rendered JSON | Delta | Exact match in JSON |")
    lines.append("|---|---:|---:|---:|---|")
    for scene in SCENES:
        reported = ours.scores[scene]
        scene_entries = entries_by_scene.get(scene, [])
        exact_rendered = find_exact_entry(scene_entries, reported, "rendered")
        exact_gt = find_exact_entry(scene_entries, reported, "gt")
        best_rendered = find_best_entry(scene_entries, "rendered")
        if best_rendered is None:
            best_rendered_value = "-"
            delta_value = "-"
        else:
            best_rendered_value = fmt(best_rendered.score)
            delta_value = f"{reported - best_rendered.score:+.3f}"
        if exact_rendered is not None:
            exact_status = f"rendered: {exact_rendered.path}"
        elif exact_gt is not None:
            exact_status = f"gt only: {exact_gt.path}"
        else:
            exact_status = "none"
        lines.append(
            f"| {SCENE_LABELS[scene]} | {fmt(reported)} | {best_rendered_value} | {delta_value} | {exact_status} |"
        )
    lines.append("")
    lines.append("## Source details")
    lines.append("")
    for scene in SCENES:
        reported = ours.scores[scene]
        scene_entries = entries_by_scene.get(scene, [])
        exact_rendered = find_exact_entry(scene_entries, reported, "rendered")
        exact_gt = find_exact_entry(scene_entries, reported, "gt")
        best_rendered = find_best_entry(scene_entries, "rendered")
        lines.append(f"### {SCENE_LABELS[scene]}")
        lines.append("")
        lines.append(f"- Reported score: {fmt(reported)}")
        lines.append(f"- Best rendered JSON: {summarise_entry(best_rendered)}")
        if best_rendered is not None:
            lines.append(f"- Best rendered config: {best_rendered.config or '-'}")
            lines.append(f"- Best rendered checkpoint: {best_rendered.checkpoint or '-'}")
        lines.append(
            f"- Exact rendered match: {summarise_entry(exact_rendered)}"
        )
        lines.append(f"- Exact GT match: {summarise_entry(exact_gt)}")
        if exact_rendered is None:
            lines.append(
                "- Action: do not freeze this scene into the manuscript until the exact "
                "source run is identified or the report number is revised."
            )
        else:
            lines.append("- Action: this scene already has direct rendered JSON backing.")
        lines.append("")
    return "\n".join(lines) + "\n"


def build_main_markdown(records: list[MethodRecord]) -> str:
    lines = []
    lines.append("# RADIO-GS Paper Submission Main Table")
    lines.append("")
    lines.append(
        "This table packages the current LERF-OVS open-vocabulary grounding "
        "comparison that is most suitable for the submission main paper table."
    )
    lines.append("")
    header = ["Method", "Venue"] + [SCENE_LABELS[scene] for scene in SCENES] + ["Macro"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    best_by_scene = {
        scene: max(record.scores[scene] for record in records) for scene in SCENES
    }
    best_macro = max(record.macro for record in records)
    for record in records:
        row = [record.name, record.venue]
        for scene in SCENES:
            score = fmt(record.scores[scene])
            if record.scores[scene] == best_by_scene[scene]:
                score = f"**{score}**"
            row.append(score)
        macro = fmt(record.macro)
        if record.macro == best_macro:
            macro = f"**{macro}**"
        row.append(macro)
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- RADIO-GS row is parsed from the internal report at "
        "`output/radio_gs/reports/sota_comparison_table.md`."
    )
    lines.append(
        "- Published baseline values match the numbers currently used by this "
        "repository for LERF-OVS comparison. Before submission freeze, re-check the "
        "exact table identifiers in the original papers."
    )
    lines.append(
        "- Use `output/radio_gs/reports/paper_submission_result_audit.md` to verify "
        "whether each RADIO-GS scene score is directly backed by a rendered JSON file."
    )
    lines.append("")
    lines.append("## Sources")
    lines.append("")
    for record in records:
        lines.append(
            f"- **{record.name}**: {record.paper_title}. {record.venue}. "
            f"Source: {record.source_url}"
        )
    lines.append("")
    lines.append("## Current readiness snapshot")
    lines.append("")
    lines.append("| Area | Completion | Comment |")
    lines.append("|---|---:|---|")
    for area, completion, comment in READINESS_AREAS:
        lines.append(f"| {area} | {completion * 100:.0f}% | {comment} |")
    return "\n".join(lines) + "\n"


def build_main_latex(records: list[MethodRecord]) -> str:
    best_by_scene = {
        scene: max(record.scores[scene] for record in records) for scene in SCENES
    }
    best_macro = max(record.macro for record in records)
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{LERF-OVS open-vocabulary grounding comparison. Values are "
        r"LocAcc. Best per column is in \textbf{bold}.}"
    )
    lines.append(r"  \label{tab:lerf_ovs_main}")
    lines.append(r"  \begin{tabular}{lccccc}")
    lines.append(r"    \toprule")
    lines.append(
        r"    Method & Figurines & Ramen & Teatime & Waldo Kitchen & Macro \\" 
    )
    lines.append(r"    \midrule")
    for record in records:
        values = []
        for scene in SCENES:
            score = fmt(record.scores[scene])
            if record.scores[scene] == best_by_scene[scene]:
                score = rf"\textbf{{{score}}}"
            values.append(score)
        macro = fmt(record.macro)
        if record.macro == best_macro:
            macro = rf"\textbf{{{macro}}}"
        values.append(macro)
        lines.append(f"    {record.name} & " + " & ".join(values) + r" \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def build_benchmark_sheet(records: list[MethodRecord]) -> str:
    lines = []
    lines.append("# RADIO-GS Published Benchmark Targets")
    lines.append("")
    lines.append(
        "This sheet lists the published methods that should anchor the paper's "
        "comparison section. The first block is the primary main-table target set."
    )
    lines.append("")
    lines.append("## Primary main-table targets")
    lines.append("")
    lines.append("| Method | Paper | Venue | Why it belongs in the main table | Source |")
    lines.append("|---|---|---|---|---|")
    for record in records:
        if record.name == "RADIO-GS":
            continue
        lines.append(
            f"| {record.name} | {record.paper_title} | {record.venue} | {record.notes} | {record.source_url} |"
        )
    lines.append("")
    lines.append("## Supplementary published methods")
    lines.append("")
    lines.append("| Method | Paper | Venue | Best use in the paper | Source |")
    lines.append("|---|---|---|---|---|")
    for method in AUXILIARY_PUBLISHED_METHODS:
        lines.append(
            f"| {method['name']} | {method['paper_title']} | {method['venue']} | {method['fit']} | {method['source_url']} |"
        )
    lines.append("")
    lines.append("## Important exclusions")
    lines.append("")
    lines.append(
        "- `Feature3DGS-style` should stay an internal reproduced baseline or ablation label. "
        "It should not be presented as a published external SOTA method."
    )
    lines.append(
        "- Replica room_0 depth results are currently strong supporting evidence, but they do not "
        "yet replace the need for published open-vocabulary 3D grounding comparisons."
    )
    lines.append("")
    lines.append("## Next benchmark actions")
    lines.append("")
    lines.append("1. Freeze a single four-scene LERF-OVS main table with LERF, LangSplat, LEGaussians, and RADIO-GS.")
    lines.append("2. Re-check every external number against the exact original paper table before paper freeze.")
    lines.append("3. Add one cross-domain benchmark, ideally ScanNet, to support a stronger generalization claim.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper submission tables for RADIO-GS")
    parser.add_argument(
        "--output_dir",
        default="output/radio_gs/reports",
        help="Directory where the generated markdown and LaTeX files will be stored.",
    )
    parser.add_argument(
        "--report_path",
        default=str(DEFAULT_REPORT_PATH),
        help="Internal markdown report used as the current RADIO-GS source of truth.",
    )
    parser.add_argument(
        "--lerf_eval_dir",
        default=str(DEFAULT_EVAL_ROOT),
        help="Directory containing LERF-OVS evaluation JSON files for provenance audit.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    report_path = Path(args.report_path)
    eval_root = Path(args.lerf_eval_dir)
    ours = load_reported_ours(report_path)
    records = PUBLISHED_BASELINES + [ours]
    result_entries = collect_result_entries(eval_root)

    write_text(output_dir / "paper_submission_main_table.md", build_main_markdown(records))
    write_text(output_dir / "paper_submission_main_table.tex", build_main_latex(records))
    write_text(output_dir / "paper_benchmark_targets.md", build_benchmark_sheet(records))
    write_text(
        output_dir / "paper_submission_result_audit.md",
        build_result_audit_markdown(report_path, ours, result_entries),
    )


if __name__ == "__main__":
    main()
