#!/usr/bin/env python3
"""Build 2D teacher-vs-CTF-GS feature-usability evidence tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTROLLED = REPO_ROOT / "paper/artifacts/controlled_evidence_table.json"
DEFAULT_SAM_DINO = (
    REPO_ROOT
    / "output/lerf_sam_dino_tasks/formal_v12c_dino_sam3_boundary_v9readout_gpu_20260528/lerf_sam_dino_task_aggregate.json"
)
DEFAULT_RANSAC = (
    REPO_ROOT
    / "output/lerf_sam_dino_tasks/formal_v8_mutual_homography_ransac_all_20260514/lerf_sam_dino_task_aggregate.json"
)
DEFAULT_PROTOTYPE = (
    REPO_ROOT / "output/lerf_adaptor_downstream/mainline/lerf_adaptor_downstream_aggregate.json"
)
DEFAULT_JSON = REPO_ROOT / "paper/artifacts/teacher_vs_ctfgs_2d_usability_20260525.json"
DEFAULT_MARKDOWN = REPO_ROOT / "paper/artifacts/teacher_vs_ctfgs_2d_usability_20260525.md"
DEFAULT_LATEX = REPO_ROOT / "paper/tables/teacher_vs_ctfgs_2d_usability_20260525.tex"


def _load_json(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value: object) -> float:
    return float(value)


def _fmt(value: object) -> str:
    return f"{float(value):.4f}"


def _fmt_delta(value: object) -> str:
    return f"{float(value):+.4f}"


def _find_method(rows: list[Mapping[str, object]], method: str) -> Mapping[str, object]:
    for row in rows:
        if row.get("method") == method:
            return row
    raise KeyError(f"Missing method row: {method}")


def _metric_pair(
    metrics: Mapping[str, Mapping[str, object]],
    *,
    task: str,
    primary_metric: str,
    primary_key: str,
    secondary_metric: str,
    secondary_key: str,
    count_key: str,
) -> dict[str, object]:
    teacher = metrics["teacher"]
    rendered = metrics["rendered"]
    teacher_primary = _float(teacher[primary_key])
    rendered_primary = _float(rendered[primary_key])
    teacher_secondary = _float(teacher[secondary_key])
    rendered_secondary = _float(rendered[secondary_key])
    delta_primary = rendered_primary - teacher_primary
    if delta_primary > 0:
        winner = "rendered"
    elif delta_primary < 0:
        winner = "teacher"
    else:
        winner = "tie"
    return {
        "task": task,
        "primary_metric": primary_metric,
        "teacher_primary": teacher_primary,
        "rendered_primary": rendered_primary,
        "delta_primary": delta_primary,
        "secondary_metric": secondary_metric,
        "teacher_secondary": teacher_secondary,
        "rendered_secondary": rendered_secondary,
        "delta_secondary": rendered_secondary - teacher_secondary,
        "n": int(rendered[count_key]),
        "winner": winner,
    }


def _build_text_grounding_rows(controlled: Mapping[str, object]) -> list[dict[str, object]]:
    rows = [dict(row) for row in controlled["rows"]]  # type: ignore[index]
    teacher = _find_method(rows, "Frame-wise RADIO teacher")
    teacher_loc = _float(teacher["lerf_loc_acc"])
    teacher_miou = _float(teacher["lerf_miou"])
    methods = [
        "Frame-wise RADIO teacher",
        "Nearest-view RADIO cache",
        "Per-Gaussian 1280-D RADIO memory",
        "Full CTF-GS",
    ]
    output = []
    for method in methods:
        row = _find_method(rows, method)
        loc = _float(row["lerf_loc_acc"])
        miou = _float(row["lerf_miou"])
        output.append(
            {
                "method": method,
                "loc_acc": loc,
                "miou": miou,
                "delta_loc_acc": loc - teacher_loc,
                "delta_miou": miou - teacher_miou,
                "source": row.get("source", ""),
                "note": row.get("note", ""),
            }
        )
    return output


def _build_frozen_head_rows(sam_dino: Mapping[str, object]) -> list[dict[str, object]]:
    macro = sam_dino["macro"]  # type: ignore[index]
    sam3 = macro["sam3"]  # type: ignore[index]
    dino = macro["dino_v3"]  # type: ignore[index]
    return [
        _metric_pair(
            sam3["point_prompt_segmentation"],  # type: ignore[index]
            task="SAM3 point prompt",
            primary_metric="mIoU",
            primary_key="miou",
            secondary_metric="LocAcc",
            secondary_key="loc_acc",
            count_key="n_samples",
        ),
        _metric_pair(
            sam3["box_prompt_segmentation"],  # type: ignore[index]
            task="SAM3 box prompt",
            primary_metric="mIoU",
            primary_key="miou",
            secondary_metric="LocAcc",
            secondary_key="loc_acc",
            count_key="n_samples",
        ),
        _metric_pair(
            sam3["mask_prompt_propagation"],  # type: ignore[index]
            task="SAM3 mask propagation",
            primary_metric="mIoU",
            primary_key="miou",
            secondary_metric="LocAcc",
            secondary_key="loc_acc",
            count_key="n_samples",
        ),
        _metric_pair(
            dino["dense_matching"],  # type: ignore[index]
            task="DINOv3 dense matching",
            primary_metric="Mean score",
            primary_key="mean_score",
            secondary_metric="HitRate",
            secondary_key="hit_rate",
            count_key="n_matches",
        ),
        _metric_pair(
            dino["mask_propagation"],  # type: ignore[index]
            task="DINOv3 mask propagation",
            primary_metric="mIoU",
            primary_key="miou",
            secondary_metric="LocAcc",
            secondary_key="loc_acc",
            count_key="n_samples",
        ),
    ]


def _build_prototype_rows(prototype: Mapping[str, object]) -> list[dict[str, object]]:
    macro = prototype["macro"]  # type: ignore[index]
    sam3 = macro["sam3"]  # type: ignore[index]
    dino = macro["dino_v3"]  # type: ignore[index]
    rows = [
        _metric_pair(
            sam3["prototype_segmentation"],  # type: ignore[index]
            task="SAM3 prototype segmentation",
            primary_metric="mIoU",
            primary_key="miou",
            secondary_metric="LocAcc",
            secondary_key="loc_acc",
            count_key="n_iou_samples",
        ),
        _metric_pair(
            dino["source_target_matching"],  # type: ignore[index]
            task="DINOv3 source-target matching",
            primary_metric="mIoU",
            primary_key="miou",
            secondary_metric="LocAcc",
            secondary_key="loc_acc",
            count_key="n_iou_samples",
        ),
    ]
    if "prototype_segmentation" in dino:
        rows.append(
            _metric_pair(
                dino["prototype_segmentation"],  # type: ignore[index]
                task="DINOv3 prototype segmentation",
                primary_metric="mIoU",
                primary_key="miou",
                secondary_metric="LocAcc",
                secondary_key="loc_acc",
                count_key="n_iou_samples",
            )
        )
    if "source_target_matching" in sam3:
        rows.append(
            _metric_pair(
                sam3["source_target_matching"],  # type: ignore[index]
                task="SAM3 source-target matching",
                primary_metric="mIoU",
                primary_key="miou",
                secondary_metric="LocAcc",
                secondary_key="loc_acc",
                count_key="n_iou_samples",
            )
        )
    return rows


def _build_optional_rows(
    path: Path | None,
    source_name: str,
    *,
    schema: str,
) -> list[dict[str, object]]:
    if path is None or not path.exists():
        return []
    loaded = _load_json(path)
    if schema == "formal":
        rows = _build_frozen_head_rows(loaded)
    elif schema == "prototype":
        rows = _build_prototype_rows(loaded)
    else:
        raise ValueError(f"Unknown optional schema: {schema}")
    for row in rows:
        row["source_name"] = source_name
    return rows


def _build_summary(
    text_rows: list[dict[str, object]],
    frozen_rows: list[dict[str, object]],
) -> dict[str, object]:
    full = next(row for row in text_rows if row["method"] == "Full CTF-GS")
    primary_total = 1 + len(frozen_rows)
    primary_rendered_wins = int(full["delta_miou"] > 0)
    primary_rendered_wins += sum(1 for row in frozen_rows if row["delta_primary"] > 0)
    caveats: list[str] = []
    for row in frozen_rows:
        if row["delta_primary"] < 0:
            caveats.append(
                f"{row['task']} {row['primary_metric']} remains teacher-stronger "
                f"({_fmt(row['rendered_primary'])} vs {_fmt(row['teacher_primary'])})."
            )
        if row["delta_secondary"] < 0:
            caveats.append(
                f"{row['task']} {row['secondary_metric']} remains teacher-stronger "
                f"({_fmt(row['rendered_secondary'])} vs {_fmt(row['teacher_secondary'])})."
            )
    if primary_rendered_wins == primary_total:
        claim_sentence = (
            "CTF-GS rendered features outperform the frame-wise RADIO teacher "
            "on all selected primary downstream feature-usability metrics; "
            "secondary LocAcc/HitRate caveats are reported separately."
        )
    else:
        claim_sentence = (
            "CTF-GS rendered features improve selected downstream feature-usability "
            "metrics over frame-wise RADIO teacher features, while caveats remain "
            "under the same frozen readout."
        )
    return {
        "primary_rendered_wins": primary_rendered_wins,
        "primary_total": primary_total,
        "universal_superiority": primary_rendered_wins == primary_total
        and not caveats,
        "caveats": caveats,
        "claim_sentence": claim_sentence,
    }


def build_report(
    controlled_path: Path,
    sam_dino_path: Path,
    *,
    ransac_path: Path | None = None,
    prototype_path: Path | None = None,
) -> dict[str, object]:
    controlled = _load_json(controlled_path)
    sam_dino = _load_json(sam_dino_path)
    text_rows = _build_text_grounding_rows(controlled)
    frozen_rows = _build_frozen_head_rows(sam_dino)
    return {
        "text_grounding_rows": text_rows,
        "frozen_head_rows": frozen_rows,
        "diagnostic_rows": {
            "dino_homography_ransac": _build_optional_rows(
                ransac_path,
                "formal_v8_mutual_homography_ransac_all_20260514",
                schema="formal",
            ),
            "prototype_adaptor": _build_optional_rows(
                prototype_path,
                "lerf_adaptor_downstream_mainline",
                schema="prototype",
            ),
        },
        "summary": _build_summary(text_rows, frozen_rows),
        "sources": {
            "controlled_evidence": str(controlled_path),
            "sam_dino_formal": str(sam_dino_path),
            "dino_homography_ransac": str(ransac_path) if ransac_path else "",
            "prototype_adaptor": str(prototype_path) if prototype_path else "",
        },
    }


def _write_markdown(report: Mapping[str, object], path: Path) -> None:
    lines = [
        "# Teacher vs CTF-GS 2D Feature Usability",
        "",
        "This report consolidates same-evaluator 2D evidence for selected downstream tasks. "
        "It supports a selected downstream tasks claim rather than universal feature superiority.",
        "",
        "## LERF Rendered-View Text Grounding and Feature Memory",
        "",
        "| Method | LocAcc | mIoU | Delta LocAcc vs teacher | Delta mIoU vs teacher |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["text_grounding_rows"]:  # type: ignore[index]
        lines.append(
            "| {method} | {loc} | {miou} | {dloc} | {dmiou} |".format(
                method=row["method"],
                loc=_fmt(row["loc_acc"]),
                miou=_fmt(row["miou"]),
                dloc=_fmt_delta(row["delta_loc_acc"]),
                dmiou=_fmt_delta(row["delta_miou"]),
            )
        )
    lines.extend(
        [
            "",
            "## Frozen-Head Downstream Tasks",
            "",
            "| Task | Primary | Teacher | CTF-GS rendered | Delta | Secondary | Teacher | CTF-GS rendered | Delta | N | Winner |",
            "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["frozen_head_rows"]:  # type: ignore[index]
        lines.append(
            "| {task} | {primary} | {tp} | {rp} | {dp} | {secondary} | {ts} | {rs} | {ds} | {n} | {winner} |".format(
                task=row["task"],
                primary=row["primary_metric"],
                tp=_fmt(row["teacher_primary"]),
                rp=_fmt(row["rendered_primary"]),
                dp=_fmt_delta(row["delta_primary"]),
                secondary=row["secondary_metric"],
                ts=_fmt(row["teacher_secondary"]),
                rs=_fmt(row["rendered_secondary"]),
                ds=_fmt_delta(row["delta_secondary"]),
                n=int(row["n"]),
                winner=row["winner"],
            )
        )
    summary = report["summary"]  # type: ignore[index]
    claim_sentence = summary.get(  # type: ignore[union-attr]
        "claim_sentence",
        "CTF-GS rendered features improve selected downstream tasks, with "
        "explicit caveats where the frame-wise teacher remains stronger.",
    )
    lines.extend(
        [
            "",
            "## Claim-Safe Summary",
            "",
            "- Primary rendered wins: "
            f"{summary['primary_rendered_wins']} / {summary['primary_total']}.",
            f"- Universal superiority claim allowed: {summary['universal_superiority']}.",
            f"- Recommended wording: {claim_sentence}",
            "",
            "## Caveats",
            "",
        ]
    )
    caveats = list(summary["caveats"])  # type: ignore[index]
    if caveats:
        lines.extend(f"- {item}" for item in caveats)
    else:
        lines.append("- No teacher-stronger frozen-head metrics in the selected primary/secondary set.")
    lines.extend(["", "## Sources", ""])
    for key, value in report["sources"].items():  # type: ignore[index]
        if value:
            lines.append(f"- {key}: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex(report: Mapping[str, object], path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{2D feature-usability comparison between frame-wise RADIO teacher features and rendered CTF-GS features under frozen downstream heads. CTF-GS wins all selected primary metrics; secondary LocAcc/HitRate caveats are reported in the artifact.}",
        r"\label{tab:teacher_vs_ctfgs_2d_usability}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{l l r r r}",
        r"\toprule",
        r"Task & Metric & Teacher & CTF-GS & $\Delta$ \\",
        r"\midrule",
    ]
    full = next(
        row
        for row in report["text_grounding_rows"]  # type: ignore[index]
        if row["method"] == "Full CTF-GS"
    )
    teacher = next(
        row
        for row in report["text_grounding_rows"]  # type: ignore[index]
        if row["method"] == "Frame-wise RADIO teacher"
    )
    lines.append(
        "LERF text grounding & mIoU & "
        f"{_fmt(teacher['miou'])} & {_fmt(full['miou'])} & {_fmt_delta(full['delta_miou'])} \\\\"
    )
    for row in report["frozen_head_rows"]:  # type: ignore[index]
        lines.append(
            f"{row['task']} & {row['primary_metric']} & "
            f"{_fmt(row['teacher_primary'])} & {_fmt(row['rendered_primary'])} & "
            f"{_fmt_delta(row['delta_primary'])} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    report: Mapping[str, object],
    json_path: Path,
    markdown_path: Path,
    latex_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(report, markdown_path)
    _write_latex(report, latex_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controlled", type=Path, default=DEFAULT_CONTROLLED)
    parser.add_argument("--sam_dino", type=Path, default=DEFAULT_SAM_DINO)
    parser.add_argument("--ransac", type=Path, default=DEFAULT_RANSAC)
    parser.add_argument("--prototype", type=Path, default=DEFAULT_PROTOTYPE)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--latex", type=Path, default=DEFAULT_LATEX)
    args = parser.parse_args()

    built = build_report(
        args.controlled,
        args.sam_dino,
        ransac_path=args.ransac if args.ransac.exists() else None,
        prototype_path=args.prototype if args.prototype.exists() else None,
    )
    write_outputs(built, args.json, args.markdown, args.latex)
    print(f"Wrote {args.json}")
    print(f"Wrote {args.markdown}")
    print(f"Wrote {args.latex}")


if __name__ == "__main__":
    main()
