#!/usr/bin/env python3
"""Summarize controlled seed-7 LERF component ablations."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "output" / "radio_gs"

SCENES = ("figurines", "ramen", "teatime", "waldo_kitchen")
SCENE_LABELS = {
    "figurines": "Figurines",
    "ramen": "Ramen",
    "teatime": "Teatime",
    "waldo_kitchen": "Waldo Kitchen",
}


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    exp_template: str
    config_template: str
    note: str


VARIANTS = (
    Variant(
        key="full",
        label="Full RADIO-GS",
        exp_template="lerf_{scene}_v14_fdh_ws240_240ep_seed7",
        config_template="radio_gs/configs/generated/seeds/lerf_hybrid_v14_{scene}_fdh_ws240_240ep_seed7.yaml",
        note="hybrid + HCD + refiner + FDH warm-start",
    ),
    Variant(
        key="no_fdh",
        label="w/o FDH warm-start",
        exp_template="lerf_{scene}_v14_nofdh_240ep_seed7",
        config_template="radio_gs/configs/generated/seeds/lerf_hybrid_v14_{scene}_nofdh_240ep_seed7.yaml",
        note="same hybrid/HCD/refiner, no frozen-depth-head stage",
    ),
    Variant(
        key="no_refiner",
        label="w/o refiner",
        exp_template="lerf_{scene}_component_no_refiner_seed7",
        config_template="radio_gs/configs/generated/ablation/lerf_{scene}_component_no_refiner_seed7.yaml",
        note="refiner disabled during the FDH refinement run",
    ),
    Variant(
        key="no_hybrid",
        label="w/o hybrid",
        exp_template="lerf_{scene}_component_no_hybrid_seed7",
        config_template="radio_gs/configs/generated/ablation/lerf_{scene}_component_no_hybrid_seed7.yaml",
        note="explicit per-Gaussian compact field, HCD/refiner retained",
    ),
    Variant(
        key="direct_codec",
        label="w/o HCD",
        exp_template="lerf_{scene}_component_direct_codec_seed7",
        config_template="radio_gs/configs/generated/ablation/lerf_{scene}_component_direct_codec_seed7.yaml",
        note="direct 1x1 projection codec replaces HCD",
    ),
)


def scene_config_template(template: str, scene: str) -> str:
    return template.format(scene=scene)


def exp_dir_for(variant: Variant, scene: str) -> Path:
    return OUTPUT_ROOT / variant.exp_template.format(scene=scene)


def config_path_for(variant: Variant, scene: str) -> Path:
    return REPO_ROOT / scene_config_template(variant.config_template, scene)


def _parse_temp_from_dir(path: Path) -> float | None:
    name = path.name
    if not name.startswith("T"):
        return None
    try:
        return float(name[1:].replace("p", "."))
    except ValueError:
        return None


def _result_from_lerf_json(path: Path, scene: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["scenes"][scene]["rendered"]
        temp = _parse_temp_from_dir(path.parent)
        return {
            "loc_acc": float(metrics["loc_acc"]),
            "miou": float(metrics["miou"]),
            "loc_total": int(metrics["loc_total"]),
            "temp": temp,
            "source": str(path),
        }
    except Exception:
        return None


def _summary_best(summary_path: Path, scene: str) -> dict[str, Any] | None:
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        best = summary.get("best") or {}
        loc_acc = best.get("loc_acc", best.get("rendered_loc_acc"))
        miou = best.get("miou", best.get("rendered_miou"))
        if loc_acc is None or miou is None:
            return None
        output_dir = Path(str(best.get("output_dir", summary_path.parent)))
        result_path = output_dir / "lerf_ovs_results.json"
        return {
            "loc_acc": float(loc_acc),
            "miou": float(miou),
            "loc_total": int(best.get("loc_total", best.get("rendered_loc_total", 0))),
            "temp": best.get("temp", best.get("temperature")),
            "source": str(result_path if result_path.exists() else summary_path),
            "summary": str(summary_path),
            "checkpoint": summary.get("checkpoint"),
            "config": summary.get("config"),
        }
    except Exception:
        return None


def collect_best_result(exp_dir: Path, scene: str) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for summary_path in sorted(exp_dir.glob("lerf_eval_*/summary.json")):
        parsed = _summary_best(summary_path, scene)
        if parsed is not None:
            candidates.append(parsed)
    for result_path in sorted(exp_dir.glob("lerf_eval_*/T*/lerf_ovs_results.json")):
        parsed = _result_from_lerf_json(result_path, scene)
        if parsed is not None:
            candidates.append(parsed)
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item["loc_acc"], item["miou"]))


def collect_all() -> dict[str, dict[str, dict[str, Any] | None]]:
    data: dict[str, dict[str, dict[str, Any] | None]] = {}
    for variant in VARIANTS:
        scene_rows: dict[str, dict[str, Any] | None] = {}
        for scene in SCENES:
            result = collect_best_result(exp_dir_for(variant, scene), scene)
            if result is not None:
                result = {
                    **result,
                    "scene": scene,
                    "variant": variant.key,
                    "exp_dir": str(exp_dir_for(variant, scene)),
                    "config_path": str(config_path_for(variant, scene)),
                }
            scene_rows[scene] = result
        data[variant.key] = scene_rows
    return data


def _macro(rows: dict[str, dict[str, Any] | None], key: str) -> float | None:
    values = [float(row[key]) for row in rows.values() if row is not None]
    if len(values) != len(SCENES):
        return None
    return sum(values) / len(values)


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _display_path(path_like: str | Path) -> str:
    path = Path(path_like)
    try:
        return str(path.relative_to(REPO_ROOT))
    except Exception:
        pass
    try:
        rel_output = path.resolve().relative_to(OUTPUT_ROOT.resolve())
        return str(Path("output") / "radio_gs" / rel_output)
    except Exception:
        pass
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(path)


def build_markdown(data: dict[str, dict[str, dict[str, Any] | None]]) -> str:
    full_macro = _macro(data["full"], "loc_acc")
    lines = [
        "# LERF Component Ablation",
        "",
        "Protocol: controlled seed-7 LERF-OVS evaluation. Each row is selected by the same scene-specific temperature sweep over rendered features; ties are resolved by mIoU. Full/component rows follow the same scene-level FDH route defined by their configs, while the no-FDH row removes the frozen-depth-head stage. The current-best selector table is intentionally kept separate.",
        "",
        "## Macro Summary",
        "",
        "| Variant | Scenes | Macro LocAcc | Delta | Macro mIoU | Note |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for variant in VARIANTS:
        rows = data[variant.key]
        done = sum(1 for row in rows.values() if row is not None)
        loc = _macro(rows, "loc_acc")
        miou = _macro(rows, "miou")
        delta = loc - full_macro if loc is not None and full_macro is not None else None
        lines.append(
            f"| {variant.label} | {done}/4 | {_fmt(loc)} | {_fmt(delta)} | {_fmt(miou)} | {variant.note} |"
        )

    lines.extend(["", "## Per-Scene LocAcc / mIoU", ""])
    for scene in SCENES:
        lines.append(f"### {SCENE_LABELS[scene]}")
        lines.append("")
        lines.append("| Variant | LocAcc | mIoU | Temp | Source |")
        lines.append("|---|---:|---:|---:|---|")
        for variant in VARIANTS:
            row = data[variant.key][scene]
            if row is None:
                lines.append(f"| {variant.label} | - | - | - | pending |")
                continue
            temp = row.get("temp")
            temp_text = "-" if temp is None else f"{float(temp):.1f}"
            lines.append(
                f"| {variant.label} | {_fmt(row['loc_acc'])} | {_fmt(row['miou'])} | {temp_text} | `{_display_path(row['source'])}` |"
            )
        lines.append("")

    lines.extend(["## Provenance", ""])
    lines.append("| Variant | Scene | Config | Experiment dir | Status |")
    lines.append("|---|---|---|---|---|")
    for variant in VARIANTS:
        for scene in SCENES:
            row = data[variant.key][scene]
            status = "ready" if row is not None else "pending"
            lines.append(
                f"| {variant.label} | {SCENE_LABELS[scene]} | `{_display_path(config_path_for(variant, scene))}` | `{_display_path(exp_dir_for(variant, scene))}` | {status} |"
            )
    lines.append("")
    return "\n".join(lines)


def build_latex(data: dict[str, dict[str, dict[str, Any] | None]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Controlled LERF-OVS component ablation on seed 7. The table reports rendered-feature LocAcc per scene plus macro LocAcc and mIoU.}",
        r"\label{tab:component_ablation}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Variant & Fig. & Ramen & Tea. & Waldo & LocAcc & mIoU \\",
        r"\midrule",
    ]
    for variant in VARIANTS:
        rows = data[variant.key]
        scene_cells = [
            _fmt(rows[scene]["loc_acc"], 3) if rows[scene] is not None else r"\textemdash"
            for scene in SCENES
        ]
        loc = _macro(rows, "loc_acc")
        miou = _macro(rows, "miou")
        loc_cell = _fmt(loc, 3) if loc is not None else r"\textemdash"
        miou_cell = _fmt(miou, 3) if miou is not None else r"\textemdash"
        label = r"Full \method{}" if variant.label == "Full GaussFM" else variant.label
        lines.append(
            f"{label} & "
            + " & ".join(scene_cells)
            + f" & {loc_cell}"
            + f" & {miou_cell} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def write_json(path: Path, data: dict[str, dict[str, dict[str, Any] | None]]) -> None:
    payload = {
        "protocol": "controlled seed-7 LERF-OVS rendered-feature temperature sweep",
        "scenes": list(SCENES),
        "variants": [
            {
                "key": variant.key,
                "label": variant.label,
                "note": variant.note,
            }
            for variant in VARIANTS
        ],
        "results": data,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--markdown",
        default=str(OUTPUT_ROOT / "reports" / "lerf_component_ablation.md"),
    )
    parser.add_argument(
        "--json",
        default=str(OUTPUT_ROOT / "reports" / "lerf_component_ablation.json"),
    )
    parser.add_argument(
        "--latex",
        default=str(REPO_ROOT / "paper" / "lerf_component_ablation_table.tex"),
    )
    args = parser.parse_args()

    data = collect_all()
    md_path = Path(args.markdown)
    json_path = Path(args.json)
    latex_path = Path(args.latex)
    for path in (md_path, json_path, latex_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(build_markdown(data), encoding="utf-8")
    write_json(json_path, data)
    latex_path.write_text(build_latex(data), encoding="utf-8")
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {latex_path}")


if __name__ == "__main__":
    main()
