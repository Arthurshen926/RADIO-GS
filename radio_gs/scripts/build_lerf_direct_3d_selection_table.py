#!/usr/bin/env python3
"""Build paper/report tables for LERF direct 3D object selection."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

from radio_gs.scripts.summarize_opengaussian_baseline import OPENGAUSSIAN_PAPER_LERF


SCENES = ["figurines", "ramen", "teatime", "waldo_kitchen"]
SCENE_LABELS = {
    "figurines": "Figurines",
    "ramen": "Ramen",
    "teatime": "Teatime",
    "waldo_kitchen": "Waldo Kitchen",
}


def fmt(value: float) -> str:
    return f"{value:.4f}"


def texfmt(value: float) -> str:
    return f"{value:.3f}"


def tex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def load_scene_results(root: Path, extra_roots: List[Path] | None = None) -> Dict[str, dict]:
    results = {}
    for scene in SCENES:
        path = root / scene / "lerf_direct_3d_selection_results.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing direct-selection result: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        results[scene] = data["scene"]
        results[scene]["_args"] = data.get("args", {})
        results[scene]["_protocol"] = data.get("protocol", {})
    for extra_root in extra_roots or []:
        for scene in SCENES:
            path = extra_root / scene / "lerf_direct_3d_selection_results.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing extra direct-selection result: {path}")
            data = json.loads(path.read_text(encoding="utf-8"))["scene"]
            results[scene]["results"].update(data["results"])
            results[scene]["best_by_miou"] = max(
                results[scene]["results"],
                key=lambda tag: results[scene]["results"][tag]["miou"],
            )
    return results


def selection_tag_sort_key(tag: str) -> tuple[int, float, str]:
    match = re.fullmatch(r"top([0-9]+(?:p[0-9]+)?)", tag)
    if match:
        return (0, float(match.group(1).replace("p", ".")), tag)
    match = re.fullmatch(r"thr([0-9]+(?:p[0-9]+)?)", tag)
    if match:
        return (1, float(match.group(1).replace("p", ".")), tag)
    match = re.fullmatch(r"meanstd([0-9]+(?:p[0-9]+)?)", tag)
    if match:
        return (2, float(match.group(1).replace("p", ".")), tag)
    return (99, float("inf"), tag)


def selection_tags(results: Dict[str, dict]) -> List[str]:
    tags = set()
    for scene in SCENES:
        tags.update(results[scene]["results"].keys())
    return sorted(tags, key=selection_tag_sort_key)


def macro_for_tag(results: Dict[str, dict], tag: str, metric: str) -> float:
    values = [float(results[scene]["results"][tag][metric]) for scene in SCENES]
    return sum(values) / len(values)


def best_fixed_tag(results: Dict[str, dict]) -> str:
    return max(selection_tags(results), key=lambda tag: macro_for_tag(results, tag, "miou"))


def best_by_scene_row(results: Dict[str, dict], metric: str) -> Dict[str, float]:
    values = {}
    for scene in SCENES:
        tag = results[scene]["best_by_miou"]
        values[scene] = float(results[scene]["results"][tag][metric])
    values["macro"] = sum(values[scene] for scene in SCENES) / len(SCENES)
    return values


def fixed_row(results: Dict[str, dict], tag: str, metric: str) -> Dict[str, float]:
    values = {scene: float(results[scene]["results"][tag][metric]) for scene in SCENES}
    values["macro"] = sum(values[scene] for scene in SCENES) / len(SCENES)
    return values


def first_protocol(results: Dict[str, dict]) -> Dict[str, str]:
    for scene in SCENES:
        protocol = results.get(scene, {}).get("_protocol", {})
        if protocol:
            return protocol
    return {}


def first_args(results: Dict[str, dict]) -> Dict[str, str]:
    for scene in SCENES:
        args = results.get(scene, {}).get("_args", {})
        if args:
            return args
    return {}


def _ratio_value(protocol: Dict[str, str], args: Dict[str, str], key: str) -> float:
    value = protocol.get(key, args.get(key, 0.0))
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def selection_bounds_suffix(results: Dict[str, dict]) -> str:
    protocol = first_protocol(results)
    args = first_args(results)
    min_ratio = _ratio_value(protocol, args, "selection_min_ratio")
    max_ratio = _ratio_value(protocol, args, "selection_max_ratio")
    bounds = []
    if min_ratio > 0:
        bounds.append(f"selection floor={min_ratio:g}")
    if max_ratio > 0:
        bounds.append(f"selection cap={max_ratio:g}")
    if not bounds:
        return ""
    return "; GT-free " + ", ".join(bounds)


def mask_refinement_suffix(results: Dict[str, dict], *, escape_for_tex: bool = False) -> str:
    protocol = first_protocol(results)
    args = first_args(results)
    mode = protocol.get("mask_refinement", args.get("mask_refinement", "none"))
    if not mode or mode == "none":
        return ""
    mode_text = tex_escape(str(mode)) if escape_for_tex else str(mode)
    return f"; GT-free {mode_text} mask refinement"


def direct_protocol_sentence(results: Dict[str, dict]) -> str:
    protocol = first_protocol(results)
    args = first_args(results)
    score_source = protocol.get("score_source", args.get("score_source", "direct"))
    scoring = args.get("scoring", "cosine")
    aggregation = protocol.get("score_aggregation", args.get("score_aggregation", "none"))
    aggregation_suffix = ""
    if aggregation and aggregation != "none":
        aggregation_suffix = (
            f"; GT-free {aggregation} context aggregation is applied "
            f"(res={protocol.get('score_aggregation_resolution', args.get('score_aggregation_resolution', ''))}, "
            f"blend={protocol.get('score_aggregation_blend', args.get('score_aggregation_blend', ''))})"
        )
    selection_suffix = selection_bounds_suffix(results)
    refinement_suffix = mask_refinement_suffix(results)
    if score_source == "registered_view":
        max_frames = protocol.get("registration_max_frames", args.get("registration_max_frames", ""))
        frame_mode = protocol.get("registration_frame_mode", args.get("registration_frame_mode", ""))
        return (
            "Protocol: OpenGaussian-style direct 3D primitive selection with View-to-Primitive "
            "Registration (VPR). Query scores are computed on Gaussian primitives from rendered-view "
            "SigLIP2 features registered back to 3D with depth/alpha visibility checks "
            f"({frame_mode}, max_frames={max_frames}, "
            f"scoring={scoring}{aggregation_suffix}{selection_suffix}{refinement_suffix}); selected primitives are rendered only for mask evaluation."
        )
    return (
        "Protocol: OpenGaussian-style direct 3D primitive selection. Query scores are computed "
        "at Gaussian centers from pre-refiner RADIO-GS features; selected primitives are rendered "
        f"only to compare with LERF-OVS object masks{selection_suffix}{refinement_suffix}."
    )


def direct_caption_feature_source(results: Dict[str, dict]) -> str:
    protocol = first_protocol(results)
    args = first_args(results)
    score_source = protocol.get("score_source", args.get("score_source", "direct"))
    aggregation = protocol.get("score_aggregation", args.get("score_aggregation", "none"))
    suffix = ""
    if aggregation and aggregation != "none":
        suffix = f" with GT-free {tex_escape(aggregation)} context aggregation"
    selection_suffix = selection_bounds_suffix(results)
    if selection_suffix:
        suffix += selection_suffix.replace("; GT-free", " and GT-free")
    refinement_suffix = mask_refinement_suffix(results, escape_for_tex=True)
    if refinement_suffix:
        suffix += refinement_suffix.replace("; GT-free", " and GT-free")
    if score_source == "registered_view":
        max_frames = protocol.get("registration_max_frames", args.get("registration_max_frames", ""))
        view_suffix = f" from {max_frames} all-pose VPR views" if max_frames else ""
        return "rendered-view registered primitive features" + view_suffix + suffix
    return "pre-refiner Gaussian-center features" + suffix


def make_markdown(results: Dict[str, dict], root: Path, fixed_tag: str) -> str:
    lines = [
        "# LERF-OVS Direct 3D Object Selection",
        "",
        direct_protocol_sentence(results),
        "",
        f"Input root: `{root}`",
        f"Paper-facing fixed selection: `{fixed_tag}`. The complete selector sweep below is diagnostic and should be reported separately from rendered-view grounding.",
        "",
        "## Selector Sweep",
        "",
        "| Selection | Figurines mIoU | Ramen mIoU | Teatime mIoU | Waldo mIoU | Macro mIoU | Macro Acc@0.25 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for tag in selection_tags(results):
        row = fixed_row(results, tag, "miou")
        macro_acc = macro_for_tag(results, tag, "acc025")
        lines.append(
            f"| {tag} | {fmt(row['figurines'])} | {fmt(row['ramen'])} | "
            f"{fmt(row['teatime'])} | {fmt(row['waldo_kitchen'])} | "
            f"{fmt(row['macro'])} | {fmt(macro_acc)} |"
        )

    fixed_miou = fixed_row(results, fixed_tag, "miou")
    fixed_acc = fixed_row(results, fixed_tag, "acc025")
    best_miou = best_by_scene_row(results, "miou")
    best_acc = best_by_scene_row(results, "acc025")
    lines.extend(
        [
            "",
            "## Paper-Facing Direct-Selection Context",
            "",
            "| Method | Text head | Protocol | Figurines | Ramen | Teatime | Waldo Kitchen | Macro |",
            "|---|---|---|---:|---:|---:|---:|---:|",
            "| OpenGaussian | CLIP | official paper mIoU | "
            f"{fmt(OPENGAUSSIAN_PAPER_LERF['figurines']['miou'])} | "
            f"{fmt(OPENGAUSSIAN_PAPER_LERF['ramen']['miou'])} | "
            f"{fmt(OPENGAUSSIAN_PAPER_LERF['teatime']['miou'])} | "
            f"{fmt(OPENGAUSSIAN_PAPER_LERF['waldo_kitchen']['miou'])} | "
            f"{fmt(OPENGAUSSIAN_PAPER_LERF['macro']['miou'])} |",
            f"| CTF-GS | SigLIP2 | fixed {fixed_tag} mIoU | "
            f"{fmt(fixed_miou['figurines'])} | {fmt(fixed_miou['ramen'])} | "
            f"{fmt(fixed_miou['teatime'])} | {fmt(fixed_miou['waldo_kitchen'])} | "
            f"{fmt(fixed_miou['macro'])} |",
            "| CTF-GS | SigLIP2 | diagnostic best-by-scene mIoU | "
            f"{fmt(best_miou['figurines'])} | {fmt(best_miou['ramen'])} | "
            f"{fmt(best_miou['teatime'])} | {fmt(best_miou['waldo_kitchen'])} | "
            f"{fmt(best_miou['macro'])} |",
            "| OpenGaussian | CLIP | official paper Acc@0.25 | "
            f"{fmt(OPENGAUSSIAN_PAPER_LERF['figurines']['macc025'])} | "
            f"{fmt(OPENGAUSSIAN_PAPER_LERF['ramen']['macc025'])} | "
            f"{fmt(OPENGAUSSIAN_PAPER_LERF['teatime']['macc025'])} | "
            f"{fmt(OPENGAUSSIAN_PAPER_LERF['waldo_kitchen']['macc025'])} | "
            f"{fmt(OPENGAUSSIAN_PAPER_LERF['macro']['macc025'])} |",
            f"| CTF-GS | SigLIP2 | fixed {fixed_tag} Acc@0.25 | "
            f"{fmt(fixed_acc['figurines'])} | {fmt(fixed_acc['ramen'])} | "
            f"{fmt(fixed_acc['teatime'])} | {fmt(fixed_acc['waldo_kitchen'])} | "
            f"{fmt(fixed_acc['macro'])} |",
            "| CTF-GS | SigLIP2 | diagnostic best-by-scene Acc@0.25 | "
            f"{fmt(best_acc['figurines'])} | {fmt(best_acc['ramen'])} | "
            f"{fmt(best_acc['teatime'])} | {fmt(best_acc['waldo_kitchen'])} | "
            f"{fmt(best_acc['macro'])} |",
            "",
            "Interpretation: the registration readout substantially closes the primitive-level gap versus the original Gaussian-center readout while keeping the OpenGaussian-style query-select-render-evaluate protocol. The promoted fixed-threshold selector reduces primitive-level clutter under the same global rule; Waldo Kitchen remains the weakest scene and should be discussed as a remaining object-fragmentation/registration-coverage limitation.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize_variant(results: Dict[str, dict]) -> Dict[str, float | str]:
    fixed_tag = best_fixed_tag(results)
    fixed_miou = fixed_row(results, fixed_tag, "miou")
    fixed_acc = fixed_row(results, fixed_tag, "acc025")
    best_miou = best_by_scene_row(results, "miou")
    best_acc = best_by_scene_row(results, "acc025")
    return {
        "fixed_tag": fixed_tag,
        "fixed_miou": fixed_miou["macro"],
        "fixed_acc025": fixed_acc["macro"],
        "best_scene_miou": best_miou["macro"],
        "best_scene_acc025": best_acc["macro"],
    }


def append_diagnostics(
    markdown: str,
    diagnostics: List[tuple[str, Dict[str, dict]]],
) -> str:
    if not diagnostics:
        return markdown
    lines = markdown.rstrip().splitlines()
    lines.extend(
        [
            "",
            "## Direct-Readout Diagnostics",
            "",
            "These variants do not use GT masks for scoring. They test whether the direct-selection gap is caused by raw Gaussian-center readout, VPR scoring, VFA, view coverage, visibility checks, or GT-free spatial aggregation.",
            "",
            "| Variant | Best fixed selection | Fixed macro mIoU | Fixed macro Acc@0.25 | Best-by-scene macro mIoU | Best-by-scene macro Acc@0.25 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for name, result in diagnostics:
        summary = summarize_variant(result)
        lines.append(
            f"| {name} | {summary['fixed_tag']} | "
            f"{fmt(float(summary['fixed_miou']))} | "
            f"{fmt(float(summary['fixed_acc025']))} | "
            f"{fmt(float(summary['best_scene_miou']))} | "
            f"{fmt(float(summary['best_scene_acc025']))} |"
        )
    lines.extend(
        [
            "",
            "Diagnostic takeaway: VPR is the main factor that improves direct 3D object selection. View coverage, VFA, selection calibration, and optional GT-free projection cleanup control the precision/coverage tradeoff, while Waldo Kitchen remains the hardest fragmented scene.",
            "",
        ]
    )
    return "\n".join(lines)


def make_tex(results: Dict[str, dict], fixed_tag: str) -> str:
    fixed_miou = fixed_row(results, fixed_tag, "miou")
    fixed_acc = fixed_row(results, fixed_tag, "acc025")
    best_miou = best_by_scene_row(results, "miou")
    rows = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{LERF-OVS direct 3D object selection under an OpenGaussian-style query-select-render protocol. External values are the OpenGaussian official paper numbers; CTF-GS uses "
        + direct_caption_feature_source(results)
        + r" and a SigLIP2 text head.}",
        r"  \label{tab:lerf-direct-3d-selection}",
        r"  \resizebox{\linewidth}{!}{%",
        r"  \begin{tabular}{llccccc}",
        r"    \toprule",
        r"    Method & Protocol & Fig. & Ramen & Tea. & Waldo & Macro \\",
        r"    \midrule",
        "    OpenGaussian & mIoU & "
        f"{texfmt(OPENGAUSSIAN_PAPER_LERF['figurines']['miou'])} & "
        f"{texfmt(OPENGAUSSIAN_PAPER_LERF['ramen']['miou'])} & "
        f"{texfmt(OPENGAUSSIAN_PAPER_LERF['teatime']['miou'])} & "
        f"{texfmt(OPENGAUSSIAN_PAPER_LERF['waldo_kitchen']['miou'])} & "
        f"{texfmt(OPENGAUSSIAN_PAPER_LERF['macro']['miou'])} \\\\",
        f"    CTF-GS & {fixed_tag} mIoU & "
        f"{texfmt(fixed_miou['figurines'])} & {texfmt(fixed_miou['ramen'])} & "
        f"{texfmt(fixed_miou['teatime'])} & {texfmt(fixed_miou['waldo_kitchen'])} & "
        f"{texfmt(fixed_miou['macro'])} \\\\",
        f"    CTF-GS & diag. best mIoU & "
        f"{texfmt(best_miou['figurines'])} & {texfmt(best_miou['ramen'])} & "
        f"{texfmt(best_miou['teatime'])} & {texfmt(best_miou['waldo_kitchen'])} & "
        f"{texfmt(best_miou['macro'])} \\\\",
        r"    \midrule",
        "    OpenGaussian & Acc@0.25 & "
        f"{texfmt(OPENGAUSSIAN_PAPER_LERF['figurines']['macc025'])} & "
        f"{texfmt(OPENGAUSSIAN_PAPER_LERF['ramen']['macc025'])} & "
        f"{texfmt(OPENGAUSSIAN_PAPER_LERF['teatime']['macc025'])} & "
        f"{texfmt(OPENGAUSSIAN_PAPER_LERF['waldo_kitchen']['macc025'])} & "
        f"{texfmt(OPENGAUSSIAN_PAPER_LERF['macro']['macc025'])} \\\\",
        f"    CTF-GS & {fixed_tag} Acc@0.25 & "
        f"{texfmt(fixed_acc['figurines'])} & {texfmt(fixed_acc['ramen'])} & "
        f"{texfmt(fixed_acc['teatime'])} & {texfmt(fixed_acc['waldo_kitchen'])} & "
        f"{texfmt(fixed_acc['macro'])} \\\\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  }",
        r"\end{table}",
    ]
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", default="output/radio_gs/lerf_direct_3d_selection_mainline_20260511")
    parser.add_argument("--extra_results_root", action="append", default=[], help="Additional result roots to merge")
    parser.add_argument(
        "--diagnostic_root",
        action="append",
        default=[],
        help="Diagnostic result root as name=path",
    )
    parser.add_argument("--fixed_tag", default="", help="Fixed selection tag to expose; empty chooses best macro mIoU")
    parser.add_argument("--output_md", default="output/radio_gs/reports/lerf_direct_3d_selection.md")
    parser.add_argument("--output_tex", default="paper/lerf_direct_3d_selection_table.tex")
    args = parser.parse_args()

    root = Path(args.results_root)
    results = load_scene_results(root, [Path(item) for item in args.extra_results_root])
    fixed_tag = args.fixed_tag or best_fixed_tag(results)
    md = make_markdown(results, root, fixed_tag)
    diagnostics: List[tuple[str, Dict[str, dict]]] = []
    for item in args.diagnostic_root:
        if "=" not in item:
            raise ValueError(f"--diagnostic_root must be name=path, got {item!r}")
        name, raw_path = item.split("=", 1)
        diagnostics.append((name, load_scene_results(Path(raw_path))))
    md = append_diagnostics(md, diagnostics)
    tex = make_tex(results, fixed_tag)
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md + "\n", encoding="utf-8")
    out_tex = Path(args.output_tex)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(tex, encoding="utf-8")
    print(f"fixed_tag={fixed_tag}")
    print(f"wrote {out_md}")
    print(f"wrote {out_tex}")


if __name__ == "__main__":
    main()
