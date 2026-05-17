#!/usr/bin/env python3
"""Summarize LERF direct-3D threshold sweeps across scenes."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "output" / "radio_gs" / "reports"
DEFAULT_JSON = REPORT_DIR / "lerf_sam3_box_global_threshold_sweep_20260516.json"
DEFAULT_MARKDOWN = REPORT_DIR / "lerf_sam3_box_global_threshold_sweep_20260516.md"
DIRECT3D_SCENES = ("figurines", "ramen", "teatime", "waldo_kitchen")
METRIC_KEYS = ("miou", "acc025", "acc050", "boundary_f", "trimap_iou")


def _round4(value: float) -> float:
    return round(float(value), 4)


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return _round4(sum(float(row.get(key, 0.0)) for row in rows) / len(rows))


def _selection_value(tag: str) -> float:
    match = re.search(r"(\d+)p(\d+)", tag)
    if not match:
        return float("inf")
    return float(f"{match.group(1)}.{match.group(2)}")


def _tag_sort_key(tag: str) -> tuple[float, str]:
    return (_selection_value(tag), tag)


def _load_scene_payload(root: Path, scene: str) -> tuple[Path, dict[str, Any]] | None:
    preferred = [
        root / scene / "lerf_direct_3d_selection_results.json",
        root / scene / scene / "lerf_direct_3d_selection_results.json",
    ]
    for path in preferred:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("scene", {}).get("scene") == scene:
            return path, payload

    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(root.glob("**/lerf_direct_3d_selection_results.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("scene", {}).get("scene") == scene:
            matches.append((path, payload))
    if not matches:
        return None
    return min(matches, key=lambda item: (len(item[0].parts), str(item[0])))


def _scene_entries(
    root: Path,
    expected_scenes: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    entries: list[dict[str, Any]] = []
    missing: list[str] = []
    warnings: list[str] = []
    for scene in expected_scenes:
        match = _load_scene_payload(root, scene)
        if match is None:
            missing.append(scene)
            warnings.append(f"missing direct-3D scene result for {scene} under {root}")
            continue
        path, payload = match
        scene_payload = payload.get("scene", {})
        results = scene_payload.get("results", {})
        if not results:
            warnings.append(f"{scene}: no threshold results in {path}")
            continue
        entries.append(
            {
                "scene": scene,
                "path": str(path),
                "results": results,
                "best_by_miou": scene_payload.get("best_by_miou"),
                "protocol": payload.get("protocol", {}),
            }
        )
    return entries, missing, warnings


def _row(scene: str, tag: str, metrics: dict[str, Any], path: str) -> dict[str, Any]:
    return {
        "scene": scene,
        "selection": tag,
        "miou": _round4(float(metrics.get("miou", 0.0))),
        "acc025": _round4(float(metrics.get("acc025", 0.0))),
        "acc050": _round4(float(metrics.get("acc050", 0.0))),
        "boundary_f": _round4(float(metrics.get("boundary_f", 0.0))),
        "trimap_iou": _round4(float(metrics.get("trimap_iou", 0.0))),
        "n": int(metrics.get("n", 0)),
        "source": path,
    }


def _readout(
    selection: str,
    selector_policy: str,
    rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    macro_rows = raw_rows if raw_rows is not None else rows
    return {
        "selection": selection,
        "selector_policy": selector_policy,
        "scene_count": len(rows),
        "macro_miou": _mean_metric(macro_rows, "miou"),
        "macro_acc025": _mean_metric(macro_rows, "acc025"),
        "macro_acc050": _mean_metric(macro_rows, "acc050"),
        "macro_boundary_f": _mean_metric(macro_rows, "boundary_f"),
        "macro_trimap_iou": _mean_metric(macro_rows, "trimap_iou"),
        "rows": rows,
    }


def _fixed_tag_readout(
    entries: list[dict[str, Any]],
    tag: str,
    *,
    selector_policy: str,
) -> tuple[dict[str, Any], list[str]]:
    rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for entry in entries:
        results = entry["results"]
        scene = entry["scene"]
        if tag not in results:
            warnings.append(f"{scene}: missing threshold tag {tag}")
            continue
        metrics = results[tag]
        raw_rows.append({key: float(metrics.get(key, 0.0)) for key in METRIC_KEYS})
        rows.append(_row(scene, tag, metrics, entry["path"]))
    return _readout(tag, selector_policy, rows, raw_rows), warnings


def _scene_locked_readout(entries: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for entry in entries:
        results = entry["results"]
        tag = entry.get("best_by_miou")
        if tag not in results:
            tag = max(results, key=lambda item: float(results[item].get("miou", 0.0)))
        metrics = results[tag]
        raw_rows.append({key: float(metrics.get(key, 0.0)) for key in METRIC_KEYS})
        rows.append(_row(entry["scene"], str(tag), metrics, entry["path"]))
    return _readout("scene_locked", "diagnostic_scene_locked_best_by_miou", rows, raw_rows)


def _best_fixed_macro_readout(entries: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not entries:
        empty = _readout("", "diagnostic_posthoc_best_fixed_threshold", [])
        return empty, []
    common_tags = set(entries[0]["results"].keys())
    for entry in entries[1:]:
        common_tags &= set(entry["results"].keys())

    sweep_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for tag in sorted(common_tags, key=_tag_sort_key):
        readout, _ = _fixed_tag_readout(
            entries,
            tag,
            selector_policy="diagnostic_posthoc_best_fixed_threshold",
        )
        sweep_rows.append(
            {
                "selection": tag,
                "macro_miou": readout["macro_miou"],
                "macro_acc025": readout["macro_acc025"],
                "macro_acc050": readout["macro_acc050"],
                "macro_boundary_f": readout["macro_boundary_f"],
                "macro_trimap_iou": readout["macro_trimap_iou"],
            }
        )
        if best is None or float(readout["macro_miou"]) > float(best["macro_miou"]):
            best = readout

    if best is None:
        best = _readout("", "diagnostic_posthoc_best_fixed_threshold", [])
    return best, sweep_rows


def summarize_run(
    label: str,
    root: str | Path,
    *,
    protocol_tag: str,
    expected_scenes: tuple[str, ...] = DIRECT3D_SCENES,
) -> dict[str, Any]:
    root_path = Path(root)
    entries, missing_scenes, warnings = _scene_entries(root_path, expected_scenes)
    fixed, fixed_warnings = _fixed_tag_readout(
        entries,
        protocol_tag,
        selector_policy=f"fixed_global_threshold:{protocol_tag}",
    )
    best_fixed, fixed_sweep = _best_fixed_macro_readout(entries)
    scene_locked = _scene_locked_readout(entries)
    warnings.extend(fixed_warnings)
    return {
        "label": label,
        "source_root": str(root_path),
        "scene_count": len(entries),
        "missing_scenes": missing_scenes,
        "protocol": entries[0]["protocol"] if entries else {},
        "fixed_global_threshold": fixed,
        "best_fixed_macro_threshold": best_fixed,
        "scene_locked_best": scene_locked,
        "fixed_threshold_sweep": fixed_sweep,
        "warnings": warnings,
    }


def summarize_runs(
    runs: list[tuple[str, str | Path]],
    *,
    protocol_tag: str,
    expected_scenes: tuple[str, ...] = DIRECT3D_SCENES,
) -> dict[str, Any]:
    summaries = [
        summarize_run(label, root, protocol_tag=protocol_tag, expected_scenes=expected_scenes)
        for label, root in runs
    ]
    warnings = [
        "best_fixed_macro_threshold and scene_locked_best are diagnostic/post-hoc readouts; fixed_global_threshold is the strict global-threshold protocol.",
    ]
    for run in summaries:
        warnings.extend(run.get("warnings", []))
    return {
        "generated": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "protocol_tag": protocol_tag,
        "expected_scenes": list(expected_scenes),
        "runs": summaries,
        "warnings": warnings,
    }


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "0.0000"


def build_markdown(summary: dict[str, Any]) -> str:
    protocol_tag = summary["protocol_tag"]
    lines = [
        "# LERF SAM3 Box Global Threshold Sweep",
        "",
        "Fixed global threshold is the strict paper-facing readout. Best fixed macro threshold and scene-locked best are diagnostic/post-hoc upper-bound readouts unless selected on held-out validation scenes.",
        "",
        f"- Strict fixed global threshold: `{protocol_tag}`",
        "",
        "| Run | Scenes | Fixed global threshold | Fixed mIoU | Fixed Acc@0.25 | Fixed Acc@0.50 | Boundary-F | Trimap IoU | Best fixed tag | Best fixed mIoU | Scene-locked mIoU | Source root |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for run in summary.get("runs", []):
        fixed = run["fixed_global_threshold"]
        best_fixed = run["best_fixed_macro_threshold"]
        scene_locked = run["scene_locked_best"]
        lines.append(
            "| {label} | {scenes} | `{fixed_tag}` | {fixed_miou} | {fixed_acc025} | {fixed_acc050} | {fixed_boundary} | {fixed_trimap} | `{best_tag}` | {best_miou} | {scene_miou} | `{source}` |".format(
                label=run["label"],
                scenes=run["scene_count"],
                fixed_tag=fixed["selection"],
                fixed_miou=_fmt(fixed["macro_miou"]),
                fixed_acc025=_fmt(fixed["macro_acc025"]),
                fixed_acc050=_fmt(fixed["macro_acc050"]),
                fixed_boundary=_fmt(fixed["macro_boundary_f"]),
                fixed_trimap=_fmt(fixed["macro_trimap_iou"]),
                best_tag=best_fixed["selection"],
                best_miou=_fmt(best_fixed["macro_miou"]),
                scene_miou=_fmt(scene_locked["macro_miou"]),
                source=run["source_root"],
            )
        )

    for run in summary.get("runs", []):
        fixed = run["fixed_global_threshold"]
        best_fixed = run["best_fixed_macro_threshold"]
        scene_locked = run["scene_locked_best"]
        lines.extend(
            [
                "",
                f"## {run['label']}",
                "",
                f"- Source root: `{run['source_root']}`",
                f"- Missing scenes: `{', '.join(run['missing_scenes']) if run['missing_scenes'] else 'none'}`",
                f"- Fixed global threshold `{fixed['selection']}`: macro mIoU `{_fmt(fixed['macro_miou'])}`, Acc@0.50 `{_fmt(fixed['macro_acc050'])}`, Boundary-F `{_fmt(fixed['macro_boundary_f'])}`, Trimap IoU `{_fmt(fixed['macro_trimap_iou'])}`.",
                f"- Diagnostic best fixed macro threshold `{best_fixed['selection']}`: macro mIoU `{_fmt(best_fixed['macro_miou'])}`.",
                f"- Diagnostic scene-locked best: macro mIoU `{_fmt(scene_locked['macro_miou'])}`.",
                "",
                "### Fixed Global Threshold Scene Rows",
                "",
                "| Scene | Selection | mIoU | Acc@0.25 | Acc@0.50 | Boundary-F | Trimap IoU | N | Source JSON |",
                "|---|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in fixed.get("rows", []):
            lines.append(
                "| {scene} | `{selection}` | {miou} | {acc025} | {acc050} | {boundary} | {trimap} | {n} | `{source}` |".format(
                    scene=row["scene"],
                    selection=row["selection"],
                    miou=_fmt(row["miou"]),
                    acc025=_fmt(row["acc025"]),
                    acc050=_fmt(row["acc050"]),
                    boundary=_fmt(row["boundary_f"]),
                    trimap=_fmt(row["trimap_iou"]),
                    n=int(row["n"]),
                    source=row["source"],
                )
            )

    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in summary.get("warnings", []))
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    summary: dict[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> dict[str, Path]:
    json_out = Path(json_path)
    markdown_out = Path(markdown_path)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    markdown_out.write_text(build_markdown(summary), encoding="utf-8")
    return {"json": json_out, "markdown": markdown_out}


def _parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError("--run must be formatted as label=root_path")
    label, root = spec.split("=", 1)
    if not label.strip() or not root.strip():
        raise ValueError("--run must include both label and root_path")
    return label.strip(), Path(root.strip())


def main(argv: list[str] | None = None) -> dict[str, Path]:
    parser = argparse.ArgumentParser(description="Summarize direct-3D threshold sweeps")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Sweep run formatted as label=root_path. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--protocol_tag",
        default="thr0p25",
        help="Strict fixed global threshold tag to report as the paper-facing readout.",
    )
    parser.add_argument(
        "--output_json",
        default=str(DEFAULT_JSON),
        help="Output JSON manifest path.",
    )
    parser.add_argument(
        "--output_md",
        default=str(DEFAULT_MARKDOWN),
        help="Output markdown report path.",
    )
    args = parser.parse_args(argv)

    summary = summarize_runs(
        [_parse_run_spec(spec) for spec in args.run],
        protocol_tag=args.protocol_tag,
    )
    paths = write_outputs(summary, json_path=args.output_json, markdown_path=args.output_md)
    print(f"Wrote {paths['json']}")
    print(f"Wrote {paths['markdown']}")
    return paths


if __name__ == "__main__":
    main()
