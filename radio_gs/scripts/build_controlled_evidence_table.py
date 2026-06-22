"""Build a controlled evidence table from existing RADIO-GS paper artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "output" / "radio_gs" / "reports"
DEFAULT_COMPONENT_JSON = REPORT_DIR / "lerf_component_ablation.json"
DEFAULT_FREEZE_MANIFEST = REPORT_DIR / "submission_freeze_manifest.json"
DEFAULT_STORAGE_REPORT = REPORT_DIR / "storage_footprint_report.md"
DEFAULT_PROFILE_REPORT = REPORT_DIR / "submission_freeze_profile_summary.md"
DEFAULT_NEAREST_VIEW = REPORT_DIR / "lerf_nearest_view_cache_baseline.json"
DEFAULT_PER_GAUSSIAN_1280D = REPORT_DIR / "lerf_per_gaussian_1280d_baseline.json"
DEFAULT_MARKDOWN = REPORT_DIR / "controlled_evidence_table.md"
DEFAULT_JSON = REPORT_DIR / "controlled_evidence_table.json"


def _round4(value: float) -> float:
    return round(float(value), 4)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def teacher_row(teacher_loc_acc: float, teacher_miou: float) -> dict[str, Any]:
    return {
        "method": "Frame-wise RADIO",
        "compact": "no",
        "3d_memory": "no",
        "novel_view_feature": "no",
        "direct_3d_query": "no",
        "lerf_loc_acc": _round4(teacher_loc_acc),
        "lerf_miou": _round4(teacher_miou),
        "direct3d": "not applicable",
        "storage": "per-frame feature cache",
        "runtime": "not profiled here",
        "source": "paper/radio_gs_draft.tex frame-wise-RADIO-vs-rendered table",
    }


def nearest_view_row(path: str | Path) -> dict[str, Any] | None:
    report_path = Path(path)
    if not report_path.exists():
        return None
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    macro = payload.get("macro", {})
    distance = payload.get("mean_nearest_distance")
    runtime_note = (
        f"mean nearest distance {float(distance):.4f}"
        if distance is not None
        else "see nearest-view report"
    )
    return {
        "method": "Nearest-view RADIO cache",
        "compact": "no",
        "3d_memory": "no",
        "novel_view_feature": "cache-only",
        "direct_3d_query": "no",
        "lerf_loc_acc": _round4(float(macro.get("loc_acc", 0.0))),
        "lerf_miou": _round4(float(macro.get("miou", 0.0))),
        "direct3d": "not applicable",
        "storage": "per-frame feature cache",
        "runtime": runtime_note,
        "source": "lerf_nearest_view_cache_baseline.json",
        "note": "unwarped nearest cached teacher frame; target frame excluded",
    }


def per_gaussian_1280d_row(path: str | Path) -> dict[str, Any] | None:
    report_path = Path(path)
    if not report_path.exists():
        return None
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    macro = payload.get("macro", {})
    registered_fraction = payload.get("mean_registered_fraction")
    storage_mib = payload.get("mean_storage_mib")
    storage_note = (
        f"{float(storage_mib):.1f} MiB mean fp16 feature storage"
        if storage_mib is not None
        else "see per-Gaussian 1280-D report"
    )
    runtime_note = (
        f"registered fraction {float(registered_fraction):.4f}"
        if registered_fraction is not None
        else "see per-Gaussian 1280-D report"
    )
    return {
        "method": "Per-Gaussian 1280-D RADIO memory",
        "compact": "no",
        "3d_memory": "yes",
        "novel_view_feature": "yes",
        "direct_3d_query": "partial",
        "lerf_loc_acc": _round4(float(macro.get("loc_acc", 0.0))),
        "lerf_miou": _round4(float(macro.get("miou", 0.0))),
        "direct3d": "not evaluated",
        "storage": storage_note,
        "runtime": runtime_note,
        "source": "lerf_per_gaussian_1280d_baseline.json",
        "note": "registered fp16 frame-wise RADIO features attached to Gaussian primitives",
    }


def _macro_for_variant(component_payload: dict[str, Any], key: str) -> tuple[float, float]:
    rows = component_payload.get("results", {}).get(key, {})
    locs = [float(row.get("loc_acc", 0.0)) for row in rows.values()]
    mious = [float(row.get("miou", 0.0)) for row in rows.values()]
    return _round4(_mean(locs)), _round4(_mean(mious))


def _direct3d_summary(freeze_manifest: dict[str, Any]) -> str:
    strict = {
        item.get("label", ""): item
        for item in freeze_manifest.get("direct3d_readouts", [])
        if str(item.get("selector_policy", "")).startswith("fixed")
    }
    vpr = strict.get("VPR fixed threshold + RGB snap")
    sam3 = strict.get("direct field + official SAM3 box, pad16 fixed global threshold")
    parts = []
    if vpr:
        parts.append(f"VPR {float(vpr['macro_miou']):.4f}/{float(vpr['macro_acc025']):.4f}")
    if sam3:
        parts.append(f"SAM3-box {float(sam3['macro_miou']):.4f}/{float(sam3['macro_acc025']):.4f}")
    return "; ".join(parts) if parts else "not evaluated"


def _parse_mean_storage_saving(markdown_path: str | Path) -> dict[str, Any]:
    path = Path(markdown_path)
    if not path.exists():
        return {"mean_saving": None}
    savings: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or " MiB " not in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 5:
            continue
        match = re.search(r"([0-9.]+)x", cells[4])
        if match:
            savings.append(float(match.group(1)))
    return {"mean_saving": _mean(savings) if savings else None}


def _parse_mean_lerf_runtime(markdown_path: str | Path) -> dict[str, Any]:
    path = Path(markdown_path)
    if not path.exists():
        return {"mean_lerf_wall_seconds": None}
    times: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "freeze_lerf_" not in line:
            continue
        cells = [cell.strip(" `") for cell in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        match = re.search(r"([0-9.]+)\s*s", cells[1])
        if match:
            times.append(float(match.group(1)))
    return {"mean_lerf_wall_seconds": _mean(times) if times else None}


def build_rows(
    *,
    component_path: str | Path,
    freeze_manifest: dict[str, Any],
    storage_summary: dict[str, Any],
    profile_summary: dict[str, Any],
    nearest_view_path: str | Path | None = DEFAULT_NEAREST_VIEW,
    per_gaussian_1280d_path: str | Path | None = DEFAULT_PER_GAUSSIAN_1280D,
) -> list[dict[str, Any]]:
    component_payload = json.loads(Path(component_path).read_text(encoding="utf-8"))
    rows = [teacher_row(0.7985, 0.4634)]
    if nearest_view_path is not None:
        nearest = nearest_view_row(nearest_view_path)
        if nearest is not None:
            rows.append(nearest)
    if per_gaussian_1280d_path is not None:
        explicit = per_gaussian_1280d_row(per_gaussian_1280d_path)
        if explicit is not None:
            rows.append(explicit)

    direct3d = _direct3d_summary(freeze_manifest)
    mean_saving = storage_summary.get("mean_saving")
    runtime = profile_summary.get("mean_lerf_wall_seconds")
    full_storage = (
        f"{float(mean_saving):.2f}x mean compact checkpoint saving"
        if mean_saving is not None
        else "see storage report"
    )
    full_runtime = (
        f"{float(runtime):.1f}s mean LERF overlay"
        if runtime is not None
        else "see profile report"
    )

    for variant in component_payload.get("variants", []):
        key = variant["key"]
        loc_acc, miou = _macro_for_variant(component_payload, key)
        is_full = key == "full"
        rows.append(
            {
                "method": str(variant["label"]).replace("RADIO-GS", "GaussFM"),
                "compact": "yes",
                "3d_memory": "yes",
                "novel_view_feature": "yes",
                "direct_3d_query": "yes" if is_full else "not evaluated",
                "lerf_loc_acc": float(freeze_manifest["lerf"]["macro_loc_acc"]) if is_full else loc_acc,
                "lerf_miou": float(freeze_manifest["lerf"]["macro_miou"]) if is_full else miou,
                "direct3d": direct3d if is_full else "not evaluated",
                "storage": full_storage if is_full else "not separately measured",
                "runtime": full_runtime if is_full else "not separately profiled",
                "source": "submission freeze manifest" if is_full else "lerf_component_ablation.json",
                "note": variant.get("note", ""),
            }
        )
    return rows


def build_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Controlled Evidence Table",
        "",
        "This table consolidates existing frozen-protocol evidence without inventing unmeasured rows. `not evaluated` means that the artifact set does not contain that measurement for the variant.",
        "",
        "| Method | Compact | 3D memory | Novel-view feature | Direct 3D query | LERF LocAcc | LERF mIoU | Direct 3D mIoU/Acc@0.25 | Storage | Runtime | Source |",
        "|---|---|---|---|---|---:|---:|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {compact} | {memory} | {novel} | {direct} | {loc:.4f} | {miou:.4f} | {direct3d} | {storage} | {runtime} | {source} |".format(
                method=row["method"],
                compact=row["compact"],
                memory=row["3d_memory"],
                novel=row["novel_view_feature"],
                direct=row["direct_3d_query"],
                loc=float(row["lerf_loc_acc"]),
                miou=float(row["lerf_miou"]),
                direct3d=row["direct3d"],
                storage=row["storage"],
                runtime=row["runtime"],
                source=row["source"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    rows: list[dict[str, Any]],
    markdown_path: str | Path,
    json_path: str | Path,
) -> dict[str, Path]:
    markdown_out = Path(markdown_path)
    json_out = Path(json_path)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.write_text(build_markdown(rows), encoding="utf-8")
    json_out.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    return {"markdown": markdown_out, "json": json_out}


def main(argv: list[str] | None = None) -> dict[str, Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component_json", default=str(DEFAULT_COMPONENT_JSON))
    parser.add_argument("--freeze_manifest", default=str(DEFAULT_FREEZE_MANIFEST))
    parser.add_argument("--storage_report", default=str(DEFAULT_STORAGE_REPORT))
    parser.add_argument("--profile_report", default=str(DEFAULT_PROFILE_REPORT))
    parser.add_argument("--nearest_view_report", default=str(DEFAULT_NEAREST_VIEW))
    parser.add_argument("--per_gaussian_1280d_report", default=str(DEFAULT_PER_GAUSSIAN_1280D))
    parser.add_argument("--output_md", default=str(DEFAULT_MARKDOWN))
    parser.add_argument("--output_json", default=str(DEFAULT_JSON))
    args = parser.parse_args(argv)

    freeze_manifest = json.loads(Path(args.freeze_manifest).read_text(encoding="utf-8"))
    rows = build_rows(
        component_path=args.component_json,
        freeze_manifest=freeze_manifest,
        storage_summary=_parse_mean_storage_saving(args.storage_report),
        profile_summary=_parse_mean_lerf_runtime(args.profile_report),
        nearest_view_path=args.nearest_view_report,
        per_gaussian_1280d_path=args.per_gaussian_1280d_report,
    )
    paths = write_outputs(rows, args.output_md, args.output_json)
    print(f"Wrote {paths['markdown']}")
    print(f"Wrote {paths['json']}")
    return paths


if __name__ == "__main__":
    main()
