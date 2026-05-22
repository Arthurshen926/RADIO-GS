#!/usr/bin/env python3
"""Validate paper-facing final_rows.yaml against source artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from radio_gs.scripts.build_scannet_vala8_report import SCAN_SPLITS, VALA8_SCENES


DIRECT_SCANNET_SOURCE_ARGS = {
    "scene_list": ",".join(VALA8_SCENES),
    "class_splits": "19,15,10",
    "query_mode": "gaussian_index",
    "candidate_k": "0",
    "opacity_filter_mode": "label_index",
    "logit_calibration": "none",
    "logit_calibration_alpha": "1.0",
    "gaussian_index_position_mode": "label_point",
    "prompt_templates": "{query}",
    "use_summary_head": "True",
    "use_point_summary_adapter": "False",
}

CONTEXTUAL_SCANNET_SOURCE_ARGS = {
    "scene_list": ",".join(VALA8_SCENES),
    "class_splits": "19,15,10",
    "query_mode": "knn",
    "k": "8",
    "candidate_k": "32",
    "opacity_filter_mode": "auto",
    "logit_calibration": "scene_mean",
    "logit_calibration_alpha": "0.5",
    "prompt_templates": "{query}",
    "use_summary_head": "True",
    "use_point_summary_adapter": "False",
}


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _rounded4(value: Any) -> float:
    return round(float(value), 4)


def _as_str(value: Any) -> str:
    return "None" if value is None else str(value)


def _check_vala8_source_protocol(
    source: dict[str, Any],
    *,
    label: str,
    expected_args: dict[str, str],
    issues: list[str],
) -> None:
    scene_count = source.get("scene_count")
    if scene_count != len(VALA8_SCENES):
        issues.append(f"{label} scene_count must be {len(VALA8_SCENES)}, got {scene_count!r}")

    scenes = source.get("scenes")
    if scenes != list(VALA8_SCENES):
        issues.append(f"{label} scene list drift: got {scenes!r}, expected {list(VALA8_SCENES)!r}")

    rows = source.get("rows", [])
    row_scenes = [row.get("scene") for row in rows]
    if row_scenes and row_scenes != list(VALA8_SCENES):
        issues.append(f"{label} per-scene row order drift: got {row_scenes!r}")

    if len(rows) == len(VALA8_SCENES):
        for split in SCAN_SPLITS:
            for metric in ("miou", "macc"):
                try:
                    recomputed = _rounded4(sum(float(row[split][metric]) for row in rows) / len(rows))
                    actual = _rounded4(source["macro"][split][metric])
                except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
                    issues.append(f"{label} cannot recompute macro split{split}.{metric}: {exc}")
                    continue
                if actual != recomputed:
                    issues.append(
                        f"{label} macro split{split}.{metric} drift: "
                        f"source={actual:.4f} recomputed={recomputed:.4f}"
                    )

    source_args = source.get("source_args", {})
    if not isinstance(source_args, dict):
        issues.append(f"{label} source_args must be a dict")
        return
    for key, expected in expected_args.items():
        actual = _as_str(source_args.get(key))
        if actual != expected:
            issues.append(f"{label} source_args.{key} mismatch: got {actual!r}, expected {expected!r}")


def _check_contextual_scannet_row(
    payload: dict[str, Any],
    root: Path,
    issues: list[str],
) -> None:
    try:
        track = payload["tracks"]["t3_scannet_ov_point_cloud_segmentation"]
        source_path = _resolve(root, track["radio_gs_contextual_support_json"])
        row = track["rows"]["radio_gs_v67_contextual_knn_scene_mean_support"]
    except KeyError as exc:
        issues.append(f"missing ScanNet contextual registry field: {exc}")
        return

    if row.get("promoted") is not False:
        issues.append("ScanNet contextual support row must remain promoted=false")

    try:
        source = _read_json(source_path)
        macro = source["macro"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        issues.append(f"cannot read ScanNet contextual source {source_path}: {exc}")
        return

    _check_vala8_source_protocol(
        source,
        label="ScanNet contextual",
        expected_args=CONTEXTUAL_SCANNET_SOURCE_ARGS,
        issues=issues,
    )

    for split in ("19", "15", "10"):
        row_key = f"split{split}"
        for metric in ("miou", "macc"):
            expected = _rounded4(macro[split][metric])
            actual = _rounded4(row[row_key][metric])
            if actual != expected:
                issues.append(
                    f"ScanNet contextual {row_key}.{metric} drift: "
                    f"registry={actual:.4f} source={expected:.4f}"
                )


def _check_direct_scannet_row(
    payload: dict[str, Any],
    root: Path,
    issues: list[str],
) -> None:
    try:
        track = payload["tracks"]["t3_scannet_ov_point_cloud_segmentation"]
        source_path = _resolve(root, track["radio_gs_source_json"])
        row = track["rows"]["radio_gs_v67_direct_point_query"]
    except KeyError as exc:
        issues.append(f"missing ScanNet direct registry field: {exc}")
        return

    try:
        source = _read_json(source_path)
        macro = source["macro"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        issues.append(f"cannot read ScanNet direct source {source_path}: {exc}")
        return

    _check_vala8_source_protocol(
        source,
        label="ScanNet direct",
        expected_args=DIRECT_SCANNET_SOURCE_ARGS,
        issues=issues,
    )

    for split in ("19", "15", "10"):
        row_key = f"split{split}"
        for metric in ("miou", "macc"):
            expected = _rounded4(macro[split][metric])
            actual = _rounded4(row[row_key][metric])
            if actual != expected:
                issues.append(
                    f"ScanNet direct {row_key}.{metric} drift: "
                    f"registry={actual:.4f} source={expected:.4f}"
                )


def _check_opengaff_blocker(
    payload: dict[str, Any],
    root: Path,
    issues: list[str],
) -> None:
    queue = payload.get("external_reproduction_queue", {})
    p2_rows = {row.get("method"): row for row in queue.get("p2", [])}
    opengaff_row = p2_rows.get("OpenGaFF")
    if opengaff_row is None:
        issues.append("OpenGaFF missing from external_reproduction_queue.p2")
    elif "no public implementation" not in str(opengaff_row.get("status", "")):
        issues.append("OpenGaFF p2 status must record no public implementation")

    audit_path = queue.get("machine_audit", {}).get("json")
    if not audit_path:
        issues.append("external_reproduction_queue.machine_audit.json missing")
        return
    try:
        audit = _read_json(_resolve(root, audit_path))
    except (OSError, json.JSONDecodeError) as exc:
        issues.append(f"cannot read external baseline audit {audit_path}: {exc}")
        return

    baselines = {row.get("method"): row for row in audit.get("baselines", [])}
    audit_row = baselines.get("OpenGaFF")
    if audit_row is None:
        issues.append("OpenGaFF missing from external baseline audit")
        return
    if audit_row.get("exists") is not False:
        issues.append("OpenGaFF audit row must have exists=false until code is public")
    if "code will be publicly released upon acceptance" not in str(audit_row.get("blocker", "")):
        issues.append("OpenGaFF audit blocker must cite release-upon-acceptance status")


def _queue_status(payload: dict[str, Any], method: str) -> str | None:
    queue = payload.get("external_reproduction_queue", {})
    for bucket in ("p0", "p1", "p2"):
        for row in queue.get(bucket, []):
            if row.get("method") == method:
                return str(row.get("status", ""))
    return None


def _check_status_contains(status: str | None, method: str, snippets: list[str], issues: list[str]) -> None:
    if status is None:
        issues.append(f"{method} missing from external_reproduction_queue")
        return
    for snippet in snippets:
        if snippet not in status:
            issues.append(f"{method} status missing synced summary snippet: {snippet}")


def _check_completed_external_summaries(
    payload: dict[str, Any],
    root: Path,
    issues: list[str],
) -> None:
    gags_path = root / "paper/artifacts/gags_lerf_summary.json"
    if gags_path.exists():
        try:
            gags = _read_json(gags_path)
            scene_mean = gags["scene_mean"]
            weighted = gags["object_weighted"]
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            issues.append(f"cannot read GAGS summary {gags_path}: {exc}")
        else:
            _check_status_contains(
                _queue_status(payload, "GAGS"),
                "GAGS",
                [
                    f"scene-mean LocAcc {_rounded4(scene_mean['locacc']):.4f} / mIoU {_rounded4(scene_mean['miou']):.4f}",
                    (
                        f"object-weighted LocAcc {_rounded4(weighted['locacc']):.4f} / "
                        f"mIoU {_rounded4(weighted['miou']):.4f} over {int(weighted['query_count'])} queries"
                    ),
                ],
                issues,
            )

    drsplat_path = root / "paper/artifacts/drsplat_lerf_summary.json"
    if drsplat_path.exists():
        try:
            drsplat = _read_json(drsplat_path)
            macro = drsplat["macro"]
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            issues.append(f"cannot read Dr. Splat summary {drsplat_path}: {exc}")
        else:
            _check_status_contains(
                _queue_status(payload, "Dr. Splat"),
                "Dr. Splat",
                [
                    (
                        f"mIoU {_rounded4(macro['miou']):.4f} / "
                        f"Acc@0.25 {_rounded4(macro['acc025']):.4f} / "
                        f"Acc@0.5 {_rounded4(macro['acc05']):.4f} over {int(macro['count'])} objects"
                    ),
                    f"missing rendered masks counted: {int(macro['missing'])}",
                ],
                issues,
            )

    legaussians_path = root / "paper/artifacts/legaussians_lerf_summary.json"
    if legaussians_path.exists():
        try:
            legaussians = _read_json(legaussians_path)
            scene_mean = legaussians["scene_mean"]
            weighted = legaussians["object_weighted"]
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            issues.append(f"cannot read LEGaussians summary {legaussians_path}: {exc}")
        else:
            _check_status_contains(
                _queue_status(payload, "LEGaussians"),
                "LEGaussians",
                [
                    (
                        f"scene-mean mIoU {_rounded4(scene_mean['miou']):.4f} / "
                        f"Acc@0.25 {_rounded4(scene_mean['acc025']):.4f} / "
                        f"Acc@0.5 {_rounded4(scene_mean['acc05']):.4f}"
                    ),
                    (
                        f"object-weighted mIoU {_rounded4(weighted['miou']):.4f} "
                        f"over {int(weighted['count'])} objects"
                    ),
                    f"missing rendered masks counted: {int(weighted['missing'])}",
                ],
                issues,
            )

    semantic_path = root / "output/baselines/semantic_gaussians/scannet_compat_20260520/semantic_gaussians_eval_metrics.json"
    if semantic_path.exists():
        try:
            semantic = _read_json(semantic_path)
            mean_iou = semantic["metrics"]["mean_iou"]
            scene_count = len(semantic["scenes"])
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            issues.append(f"cannot read Semantic Gaussians summary {semantic_path}: {exc}")
        else:
            _check_status_contains(
                _queue_status(payload, "Semantic Gaussians"),
                "Semantic Gaussians",
                [
                    f"ScanNet-20 label-PLY mean IoU {_rounded4(mean_iou):.4f} over {scene_count} scenes",
                ],
                issues,
            )

    laga_path = root / "paper/artifacts/laga_lerf_summary.json"
    if laga_path.exists():
        try:
            laga = _read_json(laga_path)
            macro = laga["macro"]
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            issues.append(f"cannot read LaGa summary {laga_path}: {exc}")
        else:
            _check_status_contains(
                _queue_status(payload, "LaGa"),
                "LaGa",
                [
                    (
                        f"mIoU {_rounded4(macro['miou']):.4f} / "
                        f"Acc@0.25 {_rounded4(macro['acc025']):.4f} / "
                        f"Acc@0.5 {_rounded4(macro['acc05']):.4f} over {int(macro['count'])} objects"
                    ),
                    f"missing rendered masks counted: {int(macro['missing'])}",
                ],
                issues,
            )


def validate_registry(final_rows_path: str | Path, *, root: str | Path = ".") -> list[str]:
    root_path = Path(root)
    payload = _read_yaml(Path(final_rows_path))
    issues: list[str] = []
    _check_direct_scannet_row(payload, root_path, issues)
    _check_contextual_scannet_row(payload, root_path, issues)
    _check_opengaff_blocker(payload, root_path, issues)
    _check_completed_external_summaries(payload, root_path, issues)
    return issues


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "final_rows",
        nargs="?",
        default="paper/artifacts/final_rows.yaml",
        help="Path to final_rows.yaml",
    )
    parser.add_argument("--root", default=".", help="Repository root for relative artifact paths")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    issues = validate_registry(args.final_rows, root=args.root)
    if issues:
        for issue in issues:
            print(f"ERROR: {issue}")
        raise SystemExit(1)
    print("final_rows registry ok")


if __name__ == "__main__":
    main()
