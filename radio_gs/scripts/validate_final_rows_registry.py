#!/usr/bin/env python3
"""Validate paper-facing final_rows.yaml against source artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from radio_gs.scripts.build_scannet_vala8_report import SCAN_SPLITS, VALA8_SCENES

SCANNET_TABLE = Path("paper/scannet_published_context_table.tex")
LERF_SCENES = ("figurines", "ramen", "teatime", "waldo_kitchen")

PROMOTED_SCANNET_SOURCE_ARGS = {
    "scene_list": ",".join(VALA8_SCENES),
    "class_splits": "19,15,10",
    "query_mode": "knn",
    "k": "16",
    "candidate_k": "80",
    "opacity_filter_mode": "auto",
    "logit_calibration": "scene_mean",
    "logit_calibration_alpha": "0.45",
    "logit_smoothing": "spatial_knn",
    "logit_smoothing_k": "12",
    "logit_smoothing_alpha": "1.0",
    "logit_smoothing_iterations": "1",
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


def _pct(value: Any) -> str:
    return f"{100.0 * float(value):.2f}"


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
    if not isinstance(rows, list):
        issues.append(f"{label} rows must be a list")
        rows = []
    if len(rows) != len(VALA8_SCENES):
        issues.append(f"{label} must contain exactly {len(VALA8_SCENES)} per-scene rows, got {len(rows)}")
    row_scenes = [row.get("scene") for row in rows]
    if row_scenes != list(VALA8_SCENES):
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


def _check_promoted_lerf_rows(
    payload: dict[str, Any],
    root: Path,
    issues: list[str],
) -> None:
    try:
        t1 = payload["tracks"]["t1_lerf_rendered_view_ovs"]
        t1_row = t1["rows"]["ctfgs_rendered"]
        t1_source_path = _resolve(root, t1["source_json"])
        t1_source = _read_json(t1_source_path)
    except (KeyError, OSError, json.JSONDecodeError) as exc:
        issues.append(f"cannot load promoted LERF 2D source: {exc}")
    else:
        if t1_row.get("promoted") is not True:
            issues.append("LERF 2D ctfgs_rendered row must be promoted=true")
        for registry_key, source_key in (("miou", "miou"), ("locacc", "loc_acc")):
            try:
                registry_value = _rounded4(t1_row[registry_key])
                source_value = _rounded4(t1_source["macro"][source_key])
            except (KeyError, TypeError, ValueError) as exc:
                issues.append(f"cannot compare LERF 2D macro {registry_key}: {exc}")
                continue
            if registry_value != source_value:
                issues.append(
                    f"LERF 2D promoted {registry_key} drift: "
                    f"registry={registry_value:.4f} source={source_value:.4f}"
                )
        registry_scenes = t1_row.get("per_scene", {})
        if set(registry_scenes) != set(LERF_SCENES):
            issues.append(f"LERF 2D registry per_scene must contain exactly {list(LERF_SCENES)!r}")
        for scene in LERF_SCENES:
            for registry_key, source_key in (("miou", "miou"), ("acc", "loc_acc")):
                try:
                    registry_value = _rounded4(registry_scenes[scene][registry_key])
                    source_value = _rounded4(t1_source["scenes"][scene][source_key])
                except (KeyError, TypeError, ValueError) as exc:
                    issues.append(f"cannot compare LERF 2D {scene}.{registry_key}: {exc}")
                    continue
                if registry_value != source_value:
                    issues.append(
                        f"LERF 2D promoted {scene}.{registry_key} drift: "
                        f"registry={registry_value:.4f} source={source_value:.4f}"
                    )

    try:
        t2 = payload["tracks"]["t2_lerf_direct_3d_selection"]
        t2_row = t2["rows"]["ctfgs_compact_prompt_ensemble_score_component_guard_thr0p55"]
        t2_source_path = _resolve(root, t2["source_json"])
        t2_source = _read_json(t2_source_path)
        t2_source_row = t2_source["rows"][t2_row["source_row"]]
    except (KeyError, OSError, json.JSONDecodeError) as exc:
        issues.append(f"cannot load promoted LERF Direct3D source: {exc}")
    else:
        if t2_row.get("promoted") is not True:
            issues.append("LERF Direct3D compact score-component row must be promoted=true")
        for metric in ("miou", "acc025"):
            try:
                registry_value = _rounded4(t2_row[metric])
                source_value = _rounded4(t2_source_row[metric])
            except (KeyError, TypeError, ValueError) as exc:
                issues.append(f"cannot compare LERF Direct3D macro {metric}: {exc}")
                continue
            if registry_value != source_value:
                issues.append(
                    f"LERF Direct3D promoted {metric} drift: "
                    f"registry={registry_value:.4f} source={source_value:.4f}"
                )
        registry_scenes = t2_row.get("per_scene", {})
        if set(registry_scenes) != set(LERF_SCENES):
            issues.append(f"LERF Direct3D registry per_scene must contain exactly {list(LERF_SCENES)!r}")
        for scene in LERF_SCENES:
            for metric in ("miou", "acc025"):
                try:
                    registry_value = _rounded4(registry_scenes[scene][metric])
                    source_value = _rounded4(t2_source_row["per_scene"][scene][metric])
                except (KeyError, TypeError, ValueError) as exc:
                    issues.append(f"cannot compare LERF Direct3D {scene}.{metric}: {exc}")
                    continue
                if registry_value != source_value:
                    issues.append(
                        f"LERF Direct3D promoted {scene}.{metric} drift: "
                        f"registry={registry_value:.4f} source={source_value:.4f}"
                    )


def _check_promoted_scannet_row(
    payload: dict[str, Any],
    root: Path,
    issues: list[str],
) -> None:
    try:
        track = payload["tracks"]["t3_scannet_ov_point_cloud_segmentation"]
        source_json = track["radio_gs_source_json"]
        row = track["rows"]["radio_gs_dino_cv_contextual_knn_scene_mean_support"]
    except KeyError as exc:
        issues.append(f"missing ScanNet promoted registry field: {exc}")
        return

    if row.get("promoted") is not True:
        issues.append("ScanNet promoted support row must be promoted=true")

    source_path = _resolve(root, source_json)
    try:
        source = _read_json(source_path)
    except (OSError, json.JSONDecodeError) as exc:
        issues.append(f"cannot read ScanNet promoted source {source_path}: {exc}")
    else:
        _check_vala8_source_protocol(
            source,
            label="ScanNet promoted",
            expected_args=PROMOTED_SCANNET_SOURCE_ARGS,
            issues=issues,
        )
        for split in ("19", "15", "10"):
            row_key = f"split{split}"
            for metric in ("miou", "macc"):
                try:
                    registry_value = _rounded4(row[row_key][metric])
                    source_value = _rounded4(source["macro"][split][metric])
                except (KeyError, TypeError, ValueError) as exc:
                    issues.append(f"cannot compare ScanNet promoted {row_key}.{metric}: {exc}")
                    continue
                if registry_value != source_value:
                    issues.append(
                        f"ScanNet promoted {row_key}.{metric} drift: "
                        f"registry={registry_value:.4f} source={source_value:.4f}"
                    )

        registry_scenes = row.get("per_scene", {})
        if set(registry_scenes) != set(VALA8_SCENES):
            issues.append(
                f"ScanNet promoted registry per_scene must contain exactly {list(VALA8_SCENES)!r}"
            )
        source_rows = {source_row.get("scene"): source_row for source_row in source.get("rows", [])}
        for scene in VALA8_SCENES:
            for split in ("19", "15", "10"):
                row_key = f"split{split}"
                for metric in ("miou", "macc"):
                    try:
                        registry_value = _rounded4(registry_scenes[scene][row_key][metric])
                        source_value = _rounded4(source_rows[scene][split][metric])
                    except (KeyError, TypeError, ValueError) as exc:
                        issues.append(f"cannot compare ScanNet {scene}.{row_key}.{metric}: {exc}")
                        continue
                    if registry_value != source_value:
                        issues.append(
                            f"ScanNet promoted {scene}.{row_key}.{metric} drift: "
                            f"registry={registry_value:.4f} source={source_value:.4f}"
                        )

        summary_json = track.get("paper_summary_json")
        if summary_json:
            summary_path = _resolve(root, summary_json)
            try:
                summary = _read_json(summary_path)
            except (OSError, json.JSONDecodeError) as exc:
                issues.append(f"cannot read ScanNet paper summary {summary_path}: {exc}")
            else:
                _check_vala8_source_protocol(
                    summary,
                    label="ScanNet paper summary",
                    expected_args=PROMOTED_SCANNET_SOURCE_ARGS,
                    issues=issues,
                )
                summary_rows = {item.get("scene"): item for item in summary.get("rows", [])}
                for split in ("19", "15", "10"):
                    for metric in ("miou", "macc"):
                        try:
                            summary_value = _rounded4(summary["macro"][split][metric])
                            source_value = _rounded4(source["macro"][split][metric])
                        except (KeyError, TypeError, ValueError) as exc:
                            issues.append(f"cannot compare ScanNet paper summary split{split}.{metric}: {exc}")
                            continue
                        if summary_value != source_value:
                            issues.append(
                                f"ScanNet paper summary split{split}.{metric} drift: "
                                f"summary={summary_value:.4f} source={source_value:.4f}"
                            )
                for scene in VALA8_SCENES:
                    for split in ("19", "15", "10"):
                        for metric in ("miou", "macc"):
                            try:
                                summary_value = _rounded4(summary_rows[scene][split][metric])
                                source_value = _rounded4(source_rows[scene][split][metric])
                            except (KeyError, TypeError, ValueError) as exc:
                                issues.append(
                                    f"cannot compare ScanNet paper summary {scene}.split{split}.{metric}: {exc}"
                                )
                                continue
                            if summary_value != source_value:
                                issues.append(
                                    f"ScanNet paper summary {scene}.split{split}.{metric} drift: "
                                    f"summary={summary_value:.4f} source={source_value:.4f}"
                                )

    for name, context_row in track.get("rows", {}).items():
        if name.endswith("_published_context") and context_row.get("reproduced_local") is not False:
            issues.append(f"ScanNet published context row {name} must have reproduced_local=false")

    table_path = root / SCANNET_TABLE
    try:
        table_text = table_path.read_text(encoding="utf-8")
    except OSError as exc:
        issues.append(f"cannot read ScanNet paper table {table_path}: {exc}")
        return

    expected_cells: list[str] = []
    for split in ("19", "15", "10"):
        row_key = f"split{split}"
        for metric in ("miou", "macc"):
            try:
                expected_cells.append(_pct(row[row_key][metric]))
            except (KeyError, TypeError, ValueError) as exc:
                issues.append(f"cannot format ScanNet promoted {row_key}.{metric}: {exc}")
                return

    expected_snippet = "\\method{} & " + " & ".join(expected_cells)
    if expected_snippet not in table_text:
        issues.append(
            "ScanNet promoted registry/table drift: missing table row snippet "
            f"{expected_snippet}"
        )


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
    _check_promoted_lerf_rows(payload, root_path, issues)
    _check_promoted_scannet_row(payload, root_path, issues)
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
