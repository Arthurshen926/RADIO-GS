#!/usr/bin/env python3
"""Fail-closed aggregation for the four-scene Dr. Splat LERF L3 audit.

The input reports are produced one scene at a time by
``eval_drsplat_lerf_masks.py``.  This closeout validates their object-level
contents and companion model artifacts before reporting both aggregation
domains:

* the unweighted four-scene macro used for the paper-context comparison; and
* the 208-query micro, retained as a separate local diagnostic.

This tool deliberately labels the result as a scale-paired compatibility
reproduction.  It cannot turn local Occam-start/PQ checkpoints and the VALA
mask adapter into an official Dr. Splat checkpoint reproduction.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Mapping, Optional, Sequence


SCENES = ("figurines", "teatime", "ramen", "waldo_kitchen")

EXPECTED_FRAME_COUNTS: dict[str, dict[str, int]] = {
    "figurines": {
        "frame_00041": 17,
        "frame_00105": 13,
        "frame_00152": 11,
        "frame_00195": 15,
    },
    "teatime": {
        "frame_00002": 11,
        "frame_00025": 11,
        "frame_00043": 10,
        "frame_00107": 11,
        "frame_00129": 7,
        "frame_00140": 9,
    },
    "ramen": {
        "frame_00006": 10,
        "frame_00024": 8,
        "frame_00060": 8,
        "frame_00065": 11,
        "frame_00081": 12,
        "frame_00119": 8,
        "frame_00128": 14,
    },
    "waldo_kitchen": {
        "frame_00053": 6,
        "frame_00066": 4,
        "frame_00089": 3,
        "frame_00140": 5,
        "frame_00154": 4,
    },
}

EXPECTED_RENDER_COUNTS = {
    "figurines": 84,
    "teatime": 84,
    "ramen": 98,
    "waldo_kitchen": 90,
}

# These are the OpenGaFF-reported Dr. Splat context values used by the local
# paper table.  They are not claimed to come from an official released
# Dr. Splat evaluation package.
PAPER_SCENE_PERCENT = {
    "figurines": {"miou": 54.42, "acc025": 80.36},
    "teatime": {"miou": 57.35, "acc025": 77.97},
    "ramen": {"miou": 24.33, "acc025": 35.21},
    "waldo_kitchen": {"miou": 37.05, "acc025": 63.64},
}
PAPER_MACRO_PERCENT = {"miou": 43.29, "acc025": 64.30}

EXPECTED_PROTOCOL = "Dr. Splat/VALA LERF nested mask IoU"
EXPECTED_TEATIME_SHA256 = (
    "b999989e8536e26171fa292d910ba4e4004867e811a7e82ace88bb5c98651bac"
)
EXPECTED_PQ_SHA256 = (
    "40eff0447ef57655698667e05e59c2e07bb3603e0f1197a9449b5f07d3cf1de3"
)


class AuditError(ValueError):
    """Raised when an input violates the frozen closeout contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_finite_fraction(value: object, label: str) -> float:
    _require(not isinstance(value, bool), f"{label}: boolean is not a metric")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise AuditError(f"{label}: expected a finite number") from error
    _require(math.isfinite(result), f"{label}: metric is not finite")
    _require(0.0 <= result <= 1.0, f"{label}: metric is outside [0, 1]")
    return result


def _require_close(actual: object, expected: float, label: str) -> None:
    actual_float = _as_finite_fraction(actual, label)
    _require(
        math.isclose(actual_float, expected, rel_tol=0.0, abs_tol=1e-12),
        f"{label}: stored={actual_float!r}, recomputed={expected!r}",
    )


def _safe_parse_namespace(path: Path) -> dict[str, object]:
    """Parse a saved ``Namespace(...)`` without executing the file."""

    raw = path.read_text(encoding="utf-8").strip()
    try:
        body = ast.parse(raw, mode="eval").body
    except SyntaxError as error:
        raise AuditError(f"{path}: invalid cfg_args syntax") from error
    _require(isinstance(body, ast.Call), f"{path}: cfg_args is not a call")
    _require(
        isinstance(body.func, ast.Name) and body.func.id == "Namespace",
        f"{path}: cfg_args must be Namespace(...) ",
    )
    _require(not body.args, f"{path}: positional Namespace arguments are forbidden")
    values: dict[str, object] = {}
    for keyword in body.keywords:
        _require(keyword.arg is not None, f"{path}: **kwargs are forbidden")
        _require(keyword.arg not in values, f"{path}: duplicate {keyword.arg}")
        try:
            values[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, SyntaxError) as error:
            raise AuditError(
                f"{path}: non-literal value for {keyword.arg}"
            ) from error
    return values


def _metrics_from_ious(ious: Sequence[float]) -> dict[str, object]:
    _require(bool(ious), "cannot aggregate an empty IoU list")
    count = len(ious)
    acc025_hits = sum(value > 0.25 for value in ious)
    acc05_hits = sum(value > 0.5 for value in ious)
    return {
        "miou": sum(ious) / count,
        "acc025": acc025_hits / count,
        "acc05": acc05_hits / count,
        "count": count,
        "acc025_hits": acc025_hits,
        "acc05_hits": acc05_hits,
    }


def _validate_cfg(scene: str, model_root: Path) -> dict[str, object]:
    cfg_path = model_root / "cfg_args"
    checkpoint_path = model_root / "chkpnt0.pth"
    _require(cfg_path.is_file(), f"{scene}: missing {cfg_path}")
    _require(checkpoint_path.is_file(), f"{scene}: missing {checkpoint_path}")
    _require(checkpoint_path.stat().st_size > 0, f"{scene}: empty checkpoint")

    cfg = _safe_parse_namespace(cfg_path)
    expected = {
        "sh_degree": 3,
        "images": "images",
        "resolution": -1,
        "data_device": "cuda",
        "feature_level": 3,
        "name_extra": "l3paired",
        "mode": "mean",
        "topk": 45,
        "use_pq": True,
        "eval": True,
        "iterations": 0,
        "test_iterations": [0],
        "save_iterations": [0, 0],
        "checkpoint_iterations": [0],
        "language_features_name": "language_features_dim3",
    }
    for key, value in expected.items():
        _require(cfg.get(key) == value, f"{scene}: cfg_args {key} != {value!r}")
    _require(
        cfg.get("model_path") == str(model_root),
        f"{scene}: cfg_args model_path does not bind the rendered model",
    )

    source_path = Path(str(cfg.get("source_path", "")))
    start_checkpoint = Path(str(cfg.get("start_checkpoint", "")))
    pq_index = Path(str(cfg.get("pq_index", "")))
    _require(source_path.name == scene, f"{scene}: source_path scene mismatch")
    _require(source_path.is_dir(), f"{scene}: source_path is unavailable")
    _require(
        start_checkpoint.name == "chkpnt30000.pth"
        and start_checkpoint.parent.name == scene,
        f"{scene}: unexpected RGB start checkpoint",
    )
    _require(start_checkpoint.is_file(), f"{scene}: RGB start checkpoint unavailable")
    _require(pq_index.is_file(), f"{scene}: PQ index unavailable")
    pq_sha256 = _sha256_path(pq_index)
    _require(
        pq_sha256 == EXPECTED_PQ_SHA256,
        f"{scene}: PQ index SHA-256 changed: {pq_sha256}",
    )

    return {
        "cfg_args_path": str(cfg_path),
        "cfg_args_sha256": _sha256_path(cfg_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "source_path": str(source_path),
        "start_checkpoint": str(start_checkpoint),
        "pq_index": str(pq_index),
        "pq_index_sha256": pq_sha256,
    }


def validate_scene_report(
    scene: str,
    report_path: Path,
    *,
    expected_sha256: Optional[str] = None,
) -> dict[str, object]:
    """Load one report and validate metrics, denominator, paths, and L3 config."""

    _require(scene in SCENES, f"unknown scene: {scene}")
    _require(report_path.is_file(), f"{scene}: missing report {report_path}")
    raw = report_path.read_bytes()
    report_sha256 = _sha256_bytes(raw)
    if expected_sha256 is not None:
        _require(
            report_sha256 == expected_sha256,
            f"{scene}: report SHA-256 changed: {report_sha256}",
        )
    try:
        report = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuditError(f"{scene}: invalid report JSON") from error
    _require(isinstance(report, Mapping), f"{scene}: report root is not an object")

    frozen_top_level = {
        "protocol": EXPECTED_PROTOCOL,
        "mask_thresh": "0.4",
        "threshold": 10,
        "ablation_type": "none",
        "prediction_dir": "renders_silhouette",
    }
    for key, expected in frozen_top_level.items():
        _require(
            report.get(key) == expected,
            f"{scene}: report {key} != {expected!r}",
        )

    scenes = report.get("scenes")
    _require(isinstance(scenes, Mapping), f"{scene}: missing scenes mapping")
    _require(set(scenes) == {scene}, f"{scene}: report contains the wrong scenes")
    row = scenes[scene]
    _require(isinstance(row, Mapping), f"{scene}: scene row is not an object")

    expected_frame_counts = EXPECTED_FRAME_COUNTS[scene]
    expected_count = sum(expected_frame_counts.values())
    _require(row.get("count") == expected_count, f"{scene}: wrong object denominator")
    _require(row.get("missing") == 0, f"{scene}: missing predictions are forbidden")
    objects = row.get("objects")
    _require(isinstance(objects, list), f"{scene}: objects must be a list")
    _require(len(objects) == expected_count, f"{scene}: object list length mismatch")

    ious: list[float] = []
    frame_counts = {frame: 0 for frame in expected_frame_counts}
    object_keys: set[tuple[str, str]] = set()
    model_roots: set[Path] = set()
    for index, obj in enumerate(objects):
        label = f"{scene}.objects[{index}]"
        _require(isinstance(obj, Mapping), f"{label}: not an object")
        frame = obj.get("frame")
        query = obj.get("query")
        _require(isinstance(frame, str), f"{label}: invalid frame")
        _require(frame in frame_counts, f"{label}: unexpected frame {frame!r}")
        _require(isinstance(query, str) and query, f"{label}: invalid query")
        key = (frame, query)
        _require(key not in object_keys, f"{label}: duplicate frame/query {key!r}")
        object_keys.add(key)
        frame_counts[frame] += 1
        _require(obj.get("missing") is False, f"{label}: prediction marked missing")
        ious.append(_as_finite_fraction(obj.get("iou"), f"{label}.iou"))

        pred_path = Path(str(obj.get("pred_path", "")))
        gt_path = Path(str(obj.get("gt_path", "")))
        _require(pred_path.is_file(), f"{label}: prediction path is unavailable")
        _require(gt_path.is_file(), f"{label}: GT path is unavailable")
        _require(pred_path.parent.name == frame, f"{label}: prediction frame mismatch")
        _require(pred_path.name == f"{query}.png", f"{label}: prediction query mismatch")
        _require(
            pred_path.parent.parent.name == "renders_silhouette"
            and pred_path.parent.parent.parent.name == "predictions_mask_0.4",
            f"{label}: prediction is outside the frozen mask layout",
        )
        _require(
            gt_path.name == f"{frame}.json" and gt_path.parent.name == scene,
            f"{label}: GT scene/frame mismatch",
        )
        model_roots.add(pred_path.parents[3])

    _require(frame_counts == expected_frame_counts, f"{scene}: per-frame counts changed")
    _require(len(model_roots) == 1, f"{scene}: predictions span multiple models")
    model_root = next(iter(model_roots))
    expected_model_name = f"{scene}_3_l3paired_topk45_weight_128"
    _require(
        model_root.name == expected_model_name,
        f"{scene}: expected L3 paired model {expected_model_name}",
    )

    recomputed = _metrics_from_ious(ious)
    for metric in ("miou", "acc025", "acc05"):
        _require_close(row.get(metric), float(recomputed[metric]), f"{scene}.{metric}")

    macro = report.get("macro")
    _require(isinstance(macro, Mapping), f"{scene}: missing report macro")
    _require(macro.get("count") == expected_count, f"{scene}: macro count mismatch")
    _require(macro.get("missing") == 0, f"{scene}: macro missing must be zero")
    for metric in ("miou", "acc025", "acc05"):
        _require_close(
            macro.get(metric), float(recomputed[metric]), f"{scene}.macro.{metric}"
        )

    pred_root = model_root / "predictions_mask_0.4" / "renders_silhouette"
    _require(
        row.get("pred_root") == str(pred_root),
        f"{scene}: scene pred_root does not bind the validated predictions",
    )
    rendered_pngs = sorted(pred_root.rglob("*.png"))
    _require(
        len(rendered_pngs) == EXPECTED_RENDER_COUNTS[scene],
        f"{scene}: rendered silhouette count is {len(rendered_pngs)}, expected "
        f"{EXPECTED_RENDER_COUNTS[scene]}",
    )
    model_artifacts = _validate_cfg(scene, model_root)

    return {
        "scene": scene,
        "report_path": str(report_path),
        "report_sha256": report_sha256,
        "model_root": str(model_root),
        "rendered_silhouette_pngs": len(rendered_pngs),
        "metrics": recomputed,
        "ious": ious,
        "artifacts": model_artifacts,
    }


def aggregate_reports(
    report_paths: Mapping[str, Path],
    *,
    required_hashes: Optional[Mapping[str, str]] = None,
) -> dict[str, object]:
    """Validate and aggregate exactly the frozen four-scene cohort."""

    _require(set(report_paths) == set(SCENES), "reports must cover exactly four scenes")
    required_hashes = required_hashes or {}
    validated = {
        scene: validate_scene_report(
            scene,
            report_paths[scene],
            expected_sha256=required_hashes.get(scene),
        )
        for scene in SCENES
    }

    scene_rows: dict[str, object] = {}
    for scene in SCENES:
        metrics = validated[scene]["metrics"]
        assert isinstance(metrics, Mapping)
        paper = PAPER_SCENE_PERCENT[scene]
        scene_rows[scene] = {
            "objects": int(metrics["count"]),
            "metrics_fraction": {
                key: float(metrics[key]) for key in ("miou", "acc025", "acc05")
            },
            "metrics_percent": {
                key: 100.0 * float(metrics[key])
                for key in ("miou", "acc025", "acc05")
            },
            "acc025_hits": int(metrics["acc025_hits"]),
            "acc05_hits": int(metrics["acc05_hits"]),
            "paper_context_percent": dict(paper),
            "delta_to_paper_points": {
                "miou": 100.0 * float(metrics["miou"]) - paper["miou"],
                "acc025": 100.0 * float(metrics["acc025"]) - paper["acc025"],
            },
        }

    scene_macro_fraction = {
        metric: sum(
            float(validated[scene]["metrics"][metric])  # type: ignore[index]
            for scene in SCENES
        )
        / len(SCENES)
        for metric in ("miou", "acc025", "acc05")
    }
    all_ious = [
        float(iou)
        for scene in SCENES
        for iou in validated[scene]["ious"]  # type: ignore[union-attr]
    ]
    query_micro = _metrics_from_ious(all_ious)
    _require(query_micro["count"] == 208, "four-scene query total is not 208")

    scene_macro_percent = {
        key: 100.0 * value for key, value in scene_macro_fraction.items()
    }
    query_micro_percent = {
        key: 100.0 * float(query_micro[key])
        for key in ("miou", "acc025", "acc05")
    }

    return {
        "schema_version": 1,
        "audit_id": "drsplat_lerf_l3_scale_paired_compatibility",
        "evidence_class": "scale_paired_compatibility_reproduction",
        "strict_checkpoint_reproduction": False,
        "paper_comparison": "diagnostic_only",
        "comparability_notes": [
            "Fixed feature_level=3 is paired across all four scenes.",
            "Geometry starts are local OccamLGS-compatible RGB checkpoints, "
            "not released Dr. Splat pretrained checkpoints.",
            "Masks use the local VALA single-checkpoint adapter and threshold "
            "0.4, not an official released Dr. Splat evaluator.",
            "OpenGaFF-reported Dr. Splat values are paper context; only the "
            "scene-equal macro is used for their delta.",
        ],
        "protocol": {
            "benchmark": "LERF-OVS direct 3D open-vocabulary object selection",
            "feature_level": 3,
            "name_extra": "l3paired",
            "topk": 45,
            "use_pq": True,
            "mask_threshold": 0.4,
            "png_binary_threshold": 10,
            "acc025_comparison": "strict IoU > 0.25",
            "acc05_comparison": "strict IoU > 0.5",
            "paper_delta_aggregation": "unweighted_scene_equal_macro",
            "secondary_aggregation": "208_query_micro",
        },
        "inputs": {
            scene: {
                key: value
                for key, value in validated[scene].items()
                if key not in {"scene", "metrics", "ious"}
            }
            for scene in SCENES
        },
        "scenes": scene_rows,
        "scene_equal_macro": {
            "scenes": 4,
            "metrics_fraction": scene_macro_fraction,
            "metrics_percent": scene_macro_percent,
            "paper_reported_percent": dict(PAPER_MACRO_PERCENT),
            "delta_to_paper_points": {
                "miou": scene_macro_percent["miou"] - PAPER_MACRO_PERCENT["miou"],
                "acc025": scene_macro_percent["acc025"]
                - PAPER_MACRO_PERCENT["acc025"],
            },
        },
        "query_micro": {
            "objects": int(query_micro["count"]),
            "metrics_fraction": {
                key: float(query_micro[key]) for key in ("miou", "acc025", "acc05")
            },
            "metrics_percent": query_micro_percent,
            "acc025_hits": int(query_micro["acc025_hits"]),
            "acc05_hits": int(query_micro["acc05_hits"]),
            "paper_value_available": False,
            "note": (
                "The paper headline is scene-equal; this 208-query micro is a "
                "separate local diagnostic."
            ),
        },
        "paper_context": {
            "source": "OpenGaFF-reported Dr. Splat row used by the local paper table",
            "scene_percent": PAPER_SCENE_PERCENT,
            "scene_equal_macro_reported_percent": PAPER_MACRO_PERCENT,
            "scene_equal_macro_recomputed_from_rounded_scene_rows_percent": {
                "miou": sum(row["miou"] for row in PAPER_SCENE_PERCENT.values())
                / len(SCENES),
                "acc025": sum(
                    row["acc025"] for row in PAPER_SCENE_PERCENT.values()
                )
                / len(SCENES),
            },
            "acc025_query_micro_derived_from_integer_hits_percent": 100.0
            * (45 + 46 + 25 + 14)
            / 208,
            "query_micro_warning": (
                "No exact paper mIoU query-micro is reported; do not compare "
                "the local micro to the paper macro."
            ),
        },
    }


def write_json_exclusive(path: Path, payload: Mapping[str, object]) -> str:
    """Atomically publish JSON without ever replacing an existing artifact."""

    _require(path.is_absolute(), "output path must be absolute")
    _require(path.parent.is_dir(), f"output parent does not exist: {path.parent}")
    _require(not path.exists(), f"refusing to overwrite existing output: {path}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.tmp."
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise AuditError(f"refusing to overwrite existing output: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256_bytes(encoded)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teatime-report", type=Path, required=True)
    parser.add_argument(
        "--remaining-root",
        type=Path,
        required=True,
        help="Directory containing the other three <scene>_l3_mask0p4_eval.json files.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--expected-teatime-sha256",
        default=EXPECTED_TEATIME_SHA256,
        help="Immutable Teatime report fingerprint.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _require(not args.output_json.exists(), f"output already exists: {args.output_json}")
    report_paths = {
        "figurines": args.remaining_root / "figurines_l3_mask0p4_eval.json",
        "teatime": args.teatime_report,
        "ramen": args.remaining_root / "ramen_l3_mask0p4_eval.json",
        "waldo_kitchen": args.remaining_root / "waldo_kitchen_l3_mask0p4_eval.json",
    }
    result = aggregate_reports(
        report_paths,
        required_hashes={"teatime": args.expected_teatime_sha256},
    )
    output_sha256 = write_json_exclusive(args.output_json, result)
    macro = result["scene_equal_macro"]
    micro = result["query_micro"]
    assert isinstance(macro, Mapping) and isinstance(micro, Mapping)
    macro_percent = macro["metrics_percent"]
    micro_percent = micro["metrics_percent"]
    assert isinstance(macro_percent, Mapping) and isinstance(micro_percent, Mapping)
    print(f"Wrote {args.output_json}")
    print(f"SHA-256 {output_sha256}")
    print(
        "Scene-equal macro: "
        f"mIoU={float(macro_percent['miou']):.4f} "
        f"Acc@0.25={float(macro_percent['acc025']):.4f} "
        f"Acc@0.5={float(macro_percent['acc05']):.4f}"
    )
    print(
        "208-query micro: "
        f"mIoU={float(micro_percent['miou']):.4f} "
        f"Acc@0.25={float(micro_percent['acc025']):.4f} "
        f"Acc@0.5={float(micro_percent['acc05']):.4f}"
    )
    print("Evidence class: scale-paired compatibility reproduction (not strict checkpoint)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as error:
        print(f"audit failed: {error}", file=sys.stderr)
        raise SystemExit(2)
