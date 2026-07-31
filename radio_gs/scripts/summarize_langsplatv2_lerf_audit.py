#!/usr/bin/env python3
"""Summarize the exact-camera LangSplatV2 LERF-2D protocol audit.

The released evaluator logs scene metrics to four decimal places but does not
serialize per-query IoUs. Consequently, the 208-query mIoU below is a weighted
reconstruction from the four rounded scene means. Localization hit counts are
recoverable exactly because every scene's query count is known.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from radio_gs.scripts.eval_opengaussian_lerf_baseline import SCENE_GT_FRAMES


SCENE_ORDER = ("figurines", "teatime", "ramen", "waldo_kitchen")
PAPER_ROWS = {
    "figurines": {"miou": 0.564, "loc_acc": 0.821},
    "teatime": {"miou": 0.722, "loc_acc": 0.932},
    "ramen": {"miou": 0.518, "loc_acc": 0.747},
    "waldo_kitchen": {"miou": 0.591, "loc_acc": 0.955},
}
PAPER_OVERALL = {"miou": 0.599, "loc_acc": 0.841}
MIOU_PATTERN = re.compile(r"iou chosen:\s*([0-9.]+)")
LOC_PATTERN = re.compile(r"Localization accuracy:\s*([0-9.]+)")


def _load_colmap_image_names(scene_root: Path, baseline_root: Path) -> list[str]:
    loader_path = baseline_root / "scene" / "colmap_loader.py"
    spec = importlib.util.spec_from_file_location("_langsplatv2_colmap_loader", loader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load COLMAP reader from {loader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sparse_root = scene_root / "sparse" / "0"
    binary_path = sparse_root / "images.bin"
    text_path = sparse_root / "images.txt"
    if binary_path.exists():
        extrinsics = module.read_extrinsics_binary(str(binary_path))
    elif text_path.exists():
        extrinsics = module.read_extrinsics_text(str(text_path))
    else:
        raise FileNotFoundError(f"missing COLMAP images.bin/images.txt under {sparse_root}")
    names = sorted(Path(image.name).stem for image in extrinsics.values())
    if len(names) != len(set(names)):
        raise ValueError(f"{scene_root.name}: duplicate COLMAP image stems")
    return names


def _annotation_queries(annotation_path: Path) -> tuple[str, list[str]]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    image_name = str(payload["info"]["name"])
    queries = list(dict.fromkeys(str(item["category"]) for item in payload["objects"]))
    return image_name, queries


def _read_namespace_config(path: Path) -> dict[str, object]:
    expression = ast.parse(path.read_text(encoding="utf-8"), mode="eval").body
    if not isinstance(expression, ast.Call):
        raise ValueError(f"{path}: expected Namespace(...)")
    if not isinstance(expression.func, ast.Name) or expression.func.id != "Namespace":
        raise ValueError(f"{path}: expected Namespace(...)")
    if expression.args:
        raise ValueError(f"{path}: positional Namespace arguments are unsupported")
    config: dict[str, object] = {}
    for keyword in expression.keywords:
        if keyword.arg is None:
            raise ValueError(f"{path}: expanded Namespace arguments are unsupported")
        config[keyword.arg] = ast.literal_eval(keyword.value)
    return config


def validate_checkpoint_cohort(
    scene: str,
    checkpoint_root: Path,
    data_root: Path,
) -> dict[str, Any]:
    levels: dict[str, dict[str, Any]] = {}
    cohort_configs: list[dict[str, object]] = []
    for level in (1, 2, 3):
        level_root = checkpoint_root / f"{scene}_0_{level}"
        cfg_path = level_root / "cfg_args"
        checkpoint_path = level_root / "chkpnt10000.pth"
        if not cfg_path.exists() or not checkpoint_path.exists():
            raise FileNotFoundError(f"incomplete checkpoint cohort: {level_root}")
        config = _read_namespace_config(cfg_path)
        if Path(str(config.get("source_path", ""))).resolve() != (data_root / scene).resolve():
            raise ValueError(f"{cfg_path}: source_path does not match {scene}")
        if config.get("eval") is not True:
            raise ValueError(f"{cfg_path}: eval must be exactly True for LLFF role audit")
        if config.get("feature_level") != level:
            raise ValueError(f"{cfg_path}: feature_level does not match level {level}")
        if Path(str(config.get("model_path", ""))).name != level_root.name:
            raise ValueError(f"{cfg_path}: model_path does not identify {level_root.name}")
        cohort_configs.append(
            {
                key: value
                for key, value in config.items()
                if key not in {"feature_level", "model_path"}
            }
        )
        levels[str(level)] = {
            "cfg_args": str(cfg_path),
            "cfg_args_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
            "checkpoint": str(checkpoint_path),
            "checkpoint_size_bytes": checkpoint_path.stat().st_size,
            "source_path": config["source_path"],
            "eval": config["eval"],
            "feature_level": config["feature_level"],
        }
    for level, config in enumerate(cohort_configs[1:], start=2):
        if config != cohort_configs[0]:
            raise ValueError(
                f"{scene}: checkpoint configs differ between levels 1 and {level}"
            )
    return {
        "scene": scene,
        "iteration": 10000,
        "index": 0,
        "cfg_eval_verified": True,
        "levels": levels,
    }


def build_camera_manifest(
    scene: str,
    data_root: Path,
    label_root: Path,
    baseline_root: Path,
) -> dict[str, Any]:
    camera_names = _load_colmap_image_names(data_root / scene, baseline_root)
    index_by_name = {name: index for index, name in enumerate(camera_names)}
    frames: dict[str, dict[str, Any]] = {}
    for frame in SCENE_GT_FRAMES[scene]:
        annotation_path = label_root / scene / f"{frame}.json"
        annotation_image_name, queries = _annotation_queries(annotation_path)
        annotation_stem = Path(annotation_image_name).stem
        if annotation_stem != frame:
            raise ValueError(
                f"{annotation_path}: annotation image stem {annotation_stem!r} != {frame!r}"
            )
        if frame not in index_by_name:
            raise KeyError(f"{scene}/{frame}: no exact COLMAP camera-name match")
        camera_index = index_by_name[frame]
        frames[frame] = {
            "annotation_path": str(annotation_path),
            "annotation_image_name": annotation_image_name,
            "resolved_camera_name": frame,
            "sorted_camera_index": camera_index,
            "camera_role": "test" if camera_index % 8 == 0 else "train",
            "query_count": len(queries),
            "queries": queries,
        }
    return {
        "schema_version": 1,
        "method": "LangSplatV2",
        "scene": scene,
        "camera_resolution": "exact annotation-image stem across train+test camera union",
        "split": {
            "dataset_eval": True,
            "llffhold": 8,
            "ordering": "COLMAP cameras sorted lexicographically by image_name",
            "train_rule": "sorted_camera_index % 8 != 0",
            "test_rule": "sorted_camera_index % 8 == 0",
            "test_only_required": False,
        },
        "frames": frames,
        "frame_count": len(frames),
        "query_count": sum(int(item["query_count"]) for item in frames.values()),
        "camera_role_counts": {
            role: sum(item["camera_role"] == role for item in frames.values())
            for role in ("train", "test")
        },
    }


def _latest_complete_log(log_root: Path, scene: str) -> tuple[Path, float, float]:
    candidates = sorted(
        (log_root / f"{scene}_0").glob("*.log"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    for path in candidates:
        text = path.read_text(encoding="utf-8")
        miou_match = MIOU_PATTERN.search(text)
        loc_match = LOC_PATTERN.search(text)
        if miou_match and loc_match:
            return path, float(miou_match.group(1)), float(loc_match.group(1))
    raise FileNotFoundError(f"{scene}: no complete evaluator log under {log_root}")


def _git_output(baseline_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(baseline_root), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_cohorts = {
        scene: validate_checkpoint_cohort(
            scene,
            args.checkpoint_root,
            args.data_root,
        )
        for scene in SCENE_ORDER
    }
    manifests = {
        scene: build_camera_manifest(
            scene,
            args.data_root,
            args.label_root,
            args.baseline_root,
        )
        for scene in SCENE_ORDER
    }
    scene_metrics: dict[str, dict[str, Any]] = {}
    for scene in SCENE_ORDER:
        log_path, miou, loc_acc = _latest_complete_log(args.log_root, scene)
        objects = int(manifests[scene]["query_count"])
        loc_hits = round(loc_acc * objects)
        recovered_loc_acc = loc_hits / objects
        if abs(recovered_loc_acc - loc_acc) > 0.000051:
            raise ValueError(
                f"{scene}: logged LocAcc={loc_acc} cannot be reconciled with {objects} queries"
            )
        scene_metrics[scene] = {
            "miou": miou,
            "loc_acc": recovered_loc_acc,
            "loc_hits": loc_hits,
            "objects": objects,
            "log_path": str(log_path),
            "log_precision_decimal_places": 4,
            "paper": PAPER_ROWS[scene],
            "delta_vs_paper": {
                "miou": miou - PAPER_ROWS[scene]["miou"],
                "loc_acc": recovered_loc_acc - PAPER_ROWS[scene]["loc_acc"],
            },
        }

    object_total = sum(int(item["objects"]) for item in scene_metrics.values())
    scene_macro = {
        "miou": float(np.mean([float(item["miou"]) for item in scene_metrics.values()])),
        "loc_acc": float(np.mean([float(item["loc_acc"]) for item in scene_metrics.values()])),
        "scenes": len(scene_metrics),
        "aggregation": "scene_equal_macro",
    }
    query_micro = {
        "miou": (
            sum(float(item["miou"]) * int(item["objects"]) for item in scene_metrics.values())
            / object_total
        ),
        "loc_acc": (
            sum(int(item["loc_hits"]) for item in scene_metrics.values()) / object_total
        ),
        "loc_hits": sum(int(item["loc_hits"]) for item in scene_metrics.values()),
        "objects": object_total,
        "aggregation": "query_weighted_micro",
        "miou_precision_note": (
            "reconstructed from evaluator logs rounded to four decimal places; "
            "the released evaluator does not serialize per-query IoUs"
        ),
    }

    eval_patch = _git_output(args.baseline_root, "diff", "--", "eval_lerf.py")
    full_patch = _git_output(args.baseline_root, "diff")
    revision = _git_output(args.baseline_root, "rev-parse", "HEAD").strip()
    paper_scene_macro = {
        metric: float(np.mean([PAPER_ROWS[scene][metric] for scene in SCENE_ORDER]))
        for metric in ("miou", "loc_acc")
    }
    paper_query_micro = {
        metric: (
            sum(
                PAPER_ROWS[scene][metric] * int(scene_metrics[scene]["objects"])
                for scene in SCENE_ORDER
            )
            / object_total
        )
        for metric in ("miou", "loc_acc")
    }
    matched_paper_delta = {
        "miou": scene_macro["miou"] - PAPER_OVERALL["miou"],
        "miou_local_aggregation": "scene_equal_macro",
        "loc_acc": query_micro["loc_acc"] - PAPER_OVERALL["loc_acc"],
        "loc_acc_local_aggregation": "query_weighted_micro",
    }
    return {
        "schema_version": 1,
        "method": "LangSplatV2",
        "benchmark": "LERF-2D",
        "status": "complete_exact_camera_four_scene_cohort",
        "scene_order": list(SCENE_ORDER),
        "scene_metrics": scene_metrics,
        "aggregates": {
            "scene_equal_macro": scene_macro,
            "query_weighted_micro": query_micro,
        },
        "paper_context": {
            "per_scene_rows": PAPER_ROWS,
            "published_overall": PAPER_OVERALL,
            "computed_scene_equal_from_rounded_scene_rows": paper_scene_macro,
            "computed_query_micro_from_rounded_scene_rows_and_local_query_counts": paper_query_micro,
            "aggregation_inference": {
                "miou": (
                    "published 0.599 matches the four-scene equal macro of the "
                    "rounded scene rows (0.59875)"
                ),
                "loc_acc": (
                    "published 0.841 matches the 208-query weighted micro of the "
                    "rounded scene rows; their four-scene equal macro is 0.86375"
                ),
                "status": "mixed_overall_aggregation",
            },
            "matched_local_delta_vs_published_overall": matched_paper_delta,
            "note": (
                "Published overall values are retained verbatim. A single uniform aggregation "
                "must not be assigned: mIoU matches scene-macro, whereas LocAcc matches "
                "208-query micro."
            ),
        },
        "protocol": {
            "camera_mapping": "exact annotation filename across the train+test camera union",
            "mixed_camera_roles_allowed": True,
            "cfg_eval_verified_for_all_three_levels": True,
            "mask_threshold": 0.4,
            "checkpoint": 10000,
            "quick_render": True,
            "topk": 4,
            "scene_aggregation_reported": [
                "scene_equal_macro",
                "query_weighted_micro",
            ],
        },
        "provenance": {
            "baseline_root": str(args.baseline_root),
            "baseline_git_revision": revision,
            "eval_lerf_patch_sha256": hashlib.sha256(eval_patch.encode("utf-8")).hexdigest(),
            "full_tracked_diff_sha256": hashlib.sha256(full_patch.encode("utf-8")).hexdigest(),
            "eval_lerf_has_local_patch": bool(eval_patch),
            "data_root": str(args.data_root),
            "label_root": str(args.label_root),
            "log_root": str(args.log_root),
            "checkpoint_root": str(args.checkpoint_root),
        },
        "checkpoint_cohorts": checkpoint_cohorts,
        "camera_manifests": {
            scene: str(args.output_dir / "camera_manifests" / f"{scene}.json")
            for scene in SCENE_ORDER
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("/root/baselines/LangSplatV2"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/pool/sqy/3d_understanding/lerf_ovs"),
    )
    parser.add_argument(
        "--label-root",
        type=Path,
        default=Path("/mnt/pool/sqy/3d_understanding/lerf_ovs/label"),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path(
            "/root/RADIO-GS/output/baselines/langsplatv2/lerf_compat_20260518"
        ),
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path(
            "/root/RADIO-GS/output/protocol_audit_20260731/"
            "langsplatv2_lerf2d_view_fix"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/root/RADIO-GS/output/protocol_audit_20260731/"
            "langsplatv2_lerf2d_view_fix"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = summarize(args)
    manifest_dir = args.output_dir / "camera_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for scene in SCENE_ORDER:
        manifest = build_camera_manifest(
            scene,
            args.data_root,
            args.label_root,
            args.baseline_root,
        )
        path = manifest_dir / f"{scene}.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_path = args.output_dir / "cohort_summary.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    scene_macro = result["aggregates"]["scene_equal_macro"]
    query_micro = result["aggregates"]["query_weighted_micro"]
    print(
        f"scene-equal: mIoU={scene_macro['miou']:.4f} "
        f"LocAcc={scene_macro['loc_acc']:.4f}; "
        f"query-micro ({query_micro['objects']}): mIoU={query_micro['miou']:.4f} "
        f"LocAcc={query_micro['loc_acc']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
