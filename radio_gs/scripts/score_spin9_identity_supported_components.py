#!/usr/bin/env python3
"""Score sealed SPIn baseline and frozen identity-supported subset predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from radio_gs.data.promptable_nvs_manifest import validate_manifest
from radio_gs.evaluation.promptable_segmentation import evaluate_binary_scores, load_ground_truth_mask
from radio_gs.scripts.filter_spin9_sam3_components_by_identity_support import (
    CANDIDATE_ID,
    LOCAL_DENSITY_CANDIDATE_ID,
)
from radio_gs.scripts.predict_nvos_method_v1_field_box_sam3 import _write_json
from radio_gs.scripts.predict_nvos_method_v1_transient_sam import _sha256


def _load(path: str | Path, expected_sha: str, label: str) -> tuple[dict[str, Any], Path]:
    source = Path(path).expanduser().resolve(strict=True)
    if len(expected_sha) != 64 or _sha256(source) != expected_sha:
        raise ValueError(f"{label} SHA-256 differs")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, source


def _prediction_path(manifest: Mapping[str, Any], source: Path, scene: str, frame: str) -> Path:
    root = Path(str(manifest.get("prediction_root", ".")))
    if not root.is_absolute():
        root = source.parent / root
    path = (root / str(manifest["predictions"][scene][frame])).resolve(strict=True)
    if _sha256(path) != str(manifest["prediction_sha256"][scene][frame]):
        raise ValueError(f"prediction differs: {scene}/{frame}")
    return path


def _evaluate(
    manifest: Mapping[str, Any], source: Path, normalized: Mapping[str, Any], scenes: Sequence[str]
) -> dict[str, Any]:
    index = {str(row["scene_id"]): row for row in normalized["scenes"]}
    scene_rows: list[dict[str, Any]] = []
    for scene_id in scenes:
        scene = index[scene_id]
        frame_values = (
            scene["frames"].values()
            if isinstance(scene["frames"], Mapping)
            else scene["frames"]
        )
        frame_index = {str(row["frame_id"]): row for row in frame_values}
        frames: list[dict[str, Any]] = []
        for frame_id in scene["evaluation_frame_ids"]:
            frame_id = str(frame_id)
            score_path = _prediction_path(manifest, source, scene_id, frame_id)
            gt_path = Path(str(frame_index[frame_id]["ground_truth"])).resolve(strict=True)
            score = np.load(score_path, allow_pickle=False)
            target = load_ground_truth_mask(gt_path)
            metrics = evaluate_binary_scores(score, target, threshold=0.0)
            frames.append({"frame_id": frame_id, **metrics})
        scene_rows.append({
            "scene_id": scene_id,
            "frame_count": len(frames),
            "foreground_iou": float(np.mean([row["foreground_iou"] for row in frames])),
            "pixel_accuracy": float(np.mean([row["pixel_accuracy"] for row in frames])),
            "frames": frames,
        })
    return {
        "scene_count": len(scene_rows),
        "scene_macro_foreground_iou": float(np.mean([row["foreground_iou"] for row in scene_rows])),
        "scene_macro_pixel_accuracy": float(np.mean([row["pixel_accuracy"] for row in scene_rows])),
        "scenes": scene_rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--expected-baseline-manifest-sha256", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--expected-candidate-manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    dataset_path = Path(args.dataset_manifest).expanduser().resolve(strict=True)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    normalized = validate_manifest(dataset, check_files=False)
    baseline, baseline_path = _load(args.baseline_manifest, args.expected_baseline_manifest_sha256, "baseline manifest")
    candidate, candidate_path = _load(args.candidate_manifest, args.expected_candidate_manifest_sha256, "candidate manifest")
    scenes = [str(value) for value in candidate.get("scene_order", [])]
    if (
        not scenes
        or baseline.get("scene_order") != scenes
        or baseline.get("protocol_hash") != normalized["protocol_hash"]
        or candidate.get("protocol_hash") != normalized["protocol_hash"]
        or candidate.get("candidate_id")
        not in (CANDIDATE_ID, LOCAL_DENSITY_CANDIDATE_ID)
        or baseline.get("evaluation_performed") is not False
        or candidate.get("evaluation_performed") is not False
        or baseline.get("target_mask_opened") is not False
        or candidate.get("target_mask_opened") is not False
    ):
        raise ValueError("sealed SPIn confirmation barrier differs")
    base = _evaluate(baseline, baseline_path, normalized, scenes)
    result = _evaluate(candidate, candidate_path, normalized, scenes)
    report = {
        "schema_version": 1,
        "artifact_type": "radio_gs_spin9_identity_supported_components_independent_result",
        "candidate_id": candidate["candidate_id"],
        "pre_gt_barrier": {
            "baseline_manifest": str(baseline_path),
            "baseline_manifest_sha256": args.expected_baseline_manifest_sha256,
            "candidate_manifest": str(candidate_path),
            "candidate_manifest_sha256": args.expected_candidate_manifest_sha256,
            "frozen_rule_before_first_spin_target_metric": True,
        },
        "baseline": base,
        "candidate": result,
        "delta": {
            "scene_macro_foreground_iou": result["scene_macro_foreground_iou"] - base["scene_macro_foreground_iou"],
            "scene_macro_pixel_accuracy": result["scene_macro_pixel_accuracy"] - base["scene_macro_pixel_accuracy"],
        },
        "eligibility": {
            "independent_cross_benchmark_confirmation": True,
            "development_subset": bool(candidate.get("development_subset", False)),
            "threshold_selected_on_spin_target_metrics": False,
        },
    }
    output = Path(args.output).expanduser().resolve()
    _write_json(output, report)
    print(json.dumps({"output": str(output), "baseline": base["scene_macro_foreground_iou"], "candidate": result["scene_macro_foreground_iou"], "delta": report["delta"]["scene_macro_foreground_iou"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
