#!/usr/bin/env python3
"""Bind the strict-unseen NVOS current-field screen to immutable inputs/results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from radio_gs.data.promptable_nvs_manifest import NVOS_TASKS, validate_manifest


ARTIFACT_TYPE = "nvos-current-field-strict-unseen-exact-authority-v1"
REQUIRED_SOURCE_PATHS = (
    "radio_gs/data/promptable_nvs_manifest.py",
    "radio_gs/evaluation/promptable_feature_readout.py",
    "radio_gs/evaluation/promptable_segmentation.py",
    "radio_gs/field/checkpoint.py",
    "radio_gs/rendering/coefficient_renderer.py",
    "radio_gs/scripts/render_promptable_nvs_features.py",
    "radio_gs/scripts/predict_promptable_nvs_feature_readout.py",
    "radio_gs/scripts/eval_promptable_nvs_segmentation.py",
)


class AuthorityError(ValueError):
    """Raised when an alleged exact result is not completely bound."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuthorityError(f"Expected a JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorityError(message)


def validate_protocol_semantics(protocol: Mapping[str, Any]) -> None:
    required = {
        "benchmark": "NVOS",
        "aggregation": "per_frame_then_per_scene_then_dataset_scene_macro",
        "resize": "nearest",
        "prediction_representation": "continuous_margin",
        "threshold_comparison": "greater_or_equal",
        "score_semantics": "cosine_similarity_foreground_minus_background",
        "score_temperature": "none",
        "target_rgb_during_field_training": "forbidden",
        "target_rgb_at_query": "forbidden",
        "target_mask_use": "scoring_only",
        "within_scene_aggregation": "single_official_target",
    }
    for key, expected in required.items():
        _require(protocol.get(key) == expected, f"Frozen protocol {key!r} drifted")
    _require(
        protocol.get("cohort") == list(NVOS_TASKS),
        "Frozen NVOS full8 cohort/order drifted",
    )
    threshold = protocol.get("threshold")
    _require(
        isinstance(threshold, Mapping)
        and threshold.get("mode") == "fixed"
        and float(threshold.get("value", float("nan"))) == 0.0,
        "NVOS exact requires fixed score >= 0",
    )


def validate_target_fence(scene: Mapping[str, Any]) -> tuple[str, str]:
    scene_id = str(scene.get("scene_id") or "")
    evaluation_ids = scene.get("evaluation_frame_ids")
    prompt_ids = scene.get("prompt_frame_ids")
    _require(
        isinstance(evaluation_ids, list) and len(evaluation_ids) == 1,
        f"{scene_id}: expected one target",
    )
    _require(
        isinstance(prompt_ids, list) and len(prompt_ids) == 1,
        f"{scene_id}: expected one prompt frame",
    )
    target_id, prompt_id = str(evaluation_ids[0]), str(prompt_ids[0])
    training = scene.get("training_frames")
    _require(isinstance(training, list), f"{scene_id}: training_frames missing")
    training_ids = {str(row.get("frame_id")) for row in training if isinstance(row, Mapping)}
    _require(target_id not in training_ids, f"{scene_id}: target leaked into training")
    _require(
        scene.get("excluded_training_frame_ids") == [target_id],
        f"{scene_id}: target exclusion declaration drifted",
    )
    _require(
        scene.get("target_rgb_policy") == "excluded_from_field_training_and_query",
        f"{scene_id}: target RGB policy drifted",
    )
    return prompt_id, target_id


def validate_score_file(path: Path, expected_sha256: str) -> dict[str, Any]:
    _require(path.is_file(), f"Score is missing: {path}")
    actual_sha = _sha256(path)
    _require(actual_sha == expected_sha256, f"Score SHA drifted: {path}")
    values = np.load(path, allow_pickle=False)
    _require(values.dtype == np.float32, f"Score must be float32: {path}")
    _require(values.ndim == 2 and values.size > 0, f"Score must be nonempty 2-D: {path}")
    _require(bool(np.isfinite(values).all()), f"Score contains NaN/Inf: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": actual_sha,
        "dtype": "float32",
        "shape": list(values.shape),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def _frame_map(scene: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    frames = scene.get("frames")
    _require(isinstance(frames, list), f"{scene.get('scene_id')}: frames missing")
    return {
        str(frame.get("frame_id")): frame
        for frame in frames
        if isinstance(frame, Mapping)
    }


def build_authority(
    *,
    repo_root: Path,
    protocol_freeze: Path,
    dataset_manifest: Path,
    queue_root: Path,
    canonical_field_root: Path,
    render_root: Path,
    prediction_manifest_path: Path,
    evaluation_report_path: Path,
) -> dict[str, Any]:
    raw_manifest = _load_json(dataset_manifest)
    normalized = validate_manifest(raw_manifest, check_files=True)
    protocol_hash = str(normalized["protocol_hash"])
    validate_protocol_semantics(normalized["protocol"])
    raw_scenes = {
        str(scene.get("scene_id")): scene
        for scene in raw_manifest.get("scenes", [])
        if isinstance(scene, Mapping)
    }
    _require(list(raw_scenes) == list(NVOS_TASKS), "Raw manifest cohort/order drifted")

    prediction_manifest = _load_json(prediction_manifest_path)
    evaluation = _load_json(evaluation_report_path)
    _require(
        prediction_manifest.get("kind") == "promptable_nvs_continuous_score_predictions",
        "Prediction manifest kind drifted",
    )
    _require(
        prediction_manifest.get("protocol_hash") == protocol_hash
        and evaluation.get("protocol_hash") == protocol_hash,
        "Prediction/evaluation protocol hash mismatch",
    )
    prediction_input = prediction_manifest.get("input")
    _require(isinstance(prediction_input, Mapping), "Prediction input binding missing")
    _require(
        prediction_input.get("dataset_manifest_sha256") == _sha256(dataset_manifest),
        "Prediction dataset-manifest SHA mismatch",
    )
    method = prediction_manifest.get("method")
    _require(isinstance(method, Mapping), "Prediction method metadata missing")
    _require(method.get("readout") == "reference_prototype_cosine_margin", "Readout drifted")
    _require(method.get("score_semantics") == normalized["protocol"]["score_semantics"], "Score semantics drifted")
    _require(method.get("threshold") == {"mode": "fixed", "value": 0.0, "source": "input_manifest"}, "Prediction threshold drifted")
    prediction_safety = prediction_manifest.get("safety")
    _require(
        isinstance(prediction_safety, Mapping)
        and prediction_safety.get("evaluation_ground_truth_opened") is False
        and prediction_safety.get("evaluation_performed") is False,
        "Prediction-stage target-GT fence is not attested",
    )

    prediction_scene_rows = {
        str(row.get("scene_id")): row
        for row in prediction_manifest.get("scenes", [])
        if isinstance(row, Mapping)
    }
    evaluation_scene_rows = evaluation.get("scenes")
    _require(
        isinstance(evaluation_scene_rows, list)
        and [str(row.get("scene_id")) for row in evaluation_scene_rows] == list(NVOS_TASKS),
        "Evaluation full8 cohort/order drifted",
    )
    _require(
        evaluation.get("dataset", {}).get("num_scenes") == 8
        and evaluation.get("dataset", {}).get("num_frames") == 8,
        "Evaluation is not full8/single-target",
    )
    validate_protocol_semantics(evaluation.get("protocol", {}))
    _require(
        evaluation.get("thresholds", {}).get("policy") == {"mode": "fixed", "value": 0.0},
        "Evaluation threshold policy drifted",
    )

    scene_authorities: list[dict[str, Any]] = []
    prediction_base = prediction_manifest_path.parent
    for scene_id in NVOS_TASKS:
        raw_scene = raw_scenes[scene_id]
        prompt_id, target_id = validate_target_fence(raw_scene)
        frames = _frame_map(raw_scene)
        _require(prompt_id in frames and target_id in frames, f"{scene_id}: frame records missing")
        prompt_camera = str(frames[prompt_id].get("camera_name") or "")
        target_camera = str(frames[target_id].get("camera_name") or "")

        render_manifest_path = render_root / scene_id / "render_manifest.json"
        render = _load_json(render_manifest_path)
        _require(render.get("protocol_hash") == protocol_hash, f"{scene_id}: render protocol drifted")
        _require(render.get("scene_id") == scene_id, f"{scene_id}: render scene drifted")
        _require(
            render.get("render_mode") == "canonical_mpr_v3_affine_normalized_splat",
            f"{scene_id}: render is not canonical-mpr-v3",
        )
        _require(render.get("manifest_file_sha256") == _sha256(dataset_manifest), f"{scene_id}: render manifest SHA drifted")
        safety = render.get("safety")
        _require(
            isinstance(safety, Mapping)
            and safety.get("rgb_files_opened") is False
            and safety.get("segmentation_masks_opened") is False
            and safety.get("evaluation_ground_truth_opened") is False
            and safety.get("rgb_refiner_used") is False,
            f"{scene_id}: render target-data fence is not attested",
        )
        config = Path(str(render.get("config"))).resolve()
        checkpoint = Path(str(render.get("checkpoint"))).resolve()
        canonical_field = Path(str(render.get("canonical_field_checkpoint"))).resolve()
        camera_map = Path(str(render.get("camera_map"))).resolve()
        _require(config == (queue_root / "scenes" / scene_id / "gaussfm_main_track.yaml").resolve(), f"{scene_id}: config path drifted")
        _require(checkpoint == (queue_root / "scenes" / scene_id / "feature_field/checkpoints/best.pth").resolve(), f"{scene_id}: checkpoint path drifted")
        _require(
            canonical_field
            == (
                canonical_field_root
                / scene_id
                / "canonical_d256_l128_capability_first.pth"
            ).resolve(),
            f"{scene_id}: canonical field path drifted",
        )
        _require(camera_map == (queue_root / "scenes" / scene_id / "rgb_to_colmap_camera_mapping.json").resolve(), f"{scene_id}: camera-map path drifted")
        for path, key in ((config, "config_sha256"), (checkpoint, "checkpoint_sha256"), (camera_map, "camera_map_sha256")):
            _require(path.is_file() and render.get(key) == _sha256(path), f"{scene_id}: {key} mismatch")
        _require(
            canonical_field.is_file()
            and render.get("canonical_field_checkpoint_sha256")
            == _sha256(canonical_field),
            f"{scene_id}: canonical field SHA mismatch",
        )
        _require(
            render.get("canonical_render_contract")
            == {
                "normalized_splat": True,
                "affine_decode_after_splat": True,
                "reliability_splat": False,
                "screen_refiner": False,
            },
            f"{scene_id}: canonical render contract drifted",
        )
        render_outputs = render.get("outputs")
        _require(isinstance(render_outputs, list) and len(render_outputs) == 2, f"{scene_id}: render must contain prompt+target only")
        roles = {str(row.get("role")): row for row in render_outputs if isinstance(row, Mapping)}
        _require(set(roles) == {"prompt", "evaluation"}, f"{scene_id}: render roles drifted")
        _require(roles["prompt"].get("frame_id") == prompt_id and roles["prompt"].get("camera_name") == prompt_camera, f"{scene_id}: prompt render drifted")
        _require(roles["evaluation"].get("frame_id") == target_id and roles["evaluation"].get("camera_name") == target_camera, f"{scene_id}: target render drifted")

        prediction_scene = prediction_scene_rows.get(scene_id)
        _require(isinstance(prediction_scene, Mapping), f"{scene_id}: prediction scene missing")
        _require(prediction_scene.get("prompt_frame_id") == prompt_id, f"{scene_id}: prompt feature drifted")
        prompt_feature = Path(str(prediction_scene.get("prompt_feature_path"))).resolve()
        _require(prompt_feature == Path(str(roles["prompt"].get("feature_path"))).resolve(), f"{scene_id}: prompt feature is not rendered output")
        _require(_sha256(prompt_feature) == prediction_scene.get("prompt_feature_sha256"), f"{scene_id}: prompt feature SHA drifted")
        outputs = prediction_scene.get("outputs")
        _require(isinstance(outputs, list) and len(outputs) == 1, f"{scene_id}: expected one score")
        output = outputs[0]
        _require(output.get("frame_id") == target_id and output.get("score_dtype") == "float32", f"{scene_id}: target score metadata drifted")
        target_feature = Path(str(output.get("feature_path"))).resolve()
        _require(target_feature == Path(str(roles["evaluation"].get("feature_path"))).resolve(), f"{scene_id}: target feature is not rendered output")
        _require(_sha256(target_feature) == output.get("feature_sha256"), f"{scene_id}: target feature SHA drifted")
        score_path = (prediction_base / str(output.get("score_path"))).resolve()
        score = validate_score_file(score_path, str(output.get("score_sha256")))
        _require(
            prediction_manifest.get("prediction_sha256", {}).get(scene_id, {}).get(target_id) == score["sha256"],
            f"{scene_id}: score index SHA drifted",
        )
        scene_authorities.append(
            {
                "scene_id": scene_id,
                "prompt_frame_id": prompt_id,
                "target_frame_id": target_id,
                "target_absent_from_training": True,
                "render_manifest": {"path": str(render_manifest_path.resolve()), "sha256": _sha256(render_manifest_path)},
                "camera_map": {"path": str(camera_map), "sha256": _sha256(camera_map)},
                "config": {"path": str(config), "sha256": _sha256(config)},
                "geometry_carrier_checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
                "canonical_field_checkpoint": {"path": str(canonical_field), "sha256": _sha256(canonical_field)},
                "prompt_feature": {"path": str(prompt_feature), "sha256": _sha256(prompt_feature)},
                "target_feature": {"path": str(target_feature), "sha256": _sha256(target_feature)},
                "score": score,
            }
        )

    foreground_iou = float(evaluation["dataset"]["foreground_iou"])
    pixel_accuracy = float(evaluation["dataset"]["pixel_accuracy"])
    _require(math.isfinite(foreground_iou) and math.isfinite(pixel_accuracy), "Non-finite aggregate")
    scene_miou = sum(float(row["foreground_iou"]) for row in evaluation_scene_rows) / 8.0
    scene_macc = sum(float(row["pixel_accuracy"]) for row in evaluation_scene_rows) / 8.0
    _require(abs(scene_miou - foreground_iou) <= 1e-15, "IoU is not equal scene macro")
    _require(abs(scene_macc - pixel_accuracy) <= 1e-15, "Accuracy is not equal scene macro")

    source_artifacts = []
    for relative in REQUIRED_SOURCE_PATHS:
        path = (repo_root / relative).resolve()
        _require(path.is_file(), f"Required source missing: {path}")
        source_artifacts.append({"path": str(path), "sha256": _sha256(path)})
    _require(protocol_freeze.is_file(), f"Protocol freeze missing: {protocol_freeze}")
    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "benchmark": "NVOS",
        "method": "RADIO-GS canonical-mpr-v3 strict-unseen feature field",
        "status": "complete_exact_full8_canonical_mpr_v3_screen",
        "comparability": {
            "strict_unseen_target_rgb_and_mask": True,
            "same_training_visibility_as_frozen_LUDVIG_baseline": False,
            "note": "This is the strict-unseen RADIO-GS track; LUDVIG's frozen baseline is all-view.",
        },
        "protocol": {
            "protocol_hash": protocol_hash,
            "dataset_manifest": {"path": str(dataset_manifest.resolve()), "sha256": _sha256(dataset_manifest)},
            "evaluation_protocol_freeze": {"path": str(protocol_freeze.resolve()), "sha256": _sha256(protocol_freeze)},
            "score": "float32 cosine(foreground)-cosine(background)",
            "threshold": "score >= 0",
            "resize": "nearest",
            "aggregation": "one target per task, equal macro over full8",
        },
        "safety": {
            "render_opens_rgb": False,
            "render_opens_masks": False,
            "prediction_opens_target_ground_truth": False,
            "target_rgb_excluded_from_field_training": True,
            "target_rgb_excluded_at_query": True,
            "target_mask_scoring_only": True,
        },
        "source_artifacts": source_artifacts,
        "prediction_manifest": {"path": str(prediction_manifest_path.resolve()), "sha256": _sha256(prediction_manifest_path)},
        "evaluation_report": {"path": str(evaluation_report_path.resolve()), "sha256": _sha256(evaluation_report_path)},
        "scenes": scene_authorities,
        "metrics": {
            "foreground_iou": foreground_iou,
            "pixel_accuracy": pixel_accuracy,
            "num_scenes": 8,
            "num_frames": 8,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--protocol-freeze", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--canonical-field-root", type=Path, required=True)
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Authority exists (use --overwrite): {output}")
    authority = build_authority(
        repo_root=args.repo_root.expanduser().resolve(),
        protocol_freeze=args.protocol_freeze.expanduser().resolve(),
        dataset_manifest=args.dataset_manifest.expanduser().resolve(),
        queue_root=args.queue_root.expanduser().resolve(),
        canonical_field_root=args.canonical_field_root.expanduser().resolve(),
        render_root=args.render_root.expanduser().resolve(),
        prediction_manifest_path=args.prediction_manifest.expanduser().resolve(),
        evaluation_report_path=args.evaluation_report.expanduser().resolve(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(authority, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps({"output": str(output), "sha256": _sha256(output), "metrics": authority["metrics"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
