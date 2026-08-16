#!/usr/bin/env python3
"""Convert sealed Method-v1 field prompts into transient official-SAM3 masks.

This prediction-only stage opens the declared NVOS target RGB after a signed
field prompt has been sealed.  It never opens a target mask or metric.  Ten
deterministic trials each use three positive and three negative points; no
graph, component filter, target calibration, or proposal selection is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch

from radio_gs.data.promptable_nvs_manifest import (
    validate_manifest as validate_dataset_manifest,
)
from radio_gs.five_benchmark_method_v1 import (
    METHOD_ID,
    validate_method_authority,
)
from radio_gs.querying.transient_rgb_sam import (
    FROZEN_POLICY,
    PromptMode,
    aggregate_sam_trials,
    deterministic_signed_point_trials,
    transient_adapter_contract,
)
from radio_gs.scripts.build_sam3_foundation_cache import (
    _load_sam3_model,
    sam3_autocast_context,
    set_requested_cuda_device,
)
from radio_gs.scripts.validate_nvos_rgb_assisted_contract import (
    DEFAULT_CONTRACT as DEFAULT_EVALUATION_CONTRACT,
    validate_contract as validate_evaluation_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METHOD_AUTHORITY = (
    REPO_ROOT / "paper/artifacts/five_benchmark_method_v1_authority_20260815.json"
)
DEFAULT_SAM3_CHECKPOINT = REPO_ROOT / "checkpoints/sam3_modelscope/sam3.pt"
FROZEN_SAM3_CHECKPOINT_SHA256 = (
    "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
)
SAM_WIDTH = 1008
SAM_HEIGHT = 756


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str | Path, *, label: str) -> tuple[dict[str, Any], Path, str]:
    source = Path(path).expanduser().resolve(strict=True)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, source, _sha256(source)


def _write_json_noclobber(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_numpy_noclobber(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.asarray(value, dtype=np.float32), allow_pickle=False)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_torch_noclobber(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(dict(value), temporary)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_png_noclobber(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".png", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        Image.fromarray(np.asarray(value, dtype=np.uint8), mode="L").save(temporary)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _raw_scene(manifest: Mapping[str, Any], scene_id: str) -> Mapping[str, Any]:
    scenes = [
        row
        for row in manifest.get("scenes", [])
        if isinstance(row, Mapping) and str(row.get("scene_id")) == scene_id
    ]
    if len(scenes) != 1:
        raise ValueError(f"expected exactly one frozen scene {scene_id!r}")
    return scenes[0]


def _raw_frame(scene: Mapping[str, Any], frame_id: str) -> Mapping[str, Any]:
    frames = scene.get("frames", [])
    values = list(frames.values()) if isinstance(frames, Mapping) else list(frames)
    matches = [
        row
        for row in values
        if isinstance(row, Mapping) and str(row.get("frame_id")) == frame_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one frozen frame {frame_id!r}")
    return matches[0]


def _prediction_override(
    manifest: Mapping[str, Any], scene_id: str, frame_id: str
) -> tuple[Path, str]:
    predictions = manifest.get("predictions")
    hashes = manifest.get("prediction_sha256")
    if not isinstance(predictions, Mapping) or not isinstance(hashes, Mapping):
        raise ValueError("signed field prompt manifest lacks predictions or hashes")
    scene_predictions = predictions.get(scene_id)
    scene_hashes = hashes.get(scene_id)
    if not isinstance(scene_predictions, Mapping) or not isinstance(scene_hashes, Mapping):
        raise ValueError(f"signed field prompt is absent for {scene_id}")
    value = scene_predictions.get(frame_id)
    expected = str(scene_hashes.get(frame_id, ""))
    if not value or len(expected) != 64:
        raise ValueError(f"signed field prompt authority is incomplete for {scene_id}")
    return Path(str(value)), expected


def _validate_readout_authority(authority: Mapping[str, Any]) -> None:
    validate_method_authority(authority)
    readout = authority.get("readouts", {}).get("nvos", {})
    expected = {
        "operator": "signed_field_prompt_to_query_transient_target_rgb_frozen_sam",
        "trials": 10,
        "positive_points": 3,
        "negative_points": 3,
        "reference_mask_selection": False,
        "graph_or_connected_component": False,
    }
    if not isinstance(readout, Mapping) or any(
        readout.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("frozen Method-v1 NVOS readout authority differs")


def load_signed_field_prompt(
    *,
    dataset_manifest_path: str | Path,
    prompt_manifest_path: str | Path,
    method_authority_path: str | Path,
    evaluation_contract_path: str | Path,
    scene_id: str,
) -> dict[str, Any]:
    """Resolve one target RGB and signed score without opening target GT."""

    dataset, dataset_path, dataset_sha = _load_json(
        dataset_manifest_path, label="NVOS dataset manifest"
    )
    normalized = validate_dataset_manifest(dataset, check_files=False)
    authority, authority_path, authority_sha = _load_json(
        method_authority_path, label="Method-v1 authority"
    )
    _validate_readout_authority(authority)
    contract = validate_evaluation_contract(evaluation_contract_path)
    if (
        Path(contract["dataset_manifest"]) != dataset_path
        or contract["dataset_manifest_sha256"] != dataset_sha
        or Path(contract["method_authority"]) != authority_path
        or contract["method_authority_sha256"] != authority_sha
    ):
        raise ValueError("NVOS RGB-assisted contract authority binding differs")
    frozen_cohort = [str(value) for value in authority["frozen_cohorts"]["nvos"]]
    if scene_id not in frozen_cohort:
        raise ValueError(f"scene is outside frozen Method-v1 NVOS cohort: {scene_id}")

    prompt_manifest, prompt_path, prompt_sha = _load_json(
        prompt_manifest_path, label="signed field prompt manifest"
    )
    if (
        prompt_manifest.get("kind") != "promptable_nvs_continuous_score_predictions"
        or prompt_manifest.get("protocol_hash") != normalized["protocol_hash"]
        or prompt_manifest.get("method", {}).get("readout")
        != "reference_prototype_cosine_margin"
    ):
        raise ValueError("signed field prompt manifest contract differs")
    prompt_scenes = [
        row
        for row in prompt_manifest.get("scenes", [])
        if isinstance(row, Mapping) and str(row.get("scene_id")) == scene_id
    ]
    if len(prompt_scenes) != 1:
        raise ValueError(f"signed field prompt scene authority differs for {scene_id}")
    render_authority = prompt_scenes[0].get("feature_render_authority")
    if (
        not isinstance(render_authority, Mapping)
        or render_authority.get("field_checkpoint_schema") != "factorized-v2"
        or len(str(render_authority.get("field_checkpoint_sha256") or "")) != 64
    ):
        raise ValueError("signed field prompt lacks factorized-v2 field authority")
    render_authority_path = Path(str(render_authority.get("path", ""))).resolve(
        strict=True
    )
    if _sha256(render_authority_path) != render_authority.get("sha256"):
        raise ValueError("signed field prompt render authority SHA256 differs")

    normalized_scenes = {
        str(row["scene_id"]): row for row in normalized["scenes"]
    }
    scene = normalized_scenes.get(scene_id)
    if scene is None or len(scene["evaluation_frame_ids"]) != 1:
        raise ValueError(f"{scene_id} must have one frozen evaluation frame")
    frame_id = str(scene["evaluation_frame_ids"][0])
    relative_score, expected_score_sha = _prediction_override(
        prompt_manifest, scene_id, frame_id
    )
    prediction_root = Path(str(prompt_manifest.get("prediction_root", ".")))
    if not prediction_root.is_absolute():
        prediction_root = prompt_path.parent / prediction_root
    score_path = relative_score if relative_score.is_absolute() else prediction_root / relative_score
    score_path = score_path.resolve(strict=True)
    if _sha256(score_path) != expected_score_sha:
        raise ValueError("signed field prompt SHA256 differs")
    signed_margin = np.load(score_path, allow_pickle=False)
    if (
        signed_margin.ndim != 2
        or min(signed_margin.shape) <= 0
        or not bool(np.isfinite(signed_margin).all())
    ):
        raise ValueError("signed field prompt must be a finite [H,W] margin")

    raw_scene = _raw_scene(dataset, scene_id)
    if str(raw_scene.get("prompt", {}).get("type")) != "positive_negative_scribbles":
        raise ValueError("Method-v1 NVOS requires signed source scribbles")
    raw_frame = _raw_frame(raw_scene, frame_id)
    target_rgb = Path(str(raw_frame.get("rgb_path", ""))).expanduser().resolve(strict=True)
    if target_rgb.stem != str(raw_frame.get("camera_name", frame_id)):
        raise ValueError("target RGB frame/camera identity differs")
    return {
        "scene_id": scene_id,
        "frame_id": frame_id,
        "signed_margin": signed_margin.astype(np.float32, copy=False),
        "signed_margin_path": score_path,
        "signed_margin_sha256": expected_score_sha,
        "target_rgb_path": target_rgb,
        "target_rgb_sha256": _sha256(target_rgb),
        "dataset_manifest": dataset_path,
        "dataset_manifest_sha256": dataset_sha,
        "protocol_hash": normalized["protocol_hash"],
        "prompt_manifest": prompt_path,
        "prompt_manifest_sha256": prompt_sha,
        "method_authority": authority_path,
        "method_authority_sha256": authority_sha,
        "evaluation_contract": Path(contract["contract"]),
        "evaluation_contract_sha256": contract["contract_sha256"],
        "evaluation_contract_id": contract["contract_id"],
        "feature_render_authority": dict(render_authority),
        "legacy_dataset_target_rgb_policy": raw_scene.get("target_rgb_policy"),
    }


@torch.inference_mode()
def run_sam_trials(
    processor: Any,
    image: Image.Image,
    signed_margin: np.ndarray,
    *,
    device: str,
    amp_dtype: torch.dtype | None,
) -> dict[str, Any]:
    """Run the frozen no-selection SAM3 point interface."""

    margin = np.asarray(signed_margin, dtype=np.float32)
    positive = np.maximum(margin, 0.0)
    negative = np.maximum(-margin, 0.0)
    points, labels = deterministic_signed_point_trials(
        positive,
        negative,
        image_shape=(image.height, image.width),
        policy=FROZEN_POLICY,
    )
    with sam3_autocast_context(device, amp_dtype):
        state = processor.set_image(image)
    masks: list[np.ndarray] = []
    qualities: list[np.ndarray] = []
    low_resolution_shapes: list[list[int]] = []
    for trial_points, trial_labels in zip(points, labels):
        with sam3_autocast_context(device, amp_dtype):
            candidate_masks, quality, low_resolution = processor.model.predict_inst(
                state,
                point_coords=trial_points.astype(np.float32, copy=False),
                point_labels=trial_labels.astype(np.int32, copy=False),
                multimask_output=False,
            )
        candidate_masks = np.asarray(candidate_masks)
        quality = np.asarray(quality, dtype=np.float32).reshape(-1)
        low_resolution = np.asarray(low_resolution)
        if candidate_masks.shape != (1, image.height, image.width):
            raise ValueError(f"unexpected official SAM3 mask shape {candidate_masks.shape}")
        if quality.shape != (1,) or not bool(np.isfinite(quality).all()):
            raise ValueError("official SAM3 quality output differs")
        masks.append(candidate_masks.astype(np.float32, copy=False))
        qualities.append(quality)
        low_resolution_shapes.append(list(low_resolution.shape))
    trial_masks = np.stack(masks, axis=0)
    probability = aggregate_sam_trials(trial_masks, policy=FROZEN_POLICY)
    if probability.shape != (1, image.height, image.width):
        raise RuntimeError("frozen SAM3 aggregation shape differs")
    probability = probability[0]
    return {
        "probability": probability,
        "continuous_margin": probability - float(FROZEN_POLICY.signed_vote_threshold),
        "binary_mask": probability >= float(FROZEN_POLICY.signed_vote_threshold),
        "trial_masks": trial_masks,
        "point_coordinates_xy": points,
        "point_labels": labels,
        "quality": np.concatenate(qualities),
        "low_resolution_shapes": low_resolution_shapes,
    }


def predict(args: argparse.Namespace) -> dict[str, Any]:
    scene_ids = [str(value) for value in args.scene_ids]
    if not scene_ids or len(set(scene_ids)) != len(scene_ids):
        raise ValueError("--scene-id must be a nonempty unique list")
    checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    checkpoint_sha = _sha256(checkpoint)
    if checkpoint_sha != str(args.expected_checkpoint_sha256):
        raise ValueError("official SAM3 checkpoint SHA256 differs")
    sources = [
        load_signed_field_prompt(
            dataset_manifest_path=args.manifest,
            prompt_manifest_path=args.signed_field_prompt_manifest,
            method_authority_path=args.method_authority,
            evaluation_contract_path=args.evaluation_contract,
            scene_id=scene_id,
        )
        for scene_id in scene_ids
    ]
    output_root = Path(args.output_dir).expanduser().resolve()
    manifest_path = output_root / "prediction_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)

    set_requested_cuda_device(args.device)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    processor = _load_sam3_model(
        checkpoint_path=str(checkpoint),
        device=args.device,
        confidence_threshold=0.0,
        dtype="bfloat16",
        resolution=SAM_WIDTH,
        point_only=True,
    )
    started = time.time()
    predictions: dict[str, dict[str, str]] = {}
    prediction_hashes: dict[str, dict[str, str]] = {}
    receipts: list[dict[str, Any]] = []
    for source in sources:
        scene_id = str(source["scene_id"])
        frame_id = str(source["frame_id"])
        target = Image.open(source["target_rgb_path"]).convert("RGB")
        original_size = list(target.size)
        target = target.resize((SAM_WIDTH, SAM_HEIGHT), Image.Resampling.LANCZOS)
        result = run_sam_trials(
            processor,
            target,
            source["signed_margin"],
            device=args.device,
            amp_dtype=torch.bfloat16 if str(args.device).startswith("cuda") else None,
        )
        score_rel = Path("scores") / scene_id / f"{frame_id}.npy"
        score_path = output_root / score_rel
        trial_path = output_root / "trials" / scene_id / f"{frame_id}.pt"
        png_path = output_root / "masks" / scene_id / f"{frame_id}.png"
        _write_numpy_noclobber(score_path, result["continuous_margin"])
        _write_torch_noclobber(
            trial_path,
            {
                "trial_masks": torch.from_numpy(result["trial_masks"].copy()),
                "aggregate_probability": torch.from_numpy(result["probability"].copy()),
                "continuous_margin": torch.from_numpy(result["continuous_margin"].copy()),
                "point_coordinates_xy": torch.from_numpy(
                    result["point_coordinates_xy"].copy()
                ),
                "point_labels": torch.from_numpy(result["point_labels"].copy()),
                "quality": torch.from_numpy(result["quality"].copy()),
            },
        )
        _write_png_noclobber(
            png_path, result["binary_mask"].astype(np.uint8) * 255
        )
        score_sha = _sha256(score_path)
        predictions[scene_id] = {frame_id: score_rel.as_posix()}
        prediction_hashes[scene_id] = {frame_id: score_sha}
        receipt = {
            "schema_version": 1,
            "artifact_type": "radio_gs_method_v1_nvos_transient_sam_receipt",
            "method_id": METHOD_ID,
            "scene_id": scene_id,
            "frame_id": frame_id,
            "signed_field_prompt": {
                "path": str(source["signed_margin_path"]),
                "sha256": source["signed_margin_sha256"],
                "sealed_before_target_rgb_open": True,
            },
            "target_rgb": {
                "path": str(source["target_rgb_path"]),
                "sha256": source["target_rgb_sha256"],
                "original_size_wh": original_size,
                "sam_size_wh": [SAM_WIDTH, SAM_HEIGHT],
            },
            "output": {
                "continuous_margin_path": str(score_path),
                "continuous_margin_sha256": score_sha,
                "trial_artifact_path": str(trial_path),
                "trial_artifact_sha256": _sha256(trial_path),
                "mask_png_path": str(png_path),
                "mask_png_sha256": _sha256(png_path),
                "foreground_pixels": int(result["binary_mask"].sum()),
                "foreground_fraction": float(result["binary_mask"].mean()),
            },
            "policy": transient_adapter_contract(PromptMode.SIGNED_SCRIBBLE),
            "multimask_output": False,
            "candidate_selection": "none_exactly_one_candidate_per_trial",
            "aggregation": "mean_of_ten_official_binary_masks",
            "continuous_margin": "aggregate_probability_minus_0.5",
            "low_resolution_shapes": result["low_resolution_shapes"],
            "quality": result["quality"].tolist(),
            "authorities": {
                "dataset_manifest": str(source["dataset_manifest"]),
                "dataset_manifest_sha256": source["dataset_manifest_sha256"],
                "protocol_hash": source["protocol_hash"],
                "signed_field_prompt_manifest": str(source["prompt_manifest"]),
                "signed_field_prompt_manifest_sha256": source[
                    "prompt_manifest_sha256"
                ],
                "feature_render_authority": source["feature_render_authority"],
                "method_authority": str(source["method_authority"]),
                "method_authority_sha256": source["method_authority_sha256"],
                "evaluation_contract": str(source["evaluation_contract"]),
                "evaluation_contract_sha256": source[
                    "evaluation_contract_sha256"
                ],
                "evaluation_contract_id": source["evaluation_contract_id"],
                "official_sam3_checkpoint": str(checkpoint),
                "official_sam3_checkpoint_sha256": checkpoint_sha,
            },
            "safety": {
                "target_rgb_opened": True,
                "target_mask_opened": False,
                "target_metric_opened": False,
                "reference_mask_selection": False,
                "graph_used": False,
                "connected_component_used": False,
                "legacy_dataset_target_rgb_policy": source[
                    "legacy_dataset_target_rgb_policy"
                ],
                "target_rgb_access_authority": "Method-v1 authority target_access",
            },
        }
        receipt_path = output_root / "receipts" / f"{scene_id}.json"
        _write_json_noclobber(receipt_path, receipt)
        receipts.append(
            {
                "scene_id": scene_id,
                "path": str(receipt_path),
                "sha256": _sha256(receipt_path),
            }
        )
    protocol_hashes = {str(source["protocol_hash"]) for source in sources}
    if len(protocol_hashes) != 1:
        raise RuntimeError("selected scenes disagree on protocol hash")
    output_manifest = {
        "schema_version": 1,
        "kind": "promptable_nvs_method_v1_transient_sam_predictions",
        "protocol_hash": next(iter(protocol_hashes)),
        "prediction_root": ".",
        "predictions": predictions,
        "prediction_sha256": prediction_hashes,
        "method": {
            "id": METHOD_ID,
            "readout": "signed_field_prompt_to_query_transient_target_rgb_frozen_sam",
            "trials": FROZEN_POLICY.trials,
            "positive_points_per_trial": FROZEN_POLICY.positive_points_per_trial,
            "negative_points_per_trial": FROZEN_POLICY.negative_points_per_trial,
            "threshold": {"mode": "fixed", "value": 0.0},
            "score_semantics": "mean_binary_sam_vote_minus_0.5",
            "reference_mask_selection": False,
            "graph_or_connected_component": False,
        },
        "evaluation_contract": {
            "path": str(sources[0]["evaluation_contract"]),
            "sha256": sources[0]["evaluation_contract_sha256"],
            "contract_id": sources[0]["evaluation_contract_id"],
        },
        "receipts": receipts,
        "elapsed_seconds": float(time.time() - started),
        "evaluation_performed": False,
        "target_mask_opened": False,
        "target_metric_opened": False,
    }
    _write_json_noclobber(manifest_path, output_manifest)
    return {**output_manifest, "prediction_manifest_path": str(manifest_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--signed-field-prompt-manifest", required=True)
    parser.add_argument("--scene-id", action="append", dest="scene_ids", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-authority", default=str(DEFAULT_METHOD_AUTHORITY))
    parser.add_argument(
        "--evaluation-contract", default=str(DEFAULT_EVALUATION_CONTRACT)
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_SAM3_CHECKPOINT))
    parser.add_argument(
        "--expected-checkpoint-sha256", default=FROZEN_SAM3_CHECKPOINT_SHA256
    )
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = predict(build_parser().parse_args(argv))
    print(
        json.dumps(
            {
                "prediction_manifest": report["prediction_manifest_path"],
                "scenes": sorted(report["predictions"]),
                "evaluation_performed": False,
                "target_mask_opened": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
