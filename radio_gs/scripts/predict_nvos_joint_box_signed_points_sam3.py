#!/usr/bin/env python3
"""Run one official SAM3 decode with a field box and signed points jointly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch

from radio_gs.evaluation.promptable_segmentation import resize_mask_nearest
from radio_gs.five_benchmark_method_v1 import METHOD_ID
from radio_gs.querying.transient_rgb_sam import (
    FROZEN_POLICY,
    aggregate_sam_trials,
    deterministic_signed_point_trials,
)
from radio_gs.scripts.build_sam3_foundation_cache import (
    _load_sam3_model,
    sam3_autocast_context,
    set_requested_cuda_device,
)
from radio_gs.scripts.predict_nvos_method_v1_transient_sam import (
    DEFAULT_EVALUATION_CONTRACT,
    DEFAULT_METHOD_AUTHORITY,
    DEFAULT_SAM3_CHECKPOINT,
    FROZEN_SAM3_CHECKPOINT_SHA256,
    SAM_HEIGHT,
    SAM_WIDTH,
    _sha256,
    _write_json_noclobber,
    _write_numpy_noclobber,
    _write_png_noclobber,
    _write_torch_noclobber,
    load_signed_field_prompt,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION = (
    REPO_ROOT
    / "paper/artifacts/nvos_joint_box_signed_points_sam3_preregistration_20260817.json"
)
CANDIDATE_ID = "nvos-method-v1-joint-box-signed-points-sam3-v1"


def padded_mask_box_xyxy(mask: np.ndarray, *, padding: int) -> np.ndarray | None:
    """Return a clipped pixel-coordinate XYXY box around a binary mask."""

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError("coarse mask must be two-dimensional")
    rows, columns = np.where(values)
    if rows.size == 0:
        return None
    height, width = values.shape
    pad = int(padding)
    if pad < 0:
        raise ValueError("box padding must be nonnegative")
    return np.asarray(
        [
            max(0, int(columns.min()) - pad),
            max(0, int(rows.min()) - pad),
            min(width, int(columns.max()) + 1 + pad),
            min(height, int(rows.max()) + 1 + pad),
        ],
        dtype=np.float32,
    )


@torch.inference_mode()
def run_joint_trials(
    processor: Any,
    image: Image.Image,
    signed_margin: np.ndarray,
    *,
    device: str,
    padding: int = 16,
) -> dict[str, Any]:
    margin = np.asarray(signed_margin, dtype=np.float32)
    points, labels = deterministic_signed_point_trials(
        np.maximum(margin, 0.0),
        np.maximum(-margin, 0.0),
        image_shape=(image.height, image.width),
        policy=FROZEN_POLICY,
    )
    coarse = resize_mask_nearest(
        margin >= 0.0, (image.height, image.width)
    ).astype(bool)
    box = padded_mask_box_xyxy(coarse, padding=padding)
    if box is None:
        raise ValueError("sealed field prompt has empty nonnegative support")
    amp_dtype = torch.bfloat16 if str(device).startswith("cuda") else None
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
                box=box,
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
    probability = aggregate_sam_trials(trial_masks, policy=FROZEN_POLICY)[0]
    return {
        "probability": probability,
        "continuous_margin": probability - 0.5,
        "binary_mask": probability >= 0.5,
        "trial_masks": trial_masks,
        "point_coordinates_xy": points,
        "point_labels": labels,
        "box_xyxy": box,
        "quality": np.concatenate(qualities),
        "low_resolution_shapes": low_resolution_shapes,
    }


def predict(args: argparse.Namespace) -> dict[str, Any]:
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if (
        preregistration.get("status")
        != "frozen_before_first_joint_box_point_prediction"
        or preregistration.get("candidate_id") != CANDIDATE_ID
    ):
        raise ValueError("joint box-point preregistration differs")
    scene_ids = [str(value) for value in args.scene_ids]
    if len(scene_ids) != 8 or len(set(scene_ids)) != 8:
        raise ValueError("joint box-point prediction requires the frozen full8")
    sources = [
        load_signed_field_prompt(
            dataset_manifest_path=args.manifest,
            prompt_manifest_path=args.signed_field_prompt_manifest,
            method_authority_path=args.method_authority,
            evaluation_contract_path=args.evaluation_contract,
            scene_id=scene,
        )
        for scene in scene_ids
    ]
    output_root = Path(args.output_dir).expanduser().resolve()
    manifest_path = output_root / "prediction_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    checkpoint_sha = _sha256(checkpoint)
    if checkpoint_sha != str(args.expected_checkpoint_sha256):
        raise ValueError("official SAM3 checkpoint SHA-256 differs")
    set_requested_cuda_device(args.device)
    processor = _load_sam3_model(
        checkpoint_path=str(checkpoint),
        device=args.device,
        confidence_threshold=0.0,
        dtype="bfloat16",
        resolution=SAM_WIDTH,
        point_only=True,
        build_on_cpu=True,
    )
    started = time.time()
    predictions: dict[str, dict[str, str]] = {}
    prediction_hashes: dict[str, dict[str, str]] = {}
    receipts: list[dict[str, str]] = []
    for source in sources:
        scene = str(source["scene_id"])
        frame = str(source["frame_id"])
        image = Image.open(source["target_rgb_path"]).convert("RGB")
        original_size = list(image.size)
        image = image.resize((SAM_WIDTH, SAM_HEIGHT), Image.Resampling.LANCZOS)
        result = run_joint_trials(
            processor,
            image,
            source["signed_margin"],
            device=args.device,
            padding=16,
        )
        relative = Path("scores") / scene / f"{frame}.npy"
        score_path = output_root / relative
        trial_path = output_root / "trials" / scene / f"{frame}.pt"
        png_path = output_root / "masks" / scene / f"{frame}.png"
        _write_numpy_noclobber(score_path, result["continuous_margin"])
        _write_torch_noclobber(
            trial_path,
            {
                "trial_masks": torch.from_numpy(result["trial_masks"].copy()),
                "aggregate_probability": torch.from_numpy(result["probability"].copy()),
                "point_coordinates_xy": torch.from_numpy(
                    result["point_coordinates_xy"].copy()
                ),
                "point_labels": torch.from_numpy(result["point_labels"].copy()),
                "box_xyxy": torch.from_numpy(result["box_xyxy"].copy()),
                "quality": torch.from_numpy(result["quality"].copy()),
            },
        )
        _write_png_noclobber(png_path, result["binary_mask"].astype(np.uint8) * 255)
        score_sha = _sha256(score_path)
        predictions[scene] = {frame: relative.as_posix()}
        prediction_hashes[scene] = {frame: score_sha}
        receipt = {
            "schema_version": 1,
            "artifact_type": "radio_gs_nvos_joint_box_signed_points_sam3_receipt",
            "candidate_id": CANDIDATE_ID,
            "method_id": METHOD_ID,
            "scene_id": scene,
            "frame_id": frame,
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
            "joint_prompt": {
                "box_xyxy": result["box_xyxy"].tolist(),
                "box_padding_pixels": 16,
                "trials": 10,
                "positive_points_per_trial": 3,
                "negative_points_per_trial": 3,
                "multimask_output": False,
                "aggregation": "mean_binary_vote",
                "threshold": 0.5,
                "quality": result["quality"].tolist(),
                "low_resolution_shapes": result["low_resolution_shapes"],
            },
            "output": {
                "score": {"path": str(score_path), "sha256": score_sha},
                "trials": {"path": str(trial_path), "sha256": _sha256(trial_path)},
                "mask": {"path": str(png_path), "sha256": _sha256(png_path)},
            },
            "authorities": {
                "dataset_manifest": str(source["dataset_manifest"]),
                "dataset_manifest_sha256": source["dataset_manifest_sha256"],
                "prompt_manifest": str(source["prompt_manifest"]),
                "prompt_manifest_sha256": source["prompt_manifest_sha256"],
                "evaluation_contract": str(source["evaluation_contract"]),
                "evaluation_contract_sha256": source["evaluation_contract_sha256"],
                "preregistration": str(PREREGISTRATION),
                "preregistration_sha256": _sha256(PREREGISTRATION),
                "official_sam3_checkpoint_sha256": checkpoint_sha,
            },
            "safety": {
                "target_rgb_opened": True,
                "target_mask_opened": False,
                "target_metric_opened": False,
                "target_dependent_candidate_selection": False,
            },
        }
        receipt_path = output_root / "receipts" / f"{scene}.json"
        _write_json_noclobber(receipt_path, receipt)
        receipts.append(
            {"scene_id": scene, "path": str(receipt_path), "sha256": _sha256(receipt_path)}
        )
    protocol_hashes = {str(source["protocol_hash"]) for source in sources}
    if len(protocol_hashes) != 1:
        raise ValueError("full8 sources disagree on protocol hash")
    manifest = {
        "schema_version": 1,
        "kind": "promptable_nvs_method_v1_joint_box_signed_points_sam3_predictions",
        "candidate_id": CANDIDATE_ID,
        "protocol_hash": next(iter(protocol_hashes)),
        "scene_order": scene_ids,
        "prediction_root": ".",
        "predictions": predictions,
        "prediction_sha256": prediction_hashes,
        "receipts": receipts,
        "elapsed_seconds": float(time.time() - started),
        "all_eight_predictions_sealed": True,
        "evaluation_performed": False,
        "target_mask_opened": False,
        "target_metric_opened": False,
    }
    _write_json_noclobber(manifest_path, manifest)
    return {**manifest, "prediction_manifest": str(manifest_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--signed-field-prompt-manifest", required=True)
    parser.add_argument("--scene-id", action="append", dest="scene_ids", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-authority", default=str(DEFAULT_METHOD_AUTHORITY))
    parser.add_argument("--evaluation-contract", default=str(DEFAULT_EVALUATION_CONTRACT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_SAM3_CHECKPOINT))
    parser.add_argument(
        "--expected-checkpoint-sha256", default=FROZEN_SAM3_CHECKPOINT_SHA256
    )
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = predict(build_parser().parse_args(argv))
    print(json.dumps({"prediction_manifest": report["prediction_manifest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
