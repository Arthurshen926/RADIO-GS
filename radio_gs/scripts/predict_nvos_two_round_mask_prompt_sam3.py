#!/usr/bin/env python3
"""Run fixed NVOS round-two official SAM3 mask self-prompts.

The first-round field/box/point observations have already been lifted with an
exact registered-view compositor and rerendered by the companion consensus
stage.  This stage opens only the contract-authorized target RGB, turns the
sealed rerender into the checkpoint-authoritative SAM mask-logit input, and combines
that mask prompt with the same deterministic signed field points and a fixed
padding-16 rerender box.  Ten single-candidate trials are averaged; no target
mask, target metric, or scene-specific choice is available before sealing all
requested predictions.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

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
    load_signed_field_prompt,
)


CANDIDATE_ID = "nvos-method-v1-two-round-exact-logodds-sam3-v1"
PROBABILITY_EPSILON = 0.05
MASK_INPUT_SIZE = (288, 288)
BOX_PADDING_PIXELS = 16
ROUND2_MODE = "official_sam3_mask_logits_plus_signed_points_plus_pad16_box_ten_trial_mean"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_numpy(path: Path, value: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.asarray(value, dtype=np.float32), allow_pickle=False)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(path)


def _prediction_path(
    manifest: Mapping[str, Any], manifest_path: Path, scene: str, frame: str
) -> Path:
    relative = Path(str(manifest["predictions"][scene][frame]))
    root = Path(str(manifest.get("prediction_root", ".")))
    if not root.is_absolute():
        root = manifest_path.parent / root
    source = (relative if relative.is_absolute() else root / relative).resolve(strict=True)
    if _sha256(source) != str(manifest["prediction_sha256"][scene][frame]):
        raise ValueError(f"sealed prediction SHA-256 differs: {scene}/{frame}")
    return source


def rerender_mask_logits(
    probability: np.ndarray,
    *,
    size: int | tuple[int, int] = MASK_INPUT_SIZE,
    epsilon: float = PROBABILITY_EPSILON,
) -> np.ndarray:
    """Convert a full-resolution rerender posterior into official mask logits."""

    value = torch.from_numpy(np.asarray(probability, dtype=np.float32)).view(
        1, 1, *probability.shape
    )
    if value.ndim != 4 or min(value.shape[-2:]) <= 0:
        raise ValueError("rerender probability must be a nonempty [H,W] map")
    if not bool(torch.isfinite(value).all()) or bool(((value < 0) | (value > 1)).any()):
        raise ValueError("rerender probability must be finite in [0,1]")
    output_size = (int(size), int(size)) if isinstance(size, int) else tuple(map(int, size))
    if len(output_size) != 2 or min(output_size) <= 0:
        raise ValueError("official mask input size must contain two positive values")
    resized = F.interpolate(value, size=output_size, mode="bilinear", align_corners=False)
    logits = torch.logit(resized.clamp(float(epsilon), 1.0 - float(epsilon)))
    return logits[0].numpy().astype(np.float32, copy=False)


def probability_box_xyxy(
    probability: np.ndarray,
    *,
    threshold: float = 0.5,
    padding_pixels: int = BOX_PADDING_PIXELS,
) -> np.ndarray | None:
    """Return a fixed padded absolute XYXY box around a rerender posterior."""

    value = np.asarray(probability, dtype=np.float32)
    if value.ndim != 2 or not bool(np.isfinite(value).all()):
        raise ValueError("rerender probability must be a finite [H,W] map")
    rows, columns = np.nonzero(value >= float(threshold))
    if not rows.size:
        return None
    height, width = value.shape
    x0 = max(0, int(columns.min()) - int(padding_pixels))
    y0 = max(0, int(rows.min()) - int(padding_pixels))
    x1 = min(width - 1, int(columns.max()) + int(padding_pixels))
    y1 = min(height - 1, int(rows.max()) + int(padding_pixels))
    return np.asarray([x0, y0, x1, y1], dtype=np.float32)


def _load_consensus_manifest(path: str | Path, expected_sha256: str) -> tuple[dict[str, Any], Path]:
    source = Path(path).expanduser().resolve(strict=True)
    if len(str(expected_sha256)) != 64 or _sha256(source) != str(expected_sha256):
        raise ValueError("exact consensus manifest SHA-256 differs")
    value = json.loads(source.read_text(encoding="utf-8"))
    if (
        value.get("kind") != "promptable_nvs_method_v1_two_round_exact_consensus_rerender"
        or value.get("candidate_id") != CANDIDATE_ID
        or bool(value.get("target_rgb_opened", True))
        or bool(value.get("target_mask_opened", True))
        or bool(value.get("target_metric_opened", True))
    ):
        raise ValueError("exact consensus manifest contract differs")
    return value, source


@torch.inference_mode()
def run_round2_trials(
    processor: Any,
    image: Image.Image,
    signed_margin: np.ndarray,
    rerender_probability: np.ndarray,
    *,
    device: str,
    amp_dtype: torch.dtype | None,
) -> dict[str, Any]:
    """Execute the frozen official SAM3 round-two prompt exactly ten times."""

    if (image.height, image.width) != (SAM_HEIGHT, SAM_WIDTH):
        raise ValueError("round-two image differs from frozen SAM raster")
    probability = np.asarray(rerender_probability, dtype=np.float32)
    if probability.shape != (SAM_HEIGHT, SAM_WIDTH):
        raise ValueError("round-two rerender differs from frozen SAM raster")
    margin = np.asarray(signed_margin, dtype=np.float32)
    points, labels = deterministic_signed_point_trials(
        np.maximum(margin, 0.0),
        np.maximum(-margin, 0.0),
        image_shape=(image.height, image.width),
        policy=FROZEN_POLICY,
    )
    prompt_encoder = processor.model.inst_interactive_predictor.model.sam_prompt_encoder
    official_mask_input_size = tuple(map(int, prompt_encoder.mask_input_size))
    if official_mask_input_size != MASK_INPUT_SIZE:
        raise ValueError(
            "official SAM3 prompt-encoder mask input size differs from frozen checkpoint authority"
        )
    mask_input = rerender_mask_logits(probability, size=official_mask_input_size)
    box = probability_box_xyxy(probability)
    if box is None:
        raise ValueError("fixed rerender self-prompt is empty")
    with sam3_autocast_context(device, amp_dtype):
        state = processor.set_image(image)
    masks: list[np.ndarray] = []
    qualities: list[float] = []
    low_resolution_shapes: list[list[int]] = []
    for trial_points, trial_labels in zip(points, labels):
        with sam3_autocast_context(device, amp_dtype):
            candidate_masks, quality, low_resolution = processor.model.predict_inst(
                state,
                point_coords=trial_points.astype(np.float32, copy=False),
                point_labels=trial_labels.astype(np.int32, copy=False),
                box=box,
                mask_input=mask_input,
                multimask_output=False,
            )
        candidate = np.asarray(candidate_masks)
        score = np.asarray(quality, dtype=np.float32).reshape(-1)
        low_resolution = np.asarray(low_resolution)
        if candidate.shape != (1, SAM_HEIGHT, SAM_WIDTH):
            raise ValueError(f"unexpected official round-two SAM3 shape: {candidate.shape}")
        if score.shape != (1,) or not bool(np.isfinite(score).all()):
            raise ValueError("official round-two SAM3 quality output differs")
        if low_resolution.shape != (1, *MASK_INPUT_SIZE):
            raise ValueError("official round-two SAM3 low-resolution output differs")
        masks.append(candidate.astype(np.float32, copy=False))
        qualities.append(float(score[0]))
        low_resolution_shapes.append(list(low_resolution.shape))
    trial_masks = np.stack(masks, axis=0)
    output_probability = aggregate_sam_trials(trial_masks, policy=FROZEN_POLICY)[0]
    return {
        "probability": output_probability.astype(np.float32, copy=False),
        "continuous_margin": (
            output_probability - float(FROZEN_POLICY.signed_vote_threshold)
        ).astype(np.float32, copy=False),
        "trial_masks": trial_masks,
        "quality": qualities,
        "point_coordinates_xy": points,
        "point_labels": labels,
        "mask_input": mask_input,
        "box_xyxy": box,
        "low_resolution_shapes": low_resolution_shapes,
    }


def _write_torch(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(value), temporary)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(path)


def predict(args: argparse.Namespace) -> dict[str, Any]:
    scene_ids = [str(value) for value in args.scene_ids]
    if not scene_ids or len(scene_ids) != len(set(scene_ids)):
        raise ValueError("--scene-id must be a nonempty unique list")
    consensus, consensus_path = _load_consensus_manifest(
        args.consensus_manifest, args.expected_consensus_manifest_sha256
    )
    if scene_ids != [str(value) for value in consensus.get("scene_order", [])]:
        raise ValueError("round-two requested scene order differs from exact consensus")
    checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    checkpoint_sha = _sha256(checkpoint)
    if checkpoint_sha != str(args.expected_checkpoint_sha256):
        raise ValueError("official SAM3 checkpoint SHA-256 differs")
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
    if {str(source["protocol_hash"]) for source in sources} != {
        str(consensus.get("protocol_hash"))
    }:
        raise ValueError("round-two sources disagree with consensus protocol hash")
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
    amp_dtype = torch.bfloat16 if str(args.device).startswith("cuda") else None
    predictions: dict[str, dict[str, str]] = {}
    prediction_hashes: dict[str, dict[str, str]] = {}
    receipts: list[dict[str, str]] = []
    started = time.time()
    for source in sources:
        scene_id = str(source["scene_id"])
        frame_id = str(source["frame_id"])
        rerender_path = _prediction_path(consensus, consensus_path, scene_id, frame_id)
        rerender = np.load(rerender_path, allow_pickle=False)
        target = Image.open(source["target_rgb_path"]).convert("RGB")
        original_size = list(target.size)
        target = target.resize((SAM_WIDTH, SAM_HEIGHT), Image.Resampling.LANCZOS)
        result = run_round2_trials(
            processor,
            target,
            source["signed_margin"],
            rerender,
            device=args.device,
            amp_dtype=amp_dtype,
        )
        relative = Path("scores") / scene_id / f"{frame_id}.npy"
        score_path = output_root / relative
        trial_path = output_root / "trials" / scene_id / f"{frame_id}.pt"
        score_sha = _write_numpy(score_path, result["continuous_margin"])
        trial_sha = _write_torch(
            trial_path,
            {
                "trial_masks": torch.from_numpy(result["trial_masks"].copy()),
                "aggregate_probability": torch.from_numpy(result["probability"].copy()),
                "continuous_margin": torch.from_numpy(result["continuous_margin"].copy()),
                "point_coordinates_xy": torch.from_numpy(result["point_coordinates_xy"].copy()),
                "point_labels": torch.from_numpy(result["point_labels"].copy()),
                "mask_input": torch.from_numpy(result["mask_input"].copy()),
                "box_xyxy": torch.from_numpy(result["box_xyxy"].copy()),
                "quality": torch.tensor(result["quality"], dtype=torch.float32),
            },
        )
        receipt = {
            "schema_version": 1,
            "artifact_type": "radio_gs_nvos_two_round_mask_prompt_sam3_receipt",
            "candidate_id": CANDIDATE_ID,
            "method_id": METHOD_ID,
            "scene_id": scene_id,
            "frame_id": frame_id,
            "exact_consensus_rerender": {"path": str(rerender_path), "sha256": _sha256(rerender_path)},
            "signed_field_prompt": {
                "path": str(source["signed_margin_path"]),
                "sha256": source["signed_margin_sha256"],
            },
            "target_rgb": {
                "path": str(source["target_rgb_path"]),
                "sha256": source["target_rgb_sha256"],
                "original_size_wh": original_size,
                "sam_size_wh": [SAM_WIDTH, SAM_HEIGHT],
            },
            "round2": {
                "mode": ROUND2_MODE,
                "mask_input_size": list(MASK_INPUT_SIZE),
                "mask_input_probability_epsilon": PROBABILITY_EPSILON,
                "box_padding_pixels": BOX_PADDING_PIXELS,
                "box_xyxy": result["box_xyxy"].tolist(),
                "trials": FROZEN_POLICY.trials,
                "positive_points_per_trial": FROZEN_POLICY.positive_points_per_trial,
                "negative_points_per_trial": FROZEN_POLICY.negative_points_per_trial,
                "multimask_output": False,
                "candidate_selection": "none_exactly_one_candidate_per_trial",
                "aggregation": "mean_of_ten_official_binary_masks",
                "quality": result["quality"],
                "low_resolution_shapes": result["low_resolution_shapes"],
            },
            "output": {
                "continuous_margin": {"path": str(score_path), "sha256": score_sha},
                "trial_artifact": {"path": str(trial_path), "sha256": trial_sha},
                "foreground_pixels": int((result["continuous_margin"] >= 0).sum()),
                "foreground_fraction": float((result["continuous_margin"] >= 0).mean()),
            },
            "authorities": {
                "consensus_manifest": str(consensus_path),
                "consensus_manifest_sha256": args.expected_consensus_manifest_sha256,
                "dataset_manifest": str(source["dataset_manifest"]),
                "dataset_manifest_sha256": source["dataset_manifest_sha256"],
                "protocol_hash": source["protocol_hash"],
                "evaluation_contract": str(source["evaluation_contract"]),
                "evaluation_contract_sha256": source["evaluation_contract_sha256"],
                "evaluation_contract_id": source["evaluation_contract_id"],
                "official_sam3_checkpoint": str(checkpoint),
                "official_sam3_checkpoint_sha256": checkpoint_sha,
            },
            "safety": {
                "target_rgb_opened": True,
                "target_mask_opened": False,
                "target_metric_opened": False,
                "target_metric_used_for_selection": False,
                "scene_specific_parameter": False,
                "graph_used": False,
                "connected_component_used": False,
            },
        }
        receipt_path = output_root / "receipts" / f"{scene_id}.json"
        _write_json(receipt_path, receipt)
        predictions[scene_id] = {frame_id: relative.as_posix()}
        prediction_hashes[scene_id] = {frame_id: score_sha}
        receipts.append(
            {"scene_id": scene_id, "path": str(receipt_path), "sha256": _sha256(receipt_path)}
        )

    manifest = {
        "schema_version": 1,
        "kind": "promptable_nvs_method_v1_two_round_mask_prompt_sam3_predictions",
        "candidate_id": CANDIDATE_ID,
        "method_id": METHOD_ID,
        "protocol_hash": consensus["protocol_hash"],
        "scene_order": scene_ids,
        "prediction_root": ".",
        "predictions": predictions,
        "prediction_sha256": prediction_hashes,
        "receipts": receipts,
        "exact_consensus_manifest": {
            "path": str(consensus_path),
            "sha256": args.expected_consensus_manifest_sha256,
        },
        "method": {
            "round1": "sealed_field_box_and_point_sam",
            "transport": "exact_W_transpose_visibility_logodds_consensus_and_same_W_rerender",
            "round2": ROUND2_MODE,
            "threshold": {"mode": "fixed", "value": 0.0},
        },
        "elapsed_seconds": float(time.time() - started),
        "all_requested_predictions_sealed": True,
        "evaluation_performed": False,
        "target_rgb_opened": True,
        "target_mask_opened": False,
        "target_metric_opened": False,
    }
    _write_json(manifest_path, manifest)
    return {**manifest, "prediction_manifest_path": str(manifest_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--signed-field-prompt-manifest", required=True)
    parser.add_argument("--consensus-manifest", required=True)
    parser.add_argument("--expected-consensus-manifest-sha256", required=True)
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
    print(
        json.dumps(
            {
                "prediction_manifest": report["prediction_manifest_path"],
                "scenes": report["scene_order"],
                "evaluation_performed": False,
                "target_mask_opened": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
