#!/usr/bin/env python3
"""Run the frozen SPIn Method-v1 reference-calibrated transient SAM readout.

Every signed field margin and field receipt is verified before the first RGB
image is opened.  SAM then emits three candidates for each of ten deterministic
signed-point trials.  Candidate index and threshold are selected once on the
single permitted reference mask and transferred unchanged to every target.
Evaluation masks and target metrics are never opened in this stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch

from radio_gs.data.promptable_nvs_manifest import (
    validate_manifest as validate_dataset_manifest,
)
from radio_gs.evaluation.promptable_segmentation import (
    load_ground_truth_mask,
    resize_mask_nearest,
)
from radio_gs.five_benchmark_method_v1 import METHOD_ID, validate_method_authority
from radio_gs.querying.transient_rgb_sam import (
    FROZEN_POLICY,
    PromptMode,
    aggregate_sam_trials,
    calibrate_full_reference_interface,
    deterministic_signed_point_trials,
    transient_adapter_contract,
)
from radio_gs.scripts.build_sam3_foundation_cache import (
    _load_sam3_model,
    sam3_autocast_context,
    set_requested_cuda_device,
)
from radio_gs.scripts.predict_nvos_method_v1_transient_sam import (
    DEFAULT_SAM3_CHECKPOINT,
    FROZEN_SAM3_CHECKPOINT_SHA256,
    SAM_HEIGHT,
    SAM_WIDTH,
)
from radio_gs.scripts.run_spin9_method_v1_scene import (
    DATASET_MANIFEST,
    DEFAULT_RUN_ROOT,
    METHOD_AUTHORITY,
)
from radio_gs.utils.immutable_artifacts import load_json_object, sha256_file


DEFAULT_OUTPUT_ROOT = DEFAULT_RUN_ROOT / "method_v1_readout/transient_sam"
DEFAULT_SIGNED_ROOT = DEFAULT_RUN_ROOT / "method_v1_readout/signed_field"
READOUT_PREREGISTRATION = (
    Path(__file__).resolve().parents[2]
    / "paper/artifacts/spin9_method_v1_transient_readout_preregistration_20260816.json"
)


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _write_numpy(path: Path, value: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.asarray(value, dtype=np.float32), allow_pickle=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _scene_row(rows: Sequence[Mapping[str, Any]], scene_id: str) -> Mapping[str, Any]:
    matches = [row for row in rows if str(row.get("scene_id")) == scene_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one frozen SPIn scene {scene_id!r}")
    return matches[0]


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


def _load_margin(row: Mapping[str, Any], *, label: str) -> np.ndarray:
    path = Path(str(row.get("path", ""))).resolve(strict=True)
    if sha256_file(path) != row.get("sha256"):
        raise ValueError(f"{label} signed margin SHA-256 differs")
    margin = np.load(path, allow_pickle=False)
    if (
        margin.ndim != 2
        or min(margin.shape) <= 0
        or not bool(np.isfinite(margin).all())
    ):
        raise ValueError(f"{label} signed margin must be finite [H,W]")
    return margin.astype(np.float32, copy=False)


def verify_signed_full9_before_rgb(
    *,
    signed_root: Path,
    dataset_manifest: Path = DATASET_MANIFEST,
    method_authority: Path = METHOD_AUTHORITY,
    scene_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Verify requested scalar/field receipts without reading RGB or target masks.

    The default remains the preregistered full Available-Nine barrier.  An
    explicit scene list is a development-only subset barrier and is labelled as
    such in the returned metadata and prediction manifest.
    """

    dataset, dataset_sha, dataset_path = load_json_object(
        dataset_manifest, label="SPIn Available-Nine dataset manifest"
    )
    normalized = validate_dataset_manifest(dataset, check_files=False)
    authority, authority_sha, authority_path = load_json_object(
        method_authority, label="Method-v1 authority"
    )
    validate_method_authority(authority)
    manifest_order = [str(value) for value in dataset["protocol"]["cohort"]]
    normalized_order = [str(row["scene_id"]) for row in normalized["scenes"]]
    method_scenes = [
        str(value) for value in authority["frozen_cohorts"]["spin_nerf_available9"]
    ]
    if normalized_order != manifest_order or set(method_scenes) != set(manifest_order):
        raise ValueError("SPIn Method-v1 and dataset Available-Nine cohorts differ")
    if scene_ids is None:
        requested_order = manifest_order
        development_subset = False
    else:
        requested_order = [str(value) for value in scene_ids]
        if not requested_order or len(requested_order) != len(set(requested_order)):
            raise ValueError("development subset scenes must be non-empty and unique")
        unknown = sorted(set(requested_order) - set(manifest_order))
        if unknown:
            raise ValueError(f"unknown SPIn development subset scenes: {unknown}")
        requested_order = [value for value in manifest_order if value in requested_order]
        development_subset = len(requested_order) != len(manifest_order)
    prereg, prereg_sha, prereg_path = load_json_object(
        READOUT_PREREGISTRATION, label="SPIn Method-v1 readout preregistration"
    )
    if prereg.get("status") != "frozen_before_first_method_v1_spin9_target_readout":
        raise ValueError("SPIn Method-v1 readout preregistration differs")

    normalized_index = {str(row["scene_id"]): row for row in normalized["scenes"]}
    verified: dict[str, Any] = {}
    for scene_id in requested_order:
        receipt_path = (signed_root / "scenes" / scene_id / "receipt.json").resolve(
            strict=True
        )
        receipt, receipt_sha, _source = load_json_object(
            receipt_path, label=f"{scene_id} signed-field receipt"
        )
        scene = normalized_index[scene_id]
        target_ids = [str(value) for value in scene["evaluation_frame_ids"]]
        target_rows = receipt.get("target_scores")
        if (
            receipt.get("artifact_type")
            != "radio_gs_method_v1_spin9_signed_field_receipt"
            or receipt.get("method_id") != METHOD_ID
            or receipt.get("scene_id") != scene_id
            or receipt.get("protocol_hash") != normalized["protocol_hash"]
            or receipt.get("authorities", {}).get("method_sha256") != authority_sha
            or receipt.get("authorities", {}).get("readout_preregistration_sha256")
            != prereg_sha
            or receipt.get("safety", {}).get("target_rgb_opened") is not False
            or receipt.get("safety", {}).get("evaluation_masks_opened") is not False
            or receipt.get("safety", {}).get("target_metrics_opened") is not False
            or not isinstance(target_rows, list)
            or [str(row.get("frame_id")) for row in target_rows] != target_ids
        ):
            raise ValueError(f"{scene_id} signed-field receipt contract differs")
        reference_row = receipt.get("reference_score")
        if not isinstance(reference_row, Mapping):
            raise ValueError(f"{scene_id} reference signed margin is absent")
        _load_margin(reference_row, label=f"{scene_id}/reference")
        target_index: dict[str, Mapping[str, Any]] = {}
        for row in target_rows:
            _load_margin(row, label=f"{scene_id}/{row['frame_id']}")
            target_index[str(row["frame_id"])] = row
        field = Path(str(receipt.get("field", {}).get("path", ""))).resolve(strict=True)
        if sha256_file(field) != receipt.get("field", {}).get("sha256"):
            raise ValueError(f"{scene_id} final Method-v1 field SHA-256 differs")
        verified[scene_id] = {
            "receipt": receipt,
            "receipt_path": receipt_path,
            "receipt_sha256": receipt_sha,
            "reference_score": dict(reference_row),
            "target_scores": target_index,
        }
    return {
        "dataset": dataset,
        "dataset_path": dataset_path,
        "dataset_sha256": dataset_sha,
        "normalized": normalized,
        "method_authority": authority_path,
        "method_authority_sha256": authority_sha,
        "readout_preregistration": prereg_path,
        "readout_preregistration_sha256": prereg_sha,
        "scene_order": requested_order,
        "development_subset": development_subset,
        "verified": verified,
        "all_signed_margins_and_fields_verified_before_first_rgb_open": True,
    }


@torch.inference_mode()
def run_candidate_trials(
    processor: Any,
    image: Image.Image,
    signed_margin: np.ndarray,
    *,
    device: str,
    amp_dtype: torch.dtype | None,
) -> dict[str, Any]:
    margin = np.asarray(signed_margin, dtype=np.float32)
    points, labels = deterministic_signed_point_trials(
        np.maximum(margin, 0.0),
        np.maximum(-margin, 0.0),
        image_shape=(image.height, image.width),
        policy=FROZEN_POLICY,
    )
    with sam3_autocast_context(device, amp_dtype):
        state = processor.set_image(image)
    masks: list[np.ndarray] = []
    qualities: list[np.ndarray] = []
    for trial_points, trial_labels in zip(points, labels):
        with sam3_autocast_context(device, amp_dtype):
            candidate_masks, quality, _low_resolution = processor.model.predict_inst(
                state,
                point_coords=trial_points.astype(np.float32, copy=False),
                point_labels=trial_labels.astype(np.int32, copy=False),
                multimask_output=True,
            )
        candidate_masks = np.asarray(candidate_masks, dtype=np.float32)
        quality = np.asarray(quality, dtype=np.float32).reshape(-1)
        if (
            candidate_masks.shape != (3, image.height, image.width)
            or quality.shape != (3,)
            or not bool(np.isfinite(candidate_masks).all())
            or not bool(np.isfinite(quality).all())
        ):
            raise ValueError(
                f"unexpected official SAM3 multimask output {candidate_masks.shape}"
            )
        masks.append(candidate_masks)
        qualities.append(quality)
    trials = np.stack(masks, axis=0)
    probabilities = aggregate_sam_trials(trials, policy=FROZEN_POLICY)
    if probabilities.shape != (3, image.height, image.width):
        raise RuntimeError("frozen SPIn SAM3 aggregation shape differs")
    return {
        "probabilities": probabilities,
        "points": points,
        "labels": labels,
        "quality": np.stack(qualities, axis=0),
    }


def _validate_existing_scene(
    scene_root: Path, *, scene_id: str, target_ids: Sequence[str]
) -> dict[str, Any] | None:
    receipt_path = scene_root / "receipt.json"
    if not receipt_path.is_file():
        if scene_root.exists():
            raise RuntimeError(
                f"partial transient SAM output must be audited: {scene_root}"
            )
        return None
    receipt, _digest, _source = load_json_object(
        receipt_path, label=f"{scene_id} transient SAM receipt"
    )
    outputs = receipt.get("outputs")
    if (
        receipt.get("artifact_type") != "radio_gs_method_v1_spin9_transient_sam_receipt"
        or receipt.get("method_id") != METHOD_ID
        or receipt.get("scene_id") != scene_id
        or not isinstance(outputs, list)
        or [str(row.get("frame_id")) for row in outputs] != list(target_ids)
        or receipt.get("safety", {}).get("target_mask_opened") is not False
        or receipt.get("safety", {}).get("target_metric_opened") is not False
    ):
        raise ValueError(f"{scene_id} existing transient SAM receipt differs")
    for row in outputs:
        path = Path(str(row["path"])).resolve(strict=True)
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"{scene_id}/{row['frame_id']} prediction SHA-256 differs")
    return receipt


def predict(args: argparse.Namespace) -> dict[str, Any]:
    signed_root = Path(args.signed_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    manifest_path = output_root / "prediction_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    requested_scenes = args.scene_ids if args.scene_ids else None
    barrier = verify_signed_full9_before_rgb(
        signed_root=signed_root, scene_ids=requested_scenes
    )
    output_root.mkdir(parents=True, exist_ok=True)
    set_requested_cuda_device(args.device)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != str(args.expected_checkpoint_sha256):
        raise ValueError("official SAM3 checkpoint SHA-256 differs")
    processor = _load_sam3_model(
        checkpoint_path=str(checkpoint),
        device=args.device,
        confidence_threshold=0.0,
        dtype="bfloat16",
        resolution=SAM_WIDTH,
        point_only=True,
    )
    amp_dtype = torch.bfloat16 if str(args.device).startswith("cuda") else None
    raw_index = {str(row["scene_id"]): row for row in barrier["dataset"]["scenes"]}
    normalized_index = {
        str(row["scene_id"]): row for row in barrier["normalized"]["scenes"]
    }
    started = time.time()

    for scene_id in barrier["scene_order"]:
        raw_scene = raw_index[scene_id]
        normalized_scene = normalized_index[scene_id]
        target_ids = [str(value) for value in normalized_scene["evaluation_frame_ids"]]
        final_scene_root = output_root / "scenes" / scene_id
        if (
            _validate_existing_scene(
                final_scene_root, scene_id=scene_id, target_ids=target_ids
            )
            is not None
        ):
            continue
        signed = barrier["verified"][scene_id]
        reference_id = str(signed["receipt"]["reference_frame_id"])
        reference_frame = _raw_frame(raw_scene, reference_id)
        reference_rgb = Path(str(reference_frame["rgb_path"])).resolve(strict=True)
        reference_mask_path = Path(str(raw_scene["prompt"]["mask_path"])).resolve(
            strict=True
        )
        reference_margin = _load_margin(
            signed["reference_score"], label=f"{scene_id}/reference"
        )
        reference_image = Image.open(reference_rgb).convert("RGB")
        reference_original_size = list(reference_image.size)
        reference_image = reference_image.resize(
            (SAM_WIDTH, SAM_HEIGHT), Image.Resampling.LANCZOS
        )
        reference_result = run_candidate_trials(
            processor,
            reference_image,
            reference_margin,
            device=args.device,
            amp_dtype=amp_dtype,
        )
        reference_mask = load_ground_truth_mask(reference_mask_path).astype(bool)
        if reference_mask.shape != (SAM_HEIGHT, SAM_WIDTH):
            reference_mask = resize_mask_nearest(
                reference_mask, (SAM_HEIGHT, SAM_WIDTH)
            ).astype(bool)
        calibration = calibrate_full_reference_interface(
            reference_result["probabilities"],
            reference_mask,
            allow_canonical_fallback=False,
            policy=FROZEN_POLICY,
        )
        if calibration.branch != "sam" or calibration.candidate_index not in (0, 1, 2):
            raise RuntimeError("SPIn reference calibration selected an invalid branch")

        staging = Path(tempfile.mkdtemp(prefix=f".{scene_id}.", dir=output_root))
        try:
            reference_candidates_final = final_scene_root / "reference_candidates.npy"
            reference_candidates_sha = _write_numpy(
                staging / "reference_candidates.npy",
                reference_result["probabilities"],
            )
            output_rows: list[dict[str, Any]] = []
            for frame_id in target_ids:
                frame = _raw_frame(raw_scene, frame_id)
                target_rgb = Path(str(frame["rgb_path"])).resolve(strict=True)
                target_margin_row = signed["target_scores"][frame_id]
                target_margin = _load_margin(
                    target_margin_row, label=f"{scene_id}/{frame_id}"
                )
                image = Image.open(target_rgb).convert("RGB")
                original_size = list(image.size)
                image = image.resize((SAM_WIDTH, SAM_HEIGHT), Image.Resampling.LANCZOS)
                result = run_candidate_trials(
                    processor,
                    image,
                    target_margin,
                    device=args.device,
                    amp_dtype=amp_dtype,
                )
                selected = result["probabilities"][calibration.candidate_index]
                continuous_margin = selected - float(calibration.threshold)
                relative = Path("scores") / f"{frame_id}.npy"
                score_sha = _write_numpy(staging / relative, continuous_margin)
                output_rows.append(
                    {
                        "frame_id": frame_id,
                        "path": str(final_scene_root / relative),
                        "sha256": score_sha,
                        "shape": list(continuous_margin.shape),
                        "target_rgb": str(target_rgb),
                        "target_rgb_sha256": sha256_file(target_rgb),
                        "target_rgb_original_size_wh": original_size,
                        "signed_field_margin": str(target_margin_row["path"]),
                        "signed_field_margin_sha256": target_margin_row["sha256"],
                        "point_coordinates_sha256": _sha256_array(result["points"]),
                        "point_labels_sha256": _sha256_array(result["labels"]),
                        "quality": result["quality"].tolist(),
                        "foreground_fraction": float((continuous_margin >= 0.0).mean()),
                    }
                )
            receipt = {
                "schema_version": 1,
                "artifact_type": "radio_gs_method_v1_spin9_transient_sam_receipt",
                "method_id": METHOD_ID,
                "scene_id": scene_id,
                "protocol_hash": barrier["normalized"]["protocol_hash"],
                "signed_field_receipt": str(signed["receipt_path"]),
                "signed_field_receipt_sha256": signed["receipt_sha256"],
                "field": signed["receipt"]["field"],
                "reference": {
                    "frame_id": reference_id,
                    "rgb": str(reference_rgb),
                    "rgb_sha256": sha256_file(reference_rgb),
                    "rgb_original_size_wh": reference_original_size,
                    "mask": str(reference_mask_path),
                    "mask_sha256": sha256_file(reference_mask_path),
                    "candidate_probabilities": str(reference_candidates_final),
                    "candidate_probabilities_sha256": reference_candidates_sha,
                    "selected_candidate": calibration.candidate_index,
                    "selected_threshold": calibration.threshold,
                    "selected_reference_iou": calibration.reference_iou,
                    "point_coordinates_sha256": _sha256_array(
                        reference_result["points"]
                    ),
                    "point_labels_sha256": _sha256_array(reference_result["labels"]),
                    "quality": reference_result["quality"].tolist(),
                },
                "outputs": output_rows,
                "policy": transient_adapter_contract(PromptMode.FULL_REFERENCE_MASK),
                "candidate_policy": {
                    "multimask_output": True,
                    "candidate_count": 3,
                    "candidate_aggregation": "candidate_wise_mean_over_ten_trials",
                    "reference_only_calibration": True,
                    "canonical_branch_fallback": False,
                    "target_transfer": "fixed_candidate_and_threshold",
                    "continuous_margin": "selected_candidate_probability_minus_reference_selected_threshold",
                },
                "authorities": {
                    "dataset_manifest": str(barrier["dataset_path"]),
                    "dataset_manifest_sha256": barrier["dataset_sha256"],
                    "method_authority": str(barrier["method_authority"]),
                    "method_authority_sha256": barrier["method_authority_sha256"],
                    "readout_preregistration": str(barrier["readout_preregistration"]),
                    "readout_preregistration_sha256": barrier[
                        "readout_preregistration_sha256"
                    ],
                    "official_sam3_checkpoint": str(checkpoint),
                    "official_sam3_checkpoint_sha256": checkpoint_sha,
                },
                "safety": {
                    "all_signed_margins_and_fields_verified_before_first_rgb_open": True,
                    "reference_mask_opened": True,
                    "target_rgb_opened": True,
                    "target_mask_opened": False,
                    "target_metric_opened": False,
                    "reference_mask_selection": True,
                    "target_metric_used_for_selection": False,
                    "graph_used": False,
                    "connected_component_used": False,
                },
            }
            _write_json(staging / "receipt.json", receipt)
            final_scene_root.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging, final_scene_root)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    predictions: dict[str, dict[str, str]] = {}
    prediction_hashes: dict[str, dict[str, str]] = {}
    receipts: list[dict[str, str]] = []
    for scene_id in barrier["scene_order"]:
        target_ids = [
            str(value) for value in normalized_index[scene_id]["evaluation_frame_ids"]
        ]
        scene_root = output_root / "scenes" / scene_id
        receipt = _validate_existing_scene(
            scene_root, scene_id=scene_id, target_ids=target_ids
        )
        assert receipt is not None
        by_frame = {str(row["frame_id"]): row for row in receipt["outputs"]}
        predictions[scene_id] = {
            frame_id: Path(by_frame[frame_id]["path"])
            .relative_to(output_root)
            .as_posix()
            for frame_id in target_ids
        }
        prediction_hashes[scene_id] = {
            frame_id: str(by_frame[frame_id]["sha256"]) for frame_id in target_ids
        }
        receipt_path = scene_root / "receipt.json"
        receipts.append(
            {
                "scene_id": scene_id,
                "path": str(receipt_path),
                "sha256": sha256_file(receipt_path),
            }
        )
    manifest = {
        "schema_version": 1,
        "kind": "promptable_nvs_method_v1_spin9_transient_sam_predictions",
        "method_id": METHOD_ID,
        "protocol_hash": barrier["normalized"]["protocol_hash"],
        "prediction_root": ".",
        "predictions": predictions,
        "prediction_sha256": prediction_hashes,
        "receipts": receipts,
        "method": {
            "operator": "signed_field_prompt_to_query_transient_target_rgb_frozen_sam",
            "trials": FROZEN_POLICY.trials,
            "positive_points_per_trial": FROZEN_POLICY.positive_points_per_trial,
            "negative_points_per_trial": FROZEN_POLICY.negative_points_per_trial,
            "multimask_output": True,
            "reference_mask_selection": True,
            "graph_or_connected_component": False,
            "score_semantics": "selected_candidate_probability_minus_reference_selected_threshold",
            "evaluator_threshold": 0.0,
        },
        "scene_order": barrier["scene_order"],
        "elapsed_seconds": float(time.time() - started),
        "evaluation_performed": False,
        "target_mask_opened": False,
        "target_metric_opened": False,
        "development_subset": barrier["development_subset"],
        "all_nine_scene_predictions_sealed": not barrier["development_subset"],
    }
    _write_json(manifest_path, manifest)
    return {**manifest, "prediction_manifest": str(manifest_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signed-root", default=str(DEFAULT_SIGNED_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_SAM3_CHECKPOINT))
    parser.add_argument(
        "--expected-checkpoint-sha256", default=FROZEN_SAM3_CHECKPOINT_SHA256
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--scene",
        dest="scene_ids",
        action="append",
        help=(
            "Explicit development-only subset scene (repeatable). The default "
            "retains the preregistered full Available-Nine barrier."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = predict(build_parser().parse_args(argv))
    print(
        json.dumps(
            {
                "prediction_manifest": report["prediction_manifest"],
                "scene_count": len(report["predictions"]),
                "evaluation_performed": False,
                "target_mask_opened": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
