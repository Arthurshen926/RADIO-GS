#!/usr/bin/env python3
"""Propagate the official NVOS scribble with official SAM3 video memory.

This is an explicitly RGB-assisted development readout.  It consumes only the
content-bound dataset manifest, the official prompt RGB/scribbles, registered
RGB frames, and the frozen SAM3 checkpoint.  The target annotation is not
opened until the prediction receipt has been sealed by the separate scorer.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch

from radio_gs.scripts.predict_nvos_sam3_video_from_registered_prompt import (
    RECEIPT_TYPE,
    _atomic_json,
    _load_json,
    _scene,
    sample_signed_points,
    sha256_file,
    shorter_cyclic_path,
)
from radio_gs.scripts.build_sam3_foundation_cache import (
    _load_sam3_model,
    sam3_autocast_context,
    set_requested_cuda_device,
)
from radio_gs.scripts.refine_lerf_coarse_receipt_official_sam3 import mask_to_box


def load_binary_scribble(path: Path) -> np.ndarray:
    values = np.asarray(Image.open(path))
    if values.ndim == 3:
        values = np.any(values > 127, axis=2)
    elif values.ndim == 2:
        values = values > 127
    else:
        raise ValueError(f"unsupported scribble shape: {values.shape}")
    return np.asarray(values, dtype=np.float32)


def registered_rows(scene: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_index: dict[int, dict[str, Any]] = {}
    for row in [*scene.get("training_frames", []), *scene.get("frames", [])]:
        index = int(row["rgb_sorted_index"])
        normalized = {
            "frame_id": str(row["frame_id"]),
            "rgb_path": str(Path(row["rgb_path"]).resolve(strict=True)),
            "rgb_sorted_index": index,
        }
        previous = by_index.get(index)
        if previous is not None and previous != normalized:
            raise ValueError("registered RGB index is ambiguous")
        by_index[index] = normalized
    rows = [by_index[index] for index in sorted(by_index)]
    if [row["rgb_sorted_index"] for row in rows] != list(range(len(rows))):
        raise ValueError("registered RGB cohort is not contiguous")
    if len({row["frame_id"] for row in rows}) != len(rows):
        raise ValueError("registered RGB frame identity is ambiguous")
    return rows


def scribble_adherence(
    mask: np.ndarray, positive: np.ndarray, negative: np.ndarray
) -> dict[str, float]:
    if positive.shape != negative.shape or positive.ndim != 2:
        raise ValueError("signed scribble shapes differ")
    binary = np.asarray(mask).astype(bool)
    if binary.ndim != 2:
        raise ValueError("prompt mask must be two-dimensional")
    resized = np.asarray(
        Image.fromarray(binary.astype(np.uint8)).resize(
            (positive.shape[1], positive.shape[0]), resample=Image.Resampling.NEAREST
        )
    ).astype(bool)
    positive_rows = positive > 0.5
    negative_rows = negative > 0.5
    if not bool(positive_rows.any()) or not bool(negative_rows.any()):
        raise ValueError("official signed scribble is empty")
    positive_recall = float(resized[positive_rows].mean())
    negative_rejection = float((~resized[negative_rows]).mean())
    return {
        "positive_recall": positive_recall,
        "negative_rejection": negative_rejection,
        "minimum_signed_adherence": min(positive_recall, negative_rejection),
    }


def choose_prompt_proposal(
    masks: np.ndarray,
    scores: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    candidates = np.asarray(masks)
    if candidates.ndim == 4 and candidates.shape[1] == 1:
        candidates = candidates[:, 0]
    if candidates.ndim == 2:
        candidates = candidates[None]
    if candidates.ndim != 3 or candidates.shape[1:] != positive.shape:
        raise ValueError("prompt proposal shapes differ")
    quality = np.asarray(scores, dtype=np.float32).reshape(-1)
    if quality.shape != (len(candidates),):
        quality = np.zeros(len(candidates), dtype=np.float32)
    best: tuple[float, float, int, int] | None = None
    best_report: dict[str, float | int] | None = None
    for index, values in enumerate(candidates):
        candidate = np.asarray(values) > 0
        report = scribble_adherence(candidate, positive, negative)
        proposal = (
            float(report["minimum_signed_adherence"]),
            float(quality[index]),
            int(candidate.sum()),
            -index,
        )
        if best is None or proposal > best:
            best = proposal
            best_report = {
                **report,
                "selected_index": index,
                "selected_score": float(quality[index]),
                "selected_pixels": int(candidate.sum()),
                "candidate_count": int(len(candidates)),
            }
    if best_report is None:
        raise ValueError("official SAM3 returned no prompt proposal")
    return np.asarray(candidates[int(best_report["selected_index"])]) > 0, best_report


def predict(args: argparse.Namespace) -> dict[str, Any]:
    from sam3.model_builder import build_sam3_video_model

    manifest_path = Path(args.manifest).expanduser().resolve(strict=True)
    checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    if sha256_file(manifest_path) != str(args.expected_manifest_sha256):
        raise ValueError("dataset manifest hash differs")
    if sha256_file(checkpoint) != str(args.expected_checkpoint_sha256):
        raise ValueError("SAM3 checkpoint hash differs")
    manifest = _load_json(manifest_path)
    scene_id = str(args.scene)
    scene = _scene(manifest, scene_id)
    prompt_id = str(scene["prompt"]["frame_id"])
    target_ids = [str(value) for value in scene.get("evaluation_frame_ids", [])]
    if len(target_ids) != 1:
        raise ValueError("NVOS video readout requires one target frame")
    target_id = target_ids[0]

    rows = registered_rows(scene)
    frame_ids = [row["frame_id"] for row in rows]
    path_indices = shorter_cyclic_path(
        len(rows), frame_ids.index(prompt_id), frame_ids.index(target_id)
    )
    image_paths = [Path(rows[index]["rgb_path"]) for index in path_indices]

    positive_path = Path(scene["prompt"]["positive_path"]).resolve(strict=True)
    negative_path = Path(scene["prompt"]["negative_path"]).resolve(strict=True)
    positive = load_binary_scribble(positive_path)
    negative = load_binary_scribble(negative_path)
    if positive.shape != negative.shape:
        raise ValueError("official signed scribble shapes differ")
    points, labels = sample_signed_points(
        positive, negative, int(args.points_per_sign)
    )

    prompt_proposal_report: dict[str, float | int] | None = None
    seed_box: np.ndarray | None = None
    if args.prompt_proposal_seed:
        set_requested_cuda_device(args.device)
        processor = _load_sam3_model(
            checkpoint_path=str(checkpoint),
            device=args.device,
            confidence_threshold=0.0,
            dtype="float32",
            resolution=positive.shape[1],
            point_only=False,
            build_on_cpu=True,
        )
        prompt_image = Image.open(image_paths[0]).convert("RGB").resize(
            (positive.shape[1], positive.shape[0]), Image.Resampling.LANCZOS
        )
        positive_box = mask_to_box(positive > 0.5, padding_pixels=16)
        if positive_box is None:
            raise ValueError("official positive scribble has no box")
        amp_dtype = torch.bfloat16 if str(args.device).startswith("cuda") else None
        with sam3_autocast_context(args.device, amp_dtype):
            image_state = processor.set_image(prompt_image)
            proposal_output = processor.add_geometric_prompt(
                positive_box, True, dict(image_state)
            )
        proposal_masks = proposal_output.get("masks")
        if proposal_masks is None:
            logits = proposal_output.get("masks_logits")
            proposal_masks = None if logits is None else logits.float() > 0.0
        if proposal_masks is None:
            raise RuntimeError("official SAM3 returned no prompt proposal masks")
        proposal_scores = proposal_output.get("scores")
        prompt_proposal, prompt_proposal_report = choose_prompt_proposal(
            proposal_masks.detach().cpu().numpy()
            if torch.is_tensor(proposal_masks)
            else np.asarray(proposal_masks),
            proposal_scores.detach().float().cpu().numpy()
            if torch.is_tensor(proposal_scores)
            else np.asarray(proposal_scores if proposal_scores is not None else []),
            positive,
            negative,
        )
        proposal_box = mask_to_box(prompt_proposal, padding_pixels=0)
        if proposal_box is None:
            raise RuntimeError("selected prompt proposal is empty")
        cx, cy, width, height = [float(value) for value in proposal_box]
        seed_box = np.asarray(
            [[cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2]],
            dtype=np.float32,
        )
        del processor, image_state, proposal_output, proposal_masks, prompt_image
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    target_mask: np.ndarray | None = None
    target_score: float | None = None
    with tempfile.TemporaryDirectory(prefix=f"radio_gs_nvos_{scene_id}_official_video_") as temporary:
        frame_root = Path(temporary)
        for index, path in enumerate(image_paths):
            os.symlink(path, frame_root / f"{index:05d}.jpg")
        model = build_sam3_video_model(
            checkpoint_path=str(checkpoint),
            load_from_HF=False,
            strict_state_dict_loading=True,
            apply_temporal_disambiguation=True,
            device=args.device,
            compile=False,
        )
        tracker = model.tracker
        tracker.backbone = model.detector.backbone
        state = tracker.init_state(video_path=str(frame_root))
        _prompt_frame, prompt_object_ids, _prompt_low, prompt_masks = tracker.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=1,
            points=points,
            labels=labels,
            box=seed_box,
        )
        prompt_positions = [
            i for i, value in enumerate(prompt_object_ids) if int(value) == 1
        ]
        if len(prompt_positions) != 1:
            raise RuntimeError("prompt NVOS object identity differs")
        prompt_mask = (
            prompt_masks[prompt_positions[0]].detach().float().cpu().squeeze().numpy()
            > 0.0
        )
        prompt_adherence = scribble_adherence(prompt_mask, positive, negative)
        for frame_index, object_ids, _low, masks, object_scores in tracker.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=len(image_paths),
            reverse=False,
            propagate_preflight=True,
        ):
            if int(frame_index) != len(image_paths) - 1:
                continue
            positions = [i for i, value in enumerate(object_ids) if int(value) == 1]
            if len(positions) != 1:
                raise RuntimeError("tracked NVOS object identity differs")
            position = positions[0]
            target_mask = masks[position].detach().float().cpu().squeeze().numpy() > 0.0
            target_score = float(torch.as_tensor(object_scores[position]).detach().cpu())
        if hasattr(tracker, "reset_state"):
            tracker.reset_state(state)
        del state, tracker, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if target_mask is None:
        raise RuntimeError("official SAM3 video tracker returned no target mask")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "target_probability.npy"
    temporary_prediction = output_dir / f".target_probability.{os.getpid()}.npy"
    with temporary_prediction.open("wb") as handle:
        np.save(handle, target_mask.astype(np.float32), allow_pickle=False)
    os.replace(temporary_prediction, prediction_path)
    receipt = {
        "schema_version": 1,
        "artifact_type": RECEIPT_TYPE,
        "scene_id": scene_id,
        "target_frame_id": target_id,
        "shape": list(target_mask.shape),
        "threshold": 0.5,
        "prediction": {"path": str(prediction_path), "sha256": sha256_file(prediction_path)},
        "prediction_sealed_before_target_ground_truth": True,
        "target_mask_opened": False,
        "target_metric_opened": False,
        "development_contract": "protocol_authorized_registered_rgb_official_scribble_sam3_video_memory",
        "strict_unseen_eligible": False,
        "tracker_score": target_score,
        "registered_path_frame_ids": [frame_ids[index] for index in path_indices],
        "registered_path_direction": "cyclic_shortest",
        "prompt_frame_id": prompt_id,
        "prompt_points": int(len(points)),
        "prompt_authority_shape": list(positive.shape),
        "prompt_seed": (
            "official_signed_scribble_selected_sam3_proposal_box_plus_points"
            if args.prompt_proposal_seed
            else "official_signed_scribble_points_only"
        ),
        "prompt_signed_adherence": prompt_adherence,
        "prompt_proposal": prompt_proposal_report,
        "inputs": {
            "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
            "positive_scribble": {"path": str(positive_path), "sha256": sha256_file(positive_path)},
            "negative_scribble": {"path": str(negative_path), "sha256": sha256_file(negative_path)},
            "registered_rgb": [
                {"path": str(path), "sha256": sha256_file(path)} for path in image_paths
            ],
        },
    }
    receipt_path = output_dir / "prediction_receipt.json"
    _atomic_json(receipt_path, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--points-per-sign", type=int, default=8)
    parser.add_argument("--prompt-proposal-seed", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    receipt = predict(parser.parse_args(argv))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
