#!/usr/bin/env python3
"""Source-only SAM3 video-track consistency sentinel.

The largest non-duplicate automatic SAM3 proposals in the first legal source
view are used only as box prompts.  Official SAM3 tracking propagates them over
the ordered source cohort.  Independent per-frame query-free SAM3 proposals
measure extent agreement, while native DINOv2 tokens measure physical-identity
stability.  No benchmark query, target/evaluation RGB, mask or metric is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
import torch
from torch.nn import functional as F

from radio_gs.models.sam3_multiscale_hierarchy import unpack_masks


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON authority is not a mapping: {path}")
    return dict(value)


def binary_iou(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=bool)
    b = np.asarray(right, dtype=bool)
    intersection = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return float(intersection / union) if union else 1.0


def select_seed_proposals(
    masks: np.ndarray,
    area: torch.Tensor,
    quality: torch.Tensor,
    maximum: int,
    maximum_pair_iou: float,
) -> list[int]:
    rank = sorted(
        range(int(masks.shape[0])),
        key=lambda index: (float(area[index]), float(quality[index]), -index),
        reverse=True,
    )
    selected: list[int] = []
    for index in rank:
        if any(
            binary_iou(masks[index], masks[other]) > float(maximum_pair_iou)
            for other in selected
        ):
            continue
        selected.append(index)
        if len(selected) == int(maximum):
            break
    return selected


def pooled_dino_descriptor(feature: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
    dense = torch.as_tensor(feature).float()
    if dense.ndim != 3:
        raise ValueError("native DINO frame feature is not [C,H,W]")
    support = F.interpolate(
        torch.from_numpy(np.asarray(mask, dtype=np.float32))[None, None],
        size=tuple(dense.shape[-2:]),
        mode="nearest",
    )[0, 0]
    value = (dense * support[None]).flatten(1).sum(dim=1)
    value /= support.sum().clamp_min(1.0)
    return F.normalize(value, dim=0, eps=1e-8)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from sam3.model_builder import build_sam3_video_model

    source_path = Path(args.source_authority).expanduser().resolve(strict=True)
    source = _load_json(source_path)
    mask_root = Path(args.mask_root).expanduser().resolve(strict=True)
    manifest_path = mask_root / args.manifest_name
    manifest = _load_json(manifest_path)
    checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    if sha256_file(checkpoint) != args.expected_checkpoint_sha256:
        raise ValueError("official SAM3 video checkpoint hash differs")
    images = source.get("images")
    selected_ids = manifest.get("selected_image_ids")
    records = {
        str(record.get("image_id", "")): dict(record)
        for record in manifest.get("images", [])
        if isinstance(record, Mapping)
    }
    if (
        source.get("contract") != "sam3-query-free-source-rgb-authority-v1"
        or not isinstance(images, list)
        or not isinstance(selected_ids, list)
        or [str(item.get("image_id", "")) for item in images] != selected_ids
        or list(records) != selected_ids
    ):
        raise ValueError("source RGB and SAM3 proposal authorities differ")
    start_view = int(args.start_view)
    if start_view < 0 or start_view >= len(images):
        raise ValueError("start view falls outside the source cohort")
    images = images[start_view:]
    selected_ids = selected_ids[start_view:]
    frame_count = min(int(args.maximum_frames), len(images))
    if frame_count < 2:
        raise ValueError("video sentinel requires at least two source frames")
    images = images[:frame_count]
    selected_ids = selected_ids[:frame_count]
    frame_paths: list[Path] = []
    mask_paths: list[Path] = []
    dino_paths: list[Path] = []
    exact_frame_numbers: list[int] = []
    for item, image_id in zip(images, selected_ids):
        image = Path(str(item.get("path", ""))).expanduser().resolve(strict=True)
        if sha256_file(image) != str(item.get("sha256", "")):
            raise ValueError("source image hash differs")
        mask_path = Path(str(records[image_id].get("output", ""))).resolve(strict=True)
        if sha256_file(mask_path) != str(records[image_id].get("output_sha256", "")):
            raise ValueError("source proposal cache hash differs")
        frame = int(image_id.removeprefix("frame_"))
        dino_path = Path(args.native_dino_frame_root).expanduser().resolve() / f"frame_{frame:05d}.pt"
        dino_path.resolve(strict=True)
        frame_paths.append(image)
        mask_paths.append(mask_path)
        dino_paths.append(dino_path)
        exact_frame_numbers.append(frame)

    dense_manifest_record: dict[str, str] | None = None
    if args.dense_frame_manifest:
        dense_manifest_path = Path(args.dense_frame_manifest).expanduser().resolve(
            strict=True
        )
        dense_manifest = _load_json(dense_manifest_path)
        image_dir = Path(str(dense_manifest.get("image_dir", ""))).expanduser().resolve(
            strict=True
        )
        dense_records = [
            dict(record)
            for record in dense_manifest.get("frames", [])
            if isinstance(record, Mapping)
            and exact_frame_numbers[0]
            <= int(record.get("frame_idx", -1))
            <= exact_frame_numbers[-1]
        ]
        if not dense_records:
            raise ValueError("dense source manifest does not bridge exact views")
        dense_frame_paths: list[Path] = []
        dense_frame_numbers: list[int] = []
        for record in dense_records:
            path = (image_dir / str(record.get("source_file", ""))).resolve(strict=True)
            if sha256_file(path) != str(record.get("source_sha256", "")):
                raise ValueError("dense source image hash differs")
            dense_frame_paths.append(path)
            dense_frame_numbers.append(int(record["frame_idx"]))
        dense_positions = {frame: index for index, frame in enumerate(dense_frame_numbers)}
        if any(frame not in dense_positions for frame in exact_frame_numbers):
            raise ValueError("dense source manifest omitted an exact anchor view")
        exact_video_positions = [dense_positions[frame] for frame in exact_frame_numbers]
        dense_manifest_record = {
            "path": str(dense_manifest_path),
            "sha256": sha256_file(dense_manifest_path),
        }
    else:
        dense_frame_paths = frame_paths
        exact_video_positions = list(range(frame_count))

    seed_payload = torch.load(mask_paths[0], map_location="cpu", weights_only=False)
    seed_height, seed_width = (int(value) for value in seed_payload["mask_shape"])
    seed_masks = unpack_masks(
        torch.as_tensor(seed_payload["packed_masks"]), width=seed_width
    )
    seed_indices = select_seed_proposals(
        seed_masks,
        torch.as_tensor(seed_payload["proposal_area_fraction"]),
        torch.as_tensor(seed_payload["quality"]),
        int(args.seed_proposals),
        float(args.maximum_seed_pair_iou),
    )
    if not seed_indices:
        raise ValueError("source frame has no seed proposal")
    seed_dino_payload = torch.load(
        dino_paths[0], map_location="cpu", weights_only=False
    )
    seed_dino = [
        pooled_dino_descriptor(seed_dino_payload["feature"], seed_masks[index])
        for index in seed_indices
    ]

    with tempfile.TemporaryDirectory(prefix="radio_gs_sam3_source_track_") as temporary:
        frame_root = Path(temporary)
        for index, source_image in enumerate(dense_frame_paths):
            os.symlink(source_image, frame_root / f"{index:05d}.jpg")
        model = build_sam3_video_model(
            checkpoint_path=str(checkpoint),
            load_from_HF=False,
            strict_state_dict_loading=True,
            apply_temporal_disambiguation=True,
            device=args.device,
            compile=False,
        )
        predictor = model.tracker
        predictor.backbone = model.detector.backbone
        state = predictor.init_state(video_path=str(frame_root))
        for object_offset, proposal_index in enumerate(seed_indices):
            box = torch.as_tensor(seed_payload["boxes_xyxy"])[proposal_index].float()
            relative = np.asarray(
                [[
                    float(box[0]) / seed_width,
                    float(box[1]) / seed_height,
                    float(box[2]) / seed_width,
                    float(box[3]) / seed_height,
                ]],
                dtype=np.float32,
            )
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=object_offset + 1,
                box=relative,
            )
        tracked: dict[int, dict[int, np.ndarray]] = {}
        scores: dict[int, dict[int, float]] = {}
        for frame_index, object_ids, _low, masks, object_scores in predictor.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=len(dense_frame_paths),
            reverse=False,
            propagate_preflight=True,
        ):
            if int(frame_index) >= len(dense_frame_paths):
                continue
            tracked[int(frame_index)] = {}
            scores[int(frame_index)] = {}
            for position, object_id in enumerate(object_ids):
                mask = (masks[position].detach().float().cpu().squeeze().numpy() > 0.0)
                tracked[int(frame_index)][int(object_id)] = mask
                scores[int(frame_index)][int(object_id)] = float(
                    torch.as_tensor(object_scores[position]).detach().cpu()
                )
        if hasattr(predictor, "reset_state"):
            predictor.reset_state(state)
        del state, predictor, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    frame_reports: list[dict[str, Any]] = []
    extent_values: list[float] = []
    identity_values: list[float] = []
    for frame_index in range(1, frame_count):
        video_frame_index = exact_video_positions[frame_index]
        payload = torch.load(mask_paths[frame_index], map_location="cpu", weights_only=False)
        height, width = (int(value) for value in payload["mask_shape"])
        proposals = unpack_masks(torch.as_tensor(payload["packed_masks"]), width=width)
        dino = torch.load(dino_paths[frame_index], map_location="cpu", weights_only=False)
        object_reports: list[dict[str, Any]] = []
        for object_offset, seed_index in enumerate(seed_indices):
            object_id = object_offset + 1
            mask = tracked.get(video_frame_index, {}).get(object_id)
            if mask is None:
                object_reports.append({"object_id": object_id, "present": False})
                continue
            if mask.shape != (height, width):
                mask = F.interpolate(
                    torch.from_numpy(mask.astype(np.float32))[None, None],
                    size=(height, width),
                    mode="nearest",
                )[0, 0].numpy() > 0.5
            proposal_ious = [binary_iou(mask, proposal) for proposal in proposals]
            best_iou = max(proposal_ious, default=0.0)
            appearance = pooled_dino_descriptor(dino["feature"], mask)
            cosine = float(torch.dot(seed_dino[object_offset], appearance))
            extent_values.append(best_iou)
            identity_values.append(cosine)
            object_reports.append(
                {
                    "object_id": object_id,
                    "seed_proposal_index": seed_index,
                    "present": True,
                    "tracker_score": scores.get(video_frame_index, {}).get(object_id),
                    "best_independent_sam3_proposal_iou": best_iou,
                    "native_dinov2_seed_cosine": cosine,
                    "tracked_area_fraction": float(mask.mean()),
                }
            )
        frame_reports.append(
            {
                "frame_index": frame_index,
                "video_frame_index": video_frame_index,
                "image_id": selected_ids[frame_index],
                "objects": object_reports,
            }
        )
    if not extent_values:
        raise RuntimeError("official SAM3 tracker returned no propagated mask")
    mean_extent = float(np.mean(extent_values))
    mean_identity = float(np.mean(identity_values))
    passed = mean_extent >= float(args.minimum_mean_proposal_iou) and mean_identity >= float(
        args.minimum_mean_dino_cosine
    )
    output = {
        "schema": "radio_gs.sam3_video_source_track_consistency.v1",
        "schema_version": 1,
        "status": "source_gate_pass" if passed else "source_gate_fail",
        "scene": str(args.scene),
        "method": {
            "seed": "largest_nonduplicate_query_free_source_sam3_proposals_as_boxes",
            "tracker": "official_sam3_video_tracker",
            "extent_measure": "best_iou_to_independent_query_free_sam3_proposal",
            "identity_measure": "native_dinov2_mask_pool_cosine_to_seed",
            "frames": frame_count,
            "dense_tracking_frames": len(dense_frame_paths),
            "source_start_view": start_view,
            "objects": len(seed_indices),
            "seed_proposal_indices": seed_indices,
        },
        "aggregate": {
            "observations": len(extent_values),
            "mean_best_independent_sam3_proposal_iou": mean_extent,
            "mean_native_dinov2_seed_cosine": mean_identity,
            "minimum_best_independent_sam3_proposal_iou": min(extent_values),
            "minimum_native_dinov2_seed_cosine": min(identity_values),
        },
        "gate": {
            "minimum_mean_proposal_iou": float(args.minimum_mean_proposal_iou),
            "minimum_mean_dino_cosine": float(args.minimum_mean_dino_cosine),
            "passed": passed,
        },
        "frames": frame_reports,
        "inputs": {
            "source_authority": {"path": str(source_path), "sha256": sha256_file(source_path)},
            "sam3_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "sam3_checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
            "native_dino_frame_root": str(Path(args.native_dino_frame_root).resolve()),
        },
        "access_audit": {
            "source_rgb_only": True,
            "query_text_opened": False,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_metrics_opened": False,
        },
        "claim_boundary": "source consistency sentinel; no 3D membership or benchmark promotion",
    }
    if dense_manifest_record is not None:
        output["inputs"]["dense_frame_manifest"] = dense_manifest_record
    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_path)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--source-authority", required=True)
    parser.add_argument("--mask-root", required=True)
    parser.add_argument("--manifest-name", default="manifest_grid8_crop2.json")
    parser.add_argument("--native-dino-frame-root", required=True)
    parser.add_argument("--dense-frame-manifest", default="")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--maximum-frames", type=int, default=8)
    parser.add_argument("--start-view", type=int, default=0)
    parser.add_argument("--seed-proposals", type=int, default=4)
    parser.add_argument("--maximum-seed-pair-iou", type=float, default=0.5)
    parser.add_argument("--minimum-mean-proposal-iou", type=float, default=0.5)
    parser.add_argument("--minimum-mean-dino-cosine", type=float, default=0.6)
    print(json.dumps(evaluate(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
