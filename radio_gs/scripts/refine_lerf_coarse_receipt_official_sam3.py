#!/usr/bin/env python3
"""Refine a sealed LERF O2 batch with official SAM3 box prompts, before scoring.

This producer deliberately runs in the official-SAM3 Torch environment.  It
reads only an immutable coarse-prediction receipt, its binary masks, the
sanitized benchmark inventory, and target RGB images.  It never opens the
polygon label directory and seals the complete refined batch before a scorer
is allowed to rasterize target masks or compute metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, ".")

from radio_gs.scripts.build_sam3_foundation_cache import (  # noqa: E402
    _load_sam3_model,
    resolve_sam3_amp_dtype,
    sam3_autocast_context,
    set_requested_cuda_device,
    validate_sam3_resolution,
)


COARSE_ARTIFACT = "lerf_o2_coarse_pre_sam3_prediction_receipt_v1"
FINAL_ARTIFACT = "lerf_target_rgb_assisted_pre_metric_prediction_receipt_v1"
TARGET_PRESET = "vala_paper_3d_target_rgb_sam3_box"


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_binary_mask(path: Path, *, height: int, width: int) -> np.ndarray:
    with Image.open(path) as image:
        value = np.asarray(image.convert("L")) > 127
    if value.shape != (height, width):
        raise ValueError(f"mask shape changed: {value.shape} vs {(height, width)}: {path}")
    return value


def _write_binary_mask_no_clobber(path: Path, mask: np.ndarray) -> None:
    value = (np.asarray(mask, dtype=bool).astype(np.uint8) * 255)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = _load_binary_mask(path, height=value.shape[0], width=value.shape[1])
        if not np.array_equal(existing, value > 127):
            raise FileExistsError(f"refusing to overwrite a different refined mask: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale mask temporary exists: {temporary}")
    Image.fromarray(value, mode="L").save(temporary, format="PNG")
    temporary.replace(path)


def _write_json_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f"refusing to overwrite a different receipt: {path}")
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        if temporary.exists():
            raise FileExistsError(f"stale receipt temporary exists: {temporary}")
        temporary.write_bytes(encoded)
        temporary.replace(path)
    return hashlib.sha256(encoded).hexdigest()


def mask_to_box(mask: np.ndarray, *, padding_pixels: int) -> Optional[list[float]]:
    pred = np.asarray(mask, dtype=bool)
    if pred.ndim != 2:
        raise ValueError(f"expected a 2D mask, got {pred.shape}")
    if not pred.any():
        return None
    height, width = pred.shape
    ys, xs = np.nonzero(pred)
    pad = max(int(padding_pixels), 0)
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, width)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, height)
    box_w = max(float(x1 - x0), 1.0)
    box_h = max(float(y1 - y0), 1.0)
    return [
        float(np.clip((float(x0) + box_w * 0.5) / max(width, 1), 0.0, 1.0)),
        float(np.clip((float(y0) + box_h * 0.5) / max(height, 1), 0.0, 1.0)),
        float(np.clip(box_w / max(width, 1), 0.0, 1.0)),
        float(np.clip(box_h / max(height, 1), 0.0, 1.0)),
    ]


def choose_candidate(
    initial_mask: np.ndarray,
    candidate_masks: np.ndarray,
    *,
    scores: Optional[np.ndarray],
    min_initial_iou: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Mirror the evaluator's fixed, GT-free overlap selector exactly."""
    initial = np.asarray(initial_mask, dtype=bool)
    report: Dict[str, Any] = {
        "accepted": False,
        "fallback_reason": "",
        "candidate_count": 0,
        "selected_index": -1,
        "best_initial_overlap": 0.0,
        "selected_score": 0.0,
    }
    masks = np.asarray(candidate_masks)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    if masks.ndim != 3 or masks.shape[-2:] != initial.shape:
        report["fallback_reason"] = "candidate_shape_mismatch"
        report["candidate_shape"] = list(masks.shape)
        return initial.copy(), report
    report["candidate_count"] = int(masks.shape[0])
    if masks.shape[0] == 0:
        report["fallback_reason"] = "empty_candidate_set"
        return initial.copy(), report
    score_arr = np.asarray(
        scores if scores is not None else np.zeros((masks.shape[0],)), dtype=np.float32
    )
    if score_arr.ndim != 1 or score_arr.shape[0] != masks.shape[0]:
        score_arr = np.zeros((masks.shape[0],), dtype=np.float32)
    best_idx = -1
    best_overlap = -1.0
    best_score = -float("inf")
    for idx, candidate in enumerate(masks):
        candidate_bool = np.asarray(candidate) > 0
        union = float(np.logical_or(initial, candidate_bool).sum())
        overlap = (
            float(np.logical_and(initial, candidate_bool).sum()) / union if union > 0 else 0.0
        )
        candidate_score = float(score_arr[idx])
        if overlap > best_overlap + 1e-8 or (
            abs(overlap - best_overlap) <= 1e-8 and candidate_score > best_score
        ):
            best_idx, best_overlap, best_score = idx, overlap, candidate_score
    report.update(
        {
            "selected_index": int(best_idx),
            "best_initial_overlap": float(max(best_overlap, 0.0)),
            "selected_score": float(best_score if np.isfinite(best_score) else 0.0),
        }
    )
    if best_idx < 0:
        report["fallback_reason"] = "no_valid_candidate"
        return initial.copy(), report
    if best_overlap < float(min_initial_iou):
        report["fallback_reason"] = "low_initial_overlap"
        return initial.copy(), report
    report["accepted"] = True
    report["fallback_reason"] = "accepted"
    return (np.asarray(masks[best_idx]) > 0).astype(bool), report


def validate_coarse_receipt(path: str | Path, *, expected_sha256: str) -> Dict[str, Any]:
    receipt_path = Path(path).expanduser().resolve()
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    observed_sha = sha256_file(receipt_path)
    if expected_sha256 and observed_sha != expected_sha256:
        raise ValueError(f"coarse receipt SHA256 mismatch: {observed_sha} vs {expected_sha256}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    protocol = receipt.get("protocol", {})
    if (
        receipt.get("artifact_type") != COARSE_ARTIFACT
        or receipt.get("status")
        != "coarse_prediction_sealed_before_target_rgb_sam3_and_metric"
        or receipt.get("target_rgb_opened") is not False
        or receipt.get("target_annotation_coordinates_loaded") is not False
        or receipt.get("target_mask_rasterized_before_seal") is not False
        or receipt.get("target_metric_computed_before_seal") is not False
        or protocol.get("protocol_preset") != TARGET_PRESET
        or protocol.get("prediction_stage") != "coarse_o2_before_sam3_bridge"
        or protocol.get("capability_track") != "target_rgb_assisted_official_sam3_box"
        or protocol.get("strict_mainline_eligible") is not False
        or protocol.get("projection_mode") != "selected_only_alpha"
        or abs(float(protocol.get("primitive_score_threshold", -1.0)) - 0.6) > 1e-9
        or int(protocol.get("sam3_box_padding", -1)) != 16
        or int(protocol.get("sam3_resolution", -1)) != 1008
        or abs(float(protocol.get("sam3_confidence_threshold", -1.0))) > 1e-9
        or abs(float(protocol.get("sam3_min_initial_iou", -1.0)) - 0.05) > 1e-9
    ):
        raise ValueError("coarse receipt violates the fixed target-RGB SAM3-box contract")
    inventory_path = Path(str(protocol.get("sanitized_prediction_inventory", ""))).resolve()
    if (
        not inventory_path.is_file()
        or sha256_file(inventory_path) != protocol.get("sanitized_prediction_inventory_sha256")
    ):
        raise ValueError("sanitized prediction inventory changed before SAM3 refinement")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if (
        inventory.get("artifact_type") != "lerf_sanitized_prediction_inventory_v1"
        or inventory.get("scene") != receipt.get("scene")
        or inventory.get("contains_polygon_coordinates") is not False
    ):
        raise ValueError("invalid sanitized prediction inventory")
    predictions = receipt.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("coarse receipt contains no predictions")
    identities = set()
    for row in predictions:
        identity = (int(row["frame_id"]), str(row["category"]))
        if identity in identities:
            raise ValueError(f"duplicate coarse prediction identity: {identity}")
        identities.add(identity)
        mask_path = Path(str(row["prediction_path"])).resolve()
        if not mask_path.is_file() or sha256_file(mask_path) != row["prediction_sha256"]:
            raise ValueError(f"sealed coarse prediction changed: {identity}")
        mask = _load_binary_mask(
            mask_path, height=int(row["height"]), width=int(row["width"])
        )
        if int(mask.sum()) != int(row["prediction_pixels"]):
            raise ValueError(f"sealed coarse prediction pixel count changed: {identity}")
    receipt["_receipt_path"] = str(receipt_path)
    receipt["_receipt_sha256"] = observed_sha
    receipt["_inventory_path"] = str(inventory_path)
    receipt["_inventory_sha256"] = sha256_file(inventory_path)
    return receipt


def _resolve_target_rgb(scene_root: Path, frame_id: int) -> Path:
    candidates = [
        scene_root / "images" / f"frame_{frame_id:05d}{suffix}"
        for suffix in (".jpg", ".jpeg", ".png", ".JPG", ".PNG")
    ]
    matches = [path.resolve() for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one target RGB for frame {frame_id}, found {matches}"
        )
    return matches[0]


def _official_source_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", "/root/external/sam3", "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-receipt", required=True)
    parser.add_argument("--expected-coarse-receipt-sha256", required=True)
    parser.add_argument("--scene-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--final-receipt", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32",), default="float32")
    parser.add_argument("--amp-dtype", choices=("bfloat16",), default="bfloat16")
    args = parser.parse_args()

    receipt = validate_coarse_receipt(
        args.coarse_receipt, expected_sha256=args.expected_coarse_receipt_sha256
    )
    scene_root = Path(args.scene_root).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    resolution = validate_sam3_resolution(1008, allow_unsafe=False)
    set_requested_cuda_device(args.device)
    processor = _load_sam3_model(
        checkpoint_path=str(checkpoint),
        device=args.device,
        confidence_threshold=0.0,
        dtype=args.dtype,
        resolution=resolution,
        build_on_cpu=True,
    )
    amp_dtype = resolve_sam3_amp_dtype(args.device, args.amp_dtype)

    by_frame: Dict[int, list[Dict[str, Any]]] = defaultdict(list)
    for row in receipt["predictions"]:
        by_frame[int(row["frame_id"])].append(dict(row))
    final_rows = []
    rgb_records = []
    for frame_id, rows in sorted(by_frame.items()):
        image_path = _resolve_target_rgb(scene_root, frame_id)
        image_sha = sha256_file(image_path)
        with Image.open(image_path) as image_handle:
            image = image_handle.convert("RGB")
            image_size = [int(image.height), int(image.width)]
            with torch.no_grad(), sam3_autocast_context(str(processor.device), amp_dtype):
                state = processor.set_image(image)
        rgb_records.append(
            {
                "frame_id": frame_id,
                "target_rgb_path": str(image_path),
                "target_rgb_sha256": image_sha,
                "height": image_size[0],
                "width": image_size[1],
            }
        )
        for row in sorted(rows, key=lambda value: str(value["category"])):
            height, width = int(row["height"]), int(row["width"])
            if image_size != [height, width]:
                raise ValueError(
                    f"target RGB/mask shape mismatch for frame {frame_id}: "
                    f"{image_size} vs {[height, width]}"
                )
            coarse_path = Path(str(row["prediction_path"])).resolve()
            coarse = _load_binary_mask(coarse_path, height=height, width=width)
            box = mask_to_box(coarse, padding_pixels=16)
            report: Dict[str, Any] = {
                "backend": "facebookresearch/sam3_official_box",
                "attempted": True,
                "accepted": False,
                "fallback_reason": "",
                "candidate_count": 0,
                "selected_index": -1,
                "best_initial_overlap": 0.0,
                "selected_score": 0.0,
                "box_prompt_format": "normalized_cxcywh",
                "box_prompt_cxcywh_norm": box,
                "box_prompt_xyxy_pixels": None,
                "box_padding_pixels": 16,
                "min_initial_iou": 0.05,
            }
            if box is None:
                refined = coarse.copy()
                report["fallback_reason"] = "empty_initial_mask"
            else:
                cx, cy, box_w, box_h = box
                abs_w, abs_h = box_w * width, box_h * height
                x0, y0 = cx * width - abs_w * 0.5, cy * height - abs_h * 0.5
                report["box_prompt_xyxy_pixels"] = [x0, y0, x0 + abs_w, y0 + abs_h]
                with torch.no_grad(), sam3_autocast_context(
                    str(processor.device), amp_dtype
                ):
                    output = processor.add_geometric_prompt(box, True, dict(state))
                masks = output.get("masks")
                if masks is None:
                    logits = output.get("masks_logits")
                    if logits is None:
                        refined = coarse.copy()
                        report["fallback_reason"] = "missing_masks_and_logits"
                        masks = None
                    else:
                        masks = logits.float() > 0.0
                if masks is not None:
                    masks_np = (
                        masks.detach().cpu().numpy()
                        if torch.is_tensor(masks)
                        else np.asarray(masks)
                    )
                    scores = output.get("scores")
                    scores_np = (
                        scores.detach().float().cpu().numpy()
                        if torch.is_tensor(scores)
                        else np.asarray(scores if scores is not None else [], dtype=np.float32)
                    )
                    refined, chosen = choose_candidate(
                        coarse, masks_np, scores=scores_np, min_initial_iou=0.05
                    )
                    report.update(chosen)
            safe_category = str(row["category"]).replace("/", "_")
            refined_path = (
                output_root
                / "final_pred_masks"
                / "score_threshold_0p600"
                / str(receipt["scene"])
                / f"frame_{frame_id:05d}_{safe_category}.png"
            ).resolve()
            _write_binary_mask_no_clobber(refined_path, refined)
            final_rows.append(
                {
                    "frame_id": frame_id,
                    "category": str(row["category"]),
                    "prediction_path": str(refined_path),
                    "prediction_sha256": sha256_file(refined_path),
                    "coarse_prediction_path": str(coarse_path),
                    "coarse_prediction_sha256": sha256_file(coarse_path),
                    "height": height,
                    "width": width,
                    "prediction_pixels": int(refined.sum()),
                    "coarse_prediction_pixels": int(coarse.sum()),
                    "target_rgb_path": str(image_path),
                    "target_rgb_sha256": image_sha,
                    "sam3_report": report,
                }
            )
        del state

    final_rows.sort(key=lambda row: (row["frame_id"], row["category"]))
    protocol = dict(receipt["protocol"])
    protocol.update(
        {
            "prediction_stage": "official_sam3_box_before_metric",
            "mask_refinement": "official_sam3_box",
            "target_annotation_coordinates_loaded": False,
            "coarse_prediction_receipt": receipt["_receipt_path"],
            "coarse_prediction_receipt_sha256": receipt["_receipt_sha256"],
            "sanitized_prediction_inventory": receipt["_inventory_path"],
            "sanitized_prediction_inventory_sha256": receipt["_inventory_sha256"],
            "sam3_checkpoint": str(checkpoint),
            "sam3_checkpoint_sha256": sha256_file(checkpoint),
            "sam3_model_dtype": args.dtype,
            "sam3_amp_dtype": args.amp_dtype,
            "official_sam3_source": "/root/external/sam3",
            "official_sam3_source_commit": _official_source_commit(),
            "bridge_script": str(Path(__file__).resolve()),
            "bridge_script_sha256": sha256_file(__file__),
        }
    )
    payload = {
        "schema_version": 1,
        "artifact_type": FINAL_ARTIFACT,
        "status": "sealed_before_target_mask_rasterization_and_metric",
        "scene": str(receipt["scene"]),
        "selection": dict(receipt["selection"]),
        "protocol": protocol,
        "predictions": final_rows,
        "prediction_count": len(final_rows),
        "target_rgb_images": rgb_records,
        "target_rgb_opened": True,
        "target_annotation_inventory_opened": True,
        "target_annotation_coordinates_loaded": False,
        "target_mask_rasterized_before_seal": False,
        "target_metric_computed_before_seal": False,
    }
    receipt_path = Path(args.final_receipt).expanduser().resolve()
    digest = _write_json_no_clobber(receipt_path, payload)
    accepted = sum(bool(row["sam3_report"].get("accepted")) for row in final_rows)
    print(
        json.dumps(
            {
                "final_receipt": str(receipt_path),
                "final_receipt_sha256": digest,
                "predictions": len(final_rows),
                "sam3_accepted": accepted,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
