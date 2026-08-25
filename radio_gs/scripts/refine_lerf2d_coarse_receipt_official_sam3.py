#!/usr/bin/env python3
"""Refine sealed LERF-2D coarse masks with fixed official SAM3 boxes."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

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
from radio_gs.utils.immutable_artifacts import (  # noqa: E402
    file_record,
    sha256_file,
    write_bytes_noclobber,
    write_frozen_json,
)


SCHEMA_VERSION = 1
COARSE_ARTIFACT_TYPE = "radio_gs_lerf2d_occam_coarse_prediction_receipt"
FORMAL_COARSE_ARTIFACT_TYPE = (
    "radio_gs_lerf2d_formal_posterior_coarse_prediction_receipt"
)
FORMAL_COARSE_POLICY = {
    "posterior_threshold": 0.6,
    "threshold_mode": "fixed",
    "eval_at_image_resolution": True,
    "primitive_valid_normalization": True,
    "primitive_valid_coverage_power": 0.0,
    "feature_contribution_gamma": 1.0,
}
RGB_AUTHORITY_TYPE = "radio_gs_lerf2d_target_rgb_root_authority"
FINAL_ARTIFACT_TYPE = "radio_gs_lerf2d_official_sam3_box_prediction_receipt"
METHOD_NAME = "RADIO-GS current-field exact scalar + target-RGB official SAM3 box pad16"
FORMAL_METHOD_NAME = (
    "RADIO-GS retained identity_extent_posterior_v3 + target-RGB official SAM3 box pad16"
)


class Sam3Lerf2DProtocolError(ValueError):
    pass


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Sam3Lerf2DProtocolError(f"{label} must be an object")
    return value


def _require_sha(value: Any, *, label: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise Sam3Lerf2DProtocolError(f"{label} is not a lowercase SHA256")
    return text


def _npy_bytes(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return buffer.getvalue()


def _load_binary_qhw(
    path: Path, *, expected_sha256: str, expected_shape: tuple[int, int, int], label: str
) -> np.ndarray:
    if not path.is_file() or sha256_file(path) != _require_sha(expected_sha256, label=label):
        raise Sam3Lerf2DProtocolError(f"{label} changed after sealing")
    value = np.load(path, allow_pickle=False)
    if (
        not isinstance(value, np.ndarray)
        or value.shape != expected_shape
        or value.dtype != np.uint8
        or not bool(np.isin(value, [0, 1]).all())
    ):
        raise Sam3Lerf2DProtocolError(f"{label} is not bound binary uint8 QHW")
    return value.astype(bool, copy=False)


def _bundle_member(root: Path, raw: Any, *, label: str) -> Path:
    relative = Path(str(raw or ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise Sam3Lerf2DProtocolError(f"{label} is not a safe relative path")
    path = (root / relative).resolve(strict=True)
    try:
        path.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise Sam3Lerf2DProtocolError(f"{label} escaped its receipt root") from error
    return path


def _validate_live_file_record(value: Any, *, label: str) -> Path:
    record = _require_mapping(value, label=label)
    path = Path(str(record.get("path", ""))).expanduser().resolve(strict=True)
    if not path.is_file() or sha256_file(path) != _require_sha(
        record.get("sha256"), label=f"{label}.sha256"
    ):
        raise Sam3Lerf2DProtocolError(f"{label} changed after sealing")
    return path


def _load_coarse_receipt(path: Path, *, expected_sha256: str) -> tuple[dict[str, Any], str]:
    receipt_path = path.expanduser().resolve(strict=True)
    digest = sha256_file(receipt_path)
    if digest != _require_sha(expected_sha256, label="coarse receipt SHA256"):
        raise Sam3Lerf2DProtocolError("coarse receipt SHA256 differs")
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    source = _require_mapping(payload.get("source_access"), label="source_access")
    common_invalid = (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status")
        != "coarse_predictions_sealed_before_target_rgb_or_gt_access"
        or source.get("target_rgb_opened") is not False
        or any(
            source.get(key) is not False
            for key in (
                "benchmark_annotation_json_opened",
                "benchmark_segmentation_opened",
                "benchmark_bboxes_opened",
                "benchmark_masks_opened",
                "benchmark_metrics_opened",
                "candidate_selected_with_gt",
            )
        )
    )
    artifact_type = payload.get("artifact_type")
    legacy = artifact_type == COARSE_ARTIFACT_TYPE
    formal = artifact_type == FORMAL_COARSE_ARTIFACT_TYPE
    policy_invalid = (
        legacy
        and payload.get("policy")
        != {"activation_kernel": 30, "mask_threshold": 0.5, "smooth_kernel": 7}
    ) or (formal and payload.get("policy") != FORMAL_COARSE_POLICY)
    if common_invalid or not (legacy or formal) or policy_invalid:
        raise Sam3Lerf2DProtocolError("coarse receipt violates the frozen contract")
    _validate_live_file_record(payload.get("implementation"), label="coarse producer")
    if legacy:
        authority = _require_mapping(payload.get("score_manifest"), label="score_manifest")
        authority_label = "score manifest"
    else:
        authority = _require_mapping(
            payload.get("query_authority_manifest"), label="query_authority_manifest"
        )
        authority_label = "query authority manifest"
        _validate_live_file_record(payload.get("posterior_source"), label="posterior source")
        _validate_live_file_record(payload.get("config"), label="formal config")
        _validate_live_file_record(payload.get("checkpoint"), label="formal checkpoint")
    authority_path = Path(str(authority.get("path", ""))).resolve(strict=True)
    if sha256_file(authority_path) != _require_sha(
        authority.get("sha256"), label=f"{authority_label} SHA256"
    ):
        raise Sam3Lerf2DProtocolError(f"{authority_label} changed after coarse sealing")
    return payload, digest


def _final_policy(coarse: Mapping[str, Any]) -> dict[str, Any]:
    common = {
        "sam3_box_padding_pixels": 16,
        "sam3_resolution": 1008,
        "sam3_confidence_threshold": 0.0,
        "sam3_min_initial_iou": 0.05,
        "candidate_selector": "coarse_mask_iou_then_official_score_tie_break",
    }
    if coarse.get("artifact_type") == FORMAL_COARSE_ARTIFACT_TYPE:
        return {
            "coarse_artifact_type": FORMAL_COARSE_ARTIFACT_TYPE,
            "coarse_policy": dict(FORMAL_COARSE_POLICY),
            "binary_mask_materialization": "interpolated_logit_gt_zero_exact",
            **common,
        }
    return {
        "coarse_activation_kernel": 30,
        "coarse_mask_threshold": 0.5,
        "coarse_smooth_kernel": 7,
        **common,
    }


def _load_rgb_authority(path: Path, *, expected_sha256: str) -> tuple[dict[str, Any], str]:
    authority_path = path.expanduser().resolve(strict=True)
    digest = sha256_file(authority_path)
    if digest != _require_sha(expected_sha256, label="RGB authority SHA256"):
        raise Sam3Lerf2DProtocolError("RGB authority SHA256 differs")
    payload = json.loads(authority_path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("artifact_type") != RGB_AUTHORITY_TYPE
        or payload.get("target_rgb_authorized") is not True
        or any(
            payload.get(key) is not False
            for key in (
                "benchmark_masks_opened",
                "benchmark_segmentation_opened",
                "benchmark_bboxes_opened",
                "benchmark_metrics_opened",
            )
        )
    ):
        raise Sam3Lerf2DProtocolError("RGB authority violates the no-GT contract")
    return payload, digest


def _resolve_rgb(authority: Mapping[str, Any], scene: str, camera_name: str) -> Path:
    scenes = _require_mapping(authority.get("scenes"), label="RGB authority scenes")
    entry = _require_mapping(scenes.get(scene), label=f"RGB authority {scene}")
    root = Path(str(entry.get("scene_root", ""))).expanduser().resolve(strict=True)
    subdir = Path(str(entry.get("image_subdir", "images")))
    if subdir.is_absolute() or ".." in subdir.parts:
        raise Sam3Lerf2DProtocolError(f"{scene}: invalid RGB subdirectory")
    unresolved = root / subdir / camera_name
    resolved = unresolved.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise Sam3Lerf2DProtocolError(f"{scene}: RGB escaped scene root") from error
    if resolved != unresolved.absolute() or not resolved.is_file():
        raise Sam3Lerf2DProtocolError(f"{scene}: RGB must be a regular non-symlink file")
    return resolved


def mask_to_box(mask: np.ndarray, *, padding_pixels: int = 16) -> Optional[list[float]]:
    pred = np.asarray(mask, dtype=bool)
    if pred.ndim != 2:
        raise ValueError(f"expected 2D mask, got {pred.shape}")
    if not pred.any():
        return None
    height, width = pred.shape
    ys, xs = np.nonzero(pred)
    x0, x1 = max(int(xs.min()) - padding_pixels, 0), min(
        int(xs.max()) + padding_pixels + 1, width
    )
    y0, y1 = max(int(ys.min()) - padding_pixels, 0), min(
        int(ys.max()) + padding_pixels + 1, height
    )
    box_w, box_h = float(max(x1 - x0, 1)), float(max(y1 - y0, 1))
    return [
        (float(x0) + box_w * 0.5) / width,
        (float(y0) + box_h * 0.5) / height,
        box_w / width,
        box_h / height,
    ]


def choose_candidate(
    coarse: np.ndarray,
    candidates: np.ndarray,
    *,
    scores: Optional[np.ndarray],
    min_initial_iou: float = 0.05,
) -> Tuple[np.ndarray, dict[str, Any]]:
    initial = np.asarray(coarse, dtype=bool)
    masks = np.asarray(candidates)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    report: dict[str, Any] = {
        "attempted": True,
        "accepted": False,
        "fallback_reason": "",
        "candidate_count": int(masks.shape[0]) if masks.ndim == 3 else 0,
        "selected_index": -1,
        "best_initial_overlap": 0.0,
        "selected_score": 0.0,
    }
    if masks.ndim != 3 or masks.shape[-2:] != initial.shape or masks.shape[0] == 0:
        report["fallback_reason"] = "candidate_shape_mismatch"
        return initial.copy(), report
    score_arr = np.asarray(
        scores if scores is not None else np.zeros(masks.shape[0]), dtype=np.float32
    )
    if score_arr.ndim != 1 or score_arr.shape[0] != masks.shape[0]:
        score_arr = np.zeros(masks.shape[0], dtype=np.float32)
    best_idx, best_overlap, best_score = -1, -1.0, -float("inf")
    for index, candidate in enumerate(masks):
        candidate = np.asarray(candidate) > 0
        union = float(np.logical_or(initial, candidate).sum())
        overlap = float(np.logical_and(initial, candidate).sum()) / union if union else 0.0
        candidate_score = float(score_arr[index])
        if overlap > best_overlap + 1e-8 or (
            abs(overlap - best_overlap) <= 1e-8 and candidate_score > best_score
        ):
            best_idx, best_overlap, best_score = index, overlap, candidate_score
    report.update(
        {
            "selected_index": best_idx,
            "best_initial_overlap": max(best_overlap, 0.0),
            "selected_score": best_score if np.isfinite(best_score) else 0.0,
        }
    )
    if best_overlap < min_initial_iou:
        report["fallback_reason"] = "low_initial_overlap"
        return initial.copy(), report
    report["accepted"] = True
    report["fallback_reason"] = "accepted"
    return (masks[best_idx] > 0).astype(bool), report


@torch.inference_mode()
def add_geometric_prompt_binary_memory_efficient(
    processor: Any, box: list[float], state: dict[str, Any]
) -> dict[str, Any]:
    """Run official SAM3 without a redundant full-resolution probability copy.

    The official processor returns ``sigmoid(interpolate(mask_logits))`` and
    this readout thresholds it at 0.5.  ``sigmoid(x) > 0.5`` is exactly
    ``x > 0``, so direct logit thresholding preserves every binary mask bit.
    """

    from sam3.model import box_ops
    from sam3.model.data_misc import interpolate

    if "backbone_out" not in state:
        raise ValueError("official SAM3 image state has no backbone output")
    if "language_features" not in state["backbone_out"]:
        state["backbone_out"].update(
            processor.model.backbone.forward_text(["visual"], device=processor.device)
        )
    if "geometric_prompt" not in state:
        state["geometric_prompt"] = processor.model._get_dummy_prompt()
    prompt_boxes = torch.tensor(
        box, device=processor.device, dtype=torch.float32
    ).view(1, 1, 4)
    prompt_labels = torch.ones((1, 1), device=processor.device, dtype=torch.bool)
    state["geometric_prompt"].append_boxes(prompt_boxes, prompt_labels)
    outputs = processor.model.forward_grounding(
        backbone_out=state["backbone_out"],
        find_input=processor.find_stage,
        geometric_prompt=state["geometric_prompt"],
        find_target=None,
    )
    probabilities = outputs["pred_logits"].sigmoid()
    presence = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
    probabilities = (probabilities * presence).squeeze(-1)
    keep = probabilities > processor.confidence_threshold
    probabilities = probabilities[keep]
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(outputs["pred_boxes"][keep])
    height, width = int(state["original_height"]), int(state["original_width"])
    scale = torch.tensor(
        [width, height, width, height], device=processor.device
    )
    boxes_xyxy = boxes_xyxy * scale[None, :]
    resized_logits = interpolate(
        outputs["pred_masks"][keep].unsqueeze(1),
        (height, width),
        mode="bilinear",
        align_corners=False,
    )
    state["masks"] = resized_logits > 0.0
    state["boxes"] = boxes_xyxy
    state["scores"] = probabilities
    return state


def _official_commit() -> str:
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


def refine(
    *,
    coarse_receipt: Path,
    coarse_receipt_sha256: str,
    rgb_authority: Path,
    rgb_authority_sha256: str,
    checkpoint: Path,
    output_dir: Path,
    device: str = "cuda",
) -> dict[str, Any]:
    coarse, coarse_sha = _load_coarse_receipt(
        coarse_receipt, expected_sha256=coarse_receipt_sha256
    )
    rgb, rgb_sha = _load_rgb_authority(
        rgb_authority, expected_sha256=rgb_authority_sha256
    )
    raw_scenes = _require_mapping(coarse.get("scenes"), label="coarse scenes")
    authority_scenes = _require_mapping(rgb.get("scenes"), label="RGB scenes")
    if not set(raw_scenes).issubset(set(authority_scenes)):
        raise Sam3Lerf2DProtocolError("RGB authority does not cover coarse scenes")
    checkpoint_path = checkpoint.expanduser().resolve(strict=True)
    output = output_dir.expanduser().absolute()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"final output must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    resolution = validate_sam3_resolution(1008, allow_unsafe=False)
    set_requested_cuda_device(device)
    processor = _load_sam3_model(
        checkpoint_path=str(checkpoint_path),
        device=device,
        confidence_threshold=0.0,
        dtype="float32",
        resolution=resolution,
        build_on_cpu=True,
    )
    amp_dtype = resolve_sam3_amp_dtype(device, "bfloat16")
    total_queries, accepted = 0, 0
    published_scenes: dict[str, Any] = {}
    for scene, scene_value in raw_scenes.items():
        frames = _require_mapping(
            _require_mapping(scene_value, label=scene).get("frames"), label=f"{scene}.frames"
        )
        published_frames: dict[str, Any] = {}
        for frame, frame_value in frames.items():
            entry = _require_mapping(frame_value, label=f"{scene}/{frame}")
            shape = entry.get("prediction_shape_qhw")
            if not isinstance(shape, list) or len(shape) != 3:
                raise Sam3Lerf2DProtocolError(f"{scene}/{frame}: coarse shape is malformed")
            expected_shape = tuple(int(value) for value in shape)
            coarse_path = _bundle_member(
                coarse_receipt.resolve(strict=True).parent,
                entry.get("coarse_prediction_file"),
                label=f"{scene}/{frame}.coarse_prediction_file",
            )
            coarse_qhw = _load_binary_qhw(
                coarse_path,
                expected_sha256=str(entry.get("coarse_prediction_sha256")),
                expected_shape=expected_shape,
                label=f"{scene}/{frame} coarse prediction",
            )
            query_ids, query_texts, query_rows = (
                entry.get("query_ids"),
                entry.get("query_texts"),
                entry.get("queries"),
            )
            if (
                not isinstance(query_ids, list)
                or not isinstance(query_texts, list)
                or not isinstance(query_rows, list)
                or len(query_ids) != expected_shape[0]
                or len(query_texts) != expected_shape[0]
                or len(query_rows) != expected_shape[0]
            ):
                raise Sam3Lerf2DProtocolError(f"{scene}/{frame}: coarse query axis differs")
            rgb_path = _resolve_rgb(rgb, scene, str(entry.get("camera_name", "")))
            rgb_digest = sha256_file(rgb_path)
            with Image.open(rgb_path) as image_handle:
                image = image_handle.convert("RGB")
                if (image.height, image.width) != expected_shape[-2:]:
                    raise Sam3Lerf2DProtocolError(f"{scene}/{frame}: RGB resolution differs")
                with torch.no_grad(), sam3_autocast_context(str(processor.device), amp_dtype):
                    state = processor.set_image(image)
            # The official image encoder transiently reserves CUDA allocator
            # blocks that are no longer live once the immutable image state is
            # materialized.  Return only those unused blocks before the box
            # decoder; this does not move tensors or alter model precision.
            if str(processor.device).startswith("cuda"):
                torch.cuda.empty_cache()
            final_qhw = np.zeros(expected_shape, dtype=np.uint8)
            final_query_rows = []
            for index in range(expected_shape[0]):
                initial = coarse_qhw[index]
                box = mask_to_box(initial, padding_pixels=16)
                report: dict[str, Any] = {
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
                    "box_padding_pixels": 16,
                    "min_initial_iou": 0.05,
                }
                if box is None:
                    prediction = initial.copy()
                    report["fallback_reason"] = "empty_initial_mask"
                else:
                    with torch.no_grad(), sam3_autocast_context(
                        str(processor.device), amp_dtype
                    ):
                        if coarse.get("artifact_type") == FORMAL_COARSE_ARTIFACT_TYPE:
                            sam_output = add_geometric_prompt_binary_memory_efficient(
                                processor, box, dict(state)
                            )
                        else:
                            sam_output = processor.add_geometric_prompt(
                                box, True, dict(state)
                            )
                    masks = sam_output.get("masks")
                    if masks is None and sam_output.get("masks_logits") is not None:
                        masks = sam_output["masks_logits"].float() > 0.0
                    if masks is None:
                        prediction = initial.copy()
                        report["fallback_reason"] = "missing_masks_and_logits"
                    else:
                        masks_np = (
                            masks.detach().cpu().numpy()
                            if torch.is_tensor(masks)
                            else np.asarray(masks)
                        )
                        scores = sam_output.get("scores")
                        scores_np = (
                            scores.detach().float().cpu().numpy()
                            if torch.is_tensor(scores)
                            else np.asarray(
                                scores if scores is not None else [], dtype=np.float32
                            )
                        )
                        prediction, selected = choose_candidate(
                            initial, masks_np, scores=scores_np, min_initial_iou=0.05
                        )
                        report.update(selected)
                final_qhw[index] = prediction.astype(np.uint8)
                accepted += int(report["accepted"])
                total_queries += 1
                coarse_query = _require_mapping(
                    query_rows[index], label=f"{scene}/{frame}.queries[{index}]"
                )
                if (
                    coarse_query.get("query_id") != query_ids[index]
                    or coarse_query.get("query_text") != query_texts[index]
                ):
                    raise Sam3Lerf2DProtocolError(f"{scene}/{frame}: coarse query row differs")
                final_query_rows.append({**dict(coarse_query), "boundary": report})
            relative = Path("final_masks") / scene / f"{frame}.npy"
            final_path = output / relative
            write_bytes_noclobber(final_path, _npy_bytes(final_qhw))
            published_frames[frame] = {
                **{key: entry[key] for key in (
                    "annotation_sha256",
                    "camera_name",
                    "query_ids",
                    "query_texts",
                    "resolution_hw",
                    "score_map",
                    "coarse_prediction_file",
                    "coarse_prediction_sha256",
                    "prediction_shape_qhw",
                )},
                "coarse_prediction_receipt_root": str(
                    coarse_receipt.resolve(strict=True).parent
                ),
                "target_rgb": {"path": str(rgb_path), "sha256": rgb_digest},
                "prediction_file": str(relative),
                "prediction_sha256": sha256_file(final_path),
                "queries": final_query_rows,
            }
            del state
        published_scenes[scene] = {"frames": published_frames}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": FINAL_ARTIFACT_TYPE,
        "status": "sealed_before_benchmark_mask_or_metric_access",
        "method": (
            FORMAL_METHOD_NAME
            if coarse.get("artifact_type") == FORMAL_COARSE_ARTIFACT_TYPE
            else METHOD_NAME
        ),
        "policy": _final_policy(coarse),
        "coarse_prediction_receipt": {
            "path": str(coarse_receipt.resolve(strict=True)),
            "sha256": coarse_sha,
            "score_manifest": coarse.get("score_manifest"),
            "query_authority_manifest": coarse.get("query_authority_manifest"),
            "posterior_source": coarse.get("posterior_source"),
        },
        "rgb_root_authority": {
            "path": str(rgb_authority.resolve(strict=True)),
            "sha256": rgb_sha,
        },
        "sam3": {
            "checkpoint": {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
            "official_source": "/root/external/sam3",
            "official_source_commit": _official_commit(),
            "model_dtype": "float32",
            "amp_dtype": "bfloat16",
        },
        "implementation": file_record(Path(__file__).resolve()),
        "source_access": {
            "target_rgb_opened": True,
            "benchmark_annotation_json_opened": False,
            "benchmark_segmentation_opened": False,
            "benchmark_bboxes_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_metrics_opened": False,
            "candidate_selected_with_gt": False,
        },
        "cohort": {
            "scenes": list(raw_scenes),
            "queries": total_queries,
            "sam3_candidate_accepted": accepted,
        },
        "scenes": published_scenes,
    }
    write_frozen_json(output / "prediction_receipt.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-receipt", type=Path, required=True)
    parser.add_argument("--coarse-receipt-sha256", required=True)
    parser.add_argument("--rgb-authority", type=Path, required=True)
    parser.add_argument("--rgb-authority-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    payload = refine(
        coarse_receipt=args.coarse_receipt,
        coarse_receipt_sha256=args.coarse_receipt_sha256,
        rgb_authority=args.rgb_authority,
        rgb_authority_sha256=args.rgb_authority_sha256,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
    )
    print(json.dumps({"status": payload["status"], "cohort": payload["cohort"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
