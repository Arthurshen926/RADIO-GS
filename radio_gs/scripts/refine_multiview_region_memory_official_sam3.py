#!/usr/bin/env python3
"""Refine sealed source-view anchors with fixed official SAM3 boxes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
from PIL import Image
import torch

from radio_gs.scripts.build_sam3_foundation_cache import (
    _load_sam3_model,
    resolve_sam3_amp_dtype,
    sam3_autocast_context,
    set_requested_cuda_device,
    validate_sam3_resolution,
)
from radio_gs.scripts.refine_lerf_coarse_receipt_official_sam3 import (
    choose_candidate,
    mask_to_box,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    sha256_file,
    validate_file_record,
    write_bytes_noclobber,
    write_frozen_json,
)


def _load_region_memory_helpers():
    """Load the leaf helper without importing optional query dependencies."""

    helper_path = (
        Path(__file__).resolve().parents[1]
        / "querying"
        / "multiview_region_memory.py"
    )
    module_name = "radio_gs_multiview_region_memory_leaf"
    spec = importlib.util.spec_from_file_location(module_name, helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load multiview-region helper {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_REGION_MEMORY_HELPERS = _load_region_memory_helpers()
METHOD = _REGION_MEMORY_HELPERS.METHOD
method_contract = _REGION_MEMORY_HELPERS.method_contract


ARTIFACT_TYPE = "multiview_region_memory_source_sam3_prediction_receipt_v1"
VIEW_ARTIFACT_TYPE = "multiview_region_memory_source_sam3_view_receipt_v1"
COARSE_ARTIFACT_TYPE = "multiview_region_memory_coarse_source_prediction_receipt_v1"
DECODER_HEIGHT = 756
DECODER_WIDTH = 1008
SOURCE_VIEW_COUNT = 3


def _load_mask(value: Mapping[str, Any], *, label: str) -> torch.Tensor:
    path = Path(str(value.get("path", ""))).expanduser().resolve(strict=True)
    if sha256_file(path) != str(value.get("sha256", "")):
        raise ValueError(f"{label} changed after sealing")
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L")) > 127
    if mask.shape != (DECODER_HEIGHT, DECODER_WIDTH):
        raise ValueError(f"{label} has the wrong decoder shape")
    if int(mask.sum()) != int(value.get("pixels", -1)):
        raise ValueError(f"{label} pixel count changed")
    return torch.from_numpy(mask.copy()).bool()


def _save_mask(path: Path, value: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(value, dtype=bool)
    buffer = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(buffer, format="PNG")
    write_bytes_noclobber(path, buffer.getvalue())
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "height": int(mask.shape[0]),
        "width": int(mask.shape[1]),
        "pixels": int(mask.sum()),
    }


def _validate_coarse(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str, Path]:
    coarse, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="multiview region-memory coarse receipt",
    )
    access = coarse.get("source_access")
    selection = coarse.get("selection")
    if (
        coarse.get("schema_version") != 1
        or coarse.get("artifact_type") != COARSE_ARTIFACT_TYPE
        or coarse.get("status")
        != "coarse_source_predictions_sealed_before_source_rgb_sam3_or_target_access"
        or coarse.get("method") != METHOD
        or coarse.get("method_contract") != method_contract()
        or coarse.get("selection_count") != SOURCE_VIEW_COUNT
        or not isinstance(selection, list)
        or len(selection) != SOURCE_VIEW_COUNT
        or not isinstance(access, Mapping)
        or access.get("source_rgb_opened") is not False
        or access.get("official_sam3_loaded") is not False
        or any(
            access.get(key) is not False
            for key in (
                "target_rgb_opened",
                "target_mask_opened",
                "target_metric_opened",
                "candidate_selected_with_gt",
            )
        )
    ):
        raise ValueError("coarse receipt violates the source-SAM3 contract")
    validate_file_record(coarse.get("implementation"), label="coarse producer")
    inventory = coarse.get("source_inventory")
    if not isinstance(inventory, Mapping):
        raise ValueError("coarse receipt lacks source inventory binding")
    inventory_payload, inventory_sha, _ = load_json_object(
        inventory.get("path", ""),
        expected_sha256=inventory.get("sha256"),
        label="source inventory",
    )
    scene = inventory_payload.get("scenes", {}).get(coarse.get("scene_id"))
    if (
        not isinstance(scene, Mapping)
        or scene.get("source_rgb_inventory_sha256")
        != inventory.get("scene_source_rgb_inventory_sha256")
        or scene.get("safety", {}).get("target_rgb_content_opened_or_hashed") is not False
        or inventory_sha != inventory.get("sha256")
    ):
        raise ValueError("coarse source inventory changed")
    forbidden = {str(value) for value in scene["forbidden_target_frame_ids"]}
    reference = str(scene["reference_frame_id"])
    identities = []
    for rank, row in enumerate(selection):
        if not isinstance(row, Mapping) or row.get("selection_rank") != rank:
            raise ValueError("coarse selection order differs")
        frame_id = str(row.get("frame_id", ""))
        if not frame_id or frame_id == reference or frame_id in forbidden:
            raise ValueError("coarse selection contains reference or target frame")
        _load_mask(row.get("coarse_mask", {}), label=f"coarse selection {rank}")
        source_path = Path(str(row.get("source_rgb_path", ""))).resolve(strict=True)
        if source_path.name in set(scene["forbidden_target_rgb_names"]):
            raise ValueError("coarse selection source path is a forbidden target")
        if sha256_file(source_path) != row.get("source_rgb_sha256"):
            raise ValueError("selected source RGB changed")
        identities.append(frame_id)
    if len(set(identities)) != SOURCE_VIEW_COUNT:
        raise ValueError("coarse source identities are not unique")
    return coarse, digest, source


def run(args: argparse.Namespace) -> dict[str, Any]:
    coarse, coarse_sha, coarse_path = _validate_coarse(
        Path(args.coarse_receipt), args.coarse_receipt_sha256
    )
    checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    if sha256_file(checkpoint) != args.checkpoint_sha256:
        raise ValueError("official SAM3 checkpoint changed")
    output_root = Path(args.output_root).expanduser().absolute()
    output_root.mkdir(parents=True, exist_ok=True)
    final_receipt = Path(args.output_receipt).expanduser().absolute()
    if final_receipt.exists():
        raise FileExistsError(final_receipt)

    resolution = validate_sam3_resolution(1008, allow_unsafe=False)
    set_requested_cuda_device(args.device)
    processor = _load_sam3_model(
        checkpoint_path=str(checkpoint),
        device=args.device,
        confidence_threshold=0.0,
        dtype="float32",
        resolution=resolution,
        build_on_cpu=True,
    )
    amp_dtype = resolve_sam3_amp_dtype(args.device, "bfloat16")
    view_receipts: list[dict[str, Any]] = []
    accepted = 0
    for row in coarse["selection"]:
        rank = int(row["selection_rank"])
        frame_id = str(row["frame_id"])
        source_path = Path(str(row["source_rgb_path"])).resolve(strict=True)
        source_sha = sha256_file(source_path)
        if source_sha != row["source_rgb_sha256"]:
            raise ValueError(f"{frame_id}: selected source RGB changed before SAM3")
        coarse_mask = _load_mask(row["coarse_mask"], label=f"{frame_id} coarse mask")
        coarse_np = coarse_mask.numpy()
        box = mask_to_box(coarse_np, padding_pixels=16)
        if box is None:
            raise ValueError(f"{frame_id}: selected coarse mask is empty")
        with Image.open(source_path) as image_handle:
            original_size = [int(image_handle.height), int(image_handle.width)]
            image = image_handle.convert("RGB").resize(
                (DECODER_WIDTH, DECODER_HEIGHT), Image.Resampling.LANCZOS
            )
        resized_rgb = np.asarray(image).copy()
        resized_rgb_sha = hashlib.sha256(resized_rgb.tobytes(order="C")).hexdigest()
        with torch.no_grad(), sam3_autocast_context(str(processor.device), amp_dtype):
            state = processor.set_image(image)
            output = processor.add_geometric_prompt(box, True, dict(state))
        masks = output.get("masks")
        if masks is None:
            logits = output.get("masks_logits")
            if logits is None:
                raise ValueError(f"{frame_id}: SAM3 returned neither masks nor logits")
            masks = logits.float() > 0
        masks_np = masks.detach().cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks)
        scores = output.get("scores")
        scores_np = (
            scores.detach().float().cpu().numpy()
            if torch.is_tensor(scores)
            else np.asarray(scores if scores is not None else [], dtype=np.float32)
        )
        refined, report = choose_candidate(
            coarse_np,
            masks_np,
            scores=scores_np,
            min_initial_iou=0.05,
        )
        report.update(
            {
                "backend": "facebookresearch/sam3_official_box",
                "box_prompt_format": "normalized_cxcywh",
                "box_prompt_cxcywh_norm": [float(value) for value in box],
                "box_padding_pixels": 16,
                "decoder_resolution_wh": [DECODER_WIDTH, DECODER_HEIGHT],
                "confidence_threshold": 0.0,
                "minimum_projected_anchor_overlap": 0.05,
            }
        )
        accepted += int(bool(report.get("accepted")))
        mask_path = output_root / "proposal_masks" / f"{rank:02d}_{frame_id}.png"
        mask_record = _save_mask(mask_path, refined)
        view_payload = {
            "schema_version": 1,
            "artifact_type": VIEW_ARTIFACT_TYPE,
            "status": "source_proposal_sealed_before_target_access",
            "scene_id": coarse["scene_id"],
            "selection_rank": rank,
            "assignment_view_index": int(row["assignment_view_index"]),
            "frame_id": frame_id,
            "source_rgb": {
                "path": str(source_path),
                "sha256": source_sha,
                "original_size_hw": original_size,
                "decoder_size_hw": [DECODER_HEIGHT, DECODER_WIDTH],
                "resized_rgb_tensor_sha256": resized_rgb_sha,
            },
            "coarse_mask": dict(row["coarse_mask"]),
            "proposal_mask": mask_record,
            "sam3_report": report,
            "coarse_receipt": {"path": str(coarse_path), "sha256": coarse_sha},
            "source_access": {
                "source_rgb_opened": True,
                "target_rgb_opened": False,
                "target_mask_opened": False,
                "target_metric_opened": False,
                "candidate_selected_with_gt": False,
            },
        }
        view_path = output_root / "view_receipts" / f"{rank:02d}_{frame_id}.json"
        write_frozen_json(view_path, view_payload)
        view_receipts.append(
            {
                "selection_rank": rank,
                "frame_id": frame_id,
                "path": str(view_path.resolve()),
                "sha256": sha256_file(view_path),
                "proposal_mask": mask_record,
                "sam3_accepted": bool(report.get("accepted")),
                "best_initial_overlap": float(report.get("best_initial_overlap", 0.0)),
            }
        )
        del state, output, masks, image, resized_rgb
    torch.cuda.synchronize()
    payload = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "status": "all_source_proposals_sealed_before_target_access",
        "scene_id": coarse["scene_id"],
        "method": METHOD,
        "method_contract": method_contract(),
        "coarse_receipt": {"path": str(coarse_path), "sha256": coarse_sha},
        "checkpoint": {"path": str(checkpoint), "sha256": args.checkpoint_sha256},
        "view_receipts": view_receipts,
        "view_count": len(view_receipts),
        "accepted_view_count": accepted,
        "implementation": file_record(Path(__file__).resolve()),
        "source_access": {
            "source_rgb_opened": True,
            "target_rgb_opened": False,
            "target_mask_opened": False,
            "target_metric_opened": False,
            "candidate_selected_with_gt": False,
        },
    }
    write_frozen_json(final_receipt, payload)
    return {
        "receipt": str(final_receipt.resolve()),
        "sha256": sha256_file(final_receipt),
        "accepted": accepted,
        "views": len(view_receipts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-receipt", required=True)
    parser.add_argument("--coarse-receipt-sha256", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-receipt", required=True)
    parser.add_argument("--device", default="cuda")
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
