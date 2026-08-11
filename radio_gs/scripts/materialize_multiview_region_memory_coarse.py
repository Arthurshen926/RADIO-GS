#!/usr/bin/env python3
"""Seal source-only projected-anchor masks before official SAM3 is loaded."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from radio_gs.querying.multiview_region_memory import (
    METHOD,
    method_contract,
    project_anchor_to_feature_view,
    select_source_views,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_torch_mapping,
    sha256_file,
    write_bytes_noclobber,
    write_frozen_json,
)


ARTIFACT_TYPE = "multiview_region_memory_coarse_source_prediction_receipt_v1"
DECODER_HEIGHT = 756
DECODER_WIDTH = 1008
SOURCE_VIEW_COUNT = 3


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    )
    digest.update(b"\0")
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _scene(value: Mapping[str, Any], scene_id: str) -> dict[str, Any]:
    scenes = value.get("scenes")
    if not isinstance(scenes, Mapping) or not isinstance(scenes.get(scene_id), Mapping):
        raise ValueError(f"source inventory does not contain scene {scene_id!r}")
    return dict(scenes[scene_id])


def _load_binary_tensor(
    tensors: Mapping[str, Any],
    tensor_sha: Mapping[str, Any],
    name: str,
) -> torch.Tensor:
    value = tensors.get(name)
    if (
        not torch.is_tensor(value)
        or value.device.type != "cpu"
        or value.dtype != torch.bool
        or value.shape != (DECODER_HEIGHT, DECODER_WIDTH)
        or _tensor_sha256(value.contiguous()) != tensor_sha.get(name)
    ):
        raise ValueError(f"reference completion tensor {name!r} differs")
    return value.contiguous()


def _save_mask(path: Path, value: torch.Tensor) -> dict[str, Any]:
    mask = torch.as_tensor(value, device="cpu").bool().numpy()
    buffer = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(buffer, format="PNG")
    write_bytes_noclobber(path, buffer.getvalue())
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "pixels": int(mask.sum()),
        "height": int(mask.shape[0]),
        "width": int(mask.shape[1]),
    }


def _upsample_seed(value: torch.Tensor) -> torch.Tensor:
    seed = torch.as_tensor(value, device="cpu").bool()
    return (
        F.interpolate(
            seed.float()[None, None],
            size=(DECODER_HEIGHT, DECODER_WIDTH),
            mode="nearest",
        )[0, 0]
        > 0.5
    ).contiguous()


def _lift_reference_positive_anchor(
    reference_mask: torch.Tensor,
    assignment: Mapping[str, torch.Tensor],
    *,
    feature_height: int,
    feature_width: int,
    num_gaussians: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    feature_mask = (
        F.interpolate(
            reference_mask.float()[None, None],
            size=(int(feature_height), int(feature_width)),
            mode="nearest",
        )[0, 0]
        > 0.5
    ).reshape(-1)
    gaussian_ids = torch.as_tensor(assignment["gaussian_ids"]).long().reshape(-1)
    pixel_ids = torch.as_tensor(assignment["pixel_ids"]).long().reshape(-1)
    weights = torch.as_tensor(assignment["weights"]).float().reshape(-1)
    if (
        gaussian_ids.shape != pixel_ids.shape
        or gaussian_ids.shape != weights.shape
        or gaussian_ids.numel() == 0
        or int(gaussian_ids.min()) < 0
        or int(gaussian_ids.max()) >= int(num_gaussians)
        or int(pixel_ids.min()) < 0
        or int(pixel_ids.max()) >= feature_mask.numel()
        or not bool(torch.isfinite(weights).all())
        or bool((weights <= 0).any())
    ):
        raise ValueError("reference source assignment is malformed")
    positive_hits = feature_mask[pixel_ids]
    positive_mass = torch.zeros(int(num_gaussians), dtype=torch.float32)
    positive_mass.index_add_(0, gaussian_ids, weights * positive_hits.float())
    probability = (positive_mass > 0).float()
    confidence = -torch.expm1(-positive_mass)
    return probability, confidence, {
        "reference_feature_positive_pixels": int(feature_mask.sum()),
        "positive_anchor_primitive_rows": int((positive_mass > 0).sum()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    inventory, inventory_sha, inventory_path = load_json_object(
        args.source_inventory,
        expected_sha256=args.source_inventory_sha256,
        label="multiview source inventory",
    )
    if (
        inventory.get("artifact_type")
        != "nvos_multiview_region_memory_source_inventory_v1"
        or inventory.get("status")
        != "source_rgb_and_assignment_authority_sealed_before_sam3_or_target_access"
        or inventory.get("global_safety", {}).get("target_rgb_content_opened_or_hashed")
        is not False
        or inventory.get("global_safety", {}).get("target_mask_opened") is not False
        or inventory.get("method_contract") != method_contract()
    ):
        raise ValueError("source inventory violates the region-memory contract")
    scene = _scene(inventory, args.scene_id)
    if scene.get("safety", {}).get("target_rgb_content_opened_or_hashed") is not False:
        raise ValueError("source inventory reports target RGB access")

    completion, completion_sha, completion_path = load_torch_mapping(
        args.reference_completion,
        expected_sha256=args.reference_completion_sha256,
        map_location="cpu",
        label="reference object completion",
    )
    completion_receipt, completion_receipt_sha, completion_receipt_path = load_json_object(
        args.reference_completion_receipt,
        expected_sha256=args.reference_completion_receipt_sha256,
        label="reference object completion receipt",
    )
    authority = completion.get("authority")
    tensors = completion.get("tensors")
    tensor_sha = completion.get("tensor_sha256")
    if (
        completion.get("artifact_type") != "radio_gs.nvos_sam3_reference_completion"
        or completion.get("schema_version") != 1
        or not isinstance(authority, Mapping)
        or not isinstance(tensors, Mapping)
        or not isinstance(tensor_sha, Mapping)
        or authority.get("scene_id") != args.scene_id
        or authority.get("frame_id") != scene.get("reference_frame_id")
        or authority.get("target_rgb_opened") is not False
        or authority.get("target_mask_opened") is not False
        or completion_receipt.get("artifact_sha256") != completion_sha
        or completion_receipt.get("target_rgb_opened") is not False
        or completion_receipt.get("target_mask_opened") is not False
        or completion_receipt.get("target_metric_opened") is not False
    ):
        raise ValueError("reference object completion authority differs")
    completed_positive = _load_binary_tensor(
        tensors, tensor_sha, "completed_positive"
    )
    raw_positive = _load_binary_tensor(tensors, tensor_sha, "raw_positive")
    raw_negative = _load_binary_tensor(tensors, tensor_sha, "raw_negative")
    if bool((raw_positive & raw_negative).any()):
        raise ValueError("reference positive and negative prompts overlap")
    if not bool(completed_positive[raw_positive].all()) or bool(
        completed_positive[raw_negative].any()
    ):
        raise ValueError("reference completion did not preserve signed prompts")

    responsibility_record = scene.get("assets", {}).get("responsibility")
    if not isinstance(responsibility_record, Mapping):
        raise ValueError("source responsibility record is missing")
    responsibility, responsibility_sha, responsibility_path = load_torch_mapping(
        responsibility_record["path"],
        expected_sha256=responsibility_record["sha256"],
        map_location="cpu",
        label="source multiview responsibility",
    )
    metadata = responsibility.get("metadata")
    assignments = responsibility.get("assignments")
    source_views = scene.get("source_views")
    if (
        responsibility.get("schema_version") != 1
        or not isinstance(metadata, Mapping)
        or not isinstance(assignments, list)
        or not isinstance(source_views, list)
        or len(assignments) != len(source_views)
        or not assignments
        or metadata.get("assignment_mode") != "raster_gaussian_top1"
        or metadata.get("registration_weight_mode") != "alpha_depth"
        or metadata.get("benchmark_images_opened") is not False
        or metadata.get("benchmark_masks_opened") is not False
    ):
        raise ValueError("source multiview responsibility contract differs")
    for index, source_view in enumerate(source_views):
        if source_view.get("assignment_view_index") != index:
            raise ValueError("source inventory assignment order changed")
    feature_height, feature_width = map(int, scene["feature_grid_hw"])
    maximum_gaussian_id = max(
        int(torch.as_tensor(value["gaussian_ids"]).max()) for value in assignments
    )
    num_gaussians = maximum_gaussian_id + 1
    reference_index = int(scene["reference_assignment_view_index"])
    anchor_probability, anchor_confidence, anchor_report = (
        _lift_reference_positive_anchor(
            completed_positive,
            assignments[reference_index],
            feature_height=feature_height,
            feature_width=feature_width,
            num_gaussians=num_gaussians,
        )
    )
    projections = [
        project_anchor_to_feature_view(
            anchor_probability,
            anchor_confidence,
            assignment,
            height=feature_height,
            width=feature_width,
        )
        for assignment in assignments
    ]
    selected = select_source_views(
        [str(value["frame_id"]) for value in source_views],
        projections,
        count=SOURCE_VIEW_COUNT,
        reference_frame_id=str(scene["reference_frame_id"]),
        forbidden_frame_ids=[str(value) for value in scene["forbidden_target_frame_ids"]],
    )

    output_root = Path(args.output_root).expanduser().absolute()
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for rank, selected_view in enumerate(selected):
        source = dict(source_views[selected_view.view_index])
        projection = projections[selected_view.view_index]
        decoder_mask = _upsample_seed(projection.seed)
        mask_path = output_root / "coarse_masks" / f"{rank:02d}_{source['frame_id']}.png"
        mask_record = _save_mask(mask_path, decoder_mask)
        rows.append(
            {
                "selection_rank": rank,
                "assignment_view_index": selected_view.view_index,
                "frame_id": selected_view.frame_id,
                "source_rgb_path": source["rgb_path"],
                "source_rgb_sha256": source["rgb_sha256"],
                "source_rgb_bytes": source["rgb_bytes"],
                "positive_anchor_coverage": selected_view.positive_anchor_coverage,
                "assignment_reliability": selected_view.assignment_reliability,
                "selection_score": selected_view.selection_score,
                "feature_seed_pixels": int(projection.seed.sum()),
                "feature_supported_pixels": int((projection.confidence > 0).sum()),
                "coarse_mask": mask_record,
            }
        )
    payload = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "status": "coarse_source_predictions_sealed_before_source_rgb_sam3_or_target_access",
        "scene_id": args.scene_id,
        "method": METHOD,
        "method_contract": method_contract(),
        "selection_count": SOURCE_VIEW_COUNT,
        "selection": rows,
        "selection_identity_sha256": canonical_json_sha256(
            [
                [row["selection_rank"], row["assignment_view_index"], row["frame_id"]]
                for row in rows
            ]
        ),
        "reference_anchor": {
            **anchor_report,
            "reference_frame_id": scene["reference_frame_id"],
            "reference_assignment_view_index": reference_index,
            "source_completion": {
                "path": str(completion_path),
                "sha256": completion_sha,
            },
            "source_completion_receipt": {
                "path": str(completion_receipt_path),
                "sha256": completion_receipt_sha,
            },
            "raw_positive_sha256": tensor_sha["raw_positive"],
            "raw_negative_sha256": tensor_sha["raw_negative"],
        },
        "source_inventory": {
            "path": str(inventory_path),
            "sha256": inventory_sha,
            "scene_source_rgb_inventory_sha256": scene["source_rgb_inventory_sha256"],
        },
        "source_responsibility": {
            "path": str(responsibility_path),
            "sha256": responsibility_sha,
            "assignment_mode": metadata["assignment_mode"],
            "registration_weight_mode": metadata["registration_weight_mode"],
            "feature_grid_hw": [feature_height, feature_width],
            "primitive_domain_minimum_size": num_gaussians,
        },
        "implementation": file_record(Path(__file__).resolve()),
        "source_access": {
            "source_rgb_opened": False,
            "official_sam3_loaded": False,
            "target_rgb_opened": False,
            "target_mask_opened": False,
            "target_metric_opened": False,
            "candidate_selected_with_gt": False,
        },
    }
    receipt_path = Path(args.output_receipt).expanduser().absolute()
    write_frozen_json(receipt_path, payload)
    return {
        "receipt": str(receipt_path.resolve()),
        "sha256": sha256_file(receipt_path),
        "selected_frame_ids": [row["frame_id"] for row in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-inventory", required=True)
    parser.add_argument("--source-inventory-sha256", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--reference-completion", required=True)
    parser.add_argument("--reference-completion-sha256", required=True)
    parser.add_argument("--reference-completion-receipt", required=True)
    parser.add_argument("--reference-completion-receipt-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-receipt", required=True)
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
