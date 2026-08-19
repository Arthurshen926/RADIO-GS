#!/usr/bin/env python3
"""Lift query-free packed-bool official-SAM3 proposals through exact MPR."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from radio_gs.rendering.sparse_marginal_authority import (
    SPARSE_EXACT_MARGINAL_FORMULA_SHA256,
    sparse_exact_marginal_formula_contract,
)

from radio_gs.scripts.build_lerf_sam3_exact_mpr_memberships import (
    RESPONSIBILITY_SCHEMA,
    _float32_rows_sha256,
    _frame_id,
    exact_mpr_target_weights,
)
from radio_gs.scripts.materialize_official_multiview_siglip2_teacher_authority import (
    validate_responsibility_view,
)
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks
from radio_gs.utils.immutable_artifacts import canonical_json_sha256, sha256_file


SCHEMA = "radio_gs.lerf_query_free_sam3_exact_mpr_memberships.v1"
PREFLIGHT_SCHEMA = "radio_gs.lerf_query_free_sam3_p0_preflight.v1"
EXPECTED_P0_GENERATION = {
    "schema_version": 2,
    "source": "official_sam3_interactive_grid_multimask_hierarchy",
    "official_decoder": True,
    "query_free": True,
    "resolution": 1008,
    "dtype": "bfloat16",
    "minimum_quality": 0.70,
    "minimum_area_fraction": 0.001,
    "maximum_area_fraction": 0.80,
    "nms_iou": 0.85,
    "minimum_stability": 0.0,
    "stability_offset": 1.0,
    "deduplication": "containment_aware_near_duplicate_only",
    "duplicate_minimum_area_ratio": 0.90,
    "maximum_masks": 0,
    "proposal_set_type": "single_scale_point_grid_multimask",
    "multiscale_crop_pyramid": False,
    "hierarchy_parent_edges_materialized": False,
}


def validate_responsibility_authority(
    value: object,
    *,
    authority_path: Path,
    num_gaussians: int,
    xyz_sha256: str,
) -> dict[str, Any]:
    """Validate aggregate formula, view registry, and every shard digest."""

    if not isinstance(value, Mapping):
        raise ValueError("exact-MPR authority must be a mapping")
    payload = dict(value)
    metadata = payload.get("metadata")
    formula = payload.get("formula_contract")
    views = payload.get("views")
    formula_sha256 = str(payload.get("formula_sha256", ""))
    required = {
        "schema", "schema_version", "formula_contract", "formula_sha256",
        "frame_indices", "metadata", "num_gaussians", "num_pixels",
        "total_hits", "views",
    }
    if (
        set(payload) != required
        or payload.get("schema") != RESPONSIBILITY_SCHEMA
        or int(payload.get("schema_version", -1)) != 1
        or not isinstance(metadata, Mapping)
        or metadata.get("query_independent") is not True
        or any(metadata.get(key) is not False for key in (
            "benchmark_images_opened", "benchmark_masks_opened", "text_queries_opened",
        ))
        or formula != sparse_exact_marginal_formula_contract()
        or formula_sha256 != SPARSE_EXACT_MARGINAL_FORMULA_SHA256
        or formula_sha256 != canonical_json_sha256(formula)
        or metadata.get("formula_sha256") != formula_sha256
        or int(payload.get("num_gaussians", -1)) != int(num_gaussians)
        or str(metadata.get("xyz_sha256", "")) != str(xyz_sha256)
        or not isinstance(views, list)
        or not views
    ):
        raise ValueError("exact-MPR aggregate authority differs")
    frozen: list[dict[str, Any]] = []
    last: tuple[int, int] | None = None
    for position, raw in enumerate(views):
        if not isinstance(raw, Mapping) or set(raw) != {
            "frame_index", "num_hits", "relative_path", "sha256", "view_index",
        }:
            raise ValueError("exact-MPR view registry record differs")
        record = dict(raw)
        frame = int(record["frame_index"])
        view_index = int(record["view_index"])
        relative = Path(str(record["relative_path"]))
        key = (frame, view_index)
        if (
            view_index != position
            or int(record["num_hits"]) < 0
            or relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or len(str(record["sha256"])) != 64
            or (last is not None and key <= last)
        ):
            raise ValueError("exact-MPR view identity/order/path differs")
        last = key
        view_path = (authority_path.parent / relative).resolve()
        if authority_path.parent not in view_path.parents or not view_path.is_file():
            raise ValueError("exact-MPR view shard escapes or is absent")
        if sha256_file(view_path) != str(record["sha256"]):
            raise ValueError("exact-MPR view shard SHA-256 differs")
        frozen.append({**record, "resolved_path": str(view_path)})
    frame_indices = [int(value) for value in payload.get("frame_indices", [])]
    if (
        frame_indices != [int(record["frame_index"]) for record in frozen]
        or frame_indices != [int(value) for value in metadata.get("selected_frame_indices", [])]
        or int(payload.get("total_hits", -1))
        != sum(int(record["num_hits"]) for record in frozen)
        or int(payload.get("num_pixels", -1))
        != int(metadata.get("feature_height", -1)) * int(metadata.get("feature_width", -1))
    ):
        raise ValueError("exact-MPR aggregate counts or frame axis differ")
    return {**payload, "views": frozen}


def validate_automatic_mask_contract(
    payload: dict[str, Any],
    *,
    mask_path: Path,
    manifest_record: dict[str, Any],
    source_image: Path,
    expected_checkpoint_sha256: str,
    expected_grid_size: int,
    source_image_sha256: str | None = None,
    mask_cache_sha256: str | None = None,
) -> str:
    """Bind one proposal cache to uniform producer and source-image bytes."""

    metadata = dict(payload.get("metadata", {}))
    expected_contract = {
        **EXPECTED_P0_GENERATION,
        "checkpoint_sha256": str(expected_checkpoint_sha256),
        "grid_size": int(expected_grid_size),
    }
    if any(metadata.get(key) != expected for key, expected in expected_contract.items()):
        raise ValueError("automatic masks mix producer, checkpoint, grid, or schema contracts")
    if any(bool(metadata.get(key, False)) for key in (
        "labels_opened", "instances_opened", "text_opened",
    )):
        raise ValueError("automatic mask cache violates annotation-free provenance")
    resolved_image = Path(source_image).expanduser().resolve()
    resolved_mask = Path(mask_path).expanduser().resolve()
    source_sha256 = (
        str(source_image_sha256)
        if source_image_sha256 is not None
        else sha256_file(resolved_image)
    )
    output_sha256 = (
        str(mask_cache_sha256)
        if mask_cache_sha256 is not None
        else sha256_file(resolved_mask)
    )
    if (
        Path(str(metadata.get("image", ""))).expanduser().resolve() != resolved_image
        or str(metadata.get("source_image_sha256", "")) != source_sha256
        or Path(str(manifest_record.get("image", ""))).expanduser().resolve()
        != resolved_image
        or Path(str(manifest_record.get("output", ""))).expanduser().resolve()
        != resolved_mask
        or str(manifest_record.get("source_image_sha256", "")) != source_sha256
        or str(manifest_record.get("output_sha256", "")) != output_sha256
    ):
        raise ValueError("automatic mask, manifest, and source RGB byte binding differ")
    return source_sha256


def validate_automatic_mask_payload(
    payload: Mapping[str, Any],
    *,
    image_height: int,
    image_width: int,
    expected_grid_size: int,
) -> dict[str, torch.Tensor]:
    """Validate packed masks and every row-aligned proposal attribute."""

    required = {
        "packed_masks", "mask_shape", "scores", "stability", "seed_xy",
        "prompt_index", "candidate_index", "boxes_xyxy",
        "proposal_area_fraction", "proposal_count_before_deduplication",
        "decoder_logits_available", "metadata",
    }
    if set(payload) != required:
        raise ValueError("automatic proposal payload fields differ")
    shape = payload.get("mask_shape")
    if (
        not isinstance(shape, (list, tuple))
        or len(shape) != 2
        or [int(value) for value in shape] != [int(image_height), int(image_width)]
        or min(int(image_height), int(image_width)) <= 0
    ):
        raise ValueError("automatic proposal raster differs from source RGB")
    packed = torch.as_tensor(payload.get("packed_masks"))
    expected_bytes = (int(image_width) + 7) // 8
    if (
        packed.dtype != torch.uint8
        or packed.ndim != 3
        or packed.shape[1:] != (int(image_height), expected_bytes)
    ):
        raise ValueError("automatic packed mask dtype or shape differs")
    remainder = int(image_width) % 8
    if remainder and packed.numel():
        unused_mask = (~((1 << remainder) - 1)) & 0xFF
        if bool(((packed[..., -1].to(torch.int16) & unused_mask) != 0).any()):
            raise ValueError("automatic packed mask padding bits are nonzero")
    masks = torch.from_numpy(unpack_masks(packed, int(image_width)))
    count = int(masks.shape[0])
    scores = torch.as_tensor(payload.get("scores"))
    stability = torch.as_tensor(payload.get("stability"))
    area = torch.as_tensor(payload.get("proposal_area_fraction"))
    seed_xy = torch.as_tensor(payload.get("seed_xy"))
    prompt = torch.as_tensor(payload.get("prompt_index"))
    candidate = torch.as_tensor(payload.get("candidate_index"))
    boxes = torch.as_tensor(payload.get("boxes_xyxy"))
    if (
        scores.shape != (count,)
        or stability.shape != (count,)
        or area.shape != (count,)
        or seed_xy.shape != (count, 2)
        or prompt.shape != (count,)
        or candidate.shape != (count,)
        or boxes.shape != (count, 4)
        or not bool(torch.isfinite(scores.float()).all())
        or not bool(torch.isfinite(stability.float()).all())
        or not bool(torch.isfinite(area.float()).all())
        or not bool(torch.isfinite(seed_xy.float()).all())
        or bool(((scores.float() < 0) | (scores.float() > 1)).any())
        or bool(((stability.float() < 0) | (stability.float() > 1)).any())
        or bool(((area.float() < 0) | (area.float() > 1)).any())
        or bool((prompt.long() < 0).any())
        or bool((prompt.long() >= int(expected_grid_size) ** 2).any())
        or bool((candidate.long() < 0).any())
        or int(payload.get("proposal_count_before_deduplication", -1)) < count
        or not isinstance(payload.get("decoder_logits_available"), bool)
    ):
        raise ValueError("automatic proposal row attributes differ")
    if count:
        measured_area = masks.float().mean(dim=(1, 2))
        tolerance = 1.0 / (int(image_height) * int(image_width)) + 1e-7
        if not torch.allclose(area.float(), measured_area, rtol=0.0, atol=tolerance):
            raise ValueError("automatic proposal areas differ from packed masks")
        if bool((seed_xy[:, 0] < 0).any()) or bool((seed_xy[:, 0] >= image_width).any()):
            raise ValueError("automatic proposal x seed is outside source RGB")
        if bool((seed_xy[:, 1] < 0).any()) or bool((seed_xy[:, 1] >= image_height).any()):
            raise ValueError("automatic proposal y seed is outside source RGB")
        expected_x = (
            torch.remainder(prompt.long(), int(expected_grid_size)).float() + 0.5
        ) * float(image_width) / int(expected_grid_size)
        expected_y = (
            torch.div(prompt.long(), int(expected_grid_size), rounding_mode="floor").float()
            + 0.5
        ) * float(image_height) / int(expected_grid_size)
        expected_seed = torch.stack((expected_x, expected_y), dim=1)
        if not torch.allclose(seed_xy.float(), expected_seed, rtol=0.0, atol=1e-4):
            raise ValueError("automatic proposal seed and prompt index differ")
        for index, mask in enumerate(masks):
            y, x = torch.where(mask)
            if not len(x):
                raise ValueError("automatic proposal contains an empty mask")
            expected_box = torch.tensor(
                [x.min(), y.min(), x.max() + 1, y.max() + 1], dtype=torch.long
            )
            if not torch.equal(boxes[index].long(), expected_box):
                raise ValueError("automatic proposal box differs from packed mask")
    return {
        "masks": masks,
        "scores": scores.float(),
        "stability": stability.float(),
        "area": area.float(),
        "seed_xy": seed_xy.float(),
        "prompt_index": prompt.long(),
        "candidate_index": candidate.long(),
        "boxes_xyxy": boxes.long(),
    }


def validate_preflight_receipt(
    value: object,
    *,
    preflight_path: Path,
    scene: str,
    requested_frames: set[int],
    source_root: Path,
    responsibility_path: Path,
    primitive_path: Path,
    expected_generation_contract: Mapping[str, Any],
    xyz_sha256: str,
) -> dict[str, Any]:
    """Consume the SHA-bound source-to-MPR preflight, not just produce it."""

    if not isinstance(value, Mapping):
        raise ValueError("P0 preflight receipt is not an object")
    receipt = dict(value)
    authorities = receipt.get("authorities")
    source_images = receipt.get("source_images")
    if (
        receipt.get("schema") != PREFLIGHT_SCHEMA
        or receipt.get("status") != "ready_for_query_free_sparse_p0_pilot_generation"
        or str(receipt.get("scene", "")) != str(scene)
        or set(int(value) for value in receipt.get("source_frame_ids", []))
        != requested_frames
        or int(receipt.get("source_frame_count", -1)) != len(requested_frames)
        or receipt.get("generation_contract") != dict(expected_generation_contract)
        or not isinstance(authorities, Mapping)
        or not isinstance(source_images, list)
        or len(source_images) != len(requested_frames)
        or dict(receipt.get("row_contract", {})).get("xyz_sha256") != xyz_sha256
    ):
        raise ValueError("P0 preflight identity or generation contract differs")

    def authority_record(name: str, expected_path: Path | None = None) -> dict[str, Any]:
        raw = authorities.get(name)
        if not isinstance(raw, Mapping):
            raise ValueError(f"P0 preflight {name} authority is absent")
        record = dict(raw)
        path = Path(str(record.get("path", ""))).expanduser().resolve()
        if expected_path is not None and path != expected_path:
            raise ValueError(f"P0 preflight {name} authority path differs")
        if not path.is_file() or sha256_file(path) != str(record.get("sha256", "")):
            raise ValueError(f"P0 preflight {name} authority SHA-256 differs")
        return {**record, "resolved_path": str(path)}

    authority_record("responsibility", responsibility_path)
    authority_record("primitive_query_cache", primitive_path)
    authority_record("support_graph")
    authority_record("official_sam3_checkpoint")
    authority_record("construction_frame_manifest")
    by_frame: dict[int, dict[str, Any]] = {}
    for raw in source_images:
        if not isinstance(raw, Mapping):
            raise ValueError("P0 preflight source RGB record differs")
        record = dict(raw)
        frame_id = int(record.get("frame_id", -1))
        path = Path(str(record.get("path", ""))).expanduser().resolve()
        if (
            frame_id in by_frame
            or frame_id not in requested_frames
            or source_root not in path.parents
            or not path.is_file()
            or sha256_file(path) != str(record.get("sha256", ""))
            or min(int(record.get("height", 0)), int(record.get("width", 0))) <= 0
        ):
            raise ValueError("P0 preflight source RGB binding differs")
        by_frame[frame_id] = record
    if set(by_frame) != requested_frames:
        raise ValueError("P0 preflight source RGB frame set differs")
    return {**receipt, "source_images_by_frame": by_frame, "path": str(preflight_path)}


def lift_binary_masks_with_exact_mpr(
    masks: torch.Tensor,
    gaussian_ids: torch.Tensor,
    pixel_ids: torch.Tensor,
    base_weights: torch.Tensor,
    *,
    num_gaussians: int,
    feature_height: int,
    feature_width: int,
    min_membership: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    values = torch.as_tensor(masks).bool()
    if values.ndim != 3:
        raise ValueError("automatic masks must have shape [M,H,W]")
    gids = torch.as_tensor(gaussian_ids).long()
    pixels = torch.as_tensor(pixel_ids).long()
    base = torch.as_tensor(base_weights).float()
    if not (gids.shape == pixels.shape == base.shape):
        raise ValueError("exact-MPR sparse rows differ")
    if not 0.0 < float(min_membership) <= 1.0:
        raise ValueError("min_membership must lie in (0,1]")
    if (
        not bool(torch.isfinite(base).all())
        or bool((base <= 0).any())
        or bool((base > 1).any())
    ):
        raise ValueError("exact-MPR base weights must lie in (0,1]")
    denominator = base.new_zeros((int(num_gaussians),))
    if not len(gids):
        empty = torch.empty(0, dtype=torch.long, device=gids.device)
        return empty, empty.clone(), torch.empty(0, device=gids.device), denominator
    aligned = F.interpolate(
        values.float().unsqueeze(1),
        size=(int(feature_height), int(feature_width)),
        mode="nearest",
    )[:, 0].flatten(1)
    target = exact_mpr_target_weights(
        pixels, base, num_pixels=int(feature_height) * int(feature_width)
    )
    denominator.index_add_(0, gids, target)
    if not len(values):
        empty = torch.empty(0, dtype=torch.long, device=gids.device)
        return empty, empty.clone(), torch.empty(0, device=gids.device), denominator
    rows_out: list[torch.Tensor] = []
    proposals_out: list[torch.Tensor] = []
    weights_out: list[torch.Tensor] = []
    for proposal_index in range(len(aligned)):
        numerator = target.new_zeros((int(num_gaussians),))
        numerator.index_add_(0, gids, target * aligned[proposal_index, pixels])
        membership = (numerator / denominator.clamp_min(1e-12)).clamp_(0.0, 1.0)
        if not bool(torch.isfinite(membership).all()):
            raise ValueError("exact-MPR proposal membership is non-finite")
        keep = membership >= float(min_membership)
        rows = torch.where(keep)[0]
        if len(rows):
            rows_out.append(rows)
            proposals_out.append(torch.full_like(rows, proposal_index))
            weights_out.append(membership[rows])
    if not rows_out:
        empty = torch.empty(0, dtype=torch.long, device=gids.device)
        return empty, empty.clone(), torch.empty(0, device=gids.device), denominator
    return (
        torch.cat(rows_out), torch.cat(proposals_out), torch.cat(weights_out), denominator,
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    report_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or report_path.exists():
        raise FileExistsError(f"output already exists: {output}")
    authority_path = Path(args.responsibility_authority).expanduser().resolve()
    authority_bytes = authority_path.read_bytes()
    raw_authority = json.loads(authority_bytes)

    primitive_path = Path(args.primitive_cache).expanduser().resolve()
    primitive = torch.load(primitive_path, map_location="cpu", weights_only=False)
    xyz = torch.as_tensor(primitive.get("xyz")).float().cpu()
    num_gaussians = len(xyz)
    xyz_sha256 = _float32_rows_sha256(xyz)
    authority = validate_responsibility_authority(
        raw_authority,
        authority_path=authority_path,
        num_gaussians=num_gaussians,
        xyz_sha256=xyz_sha256,
    )
    metadata = dict(authority["metadata"])
    frame_ids = [int(value) for value in authority["frame_indices"]]
    feature_height = int(metadata.get("feature_height", 0))
    feature_width = int(metadata.get("feature_width", 0))
    if feature_height <= 0 or feature_width <= 0:
        raise ValueError("exact-MPR feature raster differs")

    frame_to_view = {frame: index for index, frame in enumerate(frame_ids)}
    mask_root = Path(args.mask_root).expanduser().resolve()
    manifest_path = mask_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"automatic-mask manifest is absent: {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_images = list(manifest.get("images", []))
    if not manifest_images:
        raise ValueError("automatic-mask manifest has no image records")
    mask_paths = sorted(mask_root.glob("frame_*.pt"), key=_frame_id)
    if not mask_paths:
        raise FileNotFoundError(f"query-free packed-bool masks are absent: {mask_root}")
    requested = {
        int(value) for value in str(args.frame_ids).replace(",", " ").split() if value
    }
    if requested:
        mask_paths = [path for path in mask_paths if _frame_id(path) in requested]
    if requested != {_frame_id(path) for path in mask_paths}:
        raise ValueError("requested query-free frame set is not complete")
    if len(requested) != 8:
        raise ValueError("query-free P0 pilot requires exactly eight source frames")

    source_root = Path(args.source_image_root).expanduser().resolve()
    expected_checkpoint_sha256 = str(args.expected_checkpoint_sha256)
    if len(expected_checkpoint_sha256) != 64:
        raise ValueError("expected official SAM3 checkpoint SHA-256 differs")
    expected_grid_size = int(args.expected_grid_size)
    if expected_grid_size <= 0:
        raise ValueError("expected grid size must be positive")
    expected_generation_contract = {
        **EXPECTED_P0_GENERATION,
        "checkpoint_sha256": expected_checkpoint_sha256,
        "grid_size": expected_grid_size,
    }
    if manifest.get("generation_contract") != expected_generation_contract:
        raise ValueError("automatic-mask manifest generation contract differs")
    preflight_path = Path(args.preflight).expanduser().resolve()
    preflight_bytes = preflight_path.read_bytes()
    preflight = validate_preflight_receipt(
        json.loads(preflight_bytes),
        preflight_path=preflight_path,
        scene=str(args.scene),
        requested_frames=requested,
        source_root=source_root,
        responsibility_path=authority_path,
        primitive_path=primitive_path,
        expected_generation_contract=expected_generation_contract,
        xyz_sha256=xyz_sha256,
    )
    manifest_by_frame: dict[int, dict[str, Any]] = {}
    for record in manifest_images:
        image = Path(str(record.get("image", ""))).expanduser().resolve()
        try:
            frame_id = _frame_id(image)
        except (TypeError, ValueError) as error:
            raise ValueError("automatic-mask manifest image frame differs") from error
        if frame_id in manifest_by_frame:
            raise ValueError("automatic-mask manifest repeats a source frame")
        manifest_by_frame[frame_id] = dict(record)
    if set(manifest_by_frame) != requested:
        raise ValueError("automatic-mask manifest and requested frame set differ")

    device = torch.device(args.device)
    row_chunks: list[torch.Tensor] = []
    proposal_chunks: list[torch.Tensor] = []
    weight_chunks: list[torch.Tensor] = []
    proposal_views: list[int] = []
    proposal_scores: list[float] = []
    proposal_area: list[float] = []
    proposal_stability: list[float] = []
    proposal_boxes: list[torch.Tensor] = []
    proposal_seeds: list[torch.Tensor] = []
    proposal_prompt_indices: list[int] = []
    proposal_candidate_indices: list[int] = []
    records: list[dict[str, Any]] = []
    source_image_sha256s: list[str] = []
    view_denominators: list[torch.Tensor] = []
    proposal_offset = 0
    for source_view, mask_path in enumerate(mask_paths):
        frame_id = _frame_id(mask_path)
        if frame_id not in frame_to_view:
            raise ValueError(f"automatic mask frame is not a source MPR frame: {frame_id}")
        mask_bytes = mask_path.read_bytes()
        mask_sha256 = hashlib.sha256(mask_bytes).hexdigest()
        payload = torch.load(io.BytesIO(mask_bytes), map_location="cpu", weights_only=False)
        expected_images = [
            source_root / f"frame_{frame_id:05d}{suffix}"
            for suffix in (".jpg", ".jpeg", ".png", ".JPG", ".PNG")
        ]
        matches = [path.resolve() for path in expected_images if path.is_file()]
        if len(matches) != 1:
            raise FileNotFoundError(f"cannot uniquely resolve source RGB frame {frame_id}")
        source_image = matches[0]
        source_bytes = source_image.read_bytes()
        source_image_sha256 = hashlib.sha256(source_bytes).hexdigest()
        with Image.open(io.BytesIO(source_bytes)) as image:
            image_width, image_height = image.size
        manifest_record = manifest_by_frame[frame_id]
        validated_source_sha256 = validate_automatic_mask_contract(
            payload,
            mask_path=mask_path,
            manifest_record=manifest_record,
            source_image=source_image,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            expected_grid_size=expected_grid_size,
            source_image_sha256=source_image_sha256,
            mask_cache_sha256=mask_sha256,
        )
        preflight_source = preflight["source_images_by_frame"][frame_id]
        if (
            str(preflight_source.get("sha256", "")) != validated_source_sha256
            or int(preflight_source.get("height", -1)) != int(image_height)
            or int(preflight_source.get("width", -1)) != int(image_width)
        ):
            raise ValueError("source RGB differs from consumed P0 preflight")
        source_image_sha256 = validated_source_sha256
        source_image_sha256s.append(source_image_sha256)
        attributes = validate_automatic_mask_payload(
            payload,
            image_height=int(image_height),
            image_width=int(image_width),
            expected_grid_size=expected_grid_size,
        )
        height, width = int(image_height), int(image_width)
        masks = attributes["masks"].to(device)
        scores = attributes["scores"].cpu()
        area = attributes["area"].cpu()

        authority_view = frame_to_view[frame_id]
        authority_record = authority["views"][authority_view]
        view_path = Path(str(authority_record["resolved_path"]))
        view = validate_responsibility_view(
            torch.load(view_path, map_location="cpu", weights_only=False),
            record=authority_record,
            formula_sha256=str(authority["formula_sha256"]),
            num_gaussians=num_gaussians,
        )
        if int(view.get("num_pixels", -1)) != feature_height * feature_width:
            raise ValueError("responsibility view feature raster differs")
        rows, local_proposals, membership, denominator = lift_binary_masks_with_exact_mpr(
            masks,
            torch.as_tensor(view["gaussian_ids"]).to(device),
            torch.as_tensor(view["pixel_ids"]).to(device),
            torch.as_tensor(view["base_weights"]).to(device),
            num_gaussians=num_gaussians,
            feature_height=feature_height,
            feature_width=feature_width,
            min_membership=float(args.min_membership),
        )
        if (
            denominator.shape != (num_gaussians,)
            or not bool(torch.isfinite(denominator).all())
            or bool((denominator < 0).any())
            or not bool((denominator > 0).any())
        ):
            raise ValueError("per-view exact-MPR denominator support differs")
        view_denominators.append(denominator.float().cpu())
        if len(rows):
            row_chunks.append(rows.cpu())
            proposal_chunks.append((local_proposals + proposal_offset).cpu())
            weight_chunks.append(membership.float().cpu())
        proposal_views.extend([source_view] * len(masks))
        proposal_scores.extend(scores.tolist())
        proposal_area.extend(area.tolist())
        proposal_stability.extend(attributes["stability"].tolist())
        proposal_boxes.append(attributes["boxes_xyxy"].cpu())
        proposal_seeds.append(attributes["seed_xy"].cpu())
        proposal_prompt_indices.extend(attributes["prompt_index"].tolist())
        proposal_candidate_indices.extend(attributes["candidate_index"].tolist())
        records.append({
            "frame_id": frame_id,
            "source_view_index": source_view,
            "mask_cache": str(mask_path),
            "mask_cache_sha256": mask_sha256,
            "responsibility_view": str(view_path),
            "responsibility_view_sha256": sha256_file(view_path),
            "num_proposals": len(masks),
            "num_memberships": int(len(rows)),
            "mask_shape": [height, width],
            "source_image": str(source_image),
            "source_image_sha256": source_image_sha256,
        })
        proposal_offset += len(masks)

    payload = {
        "schema": SCHEMA,
        "schema_version": 1,
        "scene": str(args.scene),
        "num_rows": num_gaussians,
        "num_proposals": proposal_offset,
        "row_indices": torch.cat(row_chunks) if row_chunks else torch.empty(0, dtype=torch.long),
        "proposal_indices": (
            torch.cat(proposal_chunks) if proposal_chunks else torch.empty(0, dtype=torch.long)
        ),
        "weights": torch.cat(weight_chunks) if weight_chunks else torch.empty(0),
        "proposal_view_indices": torch.tensor(proposal_views, dtype=torch.long),
        "proposal_scores": torch.tensor(proposal_scores, dtype=torch.float32),
        "proposal_area_fraction": torch.tensor(proposal_area, dtype=torch.float32),
        "proposal_stability": torch.tensor(proposal_stability, dtype=torch.float32),
        "proposal_boxes_xyxy": (
            torch.cat(proposal_boxes) if proposal_boxes else torch.empty((0, 4), dtype=torch.long)
        ),
        "proposal_seed_xy": (
            torch.cat(proposal_seeds) if proposal_seeds else torch.empty((0, 2))
        ),
        "proposal_prompt_index": torch.tensor(proposal_prompt_indices, dtype=torch.long),
        "proposal_candidate_index": torch.tensor(proposal_candidate_indices, dtype=torch.long),
        "view_denominator": torch.stack(view_denominators),
        "view_observed": torch.stack(view_denominators) > 0,
        "metadata": {
            "query_independent_proposal_set": True,
            "query_independent_mask_hierarchy": False,
            "hierarchy_parent_edges_materialized": False,
            "proposal_set_type": "single_scale_point_grid_multimask",
            "official_decoder": True,
            "mask_tensor_semantics": "packed_boolean",
            "mask_raster_alignment": "nearest_label_resample_to_exact_mpr_raster",
            "membership_lifting": "exact_front_to_back_marginal_target_weight",
            "min_membership": float(args.min_membership),
            "feature_height": feature_height,
            "feature_width": feature_width,
            "xyz_sha256": xyz_sha256,
            "source_view_count": len(records),
            "view_support_semantics": (
                "per_view_exact_mpr_target_weight_denominator_and_positive_observed_mask"
            ),
            "source_records": records,
            "primitive_cache": str(primitive_path),
            "primitive_cache_sha256": sha256_file(primitive_path),
            "responsibility_authority": str(authority_path),
            "responsibility_authority_sha256": hashlib.sha256(authority_bytes).hexdigest(),
            "automatic_mask_manifest": str(manifest_path),
            "automatic_mask_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "preflight_receipt": str(preflight_path),
            "preflight_receipt_sha256": hashlib.sha256(preflight_bytes).hexdigest(),
            "official_sam3_checkpoint_sha256": expected_checkpoint_sha256,
            "source_image_root": str(source_root),
            "source_image_sha256s": source_image_sha256s,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "evaluation_rgb_opened": False,
            "text_queries_opened": False,
            "grid_size": expected_grid_size,
            "formal_stage_a_complete": False,
            "sparse_p0_pilot_complete": expected_grid_size == 12,
            "capability_track": (
                "query_free_sparse_p0_grid12_sha_bound_pilot"
                if expected_grid_size == 12
                else "query_free_p0_undercoverage_smoke_not_final_grid12"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "scene": str(args.scene),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "num_rows": num_gaussians,
        "num_proposals": proposal_offset,
        "num_memberships": int(payload["row_indices"].numel()),
        "source_view_count": len(records),
        "query_independent_proposal_set": True,
        "query_independent_mask_hierarchy": False,
        "grid_size": payload["metadata"]["grid_size"],
        "formal_stage_a_complete": False,
        "sparse_p0_pilot_complete": payload["metadata"]["sparse_p0_pilot_complete"],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--responsibility-authority", required=True)
    parser.add_argument("--mask-root", required=True)
    parser.add_argument("--frame-ids", required=True)
    parser.add_argument("--primitive-cache", required=True)
    parser.add_argument("--source-image-root", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-grid-size", type=int, required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--min-membership", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    print(json.dumps(build(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
