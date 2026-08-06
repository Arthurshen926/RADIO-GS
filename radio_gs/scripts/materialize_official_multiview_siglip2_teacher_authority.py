#!/usr/bin/env python3
"""Build a formal per-region/per-view official SigLIP2 teacher authority.

Every active descriptor is produced by re-encoding an RGB crop whose box is
derived from exact-marginal raster hits of the corresponding AcceptedV2
region.  The region anchor must be visible in that view.  The genuine first
C-RADIO 1280-D summary slot is projected by the same official
``_heads.siglip2-g`` module used for AcceptedV2 e0 and normalized in float32.

Whole-image summaries, cached RADIO summary tensors, semantic-cache payloads,
queries, labels, masks, and metrics are not accepted as inputs.  Preflight is
CPU-only; model execution happens only after every caller-SHA-bound authority
has passed and the output path has been proven absent.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import pil_to_tensor

from radio_gs.interfaces.frozen_radio_views import OfficialCropSummaryRuntime
from radio_gs.interfaces.factorized_primitive_state import (
    load_factorized_primitive_state,
)
from radio_gs.scripts import materialize_full_scalar_clean_training_shard as shard
from radio_gs.scripts.train_surface_region_full_scalar_residual import (
    canonical_physical_space_id,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_torch_mapping,
    sha256_file,
    write_torch_noclobber,
)


SOURCE_RGB_SCENE_SCHEMA = "radio_gs.clean_source_rgb_scene_authority.v1"
RESPONSIBILITY_SCHEMA = (
    "radio_gs.sparse_exact_marginal_responsibility_authority.v1"
)
RESPONSIBILITY_VIEW_SCHEMA = (
    "radio_gs.sparse_exact_marginal_responsibility_view.v1"
)
SCHEMA_VERSION = 1
UPSTREAM_CHAIN = (
    "clean source frame contract + sealed RGB frame manifest -> clean geometry "
    "and exact-marginal responsibility view shards -> AcceptedV2 canonical "
    "region authority -> official C-RADIO crop re-encoding -> this authority"
)


def source_rgb_scene_authority_contract() -> dict[str, Any]:
    return {
        "schema": SOURCE_RGB_SCENE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "frame_order": "strict_lexicographic_zero_padded_frame_id",
        "source_path": "safe_relative_path_below_caller_supplied_rgb_root",
        "image_identity": "caller_sha256_verified_exact_source_rgb_bytes",
        "field_frame_authority": (
            "sha256_of_scene_field_contract_frame_manifest_frame_and_rgb_identity"
        ),
        "query_independent": True,
    }


def _content_sha256(value: Mapping[str, Any]) -> str:
    content = dict(value)
    content.pop("authority_sha256", None)
    return canonical_json_sha256(content)


def _frame_authority(
    *,
    scene_id: str,
    field_source_contract_file_sha256: str,
    field_frame_manifest_sha256: str,
    frame_id: str,
    source_relative_path: str,
    source_image_sha256: str,
    source_image_height: int,
    source_image_width: int,
) -> str:
    return canonical_json_sha256(
        {
            "scene_id": scene_id,
            "field_source_contract_file_sha256": (
                field_source_contract_file_sha256
            ),
            "field_frame_manifest_sha256": field_frame_manifest_sha256,
            "frame_id": frame_id,
            "source_relative_path": source_relative_path,
            "source_image_sha256": source_image_sha256,
            "source_image_height": int(source_image_height),
            "source_image_width": int(source_image_width),
        }
    )


def build_source_rgb_scene_authority(
    *,
    scene_id: str,
    field_source_contract_file_sha256: str,
    field_frame_manifest_sha256: str,
    feature_frame_manifest_file_sha256: str,
    frame_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the portable source authority after an external RGB hash pass."""

    scene = str(scene_id)
    field_file = shard._require_sha256(
        field_source_contract_file_sha256,
        label="field source contract file",
    )
    field_frames = shard._require_sha256(
        field_frame_manifest_sha256, label="field frame manifest"
    )
    feature_file = shard._require_sha256(
        feature_frame_manifest_file_sha256,
        label="feature frame manifest file",
    )
    frames: list[dict[str, Any]] = []
    for raw in frame_records:
        if not isinstance(raw, Mapping) or set(raw) != {
            "frame_id", "source_relative_path", "source_image_sha256",
            "source_image_height", "source_image_width",
        }:
            raise ValueError("source RGB input frame record differs")
        frame_id = str(raw["frame_id"])
        relative = Path(str(raw["source_relative_path"]))
        if (
            not frame_id
            or relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
        ):
            raise ValueError("source RGB frame identity/path is unsafe")
        image_sha = shard._require_sha256(
            raw["source_image_sha256"], label="source RGB image"
        )
        image_height = int(raw["source_image_height"])
        image_width = int(raw["source_image_width"])
        if min(image_height, image_width) <= 0:
            raise ValueError("source RGB image dimensions must be positive")
        path = relative.as_posix()
        frames.append(
            {
                "frame_id": frame_id,
                "source_relative_path": path,
                "source_image_sha256": image_sha,
                "source_image_height": image_height,
                "source_image_width": image_width,
                "field_frame_authority_sha256": _frame_authority(
                    scene_id=scene,
                    field_source_contract_file_sha256=field_file,
                    field_frame_manifest_sha256=field_frames,
                    frame_id=frame_id,
                    source_relative_path=path,
                    source_image_sha256=image_sha,
                    source_image_height=image_height,
                    source_image_width=image_width,
                ),
            }
        )
    frames.sort(key=lambda item: item["frame_id"])
    payload: dict[str, Any] = {
        "schema": SOURCE_RGB_SCENE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "contract": source_rgb_scene_authority_contract(),
        "contract_sha256": canonical_json_sha256(
            source_rgb_scene_authority_contract()
        ),
        "scene_id": scene,
        "physical_space_id": canonical_physical_space_id(scene),
        "field_source_contract_file_sha256": field_file,
        "field_frame_manifest_sha256": field_frames,
        "feature_frame_manifest_file_sha256": feature_file,
        "frame_records": frames,
        "source_access": shard._authority_access(source_rgb_used=True),
    }
    payload["authority_sha256"] = _content_sha256(payload)
    return validate_source_rgb_scene_authority(payload)


def validate_source_rgb_scene_authority(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("source RGB scene authority must be a mapping")
    payload = dict(value)
    required = {
        "schema", "schema_version", "contract", "contract_sha256",
        "scene_id", "physical_space_id", "field_source_contract_file_sha256",
        "field_frame_manifest_sha256", "feature_frame_manifest_file_sha256",
        "frame_records", "source_access", "authority_sha256",
    }
    contract = source_rgb_scene_authority_contract()
    scene = str(payload.get("scene_id", ""))
    if (
        set(payload) != required
        or payload.get("schema") != SOURCE_RGB_SCENE_SCHEMA
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("contract") != contract
        or payload.get("contract_sha256") != canonical_json_sha256(contract)
        or payload.get("physical_space_id") != canonical_physical_space_id(scene)
        or payload.get("source_access")
        != shard._authority_access(source_rgb_used=True)
    ):
        raise ValueError("source RGB scene authority contract differs")
    field_file = shard._require_sha256(
        payload.get("field_source_contract_file_sha256"),
        label="field source contract file",
    )
    field_frames = shard._require_sha256(
        payload.get("field_frame_manifest_sha256"),
        label="field frame manifest",
    )
    shard._require_sha256(
        payload.get("feature_frame_manifest_file_sha256"),
        label="feature frame manifest file",
    )
    frames = payload.get("frame_records")
    if not isinstance(frames, list) or not frames:
        raise ValueError("source RGB scene authority has no frames")
    frozen: list[dict[str, Any]] = []
    for raw in frames:
        if not isinstance(raw, Mapping) or set(raw) != {
            "frame_id", "source_relative_path", "source_image_sha256",
            "source_image_height", "source_image_width",
            "field_frame_authority_sha256",
        }:
            raise ValueError("source RGB authority frame fields differ")
        frame_id = str(raw["frame_id"])
        relative = Path(str(raw["source_relative_path"]))
        image_sha = shard._require_sha256(
            raw["source_image_sha256"], label="source RGB image"
        )
        image_height = int(raw["source_image_height"])
        image_width = int(raw["source_image_width"])
        if (
            not frame_id
            or min(image_height, image_width) <= 0
            or relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or raw["field_frame_authority_sha256"]
            != _frame_authority(
                scene_id=scene,
                field_source_contract_file_sha256=field_file,
                field_frame_manifest_sha256=field_frames,
                frame_id=frame_id,
                source_relative_path=relative.as_posix(),
                source_image_sha256=image_sha,
                source_image_height=image_height,
                source_image_width=image_width,
            )
        ):
            raise ValueError("source RGB authority frame identity differs")
        frozen.append(
            {
                "frame_id": frame_id,
                "source_relative_path": relative.as_posix(),
                "source_image_sha256": image_sha,
                "source_image_height": image_height,
                "source_image_width": image_width,
                "field_frame_authority_sha256": str(
                    raw["field_frame_authority_sha256"]
                ),
            }
        )
    frame_ids = [item["frame_id"] for item in frozen]
    widths = {len(value) for value in frame_ids}
    if (
        frame_ids != sorted(frame_ids)
        or len(set(frame_ids)) != len(frame_ids)
        or len(widths) != 1
        or any(not value.isdecimal() for value in frame_ids)
        or any(value != f"{int(value):0{len(value)}d}" for value in frame_ids)
    ):
        raise ValueError("source RGB authority frame order differs")
    if payload.get("authority_sha256") != _content_sha256(payload):
        raise ValueError("source RGB scene authority content SHA-256 differs")
    return {**payload, "frame_records": frozen}


def validate_rgb_root(
    authority: Mapping[str, Any], root: str | Path
) -> dict[str, Path]:
    source = Path(root).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(
            f"source RGB root is missing: {source}. Required chain: {UPSTREAM_CHAIN}"
        )
    resolved: dict[str, Path] = {}
    for record in authority["frame_records"]:
        path = (source / record["source_relative_path"]).resolve()
        if source not in path.parents or not path.is_file():
            raise FileNotFoundError(f"source RGB frame is missing/unsafe: {path}")
        if sha256_file(path) != record["source_image_sha256"]:
            raise ValueError("source RGB frame SHA-256 differs")
        with Image.open(path) as image:
            width, height = image.size
        if (
            int(height) != int(record["source_image_height"])
            or int(width) != int(record["source_image_width"])
        ):
            raise ValueError("source RGB frame dimensions differ")
        resolved[str(record["frame_id"])] = path
    return resolved


def _validate_responsibility_authority(
    value: object,
    *,
    accepted: Mapping[str, Any],
    source_frames: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("exact-marginal responsibility authority must be a mapping")
    payload = dict(value)
    metadata = payload.get("metadata")
    views = payload.get("views")
    formula = payload.get("formula_contract")
    feature_height = int(metadata.get("feature_height", -1)) if isinstance(
        metadata, Mapping
    ) else -1
    feature_width = int(metadata.get("feature_width", -1)) if isinstance(
        metadata, Mapping
    ) else -1
    formula_sha256 = str(payload.get("formula_sha256", ""))
    if (
        set(payload) != {
            "schema", "schema_version", "formula_contract", "formula_sha256",
            "frame_indices", "metadata", "num_gaussians", "num_pixels",
            "total_hits", "views",
        }
        or
        payload.get("schema") != RESPONSIBILITY_SCHEMA
        or payload.get("schema_version") != 1
        or not isinstance(metadata, Mapping)
        or metadata.get("query_independent") is not True
        or metadata.get("benchmark_images_opened") is not False
        or metadata.get("benchmark_masks_opened") is not False
        or metadata.get("text_queries_opened") is not False
        or not isinstance(formula, Mapping)
        or formula.get("query_independent") is not True
        or formula.get("feature_independent") is not True
        or formula_sha256 != canonical_json_sha256(formula)
        or metadata.get("formula_sha256") != formula_sha256
        or min(feature_height, feature_width) <= 0
        or int(payload.get("num_pixels", -1))
        != feature_height * feature_width
        or int(payload.get("num_gaussians", -1))
        != int(accepted["accepted_base_valid"].numel())
        or str(metadata.get("xyz_sha256", ""))
        != str(accepted["geometry_fingerprint"]["xyz_sha256"])
        or not isinstance(views, list)
        or not views
    ):
        raise ValueError("exact-marginal responsibility authority differs")
    frozen_views: list[dict[str, Any]] = []
    last_key: tuple[int, int] | None = None
    for raw in views:
        if not isinstance(raw, Mapping) or set(raw) != {
            "frame_index", "num_hits", "relative_path", "sha256", "view_index"
        }:
            raise ValueError("responsibility view record differs")
        frame = int(raw["frame_index"])
        view = int(raw["view_index"])
        relative = Path(str(raw["relative_path"]))
        key = (frame, view)
        if (
            frame not in source_frames
            or view < 0
            or int(raw["num_hits"]) <= 0
            or relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or (last_key is not None and key <= last_key)
        ):
            raise ValueError("responsibility view identity/order differs")
        last_key = key
        frozen_views.append(
            {
                "frame_index": frame,
                "num_hits": int(raw["num_hits"]),
                "relative_path": relative.as_posix(),
                "sha256": shard._require_sha256(
                    raw["sha256"], label="responsibility view file"
                ),
                "view_index": view,
            }
        )
    if (
        payload.get("frame_indices")
        != [record["frame_index"] for record in frozen_views]
        or int(payload.get("total_hits", -1))
        != sum(record["num_hits"] for record in frozen_views)
    ):
        raise ValueError("responsibility aggregate view authority differs")
    return {**payload, "views": frozen_views}


def validate_responsibility_view(
    value: object,
    *,
    record: Mapping[str, Any],
    formula_sha256: str,
    num_gaussians: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema", "schema_version", "formula_sha256", "view_index",
        "frame_index", "num_gaussians", "num_pixels", "gaussian_ids",
        "pixel_ids", "base_weights",
    }:
        raise ValueError("responsibility view payload fields differ")
    payload = dict(value)
    gaussian = torch.as_tensor(payload["gaussian_ids"]).long().cpu()
    pixels = torch.as_tensor(payload["pixel_ids"]).long().cpu()
    weights = torch.as_tensor(payload["base_weights"]).float().cpu()
    count = int(record["num_hits"])
    num_pixels = int(payload.get("num_pixels", -1))
    if (
        payload.get("schema") != RESPONSIBILITY_VIEW_SCHEMA
        or payload.get("schema_version") != 1
        or payload.get("formula_sha256") != formula_sha256
        or int(payload.get("view_index", -1)) != int(record["view_index"])
        or int(payload.get("frame_index", -1)) != int(record["frame_index"])
        or int(payload.get("num_gaussians", -1)) != int(num_gaussians)
        or gaussian.shape != (count,)
        or pixels.shape != (count,)
        or weights.shape != (count,)
        or num_pixels <= 0
        or bool((gaussian < 0).any())
        or bool((gaussian >= num_gaussians).any())
        or bool((pixels < 0).any())
        or bool((pixels >= num_pixels).any())
        or not bool(torch.isfinite(weights).all())
        or bool((weights < 0).any())
        or not bool((weights > 0).any())
    ):
        raise ValueError("responsibility view tensor authority differs")
    return {
        **payload,
        "gaussian_ids": gaussian,
        "pixel_ids": pixels,
        "base_weights": weights,
    }


def region_view_crop_evidence(
    accepted: Mapping[str, Any],
    responsibility_view: Mapping[str, Any],
    *,
    feature_height: int,
    feature_width: int,
    image_height: int,
    image_width: int,
    region_batch_size: int = 4096,
    region_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map exact visible region hits to source-RGB tight crop boxes."""

    grid_h, grid_w = int(feature_height), int(feature_width)
    rgb_h, rgb_w = int(image_height), int(image_width)
    size = int(region_batch_size)
    if min(grid_h, grid_w, rgb_h, rgb_w, size) <= 0:
        raise ValueError("crop evidence dimensions/batch size must be positive")
    if int(responsibility_view["num_pixels"]) != grid_h * grid_w:
        raise ValueError("responsibility feature grid differs")
    num_gaussians = int(accepted["accepted_base_valid"].numel())
    gaussian = responsibility_view["gaussian_ids"].long()
    pixels = responsibility_view["pixel_ids"].long()
    weights = responsibility_view["base_weights"].float()
    positive = weights > 0
    gaussian = gaussian[positive]
    pixels = pixels[positive]
    counts = torch.bincount(gaussian, minlength=num_gaussians).long()
    inf = torch.iinfo(torch.long).max
    ymin = torch.full((num_gaussians,), inf, dtype=torch.long)
    xmin = torch.full((num_gaussians,), inf, dtype=torch.long)
    ymax = torch.full((num_gaussians,), -1, dtype=torch.long)
    xmax = torch.full((num_gaussians,), -1, dtype=torch.long)
    if gaussian.numel():
        y = torch.div(pixels, grid_w, rounding_mode="floor")
        x = torch.remainder(pixels, grid_w)
        ymin.scatter_reduce_(0, gaussian, y, reduce="amin", include_self=True)
        xmin.scatter_reduce_(0, gaussian, x, reduce="amin", include_self=True)
        ymax.scatter_reduce_(0, gaussian, y, reduce="amax", include_self=True)
        xmax.scatter_reduce_(0, gaussian, x, reduce="amax", include_self=True)
    all_rows = accepted["region_rows"].long().cpu()
    all_mask = accepted["token_mask"].bool().cpu()
    all_anchors = accepted["anchor_index"].long().cpu()
    if region_indices is None:
        selected = torch.arange(all_rows.shape[0], dtype=torch.long)
    else:
        selected = torch.as_tensor(region_indices).long().cpu().reshape(-1)
        if (
            selected.numel() <= 0
            or bool((selected < 0).any())
            or bool((selected >= all_rows.shape[0]).any())
            or selected.unique().numel() != selected.numel()
        ):
            raise ValueError("crop evidence region indices differ")
    rows = all_rows[selected]
    token_mask = all_mask[selected]
    anchors = all_anchors[selected]
    regions = int(rows.shape[0])
    boxes = torch.full((regions, 4), -1, dtype=torch.long)
    support_hits = torch.zeros(regions, dtype=torch.long)
    visible_mask = torch.zeros(regions, dtype=torch.bool)
    visible_primitive_counts = torch.zeros(regions, dtype=torch.long)
    for start in range(0, regions, size):
        stop = min(start + size, regions)
        batch_rows = rows[start:stop].clamp_min(0)
        batch_mask = token_mask[start:stop]
        primitive_visible = counts[batch_rows] > 0
        active_visible = batch_mask & primitive_visible
        batch_hits = (counts[batch_rows] * batch_mask).sum(dim=1)
        batch_primitive_counts = active_visible.sum(dim=1).long()
        anchor_rows = batch_rows[
            torch.arange(stop - start), anchors[start:stop]
        ]
        valid = (counts[anchor_rows] > 0) & (batch_hits > 0)
        batch_ymin = ymin[batch_rows].masked_fill(~active_visible, inf).amin(1)
        batch_xmin = xmin[batch_rows].masked_fill(~active_visible, inf).amin(1)
        batch_ymax = ymax[batch_rows].masked_fill(~active_visible, -1).amax(1)
        batch_xmax = xmax[batch_rows].masked_fill(~active_visible, -1).amax(1)
        # ``inf * rgb_size`` overflows signed long for a fully invisible
        # region.  Replace invalid intermediates before the integer mapping;
        # exact -1 padding is restored immediately afterwards.
        batch_ymin = torch.where(valid, batch_ymin, torch.zeros_like(batch_ymin))
        batch_xmin = torch.where(valid, batch_xmin, torch.zeros_like(batch_xmin))
        batch_ymax = torch.where(valid, batch_ymax, torch.zeros_like(batch_ymax))
        batch_xmax = torch.where(valid, batch_xmax, torch.zeros_like(batch_xmax))
        y0 = torch.div(batch_ymin * rgb_h, grid_h, rounding_mode="floor")
        x0 = torch.div(batch_xmin * rgb_w, grid_w, rounding_mode="floor")
        y1 = torch.div(
            (batch_ymax + 1) * rgb_h + grid_h - 1,
            grid_h,
            rounding_mode="floor",
        )
        x1 = torch.div(
            (batch_xmax + 1) * rgb_w + grid_w - 1,
            grid_w,
            rounding_mode="floor",
        )
        mapped = torch.stack((y0, x0, y1, x1), dim=-1)
        mapped[~valid] = -1
        boxes[start:stop] = mapped
        support_hits[start:stop] = torch.where(
            valid, batch_hits, torch.zeros_like(batch_hits)
        )
        visible_mask[start:stop] = valid
        visible_primitive_counts[start:stop] = torch.where(
            valid,
            batch_primitive_counts,
            torch.zeros_like(batch_primitive_counts),
        )
    if bool((boxes[visible_mask, 2] > rgb_h).any()) or bool(
        (boxes[visible_mask, 3] > rgb_w).any()
    ):
        raise RuntimeError("region-view crop exceeds source RGB")
    return boxes, support_hits, visible_primitive_counts, visible_mask


def encode_region_crops_with_summary_head_parity(
    runtime: OfficialCropSummaryRuntime,
    crops: torch.Tensor,
) -> torch.Tensor:
    """Encode crop summaries and prove the AcceptedV2 head identity."""

    _spatial, summary_token, descriptor = runtime.encode_training_pair(crops)
    explicit = F.normalize(
        runtime.summary_head(summary_token[:, None])[:, 0].float(),
        dim=-1,
        eps=1e-8,
    )
    if not torch.equal(descriptor, explicit):
        raise RuntimeError("official crop descriptor differs from AcceptedV2 head")
    return descriptor


def select_topk_region_views(
    visible: torch.Tensor,
    visible_primitive_counts: torch.Tensor,
    support_hit_counts: torch.Tensor,
    crop_boxes_tlbr: torch.Tensor,
    responsibility_view_indices: Sequence[int],
) -> dict[str, torch.Tensor]:
    """Select at most four source views per sampled region without labels."""

    mask = torch.as_tensor(visible).bool().cpu()
    primitive = torch.as_tensor(visible_primitive_counts).long().cpu()
    hits = torch.as_tensor(support_hit_counts).long().cpu()
    boxes = torch.as_tensor(crop_boxes_tlbr).long().cpu()
    sealed_view_indices = [int(value) for value in responsibility_view_indices]
    if (
        mask.ndim != 2
        or primitive.shape != mask.shape
        or hits.shape != mask.shape
        or boxes.shape != (*mask.shape, 4)
        or len(sealed_view_indices) != mask.shape[1]
        or len(set(sealed_view_indices)) != len(sealed_view_indices)
        or any(value < 0 for value in sealed_view_indices)
    ):
        raise ValueError("teacher top-K evidence layout differs")
    pair_rows: list[int] = []
    pair_views: list[int] = []
    pair_boxes: list[torch.Tensor] = []
    pair_hits: list[int] = []
    pair_primitives: list[int] = []
    for row in range(mask.shape[0]):
        candidates = torch.where(mask[row])[0].tolist()
        candidates.sort(
            key=lambda view: (
                -int(primitive[row, view]),
                -int(hits[row, view]),
                sealed_view_indices[view],
            )
        )
        chosen = candidates[: shard.TEACHER_VIEW_CAP_PER_REGION]
        if not chosen:
            raise RuntimeError("sampled teacher region lost every source view")
        for view in chosen:
            pair_rows.append(row)
            pair_views.append(view)
            pair_boxes.append(boxes[row, view])
            pair_hits.append(int(hits[row, view]))
            pair_primitives.append(int(primitive[row, view]))
    return {
        "pair_region_indices": torch.tensor(pair_rows, dtype=torch.long),
        "pair_view_indices": torch.tensor(pair_views, dtype=torch.long),
        "pair_crop_boxes_tlbr": torch.stack(pair_boxes).long(),
        "pair_support_hit_counts": torch.tensor(pair_hits, dtype=torch.long),
        "pair_visible_primitive_counts": torch.tensor(
            pair_primitives, dtype=torch.long
        ),
    }


def build_teacher_payload(
    *,
    scene_id: str,
    source_rgb_scene_authority_sha256: str,
    canonical_region_indices: torch.Tensor,
    region_fingerprints: Sequence[str],
    view_records: Sequence[Mapping[str, Any]],
    pair_region_indices: torch.Tensor,
    pair_view_indices: torch.Tensor,
    pair_descriptors: torch.Tensor,
    pair_crop_boxes_tlbr: torch.Tensor,
    pair_support_hit_counts: torch.Tensor,
    pair_visible_primitive_counts: torch.Tensor,
    selection_audit: Mapping[str, Any],
    input_authority: Mapping[str, Any],
) -> dict[str, Any]:
    model = shard.official_teacher_model_authority()
    contract = shard.teacher_observation_authority_contract()
    payload: dict[str, Any] = {
        "schema": shard.TEACHER_OBSERVATION_SCHEMA,
        "schema_version": shard.TEACHER_OBSERVATION_SCHEMA_VERSION,
        "contract": contract,
        "contract_sha256": canonical_json_sha256(contract),
        "scene_id": str(scene_id),
        "physical_space_id": canonical_physical_space_id(str(scene_id)),
        "source_rgb_scene_authority_sha256": source_rgb_scene_authority_sha256,
        "teacher_model_authority": model,
        "teacher_model_authority_sha256": canonical_json_sha256(model),
        "canonical_region_indices": torch.as_tensor(
            canonical_region_indices
        ).long().cpu().contiguous(),
        "region_fingerprints": [str(value) for value in region_fingerprints],
        "view_records": [dict(value) for value in view_records],
        "pair_region_indices": torch.as_tensor(
            pair_region_indices
        ).long().cpu().contiguous(),
        "pair_view_indices": torch.as_tensor(
            pair_view_indices
        ).long().cpu().contiguous(),
        "pair_descriptors": torch.as_tensor(
            pair_descriptors
        ).float().cpu().contiguous(),
        "pair_crop_boxes_tlbr": (
            torch.as_tensor(pair_crop_boxes_tlbr)
            .long()
            .cpu()
            .contiguous()
        ),
        "pair_support_hit_counts": (
            torch.as_tensor(pair_support_hit_counts)
            .long()
            .cpu()
            .contiguous()
        ),
        "pair_visible_primitive_counts": torch.as_tensor(
            pair_visible_primitive_counts
        ).long().cpu().contiguous(),
        "selection_audit": dict(selection_audit),
        "input_authority": dict(input_authority),
        "source_access": shard._authority_access(source_rgb_used=True),
    }
    payload["channel_sha256"] = shard.teacher_observation_channel_sha256(
        payload
    )
    return shard.validate_teacher_observation_authority(payload)


def _required_file(path: str | Path, expected_sha256: str, *, label: str):
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"{label} is missing: {source}. Required chain: {UPSTREAM_CHAIN}"
        )
    expected = shard._require_sha256(expected_sha256, label=label)
    if sha256_file(source) != expected:
        raise ValueError(f"{label} SHA-256 differs")
    return source


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    source_path = _required_file(
        args.source_rgb_scene_authority,
        args.expected_source_rgb_scene_authority_sha256,
        label="source RGB scene authority",
    )
    accepted_path = _required_file(
        args.accepted_region_authority,
        args.expected_accepted_region_authority_sha256,
        label="AcceptedV2 canonical region authority",
    )
    state_path = _required_file(
        args.factorized_primitive_state,
        args.expected_factorized_primitive_state_sha256,
        label="factorized primitive state",
    )
    responsibility_path = _required_file(
        args.exact_marginal_responsibility_authority,
        args.expected_exact_marginal_responsibility_authority_sha256,
        label="exact-marginal responsibility authority",
    )
    radio_path = _required_file(
        args.official_radio_checkpoint,
        args.expected_official_radio_checkpoint_sha256,
        label="official C-RADIOv4-H checkpoint",
    )
    if sha256_file(radio_path) != shard.OFFICIAL_RADIO_CHECKPOINT_SHA256:
        raise ValueError("official RADIO singleton authority differs")
    source_value, _, _ = load_json_object(
        source_path,
        expected_sha256=args.expected_source_rgb_scene_authority_sha256,
        label="source RGB scene authority",
    )
    source = validate_source_rgb_scene_authority(source_value)
    accepted_value, _, _ = load_torch_mapping(
        accepted_path,
        expected_sha256=args.expected_accepted_region_authority_sha256,
        map_location="cpu",
        label="AcceptedV2 canonical region authority",
    )
    accepted = shard.validate_accepted_region_authority(accepted_value)
    geometry_input = accepted["input_authority"]["geometry_authority"]
    if sha256_file(state_path) != geometry_input[
        "factorized_primitive_state_file_sha256"
    ]:
        raise ValueError("teacher state and AcceptedV2 state lineage differ")
    state = load_factorized_primitive_state(
        state_path,
        expected_sha256=sha256_file(state_path),
        expected_field_checkpoint_sha256=geometry_input[
            "factorized_field_checkpoint_file_sha256"
        ],
        expected_factorized_radio_cache_sha256=geometry_input[
            "factorized_radio_cache_file_sha256"
        ],
    )
    if state.metadata["geometry_fingerprint"] != accepted["geometry_fingerprint"]:
        raise ValueError("teacher state and AcceptedV2 geometry differ")
    if accepted["scene_id"] != source["scene_id"]:
        raise ValueError("source RGB and AcceptedV2 scenes differ")
    rgb_paths = validate_rgb_root(source, args.source_rgb_root)
    by_frame = {int(item["frame_id"]): item for item in source["frame_records"]}
    if len(by_frame) != len(source["frame_records"]):
        raise ValueError("source RGB numeric frame IDs collide")
    responsibility_value, _, _ = load_json_object(
        responsibility_path,
        expected_sha256=(
            args.expected_exact_marginal_responsibility_authority_sha256
        ),
        label="exact-marginal responsibility authority",
    )
    responsibility = _validate_responsibility_authority(
        responsibility_value,
        accepted=accepted,
        source_frames=by_frame,
    )
    validated_view_records: list[dict[str, Any]] = []
    view_records: list[dict[str, Any]] = []
    accepted_rows = accepted["region_rows"].long().cpu()
    accepted_anchors = accepted["anchor_index"].long().cpu()
    anchor_rows = accepted_rows[
        torch.arange(accepted_rows.shape[0]), accepted_anchors
    ]
    anchor_visible = torch.zeros(accepted_rows.shape[0], dtype=torch.bool)
    sampled = torch.arange(accepted_rows.shape[0], dtype=torch.long)
    sample_count = int(sampled.numel())
    view_count = len(responsibility["views"])
    evidence_visible = torch.zeros(sample_count, view_count, dtype=torch.bool)
    evidence_primitives = torch.zeros(sample_count, view_count, dtype=torch.long)
    evidence_hits = torch.zeros(sample_count, view_count, dtype=torch.long)
    evidence_boxes = torch.full(
        (sample_count, view_count, 4), -1, dtype=torch.long
    )
    feature_h = int(responsibility["metadata"]["feature_height"])
    feature_w = int(responsibility["metadata"]["feature_width"])
    for view_position, record in enumerate(responsibility["views"]):
        view_path = (responsibility_path.parent / record["relative_path"]).resolve()
        if responsibility_path.parent not in view_path.parents:
            raise ValueError("responsibility view path escapes its authority root")
        payload, _, _ = load_torch_mapping(
            view_path,
            expected_sha256=record["sha256"],
            map_location="cpu",
            label="exact-marginal responsibility view",
        )
        validated = validate_responsibility_view(
            payload,
            record=record,
            formula_sha256=responsibility["formula_sha256"],
            num_gaussians=int(responsibility["num_gaussians"]),
        )
        positive_gaussians = validated["gaussian_ids"][
            validated["base_weights"] > 0
        ]
        gaussian_visible = torch.bincount(
            positive_gaussians,
            minlength=int(responsibility["num_gaussians"]),
        ) > 0
        anchor_visible |= gaussian_visible[anchor_rows]
        validated_view_records.append(
            {**dict(record), "resolved_path": str(view_path)}
        )
        source_record = by_frame[int(record["frame_index"])]
        view_records.append({
            **dict(source_record),
            "feature_grid_height": feature_h,
            "feature_grid_width": feature_w,
            "responsibility_view_index": int(record["view_index"]),
            "responsibility_view_file_sha256": str(record["sha256"]),
        })
        crop_boxes, hits, primitives, visible = region_view_crop_evidence(
            accepted,
            validated,
            feature_height=feature_h,
            feature_width=feature_w,
            image_height=int(source_record["source_image_height"]),
            image_width=int(source_record["source_image_width"]),
            region_batch_size=int(args.region_batch_size),
            region_indices=sampled,
        )
        evidence_boxes[:, view_position] = crop_boxes
        evidence_hits[:, view_position] = hits
        evidence_primitives[:, view_position] = primitives
        evidence_visible[:, view_position] = visible
    if not bool(anchor_visible.all()):
        raise ValueError(
            "sampled AcceptedV2 authority contains an anchor without exact-"
            "marginal source visibility"
        )
    pair_evidence = select_topk_region_views(
        evidence_visible,
        evidence_primitives,
        evidence_hits,
        evidence_boxes,
        [int(record["view_index"]) for record in validated_view_records],
    )
    row_counts = torch.bincount(
        pair_evidence["pair_region_indices"], minlength=sample_count
    )
    selection_audit = {
        "accepted_selection_audit": dict(accepted["selection_audit"]),
        "pair_count": int(pair_evidence["pair_region_indices"].numel()),
        "maximum_views_per_region": int(row_counts.max()),
    }
    input_authority = {
        "source_rgb_scene_authority_file_sha256": sha256_file(source_path),
        "source_rgb_scene_authority_content_sha256": source[
            "authority_sha256"
        ],
        "factorized_primitive_state_file_sha256": sha256_file(state_path),
        "accepted_region_authority_file_sha256": sha256_file(accepted_path),
        "accepted_region_channel_sha256": canonical_json_sha256(
            accepted["channel_sha256"]
        ),
        "accepted_region_fingerprints_sha256": canonical_json_sha256(
            shard.stable_region_fingerprints(accepted)
        ),
        "exact_marginal_responsibility_authority_file_sha256": sha256_file(
            responsibility_path
        ),
        "official_radio_checkpoint_file_sha256": sha256_file(radio_path),
        "descriptor_definition": shard.official_teacher_descriptor_definition(),
    }
    shard.validate_official_teacher_input_authority(input_authority)
    return {
        "scene_id": source["scene_id"],
        "source": source,
        "accepted": accepted,
        "state": state,
        "responsibility": responsibility,
        "responsibility_view_records": validated_view_records,
        "canonical_region_indices": accepted["canonical_region_indices"],
        "pair_evidence": pair_evidence,
        "selection_audit": selection_audit,
        "view_records": view_records,
        "rgb_paths": {
            int(frame): path for frame, path in rgb_paths.items()
        },
        "radio_path": radio_path,
        "input_authority": input_authority,
    }


def preflight_summary(prepared: Mapping[str, Any]) -> dict[str, Any]:
    accepted = prepared["accepted"]
    return {
        "status": "ready",
        "scene_id": prepared["scene_id"],
        "regions": int(accepted["region_rows"].shape[0]),
        "source_views": len(prepared["view_records"]),
        "dense_region_view_pairs_forbidden": int(
            accepted["region_rows"].shape[0]
        ) * len(prepared["view_records"]),
        "selection_audit": dict(prepared["selection_audit"]),
        "maximum_sparse_descriptor_bytes": int(
            prepared["pair_evidence"]["pair_region_indices"].numel()
        ) * shard.trainer.DESCRIPTOR_DIM * 4,
        "descriptor_definition": shard.official_teacher_descriptor_definition(),
        "whole_image_summary_substitution_used": False,
        "precomputed_radio_summary_used": False,
        "semantic_cache_final_payload_used": False,
        "source_access": shard._authority_access(source_rgb_used=True),
        "outputs_written": False,
    }


def _crop_batch(
    image: torch.Tensor,
    boxes: torch.Tensor,
    *,
    resolution: int,
    device: torch.device,
) -> torch.Tensor:
    crops = []
    for top, left, bottom, right in boxes.tolist():
        crop = image[:, top:bottom, left:right]
        if crop.numel() == 0:
            raise RuntimeError("region-view crop is empty")
        crops.append(
            F.interpolate(
                crop[None],
                size=(int(resolution), int(resolution)),
                mode="bilinear",
                align_corners=False,
            )[0]
        )
    return torch.stack(crops).to(device)


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if not bool(args.preflight_only) and (output.exists() or output.is_symlink()):
        raise FileExistsError(f"refuses to clobber official teacher: {output}")
    prepared = preflight(args)
    summary = preflight_summary(prepared)
    if bool(args.preflight_only):
        return summary
    if min(
        int(args.crop_resolution),
        int(args.crop_batch_size),
        int(args.region_batch_size),
    ) <= 0:
        raise ValueError("crop resolution and batch sizes must be positive")
    device = torch.device(args.device)
    runtime = OfficialCropSummaryRuntime.load(
        checkpoint_path=prepared["radio_path"],
        radio_repo=args.radio_repo,
        version="c-radio_v4-h",
        device=device,
    )
    if runtime.radio_checkpoint_sha256 != shard.OFFICIAL_RADIO_CHECKPOINT_SHA256:
        raise RuntimeError("official crop runtime checkpoint differs")
    accepted = prepared["accepted"]
    pair_evidence = prepared["pair_evidence"]
    pair_views = pair_evidence["pair_view_indices"]
    pair_descriptors = torch.zeros(
        pair_views.numel(), shard.trainer.DESCRIPTOR_DIM, dtype=torch.float32
    )
    for view_position, record in enumerate(
        prepared["responsibility_view_records"]
    ):
        pair_positions = torch.where(pair_views == view_position)[0]
        if pair_positions.numel() == 0:
            continue
        image_path = prepared["rgb_paths"][int(record["frame_index"])]
        with Image.open(image_path) as source_image:
            image = pil_to_tensor(source_image.convert("RGB")).float().div_(255.0)
        for start in range(0, pair_positions.numel(), int(args.crop_batch_size)):
            positions = pair_positions[start : start + int(args.crop_batch_size)]
            crops = _crop_batch(
                image,
                pair_evidence["pair_crop_boxes_tlbr"][positions],
                resolution=int(args.crop_resolution),
                device=device,
            )
            descriptor = encode_region_crops_with_summary_head_parity(
                runtime, crops
            )
            pair_descriptors[positions] = descriptor.cpu()
    payload = build_teacher_payload(
        scene_id=prepared["scene_id"],
        source_rgb_scene_authority_sha256=prepared["source"]["authority_sha256"],
        canonical_region_indices=prepared["canonical_region_indices"],
        region_fingerprints=accepted["region_fingerprints"],
        view_records=prepared["view_records"],
        pair_region_indices=pair_evidence["pair_region_indices"],
        pair_view_indices=pair_evidence["pair_view_indices"],
        pair_descriptors=pair_descriptors,
        pair_crop_boxes_tlbr=pair_evidence["pair_crop_boxes_tlbr"],
        pair_support_hit_counts=pair_evidence["pair_support_hit_counts"],
        pair_visible_primitive_counts=pair_evidence[
            "pair_visible_primitive_counts"
        ],
        selection_audit=prepared["selection_audit"],
        input_authority=prepared["input_authority"],
    )
    write_torch_noclobber(output, payload)
    return {
        **summary,
        "status": "materialized",
        "active_region_view_pairs": int(pair_views.numel()),
        "output": file_record(output),
        "outputs_written": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-rgb-scene-authority", required=True)
    parser.add_argument(
        "--expected-source-rgb-scene-authority-sha256", required=True
    )
    parser.add_argument("--source-rgb-root", required=True)
    parser.add_argument("--accepted-region-authority", required=True)
    parser.add_argument(
        "--expected-accepted-region-authority-sha256", required=True
    )
    parser.add_argument("--factorized-primitive-state", required=True)
    parser.add_argument(
        "--expected-factorized-primitive-state-sha256", required=True
    )
    parser.add_argument(
        "--exact-marginal-responsibility-authority", required=True
    )
    parser.add_argument(
        "--expected-exact-marginal-responsibility-authority-sha256",
        required=True,
    )
    parser.add_argument("--official-radio-checkpoint", required=True)
    parser.add_argument(
        "--expected-official-radio-checkpoint-sha256", required=True
    )
    parser.add_argument("--radio-repo", default="/root/RADIO")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop-resolution", type=int, default=384)
    parser.add_argument("--crop-batch-size", type=int, default=4)
    parser.add_argument("--region-batch-size", type=int, default=4096)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    print(json.dumps(materialize(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
