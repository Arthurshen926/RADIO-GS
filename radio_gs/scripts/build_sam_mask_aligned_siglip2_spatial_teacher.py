#!/usr/bin/env python3
"""Pool official SigLIP2 spatial descriptors inside source-SAM3 proposals.

SAM supplies object extent; the frozen official SigLIP2 adaptor supplies
query-independent identity.  Masks are area-resampled to the adaptor grid,
then per-pixel normalized features are pooled by fractional foreground support.
The companion context descriptor uses the proposal bounding-box shell.  This
builder never opens text, benchmark annotations, or evaluation RGB.
"""

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

from radio_gs.models.sam3_multiscale_hierarchy import unpack_masks
from radio_gs.utils.immutable_artifacts import sha256_file


SCHEMA = "radio_gs.sam_mask_aligned_siglip2_spatial_teacher.v1"


def pool_mask_aligned_spatial_descriptors(
    features: torch.Tensor,
    masks: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return foreground/context descriptors and their cosine agreement."""

    values = torch.as_tensor(features).float()
    support = torch.as_tensor(masks).bool()
    boxes = torch.as_tensor(boxes_xyxy).long()
    if values.ndim != 3 or support.ndim != 3:
        raise ValueError("features and masks must be [C,H,W] and [M,H,W]")
    if tuple(support.shape[1:]) != (int(image_height), int(image_width)):
        raise ValueError("mask raster differs from source image")
    if boxes.shape != (support.shape[0], 4):
        raise ValueError("proposal boxes must align with masks")
    if support.shape[0] == 0 or not bool(support.flatten(1).any(dim=1).all()):
        raise ValueError("proposal masks must be non-empty")
    channels, grid_height, grid_width = map(int, values.shape)
    pixels = F.normalize(values, dim=0, eps=1e-8).reshape(channels, -1).T
    foreground = F.interpolate(
        support.float().unsqueeze(1),
        size=(grid_height, grid_width),
        mode="area",
    )[:, 0]
    box_masks = torch.zeros_like(support)
    for index, (x0, y0, x1, y1) in enumerate(boxes.tolist()):
        if not (0 <= x0 < x1 <= image_width and 0 <= y0 < y1 <= image_height):
            raise ValueError("proposal box lies outside source image")
        box_masks[index, y0:y1, x0:x1] = True
    box_grid = F.interpolate(
        box_masks.float().unsqueeze(1),
        size=(grid_height, grid_width),
        mode="area",
    )[:, 0]
    context = (box_grid - foreground).clamp_min(0.0)

    def pooled(weights: torch.Tensor, fallback: torch.Tensor | None = None) -> torch.Tensor:
        flat = weights.flatten(1)
        mass = flat.sum(dim=1, keepdim=True)
        result = flat @ pixels
        if fallback is not None:
            missing = mass[:, 0] <= 1e-8
            result[missing] = fallback[missing]
            mass[missing] = 1.0
        return F.normalize(result / mass.clamp_min(1e-8), dim=-1, eps=1e-8)

    foreground_descriptor = pooled(foreground)
    context_descriptor = pooled(context, fallback=foreground_descriptor)
    agreement = (foreground_descriptor * context_descriptor).sum(dim=-1)
    return foreground_descriptor, context_descriptor, agreement


def _load_json(path: Path, *, label: str) -> tuple[dict, bytes]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is not one regular JSON file: {path}")
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be one JSON object")
    return dict(payload), raw


def _feature_records(manifest: dict) -> dict[int, dict]:
    bundle = manifest.get("output_bundle")
    if not isinstance(bundle, Mapping) or bundle.get("contract") != "radio-feature-output-bundle-v1":
        raise ValueError("source feature output bundle contract differs")
    records: dict[int, dict] = {}
    for raw in bundle.get("frames", []):
        if not isinstance(raw, Mapping) or not isinstance(raw.get("frame"), Mapping):
            raise ValueError("source feature frame record differs")
        frame = dict(raw["frame"])
        frame_id = int(frame.get("frame_idx", -1))
        if frame_id < 0 or frame_id in records:
            raise ValueError("source feature frame id repeats")
        tensors = raw.get("tensors")
        if not isinstance(tensors, list):
            raise ValueError("source feature tensor registry differs")
        siglip = [
            dict(tensor)
            for tensor in tensors
            if isinstance(tensor, Mapping)
            and str(tensor.get("relative_path", "")).startswith("siglip2/")
        ]
        if len(siglip) != 1:
            raise ValueError("source frame lacks one official SigLIP2 spatial tensor")
        records[frame_id] = {"frame": frame, "tensor": siglip[0]}
    return records


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    report_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or report_path.exists():
        raise FileExistsError(f"mask-aligned spatial teacher exists: {output}")
    mask_root = Path(args.mask_root).expanduser().resolve()
    mask_manifest_path = mask_root / str(args.mask_manifest_name)
    mask_manifest, mask_manifest_bytes = _load_json(
        mask_manifest_path, label="multiscale SAM3 manifest"
    )
    if (
        mask_manifest.get("contract")
        != "official-sam3-query-free-multiscale-hierarchy-manifest-v1"
        or dict(mask_manifest.get("generation_contract", {})).get("query_free") is not True
        or dict(mask_manifest.get("generation_contract", {})).get("official_decoder") is not True
    ):
        raise ValueError("official query-free multiscale SAM3 contract differs")
    selected_ids = [str(value) for value in mask_manifest.get("selected_image_ids", [])]
    image_records = mask_manifest.get("images")
    if not selected_ids or not isinstance(image_records, list) or len(image_records) != len(selected_ids):
        raise ValueError("multiscale SAM3 image axis differs")
    mask_by_id = {str(record.get("image_id", "")): dict(record) for record in image_records if isinstance(record, Mapping)}
    if list(mask_by_id) != selected_ids:
        raise ValueError("multiscale SAM3 image order differs")

    feature_manifest_path = Path(args.feature_manifest).expanduser().resolve()
    feature_manifest, feature_manifest_bytes = _load_json(
        feature_manifest_path, label="source feature manifest"
    )
    if str(feature_manifest.get("scene", "")) != str(args.scene):
        raise ValueError("source feature manifest scene differs")
    signature = feature_manifest.get("features")
    adaptors = signature.get("adaptors") if isinstance(signature, Mapping) else None
    if (
        not isinstance(adaptors, list)
        or not any(
            isinstance(value, Mapping)
            and value.get("name") == "siglip2-g"
            and int(value.get("dim", -1)) == 1536
            for value in adaptors
        )
    ):
        raise ValueError("official SigLIP2-G spatial feature signature differs")
    feature_root = feature_manifest_path.parent
    feature_by_frame = _feature_records(feature_manifest)

    descriptor_chunks: list[torch.Tensor] = []
    context_chunks: list[torch.Tensor] = []
    agreement_chunks: list[torch.Tensor] = []
    proposal_views: list[int] = []
    proposal_frames: list[int] = []
    records: list[dict[str, object]] = []
    for view_index, image_id in enumerate(selected_ids):
        if not image_id.startswith("frame_"):
            raise ValueError("multiscale image id is not canonical")
        frame_id = int(image_id[6:])
        if frame_id not in feature_by_frame:
            raise ValueError(f"source frame lacks official SigLIP2 features: {frame_id}")
        mask_record = mask_by_id[image_id]
        mask_path = Path(str(mask_record.get("output", ""))).expanduser().resolve()
        if mask_path.parent != mask_root or sha256_file(mask_path) != str(mask_record.get("output_sha256", "")):
            raise ValueError("multiscale mask cache byte binding differs")
        mask_payload = torch.load(mask_path, map_location="cpu", weights_only=False)
        metadata = mask_payload.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("multiscale mask metadata is absent")
        source_image = metadata.get("source_image")
        if not isinstance(source_image, Mapping) or str(source_image.get("image_id", "")) != image_id:
            raise ValueError("multiscale source image identity differs")
        height, width = (int(value) for value in mask_payload["mask_shape"])
        masks = torch.from_numpy(
            unpack_masks(torch.as_tensor(mask_payload["packed_masks"]), width=width)
        )
        boxes = torch.as_tensor(mask_payload["boxes_xyxy"]).long()

        feature_record = feature_by_frame[frame_id]
        tensor_record = feature_record["tensor"]
        feature_path = (feature_root / str(tensor_record["relative_path"])).resolve()
        if feature_root not in feature_path.parents or sha256_file(feature_path) != str(tensor_record.get("sha256", "")):
            raise ValueError("official SigLIP2 feature tensor byte binding differs")
        features = torch.load(feature_path, map_location="cpu", weights_only=False)
        if tuple(features.shape) != tuple(int(value) for value in tensor_record.get("shape", [])) or features.shape[0] != 1536:
            raise ValueError("official SigLIP2 spatial tensor shape differs")
        foreground, context, agreement = pool_mask_aligned_spatial_descriptors(
            features,
            masks,
            boxes,
            image_height=height,
            image_width=width,
        )
        count = int(foreground.shape[0])
        descriptor_chunks.append(foreground.half())
        context_chunks.append(context.half())
        agreement_chunks.append(agreement.float())
        proposal_views.extend([view_index] * count)
        proposal_frames.extend([frame_id] * count)
        records.append(
            {
                "image_id": image_id,
                "frame_id": frame_id,
                "source_view_index": view_index,
                "proposal_count": count,
                "mask_cache": str(mask_path),
                "mask_cache_sha256": str(mask_record["output_sha256"]),
                "siglip2_spatial_tensor": str(feature_path),
                "siglip2_spatial_tensor_sha256": str(tensor_record["sha256"]),
            }
        )
    descriptors = torch.cat(descriptor_chunks)
    context_descriptors = torch.cat(context_chunks)
    agreement = torch.cat(agreement_chunks)
    if not bool(torch.isfinite(descriptors).all()) or descriptors.shape[1] != 1536:
        raise ValueError("mask-aligned official SigLIP2 descriptors differ")
    payload = {
        "schema": SCHEMA,
        "schema_version": 1,
        "scene": str(args.scene),
        "descriptors": descriptors,
        "context_descriptors": context_descriptors,
        "foreground_context_cosine": agreement,
        "proposal_view_indices": torch.tensor(proposal_views, dtype=torch.long),
        "proposal_frame_indices": torch.tensor(proposal_frames, dtype=torch.long),
        "metadata": {
            "teacher_space": "official_siglip2_g_spatial_mask_aligned_pool",
            "descriptor_formula": "l2(mean(area_resample(mask)*l2(siglip2_spatial_pixel)))",
            "context_formula": "l2(mean((proposal_box-mask)*l2(siglip2_spatial_pixel)))",
            "query_independent": True,
            "source_only": True,
            "official_sam3_decoder": True,
            "benchmark_labels_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_vocabulary_opened": False,
            "evaluation_rgb_opened": False,
            "text_queries_opened": False,
            "mask_manifest": str(mask_manifest_path),
            "mask_manifest_sha256": hashlib.sha256(mask_manifest_bytes).hexdigest(),
            "feature_manifest": str(feature_manifest_path),
            "feature_manifest_sha256": hashlib.sha256(feature_manifest_bytes).hexdigest(),
            "source_records": records,
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
        "source_view_count": len(records),
        "proposal_count": int(descriptors.shape[0]),
        "descriptor_dim": int(descriptors.shape[1]),
        "mean_foreground_context_cosine": float(agreement.mean()),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--mask-root", required=True)
    parser.add_argument("--mask-manifest-name", default="manifest.json")
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    print(json.dumps(build(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
