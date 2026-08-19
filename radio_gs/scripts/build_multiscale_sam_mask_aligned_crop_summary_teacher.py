#!/usr/bin/env python3
"""Encode multiscale source-SAM3 masks with official SigLIP2 crop summaries.

This is the direct SAM+CLIP-style identity path: official SAM3 fixes the mask,
the masked RGB crop is encoded by the frozen official C-RADIOv4 SigLIP2-G
summary adaptor, and an expanded unmasked crop supplies contextual evidence.
All descriptors are constructed query-free from sealed source RGB.
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

from PIL import Image
import torch
from torchvision.transforms.functional import pil_to_tensor

from radio_gs.interfaces.frozen_radio_views import OfficialCropSummaryRuntime
from radio_gs.models.sam3_multiscale_hierarchy import unpack_masks
from radio_gs.scripts.build_sam_mask_aligned_language_teacher import (
    build_crop_pairs,
    encode_in_batches,
)
from radio_gs.utils.immutable_artifacts import sha256_file


SCHEMA = "radio_gs.multiscale_sam_mask_aligned_crop_summary_teacher.v1"


def _load_manifest(path: Path) -> tuple[dict, bytes]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"multiscale manifest is not a regular file: {path}")
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("multiscale manifest must be one object")
    return dict(payload), raw


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    report_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or report_path.exists():
        raise FileExistsError(f"crop-summary teacher exists: {output}")
    mask_root = Path(args.mask_root).expanduser().resolve()
    manifest_path = mask_root / str(args.manifest_name)
    manifest, manifest_bytes = _load_manifest(manifest_path)
    generation = manifest.get("generation_contract")
    source_authority = manifest.get("source_authority")
    if (
        manifest.get("contract")
        != "official-sam3-query-free-multiscale-hierarchy-manifest-v1"
        or not isinstance(generation, Mapping)
        or generation.get("query_free") is not True
        or generation.get("official_decoder") is not True
        or generation.get("information_inputs")
        != ["registered_source_or_mapping_rgb"]
        or not isinstance(source_authority, Mapping)
    ):
        raise ValueError("official query-free multiscale SAM3 contract differs")
    authority_path = Path(str(source_authority.get("path", ""))).expanduser().resolve()
    if sha256_file(authority_path) != str(source_authority.get("sha256", "")):
        raise ValueError("source RGB authority SHA-256 differs")
    authority_payload = json.loads(authority_path.read_text(encoding="utf-8"))
    if (
        authority_payload.get("contract") != "sam3-query-free-source-rgb-authority-v1"
        or dict(authority_payload.get("information_policy", {}))
        != {
            "registered_source_rgb_only": True,
            "query_text_used": False,
            "benchmark_ground_truth_used": False,
            "target_or_evaluation_rgb_used": False,
        }
    ):
        raise ValueError("source-only RGB information policy differs")
    source_by_id = {
        str(record["image_id"]): dict(record)
        for record in authority_payload.get("images", [])
        if isinstance(record, Mapping)
    }
    selected_ids = [str(value) for value in manifest.get("selected_image_ids", [])]
    raw_records = manifest.get("images")
    if not selected_ids or set(selected_ids) != set(source_by_id) or not isinstance(raw_records, list):
        raise ValueError("multiscale/source authority image axes differ")
    records = {
        str(record.get("image_id", "")): dict(record)
        for record in raw_records
        if isinstance(record, Mapping)
    }
    if list(records) != selected_ids:
        raise ValueError("multiscale manifest image order differs")

    checkpoint = Path(args.radio_checkpoint).expanduser().resolve()
    if sha256_file(checkpoint) != str(args.radio_checkpoint_sha256):
        raise ValueError("official C-RADIO checkpoint SHA-256 differs")
    device = torch.device(args.device)
    runtime = OfficialCropSummaryRuntime.load(
        checkpoint_path=str(checkpoint),
        radio_repo=str(Path(args.radio_repo).expanduser().resolve()),
        version=str(args.radio_version),
        device=device,
    )
    if runtime.radio_checkpoint_sha256 != str(args.radio_checkpoint_sha256):
        raise ValueError("loaded official C-RADIO checkpoint identity differs")

    masked_chunks: list[torch.Tensor] = []
    context_chunks: list[torch.Tensor] = []
    proposal_views: list[int] = []
    proposal_frames: list[int] = []
    output_records: list[dict[str, object]] = []
    for view_index, image_id in enumerate(selected_ids):
        source = source_by_id[image_id]
        image_path = Path(str(source.get("path", ""))).expanduser().resolve()
        if sha256_file(image_path) != str(source.get("sha256", "")):
            raise ValueError("source RGB bytes differ")
        record = records[image_id]
        mask_path = Path(str(record.get("output", ""))).expanduser().resolve()
        if mask_path.parent != mask_root or sha256_file(mask_path) != str(record.get("output_sha256", "")):
            raise ValueError("multiscale mask cache bytes differ")
        payload = torch.load(mask_path, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping) or dict(metadata.get("source_image", {})).get("image_id") != image_id:
            raise ValueError("multiscale source image metadata differs")
        height, width = (int(value) for value in payload["mask_shape"])
        masks = unpack_masks(torch.as_tensor(payload["packed_masks"]), width=width)
        image_bytes = image_path.read_bytes()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if image.size != (width, height):
            raise ValueError("source RGB and multiscale mask raster differ")
        image_tensor = pil_to_tensor(image).float().div_(255.0)
        masked_crops, context_crops = build_crop_pairs(
            image_tensor,
            masks,
            torch.as_tensor(payload["boxes_xyxy"]),
            context_expansion=float(args.context_expansion),
            crop_resolution=int(args.crop_resolution),
            masked_background_rgb=(0.5, 0.5, 0.5),
        )
        masked = encode_in_batches(
            runtime,
            masked_crops,
            batch_size=int(args.batch_size),
            device=device,
        )
        context = encode_in_batches(
            runtime,
            context_crops,
            batch_size=int(args.batch_size),
            device=device,
        )
        count = int(masked.shape[0])
        masked_chunks.append(masked)
        context_chunks.append(context)
        proposal_views.extend([view_index] * count)
        frame_id = int(image_id[6:])
        proposal_frames.extend([frame_id] * count)
        output_records.append(
            {
                "image_id": image_id,
                "frame_id": frame_id,
                "source_view_index": view_index,
                "source_image": str(image_path),
                "source_image_sha256": str(source["sha256"]),
                "mask_cache": str(mask_path),
                "mask_cache_sha256": str(record["output_sha256"]),
                "proposal_count": count,
            }
        )
    descriptors = torch.cat(masked_chunks)
    contexts = torch.cat(context_chunks)
    paired_cosine = (descriptors.float() * contexts.float()).sum(dim=-1)
    payload_out = {
        "schema": SCHEMA,
        "schema_version": 1,
        "scene": str(args.scene),
        "descriptors": descriptors,
        "context_descriptors": contexts,
        "foreground_context_cosine": paired_cosine,
        "proposal_view_indices": torch.tensor(proposal_views, dtype=torch.long),
        "proposal_frame_indices": torch.tensor(proposal_frames, dtype=torch.long),
        "metadata": {
            "teacher_space": "official_siglip2_crop_summary",
            "text_compatibility": "official_siglip2_g_text_space",
            "descriptor_formula": "official_c_radio_v4_h_siglip2_summary(masked_tight_crop)",
            "context_formula": "official_c_radio_v4_h_siglip2_summary(1.5x_context_crop)",
            "masked_background_rgb": [0.5, 0.5, 0.5],
            "context_expansion": float(args.context_expansion),
            "crop_resolution": int(args.crop_resolution),
            "query_independent": True,
            "source_only": True,
            "official_sam3_decoder": True,
            "benchmark_labels_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_vocabulary_opened": False,
            "evaluation_rgb_opened": False,
            "text_queries_opened": False,
            "radio_checkpoint": str(checkpoint),
            "radio_checkpoint_sha256": runtime.radio_checkpoint_sha256,
            "multiscale_manifest": str(manifest_path),
            "multiscale_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "source_records": output_records,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    torch.save(payload_out, temporary)
    os.replace(temporary, output)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "scene": str(args.scene),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "source_view_count": len(selected_ids),
        "proposal_count": int(descriptors.shape[0]),
        "descriptor_dim": int(descriptors.shape[1]),
        "mean_masked_context_cosine": float(paired_cosine.mean()),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--mask-root", required=True)
    parser.add_argument("--manifest-name", default="manifest.json")
    parser.add_argument("--radio-checkpoint", required=True)
    parser.add_argument("--radio-checkpoint-sha256", required=True)
    parser.add_argument("--radio-repo", default="/root/RADIO")
    parser.add_argument("--radio-version", default="c-radio_v4-h")
    parser.add_argument("--context-expansion", type=float, default=1.5)
    parser.add_argument("--crop-resolution", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    print(json.dumps(build(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
