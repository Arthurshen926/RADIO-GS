#!/usr/bin/env python3
"""Encode multiscale source-SAM3 masks with RADIO or native SigLIP2 crops.

This is the direct SAM+CLIP-style identity path: official SAM3 fixes the mask,
the masked RGB crop is encoded either by the frozen official C-RADIOv4
SigLIP2-G summary adaptor (control) or the independent native SigLIP2 image
tower (candidate), and an expanded unmasked crop supplies contextual evidence.
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
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor

from radio_gs.interfaces.frozen_radio_views import OfficialCropSummaryRuntime
from radio_gs.models.sam3_multiscale_hierarchy import unpack_masks
from radio_gs.scripts.build_sam_mask_aligned_language_teacher import (
    build_crop_pairs,
    encode_in_batches,
)
from radio_gs.utils.immutable_artifacts import sha256_file


RADIO_SCHEMA = "radio_gs.multiscale_sam_mask_aligned_crop_summary_teacher.v1"
NATIVE_SCHEMA = "radio_gs.multiscale_sam_mask_aligned_crop_summary_teacher.v2"


class NativeSiglip2Runtime:
    """Frozen native SigLIP2 vision tower in its paired text embedding space."""

    def __init__(self, model: torch.nn.Module, *, device: torch.device, bundle: dict[str, Any]):
        self.model = model
        self.device = device
        self.bundle = bundle

    @classmethod
    def load(cls, path: Path, *, device: torch.device) -> "NativeSiglip2Runtime":
        from transformers import SiglipVisionModel

        root = path.expanduser().resolve(strict=True)
        required = (
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "preprocessor_config.json",
        )
        records = []
        for name in required:
            source = root / name
            resolved = source.resolve(strict=True)
            if not resolved.is_file():
                raise ValueError(f"native SigLIP2 bundle lacks {name}")
            records.append(
                {
                    "name": name,
                    "resolved_path": str(resolved),
                    "bytes": resolved.stat().st_size,
                    "sha256": sha256_file(resolved),
                }
            )
        digest = hashlib.sha256(
            json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        model = SiglipVisionModel.from_pretrained(
            str(root),
            local_files_only=True,
            torch_dtype=dtype,
        ).to(device).eval()
        model.requires_grad_(False)
        return cls(
            model,
            device=device,
            bundle={"path": str(root), "sha256": digest, "files": records},
        )

    @torch.inference_mode()
    def encode(self, crops: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(crops, device=self.device).float()
        if values.ndim != 4 or values.shape[1:] != (3, 384, 384):
            raise ValueError("native SigLIP2 crops must be [B,3,384,384]")
        # Frozen processor contract: input RGB is already in [0,1] and resized;
        # only SigLIP's 0.5/0.5 normalization remains.
        pixel_values = ((values - 0.5) / 0.5).to(
            dtype=next(self.model.parameters()).dtype
        )
        output = self.model(pixel_values=pixel_values, return_dict=True)
        descriptor = F.normalize(output.pooler_output.float(), dim=-1, eps=1e-8)
        return descriptor


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

    device = torch.device(args.device)
    if str(args.encoder_backend) == "radio_siglip2_summary":
        checkpoint = Path(args.radio_checkpoint).expanduser().resolve()
        if not args.radio_checkpoint or sha256_file(checkpoint) != str(args.radio_checkpoint_sha256):
            raise ValueError("official C-RADIO checkpoint SHA-256 differs")
        runtime = OfficialCropSummaryRuntime.load(
            checkpoint_path=str(checkpoint),
            radio_repo=str(Path(args.radio_repo).expanduser().resolve()),
            version=str(args.radio_version),
            device=device,
        )
        if runtime.radio_checkpoint_sha256 != str(args.radio_checkpoint_sha256):
            raise ValueError("loaded official C-RADIO checkpoint identity differs")
        teacher_space = "official_c_radio_siglip2_crop_summary"
        text_compatibility = "official_siglip2_g_text_space"
        encoder_binding: dict[str, Any] = {
            "backend": str(args.encoder_backend),
            "radio_checkpoint": str(checkpoint),
            "radio_checkpoint_sha256": runtime.radio_checkpoint_sha256,
        }
    elif str(args.encoder_backend) == "native_siglip2_vision":
        if int(args.crop_resolution) != 384:
            raise ValueError("native SigLIP2 candidate is frozen to 384x384 crops")
        runtime = NativeSiglip2Runtime.load(
            Path(args.native_siglip2_model), device=device
        )
        teacher_space = "independent_native_siglip2_vision_pooler"
        text_compatibility = "native_siglip2_paired_text_tower_space"
        encoder_binding = {
            "backend": str(args.encoder_backend),
            "native_siglip2_bundle": runtime.bundle,
        }
    else:
        raise ValueError("unknown crop encoder backend")

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
        frame_id = int(image_id[6:])
        if int(masks.shape[0]) == 0:
            # Query-free SAM is allowed to reject every proposal in one
            # source frame.  That frame still belongs to the immutable source
            # authority, but it contributes no row to the proposal axis.
            output_records.append(
                {
                    "image_id": image_id,
                    "frame_id": frame_id,
                    "source_view_index": view_index,
                    "source_image": str(image_path),
                    "source_image_sha256": str(source["sha256"]),
                    "mask_cache": str(mask_path),
                    "mask_cache_sha256": str(record["output_sha256"]),
                    "proposal_count": 0,
                }
            )
            continue
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
    output_schema = (
        RADIO_SCHEMA
        if str(args.encoder_backend) == "radio_siglip2_summary"
        else NATIVE_SCHEMA
    )
    payload_out = {
        "schema": output_schema,
        "schema_version": 1,
        "scene": str(args.scene),
        "descriptors": descriptors,
        "context_descriptors": contexts,
        "foreground_context_cosine": paired_cosine,
        "proposal_view_indices": torch.tensor(proposal_views, dtype=torch.long),
        "proposal_frame_indices": torch.tensor(proposal_frames, dtype=torch.long),
        "metadata": {
            "teacher_space": teacher_space,
            "text_compatibility": text_compatibility,
            "descriptor_formula": f"{args.encoder_backend}(masked_tight_crop)",
            "context_formula": f"{args.encoder_backend}(1.5x_context_crop)",
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
            "encoder_binding": encoder_binding,
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
        "schema": output_schema,
        "status": "complete",
        "scene": str(args.scene),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "source_view_count": len(selected_ids),
        "proposal_count": int(descriptors.shape[0]),
        "descriptor_dim": int(descriptors.shape[1]),
        "encoder_backend": str(args.encoder_backend),
        "mean_masked_context_cosine": float(paired_cosine.mean()),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--mask-root", required=True)
    parser.add_argument("--manifest-name", default="manifest.json")
    parser.add_argument(
        "--encoder-backend",
        choices=("radio_siglip2_summary", "native_siglip2_vision"),
        default="radio_siglip2_summary",
    )
    parser.add_argument("--radio-checkpoint", default="")
    parser.add_argument("--radio-checkpoint-sha256", default="")
    parser.add_argument("--radio-repo", default="/root/RADIO")
    parser.add_argument("--radio-version", default="c-radio_v4-h")
    parser.add_argument("--native-siglip2-model", default="")
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
