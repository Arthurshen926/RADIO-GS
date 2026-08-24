#!/usr/bin/env python3
"""Pool independent native DINOv2 tokens inside source SAM3 proposals."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F

from radio_gs.models.sam3_multiscale_hierarchy import unpack_masks
from radio_gs.utils.immutable_artifacts import sha256_file


SCHEMA = "radio_gs.multiscale_sam_mask_aligned_native_dinov2_teacher.v1"
DINO_FRAME_SCHEMA = "radio_gs.native_dinov2_exact_mpr_teacher.v1"


def build(args: argparse.Namespace) -> dict[str, object]:
    output = Path(args.output).expanduser().resolve()
    report = output.with_suffix(output.suffix + ".json")
    if output.exists() or report.exists():
        raise FileExistsError(f"native DINO proposal teacher exists: {output}")
    mask_root = Path(args.mask_root).expanduser().resolve(strict=True)
    manifest_path = mask_root / str(args.manifest_name)
    manifest = json.loads(manifest_path.read_bytes())
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("contract")
        != "official-sam3-query-free-multiscale-hierarchy-manifest-v1"
        or manifest.get("generation_contract", {}).get("query_free") is not True
    ):
        raise ValueError("official source SAM3 manifest contract differs")
    selected = [str(value) for value in manifest.get("selected_image_ids", [])]
    raw_records = manifest.get("images")
    if not selected or not isinstance(raw_records, list):
        raise ValueError("source SAM3 manifest has no proposal cohort")
    records = {
        str(record.get("image_id", "")): dict(record)
        for record in raw_records
        if isinstance(record, Mapping)
    }
    if list(records) != selected:
        raise ValueError("source SAM3 manifest image order differs")
    frame_root = Path(args.native_dino_frame_root).expanduser().resolve(strict=True)
    device = torch.device(args.device)
    descriptors: list[torch.Tensor] = []
    proposal_views: list[int] = []
    proposal_frames: list[int] = []
    output_records: list[dict[str, object]] = []
    checkpoint_sha256 = ""
    for view_index, image_id in enumerate(selected):
        if not image_id.startswith("frame_"):
            raise ValueError("source image id has no canonical frame index")
        frame = int(image_id[6:])
        mask_record = records[image_id]
        mask_path = Path(str(mask_record.get("output", ""))).expanduser().resolve()
        if mask_path.parent != mask_root or sha256_file(mask_path) != str(
            mask_record.get("output_sha256", "")
        ):
            raise ValueError("source SAM3 proposal cache binding differs")
        mask_payload = torch.load(mask_path, map_location="cpu", weights_only=False)
        _height, width = (int(value) for value in mask_payload["mask_shape"])
        masks = torch.from_numpy(
            unpack_masks(torch.as_tensor(mask_payload["packed_masks"]), width=width)
        ).float()
        frame_path = frame_root / f"frame_{frame:05d}.pt"
        frame_payload = torch.load(frame_path, map_location="cpu", weights_only=False)
        feature = torch.as_tensor(frame_payload.get("feature")).float()
        source = mask_payload.get("metadata", {}).get("source_image", {})
        if (
            frame_payload.get("schema") != DINO_FRAME_SCHEMA
            or int(frame_payload.get("frame_index", -1)) != frame
            or str(frame_payload.get("source_sha256", ""))
            != str(source.get("sha256", ""))
            or feature.ndim != 3
            or not bool(torch.isfinite(feature).all())
        ):
            raise ValueError("native DINO frame/source lineage differs")
        current_checkpoint = str(frame_payload.get("checkpoint_sha256", ""))
        if not checkpoint_sha256:
            checkpoint_sha256 = current_checkpoint
        if len(current_checkpoint) != 64 or current_checkpoint != checkpoint_sha256:
            raise ValueError("native DINO checkpoint changed inside proposal cohort")
        grid_height, grid_width = map(int, feature.shape[1:])
        aligned = F.interpolate(
            masks[:, None], size=(grid_height, grid_width), mode="nearest"
        )[:, 0]
        flat_masks = aligned.flatten(1).to(device)
        flat_feature = feature.flatten(1).T.to(device)
        pooled = flat_masks @ flat_feature
        pooled /= flat_masks.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = F.normalize(pooled.float(), dim=-1, eps=1e-8).cpu()
        if not bool(torch.isfinite(pooled).all()):
            raise ValueError("native DINO proposal descriptor is non-finite")
        count = int(pooled.shape[0])
        descriptors.append(pooled.half())
        proposal_views.extend([view_index] * count)
        proposal_frames.extend([frame] * count)
        output_records.append(
            {
                "image_id": image_id,
                "frame_id": frame,
                "source_view_index": view_index,
                "mask_cache": str(mask_path),
                "mask_cache_sha256": sha256_file(mask_path),
                "native_dino_frame": str(frame_path),
                "native_dino_frame_sha256": sha256_file(frame_path),
                "proposal_count": count,
                "native_grid": [grid_height, grid_width],
            }
        )
    descriptor = torch.cat(descriptors)
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "scene": str(args.scene),
        "descriptors": descriptor,
        "proposal_view_indices": torch.tensor(proposal_views, dtype=torch.long),
        "proposal_frame_indices": torch.tensor(proposal_frames, dtype=torch.long),
        "metadata": {
            "construction": "native_dinov2_dense_tokens_mask_mean_pool",
            "teacher_role": "query_free_cross_view_instance_appearance",
            "feature_dim": int(descriptor.shape[1]),
            "proposal_count": int(descriptor.shape[0]),
            "native_dino_checkpoint_sha256": checkpoint_sha256,
            "native_dino_frame_root": str(frame_root),
            "multiscale_manifest": str(manifest_path),
            "multiscale_manifest_sha256": sha256_file(manifest_path),
            "source_records": output_records,
            "source_only": True,
            "query_independent": True,
            "c_radio_or_radio_adaptor_used": False,
            "benchmark_masks_opened": False,
            "benchmark_queries_opened": False,
            "evaluation_rgb_opened": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "output": str(output),
        "output_sha256": sha256_file(output),
        "proposals": int(descriptor.shape[0]),
        "feature_dim": int(descriptor.shape[1]),
    }
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--mask-root", required=True)
    parser.add_argument("--manifest-name", default="manifest_grid8_crop2.json")
    parser.add_argument("--native-dino-frame-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    print(json.dumps(build(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
