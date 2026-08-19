#!/usr/bin/env python3
"""Lift SHA-bound ScanNet source-view official SAM3 masks through exact MPR."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image
import torch

from radio_gs.scripts.build_lerf_query_free_sam3_exact_mpr_memberships import (
    EXPECTED_P0_GENERATION,
    lift_binary_masks_with_exact_mpr,
    validate_automatic_mask_payload,
    validate_responsibility_authority,
)
from radio_gs.scripts.build_lerf_sam3_exact_mpr_memberships import (
    _float32_rows_sha256,
)
from radio_gs.scripts.materialize_official_multiview_siglip2_teacher_authority import (
    validate_responsibility_view,
)
from radio_gs.utils.immutable_artifacts import sha256_file


SCHEMA = "radio_gs.scannet_official_sam3_exact_mpr_memberships.v1"


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    report_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or report_path.exists():
        raise FileExistsError(f"official-SAM membership output exists: {output}")

    primitive_path = Path(args.primitive_cache).expanduser().resolve()
    primitive = torch.load(primitive_path, map_location="cpu", weights_only=False)
    xyz = torch.as_tensor(primitive.get("xyz")).float().cpu()
    num_gaussians = int(xyz.shape[0])
    xyz_sha256 = _float32_rows_sha256(xyz)

    authority_path = Path(args.responsibility_authority).expanduser().resolve()
    authority_bytes = authority_path.read_bytes()
    authority = validate_responsibility_authority(
        json.loads(authority_bytes),
        authority_path=authority_path,
        num_gaussians=num_gaussians,
        xyz_sha256=xyz_sha256,
    )
    feature_height = int(authority["metadata"]["feature_height"])
    feature_width = int(authority["metadata"]["feature_width"])
    views = {int(row["frame_index"]): row for row in authority["views"]}

    mask_root = Path(args.mask_root).expanduser().resolve()
    manifest_path = mask_root / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    expected_generation = {
        **EXPECTED_P0_GENERATION,
        "checkpoint_sha256": str(args.expected_checkpoint_sha256),
        "grid_size": int(args.expected_grid_size),
    }
    if manifest.get("generation_contract") != expected_generation:
        raise ValueError("official-SAM generation contract differs")
    manifest_rows: dict[int, dict[str, object]] = {}
    for raw in manifest.get("images", []):
        record = dict(raw)
        image = Path(str(record.get("image", ""))).expanduser().resolve()
        frame = int(image.stem)
        if frame in manifest_rows:
            raise ValueError("official-SAM manifest repeats a frame")
        manifest_rows[frame] = record
    if set(manifest_rows) != set(views):
        raise ValueError("official-SAM and exact-MPR frame axes differ")

    device = torch.device(args.device)
    row_chunks: list[torch.Tensor] = []
    proposal_chunks: list[torch.Tensor] = []
    weight_chunks: list[torch.Tensor] = []
    proposal_views: list[int] = []
    proposal_frames: list[int] = []
    proposal_scores: list[torch.Tensor] = []
    proposal_stability: list[torch.Tensor] = []
    proposal_area: list[torch.Tensor] = []
    proposal_boxes: list[torch.Tensor] = []
    records: list[dict[str, object]] = []
    proposal_offset = 0

    for source_view_index, frame in enumerate(authority["frame_indices"]):
        frame = int(frame)
        record = manifest_rows[frame]
        image_path = Path(str(record["image"])).expanduser().resolve(strict=True)
        mask_path = Path(str(record["output"])).expanduser().resolve(strict=True)
        if mask_path.parent != mask_root:
            raise ValueError("official-SAM mask escaped its sealed root")
        if sha256_file(image_path) != str(record["source_image_sha256"]):
            raise ValueError("source image SHA-256 differs")
        mask_bytes = mask_path.read_bytes()
        if hashlib.sha256(mask_bytes).hexdigest() != str(record["output_sha256"]):
            raise ValueError("official-SAM mask SHA-256 differs")
        payload = torch.load(io.BytesIO(mask_bytes), map_location="cpu", weights_only=False)
        metadata = dict(payload.get("metadata", {}))
        if (
            any(metadata.get(key) != value for key, value in expected_generation.items())
            or Path(str(metadata.get("image", ""))).expanduser().resolve() != image_path
            or str(metadata.get("source_image_sha256", ""))
            != str(record["source_image_sha256"])
        ):
            raise ValueError("official-SAM payload provenance differs")
        with Image.open(image_path) as image:
            width, height = image.size
        attributes = validate_automatic_mask_payload(
            payload,
            image_height=height,
            image_width=width,
            expected_grid_size=int(args.expected_grid_size),
        )
        masks = attributes["masks"].to(device)
        view_record = views[frame]
        view_path = Path(str(view_record["resolved_path"]))
        view = validate_responsibility_view(
            torch.load(view_path, map_location="cpu", weights_only=False),
            record=view_record,
            formula_sha256=str(authority["formula_sha256"]),
            num_gaussians=num_gaussians,
        )
        rows, local_proposals, memberships, denominator = (
            lift_binary_masks_with_exact_mpr(
                masks,
                torch.as_tensor(view["gaussian_ids"]).to(device),
                torch.as_tensor(view["pixel_ids"]).to(device),
                torch.as_tensor(view["base_weights"]).to(device),
                num_gaussians=num_gaussians,
                feature_height=feature_height,
                feature_width=feature_width,
                min_membership=float(args.min_membership),
            )
        )
        if denominator.shape != (num_gaussians,) or not bool((denominator > 0).any()):
            raise ValueError("exact-MPR source-view support differs")
        if rows.numel():
            row_chunks.append(rows.cpu())
            proposal_chunks.append((local_proposals + proposal_offset).cpu())
            weight_chunks.append(memberships.float().cpu())
        count = int(masks.shape[0])
        proposal_views.extend([source_view_index] * count)
        proposal_frames.extend([frame] * count)
        proposal_scores.append(attributes["scores"].cpu())
        proposal_stability.append(attributes["stability"].cpu())
        proposal_area.append(attributes["area"].cpu())
        proposal_boxes.append(attributes["boxes_xyxy"].cpu())
        records.append(
            {
                "frame_id": frame,
                "source_view_index": source_view_index,
                "source_image": str(image_path),
                "source_image_sha256": str(record["source_image_sha256"]),
                "mask_cache": str(mask_path),
                "mask_cache_sha256": str(record["output_sha256"]),
                "responsibility_view": str(view_path),
                "responsibility_view_sha256": sha256_file(view_path),
                "num_proposals": count,
                "num_memberships": int(rows.numel()),
            }
        )
        proposal_offset += count

    sparse_rows = torch.cat(row_chunks) if row_chunks else torch.empty(0, dtype=torch.long)
    sparse_proposals = (
        torch.cat(proposal_chunks) if proposal_chunks else torch.empty(0, dtype=torch.long)
    )
    sparse_weights = torch.cat(weight_chunks) if weight_chunks else torch.empty(0)
    payload_out = {
        "schema": SCHEMA,
        "schema_version": 1,
        "scene": str(args.scene),
        "num_rows": num_gaussians,
        "num_proposals": proposal_offset,
        "row_indices": sparse_rows,
        "proposal_indices": sparse_proposals,
        "weights": sparse_weights,
        "proposal_view_indices": torch.tensor(proposal_views, dtype=torch.long),
        "proposal_frame_indices": torch.tensor(proposal_frames, dtype=torch.long),
        "proposal_scores": torch.cat(proposal_scores),
        "proposal_stability": torch.cat(proposal_stability),
        "proposal_area_fraction": torch.cat(proposal_area),
        "proposal_boxes_xyxy": torch.cat(proposal_boxes),
        "metadata": {
            "query_independent_proposal_set": True,
            "official_sam3_decoder": True,
            "proposal_set_type": "single_scale_point_grid_multimask",
            "membership_lifting": "exact_front_to_back_marginal_target_weight",
            "min_membership": float(args.min_membership),
            "xyz_sha256": xyz_sha256,
            "source_view_count": len(records),
            "source_records": records,
            "primitive_cache": str(primitive_path),
            "primitive_cache_sha256": sha256_file(primitive_path),
            "responsibility_authority": str(authority_path),
            "responsibility_authority_sha256": hashlib.sha256(authority_bytes).hexdigest(),
            "automatic_mask_manifest": str(manifest_path),
            "automatic_mask_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "official_sam3_checkpoint_sha256": str(args.expected_checkpoint_sha256),
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "evaluation_rgb_opened": False,
            "text_queries_opened": False,
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
        "num_rows": num_gaussians,
        "num_proposals": proposal_offset,
        "num_memberships": int(sparse_rows.numel()),
        "source_view_count": len(records),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--responsibility-authority", required=True)
    parser.add_argument("--primitive-cache", required=True)
    parser.add_argument("--mask-root", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-grid-size", type=int, default=12)
    parser.add_argument("--min-membership", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    print(json.dumps(build(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

