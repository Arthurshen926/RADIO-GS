#!/usr/bin/env python3
"""Lift query-free multiscale source-SAM3 hierarchies through exact MPR.

The output is a query-independent Gaussian-domain proposal forest.  Every 2D
proposal keeps its scale/crop/containment identity while its object membership
is reconstructed from exact front-to-back marginal responsibility.  No text,
benchmark mask, or target/evaluation RGB interface exists in this builder.
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

from radio_gs.models.sam3_multiscale_hierarchy import (
    unpack_masks,
    validate_multiscale_cache_payload,
    validate_source_authority_payload,
)
from radio_gs.scripts.build_lerf_query_free_sam3_exact_mpr_memberships import (
    lift_binary_masks_with_exact_mpr,
    validate_responsibility_authority,
)
from radio_gs.scripts.build_lerf_sam3_exact_mpr_memberships import (
    _float32_rows_sha256,
)
from radio_gs.scripts.materialize_official_multiview_siglip2_teacher_authority import (
    validate_responsibility_view,
)
from radio_gs.utils.immutable_artifacts import sha256_file


SCHEMA = "radio_gs.lerf_multiscale_sam3_exact_mpr_memberships.v1"
MANIFEST_CONTRACT = "official-sam3-query-free-multiscale-hierarchy-manifest-v1"


def _load_json(path: Path, *, label: str) -> tuple[dict, bytes]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is not one regular JSON file: {path}")
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain one JSON object")
    return dict(payload), raw


def _frame_id(image_id: object) -> int:
    value = str(image_id)
    if not value.startswith("frame_") or len(value) != len("frame_00000"):
        raise ValueError(f"multiscale image id has no canonical frame id: {value}")
    try:
        return int(value[6:])
    except ValueError as error:
        raise ValueError(f"multiscale image id has no numeric frame id: {value}") from error


def _validate_information_contract(contract: object, checkpoint_sha256: str) -> dict:
    if not isinstance(contract, Mapping):
        raise ValueError("multiscale generation contract is absent")
    value = dict(contract)
    if (
        value.get("schema_version") != 1
        or value.get("source")
        != "official_sam3_interactive_multiscale_crop_pyramid_v1"
        or value.get("official_decoder") is not True
        or value.get("query_free") is not True
        or value.get("information_inputs") != ["registered_source_or_mapping_rgb"]
        or value.get("forbidden_inputs")
        != ["query_text", "benchmark_ground_truth", "target_or_evaluation_rgb"]
        or str(value.get("checkpoint_sha256", "")) != str(checkpoint_sha256)
        or int(value.get("resolution", -1)) != 1008
        or value.get("mask_tensor_semantics")
        != "binary_probability_thresholded_by_official_decoder"
        or value.get("full_image_remapping")
        != "exact_integer_crop_embedding_without_resize"
    ):
        raise ValueError("multiscale generation information contract differs")
    pyramid = value.get("crop_pyramid")
    hierarchy = value.get("hierarchy")
    if (
        not isinstance(pyramid, Mapping)
        or int(pyramid.get("layers_after_full_image", -1)) < 1
        or not isinstance(hierarchy, Mapping)
        or hierarchy.get("type")
        != "direct_smallest-containing-parent_forest"
    ):
        raise ValueError("multiscale crop hierarchy contract differs")
    return value


def remap_parent_forest(
    parent_index: torch.Tensor,
    parent_edges: torch.Tensor,
    *,
    offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remap one view-local containment forest into global proposal ids."""

    parents = torch.as_tensor(parent_index).long().clone()
    edges = torch.as_tensor(parent_edges).long().clone()
    count = int(parents.numel())
    if parents.ndim != 1 or edges.ndim != 2 or tuple(edges.shape[1:]) != (2,):
        raise ValueError("local parent forest shape differs")
    if bool(((parents < -1) | (parents >= count)).any()):
        raise ValueError("local parent index is outside proposal domain")
    if edges.numel() and bool(((edges < 0) | (edges >= count)).any()):
        raise ValueError("local parent edge is outside proposal domain")
    child = torch.where(parents >= 0)[0]
    expected = (
        torch.stack((parents[child], child), dim=1)
        if child.numel()
        else torch.empty((0, 2), dtype=torch.long)
    )
    if not torch.equal(edges, expected):
        raise ValueError("parent edge list differs from parent index")
    parents[parents >= 0] += int(offset)
    if edges.numel():
        edges += int(offset)
    return parents, edges


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    report_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or report_path.exists():
        raise FileExistsError(f"output already exists: {output}")

    primitive_path = Path(args.primitive_cache).expanduser().resolve()
    primitive = torch.load(primitive_path, map_location="cpu", weights_only=False)
    xyz = torch.as_tensor(primitive.get("xyz")).float().cpu()
    num_gaussians = int(xyz.shape[0])
    xyz_sha256 = _float32_rows_sha256(xyz)

    authority_path = Path(args.responsibility_authority).expanduser().resolve()
    raw_authority, authority_bytes = _load_json(
        authority_path, label="exact-MPR authority"
    )
    authority = validate_responsibility_authority(
        raw_authority,
        authority_path=authority_path,
        num_gaussians=num_gaussians,
        xyz_sha256=xyz_sha256,
    )
    responsibility_metadata = dict(authority["metadata"])
    feature_height = int(responsibility_metadata.get("feature_height", 0))
    feature_width = int(responsibility_metadata.get("feature_width", 0))
    frame_to_view = {
        int(record["frame_index"]): record for record in authority["views"]
    }

    source_authority_path = Path(args.source_authority).expanduser().resolve()
    source_authority, source_authority_bytes = _load_json(
        source_authority_path, label="source RGB authority"
    )
    source_records = validate_source_authority_payload(source_authority)
    source_by_id = {str(record["image_id"]): record for record in source_records}
    source_sha = hashlib.sha256(source_authority_bytes).hexdigest()

    mask_root = Path(args.mask_root).expanduser().resolve()
    manifest_path = mask_root / str(args.manifest_name)
    manifest, manifest_bytes = _load_json(
        manifest_path, label="multiscale hierarchy manifest"
    )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("contract") != MANIFEST_CONTRACT
        or manifest.get("source_authority")
        != {"path": str(source_authority_path), "sha256": source_sha}
    ):
        raise ValueError("multiscale hierarchy manifest identity differs")
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("multiscale checkpoint binding is absent")
    checkpoint_sha = str(checkpoint.get("sha256", ""))
    if len(checkpoint_sha) != 64:
        raise ValueError("multiscale checkpoint SHA-256 differs")
    generation = _validate_information_contract(
        manifest.get("generation_contract"), checkpoint_sha
    )
    selected_ids = [str(value) for value in manifest.get("selected_image_ids", [])]
    manifest_images = manifest.get("images")
    if (
        not selected_ids
        or len(set(selected_ids)) != len(selected_ids)
        or set(selected_ids) != set(source_by_id)
        or not isinstance(manifest_images, list)
        or len(manifest_images) != len(selected_ids)
    ):
        raise ValueError("multiscale selected/source image axes differ")
    manifest_by_id: dict[str, dict] = {}
    for raw in manifest_images:
        if not isinstance(raw, Mapping):
            raise ValueError("multiscale manifest image record differs")
        record = dict(raw)
        image_id = str(record.get("image_id", ""))
        if image_id in manifest_by_id or image_id not in source_by_id:
            raise ValueError("multiscale manifest repeats or adds an image")
        manifest_by_id[image_id] = record
    if list(manifest_by_id) != selected_ids:
        raise ValueError("multiscale manifest image order differs")

    device = torch.device(args.device)
    row_chunks: list[torch.Tensor] = []
    proposal_chunks: list[torch.Tensor] = []
    weight_chunks: list[torch.Tensor] = []
    view_denominators: list[torch.Tensor] = []
    proposal_views: list[int] = []
    proposal_frames: list[int] = []
    aligned_chunks: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "quality", "stability", "proposal_area_fraction", "boxes_xyxy",
            "seed_xy_full", "prompt_index", "candidate_index", "crop_index",
            "crop_layer", "crop_grid_side", "crop_boxes_xyxy", "crop_scale_xy",
            "crop_window_area_fraction", "crop_area_fraction",
            "touches_crop_edge", "parent_containment", "parent_area_ratio",
        )
    }
    parent_chunks: list[torch.Tensor] = []
    edge_chunks: list[torch.Tensor] = []
    records: list[dict[str, object]] = []
    proposal_offset = 0

    for source_view_index, image_id in enumerate(selected_ids):
        frame_id = _frame_id(image_id)
        if frame_id not in frame_to_view:
            raise ValueError(f"multiscale source frame lacks exact MPR: {frame_id}")
        source = source_by_id[image_id]
        source_path = Path(str(source["path"])).expanduser().resolve()
        if not source_path.is_file() or sha256_file(source_path) != source["sha256"]:
            raise ValueError(f"source RGB byte binding differs: {image_id}")
        record = manifest_by_id[image_id]
        mask_path = Path(str(record.get("output", ""))).expanduser().resolve()
        if mask_path.parent != mask_root or not mask_path.is_file():
            raise ValueError("multiscale cache escaped mask root or is absent")
        mask_bytes = mask_path.read_bytes()
        mask_sha = hashlib.sha256(mask_bytes).hexdigest()
        if mask_sha != str(record.get("output_sha256", "")):
            raise ValueError("multiscale cache SHA-256 differs")
        payload = torch.load(io.BytesIO(mask_bytes), map_location="cpu", weights_only=False)
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("multiscale cache metadata is absent")
        expected_metadata = {
            "generation_contract": generation,
            "source_authority": manifest["source_authority"],
            "source_image": {
                "image_id": image_id,
                "path": str(source_path),
                "sha256": str(source["sha256"]),
                "rgb_role": "registered_source_or_mapping_view",
            },
            "checkpoint": dict(checkpoint),
            "runtime_binding": manifest.get("runtime_binding"),
            "producer_binding": manifest.get("producer_binding"),
        }
        count = validate_multiscale_cache_payload(
            payload, expected_metadata=expected_metadata
        )
        if int(record.get("proposal_count", -1)) != count:
            raise ValueError("multiscale manifest proposal count differs")
        height, width = (int(value) for value in payload["mask_shape"])
        masks = torch.from_numpy(
            unpack_masks(torch.as_tensor(payload["packed_masks"]), width=width)
        ).to(device)

        authority_record = frame_to_view[frame_id]
        view_path = Path(str(authority_record["resolved_path"]))
        view = validate_responsibility_view(
            torch.load(view_path, map_location="cpu", weights_only=False),
            record=authority_record,
            formula_sha256=str(authority["formula_sha256"]),
            num_gaussians=num_gaussians,
        )
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
        if denominator.shape != (num_gaussians,) or not bool((denominator > 0).any()):
            raise ValueError("multiscale per-view exact-MPR support differs")
        view_denominators.append(denominator.float().cpu())
        if rows.numel():
            row_chunks.append(rows.cpu())
            proposal_chunks.append((local_proposals + proposal_offset).cpu())
            weight_chunks.append(membership.float().cpu())
        proposal_views.extend([source_view_index] * count)
        proposal_frames.extend([frame_id] * count)
        for key in aligned_chunks:
            aligned_chunks[key].append(torch.as_tensor(payload[key]).cpu())
        parents, edges = remap_parent_forest(
            torch.as_tensor(payload["parent_index"]),
            torch.as_tensor(payload["parent_edges"]),
            offset=proposal_offset,
        )
        parent_chunks.append(parents)
        edge_chunks.append(edges)
        records.append(
            {
                "image_id": image_id,
                "frame_id": frame_id,
                "source_view_index": source_view_index,
                "source_image": str(source_path),
                "source_image_sha256": str(source["sha256"]),
                "mask_cache": str(mask_path),
                "mask_cache_sha256": mask_sha,
                "responsibility_view": str(view_path),
                "responsibility_view_sha256": sha256_file(view_path),
                "num_proposals": count,
                "num_memberships": int(rows.numel()),
                "mask_shape": [height, width],
            }
        )
        proposal_offset += count

    def concatenate(key: str, shape: tuple[int, ...] = (0,)) -> torch.Tensor:
        values = aligned_chunks[key]
        return torch.cat(values) if values else torch.empty(shape)

    payload_out: dict[str, object] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "scene": str(args.scene),
        "num_rows": num_gaussians,
        "num_proposals": proposal_offset,
        "row_indices": torch.cat(row_chunks) if row_chunks else torch.empty(0, dtype=torch.long),
        "proposal_indices": torch.cat(proposal_chunks) if proposal_chunks else torch.empty(0, dtype=torch.long),
        "weights": torch.cat(weight_chunks) if weight_chunks else torch.empty(0),
        "proposal_view_indices": torch.tensor(proposal_views, dtype=torch.long),
        "proposal_frame_indices": torch.tensor(proposal_frames, dtype=torch.long),
        "proposal_scores": concatenate("quality"),
        "proposal_stability": concatenate("stability"),
        "proposal_area_fraction": concatenate("proposal_area_fraction"),
        "proposal_boxes_xyxy": concatenate("boxes_xyxy", (0, 4)).long(),
        "proposal_seed_xy": concatenate("seed_xy_full", (0, 2)),
        "proposal_prompt_index": concatenate("prompt_index").long(),
        "proposal_candidate_index": concatenate("candidate_index").long(),
        "proposal_crop_index": concatenate("crop_index").long(),
        "proposal_crop_layer": concatenate("crop_layer").long(),
        "proposal_crop_grid_side": concatenate("crop_grid_side").long(),
        "proposal_crop_boxes_xyxy": concatenate("crop_boxes_xyxy", (0, 4)).long(),
        "proposal_crop_scale_xy": concatenate("crop_scale_xy", (0, 2)),
        "proposal_crop_window_area_fraction": concatenate("crop_window_area_fraction"),
        "proposal_crop_area_fraction": concatenate("crop_area_fraction"),
        "proposal_touches_crop_edge": concatenate("touches_crop_edge", (0, 4)).bool(),
        "proposal_parent_index": torch.cat(parent_chunks) if parent_chunks else torch.empty(0, dtype=torch.long),
        "proposal_parent_edges": torch.cat(edge_chunks) if edge_chunks else torch.empty((0, 2), dtype=torch.long),
        "proposal_parent_containment": concatenate("parent_containment"),
        "proposal_parent_area_ratio": concatenate("parent_area_ratio"),
        "view_denominator": torch.stack(view_denominators),
        "view_observed": torch.stack(view_denominators) > 0,
        "metadata": {
            "query_independent_proposal_set": True,
            "query_independent_mask_hierarchy": True,
            "hierarchy_parent_edges_materialized": True,
            "proposal_set_type": "multiscale_crop_pyramid_point_grid_multimask",
            "official_decoder": True,
            "mask_tensor_semantics": "packed_boolean",
            "mask_raster_alignment": "nearest_label_resample_to_exact_mpr_raster",
            "membership_lifting": "exact_front_to_back_marginal_target_weight",
            "min_membership": float(args.min_membership),
            "feature_height": feature_height,
            "feature_width": feature_width,
            "xyz_sha256": xyz_sha256,
            "source_view_count": len(records),
            "source_records": records,
            "primitive_cache": str(primitive_path),
            "primitive_cache_sha256": sha256_file(primitive_path),
            "responsibility_authority": str(authority_path),
            "responsibility_authority_sha256": hashlib.sha256(authority_bytes).hexdigest(),
            "source_authority": str(source_authority_path),
            "source_authority_sha256": source_sha,
            "multiscale_manifest": str(manifest_path),
            "multiscale_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "official_sam3_checkpoint_sha256": checkpoint_sha,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "evaluation_rgb_opened": False,
            "text_queries_opened": False,
            "capability_track": "query_free_source32_multiscale_exact_mpr_development_closure",
            "formal_stage_a_complete": False,
        },
    }
    if (
        payload_out["proposal_view_indices"].numel() != proposal_offset
        or payload_out["proposal_parent_index"].numel() != proposal_offset
        or payload_out["proposal_scores"].numel() != proposal_offset
    ):
        raise AssertionError("global multiscale proposal axes differ")
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
        "num_memberships": int(payload_out["row_indices"].numel()),
        "source_view_count": len(records),
        "query_independent_mask_hierarchy": True,
        "formal_stage_a_complete": False,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--responsibility-authority", required=True)
    parser.add_argument("--primitive-cache", required=True)
    parser.add_argument("--source-authority", required=True)
    parser.add_argument("--mask-root", required=True)
    parser.add_argument("--manifest-name", default="manifest.json")
    parser.add_argument("--min-membership", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    print(json.dumps(build(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
