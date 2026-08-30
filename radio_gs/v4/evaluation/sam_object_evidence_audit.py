"""Audit real query-free SAM proposals before mask-token training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radio_gs.v4.carrier import MeshCarrier, SurfaceVoxelCarrier
from radio_gs.v4.contracts.geometry_receipt import GeometryReceipt, HashedInput, sha256_file
from radio_gs.v4.evaluation.geometry_ladder import _load_cameras, _load_mesh
from radio_gs.v4.evaluation.lerf_source_mask_gate import _load_sam_records, _masks
from radio_gs.v4.evaluation.object_oracle_gate import _load_instance_labels


def _proposal_association(
    masks: torch.Tensor,
    oracle_raster: torch.Tensor,
    *,
    minimum_purity: float,
    minimum_margin: float,
    whole_recall: float,
) -> dict[str, torch.Tensor]:
    """Partial diagnostic association; ambiguous/merged masks stay unmatched."""

    binary = masks > 0.5
    overlap = binary.float().flatten(1) @ oracle_raster.reshape(-1, oracle_raster.shape[-1])
    mask_area = binary.sum((1, 2)).float().clamp_min(1)
    visible_object_area = oracle_raster.sum((0, 1)).clamp_min(1e-12)
    values, ids = overlap.topk(min(2, overlap.shape[1]), dim=-1)
    best = values[:, 0]
    second = values[:, 1] if values.shape[1] > 1 else torch.zeros_like(best)
    purity = best / mask_area
    margin = (best - second) / mask_area
    recall = best / visible_object_area[ids[:, 0]]
    associated = (purity >= minimum_purity) & (margin >= minimum_margin)
    whole = associated & (recall >= whole_recall)
    part = associated & ~whole
    return {
        "token_id": ids[:, 0],
        "purity": purity,
        "margin": margin,
        "visible_recall": recall,
        "associated": associated,
        "whole": whole,
        "part": part,
        "ambiguous": ~associated,
    }


def _reason_counts(
    *,
    visible: torch.Tensor,
    covered: torch.Tensor,
    associated: torch.Tensor,
    ground_truth_token: torch.Tensor,
    observed_tokens: torch.Tensor,
    token_labels: list[str],
) -> dict[str, Any]:
    unknown = ~associated
    annotated = ground_truth_token >= 0
    observed_object = torch.zeros_like(annotated)
    observed_object[annotated] = observed_tokens[ground_truth_token[annotated]]
    masks = {
        "A_never_visible_without_observed_token": unknown & ~visible & ~(annotated & observed_object),
        "B_visible_without_any_sam_mask": unknown & visible & ~covered,
        "C_sam_covered_without_safe_token_association": unknown & visible & covered,
        "D_safely_associated_but_not_committed": torch.zeros_like(unknown),
        "E_unseen_surface_of_an_observed_object": unknown & ~visible & annotated & observed_object,
    }
    total = int(unknown.sum())
    records = {}
    for name, selected in masks.items():
        histogram: dict[str, int] = {}
        for token_index in ground_truth_token[selected & annotated].tolist():
            label = token_labels[int(token_index)]
            histogram[label] = histogram.get(label, 0) + 1
        records[name] = {
            "count": int(selected.sum()),
            "fraction_of_unassociated": float(selected.sum()) / max(total, 1),
            "annotated_object_count": int((selected & annotated).sum()),
            "largest_raw_label_groups": [
                {"raw_label": label, "element_count": count}
                for label, count in sorted(histogram.items(), key=lambda item: (-item[1], item[0]))[:10]
            ],
        }
    return records


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.allow_instance_oracle_diagnostic:
        raise PermissionError("SAM evidence audit requires explicit instance-oracle authorization")
    transforms = Path(args.transforms).resolve(strict=True)
    mesh_path = Path(args.mesh).resolve(strict=True)
    segmentation_path = Path(args.segmentation).resolve(strict=True)
    aggregation_path = Path(args.aggregation).resolve(strict=True)
    authority_path = Path(args.source_rgb_authority).resolve(strict=True)
    selection_path = Path(args.selection_authority).resolve(strict=True)
    authority = json.loads(authority_path.read_text())
    selection = json.loads(selection_path.read_text())
    if authority.get("information_policy", {}).get("benchmark_ground_truth_used") is not False:
        raise ValueError("source RGB authority is not label-free")
    sam_paths = [Path(value).resolve(strict=True) for value in args.sam_manifest]
    sam_records = _load_sam_records(sam_paths)

    selected64 = list(map(int, selection["selections"][str(args.maximum_view_count)]["selected_frame_ids"]))
    if set(sam_records) != set(selected64):
        raise ValueError("SAM cache frames differ from the selected source cohort")
    cameras = _load_cameras(
        transforms,
        [{"frame_id": value} for value in selected64],
        args.feature_height,
        args.feature_width,
    )
    vertices, triangles = _load_mesh(mesh_path)
    vertex_instances, object_ids = _load_instance_labels(
        segmentation_path, aggregation_path, vertices.shape[0]
    )
    aggregation = json.loads(aggregation_path.read_text())
    labels_by_instance = {
        int(group["objectId"]) + 1: str(group.get("label", ""))
        for group in aggregation.get("segGroups", [])
    }
    token_labels = [labels_by_instance[value] for value in object_ids]
    oracle = MeshCarrier(vertices, triangles)
    surface = SurfaceVoxelCarrier.from_points(
        vertices,
        args.voxel_size,
        normals=oracle.normals,
        maximum_splat_radius=args.maximum_splat_radius,
        surface_band_voxels=args.surface_band_voxels,
        maximum_contributors_per_pixel=args.maximum_contributors_per_pixel,
    )
    lookup = torch.full((int(vertex_instances.max()) + 1,), -1, dtype=torch.long)
    lookup[torch.tensor(object_ids)] = torch.arange(len(object_ids))
    vertex_tokens = lookup[vertex_instances]
    vertex_membership = torch.zeros(vertices.shape[0], len(object_ids))
    annotated_vertex = vertex_tokens >= 0
    vertex_membership[annotated_vertex, vertex_tokens[annotated_vertex]] = 1

    visible_by_view, covered_by_view, associated_by_view, token_by_view = [], [], [], []
    proposal_records = []
    for frame_id, camera in zip(selected64, cameras):
        masks = _masks(Path(sam_records[frame_id]["output"]), args.feature_height, args.feature_width)
        oracle_raster = oracle.render_posterior(vertex_membership, camera)
        association = _proposal_association(
            masks,
            oracle_raster,
            minimum_purity=args.minimum_association_purity,
            minimum_margin=args.minimum_association_margin,
            whole_recall=args.whole_object_recall,
        )
        visible = torch.zeros(surface.num_elements, dtype=torch.bool)
        projection = surface.project(camera)
        visible[projection.element_ids] = True
        union = (masks > 0.5).any(0).float()
        covered_evidence = surface.lift(union, camera)
        covered = covered_evidence.mean[:, 0] > args.evidence_epsilon
        safe_union = (
            (masks[association["associated"]] > 0.5).any(0).float()
            if bool(association["associated"].any())
            else torch.zeros(args.feature_height, args.feature_width)
        )
        associated_evidence = surface.lift(safe_union, camera)
        safely_associated = associated_evidence.mean[:, 0] > args.evidence_epsilon
        observed_tokens = torch.zeros(len(object_ids), dtype=torch.bool)
        observed_tokens[association["token_id"][association["associated"]]] = True
        visible_by_view.append(visible)
        covered_by_view.append(covered)
        associated_by_view.append(safely_associated)
        token_by_view.append(observed_tokens)
        proposal_records.append({
            "frame_id": frame_id,
            "proposal_count": int(masks.shape[0]),
            "whole_object_proposal_count": int(association["whole"].sum()),
            "part_proposal_count": int(association["part"].sum()),
            "ambiguous_or_merged_proposal_count": int(association["ambiguous"].sum()),
            "mean_best_token_purity": float(association["purity"].mean()),
            "mean_best_token_margin": float(association["margin"].mean()),
        })
    visible_by_view = torch.stack(visible_by_view)
    covered_by_view = torch.stack(covered_by_view)
    associated_by_view = torch.stack(associated_by_view)
    token_by_view = torch.stack(token_by_view)

    from scipy.spatial import cKDTree
    nearest = cKDTree(vertices.numpy()).query(surface.centres.numpy(), k=1)[1]
    ground_truth_token = vertex_tokens[torch.from_numpy(np.asarray(nearest, dtype=np.int64))]
    annotated = ground_truth_token >= 0
    ladders = []
    for count in args.view_count:
        indices = list(range(count))
        visible = visible_by_view[indices].any(0)
        covered = covered_by_view[indices].any(0)
        associated = associated_by_view[indices].any(0)
        observed_tokens = token_by_view[indices].any(0)
        proposals = proposal_records[:count]
        ladders.append({
            "selected_view_count": count,
            "proposal_count": sum(row["proposal_count"] for row in proposals),
            "whole_object_proposal_count": sum(row["whole_object_proposal_count"] for row in proposals),
            "part_proposal_count": sum(row["part_proposal_count"] for row in proposals),
            "ambiguous_or_merged_proposal_count": sum(row["ambiguous_or_merged_proposal_count"] for row in proposals),
            "surface_visibility_union": float(visible.float().mean()),
            "sam_covered_surface_union": float(covered.float().mean()),
            "safely_associated_surface_union": float(associated.float().mean()),
            "annotated_safely_associated_surface_union": float(associated[annotated].float().mean()),
            "observed_token_count": int(observed_tokens.sum()),
            "unknown_reasons": _reason_counts(
                visible=visible,
                covered=covered,
                associated=associated,
                ground_truth_token=ground_truth_token,
                observed_tokens=observed_tokens,
                token_labels=token_labels,
            ),
        })
    receipt = GeometryReceipt(
        carrier="frozen_sparse_surface_real_sam_object_evidence_audit",
        coordinate_convention="source_sam_to_sparse_surface_feature_raster",
        inputs=(
            HashedInput.seal("source_rgb_authority", authority_path),
            HashedInput.seal("label_free_view_selection", selection_path),
            HashedInput.seal("camera_transforms", transforms),
            HashedInput.seal("mesh_geometry", mesh_path),
            HashedInput.seal("diagnostic_instance_segmentation", segmentation_path),
            HashedInput.seal("diagnostic_instance_aggregation", aggregation_path),
            *tuple(HashedInput.seal(f"sam_manifest_{i}", path) for i, path in enumerate(sam_paths)),
        ),
        source_rgb_opened=False,
        target_rgb_opened=False,
        benchmark_images_opened=False,
        benchmark_masks_opened=False,
        benchmark_labels_opened=True,
        metadata={
            "diagnostic_labels_used_only_to_measure_association": True,
            "matching_type": "partial_null_capable_diagnostic_not_hungarian",
            "completion_trained": False,
        },
    )
    report = {
        "schema": "radio_gs.surface_object_memory_v4.real_sam_object_evidence_audit.v1",
        "stage": "pretraining_diagnostic",
        "ladder": ladders,
        "per_view_proposals": proposal_records,
        "association_policy": {
            "minimum_purity": args.minimum_association_purity,
            "minimum_margin": args.minimum_association_margin,
            "whole_object_visible_recall": args.whole_object_recall,
            "ambiguous_masks_remain_unmatched": True,
            "part_masks_are_auxiliary_not_object_membership_targets": True,
        },
        "geometry_receipt": receipt.to_dict(),
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-rgb-authority", required=True)
    parser.add_argument("--selection-authority", required=True)
    parser.add_argument("--sam-manifest", action="append", required=True)
    parser.add_argument("--transforms", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--segmentation", required=True)
    parser.add_argument("--aggregation", required=True)
    parser.add_argument("--maximum-view-count", type=int, default=64)
    parser.add_argument("--view-count", action="append", type=int, default=[])
    parser.add_argument("--feature-height", type=int, default=60)
    parser.add_argument("--feature-width", type=int, default=81)
    parser.add_argument("--voxel-size", type=float, default=0.04)
    parser.add_argument("--maximum-splat-radius", type=int, default=1)
    parser.add_argument("--surface-band-voxels", type=float, default=1.5)
    parser.add_argument("--maximum-contributors-per-pixel", type=int, default=8)
    parser.add_argument("--minimum-association-purity", type=float, default=0.70)
    parser.add_argument("--minimum-association-margin", type=float, default=0.20)
    parser.add_argument("--whole-object-recall", type=float, default=0.50)
    parser.add_argument("--evidence-epsilon", type=float, default=1e-6)
    parser.add_argument("--allow-instance-oracle-diagnostic", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.view_count:
        args.view_count = [16, 32, 64]
    report = run(args)
    print(json.dumps(report["ladder"], indent=2))


if __name__ == "__main__":
    main()
