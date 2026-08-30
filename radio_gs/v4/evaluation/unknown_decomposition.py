"""Decompose source-lifted object unknowns before any completion training."""

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
from radio_gs.v4.evaluation.object_oracle_gate import _load_instance_labels
from radio_gs.v4.object_memory import SparseObjectAssignments
from radio_gs.v4.registration.evidence_fusion import fuse_evidence_tables


REASONS = (
    "A_never_visible_without_observed_token",
    "B_visible_without_object_mask_evidence",
    "C_mask_evidence_without_token_association",
    "D_associated_but_not_committed",
    "E_unseen_surface_of_an_observed_object",
)


def _classify_unknown(
    *,
    hard_unknown: torch.Tensor,
    visible: torch.Tensor,
    mask_covered: torch.Tensor,
    associated: torch.Tensor,
    committed: torch.Tensor,
    ground_truth_token: torch.Tensor,
    token_observed: torch.Tensor,
) -> torch.Tensor:
    """Return a mutually exclusive A--E code for every hard-unknown element."""

    size = hard_unknown.numel()
    reason = torch.full((size,), -1, dtype=torch.long)
    annotated = ground_truth_token >= 0
    observed_object = torch.zeros(size, dtype=torch.bool)
    observed_object[annotated] = token_observed[ground_truth_token[annotated]]
    reason[hard_unknown & ~visible & annotated & observed_object] = 4
    reason[hard_unknown & ~visible & (reason < 0)] = 0
    reason[hard_unknown & visible & ~mask_covered] = 1
    reason[hard_unknown & visible & mask_covered & ~associated] = 2
    reason[hard_unknown & visible & associated & ~committed] = 3
    if bool((hard_unknown & (reason < 0)).any()):
        raise RuntimeError("unknown-reason decomposition left elements unclassified")
    return reason


def _summarize_reasons(
    reason: torch.Tensor,
    hard_unknown: torch.Tensor,
    unknown_weight: torch.Tensor,
    annotated: torch.Tensor,
) -> dict[str, Any]:
    total_count = int(hard_unknown.sum())
    total_mass = float(unknown_weight.sum())
    records = {}
    for code, name in enumerate(REASONS):
        selected = reason == code
        count = int(selected.sum())
        mass = float(unknown_weight[selected].sum())
        records[name] = {
            "hard_unknown_element_count": count,
            "hard_unknown_element_fraction": count / max(total_count, 1),
            "unknown_probability_mass": mass,
            "unknown_probability_mass_fraction": mass / max(total_mass, 1e-12),
            "annotated_object_element_count": int((selected & annotated).sum()),
        }
    return records


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.allow_instance_oracle_diagnostic:
        raise PermissionError("unknown decomposition requires explicit instance-oracle authorization")
    transforms = Path(args.transforms).resolve(strict=True)
    mesh_path = Path(args.mesh).resolve(strict=True)
    segmentation_path = Path(args.segmentation).resolve(strict=True)
    aggregation_path = Path(args.aggregation).resolve(strict=True)
    authority_path = Path(args.source_authority).resolve(strict=True)
    milestone_path = Path(args.milestone_receipt).resolve(strict=True)
    milestone = json.loads(milestone_path.read_text())
    if milestone.get("decisions", {}).get("carrier_parameters_frozen") is not True:
        raise PermissionError("unknown decomposition requires the frozen carrier milestone")

    authority = torch.load(authority_path, map_location="cpu")
    metadata = authority["metadata"]
    records = list(metadata["source_records"])
    height, width = int(metadata["feature_height"]), int(metadata["feature_width"])
    cameras = _load_cameras(transforms, records, height, width)
    heldout = [index for index in range(len(cameras)) if (index + 1) % args.heldout_stride == 0]
    mapping = [index for index in range(len(cameras)) if index not in heldout]

    vertices, triangles = _load_mesh(mesh_path)
    vertex_instances, object_ids = _load_instance_labels(
        segmentation_path, aggregation_path, vertices.shape[0]
    )
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

    visible = torch.zeros(surface.num_elements, dtype=torch.bool)
    evidence = []
    for index in mapping:
        camera = cameras[index]
        oracle_projection = oracle.project(camera)
        oracle_raster = oracle.render_posterior(vertex_membership, camera)
        support = oracle_projection.pixel_weight_sum() > 0
        surface_projection = surface.project(camera)
        visible[surface_projection.element_ids] = True
        evidence.append(
            surface.lift(oracle_raster, camera, state=torch.where(support, 1, -1))
        )
    fused = fuse_evidence_tables(evidence)
    dense_evidence = fused.mean.clamp(0, 1)
    assignments = SparseObjectAssignments.from_dense(
        dense_evidence,
        unknown_weight=(1.0 - dense_evidence.sum(-1)).clamp(0, 1),
        top_k=2,
    )
    membership = assignments.to_dense()
    committed = membership.sum(-1) > args.evidence_epsilon
    mask_covered = dense_evidence.sum(-1) > args.evidence_epsilon
    # This diagnostic receives exact oracle IDs; association cannot fail here.
    associated = mask_covered.clone()
    hard_unknown = ~committed

    from scipy.spatial import cKDTree
    nearest = cKDTree(vertices.numpy()).query(surface.centres.numpy(), k=1)[1]
    ground_truth_token = vertex_tokens[torch.from_numpy(np.asarray(nearest, dtype=np.int64))]
    annotated = ground_truth_token >= 0
    token_observed = membership.sum(0) > args.evidence_epsilon
    reason = _classify_unknown(
        hard_unknown=hard_unknown,
        visible=visible,
        mask_covered=mask_covered,
        associated=associated,
        committed=committed,
        ground_truth_token=ground_truth_token,
        token_observed=token_observed,
    )
    reason_summary = _summarize_reasons(
        reason, hard_unknown, assignments.unknown_weight, annotated
    )

    stage = {
        "surface_visibility_union": float(visible.float().mean()),
        "annotated_surface_visibility_union": float(visible[annotated].float().mean()),
        "object_mask_covered_surface_union": float(mask_covered.float().mean()),
        "successfully_associated_surface_union": float(associated.float().mean()),
        "committed_membership_surface_union": float(committed.float().mean()),
        "committed_annotated_object_surface_union": float(committed[annotated].float().mean()),
        "mean_unknown_probability": float(assignments.unknown_weight.mean()),
        "hard_unknown_element_fraction": float(hard_unknown.float().mean()),
        "annotated_object_hard_unknown_fraction": float(hard_unknown[annotated].float().mean()),
    }
    receipt = GeometryReceipt(
        carrier="frozen_sparse_surface_unknown_decomposition",
        coordinate_convention="mesh_oracle_to_sparse_surface_feature_raster",
        inputs=(
            HashedInput.seal("milestone_receipt", milestone_path),
            HashedInput.seal("source_authority", authority_path),
            HashedInput.seal("camera_transforms", transforms),
            HashedInput.seal("mesh_geometry", mesh_path),
            HashedInput.seal("instance_segmentation", segmentation_path),
            HashedInput.seal("instance_aggregation", aggregation_path),
        ),
        source_rgb_opened=False,
        target_rgb_opened=False,
        benchmark_images_opened=False,
        benchmark_masks_opened=False,
        benchmark_labels_opened=True,
        metadata={
            "diagnostic_only": True,
            "mapping_view_indices": mapping,
            "heldout_view_indices": heldout,
            "association_mode": "exact_oracle_id_so_reason_C_is_structurally_zero",
        },
    )
    largest_reason = max(
        REASONS,
        key=lambda name: reason_summary[name]["annotated_object_element_count"],
    )
    report = {
        "schema": "radio_gs.surface_object_memory_v4.unknown_reason_decomposition.v1",
        "stage": "pretraining_diagnostic",
        "mapping_view_count": len(mapping),
        "heldout_view_count": len(heldout),
        "coverage_stages": stage,
        "unknown_reasons": reason_summary,
        "largest_annotated_object_unknown_reason": largest_reason,
        "diagnostic_boundaries": {
            "reason_C_measurable_in_this_oracle_id_arm": False,
            "reason_D_includes_no_threshold_relaxation": True,
            "unannotated_exterior_is_reported_separately_from_missing_object_surface": True,
            "completion_trained": False,
        },
        "counts": {
            "surface_elements": surface.num_elements,
            "annotated_object_elements": int(annotated.sum()),
            "unannotated_exterior_elements": int((~annotated).sum()),
            "hard_unknown_elements": int(hard_unknown.sum()),
            "hard_unknown_annotated_object_elements": int((hard_unknown & annotated).sum()),
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
    parser.add_argument("--milestone-receipt", required=True)
    parser.add_argument("--transforms", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--segmentation", required=True)
    parser.add_argument("--aggregation", required=True)
    parser.add_argument("--source-authority", required=True)
    parser.add_argument("--voxel-size", type=float, default=0.04)
    parser.add_argument("--maximum-splat-radius", type=int, default=1)
    parser.add_argument("--surface-band-voxels", type=float, default=1.5)
    parser.add_argument("--maximum-contributors-per-pixel", type=int, default=8)
    parser.add_argument("--heldout-stride", type=int, default=4)
    parser.add_argument("--evidence-epsilon", type=float, default=1e-6)
    parser.add_argument("--allow-instance-oracle-diagnostic", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "coverage_stages": report["coverage_stages"],
        "largest_reason": report["largest_annotated_object_unknown_reason"],
        "unknown_reasons": report["unknown_reasons"],
    }, indent=2))


if __name__ == "__main__":
    main()
