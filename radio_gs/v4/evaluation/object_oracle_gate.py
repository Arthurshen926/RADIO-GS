"""Oracle-ID object-codebook gate on a held-out camera split.

This opt-in diagnostic opens 3-D instance annotations only to establish whether
the sparse carrier and top-2 codebook can retain object extent.  Instance IDs
are never exported as deployable semantic keys and no query encoder is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radio_gs.v4.carrier import MeshCarrier, SurfaceVoxelCarrier
from radio_gs.v4.contracts.geometry_receipt import GeometryReceipt, HashedInput, sha256_file
from radio_gs.v4.contracts.method_receipt import MethodReceipt
from radio_gs.v4.evaluation.geometry_ladder import _load_cameras, _load_mesh
from radio_gs.v4.object_memory import ObjectCodebook, SparseObjectAssignments
from radio_gs.v4.registration.evidence_fusion import fuse_evidence_tables


def _load_instance_labels(
    segmentation_path: Path,
    aggregation_path: Path,
    vertex_count: int,
) -> tuple[torch.Tensor, list[int]]:
    segmentation = json.loads(segmentation_path.read_text())
    aggregation = json.loads(aggregation_path.read_text())
    segment_ids = np.asarray(segmentation.get("segIndices", []), dtype=np.int64)
    if segment_ids.shape != (vertex_count,):
        raise ValueError("instance segmentation does not align with mesh vertices")
    labels = np.zeros(vertex_count, dtype=np.int64)
    object_ids = []
    for group in aggregation.get("segGroups", []):
        instance_id = int(group["objectId"]) + 1
        selected = np.isin(segment_ids, np.asarray(group.get("segments", []), dtype=np.int64))
        if bool((labels[selected] != 0).any()):
            raise ValueError("official instance groups overlap")
        labels[selected] = instance_id
        object_ids.append(instance_id)
    if not object_ids:
        raise ValueError("instance aggregation contains no object groups")
    return torch.from_numpy(labels), sorted(object_ids)


def _soft_iou_per_token(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eligible: torch.Tensor,
) -> list[float]:
    intersection = (prediction * target).sum(0)
    union = prediction.sum(0) + target.sum(0) - intersection
    selected = eligible & (target.sum(0) > 0) & (union > 0)
    return list(map(float, (intersection[selected] / union[selected].clamp_min(1e-12))))


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.allow_instance_oracle_diagnostic:
        raise PermissionError("object oracle gate requires explicit instance-label authorization")
    transforms = Path(args.transforms).resolve(strict=True)
    mesh_path = Path(args.mesh).resolve(strict=True)
    segmentation_path = Path(args.segmentation).resolve(strict=True)
    aggregation_path = Path(args.aggregation).resolve(strict=True)
    authority_path = Path(args.source_authority).resolve(strict=True)
    geometry_gate_path = Path(args.geometry_gate).resolve(strict=True)
    geometry_gate = json.loads(geometry_gate_path.read_text())
    if geometry_gate.get("milestone_1_complete") is not True:
        raise PermissionError("object oracle remains blocked until Milestone 1 is complete")
    if geometry_gate.get("object_codebook_authorized_scope") != "oracle_only":
        raise PermissionError("geometry gate did not authorize the oracle codebook scope")

    authority = torch.load(authority_path, map_location="cpu")
    metadata = authority["metadata"]
    records = list(metadata["source_records"])
    height, width = int(metadata["feature_height"]), int(metadata["feature_width"])
    cameras = _load_cameras(transforms, records, height, width)
    if args.heldout_stride < 2:
        raise ValueError("heldout stride must be at least two")
    heldout_indices = [
        index for index in range(len(cameras)) if (index + 1) % args.heldout_stride == 0
    ]
    mapping_indices = [index for index in range(len(cameras)) if index not in heldout_indices]
    if not mapping_indices or not heldout_indices:
        raise ValueError("oracle codebook requires non-empty mapping and held-out view splits")

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
    # Compact oracle identities are diagnostic association keys, not semantic labels.
    lookup = torch.full((int(vertex_instances.max()) + 1,), -1, dtype=torch.long)
    lookup[torch.tensor(object_ids)] = torch.arange(len(object_ids))
    vertex_tokens = lookup[vertex_instances]
    vertex_membership = torch.zeros(vertices.shape[0], len(object_ids))
    annotated_vertex = vertex_tokens >= 0
    vertex_membership[annotated_vertex, vertex_tokens[annotated_vertex]] = 1

    oracle_rasters, oracle_support = [], []
    for camera in cameras:
        oracle_rasters.append(oracle.render_posterior(vertex_membership, camera))
        oracle_support.append(oracle.project(camera).pixel_weight_sum() > 0)
    mapping_evidence = [
        surface.lift(
            oracle_rasters[index],
            cameras[index],
            state=torch.where(oracle_support[index], 1, -1),
        )
        for index in mapping_indices
    ]
    fused = fuse_evidence_tables(mapping_evidence)
    dense_evidence = fused.mean.clamp(0, 1)
    dense_sum = dense_evidence.sum(-1)
    unknown = (1.0 - dense_sum).clamp(0, 1)
    assignments = SparseObjectAssignments.from_dense(
        dense_evidence,
        unknown_weight=unknown,
        top_k=2,
    )
    codebook = ObjectCodebook.from_assignments(surface.centres, assignments)
    membership = assignments.to_dense()
    eligible = membership.sum(0) >= args.minimum_token_mass

    heldout_2d_iou: list[float] = []
    heldout_visible = torch.zeros(surface.num_elements, dtype=torch.bool)
    per_view = []
    for index in heldout_indices:
        prediction = surface.render_posterior(membership, cameras[index]).reshape(-1, len(object_ids))
        target = oracle_rasters[index].reshape(-1, len(object_ids))
        values = _soft_iou_per_token(prediction, target, eligible)
        heldout_2d_iou.extend(values)
        heldout_visible[surface.project(cameras[index]).element_ids] = True
        per_view.append({
            "camera_key": cameras[index].key,
            "eligible_visible_token_count": len(values),
            "soft_iou": float(np.mean(values)) if values else None,
        })

    # Nearest official mesh vertex defines a geometry-only 3-D diagnostic target.
    from scipy.spatial import cKDTree
    nearest = cKDTree(vertices.numpy()).query(surface.centres.numpy(), k=1)[1]
    surface_vertex_tokens = vertex_tokens[torch.from_numpy(np.asarray(nearest, dtype=np.int64))]
    target_3d = torch.zeros_like(membership)
    annotated_surface = surface_vertex_tokens >= 0
    target_3d[annotated_surface, surface_vertex_tokens[annotated_surface]] = 1
    evaluate_3d = heldout_visible[:, None].expand_as(membership)
    iou_3d = _soft_iou_per_token(
        membership[evaluate_3d[:, 0]],
        target_3d[evaluate_3d[:, 0]],
        eligible,
    )
    known = membership.sum(-1) > 0
    purity = membership[known].max(-1).values / membership[known].sum(-1).clamp_min(1e-12)
    predicted_top1 = membership.argmax(-1)
    assigned = known & annotated_surface & heldout_visible
    top1_accuracy = float((predicted_top1[assigned] == surface_vertex_tokens[assigned]).float().mean()) if bool(assigned.any()) else float("nan")

    source_lifted_metrics = {
        "heldout_2d_soft_miou": float(np.mean(heldout_2d_iou)),
        "heldout_3d_soft_miou": float(np.mean(iou_3d)),
        "element_assignment_purity": float(purity.mean()),
        "heldout_visible_top1_accuracy": top1_accuracy,
        "known_element_fraction": float(known.float().mean()),
        "mean_unknown_weight": float(assignments.unknown_weight.mean()),
        "eligible_token_count": int(eligible.sum()),
        "total_oracle_token_count": len(object_ids),
        "heldout_2d_token_observation_count": len(heldout_2d_iou),
        "heldout_3d_token_observation_count": len(iou_3d),
    }
    exact_assignments = SparseObjectAssignments.from_dense(
        target_3d,
        unknown_weight=(1.0 - target_3d.sum(-1)).clamp(0, 1),
        top_k=2,
    )
    exact_membership = exact_assignments.to_dense()
    exact_eligible = exact_membership.sum(0) >= args.minimum_token_mass
    exact_2d_iou: list[float] = []
    for index in heldout_indices:
        prediction = surface.render_posterior(exact_membership, cameras[index]).reshape(-1, len(object_ids))
        target = oracle_rasters[index].reshape(-1, len(object_ids))
        exact_2d_iou.extend(_soft_iou_per_token(prediction, target, exact_eligible))
    exact_3d_iou = _soft_iou_per_token(
        exact_membership[heldout_visible],
        target_3d[heldout_visible],
        exact_eligible,
    )
    exact_known = exact_membership.sum(-1) > 0
    exact_purity = (
        exact_membership[exact_known].max(-1).values
        / exact_membership[exact_known].sum(-1).clamp_min(1e-12)
    )
    oracle_metrics = {
        "heldout_2d_soft_miou": float(np.mean(exact_2d_iou)),
        "heldout_3d_soft_miou": float(np.mean(exact_3d_iou)),
        "element_assignment_purity": float(exact_purity.mean()),
        "known_element_fraction": float(exact_known.float().mean()),
        "mean_unknown_weight": float(exact_assignments.unknown_weight.mean()),
        "eligible_token_count": int(exact_eligible.sum()),
        "total_oracle_token_count": len(object_ids),
    }
    primary = {
        "heldout_2d": oracle_metrics["heldout_2d_soft_miou"] >= args.minimum_iou,
        "heldout_3d": oracle_metrics["heldout_3d_soft_miou"] >= args.minimum_iou,
        "purity": oracle_metrics["element_assignment_purity"] >= args.minimum_purity,
        "nondegenerate_tokens": oracle_metrics["eligible_token_count"] >= args.minimum_token_count,
    }
    source_lifted_passes = (
        source_lifted_metrics["heldout_2d_soft_miou"] >= args.minimum_iou
        and source_lifted_metrics["heldout_3d_soft_miou"] >= args.minimum_iou
        and source_lifted_metrics["element_assignment_purity"] >= args.minimum_purity
        and source_lifted_metrics["eligible_token_count"] >= args.minimum_token_count
    )
    receipt = GeometryReceipt(
        carrier="sparse_surface_oracle_object_codebook",
        coordinate_convention="scannet_mesh_nerf_opengl_to_opencv_feature_raster",
        inputs=(
            HashedInput.seal("geometry_gate", geometry_gate_path),
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
            "oracle_object_id_diagnostic_only": True,
            "mapping_view_indices": mapping_indices,
            "heldout_view_indices": heldout_indices,
            "heldout_stride": args.heldout_stride,
            "top_k": 2,
            "voxel_size": args.voxel_size,
            "maximum_splat_radius": args.maximum_splat_radius,
        },
    )
    report = {
        "schema": "radio_gs.surface_object_memory_v4.object_codebook_oracle_gate.v1",
        "stage": "object_codebook_oracle",
        "same_element_posterior_used_for_2d_and_3d": True,
        "instance_ids_persisted_as_deployment_semantics": False,
        "mapping_view_count": len(mapping_indices),
        "heldout_view_count": len(heldout_indices),
        "oracle_metrics": oracle_metrics,
        "source_lifted_diagnostic_metrics": source_lifted_metrics,
        "primary_directions": primary,
        "passes_oracle_object_gate": all(primary.values()),
        "learned_soft_codebook_authorized": source_lifted_passes,
        "learned_soft_codebook_block_reason": None if source_lifted_passes else (
            "source-lifted memberships do not yet meet the held-out 2-D/3-D oracle thresholds"
        ),
        "per_heldout_view": per_view,
        "storage": {
            "element_count": surface.num_elements,
            "token_count": len(object_ids),
            "assignment_slots_per_element": 2,
            "dense_membership_values_avoided": surface.num_elements * len(object_ids) - 2 * surface.num_elements,
        },
        "geometry_receipt": receipt.to_dict(),
        "method_receipt": MethodReceipt(
            stage="object_codebook_oracle",
            carrier="sparse_surface_voxel",
            geometry_receipt_sha256=sha256_file(geometry_gate_path),
            codebook_enabled=True,
        ).to_dict(),
        "gate_thresholds": {
            "minimum_iou": args.minimum_iou,
            "minimum_purity": args.minimum_purity,
            "minimum_token_count": args.minimum_token_count,
            "minimum_token_mass": args.minimum_token_mass,
        },
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-gate", required=True)
    parser.add_argument("--transforms", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--segmentation", required=True)
    parser.add_argument("--aggregation", required=True)
    parser.add_argument("--source-authority", required=True)
    parser.add_argument("--voxel-size", type=float, default=0.04)
    parser.add_argument("--maximum-splat-radius", type=int, default=1)
    parser.add_argument("--surface-band-voxels", type=float, default=1.5)
    parser.add_argument("--maximum-contributors-per-pixel", type=int, default=8)
    parser.add_argument("--minimum-iou", type=float, default=0.5)
    parser.add_argument("--minimum-purity", type=float, default=0.9)
    parser.add_argument("--minimum-token-count", type=int, default=5)
    parser.add_argument("--minimum-token-mass", type=float, default=1.0)
    parser.add_argument("--heldout-stride", type=int, default=4)
    parser.add_argument("--allow-instance-oracle-diagnostic", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "passes": report["passes_oracle_object_gate"],
        "oracle_metrics": report["oracle_metrics"],
        "source_lifted_diagnostic_metrics": report["source_lifted_diagnostic_metrics"],
        "learned_soft_codebook_authorized": report["learned_soft_codebook_authorized"],
    }, indent=2))


if __name__ == "__main__":
    main()
