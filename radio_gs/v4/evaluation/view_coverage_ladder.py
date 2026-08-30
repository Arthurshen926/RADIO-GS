"""Query-independent uniform/greedy source-view coverage ladder."""

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


def _uniform_indices(total: int, count: int) -> list[int]:
    if count >= total:
        return list(range(total))
    return np.linspace(0, total - 1, count, dtype=np.int64).tolist()


def _greedy_indices(visibility: np.ndarray, count: int) -> list[int]:
    """Select views by newly visible surface only; labels never enter selection."""

    count = min(count, visibility.shape[0])
    union = np.zeros(visibility.shape[1], dtype=bool)
    available = np.ones(visibility.shape[0], dtype=bool)
    selected: list[int] = []
    for _ in range(count):
        gains = np.logical_and(visibility, ~union[None]).sum(1)
        gains[~available] = -1
        index = int(gains.argmax())
        selected.append(index)
        available[index] = False
        union |= visibility[index]
    return selected


def _reason_summary(
    visible: torch.Tensor,
    covered: torch.Tensor,
    ground_truth_token: torch.Tensor,
    observed_tokens: torch.Tensor,
) -> dict[str, Any]:
    unknown = ~covered
    annotated = ground_truth_token >= 0
    observed_object = torch.zeros_like(annotated)
    observed_object[annotated] = observed_tokens[ground_truth_token[annotated]]
    reason_e = unknown & ~visible & annotated & observed_object
    reason_a = unknown & ~visible & ~reason_e
    reason_b = unknown & visible & ~covered
    total = int(unknown.sum())
    return {
        "A_never_visible_without_observed_token": {
            "count": int(reason_a.sum()), "fraction": float(reason_a.sum()) / max(total, 1)
        },
        "B_visible_without_object_mask_evidence": {
            "count": int(reason_b.sum()), "fraction": float(reason_b.sum()) / max(total, 1)
        },
        "C_mask_evidence_without_token_association": {"count": 0, "fraction": 0.0},
        "D_associated_but_not_committed": {"count": 0, "fraction": 0.0},
        "E_unseen_surface_of_an_observed_object": {
            "count": int(reason_e.sum()), "fraction": float(reason_e.sum()) / max(total, 1)
        },
    }


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.allow_instance_oracle_diagnostic:
        raise PermissionError("coverage ladder requires explicit instance-oracle authorization")
    transforms = Path(args.transforms).resolve(strict=True)
    mesh_path = Path(args.mesh).resolve(strict=True)
    segmentation_path = Path(args.segmentation).resolve(strict=True)
    aggregation_path = Path(args.aggregation).resolve(strict=True)
    milestone_path = Path(args.milestone_receipt).resolve(strict=True)
    milestone = json.loads(milestone_path.read_text())
    if milestone.get("decisions", {}).get("carrier_parameters_frozen") is not True:
        raise PermissionError("coverage ladder requires a frozen carrier milestone")

    transforms_payload = json.loads(transforms.read_text())
    frame_ids = [int(Path(frame["file_path"]).stem) for frame in transforms_payload["frames"]]
    records = [{"frame_id": value} for value in frame_ids]
    cameras = _load_cameras(transforms, records, args.feature_height, args.feature_width)
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

    visibility = np.zeros((len(cameras), surface.num_elements), dtype=bool)
    mask_coverage = np.zeros_like(visibility)
    token_observed = np.zeros((len(cameras), len(object_ids)), dtype=bool)
    for index, camera in enumerate(cameras):
        surface_projection = surface.project(camera)
        visibility[index, surface_projection.element_ids.numpy()] = True
        oracle_raster = oracle.render_posterior(vertex_membership, camera)
        oracle_support = oracle.project(camera).pixel_weight_sum() > 0
        evidence = surface.lift(
            oracle_raster,
            camera,
            state=torch.where(oracle_support, 1, -1),
        )
        mask_coverage[index] = evidence.mean.sum(-1).numpy() > args.evidence_epsilon
        token_observed[index] = oracle_raster.sum((0, 1)).numpy() > args.evidence_epsilon

    from scipy.spatial import cKDTree
    nearest = cKDTree(vertices.numpy()).query(surface.centres.numpy(), k=1)[1]
    ground_truth_token = vertex_tokens[torch.from_numpy(np.asarray(nearest, dtype=np.int64))]
    annotated = ground_truth_token >= 0
    requested = sorted(set([*args.view_count, len(cameras)]))
    results = {}
    for strategy in ("uniform", "coverage_greedy"):
        strategy_records = []
        for count in requested:
            indices = (
                _uniform_indices(len(cameras), count)
                if strategy == "uniform"
                else _greedy_indices(visibility, count)
            )
            visible = torch.from_numpy(visibility[indices].any(0))
            covered = torch.from_numpy(mask_coverage[indices].any(0))
            observed_tokens = torch.from_numpy(token_observed[indices].any(0))
            strategy_records.append({
                "requested_view_count": count,
                "selected_view_count": len(indices),
                "selected_frame_ids": [frame_ids[index] for index in indices],
                "surface_visibility_union": float(visible.float().mean()),
                "annotated_surface_visibility_union": float(visible[annotated].float().mean()),
                "object_mask_covered_surface_union": float(covered.float().mean()),
                "annotated_object_mask_covered_surface_union": float(covered[annotated].float().mean()),
                "observed_token_count": int(observed_tokens.sum()),
                "unknown_reasons": _reason_summary(
                    visible, covered, ground_truth_token, observed_tokens
                ),
            })
        results[strategy] = strategy_records

    receipt = GeometryReceipt(
        carrier="frozen_sparse_surface_view_coverage_ladder",
        coordinate_convention="mesh_oracle_to_sparse_surface_feature_raster",
        inputs=(
            HashedInput.seal("milestone_receipt", milestone_path),
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
            "selection_uses_instance_labels": False,
            "selection_signal": "newly_visible_surface_elements",
            "available_source_view_count": len(cameras),
        },
    )
    report = {
        "schema": "radio_gs.surface_object_memory_v4.view_coverage_ladder.v1",
        "stage": "pretraining_diagnostic",
        "available_source_view_count": len(cameras),
        "feature_raster": [args.feature_height, args.feature_width],
        "results": results,
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
    parser.add_argument("--feature-height", type=int, default=60)
    parser.add_argument("--feature-width", type=int, default=81)
    parser.add_argument("--view-count", action="append", type=int, default=[])
    parser.add_argument("--voxel-size", type=float, default=0.04)
    parser.add_argument("--maximum-splat-radius", type=int, default=1)
    parser.add_argument("--surface-band-voxels", type=float, default=1.5)
    parser.add_argument("--maximum-contributors-per-pixel", type=int, default=8)
    parser.add_argument("--evidence-epsilon", type=float, default=1e-6)
    parser.add_argument("--allow-instance-oracle-diagnostic", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.view_count:
        args.view_count = [16, 32, 64]
    report = run(args)
    compact = {
        strategy: [
            {
                key: row[key]
                for key in (
                    "selected_view_count",
                    "surface_visibility_union",
                    "object_mask_covered_surface_union",
                    "observed_token_count",
                )
            }
            for row in rows
        ]
        for strategy, rows in report["results"].items()
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
