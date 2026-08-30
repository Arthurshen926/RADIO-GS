"""Reorder a sealed source cohort to balance new coverage and association overlap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from radio_gs.v4.carrier import MeshCarrier, SurfaceVoxelCarrier
from radio_gs.v4.contracts.geometry_receipt import GeometryReceipt, HashedInput, sha256_file
from radio_gs.v4.evaluation.geometry_ladder import _load_cameras, _load_mesh


def _association_order(
    visibility: np.ndarray,
    *,
    target_overlap: float,
    coverage_floor: float,
) -> list[int]:
    remaining = set(range(visibility.shape[0]))
    first = max(remaining, key=lambda index: (int(visibility[index].sum()), -index))
    order = [first]
    remaining.remove(first)
    covered = visibility[first].copy()
    while remaining:
        ranked = []
        for index in remaining:
            visible_count = max(int(visibility[index].sum()), 1)
            overlap = float((visibility[index] & covered).sum()) / visible_count
            new_count = int((visibility[index] & ~covered).sum())
            association_factor = coverage_floor + (1.0 - coverage_floor) * min(
                overlap / target_overlap, 1.0
            )
            ranked.append((new_count * association_factor, overlap, new_count, -index, index))
        selected = max(ranked)[-1]
        order.append(selected)
        remaining.remove(selected)
        covered |= visibility[selected]
    return order


@torch.no_grad()
def run(args: argparse.Namespace) -> dict:
    selection_path = Path(args.selection_authority).resolve(strict=True)
    transforms = Path(args.transforms).resolve(strict=True)
    mesh_path = Path(args.mesh).resolve(strict=True)
    selection = json.loads(selection_path.read_text())
    cohort = list(map(int, selection["selections"][str(args.cohort_size)]["selected_frame_ids"]))
    cameras = _load_cameras(
        transforms, [{"frame_id": value} for value in cohort], args.feature_height, args.feature_width
    )
    vertices, triangles = _load_mesh(mesh_path)
    mesh = MeshCarrier(vertices, triangles)
    surface = SurfaceVoxelCarrier.from_points(
        vertices,
        args.voxel_size,
        normals=mesh.normals,
        maximum_splat_radius=args.maximum_splat_radius,
        surface_band_voxels=args.surface_band_voxels,
        maximum_contributors_per_pixel=args.maximum_contributors_per_pixel,
    )
    visibility = np.zeros((len(cameras), surface.num_elements), dtype=bool)
    for index, camera in enumerate(cameras):
        projection = surface.project(camera)
        visibility[index, projection.element_ids.numpy()] = True
    order = _association_order(
        visibility, target_overlap=args.target_overlap, coverage_floor=args.coverage_floor
    )
    ordered_visibility = visibility[order]
    ordered_frames = [cohort[index] for index in order]
    selections = {}
    for count in sorted(set(args.view_count)):
        prefix = ordered_visibility[:count]
        covered = np.zeros(surface.num_elements, dtype=bool)
        overlaps = []
        for row in prefix:
            overlaps.append(float((row & covered).sum()) / max(int(row.sum()), 1))
            covered |= row
        selections[str(count)] = {
            "selected_frame_ids": ordered_frames[:count],
            "surface_visibility_union": float(covered.mean()),
            "mean_predecessor_overlap_after_first": float(np.mean(overlaps[1:])) if count > 1 else 0.0,
        }
    receipt = GeometryReceipt(
        carrier="frozen_sparse_surface_association_aware_source_reordering",
        coordinate_convention="feature_raster_surface_visibility",
        inputs=(
            HashedInput.seal("label_free_coverage_selection", selection_path),
            HashedInput.seal("camera_transforms", transforms),
            HashedInput.seal("mesh_geometry", mesh_path),
        ),
        source_rgb_opened=False,
        target_rgb_opened=False,
        benchmark_images_opened=False,
        benchmark_masks_opened=False,
        benchmark_labels_opened=False,
        metadata={
            "cohort_unchanged": True,
            "query_independent": True,
            "instance_or_semantic_labels_used": False,
            "score": "new_surface_coverage_times_saturated_predecessor_overlap_factor",
        },
    )
    report = {
        "schema": "radio_gs.surface_object_memory_v4.association_aware_source_order.v1",
        "cohort_size": args.cohort_size,
        "target_overlap": args.target_overlap,
        "coverage_floor": args.coverage_floor,
        "selections": selections,
        "geometry_receipt": receipt.to_dict(),
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-authority", required=True)
    parser.add_argument("--transforms", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--cohort-size", type=int, default=64)
    parser.add_argument("--view-count", action="append", type=int, required=True)
    parser.add_argument("--target-overlap", type=float, default=0.40)
    parser.add_argument("--coverage-floor", type=float, default=0.25)
    parser.add_argument("--feature-height", type=int, default=60)
    parser.add_argument("--feature-width", type=int, default=81)
    parser.add_argument("--voxel-size", type=float, default=0.04)
    parser.add_argument("--maximum-splat-radius", type=int, default=1)
    parser.add_argument("--surface-band-voxels", type=float, default=1.5)
    parser.add_argument("--maximum-contributors-per-pixel", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report["selections"], indent=2))


if __name__ == "__main__":
    main()
