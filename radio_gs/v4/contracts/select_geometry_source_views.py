"""Create a label-free source-view authority by greedy surface coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from radio_gs.v4.carrier import MeshCarrier, SurfaceVoxelCarrier
from radio_gs.v4.contracts.geometry_receipt import GeometryReceipt, HashedInput, sha256_file
from radio_gs.v4.evaluation.geometry_ladder import _load_cameras, _load_mesh
from radio_gs.v4.evaluation.view_coverage_ladder import _greedy_indices


@torch.no_grad()
def run(args: argparse.Namespace) -> dict:
    transforms = Path(args.transforms).resolve(strict=True)
    mesh_path = Path(args.mesh).resolve(strict=True)
    milestone_path = Path(args.milestone_receipt).resolve(strict=True)
    payload = json.loads(transforms.read_text())
    frame_ids = [int(Path(frame["file_path"]).stem) for frame in payload["frames"]]
    cameras = _load_cameras(
        transforms,
        [{"frame_id": value} for value in frame_ids],
        args.feature_height,
        args.feature_width,
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
    selections = {}
    for count in sorted(set(args.view_count)):
        indices = _greedy_indices(visibility, count)
        selections[str(count)] = {
            "selected_frame_ids": [frame_ids[index] for index in indices],
            "surface_visibility_union": float(visibility[indices].any(0).mean()),
        }
    receipt = GeometryReceipt(
        carrier="frozen_sparse_surface_label_free_view_selection",
        coordinate_convention="surface_visibility_at_feature_raster",
        inputs=(
            HashedInput.seal("milestone_receipt", milestone_path),
            HashedInput.seal("camera_transforms", transforms),
            HashedInput.seal("mesh_geometry", mesh_path),
        ),
        source_rgb_opened=False,
        target_rgb_opened=False,
        benchmark_images_opened=False,
        benchmark_masks_opened=False,
        benchmark_labels_opened=False,
        metadata={
            "selection_signal": "newly_visible_surface_elements",
            "query_independent": True,
            "instance_or_semantic_labels_used": False,
        },
    )
    report = {
        "schema": "radio_gs.surface_object_memory_v4.geometry_source_view_selection.v1",
        "available_frame_count": len(frame_ids),
        "feature_raster": [args.feature_height, args.feature_width],
        "selection_strategy": "greedy_newly_visible_surface",
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
    parser.add_argument("--milestone-receipt", required=True)
    parser.add_argument("--transforms", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--view-count", action="append", type=int, required=True)
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
