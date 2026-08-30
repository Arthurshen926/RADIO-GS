"""ScanNet mesh-label geometry-registration oracle ladder.

This diagnostic is opt-in because it opens benchmark vertex labels.  The
labels are used only to isolate carrier registration; they are never persisted
as method state and cannot authorize object/query promotion by themselves.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radio_gs.v4.carrier import Camera, GaussianCarrier, MeshCarrier, SurfaceCarrier, SurfaceVoxelCarrier
from radio_gs.v4.contracts.geometry_receipt import GeometryReceipt, HashedInput, sha256_file
from radio_gs.v4.registration.evidence_fusion import fuse_evidence_tables
from radio_gs.v4.registration.surface_projection import (
    boundary_leakage,
    depth_consistency,
    element_purity,
    normal_consistency,
    projection_entropy,
    soft_macro_iou,
)


def _load_mesh(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError("geometry ladder requires open3d") from error
    mesh = o3d.io.read_triangle_mesh(str(path))
    vertices = torch.from_numpy(np.asarray(mesh.vertices).copy()).float()
    triangles = torch.from_numpy(np.asarray(mesh.triangles).copy()).long()
    if not vertices.numel() or not triangles.numel():
        raise ValueError("mesh input is empty")
    return vertices, triangles


def _load_vertex_labels(path: Path) -> torch.Tensor:
    try:
        from plyfile import PlyData
    except ImportError as error:
        raise RuntimeError("geometry ladder requires plyfile") from error
    vertex = PlyData.read(str(path))["vertex"]
    if "label" not in vertex.data.dtype.names:
        raise ValueError("oracle label PLY has no vertex label property")
    return torch.from_numpy(np.asarray(vertex["label"], dtype=np.int64).copy())


def _load_cameras(
    transforms_path: Path,
    records: list[dict[str, Any]],
    height: int,
    width: int,
) -> list[Camera]:
    payload = json.loads(transforms_path.read_text())
    frames = {Path(frame["file_path"]).stem: frame for frame in payload["frames"]}
    scale_x, scale_y = width / int(payload["w"]), height / int(payload["h"])
    intrinsic = torch.tensor(
        [
            [float(payload["fl_x"]) * scale_x, 0, float(payload["cx"]) * scale_x],
            [0, float(payload["fl_y"]) * scale_y, float(payload["cy"]) * scale_y],
            [0, 0, 1],
        ]
    )
    # The prepared transforms retain NeRF/OpenGL axes (+Y up, -Z forward).
    # SurfaceCarrier uses OpenCV axes (+Y down, +Z forward), matching the
    # training dataset and exact-renderer authority.
    gl_to_cv = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float64))
    cameras = []
    for record in records:
        key = str(int(record["frame_id"]))
        if key not in frames:
            raise KeyError(f"source frame {key!r} missing from transforms")
        pose = torch.tensor(frames[key]["transform_matrix"], dtype=torch.float64) @ gl_to_cv
        cameras.append(Camera(key, intrinsic, pose, height, width))
    return cameras


def _oracle_rasters(
    oracle: MeshCarrier,
    labels: torch.Tensor,
    cameras: list[Camera],
) -> tuple[list[torch.Tensor], list[torch.Tensor], int]:
    if labels.shape != (oracle.num_elements,):
        raise ValueError("oracle labels do not match mesh vertices")
    class_count = int(labels.max()) + 1
    one_hot = torch.nn.functional.one_hot(labels, num_classes=class_count).float()
    rasters, supports = [], []
    for camera in cameras:
        raster = oracle.render_posterior(one_hot, camera)
        support = oracle.project(camera).pixel_weight_sum() > 0
        rasters.append(raster)
        supports.append(support)
    return rasters, supports, class_count


def _aggregate(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _evaluate_carrier(
    carrier: SurfaceCarrier,
    oracle: MeshCarrier,
    cameras: list[Camera],
    oracle_rasters: list[torch.Tensor],
    oracle_supports: list[torch.Tensor],
) -> dict[str, Any]:
    label_rasters = [raster.argmax(-1) for raster in oracle_rasters]
    per_view_evidence = [
        carrier.lift(raster, camera, state=torch.where(support, 1, -1))
        for raster, support, camera in zip(oracle_rasters, oracle_supports, cameras)
    ]
    roundtrip, transfer, leakage, purity, coverage = [], [], [], [], []
    entropies, depth_values, normal_values = [], [], []
    for target_index, camera in enumerate(cameras):
        target_labels = label_rasters[target_index]
        oracle_support = oracle_supports[target_index]
        same = carrier.render_posterior(per_view_evidence[target_index].mean, camera)
        roundtrip.append(soft_macro_iou(same, target_labels, oracle_support))
        other = [table for index, table in enumerate(per_view_evidence) if index != target_index]
        fused = fuse_evidence_tables(other)
        rendered = carrier.render_posterior(fused.mean, camera)
        transfer.append(soft_macro_iou(rendered, target_labels, oracle_support))
        leakage.append(boundary_leakage(rendered, target_labels, oracle_support))
        purity.append(element_purity(fused.mean, fused.weight_sum > 0))
        projection = carrier.project(camera)
        carrier_support = projection.pixel_weight_sum() > 0
        coverage.append(float((carrier_support & oracle_support).sum() / oracle_support.sum().clamp_min(1)))
        entropies.append(projection_entropy(projection))
        consistency = depth_consistency(projection, oracle.project(camera))
        if consistency is not None:
            depth_values.append(consistency)
        normal = normal_consistency(carrier, oracle, camera)
        if normal is not None:
            normal_values.append(normal)
    return {
        "same_view_mask_roundtrip_soft_miou": _aggregate(roundtrip),
        "cross_view_mask_transfer_soft_miou": _aggregate(transfer),
        "boundary_leakage": _aggregate(leakage),
        "element_purity": _aggregate(purity),
        "registration_entropy": {
            key: _aggregate([value[key] for value in entropies]) for key in entropies[0]
        },
        "oracle_surface_coverage": _aggregate(coverage),
        "depth_consistency": (
            {
                key: _aggregate([value[key] for value in depth_values])
                for key in depth_values[0]
            }
            if depth_values else None
        ),
        "normal_consistency": (
            {
                key: _aggregate([value[key] for value in normal_values])
                for key in normal_values[0]
            }
            if normal_values else None
        ),
        "per_view": {
            "same_view_mask_roundtrip_soft_miou": roundtrip,
            "cross_view_mask_transfer_soft_miou": transfer,
            "boundary_leakage": leakage,
            "element_purity": purity,
            "oracle_surface_coverage": coverage,
        },
    }


def _receipt(
    carrier: str,
    common_inputs: list[HashedInput],
    extra_inputs: list[HashedInput],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return GeometryReceipt(
        carrier=carrier,
        coordinate_convention="nerf_opengl_to_opencv_camera_to_world_pixel_centres",
        inputs=tuple(common_inputs + extra_inputs),
        source_rgb_opened=False,
        target_rgb_opened=False,
        benchmark_images_opened=False,
        benchmark_masks_opened=False,
        benchmark_labels_opened=True,
        metadata={"oracle_geometry_diagnostic": True, **metadata},
    ).to_dict()


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.allow_benchmark_label_diagnostic:
        raise PermissionError("mesh-label geometry diagnostic requires explicit authorization")
    transforms = Path(args.transforms).resolve(strict=True)
    mesh_path = Path(args.mesh).resolve(strict=True)
    label_path = Path(args.vertex_labels).resolve(strict=True)
    authority_path = Path(args.source_authority).resolve(strict=True)
    authority = torch.load(authority_path, map_location="cpu")
    metadata = authority.get("metadata", {})
    if any(metadata.get(key) is not False for key in (
        "benchmark_images_opened", "benchmark_masks_opened", "evaluation_rgb_opened", "text_queries_opened"
    )):
        raise ValueError("source authority is not sealed from benchmark/query inputs")
    records = list(metadata.get("source_records", []))
    if len(records) < 2:
        raise ValueError("geometry ladder requires at least two sealed source views")
    height, width = int(metadata["feature_height"]), int(metadata["feature_width"])
    cameras = _load_cameras(transforms, records, height, width)
    vertices, triangles = _load_mesh(mesh_path)
    labels = _load_vertex_labels(label_path)
    oracle = MeshCarrier(vertices, triangles)
    oracle_rasters, oracle_supports, class_count = _oracle_rasters(oracle, labels, cameras)
    gaussian = GaussianCarrier(
        int(authority["num_rows"]),
        {str(int(record["frame_id"])): record["responsibility_view"] for record in records},
    )
    sparse_surface = SurfaceVoxelCarrier.from_points(
        vertices,
        args.voxel_size,
        normals=oracle.normals,
        maximum_splat_radius=args.maximum_splat_radius,
        surface_band_voxels=args.surface_band_voxels,
        maximum_contributors_per_pixel=args.maximum_contributors_per_pixel,
    )
    carriers: dict[str, SurfaceCarrier] = {
        "gaussian_exact_renderer": gaussian,
        "mesh_oracle": oracle,
        "mesh_derived_sparse_surface": sparse_surface,
    }
    results = {
        name: _evaluate_carrier(carrier, oracle, cameras, oracle_rasters, oracle_supports)
        for name, carrier in carriers.items()
    }
    common = [
        HashedInput.seal("camera_transforms", transforms),
        HashedInput.seal("source_authority", authority_path),
        HashedInput.seal("mesh_geometry", mesh_path),
        HashedInput.seal("diagnostic_vertex_labels", label_path),
    ]
    mpr_inputs = [
        HashedInput.seal(f"source_projection_{index}", record["responsibility_view"])
        for index, record in enumerate(records)
    ]
    receipts = {
        "gaussian_exact_renderer": _receipt("gaussian_exact_renderer", common, mpr_inputs, {}),
        "mesh_oracle": _receipt("mesh_oracle", common, [], {}),
        "mesh_derived_sparse_surface": _receipt(
            "mesh_derived_sparse_surface",
            common,
            [],
            {
                "voxel_size": args.voxel_size,
                "maximum_splat_radius": args.maximum_splat_radius,
                "surface_band_voxels": args.surface_band_voxels,
                "maximum_contributors_per_pixel": args.maximum_contributors_per_pixel,
            },
        ),
    }
    baseline = results["gaussian_exact_renderer"]
    comparison = {}
    for name in ("mesh_oracle", "mesh_derived_sparse_surface"):
        candidate = results[name]
        comparison[name] = {
            "roundtrip_delta": candidate["same_view_mask_roundtrip_soft_miou"] - baseline["same_view_mask_roundtrip_soft_miou"],
            "transfer_delta": candidate["cross_view_mask_transfer_soft_miou"] - baseline["cross_view_mask_transfer_soft_miou"],
            "boundary_leakage_reduction": baseline["boundary_leakage"] - candidate["boundary_leakage"],
            "purity_delta": candidate["element_purity"] - baseline["element_purity"],
            "entropy_nats_reduction": baseline["registration_entropy"]["mean_entropy_nats"] - candidate["registration_entropy"]["mean_entropy_nats"],
            "effective_contributors_reduction": baseline["registration_entropy"]["effective_contributors"] - candidate["registration_entropy"]["effective_contributors"],
            "coverage_delta": candidate["oracle_surface_coverage"] - baseline["oracle_surface_coverage"],
        }
    report = {
        "schema": "radio_gs.surface_object_memory_v4.geometry_registration_ladder.v1",
        "stage": "geometry_registration_only",
        "scene_label": args.scene_label,
        "source_view_count": len(cameras),
        "raster_shape": [height, width],
        "class_count_including_unlabeled": class_count,
        "oracle_label_use": "explicit_geometry_diagnostic_only_not_method_state",
        "results": results,
        "comparison_to_gaussian": comparison,
        "geometry_receipts": receipts,
        "downstream_gates": {
            "object_codebook_started": False,
            "query_encoder_started": False,
            "compression_started": False,
        },
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transforms", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--vertex-labels", required=True)
    parser.add_argument("--source-authority", required=True)
    parser.add_argument("--scene-label", required=True)
    parser.add_argument("--voxel-size", type=float, required=True)
    parser.add_argument("--maximum-splat-radius", type=int, default=3)
    parser.add_argument("--surface-band-voxels", type=float, default=1.5)
    parser.add_argument("--maximum-contributors-per-pixel", type=int, default=8)
    parser.add_argument("--allow-benchmark-label-diagnostic", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if (
        args.voxel_size <= 0
        or args.maximum_splat_radius < 0
        or args.surface_band_voxels < 0
        or args.maximum_contributors_per_pixel <= 0
    ):
        parser.error("voxel size must be positive and splat radius non-negative")
    report = run(args)
    print(json.dumps(report["comparison_to_gaussian"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
