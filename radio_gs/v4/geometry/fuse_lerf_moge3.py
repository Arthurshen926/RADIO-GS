"""Source-only COLMAP calibration and sparse-surface fusion for LERF MoGe-3."""

from __future__ import annotations

import argparse
import json
import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from radio_gs.data.lerf_dataset import (
    _camera_params_to_intrinsics,
    _qvec_to_rotmat,
    _read_cameras_binary,
)
from radio_gs.v4.carrier import Camera
from radio_gs.v4.contracts.geometry_receipt import GeometryReceipt, HashedInput, sha256_file
from radio_gs.v4.geometry.depth_calibration import fit_constrained_affine_depth
from radio_gs.v4.geometry.tsdf_fusion import DepthObservation, SparseSurfaceFusion


@dataclass(frozen=True)
class ColmapSourceView:
    frame_index: int
    image_id: int
    camera_id: int
    width: int
    height: int
    intrinsic: torch.Tensor
    world_to_camera: torch.Tensor
    camera_to_world: torch.Tensor
    xy: torch.Tensor
    point_ids: torch.Tensor


def _frame_index(name: str) -> int:
    values = re.findall(r"\d+", Path(name).stem)
    if not values:
        raise ValueError(f"COLMAP image name has no frame index: {name}")
    return int(values[-1])


def _read_images(path: Path, cameras: dict[int, dict[str, object]]) -> dict[int, ColmapSourceView]:
    output: dict[int, ColmapSourceView] = {}
    with path.open("rb") as handle:
        (count,) = struct.unpack("<Q", handle.read(8))
        for _ in range(count):
            (image_id,) = struct.unpack("<i", handle.read(4))
            qvec = np.asarray(struct.unpack("<4d", handle.read(32)), dtype=np.float64)
            tvec = np.asarray(struct.unpack("<3d", handle.read(24)), dtype=np.float64)
            (camera_id,) = struct.unpack("<i", handle.read(4))
            name_bytes = bytearray()
            while (value := handle.read(1)) != b"\x00":
                if not value:
                    raise EOFError("truncated COLMAP image name")
                name_bytes.extend(value)
            (point_count,) = struct.unpack("<Q", handle.read(8))
            xy = np.empty((point_count, 2), dtype=np.float64)
            point_ids = np.empty(point_count, dtype=np.uint64)
            for index in range(point_count):
                x, y, point_id = struct.unpack("<ddQ", handle.read(24))
                xy[index] = (x, y)
                point_ids[index] = point_id
            camera_record = cameras[int(camera_id)]
            fx, fy, cx, cy = _camera_params_to_intrinsics(
                int(camera_record["model_id"]), np.asarray(camera_record["params"])
            )
            intrinsic = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
            world_to_camera = np.eye(4, dtype=np.float64)
            world_to_camera[:3, :3] = _qvec_to_rotmat(qvec).astype(np.float64)
            world_to_camera[:3, 3] = tvec
            pose = np.linalg.inv(world_to_camera)
            frame_index = _frame_index(name_bytes.decode("utf-8"))
            output[frame_index] = ColmapSourceView(
                frame_index=frame_index,
                image_id=int(image_id),
                camera_id=int(camera_id),
                width=int(camera_record["width"]),
                height=int(camera_record["height"]),
                intrinsic=intrinsic,
                world_to_camera=torch.from_numpy(world_to_camera).float(),
                camera_to_world=torch.from_numpy(pose).float(),
                xy=torch.from_numpy(xy).float(),
                # COLMAP encodes an unmatched point as UINT64_MAX.  Mapping
                # through int64 turns only that sentinel into -1, which can
                # never collide with a valid positive point3D id.
                point_ids=torch.from_numpy(point_ids.astype(np.int64, copy=False)),
            )
    return output


def _read_strict_source_points(
    path: Path, source_image_ids: set[int]
) -> tuple[dict[int, torch.Tensor], dict[str, int]]:
    points: dict[int, torch.Tensor] = {}
    touched_source = 0
    with path.open("rb") as handle:
        (count,) = struct.unpack("<Q", handle.read(8))
        for _ in range(count):
            point_id, x, y, z = struct.unpack("<Qddd", handle.read(32))
            handle.read(3)
            handle.read(8)
            (track_count,) = struct.unpack("<Q", handle.read(8))
            track_ids = set()
            for _track_index in range(track_count):
                image_id, _point2d_index = struct.unpack("<ii", handle.read(8))
                track_ids.add(int(image_id))
            if track_ids & source_image_ids:
                touched_source += 1
            if track_ids and track_ids <= source_image_ids:
                points[int(point_id)] = torch.tensor([x, y, z], dtype=torch.float32)
    return points, {
        "total_colmap_points": int(count),
        "points_touching_source_views": touched_source,
        "strict_source_only_points": len(points),
    }


def _bilinear(depth: torch.Tensor, xy: torch.Tensor, source_width: int, source_height: int) -> torch.Tensor:
    height, width = depth.shape
    scaled = xy.clone()
    scaled[:, 0] *= width / source_width
    scaled[:, 1] *= height / source_height
    grid = torch.empty(1, 1, scaled.shape[0], 2)
    grid[0, 0, :, 0] = (scaled[:, 0] + 0.5) * 2 / width - 1
    grid[0, 0, :, 1] = (scaled[:, 1] + 0.5) * 2 / height - 1
    return F.grid_sample(
        depth[None, None].float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )[0, 0, 0]


def _scaled_camera(view: ColmapSourceView, height: int, width: int, stride: int) -> Camera:
    full = view.intrinsic.clone()
    full[0] *= width / view.width
    full[1] *= height / view.height
    sampled_height = len(range(0, height, stride))
    sampled_width = len(range(0, width, stride))
    intrinsic = full.clone()
    intrinsic[0, 0] /= stride
    intrinsic[1, 1] /= stride
    intrinsic[0, 2] = (full[0, 2] + 0.5) / stride - 0.5
    intrinsic[1, 2] = (full[1, 2] + 0.5) / stride - 0.5
    return Camera(str(view.frame_index), intrinsic, view.camera_to_world, sampled_height, sampled_width)


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, object]:
    scene_root = Path(args.scene_root).resolve(strict=True)
    prediction_dir = Path(args.prediction_dir).resolve(strict=True)
    scene_manifest_path = Path(args.scene_manifest).resolve(strict=True)
    scene_manifest = json.loads(scene_manifest_path.read_text())
    authority_path = Path(scene_manifest["source_authority"]).resolve(strict=True)
    authority = json.loads(authority_path.read_text())
    frame_indices = [int(value) for value in authority["frame_indices"]]
    sparse = scene_root / "sparse" / "0"
    cameras = _read_cameras_binary(sparse / "cameras.bin")
    all_views = _read_images(sparse / "images.bin", cameras)
    missing = sorted(set(frame_indices) - set(all_views))
    if missing:
        raise KeyError(f"source frames missing from COLMAP model: {missing[:8]}")
    views = {index: all_views[index] for index in frame_indices}
    source_image_ids = {view.image_id for view in views.values()}
    strict_points, point_filter = _read_strict_source_points(sparse / "points3D.bin", source_image_ids)

    predictions: dict[int, dict[str, object]] = {}
    correspondence: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    view_scale_ratios = []
    for frame_index, view in views.items():
        prediction_path = prediction_dir / f"frame_{frame_index:05d}.pt"
        payload = torch.load(prediction_path, map_location="cpu")
        if payload["model_checkpoint_sha256"] != scene_manifest["model_checkpoint_sha256"]:
            raise ValueError("prediction checkpoint digest differs from scene manifest")
        predictions[frame_index] = payload
        retained = [index for index, value in enumerate(view.point_ids.tolist()) if int(value) in strict_points]
        if not retained:
            continue
        retained_tensor = torch.tensor(retained, dtype=torch.long)
        xyz_world = torch.stack([strict_points[int(value)] for value in view.point_ids[retained_tensor].tolist()])
        xyz_camera = xyz_world @ view.world_to_camera[:3, :3].T + view.world_to_camera[:3, 3]
        reference = xyz_camera[:, 2]
        depth = torch.as_tensor(payload["point_map"])[..., 2].float()
        predicted = _bilinear(depth, view.xy[retained_tensor], view.width, view.height)
        valid = torch.isfinite(predicted) & torch.isfinite(reference) & (predicted > 0) & (reference > 0)
        predicted, reference = predicted[valid], reference[valid]
        if predicted.numel() >= 3:
            correspondence[frame_index] = (predicted, reference)
            view_scale_ratios.append((reference / predicted).median())
    if len(view_scale_ratios) < 3:
        raise RuntimeError("too few strictly source-only views for global metric/COLMAP scale")
    global_scale = float(torch.stack(view_scale_ratios).median())

    observations: list[DepthObservation] = []
    calibration_records = []
    for frame_index, view in views.items():
        local_scale, local_offset = 1.0, 0.0
        if frame_index in correspondence:
            predicted_samples, reference_samples = correspondence[frame_index]
            calibration = fit_constrained_affine_depth(
                predicted_samples * global_scale,
                reference_samples,
                maximum_scale_deviation=args.maximum_local_scale_deviation,
                maximum_offset_fraction=args.maximum_offset_fraction,
            )
            calibration_records.append({"frame_index": frame_index, **asdict(calibration)})
            if calibration.accepted:
                local_scale, local_offset = calibration.scale, calibration.offset
        else:
            calibration_records.append(
                {
                    "frame_index": frame_index,
                    "scale": 1.0,
                    "offset": 0.0,
                    "sample_count": 0,
                    "median_absolute_residual": None,
                    "accepted": False,
                    "rejection_reason": "too_few_correspondences",
                }
            )
        payload = predictions[frame_index]
        point_map = torch.as_tensor(payload["point_map"]).float()
        depth = (point_map[..., 2] * global_scale * local_scale + local_offset).float()
        validity = torch.as_tensor(payload["validity"], dtype=torch.bool) & torch.isfinite(depth) & (depth > 0)
        normals = torch.as_tensor(payload["normals"]).float()
        height, width = depth.shape
        camera = _scaled_camera(view, height, width, args.pixel_stride)
        observations.append(
            DepthObservation(
                camera=camera,
                depth=depth[:: args.pixel_stride, :: args.pixel_stride],
                validity=validity[:: args.pixel_stride, :: args.pixel_stride],
                confidence=validity[:: args.pixel_stride, :: args.pixel_stride].float(),
                normals_camera=normals[:: args.pixel_stride, :: args.pixel_stride],
            )
        )
    accepted_count = sum(bool(record["accepted"]) for record in calibration_records)
    if args.minimum_accepted_views > 0 and accepted_count < args.minimum_accepted_views:
        raise RuntimeError(f"only {accepted_count} views passed bounded calibration")
    voxel_size = global_scale * args.voxel_size_metres
    fused = SparseSurfaceFusion(
        voxel_size,
        minimum_views=args.minimum_fusion_views,
        maximum_dispersion_voxels=args.maximum_dispersion_voxels,
        device=args.fusion_device,
    ).fuse(observations)
    carrier_path = Path(args.output_carrier).resolve()
    carrier_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "radio_gs.surface_object_memory_v4.calibrated_sparse_surface.v1",
            "scene_label": args.scene_label,
            "centres": fused.centres,
            "normals": fused.normals,
            "confidence": fused.confidence,
            "view_count": fused.view_count,
            "dispersion": fused.dispersion,
            "voxel_size_colmap": fused.voxel_size,
            "voxel_size_metres": args.voxel_size_metres,
            "global_colmap_units_per_metre": global_scale,
        },
        carrier_path,
    )
    receipt = GeometryReceipt(
        carrier="calibrated_moge3_sparse_surface",
        coordinate_convention="colmap_world_opencv_camera_pixel_centres",
        inputs=(
            HashedInput.seal("moge3_scene_manifest", scene_manifest_path),
            HashedInput.seal("source_authority", authority_path),
            HashedInput.seal("colmap_cameras", sparse / "cameras.bin"),
            HashedInput.seal("colmap_images", sparse / "images.bin"),
            HashedInput.seal("colmap_sparse_points", sparse / "points3D.bin"),
            HashedInput.seal("fused_surface", carrier_path),
        ),
        source_rgb_opened=True,
        target_rgb_opened=False,
        benchmark_images_opened=False,
        benchmark_masks_opened=False,
        benchmark_labels_opened=False,
        model_family="MoGe-3",
        model_checkpoint_sha256=str(scene_manifest["model_checkpoint_sha256"]),
        calibration={
            "global_colmap_units_per_metre": global_scale,
            "accepted_view_count": accepted_count,
            "rejected_view_count": len(calibration_records) - accepted_count,
            "global_scale_fallback_view_count": len(calibration_records) - accepted_count,
            "maximum_local_scale_deviation": args.maximum_local_scale_deviation,
            "maximum_offset_fraction": args.maximum_offset_fraction,
            "track_filter": "retain only points whose complete track is a subset of sealed source image ids",
            **point_filter,
        },
        metadata={
            "scene_label": args.scene_label,
            "voxel_size_metres": args.voxel_size_metres,
            "voxel_size_colmap": voxel_size,
            "pixel_stride": args.pixel_stride,
            "minimum_fusion_views": args.minimum_fusion_views,
            "maximum_dispersion_voxels": args.maximum_dispersion_voxels,
            "fusion_device": args.fusion_device,
            "surface_element_count": int(fused.centres.shape[0]),
        },
    )
    report = {
        "schema": "radio_gs.surface_object_memory_v4.lerf_moge3_fusion_report.v1",
        "scene_label": args.scene_label,
        "global_colmap_units_per_metre": global_scale,
        "calibration": calibration_records,
        "accepted_view_count": accepted_count,
        "rejected_view_count": len(calibration_records) - accepted_count,
        "global_scale_fallback_view_count": len(calibration_records) - accepted_count,
        "point_filter": point_filter,
        "surface_element_count": int(fused.centres.shape[0]),
        "voxel_size_colmap": voxel_size,
        "view_count": {
            "minimum": int(fused.view_count.min()),
            "median": float(fused.view_count.float().median()),
            "maximum": int(fused.view_count.max()),
        },
        "dispersion_metres": {
            "median": float(fused.dispersion.median() / global_scale),
            "maximum": float(fused.dispersion.max() / global_scale),
        },
        "geometry_receipt": receipt.to_dict(),
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    report_path = Path(args.output_report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-root", required=True)
    parser.add_argument("--scene-label", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--scene-manifest", required=True)
    parser.add_argument("--output-carrier", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--voxel-size-metres", type=float, default=0.04)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--minimum-fusion-views", type=int, default=2)
    parser.add_argument("--fusion-device", default="cpu")
    parser.add_argument("--maximum-dispersion-voxels", type=float, default=1.5)
    parser.add_argument("--maximum-local-scale-deviation", type=float, default=0.2)
    parser.add_argument("--maximum-offset-fraction", type=float, default=0.1)
    parser.add_argument(
        "--minimum-accepted-views",
        type=int,
        default=0,
        help="optional diagnostic floor; local acceptance does not gate global-scale fallback by default",
    )
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({key: report[key] for key in ("scene_label", "accepted_view_count", "rejected_view_count", "surface_element_count", "global_colmap_units_per_metre")}, indent=2))


if __name__ == "__main__":
    main()
