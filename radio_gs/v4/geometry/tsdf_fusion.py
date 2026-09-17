"""Confidence-weighted sparse surface fusion for calibrated depth maps.

This module deliberately does not concatenate per-view point clouds.  It
aggregates observations into shared voxels and checks source-depth agreement
and free-space violations. Within-cell dispersion is diagnostic, not a proof
of physical surface thickness. This is point fusion, not TSDF integration.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import torch

from radio_gs.v4.carrier.base import Camera


@dataclass(frozen=True)
class DepthObservation:
    camera: Camera
    depth: torch.Tensor
    validity: torch.Tensor
    confidence: torch.Tensor
    normals_camera: torch.Tensor | None = None


@dataclass(frozen=True)
class SparseSurfaceResult:
    centres: torch.Tensor
    normals: torch.Tensor
    confidence: torch.Tensor
    view_count: torch.Tensor
    dispersion: torch.Tensor
    voxel_size: float
    agreement_count: torch.Tensor | None = None
    free_space_conflict_count: torch.Tensor | None = None


class SparseSurfaceFusion:
    def __init__(
        self,
        voxel_size: float,
        *,
        minimum_views: int = 2,
        maximum_dispersion_voxels: float = 1.5,
        device: str | torch.device = "cpu",
        consistency_mode: str = "source_depth_v2",
        relative_depth_tolerance: float = 0.01,
        minimum_voxel_views: int | None = None,
    ) -> None:
        if voxel_size <= 0 or minimum_views <= 0 or maximum_dispersion_voxels <= 0:
            raise ValueError("fusion parameters must be positive")
        self.voxel_size = float(voxel_size)
        self.minimum_views = int(minimum_views)
        self.minimum_voxel_views = self.minimum_views if minimum_voxel_views is None else int(minimum_voxel_views)
        if self.minimum_voxel_views <= 0:
            raise ValueError("minimum_voxel_views must be positive")
        if minimum_voxel_views is not None and consistency_mode != "source_depth_v2":
            raise ValueError("separate voxel support requires independent source-depth checks")
        self.maximum_dispersion_voxels = float(maximum_dispersion_voxels)
        self.device = torch.device(device)
        if consistency_mode not in {"source_depth_v2", "legacy_unit_weight_v1"}:
            raise ValueError("unsupported fusion consistency mode")
        if not math.isfinite(relative_depth_tolerance) or relative_depth_tolerance < 0:
            raise ValueError("relative depth tolerance must be finite and nonnegative")
        self.consistency_mode = consistency_mode
        self.relative_depth_tolerance = float(relative_depth_tolerance)

    @staticmethod
    def _backproject(
        observation: DepthObservation,
        device: torch.device = torch.device("cpu"),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        depth = torch.as_tensor(observation.depth, dtype=torch.float32, device=device)
        validity = torch.as_tensor(observation.validity, dtype=torch.bool, device=device)
        confidence = torch.as_tensor(observation.confidence, dtype=torch.float32, device=device)
        expected = (observation.camera.height, observation.camera.width)
        if depth.shape != expected or validity.shape != expected or confidence.shape != expected:
            raise ValueError("depth observation raster differs from camera")
        valid = validity & torch.isfinite(depth) & torch.isfinite(confidence) & (depth > 0) & (confidence > 0)
        y, x = torch.where(valid)
        z = depth[y, x]
        intrinsic = observation.camera.intrinsic.float().to(device)
        points_camera = torch.stack(
            [
                (x.float() + 0.5 - intrinsic[0, 2]) * z / intrinsic[0, 0],
                (y.float() + 0.5 - intrinsic[1, 2]) * z / intrinsic[1, 1],
                z,
            ],
            dim=-1,
        )
        pose = observation.camera.camera_to_world.float().to(device)
        points_world = points_camera @ pose[:3, :3].T + pose[:3, 3]
        if observation.normals_camera is None:
            normals_world = torch.zeros_like(points_world)
        else:
            normals = torch.as_tensor(observation.normals_camera, dtype=torch.float32, device=device)
            if normals.shape != (*expected, 3):
                raise ValueError("normal map must have shape [H, W, 3]")
            normals_world = normals[y, x] @ pose[:3, :3].T
            normals_world = torch.nn.functional.normalize(normals_world, dim=-1, eps=1e-12)
        return points_world, normals_world, confidence[y, x], valid

    def fuse(self, observations: list[DepthObservation]) -> SparseSurfaceResult:
        if not observations:
            raise ValueError("fusion requires at least one depth observation")
        points, normals, weights, view_ids = [], [], [], []
        for view_index, observation in enumerate(observations):
            point, normal, weight, _ = self._backproject(observation, self.device)
            points.append(point)
            normals.append(normal)
            weights.append(weight)
            view_ids.append(torch.full((point.shape[0],), view_index, dtype=torch.long, device=self.device))
        points_tensor = torch.cat(points)
        normals_tensor = torch.cat(normals)
        weights_tensor = torch.cat(weights)
        views_tensor = torch.cat(view_ids)
        if not points_tensor.numel():
            raise RuntimeError("fusion received no valid depth observations")
        keys = torch.floor(points_tensor / self.voxel_size).to(torch.int64)
        # ``torch.unique(..., dim=0)`` compares full rows and is prohibitively
        # slow for multi-million observation sets.  Mixed-radix encoding is a
        # bijection over this batch's finite voxel bounds, so one-dimensional
        # grouping produces exactly the same partition much faster.
        shifted = keys - keys.amin(dim=0)
        spans = shifted.amax(dim=0) + 1
        maximum_int64 = torch.iinfo(torch.int64).max
        if int(spans[0]) > maximum_int64 // max(1, int(spans[1])) or (
            int(spans[0]) * int(spans[1]) > maximum_int64 // max(1, int(spans[2]))
        ):
            unique, inverse = torch.unique(keys, dim=0, return_inverse=True)
            count = unique.shape[0]
        else:
            encoded = (shifted[:, 0] * spans[1] + shifted[:, 1]) * spans[2] + shifted[:, 2]
            unique_encoded, inverse = torch.unique(encoded, return_inverse=True)
            count = unique_encoded.shape[0]
        weight_sum = torch.zeros(count, device=self.device)
        weight_sum.index_add_(0, inverse, weights_tensor)
        centres = torch.zeros(count, 3, device=self.device)
        centres.index_add_(0, inverse, points_tensor * weights_tensor[:, None])
        centres /= weight_sum.clamp_min(1e-12)[:, None]
        residual = (points_tensor - centres[inverse]).square().sum(-1)
        dispersion = torch.zeros(count, device=self.device)
        dispersion.index_add_(0, inverse, residual * weights_tensor)
        dispersion = (dispersion / weight_sum.clamp_min(1e-12)).sqrt()
        fused_normals = torch.zeros(count, 3, device=self.device)
        fused_normals.index_add_(0, inverse, normals_tensor * weights_tensor[:, None])
        fused_normals = torch.nn.functional.normalize(fused_normals, dim=-1, eps=1e-12)
        number_of_views = len(observations)
        encoded_pairs = inverse * number_of_views + views_tensor
        unique_pairs = torch.unique(encoded_pairs)
        view_count = torch.bincount(
            torch.div(unique_pairs, number_of_views, rounding_mode="floor"),
            minlength=count,
        )
        accepted = (
            (view_count >= self.minimum_voxel_views)
            & (dispersion <= self.maximum_dispersion_voxels * self.voxel_size)
        )
        agreement = torch.zeros(count, dtype=torch.long, device=self.device)
        conflict = torch.zeros_like(agreement)
        if self.consistency_mode == "source_depth_v2":
            for observation in observations:
                camera = observation.camera
                pose = torch.linalg.inv(camera.camera_to_world).float().to(self.device)
                intrinsic = camera.intrinsic.float().to(self.device)
                xyz = centres @ pose[:3, :3].T + pose[:3, 3]
                z = xyz[:, 2]
                x = torch.floor(intrinsic[0, 0] * xyz[:, 0] / z.clamp_min(1e-8) + intrinsic[0, 2]).long()
                y = torch.floor(intrinsic[1, 1] * xyz[:, 1] / z.clamp_min(1e-8) + intrinsic[1, 2]).long()
                inside = (z > 0) & (x >= 0) & (x < camera.width) & (y >= 0) & (y < camera.height)
                ids = torch.where(inside)[0]
                depth = observation.depth.to(self.device)[y[ids], x[ids]]
                valid = observation.validity.to(self.device)[y[ids], x[ids]].bool()
                weight = observation.confidence.to(self.device)[y[ids], x[ids]]
                valid &= torch.isfinite(depth) & (depth > 0) & torch.isfinite(weight) & (weight > 0)
                # A voxel centre may differ from its observations by a voxel
                # diagonal. Larger residuals need independent depth evidence.
                tolerance = math.sqrt(3) * self.voxel_size + self.relative_depth_tolerance * depth
                residual = z[ids] - depth
                agreement[ids] += (valid & (residual.abs() <= tolerance)).long()
                # Behind an observed surface is occluded/unknown, not negative.
                conflict[ids] += (valid & (residual < -tolerance)).long()
            accepted &= (agreement >= self.minimum_views) & (agreement > conflict)
        if not bool(accepted.any()):
            raise RuntimeError("fusion rejected every voxel; audit pose, calibration, and scale")
        confidence = weight_sum / torch.bincount(inverse, minlength=count).float().clamp_min(1)
        if self.consistency_mode == "source_depth_v2":
            confidence *= agreement / (agreement + conflict).clamp_min(1)
        return SparseSurfaceResult(
            centres=centres[accepted].cpu(),
            normals=fused_normals[accepted].cpu(),
            confidence=confidence[accepted].cpu(),
            view_count=view_count[accepted].cpu(),
            dispersion=dispersion[accepted].cpu(),
            voxel_size=self.voxel_size,
            agreement_count=agreement[accepted].cpu(),
            free_space_conflict_count=conflict[accepted].cpu(),
        )
