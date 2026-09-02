"""Deterministic sparse surface-voxel/surfel carrier."""

from __future__ import annotations

import hashlib
import itertools
import operator

import numpy as np
import torch

from .base import Camera, ProjectionTable, SparseAdjacency, SurfaceCarrier


def _exact_integer(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer >= {minimum}") from error
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(result)


class SurfaceVoxelCarrier(SurfaceCarrier):
    def __init__(
        self,
        centres: torch.Tensor,
        voxel_size: float,
        *,
        normals: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
        maximum_splat_radius: int,
        surface_band_voxels: float,
        maximum_contributors_per_pixel: int,
    ) -> None:
        centres = torch.as_tensor(centres, dtype=torch.float32).cpu()
        if centres.ndim != 2 or centres.shape[1] != 3 or centres.shape[0] == 0:
            raise ValueError("centres must have shape [E, 3]")
        if not torch.isfinite(centres).all():
            raise ValueError("centres must be finite")
        radius = _exact_integer(
            maximum_splat_radius, name="maximum_splat_radius", minimum=0
        )
        contributor_cap = _exact_integer(
            maximum_contributors_per_pixel,
            name="maximum_contributors_per_pixel",
            minimum=1,
        )
        if (
            not np.isfinite(voxel_size)
            or not np.isfinite(surface_band_voxels)
            or voxel_size <= 0
            or surface_band_voxels < 0
        ):
            raise ValueError("voxel_size and surface_band_voxels must be finite and non-negative")
        self.centres = centres
        self.voxel_size = float(voxel_size)
        self.maximum_splat_radius = radius
        self.surface_band_voxels = float(surface_band_voxels)
        self.maximum_contributors_per_pixel = contributor_cap
        self.normals = None if normals is None else torch.as_tensor(normals, dtype=torch.float32).cpu()
        if self.normals is not None and self.normals.shape != centres.shape:
            raise ValueError("normals must match centres")
        if self.normals is not None and not torch.isfinite(self.normals).all():
            raise ValueError("normals must be finite")
        self.confidence = (
            torch.ones(centres.shape[0])
            if confidence is None
            else torch.as_tensor(confidence, dtype=torch.float32).cpu()
        )
        if self.confidence.shape != (centres.shape[0],):
            raise ValueError("confidence must have shape [E]")
        if not torch.isfinite(self.confidence).all() or bool((self.confidence < 0).any()):
            raise ValueError("confidence must be finite and non-negative")
        self._adjacency: SparseAdjacency | None = None
        self._projection_cache: dict[str, ProjectionTable] = {}

    @classmethod
    def from_points(
        cls,
        points: torch.Tensor,
        voxel_size: float,
        *,
        normals: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
        maximum_splat_radius: int,
        surface_band_voxels: float,
        maximum_contributors_per_pixel: int,
    ) -> "SurfaceVoxelCarrier":
        points = torch.as_tensor(points, dtype=torch.float32).cpu()
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            raise ValueError("points must have shape [N, 3]")
        if not torch.isfinite(points).all():
            raise ValueError("points must be finite")
        if not np.isfinite(voxel_size) or voxel_size <= 0:
            raise ValueError("voxel_size must be finite and positive")
        radius = _exact_integer(
            maximum_splat_radius, name="maximum_splat_radius", minimum=0
        )
        contributor_cap = _exact_integer(
            maximum_contributors_per_pixel,
            name="maximum_contributors_per_pixel",
            minimum=1,
        )
        keys = torch.floor(points / voxel_size).to(torch.int64)
        unique, inverse = torch.unique(keys, dim=0, return_inverse=True)
        count = torch.bincount(inverse, minlength=unique.shape[0]).float()
        centres = torch.zeros(unique.shape[0], 3)
        centres.index_add_(0, inverse, points)
        centres /= count[:, None]
        reduced_normals = None
        if normals is not None:
            normals = torch.as_tensor(normals, dtype=torch.float32).cpu()
            if normals.shape != points.shape or not torch.isfinite(normals).all():
                raise ValueError("normals must be finite and match points")
            reduced_normals = torch.zeros_like(centres)
            reduced_normals.index_add_(0, inverse, normals)
            reduced_normals = torch.nn.functional.normalize(reduced_normals, dim=-1, eps=1e-12)
        reduced_confidence = torch.ones(unique.shape[0])
        if confidence is not None:
            confidence = torch.as_tensor(confidence, dtype=torch.float32).cpu()
            if (
                confidence.shape != (points.shape[0],)
                or not torch.isfinite(confidence).all()
                or bool((confidence < 0).any())
            ):
                raise ValueError("confidence must be finite, non-negative, and match points")
            reduced_confidence = torch.zeros(unique.shape[0])
            reduced_confidence.index_add_(0, inverse, confidence)
            reduced_confidence /= count
        return cls(
            centres,
            voxel_size,
            normals=reduced_normals,
            confidence=reduced_confidence,
            maximum_splat_radius=radius,
            surface_band_voxels=surface_band_voxels,
            maximum_contributors_per_pixel=contributor_cap,
        )

    @property
    def num_elements(self) -> int:
        return int(self.centres.shape[0])

    @staticmethod
    def _projection_cache_key(camera: Camera) -> str:
        """Bind cached projections to the complete camera, never only its label."""

        digest = hashlib.sha256()
        digest.update(camera.key.encode("utf-8"))
        digest.update(camera.intrinsic.detach().cpu().numpy().astype(np.float64, copy=False).tobytes())
        digest.update(
            camera.camera_to_world.detach().cpu().numpy().astype(np.float64, copy=False).tobytes()
        )
        digest.update(np.asarray([camera.height, camera.width], dtype=np.int64).tobytes())
        return digest.hexdigest()

    def project(self, camera: Camera) -> ProjectionTable:
        cache_key = self._projection_cache_key(camera)
        if cache_key in self._projection_cache:
            return self._projection_cache[cache_key]
        world_to_camera = torch.linalg.inv(camera.camera_to_world).float()
        homogeneous = torch.cat([self.centres, torch.ones(self.num_elements, 1)], dim=-1)
        camera_points = homogeneous @ world_to_camera.T
        z = camera_points[:, 2]
        valid = z > 1e-6
        intrinsic = camera.intrinsic.float()
        u = intrinsic[0, 0] * camera_points[:, 0] / z.clamp_min(1e-6) + intrinsic[0, 2]
        v = intrinsic[1, 1] * camera_points[:, 1] / z.clamp_min(1e-6) + intrinsic[1, 2]
        radius = torch.ceil(
            0.5 * self.voxel_size * torch.maximum(intrinsic[0, 0], intrinsic[1, 1])
            / z.clamp_min(1e-6)
        ).long()
        radius = radius.clamp(0, self.maximum_splat_radius)
        candidate_elements, candidate_pixels, candidate_depths = [], [], []
        for offset_y, offset_x in itertools.product(
            range(-self.maximum_splat_radius, self.maximum_splat_radius + 1), repeat=2
        ):
            inside_radius = (offset_x * offset_x + offset_y * offset_y) <= radius.square()
            x = torch.round(u).long() + offset_x
            y = torch.round(v).long() + offset_y
            selected = valid & inside_radius & (x >= 0) & (x < camera.width) & (y >= 0) & (y < camera.height)
            ids = torch.where(selected)[0]
            if ids.numel():
                candidate_elements.append(ids)
                candidate_pixels.append(y[ids] * camera.width + x[ids])
                candidate_depths.append(z[ids])
        if not candidate_elements:
            result = ProjectionTable(
                torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long),
                torch.empty(0), torch.empty(0), self.num_elements, camera.height, camera.width,
                metadata={"backend": "sparse_surface_voxel_zbuffer"},
            )
            self._projection_cache[cache_key] = result
            return result
        elements = torch.cat(candidate_elements).numpy()
        pixels = torch.cat(candidate_pixels).numpy()
        depths = torch.cat(candidate_depths).numpy()
        order = np.lexsort((elements, depths, pixels))
        sorted_pixels = pixels[order]
        sorted_depths = depths[order]
        first = np.ones(order.size, dtype=bool)
        first[1:] = sorted_pixels[1:] != sorted_pixels[:-1]
        group_starts = np.maximum.accumulate(np.where(first, np.arange(order.size), 0))
        group_minimum_depth = sorted_depths[group_starts]
        rank_in_pixel = np.arange(order.size) - group_starts
        in_surface_band = sorted_depths <= (
            group_minimum_depth + self.surface_band_voxels * self.voxel_size
        )
        retained = in_surface_band & (rank_in_pixel < self.maximum_contributors_per_pixel)
        chosen = order[retained]
        minimum_depth_by_choice = group_minimum_depth[retained]
        depth_weight = np.exp(
            -(depths[chosen] - minimum_depth_by_choice) / max(self.voxel_size, 1e-12)
        ).astype(np.float32)
        chosen_elements = torch.from_numpy(elements[chosen].copy())
        result = ProjectionTable(
            element_ids=chosen_elements,
            pixel_ids=torch.from_numpy(pixels[chosen].copy()),
            depths=torch.from_numpy(depths[chosen].copy()),
            weights=self.confidence[chosen_elements].clamp_min(1e-8) * torch.from_numpy(depth_weight),
            num_elements=self.num_elements,
            height=camera.height,
            width=camera.width,
            normalization="weighted_mean",
            metadata={
                "backend": "sparse_surface_voxel_zbuffer",
                "voxel_size": self.voxel_size,
                "maximum_splat_radius": self.maximum_splat_radius,
                "surface_band_voxels": self.surface_band_voxels,
                "maximum_contributors_per_pixel": self.maximum_contributors_per_pixel,
                "visible_pixel_count": int(np.unique(pixels[chosen]).size),
            },
        )
        self._projection_cache[cache_key] = result
        return result

    def neighbors(self) -> SparseAdjacency:
        if self._adjacency is None:
            keys = torch.floor(self.centres / self.voxel_size).to(torch.int64).numpy()
            lookup = {tuple(key.tolist()): index for index, key in enumerate(keys)}
            edges: list[tuple[int, int]] = []
            for index, key in enumerate(keys):
                for axis in range(3):
                    neighbor = key.copy()
                    neighbor[axis] += 1
                    other = lookup.get(tuple(neighbor.tolist()))
                    if other is not None:
                        edges.extend(((index, other), (other, index)))
            edge_index = (
                torch.tensor(edges, dtype=torch.long).T
                if edges else torch.empty(2, 0, dtype=torch.long)
            )
            self._adjacency = SparseAdjacency(
                edge_index, torch.ones(edge_index.shape[1]), self.num_elements
            )
        return self._adjacency
