"""Gaussian carrier backed only by precomputed exact renderer transport."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch

from .base import Camera, ProjectionTable, SparseAdjacency, SurfaceCarrier


class GaussianCarrier(SurfaceCarrier):
    """Baseline carrier; exact-MPR is transport, not surface authority."""

    def __init__(
        self,
        num_elements: int,
        projection_paths: Mapping[str, str | Path],
        adjacency: SparseAdjacency | None = None,
    ) -> None:
        if num_elements <= 0:
            raise ValueError("num_elements must be positive")
        self._num_elements = int(num_elements)
        self._projection_paths = {str(key): Path(value) for key, value in projection_paths.items()}
        self._adjacency = adjacency
        self._cache: dict[str, ProjectionTable] = {}

    @property
    def num_elements(self) -> int:
        return self._num_elements

    def project(self, camera: Camera) -> ProjectionTable:
        if camera.key in self._cache:
            return self._cache[camera.key]
        if camera.key not in self._projection_paths:
            raise KeyError(f"no sealed Gaussian projection for camera key {camera.key!r}")
        path = self._projection_paths[camera.key].resolve(strict=True)
        payload = torch.load(path, map_location="cpu")
        required = {"gaussian_ids", "pixel_ids", "base_weights", "num_gaussians", "num_pixels"}
        if not isinstance(payload, dict) or not required.issubset(payload):
            raise ValueError("invalid exact renderer projection payload")
        if int(payload["num_gaussians"]) != self.num_elements:
            raise ValueError("Gaussian projection element count differs from carrier")
        if int(payload["num_pixels"]) != camera.height * camera.width:
            raise ValueError("Gaussian projection raster differs from camera")
        count = torch.as_tensor(payload["pixel_ids"]).numel()
        result = ProjectionTable(
            element_ids=payload["gaussian_ids"],
            pixel_ids=payload["pixel_ids"],
            depths=torch.full((count,), torch.nan),
            weights=payload["base_weights"],
            num_elements=self.num_elements,
            height=camera.height,
            width=camera.width,
            normalization="sum",
            metadata={"backend": "gaussian_exact_renderer_transport", "path": str(path)},
        )
        self._cache[camera.key] = result
        return result

    def neighbors(self) -> SparseAdjacency:
        if self._adjacency is None:
            return SparseAdjacency(torch.empty(2, 0), torch.empty(0), self.num_elements)
        return self._adjacency
