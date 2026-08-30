"""Backend-independent surface transport contract.

The sparse projection relation is the sole bridge between pixels and carrier
elements.  In particular, downstream code must not infer semantics from an
element index or assume that elements are Gaussians.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import torch


@dataclass(frozen=True)
class Camera:
    """A pinhole camera with a camera-to-world pose."""

    key: str
    intrinsic: torch.Tensor
    camera_to_world: torch.Tensor
    height: int
    width: int

    def __post_init__(self) -> None:
        intrinsic = torch.as_tensor(self.intrinsic, dtype=torch.float64)
        pose = torch.as_tensor(self.camera_to_world, dtype=torch.float64)
        if intrinsic.shape != (3, 3):
            raise ValueError("intrinsic must have shape [3, 3]")
        if pose.shape != (4, 4):
            raise ValueError("camera_to_world must have shape [4, 4]")
        if self.height <= 0 or self.width <= 0:
            raise ValueError("camera raster dimensions must be positive")
        if not torch.isfinite(intrinsic).all() or not torch.isfinite(pose).all():
            raise ValueError("camera values must be finite")
        if not torch.allclose(pose[3], torch.tensor([0, 0, 0, 1], dtype=pose.dtype)):
            raise ValueError("camera_to_world must be homogeneous")
        object.__setattr__(self, "intrinsic", intrinsic)
        object.__setattr__(self, "camera_to_world", pose)


@dataclass(frozen=True)
class ProjectionTable:
    """Sparse visible element-to-pixel relation for one camera."""

    element_ids: torch.Tensor
    pixel_ids: torch.Tensor
    depths: torch.Tensor
    weights: torch.Tensor
    num_elements: int
    height: int
    width: int
    normalization: Literal["sum", "weighted_mean"] = "weighted_mean"
    depth_residuals: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        element_ids = torch.as_tensor(self.element_ids, dtype=torch.long).cpu()
        pixel_ids = torch.as_tensor(self.pixel_ids, dtype=torch.long).cpu()
        depths = torch.as_tensor(self.depths, dtype=torch.float32).cpu()
        weights = torch.as_tensor(self.weights, dtype=torch.float32).cpu()
        size = element_ids.numel()
        if any(value.ndim != 1 or value.numel() != size for value in (pixel_ids, depths, weights)):
            raise ValueError("projection columns must be equally sized vectors")
        if self.num_elements <= 0 or self.height <= 0 or self.width <= 0:
            raise ValueError("projection dimensions must be positive")
        if size:
            if int(element_ids.min()) < 0 or int(element_ids.max()) >= self.num_elements:
                raise ValueError("element id outside carrier domain")
            if int(pixel_ids.min()) < 0 or int(pixel_ids.max()) >= self.height * self.width:
                raise ValueError("pixel id outside raster domain")
            if not torch.isfinite(weights).all() or bool((weights < 0).any()):
                raise ValueError("projection weights must be finite and non-negative")
            finite_depth = torch.isfinite(depths)
            if bool(finite_depth.any()) and bool((depths[finite_depth] <= 0).any()):
                raise ValueError("available projection depths must be positive")
        residuals = self.depth_residuals
        if residuals is not None:
            residuals = torch.as_tensor(residuals, dtype=torch.float32).cpu()
            if residuals.shape != depths.shape:
                raise ValueError("depth residuals must match sparse contributions")
        object.__setattr__(self, "element_ids", element_ids)
        object.__setattr__(self, "pixel_ids", pixel_ids)
        object.__setattr__(self, "depths", depths)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "depth_residuals", residuals)

    @property
    def num_pixels(self) -> int:
        return self.height * self.width

    def pixel_weight_sum(self) -> torch.Tensor:
        result = torch.zeros(self.num_pixels, dtype=torch.float32)
        result.scatter_add_(0, self.pixel_ids, self.weights)
        return result.reshape(self.height, self.width)


@dataclass(frozen=True)
class EvidenceTable:
    """Sufficient statistics retained during confidence-aware registration."""

    mean: torch.Tensor
    dispersion: torch.Tensor
    weight_sum: torch.Tensor
    view_count: torch.Tensor
    positive_weight: torch.Tensor
    negative_weight: torch.Tensor
    unknown_weight: torch.Tensor
    depth_residual: torch.Tensor
    mask_disagreement: torch.Tensor

    @property
    def known_weight(self) -> torch.Tensor:
        return self.positive_weight + self.negative_weight


@dataclass(frozen=True)
class SparseAdjacency:
    """Directed, weighted surface-local edges."""

    edge_index: torch.Tensor
    weights: torch.Tensor
    num_elements: int

    def __post_init__(self) -> None:
        edges = torch.as_tensor(self.edge_index, dtype=torch.long).cpu()
        weights = torch.as_tensor(self.weights, dtype=torch.float32).cpu()
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if weights.shape != (edges.shape[1],):
            raise ValueError("edge weights must have shape [E]")
        if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= self.num_elements):
            raise ValueError("adjacency endpoint outside carrier domain")
        object.__setattr__(self, "edge_index", edges)
        object.__setattr__(self, "weights", weights)


class SurfaceCarrier(ABC):
    """Geometry-only interface consumed by v4 registration and object memory."""

    @property
    @abstractmethod
    def num_elements(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def project(self, camera: Camera) -> ProjectionTable:
        raise NotImplementedError

    @abstractmethod
    def neighbors(self) -> SparseAdjacency:
        raise NotImplementedError

    def lift(
        self,
        image_signal: torch.Tensor,
        camera: Camera,
        *,
        state: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
    ) -> EvidenceTable:
        """Lift one raster without conflating unknown and negative evidence.

        ``state`` uses ``1`` for positive, ``0`` for negative, and ``-1`` for
        unknown.  When omitted, finite pixels are known and binary signal values
        define positive/negative state.
        """

        projection = self.project(camera)
        signal = torch.as_tensor(image_signal, dtype=torch.float32).cpu()
        if signal.shape[:2] != (camera.height, camera.width):
            raise ValueError("image signal does not match camera raster")
        if signal.ndim == 2:
            signal = signal[..., None]
        if signal.ndim != 3:
            raise ValueError("image signal must have shape [H, W] or [H, W, C]")
        flat = signal.reshape(-1, signal.shape[-1])
        finite = torch.isfinite(flat).all(-1)
        if state is None:
            if signal.shape[-1] != 1:
                states = torch.where(finite, 1, -1).to(torch.int8)
            else:
                states = torch.where(
                    finite,
                    (flat[:, 0] > 0.5).to(torch.int8),
                    torch.full_like(finite, -1, dtype=torch.int8),
                )
        else:
            states = torch.as_tensor(state, dtype=torch.int8).cpu().reshape(-1)
            if states.numel() != projection.num_pixels or bool(((states < -1) | (states > 1)).any()):
                raise ValueError("state must contain only -1, 0, and 1 for every pixel")
            states = torch.where(finite, states, torch.full_like(states, -1))
        pixel_confidence = (
            torch.ones(projection.num_pixels, dtype=torch.float32)
            if confidence is None
            else torch.as_tensor(confidence, dtype=torch.float32).cpu().reshape(-1)
        )
        if pixel_confidence.numel() != projection.num_pixels:
            raise ValueError("confidence must have one value per pixel")
        if not torch.isfinite(pixel_confidence).all() or bool((pixel_confidence < 0).any()):
            raise ValueError("confidence must be finite and non-negative")

        ids, pixels = projection.element_ids, projection.pixel_ids
        weights = projection.weights * pixel_confidence[pixels]
        known = states[pixels] >= 0
        known_weights = weights * known.float()
        values = torch.nan_to_num(flat[pixels])
        channels = values.shape[-1]
        weighted_sum = torch.zeros(self.num_elements, channels)
        weighted_sum.index_add_(0, ids, values * known_weights[:, None])
        weight_sum = torch.zeros(self.num_elements)
        weight_sum.scatter_add_(0, ids, known_weights)
        mean = weighted_sum / weight_sum.clamp_min(1e-12)[:, None]
        squared_error = (values - mean[ids]).square() * known_weights[:, None]
        dispersion = torch.zeros_like(mean)
        dispersion.index_add_(0, ids, squared_error)
        dispersion = dispersion / weight_sum.clamp_min(1e-12)[:, None]

        def state_weight(value: int) -> torch.Tensor:
            result = torch.zeros(self.num_elements)
            result.scatter_add_(0, ids, weights * (states[pixels] == value).float())
            return result

        positive = state_weight(1)
        negative = state_weight(0)
        unknown = state_weight(-1)
        view_count = (weight_sum > 0).to(torch.int32)
        if projection.depth_residuals is None:
            depth_residual = torch.full((self.num_elements,), torch.nan)
        else:
            depth_sum = torch.zeros(self.num_elements)
            valid_depth = torch.isfinite(projection.depth_residuals) & known
            depth_weight = weights * valid_depth.float()
            depth_sum.scatter_add_(
                0, ids, torch.nan_to_num(projection.depth_residuals).abs() * depth_weight
            )
            depth_denominator = torch.zeros(self.num_elements)
            depth_denominator.scatter_add_(0, ids, depth_weight)
            depth_residual = depth_sum / depth_denominator.clamp_min(1e-12)
            depth_residual[depth_denominator == 0] = torch.nan
        disagreement_denominator = (positive + negative).clamp_min(1e-12)
        disagreement = torch.minimum(positive, negative) / disagreement_denominator
        return EvidenceTable(
            mean=mean,
            dispersion=dispersion,
            weight_sum=weight_sum,
            view_count=view_count,
            positive_weight=positive,
            negative_weight=negative,
            unknown_weight=unknown,
            depth_residual=depth_residual,
            mask_disagreement=disagreement,
        )

    def render_posterior(self, posterior: torch.Tensor, camera: Camera) -> torch.Tensor:
        """Render the same element posterior without a benchmark readout."""

        projection = self.project(camera)
        values = torch.as_tensor(posterior, dtype=torch.float32).cpu()
        if values.shape[0] != self.num_elements:
            raise ValueError("posterior leading dimension must equal carrier elements")
        scalar = values.ndim == 1
        if scalar:
            values = values[:, None]
        if values.ndim != 2:
            raise ValueError("posterior must have shape [E] or [E, C]")
        output = torch.zeros(projection.num_pixels, values.shape[1])
        output.index_add_(
            0,
            projection.pixel_ids,
            values[projection.element_ids] * projection.weights[:, None],
        )
        if projection.normalization == "weighted_mean":
            denominator = torch.zeros(projection.num_pixels)
            denominator.scatter_add_(0, projection.pixel_ids, projection.weights)
            output = output / denominator.clamp_min(1e-12)[:, None]
        output = output.reshape(camera.height, camera.width, -1)
        return output[..., 0] if scalar else output
