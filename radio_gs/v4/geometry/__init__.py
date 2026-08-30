"""Geometry estimation, calibration, and sparse-surface fusion."""

from .depth_calibration import AffineDepthCalibration, fit_constrained_affine_depth
from .tsdf_fusion import DepthObservation, SparseSurfaceFusion, SparseSurfaceResult

__all__ = [
    "AffineDepthCalibration",
    "fit_constrained_affine_depth",
    "DepthObservation",
    "SparseSurfaceFusion",
    "SparseSurfaceResult",
]
