"""Geometry-bearing scene-level object token records."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .sparse_assignment import SparseObjectAssignments


@dataclass(frozen=True)
class ObjectCodebook:
    centres: torch.Tensor
    scales: torch.Tensor
    confidence: torch.Tensor
    assignments: SparseObjectAssignments

    def __post_init__(self) -> None:
        centres = torch.as_tensor(self.centres, dtype=torch.float32).cpu()
        scales = torch.as_tensor(self.scales, dtype=torch.float32).cpu()
        confidence = torch.as_tensor(self.confidence, dtype=torch.float32).cpu()
        expected = (self.assignments.num_tokens, 3)
        if centres.shape != expected or scales.shape != expected:
            raise ValueError("token centres and scales must have shape [K, 3]")
        if confidence.shape != (self.assignments.num_tokens,):
            raise ValueError("token confidence must have shape [K]")
        if not torch.isfinite(centres).all() or not torch.isfinite(scales).all():
            raise ValueError("token geometry must be finite")
        if bool((scales < 0).any()) or bool((confidence < 0).any()) or bool((confidence > 1).any()):
            raise ValueError("token scale/confidence outside valid range")
        object.__setattr__(self, "centres", centres)
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "confidence", confidence)

    @classmethod
    def from_assignments(
        cls,
        element_centres: torch.Tensor,
        assignments: SparseObjectAssignments,
    ) -> "ObjectCodebook":
        points = torch.as_tensor(element_centres, dtype=torch.float32).cpu()
        if points.shape != (assignments.num_elements, 3):
            raise ValueError("element centres must align with assignments")
        dense = assignments.to_dense()
        mass = dense.sum(0)
        centres = dense.T @ points / mass.clamp_min(1e-12)[:, None]
        delta = points[:, None, :] - centres[None, :, :]
        scales = torch.sqrt((dense[..., None] * delta.square()).sum(0) / mass.clamp_min(1e-12)[:, None])
        confidence = mass / (mass + assignments.unknown_weight.sum().clamp_min(1e-12))
        return cls(centres, scales, confidence, assignments)
