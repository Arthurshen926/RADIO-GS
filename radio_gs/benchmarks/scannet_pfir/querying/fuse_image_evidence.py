"""Frozen, label-free image unary fusion."""

from __future__ import annotations

from typing import Mapping

import numpy as np


def _robust_unit_interval(values: np.ndarray) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = np.isfinite(scores)
    output = np.zeros_like(scores)
    if not bool(finite.any()):
        return output
    lo, hi = np.quantile(scores[finite], [0.05, 0.95])
    output[finite] = np.clip((scores[finite] - lo) / max(float(hi - lo), 1e-8), 0, 1)
    return output


def fuse_image_evidence(
    scores: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
) -> np.ndarray:
    """Fuse declared modalities; weights must be frozen on PFIR-dev."""

    active = [(name, float(weight)) for name, weight in weights.items() if weight > 0]
    if not active:
        raise ValueError("at least one positive modality weight is required")
    missing = [name for name, _ in active if name not in scores]
    if missing:
        raise KeyError(f"missing image evidence: {missing}")
    shapes = {np.asarray(scores[name]).reshape(-1).shape for name, _ in active}
    if len(shapes) != 1:
        raise ValueError("all image evidence rows must align")
    total = sum(weight for _, weight in active)
    fused = sum(
        weight * _robust_unit_interval(scores[name]) for name, weight in active
    )
    return (fused / total).astype(np.float32)

