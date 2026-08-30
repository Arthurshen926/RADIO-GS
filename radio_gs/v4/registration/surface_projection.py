"""Geometry-registration metrics independent of carrier backend."""

from __future__ import annotations

import math

import torch

from radio_gs.v4.carrier.base import ProjectionTable


def soft_macro_iou(probabilities: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor) -> float:
    probabilities = torch.as_tensor(probabilities, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)
    valid = torch.as_tensor(valid, dtype=torch.bool)
    values = []
    for class_id in torch.unique(labels[valid]):
        if int(class_id) == 0:
            continue
        target = (labels == class_id) & valid
        prediction = probabilities[..., int(class_id)] * valid.float()
        intersection = prediction[target].sum()
        union = prediction.sum() + target.sum() - intersection
        if union > 0:
            values.append(intersection / union)
    return float(torch.stack(values).mean()) if values else float("nan")


def boundary_leakage(probabilities: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor) -> float:
    values = []
    for class_id in torch.unique(labels[valid]):
        if int(class_id) == 0:
            continue
        target = (labels == class_id) & valid
        prediction = probabilities[..., int(class_id)] * valid.float()
        denominator = prediction.sum()
        if denominator > 0:
            values.append(prediction[~target].sum() / denominator)
    return float(torch.stack(values).mean()) if values else float("nan")


def element_purity(class_evidence: torch.Tensor, observed: torch.Tensor) -> float:
    evidence = torch.as_tensor(class_evidence, dtype=torch.float32)
    observed = torch.as_tensor(observed, dtype=torch.bool)
    foreground = evidence[:, 1:]
    denominator = foreground.sum(-1)
    selected = observed & (denominator > 0)
    if not bool(selected.any()):
        return float("nan")
    return float((foreground.max(-1).values[selected] / denominator[selected]).mean())


def projection_entropy(projection: ProjectionTable) -> dict[str, float]:
    """Normalized contribution entropy and effective contributors per hit pixel."""

    if projection.weights.numel() == 0:
        return {"normalized_entropy": float("nan"), "effective_contributors": float("nan")}
    denominator = torch.zeros(projection.num_pixels)
    denominator.scatter_add_(0, projection.pixel_ids, projection.weights)
    normalized = projection.weights / denominator[projection.pixel_ids].clamp_min(1e-12)
    terms = -normalized * normalized.clamp_min(1e-12).log()
    entropy = torch.zeros(projection.num_pixels)
    entropy.scatter_add_(0, projection.pixel_ids, terms)
    counts = torch.bincount(projection.pixel_ids, minlength=projection.num_pixels)
    hit = counts > 0
    normalizer = counts.float().clamp_min(2).log()
    normalized_entropy = entropy / normalizer
    return {
        "mean_entropy_nats": float(entropy[hit].mean()),
        "normalized_entropy": float(normalized_entropy[hit].mean()),
        "effective_contributors": float(entropy[hit].exp().mean()),
        "mean_contributors": float(counts[hit].float().mean()),
    }


def depth_consistency(candidate: ProjectionTable, oracle: ProjectionTable) -> dict[str, float] | None:
    def first_depth(table: ProjectionTable) -> torch.Tensor:
        result = torch.full((table.num_pixels,), torch.inf)
        if table.depths.numel() and bool(torch.isfinite(table.depths).any()):
            result.scatter_reduce_(0, table.pixel_ids, table.depths, reduce="amin", include_self=True)
        return result

    candidate_depth, oracle_depth = first_depth(candidate), first_depth(oracle)
    valid = torch.isfinite(candidate_depth) & torch.isfinite(oracle_depth)
    if not bool(valid.any()):
        return None
    residual = (candidate_depth[valid] - oracle_depth[valid]).abs()
    return {
        "mean_absolute_depth_residual": float(residual.mean()),
        "median_absolute_depth_residual": float(residual.median()),
        "overlap_pixel_count": int(valid.sum()),
    }


def normal_consistency(candidate, oracle, camera) -> dict[str, float] | None:
    candidate_normals = getattr(candidate, "normals", None)
    oracle_normals = getattr(oracle, "normals", None)
    if candidate_normals is None or oracle_normals is None:
        return None
    candidate_raster = candidate.render_posterior(candidate_normals, camera)
    oracle_raster = oracle.render_posterior(oracle_normals, camera)
    candidate_length = candidate_raster.norm(dim=-1)
    oracle_length = oracle_raster.norm(dim=-1)
    valid = (candidate_length > 1e-6) & (oracle_length > 1e-6)
    if not bool(valid.any()):
        return None
    first = torch.nn.functional.normalize(candidate_raster[valid], dim=-1)
    second = torch.nn.functional.normalize(oracle_raster[valid], dim=-1)
    # Mesh windings can differ while describing the same local tangent plane.
    cosine = (first * second).sum(-1).abs().clamp(0, 1)
    return {
        "mean_unsigned_cosine": float(cosine.mean()),
        "median_angular_error_degrees": float(torch.rad2deg(torch.acos(cosine)).median()),
        "overlap_pixel_count": int(valid.sum()),
    }
