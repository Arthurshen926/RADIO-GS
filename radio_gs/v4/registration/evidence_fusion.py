"""Merge per-view evidence while retaining disagreement and uncertainty."""

from __future__ import annotations

import torch

from radio_gs.v4.carrier.base import EvidenceTable


def fuse_evidence_tables(tables: list[EvidenceTable]) -> EvidenceTable:
    if not tables:
        raise ValueError("at least one evidence table is required")
    shape = tables[0].mean.shape
    if any(table.mean.shape != shape for table in tables):
        raise ValueError("evidence tables must share element and channel dimensions")
    weights = torch.stack([table.weight_sum for table in tables])
    means = torch.stack([table.mean for table in tables])
    total = weights.sum(0)
    mean = (means * weights[..., None]).sum(0) / total.clamp_min(1e-12)[..., None]
    dispersion = torch.zeros_like(mean)
    for table in tables:
        dispersion += table.weight_sum[:, None] * (
            table.dispersion + (table.mean - mean).square()
        )
    dispersion /= total.clamp_min(1e-12)[:, None]
    positive = torch.stack([table.positive_weight for table in tables]).sum(0)
    negative = torch.stack([table.negative_weight for table in tables]).sum(0)
    unknown = torch.stack([table.unknown_weight for table in tables]).sum(0)
    disagreement = torch.minimum(positive, negative) / (positive + negative).clamp_min(1e-12)
    residual_numerator = torch.zeros_like(total)
    residual_denominator = torch.zeros_like(total)
    for table in tables:
        valid = torch.isfinite(table.depth_residual)
        residual_numerator += torch.nan_to_num(table.depth_residual) * table.weight_sum * valid
        residual_denominator += table.weight_sum * valid
    residual = residual_numerator / residual_denominator.clamp_min(1e-12)
    residual[residual_denominator == 0] = torch.nan
    return EvidenceTable(
        mean=mean,
        dispersion=dispersion,
        weight_sum=total,
        view_count=torch.stack([table.view_count for table in tables]).sum(0),
        positive_weight=positive,
        negative_weight=negative,
        unknown_weight=unknown,
        depth_residual=residual,
        mask_disagreement=disagreement,
    )
