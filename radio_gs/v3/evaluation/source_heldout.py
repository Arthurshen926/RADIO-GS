"""Threshold-fixed source-heldout metrics for the first v3 gate."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SourceHeldoutMetrics:
    mask_iou: float
    brier: float
    boundary_f: float
    unknown_fp_mass: float


def evaluate_source_heldout(
    probability: torch.Tensor,
    target: torch.Tensor,
    known: torch.Tensor,
    unknown: torch.Tensor,
    boundary_probability: torch.Tensor,
    boundary_target: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> SourceHeldoutMetrics:
    score = torch.as_tensor(probability).float().reshape(-1)
    truth = torch.as_tensor(target).bool().reshape(-1)
    authority = torch.as_tensor(known).bool().reshape(-1)
    abstain = torch.as_tensor(unknown).bool().reshape(-1)
    edge_score = torch.as_tensor(boundary_probability).float().reshape(-1)
    edge_truth = torch.as_tensor(boundary_target).bool().reshape(-1)
    if not (score.shape == truth.shape == authority.shape == abstain.shape == edge_score.shape == edge_truth.shape):
        raise ValueError("source-heldout metric axes differ")
    if not 0 < threshold < 1 or not bool(authority.any()):
        raise ValueError("source-heldout threshold or authority differs")
    predicted = score >= threshold
    intersection = (predicted & truth & authority).sum().float()
    union = ((predicted | truth) & authority).sum().float()
    iou = intersection / union.clamp_min(1)
    brier = (score[authority] - truth[authority].float()).square().mean()
    edge = edge_score >= threshold
    tp = (edge & edge_truth & authority).sum().float()
    precision = tp / (edge & authority).sum().clamp_min(1)
    recall = tp / (edge_truth & authority).sum().clamp_min(1)
    boundary_f = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    unknown_fp = score[abstain].mean() if bool(abstain.any()) else score.new_zeros(())
    return SourceHeldoutMetrics(float(iou), float(brier), float(boundary_f), float(unknown_fp))


__all__ = ["SourceHeldoutMetrics", "evaluate_source_heldout"]
