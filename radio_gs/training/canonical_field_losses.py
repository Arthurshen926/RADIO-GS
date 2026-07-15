"""Minimal losses for a canonical RADIO field and frozen official views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from radio_gs.field.canonical_gaussian_field import CanonicalGaussianField
from radio_gs.interfaces.frozen_radio_views import FrozenRadioViews
from .primitive_consensus import PrimitiveConsensus, primitive_reconstruction_loss


@dataclass(frozen=True)
class CanonicalFieldLossConfig:
    mpr_weight: float = 1.0
    dino_weight: float = 0.20
    sam3_weight: float = 0.20
    relation_weight: float = 0.05
    coefficient_weight: float = 1e-5
    basis_orthogonality_weight: float = 1e-3

    def __post_init__(self) -> None:
        if min(
            self.mpr_weight,
            self.dino_weight,
            self.sam3_weight,
            self.relation_weight,
            self.coefficient_weight,
            self.basis_orthogonality_weight,
        ) < 0:
            raise ValueError("loss weights cannot be negative")


def _cosine_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(predicted.float(), target.float(), dim=-1)).mean()


def _relation_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    pair_index: torch.Tensor | None,
) -> torch.Tensor:
    if pair_index is None or pair_index.numel() == 0:
        return predicted.sum() * 0.0
    pairs = torch.as_tensor(pair_index, device=predicted.device).long()
    if pairs.ndim != 2 or pairs.shape[0] != 2:
        raise ValueError("pair_index must be [2,E]")
    pred = F.normalize(predicted.float(), dim=-1, eps=1e-8)
    teacher = F.normalize(target.float(), dim=-1, eps=1e-8)
    pred_relation = (pred[pairs[0]] * pred[pairs[1]]).sum(dim=-1)
    target_relation = (teacher[pairs[0]] * teacher[pairs[1]]).sum(dim=-1)
    return F.smooth_l1_loss(pred_relation, target_relation)


def _capability_consensus_loss(
    projected: torch.Tensor,
    consensus: PrimitiveConsensus,
    rows_cpu: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match an official view to features fused *after* per-view projection."""

    target = consensus.targets[rows_cpu].to(projected.device).float()
    valid = consensus.valid[rows_cpu].to(projected.device)
    if projected.ndim != 2 or target.shape != projected.shape:
        raise ValueError(
            "projected capability rows do not align with auxiliary MPR targets"
        )
    if not bool(valid.any()):
        zero = projected.sum() * 0.0
        return zero, target, valid
    reliability = consensus.reliability[rows_cpu, :2].mean(dim=-1).to(
        projected.device
    )
    weights = reliability[valid].clamp_min(1e-4)
    errors = 1.0 - F.cosine_similarity(
        projected[valid].float(), target[valid], dim=-1, eps=1e-8
    )
    loss = (errors * weights).sum() / weights.sum().clamp_min(1e-8)
    return loss, target, valid


def canonical_primitive_loss(
    field: CanonicalGaussianField,
    consensus: PrimitiveConsensus,
    row_indices: torch.Tensor,
    *,
    official_views: FrozenRadioViews | None = None,
    capability_targets: Mapping[str, PrimitiveConsensus] | None = None,
    pair_index: torch.Tensor | None = None,
    config: CanonicalFieldLossConfig = CanonicalFieldLossConfig(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reconstruct RADIO first; preserve official DINO/SAM capability second."""

    rows = torch.as_tensor(row_indices, device=field.local_codes.device).long()
    predicted_radio = field.radio_features(rows)
    mpr, stats = primitive_reconstruction_loss(
        predicted_radio, consensus, row_indices=rows.detach().cpu()
    )
    target_radio = consensus.targets[rows.detach().cpu()].to(predicted_radio.device).float()
    valid = consensus.valid[rows.detach().cpu()].to(predicted_radio.device)
    zero = predicted_radio.sum() * 0.0
    dino = zero
    sam3 = zero
    relation = zero
    targets = dict(capability_targets or {})
    unknown_targets = sorted(set(targets) - {"dino_v3", "sam3"})
    if unknown_targets:
        raise ValueError(f"unsupported capability target spaces: {unknown_targets}")
    if targets and official_views is None:
        raise ValueError("capability targets require frozen official RADIO views")
    if official_views is not None and bool(valid.any()):
        rows_cpu = rows.detach().cpu()
        projected_dino = official_views.project_dino_primitives(predicted_radio)
        projected_sam3 = official_views.project_sam3_primitives(predicted_radio)
        if "dino_v3" in targets:
            dino, target_dino_all, dino_valid = _capability_consensus_loss(
                projected_dino, targets["dino_v3"], rows_cpu
            )
        else:
            dino_valid = valid
            with torch.no_grad():
                target_dino_all = official_views.project_dino_primitives(target_radio)
            dino = _cosine_loss(
                projected_dino[dino_valid], target_dino_all[dino_valid]
            )
        if "sam3" in targets:
            sam3, target_sam3_all, sam3_valid = _capability_consensus_loss(
                projected_sam3, targets["sam3"], rows_cpu
            )
        else:
            sam3_valid = valid
            with torch.no_grad():
                target_sam3_all = official_views.project_sam3_primitives(target_radio)
            sam3 = _cosine_loss(
                projected_sam3[sam3_valid], target_sam3_all[sam3_valid]
            )
        if pair_index is not None and torch.equal(dino_valid, sam3_valid):
            pairs = torch.as_tensor(pair_index, device=predicted_radio.device).long()
            if pairs.ndim != 2 or pairs.shape[0] != 2:
                raise ValueError("pair_index must be [2,E]")
            keep = dino_valid[pairs[0]] & dino_valid[pairs[1]]
            remap = dino_valid.long().cumsum(dim=0) - 1
            pairs = remap[pairs[:, keep]]
            relation = 0.5 * (
                _relation_loss(
                    projected_dino[dino_valid], target_dino_all[dino_valid], pairs
                )
                + _relation_loss(
                    projected_sam3[sam3_valid], target_sam3_all[sam3_valid], pairs
                )
            )
    coefficients = field.coefficients(rows)
    compact = coefficients.square().mean()
    orthogonality = field.decoder.orthogonality_loss()
    total = (
        config.mpr_weight * mpr
        + config.dino_weight * dino
        + config.sam3_weight * sam3
        + config.relation_weight * relation
        + config.coefficient_weight * compact
        + config.basis_orthogonality_weight * orthogonality
    )
    stats.update(
        {
            "dino": dino.detach(),
            "sam3": sam3.detach(),
            "relation": relation.detach(),
            "compact": compact.detach(),
            "orthogonality": orthogonality.detach(),
            "loss": total.detach(),
        }
    )
    return total, stats


def normalized_render_reconstruction_loss(
    rendered_radio: torch.Tensor,
    teacher_radio: torch.Tensor,
    alpha: torch.Tensor,
    *,
    alpha_threshold: float = 0.02,
    cosine_weight: float = 1.0,
    huber_weight: float = 0.25,
) -> torch.Tensor:
    """Raw-view supervision after alpha-normalized coefficient rendering."""

    if rendered_radio.shape != teacher_radio.shape or rendered_radio.ndim != 4:
        raise ValueError("rendered/teacher RADIO maps must align as [B,C,H,W]")
    opacity = torch.as_tensor(alpha, device=rendered_radio.device).float()
    if opacity.ndim == 4 and opacity.shape[1] == 1:
        opacity = opacity[:, 0]
    valid = opacity >= alpha_threshold
    predicted = rendered_radio.permute(0, 2, 3, 1)[valid]
    target = teacher_radio.to(rendered_radio).permute(0, 2, 3, 1)[valid]
    if predicted.numel() == 0:
        return rendered_radio.sum() * 0.0
    cosine = _cosine_loss(predicted, target)
    huber = F.huber_loss(predicted.float(), target.float(), delta=0.1)
    return cosine_weight * cosine + huber_weight * huber
