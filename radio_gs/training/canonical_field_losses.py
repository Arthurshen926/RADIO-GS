"""Minimal losses for a canonical RADIO field and frozen official views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from radio_gs.field.canonical_gaussian_field import CanonicalGaussianField
from radio_gs.interfaces.frozen_radio_views import FrozenRadioViews
from .factorized_radio_cache import FactorizedRadioTrainingCache
from .factorized_radio_loss import (
    FACTORIZED_RADIO_RELIABILITY_POLICY_LEGACY,
    FACTORIZED_RADIO_RELIABILITY_POLICY_MATCHED_EXACT_MARGINAL_VISIBILITY_SAFE,
    factorized_radio_reconstruction_loss,
    uniform_half_confidence,
)
from .primitive_consensus import (
    PrimitiveConsensus,
    consensus_target_rows,
    primitive_reconstruction_loss,
)


@dataclass(frozen=True)
class CanonicalFieldLossConfig:
    mpr_weight: float = 1.0
    dino_weight: float = 0.20
    sam3_weight: float = 0.20
    relation_weight: float = 0.05
    coefficient_weight: float = 1e-5
    basis_orthogonality_weight: float = 1e-3

    def __post_init__(self) -> None:
        if (
            min(
                self.mpr_weight,
                self.dino_weight,
                self.sam3_weight,
                self.relation_weight,
                self.coefficient_weight,
                self.basis_orthogonality_weight,
            )
            < 0
        ):
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


def hard_boundary_relation_ranking_loss(
    predicted_dino: torch.Tensor,
    predicted_sam3: torch.Tensor,
    pair_index: torch.Tensor,
    teacher_margin: torch.Tensor,
) -> torch.Tensor:
    """Preserve exact-teacher positive/negative relation ordering.

    ``pair_index`` is ``[2,2T]``: the first T columns are anchor-positive
    pairs and the second T columns are the aligned anchor-negative pairs. The
    margin is the detached, dimensionless exact-teacher relation gap stored in
    the query-independent Field-B cache; there is no tuned scalar margin.
    """

    pairs = torch.as_tensor(pair_index, device=predicted_dino.device).long()
    margin = (
        torch.as_tensor(teacher_margin, device=predicted_dino.device)
        .float()
        .reshape(-1)
    )
    if pairs.ndim != 2 or pairs.shape[0] != 2 or pairs.shape[1] != 2 * margin.numel():
        raise ValueError("Field-B pair_index must be [2,2T] aligned with T margins")
    if margin.numel() == 0:
        return predicted_dino.sum() * 0.0
    if (
        predicted_dino.ndim != 2
        or predicted_sam3.ndim != 2
        or (predicted_dino.shape[0] != predicted_sam3.shape[0])
    ):
        raise ValueError("Field-B predicted DINO/SAM rows must align")
    if bool((pairs < 0).any()) or int(pairs.max()) >= predicted_dino.shape[0]:
        raise ValueError("Field-B pair_index is outside predicted rows")
    if not bool(torch.isfinite(margin).all()) or bool(
        ((margin < 0) | (margin > 1)).any()
    ):
        raise ValueError("Field-B teacher margins must be finite in [0,1]")

    def relation(values: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(values.float(), dim=-1, eps=1e-8)
        cosine = (normalized[pairs[0]] * normalized[pairs[1]]).sum(dim=-1)
        return (0.5 * (1.0 + cosine)).clamp(0.0, 1.0)

    dino_relation = relation(predicted_dino)
    sam_relation = relation(predicted_sam3)
    combined = torch.sqrt((dino_relation * sam_relation).clamp_min(1e-12))
    count = margin.numel()
    positive = combined[:count]
    negative = combined[count:]
    return F.relu(margin + negative - positive).mean()


def _capability_consensus_loss(
    projected: torch.Tensor,
    consensus: PrimitiveConsensus | Any,
    rows_cpu: torch.Tensor,
    *,
    reliability_policy: str = "legacy_mean",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match an official view to features fused *after* per-view projection."""

    target = consensus_target_rows(consensus, rows_cpu).to(projected.device).float()
    valid = consensus.valid[rows_cpu].to(projected.device)
    if projected.ndim != 2 or target.shape != projected.shape:
        raise ValueError(
            "projected capability rows do not align with auxiliary MPR targets"
        )
    if not bool(valid.any()):
        zero = projected.sum() * 0.0
        return zero, target, valid
    reliability = consensus.reliability[rows_cpu].to(projected.device).float()
    if reliability.ndim != 2 or reliability.shape[1] < 2:
        raise ValueError("capability reliability must provide coverage and agreement")
    if not bool(torch.isfinite(reliability).all()) or bool(
        ((reliability[:, :2] < 0) | (reliability[:, :2] > 1)).any()
    ):
        raise ValueError("capability coverage/agreement must be finite in [0,1]")
    coverage = reliability[:, 0]
    agreement = reliability[:, 1]
    if reliability_policy == "legacy_mean":
        weights_all = 0.5 * (coverage + agreement)
    elif reliability_policy == "field_a_boundary_safe":
        # A low mean-resultant agreement can identify a real occlusion or
        # object boundary, not merely a noisy observation.  Keep an equal
        # uniform-valid component so such rows retain at least half weight;
        # the other half rewards jointly well-covered, view-consistent rows.
        # This fixed mixture has no benchmark- or query-selected parameter.
        reliable = (coverage * agreement).clamp_min(0.0).sqrt()
        weights_all = uniform_half_confidence(reliable)
    elif reliability_policy in {
        "field_c_visibility_safe",
        FACTORIZED_RADIO_RELIABILITY_POLICY_MATCHED_EXACT_MARGINAL_VISIBILITY_SAFE,
    }:
        if reliability.shape[1] < 3:
            raise ValueError("Field-C reliability requires visibility purity")
        purity = reliability[:, 2]
        if not bool(torch.isfinite(purity).all()) or bool(
            ((purity < 0) | (purity > 1)).any()
        ):
            raise ValueError("Field-C visibility purity must be finite in [0,1]")
        # The geometric mean treats coverage, directional agreement, and
        # compositor purity as independent precision evidence.  The fixed
        # uniform half keeps real boundary rows trainable instead of erasing
        # them simply because their observation distribution is ambiguous.
        reliable = (coverage * agreement * purity).clamp_min(0.0).pow(1.0 / 3.0)
        weights_all = uniform_half_confidence(reliable)
    else:
        raise ValueError(
            "capability reliability policy must be legacy_mean or "
            "field_a_boundary_safe or field_c_visibility_safe or "
            f"{FACTORIZED_RADIO_RELIABILITY_POLICY_MATCHED_EXACT_MARGINAL_VISIBILITY_SAFE}"
        )
    weights = weights_all[valid].clamp_min(1e-4)
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
    capability_targets: Mapping[str, PrimitiveConsensus | Any] | None = None,
    pair_index: torch.Tensor | None = None,
    factorized_target: FactorizedRadioTrainingCache | None = None,
    factorized_reliability_policy: str = (
        FACTORIZED_RADIO_RELIABILITY_POLICY_LEGACY
    ),
    capability_reliability_policy: str = "legacy_mean",
    config: CanonicalFieldLossConfig = CanonicalFieldLossConfig(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reconstruct RADIO first; preserve official DINO/SAM capability second."""

    rows = torch.as_tensor(row_indices, device=field.local_codes.device).long()
    predicted_radio = field.radio_features(rows)
    rows_cpu = rows.detach().cpu()
    if factorized_target is None:
        mpr, stats = primitive_reconstruction_loss(
            predicted_radio, consensus, row_indices=rows_cpu
        )
        target_radio = (
            consensus_target_rows(consensus, rows_cpu)
            .to(predicted_radio.device)
            .float()
        )
        valid = consensus.valid[rows_cpu].to(predicted_radio.device)
    else:
        if factorized_target.shape != (
            int(consensus.targets.shape[0]),
            int(consensus.targets.shape[1]),
        ) or not torch.equal(factorized_target.valid, consensus.valid):
            raise ValueError("factorized RADIO target and consensus support differ")
        target_radio = (
            factorized_target.canonical_feature[rows_cpu]
            .to(predicted_radio.device)
            .float()
        )
        valid = factorized_target.valid[rows_cpu].to(predicted_radio.device)
        factorized = factorized_radio_reconstruction_loss(
            predicted_radio,
            target_radio,
            factorized_target.log_amplitude[rows_cpu].to(predicted_radio.device),
            valid,
            factorized_target.reliability[rows_cpu].to(predicted_radio.device),
            reliability_scalar_names=factorized_target.reliability_scalar_names,
            reliability_scalar_names_digest=(
                factorized_target.reliability_scalar_names_sha256
            ),
            reliability_policy=factorized_reliability_policy,
        )
        mpr = factorized.total
        stats = {
            "factorized_direction": factorized.direction.detach(),
            "factorized_log_amplitude": factorized.log_amplitude.detach(),
            "valid_ratio": valid.float().mean().detach(),
        }
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
        projected_dino = official_views.project_dino_primitives(predicted_radio)
        projected_sam3 = official_views.project_sam3_primitives(predicted_radio)
        if "dino_v3" in targets:
            dino, target_dino_all, dino_valid = _capability_consensus_loss(
                projected_dino,
                targets["dino_v3"],
                rows_cpu,
                reliability_policy=capability_reliability_policy,
            )
        else:
            dino_valid = valid
            with torch.no_grad():
                target_dino_all = official_views.project_dino_primitives(target_radio)
            dino = _cosine_loss(projected_dino[dino_valid], target_dino_all[dino_valid])
        if "sam3" in targets:
            sam3, target_sam3_all, sam3_valid = _capability_consensus_loss(
                projected_sam3,
                targets["sam3"],
                rows_cpu,
                reliability_policy=capability_reliability_policy,
            )
        else:
            sam3_valid = valid
            with torch.no_grad():
                target_sam3_all = official_views.project_sam3_primitives(target_radio)
            sam3 = _cosine_loss(projected_sam3[sam3_valid], target_sam3_all[sam3_valid])
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
