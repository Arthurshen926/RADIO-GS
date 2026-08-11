"""Source-only object-level multiview region memory primitives.

The core contract is benchmark agnostic.  A query adapter supplies one
reference-view object observation (for example, signed scribbles completed by
an interactive mask model or an official reference full mask) and an ordered
set of query-independent Gaussian/source-view assignments.  Target images,
target labels, and benchmark metrics are deliberately outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


METHOD = "source_only_object_multiview_region_memory_v1"


def method_contract() -> dict[str, object]:
    return {
        "method": METHOD,
        "accepted_reference_prompt_kinds": [
            "positive_negative_scribbles",
            "reference_binary_mask",
        ],
        "source_rgb_only": True,
        "target_rgb_mask_or_metric_input": False,
        "projection_carrier": (
            "frozen_query_independent_primitive_top1_alpha_depth_assignment"
        ),
        "view_selection": (
            "top_k_descending_positive_anchor_coverage_times_assignment_reliability_"
            "then_frame_id"
        ),
        "sam_box_policy": {
            "padding_pixels": 16,
            "resolution": 1008,
            "confidence_threshold": 0.0,
            "minimum_projected_anchor_overlap": 0.05,
            "candidate_tie_break": "official_score_then_candidate_index",
        },
        "proposal_observation_domain": "projected_anchor_padded_box_only",
        "membership_probability": "weighted_raster_adjoint_bernoulli_mean",
        "membership_confidence": "one_minus_exp_negative_visible_evidence_mass",
        "probability_and_confidence_separate": True,
        "hard_reference_anchor_policy": "bitwise_overwrite_after_multiview_pooling",
        "region_token_policy": (
            "one_l2_normalized_positive_proposal_weighted_field_token_per_view"
        ),
        "scene_specific_parameters": False,
    }


@dataclass(frozen=True)
class ProjectedAnchorView:
    probability: torch.Tensor
    confidence: torch.Tensor
    seed: torch.Tensor
    positive_anchor_coverage: float
    assignment_reliability: float
    selection_score: float


@dataclass(frozen=True)
class SelectedSourceView:
    view_index: int
    frame_id: str
    positive_anchor_coverage: float
    assignment_reliability: float
    selection_score: float


@dataclass(frozen=True)
class RegionMembership:
    probability: torch.Tensor
    confidence: torch.Tensor
    observed: torch.Tensor


def _as_probability(value: torch.Tensor, *, label: str) -> torch.Tensor:
    result = torch.as_tensor(value, device="cpu").detach().float().reshape(-1)
    if not bool(torch.isfinite(result).all()) or bool(
        ((result < 0) | (result > 1)).any()
    ):
        raise ValueError(f"{label} must be finite in [0,1]")
    return result.contiguous()


def _assignment(
    value: Mapping[str, torch.Tensor],
    *,
    num_gaussians: int,
    num_pixels: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(value, Mapping) or set(value) != {
        "gaussian_ids",
        "pixel_ids",
        "weights",
    }:
        raise ValueError("source-view assignment schema differs")
    gaussian_ids = torch.as_tensor(value["gaussian_ids"], device="cpu").long().reshape(-1)
    pixel_ids = torch.as_tensor(value["pixel_ids"], device="cpu").long().reshape(-1)
    weights = torch.as_tensor(value["weights"], device="cpu").float().reshape(-1)
    if gaussian_ids.shape != pixel_ids.shape or gaussian_ids.shape != weights.shape:
        raise ValueError("source-view assignment tensors do not align")
    if gaussian_ids.numel() == 0:
        raise ValueError("source-view assignment cannot be empty")
    if (
        int(gaussian_ids.min()) < 0
        or int(gaussian_ids.max()) >= int(num_gaussians)
        or int(pixel_ids.min()) < 0
        or int(pixel_ids.max()) >= int(num_pixels)
    ):
        raise ValueError("source-view assignment index is outside its authority")
    if not bool(torch.isfinite(weights).all()) or bool(
        ((weights <= 0) | (weights > 1.001)).any()
    ):
        raise ValueError("source-view assignment weights are invalid")
    if pixel_ids.numel() > 1 and bool((pixel_ids[1:] < pixel_ids[:-1]).any()):
        raise ValueError("source-view assignment pixel order is not canonical")
    return gaussian_ids, pixel_ids, weights


def project_anchor_to_feature_view(
    primitive_probability: torch.Tensor,
    primitive_confidence: torch.Tensor,
    assignment: Mapping[str, torch.Tensor],
    *,
    height: int,
    width: int,
) -> ProjectedAnchorView:
    """Project a registered primitive anchor to a frozen source-view grid."""

    probability = _as_probability(primitive_probability, label="anchor probability")
    confidence = _as_probability(primitive_confidence, label="anchor confidence")
    if probability.shape != confidence.shape:
        raise ValueError("anchor probability and confidence do not align")
    pixels = int(height) * int(width)
    if int(height) <= 0 or int(width) <= 0:
        raise ValueError("source-view shape must be positive")
    gaussian_ids, pixel_ids, weights = _assignment(
        assignment,
        num_gaussians=probability.numel(),
        num_pixels=pixels,
    )

    evidence = weights * confidence[gaussian_ids]
    numerator = torch.zeros(pixels, dtype=torch.float32)
    mass = torch.zeros_like(numerator)
    numerator.index_add_(
        0,
        pixel_ids,
        evidence * probability[gaussian_ids],
    )
    mass.index_add_(0, pixel_ids, evidence)
    supported = mass > 0
    projected_probability = torch.zeros_like(mass)
    projected_probability[supported] = numerator[supported] / mass[supported]
    projected_confidence = -torch.expm1(-mass)
    seed = supported & (projected_probability >= 0.5)

    positive_anchor_mass = confidence * probability
    total_positive = float(positive_anchor_mass.sum())
    if total_positive <= 0:
        raise ValueError("reference anchor has no positive evidence mass")
    visible = torch.zeros(probability.numel(), dtype=torch.bool)
    visible[gaussian_ids] = True
    coverage = float(positive_anchor_mass[visible].sum()) / total_positive
    assignment_positive = positive_anchor_mass[gaussian_ids]
    assignment_denominator = float(assignment_positive.sum())
    reliability = (
        float((assignment_positive * weights).sum()) / assignment_denominator
        if assignment_denominator > 0
        else 0.0
    )
    selection_score = coverage * reliability
    shape = (int(height), int(width))
    return ProjectedAnchorView(
        probability=projected_probability.reshape(shape),
        confidence=projected_confidence.reshape(shape),
        seed=seed.reshape(shape),
        positive_anchor_coverage=coverage,
        assignment_reliability=reliability,
        selection_score=selection_score,
    )


def select_source_views(
    frame_ids: Sequence[str],
    projected_views: Sequence[ProjectedAnchorView],
    *,
    count: int,
    reference_frame_id: str,
    forbidden_frame_ids: Sequence[str] = (),
) -> tuple[SelectedSourceView, ...]:
    """Select a fixed number of source views without RGB, GT, or metrics."""

    if len(frame_ids) != len(projected_views):
        raise ValueError("frame identities and projected views do not align")
    if int(count) <= 0:
        raise ValueError("source-view count must be positive")
    forbidden = {str(value) for value in forbidden_frame_ids}
    if str(reference_frame_id) in forbidden:
        raise ValueError("reference frame is also declared forbidden")
    identities = [str(value) for value in frame_ids]
    if len(set(identities)) != len(identities):
        raise ValueError("source frame identities must be unique")
    candidates: list[SelectedSourceView] = []
    for index, (frame_id, projection) in enumerate(zip(identities, projected_views)):
        if frame_id == str(reference_frame_id) or frame_id in forbidden:
            continue
        if projection.seed.any() and projection.selection_score > 0:
            candidates.append(
                SelectedSourceView(
                    view_index=index,
                    frame_id=frame_id,
                    positive_anchor_coverage=float(
                        projection.positive_anchor_coverage
                    ),
                    assignment_reliability=float(
                        projection.assignment_reliability
                    ),
                    selection_score=float(projection.selection_score),
                )
            )
    candidates.sort(
        key=lambda row: (
            -row.selection_score,
            -row.positive_anchor_coverage,
            -row.assignment_reliability,
            row.frame_id,
            row.view_index,
        )
    )
    if len(candidates) < int(count):
        raise ValueError("too few non-reference source views have projected anchor support")
    return tuple(candidates[: int(count)])


def sample_native_mask_at_feature_centers(
    mask: torch.Tensor,
    *,
    feature_height: int,
    feature_width: int,
) -> torch.Tensor:
    """Sample a native binary proposal at deterministic feature-cell centres."""

    value = torch.as_tensor(mask, device="cpu")
    if value.ndim != 2 or value.dtype != torch.bool:
        raise ValueError("native proposal must be a 2D bool tensor")
    if int(feature_height) <= 0 or int(feature_width) <= 0:
        raise ValueError("feature shape must be positive")
    height, width = map(int, value.shape)
    ys = torch.floor(
        (torch.arange(int(feature_height), dtype=torch.float64) + 0.5)
        * float(height)
        / float(feature_height)
    ).long().clamp(max=height - 1)
    xs = torch.floor(
        (torch.arange(int(feature_width), dtype=torch.float64) + 0.5)
        * float(width)
        / float(feature_width)
    ).long().clamp(max=width - 1)
    return value[ys[:, None], xs[None, :]].contiguous()


def aggregate_proposal_membership(
    assignments: Sequence[Mapping[str, torch.Tensor]],
    proposal_masks: Sequence[torch.Tensor],
    observation_domains: Sequence[torch.Tensor],
    view_reliability: Sequence[float],
    *,
    num_gaussians: int,
    anchor_probability: torch.Tensor | None = None,
    anchor_confidence: torch.Tensor | None = None,
    hard_anchor: torch.Tensor | None = None,
) -> RegionMembership:
    """Lift selected source proposals and preserve legal prompt anchors exactly."""

    count = len(assignments)
    if not (
        count
        == len(proposal_masks)
        == len(observation_domains)
        == len(view_reliability)
    ) or count == 0:
        raise ValueError("proposal observations must be nonempty and aligned")
    numerator = torch.zeros(int(num_gaussians), dtype=torch.float32)
    mass = torch.zeros_like(numerator)
    for assignment, proposal, domain, reliability in zip(
        assignments,
        proposal_masks,
        observation_domains,
        view_reliability,
    ):
        proposal_value = torch.as_tensor(proposal, device="cpu")
        domain_value = torch.as_tensor(domain, device="cpu")
        if (
            proposal_value.ndim != 2
            or proposal_value.dtype != torch.bool
            or domain_value.shape != proposal_value.shape
            or domain_value.dtype != torch.bool
        ):
            raise ValueError("proposal and observation domain must be aligned bool masks")
        reliability_value = float(reliability)
        if not 0 <= reliability_value <= 1:
            raise ValueError("view reliability must be in [0,1]")
        gaussian_ids, pixel_ids, weights = _assignment(
            assignment,
            num_gaussians=int(num_gaussians),
            num_pixels=proposal_value.numel(),
        )
        flat_proposal = proposal_value.reshape(-1)
        flat_domain = domain_value.reshape(-1)
        keep = flat_domain[pixel_ids]
        if not bool(keep.any()) or reliability_value == 0:
            continue
        gaussian_ids = gaussian_ids[keep]
        pixel_ids = pixel_ids[keep]
        evidence = weights[keep] * reliability_value
        numerator.index_add_(
            0,
            gaussian_ids,
            evidence * flat_proposal[pixel_ids].float(),
        )
        mass.index_add_(0, gaussian_ids, evidence)
    observed = mass > 0
    probability = torch.zeros_like(mass)
    probability[observed] = numerator[observed] / mass[observed]
    confidence = -torch.expm1(-mass)

    provided = (anchor_probability, anchor_confidence, hard_anchor)
    if any(value is not None for value in provided):
        if not all(value is not None for value in provided):
            raise ValueError("anchor overwrite requires probability, confidence, and mask")
        anchor_q = _as_probability(anchor_probability, label="anchor probability")
        anchor_c = _as_probability(anchor_confidence, label="anchor confidence")
        anchor_mask = torch.as_tensor(hard_anchor, device="cpu").bool().reshape(-1)
        if not (
            anchor_q.shape
            == anchor_c.shape
            == anchor_mask.shape
            == probability.shape
        ):
            raise ValueError("hard anchor tensors do not align with the primitive domain")
        probability[anchor_mask] = anchor_q[anchor_mask]
        confidence[anchor_mask] = anchor_c[anchor_mask]
        observed[anchor_mask] = anchor_c[anchor_mask] > 0
    return RegionMembership(probability, confidence, observed)


def pool_region_token_set(
    primitive_features: torch.Tensor,
    assignments: Sequence[Mapping[str, torch.Tensor]],
    proposal_masks: Sequence[torch.Tensor],
    view_reliability: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool one normalized field token per accepted source-view proposal."""

    features = torch.as_tensor(primitive_features, device="cpu").detach().float()
    if features.ndim != 2 or not bool(torch.isfinite(features).all()):
        raise ValueError("primitive features must be finite [N,D]")
    if not (
        len(assignments) == len(proposal_masks) == len(view_reliability)
    ) or not assignments:
        raise ValueError("token observations must be nonempty and aligned")
    tokens: list[torch.Tensor] = []
    reliability_rows: list[float] = []
    for assignment, proposal, reliability in zip(
        assignments, proposal_masks, view_reliability
    ):
        mask = torch.as_tensor(proposal, device="cpu")
        if mask.ndim != 2 or mask.dtype != torch.bool:
            raise ValueError("proposal must be a 2D bool tensor")
        reliability_value = float(reliability)
        if not 0 < reliability_value <= 1:
            raise ValueError("token view reliability must be in (0,1]")
        gaussian_ids, pixel_ids, weights = _assignment(
            assignment,
            num_gaussians=features.shape[0],
            num_pixels=mask.numel(),
        )
        keep = mask.reshape(-1)[pixel_ids]
        if not bool(keep.any()):
            raise ValueError("accepted proposal contains no assigned primitive")
        token_weights = weights[keep] * reliability_value
        token = (
            features[gaussian_ids[keep]] * token_weights[:, None]
        ).sum(dim=0) / token_weights.sum()
        norm = torch.linalg.vector_norm(token)
        if not bool(torch.isfinite(norm)) or float(norm) <= 0:
            raise ValueError("pooled region token is degenerate")
        tokens.append(token / norm)
        reliability_rows.append(reliability_value)
    return (
        torch.stack(tokens, dim=0).contiguous(),
        torch.tensor(reliability_rows, dtype=torch.float32),
    )
