"""Shared primitive-domain unary scorer for every query modality."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
import torch.nn.functional as F

from .query_spec import PrimitiveUnaryEvidence, PrototypeSet, QuerySpec
from .score_calibration import (
    SceneSpaceCalibration,
    robust_tanh_score_calibration,
)


@dataclass(frozen=True)
class EvidenceScoringConfig:
    semantic_weight: float = 1.0
    appearance_weight: float = 1.0
    boundary_weight: float = 0.35
    prototype_temperature: float = 0.07
    feature_calibration: str = "none"
    background_centroids: int = 0
    background_negative_policy: str = "pooled_mean"
    calibration_sample_size: int = 8192
    centroid_iterations: int = 4
    score_calibration: str = "none"
    score_tanh_scale: float = 2.0
    score_chunk_size: int = 65536
    negative_spatial_mode: str = "none"
    negative_spatial_steps: int = 4
    negative_spatial_decay: float = 0.8
    spatial_log_weight: float = 0.25
    spatial_floor: float = 0.01
    # Registered masks/scribbles are observations, not merely a way to build
    # appearance prototypes.  Keep their direct signed-unary contribution
    # opt-in so frozen benchmark contracts remain bit-for-bit compatible.
    registered_seed_unary_weight: float = 0.0
    # ``additive`` preserves the historical opt-in heuristic.  The
    # probability mixture instead treats prompt mass as an observation:
    # p=(1-c)*p_field+c*q, with no task-label calibration or scale-mismatched
    # addition to cosine margins.
    registered_observation_fusion: str = "additive"

    def __post_init__(self) -> None:
        if min(self.semantic_weight, self.appearance_weight, self.boundary_weight) < 0:
            raise ValueError("evidence weights must be non-negative")
        if self.prototype_temperature <= 0:
            raise ValueError("prototype_temperature must be positive")
        if (
            not math.isfinite(self.registered_seed_unary_weight)
            or self.registered_seed_unary_weight < 0
        ):
            raise ValueError(
                "registered_seed_unary_weight must be finite and non-negative"
            )
        if self.registered_observation_fusion not in {
            "additive",
            "probability_mixture",
            "hard_seed_anchored_probability",
        }:
            raise ValueError(
                "registered_observation_fusion must be additive, "
                "probability_mixture, or hard_seed_anchored_probability"
            )
        if (
            self.registered_observation_fusion
            in {"probability_mixture", "hard_seed_anchored_probability"}
            and self.registered_seed_unary_weight != 0
        ):
            raise ValueError(
                "probability fusion requires registered_seed_unary_weight=0"
            )
        if self.feature_calibration not in {"none", "diagonal_robust"}:
            raise ValueError("feature_calibration must be none or diagonal_robust")
        if self.background_negative_policy not in {
            "pooled_mean",
            "explicit_hard_max",
        }:
            raise ValueError(
                "background_negative_policy must be pooled_mean or explicit_hard_max"
            )
        if self.score_calibration not in {
            "none",
            "robust_tanh",
            "robust_tanh_centered",
            "robust_tanh_zero",
        }:
            raise ValueError("unsupported score_calibration")
        if self.negative_spatial_mode not in {
            "none",
            "truncated_graph_decay",
            "signed_geodesic",
        }:
            raise ValueError(
                "negative_spatial_mode must be none, truncated_graph_decay, "
                "or signed_geodesic"
            )
        if (
            self.background_centroids < 0
            or self.calibration_sample_size <= 0
            or self.centroid_iterations < 0
            or self.score_tanh_scale <= 0
            or self.score_chunk_size <= 0
            or self.negative_spatial_steps < 0
            or not 0 <= self.negative_spatial_decay <= 1
            or self.spatial_log_weight < 0
            or not 0 < self.spatial_floor <= 1
        ):
            raise ValueError("invalid evidence calibration parameters")


@dataclass(frozen=True)
class RegisteredForwardBetaDiagnostics:
    """Auditable terms produced by one registered-prompt mixture E-step.

    All tensor-valued diagnostics are CPU float64 vectors.  ``nll_*`` is the
    valid-alpha-weighted mean Bernoulli negative log likelihood over labeled
    pixels that have at least one capability-valid contribution.  When that
    mass is zero, both NLL values are zero and
    ``observable_labeled_alpha_mass`` records that the value is vacuous.

    This diagnostic belongs to a method primitive and makes no independent
    protocol claim.  The caller is responsible for binding it to an evaluation
    authority; the current forward-Beta candidate remains non-exact and is not
    promoted to a strict-unseen result row.
    """

    positive_expected_count: torch.Tensor
    negative_expected_count: torch.Tensor
    labeled_expected_count: torch.Tensor
    visible_contribution_mass: torch.Tensor
    labeled_contribution_mass: torch.Tensor
    labeled_coverage: torch.Tensor
    beta_confidence: torch.Tensor
    effective_confidence: torch.Tensor
    observation_probability: torch.Tensor
    fused_probability: torch.Tensor
    forward_probability_before: torch.Tensor
    forward_probability_after: torch.Tensor
    nll_before: float
    nll_after: float
    observable_labeled_alpha_mass: float
    observable_labeled_pixel_count: int
    unobservable_labeled_pixel_count: int
    valid_hit_count: int
    protocol_status: str = "method_primitive_no_independent_protocol_claim"
    # v2-only audit terms.  Keeping these optional preserves the v1 constructor
    # and serialized compact diagnostics exactly unless the independent v2
    # primitive is selected.
    raw_positive_expected_count: torch.Tensor | None = None
    raw_negative_expected_count: torch.Tensor | None = None
    field_prior_reliability: torch.Tensor | None = None
    field_prior_coverage: torch.Tensor | None = None
    field_prior_concentration: torch.Tensor | None = None
    residual_evidence_concentration: torch.Tensor | None = None
    positive_anchor_mask: torch.Tensor | None = None
    negative_anchor_mask: torch.Tensor | None = None
    positive_class_balance_scale: float | None = None
    negative_class_balance_scale: float | None = None


def shrink_unary_by_reliability(
    unary: torch.Tensor,
    reliability: torch.Tensor,
) -> torch.Tensor:
    """Shrink signed evidence toward its neutral zero, never toward background."""

    values = torch.as_tensor(unary)
    confidence = torch.as_tensor(reliability).to(
        device=values.device, dtype=values.dtype
    )
    if values.ndim != 1 or confidence.shape != values.shape:
        raise ValueError("unary and primitive reliability must align as [N]")
    if not bool(torch.isfinite(confidence).all()):
        raise ValueError("primitive reliability contains NaN or infinity")
    if bool((confidence < 0).any()) or bool((confidence > 1).any()):
        raise ValueError("primitive reliability must be in [0,1]")
    return values * confidence


def registered_seed_observation(
    positive: torch.Tensor,
    negative: torch.Tensor | None,
    *,
    confidence_mode: str = "relative_joint_max",
    mass_scale: float = 1.0,
    visible_mass: torch.Tensor | None = None,
    coverage_power: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return confidence-weighted signed purity and joint confidence.

    ``relative_joint_max`` preserves the original query-compiler arithmetic.
    ``poisson_mass`` is intended for raw raster-adjoint contribution sums:
    ``c = 1-exp(-(m+ + m-)/mass_scale)`` is the probability of at least one
    effective observation under a Poisson model, while
    ``s = c*(m+ - m-)/(m+ + m-)`` retains foreground/background purity.  It
    gives tiny raster tails tiny confidence without scene-wise calibration.

    ``poisson_mass_coverage`` additionally multiplies ``c`` by the labeled
    fraction of the primitive's visible raster footprint.  This distinction
    is important for sparse scribbles: a Gaussian that accumulates substantial
    mass from a few labeled pixels is not a strong unary when most of its
    visible footprint is unlabeled.  A full reference mask has coverage one,
    while Poisson mass still suppresses tiny raster tails.
    """

    foreground = torch.as_tensor(positive).float().reshape(-1)
    if foreground.numel() == 0 or not bool(torch.isfinite(foreground).all()):
        raise ValueError("positive registered seeds must be a finite vector")
    if bool((foreground < 0).any()):
        raise ValueError("positive registered seeds must be non-negative")
    if negative is None:
        background = torch.zeros_like(foreground)
    else:
        background = torch.as_tensor(negative).float().reshape(-1)
        if background.shape != foreground.shape:
            raise ValueError("positive and negative registered seeds must align")
        if not bool(torch.isfinite(background).all()) or bool((background < 0).any()):
            raise ValueError(
                "negative registered seeds must be finite and non-negative"
            )
    joint = foreground + background
    if not math.isfinite(float(mass_scale)) or float(mass_scale) <= 0:
        raise ValueError("registered observation mass_scale must be positive")
    mode = str(confidence_mode)
    if mode not in {
        "relative_joint_max",
        "poisson_mass",
        "poisson_mass_coverage",
    }:
        raise ValueError(
            "registered observation confidence_mode must be "
            "relative_joint_max, poisson_mass, or poisson_mass_coverage"
        )
    visible: torch.Tensor | None = None
    power = float(coverage_power)
    if mode == "poisson_mass_coverage":
        if visible_mass is None:
            raise ValueError(
                "poisson_mass_coverage requires visible raster mass"
            )
        visible = torch.as_tensor(visible_mass).float().reshape(-1)
        if (
            visible.shape != joint.shape
            or not bool(torch.isfinite(visible).all())
            or bool((visible < 0).any())
        ):
            raise ValueError(
                "visible raster mass must be a finite non-negative aligned vector"
            )
        if not math.isfinite(power) or power <= 0:
            raise ValueError("registered observation coverage_power must be positive")
        tolerance = 1e-5 * torch.maximum(
            torch.ones_like(visible),
            visible,
        )
        if bool((joint > visible + tolerance).any()):
            raise ValueError(
                "registered prompt mass cannot exceed visible raster mass"
            )
    joint_scale = joint.amax()
    if joint_scale <= 0:
        zeros = torch.zeros_like(foreground)
        return zeros, zeros
    if mode == "relative_joint_max":
        confidence = (joint / joint_scale).clamp(0.0, 1.0)
        signed = ((foreground - background) / joint_scale).clamp(-1.0, 1.0)
        return signed, confidence
    confidence = -torch.expm1(-joint / float(mass_scale))
    if mode == "poisson_mass_coverage":
        assert visible is not None
        labeled_fraction = torch.where(
            visible > 0,
            joint / visible.clamp_min(1e-30),
            torch.zeros_like(joint),
        ).clamp(0.0, 1.0)
        confidence = confidence * labeled_fraction.pow(power)
    purity = torch.where(
        joint > 0,
        (foreground - background) / joint.clamp_min(1e-30),
        torch.zeros_like(joint),
    ).clamp(-1.0, 1.0)
    signed = confidence * purity
    return signed, confidence


def registered_seed_unary(
    positive: torch.Tensor,
    negative: torch.Tensor | None,
) -> torch.Tensor:
    """Backward-compatible signed registered unary."""

    return registered_seed_observation(positive, negative)[0]


@torch.no_grad()
def registered_forward_beta_observation(
    gaussian_ids: torch.Tensor,
    pixel_ids: torch.Tensor,
    contribution_weights: torch.Tensor,
    capability_valid: torch.Tensor,
    field_prior: torch.Tensor,
    positive_pixel_mask: torch.Tensor,
    negative_pixel_mask: torch.Tensor,
    labeled_pixel_mask: torch.Tensor,
    all_pixel_mask: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[PrimitiveUnaryEvidence, RegisteredForwardBetaDiagnostics]:
    """Construct a strong registered unary from one forward-likelihood E-step.

    ``contribution_weights`` are exact front-to-back compositor weights for the
    sparse ``(pixel_ids, gaussian_ids)`` hits.  Only capability-valid rows enter
    the forward mixture.  If ``q_i`` is the supplied field probability and
    ``r_p`` its valid-contribution-normalized forward render, the expected
    alpha-weighted positive and negative counts are

    ``n+_i = q_i * sum_{p positive} A_pi / r_p`` and
    ``n-_i = (1-q_i) * sum_{p negative} A_pi / (1-r_p)``.

    Let ``mu=n+/(n++n-)`` and let ``rho`` be the labeled fraction of a row's
    visible contribution mass.  The returned observation uses

    ``c_beta=n/(1+n)``, ``c=1-(1-rho)/(1+n)``, and
    ``signed=c*(2*mu-1)``.

    Coverage therefore only strengthens the unit-pseudocount Beta update; it
    never multiplies away a sparse scribble.  Unobserved rows have exactly
    zero confidence and leave the field prior unchanged.  A full mask gives
    unit confidence to every visible capability-valid row.  Saturated zero- or
    one-probability priors use the sign-symmetric limiting responsibility
    ``A_pi`` when the corresponding labeled likelihood is zero.

    This is a CPU-only pure method primitive.  The NVOS evaluator invokes it
    only behind the explicit forward-Beta candidate switch and a validated
    protocol-authority receipt.  That candidate is deliberately non-exact and
    is not a promoted strict-unseen result row.
    """

    tensors = {
        "gaussian_ids": torch.as_tensor(gaussian_ids),
        "pixel_ids": torch.as_tensor(pixel_ids),
        "contribution_weights": torch.as_tensor(contribution_weights),
        "capability_valid": torch.as_tensor(capability_valid),
        "field_prior": torch.as_tensor(field_prior),
        "positive_pixel_mask": torch.as_tensor(positive_pixel_mask),
        "negative_pixel_mask": torch.as_tensor(negative_pixel_mask),
        "labeled_pixel_mask": torch.as_tensor(labeled_pixel_mask),
        "all_pixel_mask": torch.as_tensor(all_pixel_mask),
    }
    non_cpu = sorted(
        name for name, value in tensors.items() if value.device.type != "cpu"
    )
    if non_cpu:
        raise ValueError(
            "registered forward Beta observation is CPU-only; non-CPU inputs: "
            + ", ".join(non_cpu)
        )

    gids = tensors["gaussian_ids"]
    pids = tensors["pixel_ids"]
    weights = tensors["contribution_weights"]
    valid = tensors["capability_valid"]
    prior = tensors["field_prior"]
    positive = tensors["positive_pixel_mask"]
    negative = tensors["negative_pixel_mask"]
    labeled = tensors["labeled_pixel_mask"]
    all_pixels = tensors["all_pixel_mask"]

    integer_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
    if gids.ndim != 1 or pids.ndim != 1 or weights.ndim != 1:
        raise ValueError("sparse contribution triplets must be one-dimensional")
    if gids.dtype not in integer_dtypes or pids.dtype not in integer_dtypes:
        raise ValueError("gaussian_ids and pixel_ids must have integer dtype")
    if gids.shape != pids.shape or gids.shape != weights.shape:
        raise ValueError("sparse contribution triplets must have matching shapes")
    if valid.ndim != 1 or prior.ndim != 1 or valid.shape != prior.shape:
        raise ValueError("capability_valid and field_prior must align as [N]")
    if valid.numel() == 0:
        raise ValueError("field_prior must contain at least one primitive")
    if valid.dtype != torch.bool:
        raise ValueError("capability_valid must have boolean dtype")
    pixel_masks = {
        "positive_pixel_mask": positive,
        "negative_pixel_mask": negative,
        "labeled_pixel_mask": labeled,
        "all_pixel_mask": all_pixels,
    }
    if any(mask.ndim != 1 for mask in pixel_masks.values()):
        raise ValueError("registered pixel masks must be one-dimensional")
    if any(mask.dtype != torch.bool for mask in pixel_masks.values()):
        raise ValueError("registered pixel masks must have boolean dtype")
    pixel_count = int(all_pixels.numel())
    if pixel_count == 0 or any(
        mask.shape != all_pixels.shape for mask in pixel_masks.values()
    ):
        raise ValueError("registered pixel masks must align as non-empty [P]")
    if bool((positive & negative).any()):
        raise ValueError("positive and negative pixel masks must be disjoint")
    if not torch.equal(labeled, positive | negative):
        raise ValueError(
            "labeled_pixel_mask must equal positive_pixel_mask OR negative_pixel_mask"
        )
    if bool((labeled & ~all_pixels).any()):
        raise ValueError("labeled pixels must be a subset of all_pixel_mask")

    if not prior.dtype.is_floating_point:
        raise ValueError("field_prior must have floating-point dtype")
    if not bool(torch.isfinite(prior).all()) or bool((prior < 0).any()) or bool(
        (prior > 1).any()
    ):
        raise ValueError("field_prior must contain finite probabilities in [0,1]")
    if not weights.dtype.is_floating_point:
        raise ValueError("contribution_weights must have floating-point dtype")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()) or bool(
        (weights > 1).any()
    ):
        raise ValueError("contribution_weights must be finite values in [0,1]")
    epsilon = float(eps)
    if not math.isfinite(epsilon) or not 0.0 < epsilon < 0.5:
        raise ValueError("eps must be finite and in (0,0.5)")

    primitive_count = int(prior.numel())
    if gids.numel():
        gids_long = gids.long()
        pids_long = pids.long()
        if int(gids_long.min()) < 0 or int(gids_long.max()) >= primitive_count:
            raise ValueError("gaussian id outside field_prior")
        if int(pids_long.min()) < 0 or int(pids_long.max()) >= pixel_count:
            raise ValueError("pixel id outside registered pixel masks")
    else:
        gids_long = gids.long()
        pids_long = pids.long()

    weights64 = weights.double()
    raw_pixel_mass = torch.zeros(pixel_count, dtype=torch.float64)
    if pids_long.numel():
        raw_pixel_mass.index_add_(0, pids_long, weights64)
    alpha_tolerance = 1e-5 * torch.maximum(
        torch.ones_like(raw_pixel_mass), raw_pixel_mass
    )
    if bool((raw_pixel_mass > 1.0 + alpha_tolerance).any()):
        raise ValueError(
            "front-to-back contribution mass must not exceed one per pixel"
        )

    valid_hit = (
        valid[gids_long]
        & all_pixels[pids_long]
        & (weights64 > 0)
    )
    valid_gids = gids_long[valid_hit]
    valid_pids = pids_long[valid_hit]
    valid_weights = weights64[valid_hit]
    prior64 = prior.double()

    pixel_visible_mass = torch.zeros(pixel_count, dtype=torch.float64)
    pixel_foreground_mass = torch.zeros(pixel_count, dtype=torch.float64)
    if valid_gids.numel():
        pixel_visible_mass.index_add_(0, valid_pids, valid_weights)
        pixel_foreground_mass.index_add_(
            0,
            valid_pids,
            valid_weights * prior64[valid_gids],
        )
    supported_pixel = pixel_visible_mass > 0
    forward_before = torch.zeros(pixel_count, dtype=torch.float64)
    forward_before[supported_pixel] = (
        pixel_foreground_mass[supported_pixel]
        / pixel_visible_mass[supported_pixel]
    ).clamp(0.0, 1.0)

    positive_count = torch.zeros(primitive_count, dtype=torch.float64)
    negative_count = torch.zeros(primitive_count, dtype=torch.float64)
    visible_mass = torch.zeros(primitive_count, dtype=torch.float64)
    labeled_mass = torch.zeros(primitive_count, dtype=torch.float64)
    if valid_gids.numel():
        visible_mass.index_add_(0, valid_gids, valid_weights)
        hit_labeled = labeled[valid_pids]
        if bool(hit_labeled.any()):
            labeled_mass.index_add_(
                0,
                valid_gids[hit_labeled],
                valid_weights[hit_labeled],
            )

        hit_positive = positive[valid_pids]
        if bool(hit_positive.any()):
            pos_gids = valid_gids[hit_positive]
            pos_pids = valid_pids[hit_positive]
            pos_weights = valid_weights[hit_positive]
            pos_prior = prior64[pos_gids]
            pos_forward = forward_before[pos_pids]
            # If r=0, all contributing priors are zero.  The q/r limit under a
            # common positive perturbation is one, so responsibility is A_pi.
            safe_pos_forward = torch.where(
                pos_forward > 0,
                pos_forward,
                torch.ones_like(pos_forward),
            )
            pos_factor = torch.where(
                pos_forward > 0,
                pos_prior / safe_pos_forward,
                torch.ones_like(pos_forward),
            )
            positive_count.index_add_(0, pos_gids, pos_weights * pos_factor)

        hit_negative = negative[valid_pids]
        if bool(hit_negative.any()):
            neg_gids = valid_gids[hit_negative]
            neg_pids = valid_pids[hit_negative]
            neg_weights = valid_weights[hit_negative]
            neg_prior = prior64[neg_gids]
            neg_background = 1.0 - forward_before[neg_pids]
            # Symmetric limiting responsibility for an all-foreground prior.
            safe_neg_background = torch.where(
                neg_background > 0,
                neg_background,
                torch.ones_like(neg_background),
            )
            neg_factor = torch.where(
                neg_background > 0,
                (1.0 - neg_prior) / safe_neg_background,
                torch.ones_like(neg_background),
            )
            negative_count.index_add_(0, neg_gids, neg_weights * neg_factor)

    expected_count = positive_count + negative_count
    safe_visible_mass = torch.where(
        visible_mass > 0,
        visible_mass,
        torch.ones_like(visible_mass),
    )
    coverage = torch.where(
        visible_mass > 0,
        labeled_mass / safe_visible_mass,
        torch.zeros_like(visible_mass),
    ).clamp(0.0, 1.0)
    safe_expected_count = torch.where(
        expected_count > 0,
        expected_count,
        torch.ones_like(expected_count),
    )
    observation_probability = torch.where(
        expected_count > 0,
        positive_count / safe_expected_count,
        torch.full_like(expected_count, 0.5),
    ).clamp(0.0, 1.0)
    beta_confidence = expected_count / (1.0 + expected_count)
    effective_confidence = (
        1.0 - (1.0 - coverage) / (1.0 + expected_count)
    ).clamp(0.0, 1.0)
    # Invalid rows are method-ineligible even if malformed triplets tried to
    # attach prompt observations to them.
    effective_confidence = torch.where(
        valid,
        effective_confidence,
        torch.zeros_like(effective_confidence),
    )
    signed = (
        effective_confidence * (2.0 * observation_probability - 1.0)
    ).clamp(-1.0, 1.0)
    fused_probability = torch.where(
        valid,
        (1.0 - effective_confidence) * prior64
        + effective_confidence * observation_probability,
        prior64,
    ).clamp(0.0, 1.0)

    pixel_foreground_after = torch.zeros(pixel_count, dtype=torch.float64)
    if valid_gids.numel():
        pixel_foreground_after.index_add_(
            0,
            valid_pids,
            valid_weights * fused_probability[valid_gids],
        )
    forward_after = torch.zeros(pixel_count, dtype=torch.float64)
    forward_after[supported_pixel] = (
        pixel_foreground_after[supported_pixel]
        / pixel_visible_mass[supported_pixel]
    ).clamp(0.0, 1.0)

    observable_labeled = labeled & supported_pixel
    observable_mass = pixel_visible_mass[observable_labeled].sum()

    def weighted_nll(probability: torch.Tensor) -> float:
        if float(observable_mass) == 0.0:
            return 0.0
        observed_probability = probability[observable_labeled].clamp(
            epsilon, 1.0 - epsilon
        )
        observed_positive = positive[observable_labeled]
        loss = torch.where(
            observed_positive,
            -torch.log(observed_probability),
            -torch.log1p(-observed_probability),
        )
        return float(
            (
                loss * pixel_visible_mass[observable_labeled]
            ).sum()
            / observable_mass
        )

    evidence = PrimitiveUnaryEvidence(
        signed.float(),
        "forward_likelihood_beta_coverage_v1",
        confidence=effective_confidence.float(),
    )
    diagnostics = RegisteredForwardBetaDiagnostics(
        positive_expected_count=positive_count,
        negative_expected_count=negative_count,
        labeled_expected_count=expected_count,
        visible_contribution_mass=visible_mass,
        labeled_contribution_mass=labeled_mass,
        labeled_coverage=coverage,
        beta_confidence=beta_confidence,
        effective_confidence=effective_confidence,
        observation_probability=observation_probability,
        fused_probability=fused_probability,
        forward_probability_before=forward_before,
        forward_probability_after=forward_after,
        nll_before=weighted_nll(forward_before),
        nll_after=weighted_nll(forward_after),
        observable_labeled_alpha_mass=float(observable_mass),
        observable_labeled_pixel_count=int(observable_labeled.sum()),
        unobservable_labeled_pixel_count=int((labeled & ~supported_pixel).sum()),
        valid_hit_count=int(valid_hit.sum()),
    )
    return evidence, diagnostics


@torch.no_grad()
def registered_forward_beta_balanced_residual_observation(
    gaussian_ids: torch.Tensor,
    pixel_ids: torch.Tensor,
    contribution_weights: torch.Tensor,
    capability_valid: torch.Tensor,
    field_prior: torch.Tensor,
    primitive_reliability: torch.Tensor,
    primitive_coverage: torch.Tensor,
    positive_pixel_mask: torch.Tensor,
    negative_pixel_mask: torch.Tensor,
    labeled_pixel_mask: torch.Tensor,
    all_pixel_mask: torch.Tensor,
    *,
    anchor_threshold: float,
    eps: float = 1e-12,
) -> tuple[PrimitiveUnaryEvidence, RegisteredForwardBetaDiagnostics]:
    """Target-blind class-balanced forward-Beta residual with explicit anchors.

    The v1 E-step supplies exact compositor responsibilities, but v2 removes
    prompt area as a foreground/background prior by rescaling the two observed
    classes to the same global expected-count total.  Per-row evidence mass is
    then tempered as ``m/(1+m)``.  A query-independent canonical reliability
    ``r`` and observation-coverage component ``v`` define the bounded semantic
    prior concentration ``kappa=1+r*v``.  Thus ``kappa`` is in ``[1,2]`` and
    the non-anchor Beta residual has concentration below one: the existing
    semantic unary always retains at least half of the posterior precision.

    Rows with sufficiently strong sign-pure direct evidence reuse the solver's
    hard-seed threshold and become explicit probability/seed anchors.  This is
    not a learned or scene-specific threshold.  If neither sign has observable
    compositor mass, the method exactly restores the field prior.  If only one
    requested sign is observable, class balancing is undefined and fails
    closed rather than inventing a class prior.
    """

    reliability = torch.as_tensor(primitive_reliability)
    coverage_proxy = torch.as_tensor(primitive_coverage)
    prior = torch.as_tensor(field_prior)
    valid = torch.as_tensor(capability_valid)
    auxiliary = {
        "primitive_reliability": reliability,
        "primitive_coverage": coverage_proxy,
    }
    non_cpu = sorted(
        name for name, value in auxiliary.items() if value.device.type != "cpu"
    )
    if non_cpu:
        raise ValueError(
            "registered forward Beta v2 is CPU-only; non-CPU inputs: "
            + ", ".join(non_cpu)
        )
    if (
        reliability.ndim != 1
        or coverage_proxy.ndim != 1
        or reliability.shape != prior.shape
        or coverage_proxy.shape != prior.shape
    ):
        raise ValueError(
            "primitive reliability/coverage must align with field_prior as [N]"
        )
    for name, values in auxiliary.items():
        if not values.dtype.is_floating_point:
            raise ValueError(f"{name} must have floating-point dtype")
        if (
            not bool(torch.isfinite(values).all())
            or bool((values < 0).any())
            or bool((values > 1).any())
        ):
            raise ValueError(f"{name} must contain finite values in [0,1]")
    if valid.dtype != torch.bool or valid.shape != prior.shape:
        raise ValueError("capability_valid must align with field_prior as bool [N]")
    if bool((reliability[~valid] != 0).any()) or bool(
        (coverage_proxy[~valid] != 0).any()
    ):
        raise ValueError(
            "invalid primitive rows must have zero reliability and coverage"
        )
    threshold = float(anchor_threshold)
    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("forward Beta v2 anchor_threshold must be in (0,1]")

    # Reuse the frozen v1 validation and exact responsibility implementation.
    # Its fused posterior is deliberately ignored; only target-blind E-step
    # sufficient statistics and compositor audits are consumed below.
    _, base = registered_forward_beta_observation(
        gaussian_ids,
        pixel_ids,
        contribution_weights,
        capability_valid,
        field_prior,
        positive_pixel_mask,
        negative_pixel_mask,
        labeled_pixel_mask,
        all_pixel_mask,
        eps=eps,
    )
    raw_positive = base.positive_expected_count.double()
    raw_negative = base.negative_expected_count.double()
    valid_cpu = valid.detach().cpu()
    positive_total = raw_positive[valid_cpu].sum()
    negative_total = raw_negative[valid_cpu].sum()
    has_positive = float(positive_total) > 0.0
    has_negative = float(negative_total) > 0.0
    if has_positive != has_negative:
        raise ValueError(
            "forward Beta v2 class balancing requires observable positive and "
            "negative evidence"
        )

    if has_positive:
        # Preserve the total amount of labeled evidence while giving each sign
        # exactly half of it.  Scribble area can change evidence precision, but
        # cannot act as a foreground/background class prior.
        common_total = 0.5 * (positive_total + negative_total)
        positive_scale = float(common_total / positive_total)
        negative_scale = float(common_total / negative_total)
        positive_count = raw_positive * positive_scale
        negative_count = raw_negative * negative_scale
    else:
        positive_scale = 0.0
        negative_scale = 0.0
        positive_count = torch.zeros_like(raw_positive)
        negative_count = torch.zeros_like(raw_negative)

    balanced_count = positive_count + negative_count
    safe_balanced_count = torch.where(
        balanced_count > 0,
        balanced_count,
        torch.ones_like(balanced_count),
    )
    observation_probability = torch.where(
        balanced_count > 0,
        positive_count / safe_balanced_count,
        torch.full_like(balanced_count, 0.5),
    ).clamp(0.0, 1.0)
    residual_concentration = balanced_count / (1.0 + balanced_count)

    reliability64 = reliability.detach().double().cpu()
    coverage64 = coverage_proxy.detach().double().cpu()
    prior64 = prior.detach().double().cpu()
    prior_concentration = 1.0 + reliability64 * coverage64
    posterior_confidence = residual_concentration / (
        prior_concentration + residual_concentration
    )
    fused_probability = (
        prior_concentration * prior64
        + residual_concentration * observation_probability
    ) / (prior_concentration + residual_concentration)

    # Anchors are decided from unscaled direct evidence.  Global class
    # balancing must neither manufacture a hard seed from a tiny minority
    # raster tail nor erase a strong majority-class full-mask observation.
    raw_count = raw_positive + raw_negative
    direct_signed_strength = torch.where(
        raw_count > 0,
        (raw_positive - raw_negative).abs() / (1.0 + raw_count),
        torch.zeros_like(raw_count),
    )
    positive_anchor = (
        valid_cpu
        & (raw_positive > raw_negative)
        & (direct_signed_strength >= threshold)
    )
    negative_anchor = (
        valid_cpu
        & (raw_negative > raw_positive)
        & (direct_signed_strength >= threshold)
    )
    anchor = positive_anchor | negative_anchor
    effective_confidence = torch.where(
        anchor,
        torch.ones_like(posterior_confidence),
        posterior_confidence,
    )
    fused_probability = torch.where(
        positive_anchor,
        torch.ones_like(fused_probability),
        torch.where(negative_anchor, torch.zeros_like(fused_probability), fused_probability),
    ).clamp(0.0, 1.0)
    signed = torch.where(
        positive_anchor,
        torch.ones_like(effective_confidence),
        torch.where(
            negative_anchor,
            -torch.ones_like(effective_confidence),
            posterior_confidence * (2.0 * observation_probability - 1.0),
        ),
    ).clamp(-1.0, 1.0)
    # Capability-invalid rows are method-ineligible and exactly retain field.
    effective_confidence = torch.where(
        valid_cpu, effective_confidence, torch.zeros_like(effective_confidence)
    )
    signed = torch.where(valid_cpu, signed, torch.zeros_like(signed))
    fused_probability = torch.where(valid_cpu, fused_probability, prior64)

    pixel_count = int(base.forward_probability_before.numel())
    gids = torch.as_tensor(gaussian_ids).detach().long().cpu().reshape(-1)
    pids = torch.as_tensor(pixel_ids).detach().long().cpu().reshape(-1)
    weights = torch.as_tensor(contribution_weights).detach().double().cpu().reshape(-1)
    all_pixels = torch.as_tensor(all_pixel_mask).detach().bool().cpu().reshape(-1)
    valid_hit = valid_cpu[gids] & all_pixels[pids] & (weights > 0)
    valid_gids = gids[valid_hit]
    valid_pids = pids[valid_hit]
    valid_weights = weights[valid_hit]
    pixel_visible_mass = torch.zeros(pixel_count, dtype=torch.float64)
    pixel_foreground_after = torch.zeros(pixel_count, dtype=torch.float64)
    if valid_gids.numel():
        pixel_visible_mass.index_add_(0, valid_pids, valid_weights)
        pixel_foreground_after.index_add_(
            0, valid_pids, valid_weights * fused_probability[valid_gids]
        )
    supported_pixel = pixel_visible_mass > 0
    forward_after = torch.zeros(pixel_count, dtype=torch.float64)
    forward_after[supported_pixel] = (
        pixel_foreground_after[supported_pixel]
        / pixel_visible_mass[supported_pixel]
    ).clamp(0.0, 1.0)

    positive_pixels = torch.as_tensor(positive_pixel_mask).detach().bool().cpu()
    negative_pixels = torch.as_tensor(negative_pixel_mask).detach().bool().cpu()
    labeled_pixels = torch.as_tensor(labeled_pixel_mask).detach().bool().cpu()
    observable_labeled = labeled_pixels & supported_pixel
    observable_mass = pixel_visible_mass[observable_labeled].sum()
    epsilon = float(eps)

    def weighted_nll(probability: torch.Tensor) -> float:
        if float(observable_mass) == 0.0:
            return 0.0
        selected = probability[observable_labeled].clamp(
            epsilon, 1.0 - epsilon
        )
        loss = torch.where(
            positive_pixels[observable_labeled],
            -torch.log(selected),
            -torch.log1p(-selected),
        )
        return float(
            (loss * pixel_visible_mass[observable_labeled]).sum()
            / observable_mass
        )

    evidence = PrimitiveUnaryEvidence(
        signed.float(),
        "forward_likelihood_beta_balanced_residual_v2",
        confidence=effective_confidence.float(),
    )
    diagnostics = RegisteredForwardBetaDiagnostics(
        positive_expected_count=positive_count,
        negative_expected_count=negative_count,
        labeled_expected_count=balanced_count,
        visible_contribution_mass=base.visible_contribution_mass,
        labeled_contribution_mass=base.labeled_contribution_mass,
        labeled_coverage=base.labeled_coverage,
        beta_confidence=posterior_confidence,
        effective_confidence=effective_confidence,
        observation_probability=observation_probability,
        fused_probability=fused_probability,
        forward_probability_before=base.forward_probability_before,
        forward_probability_after=forward_after,
        nll_before=weighted_nll(base.forward_probability_before),
        nll_after=weighted_nll(forward_after),
        observable_labeled_alpha_mass=float(observable_mass),
        observable_labeled_pixel_count=int(observable_labeled.sum()),
        unobservable_labeled_pixel_count=int(
            (labeled_pixels & ~supported_pixel).sum()
        ),
        valid_hit_count=int(valid_hit.sum()),
        protocol_status=(
            "target_blind_method_primitive_v2_no_independent_protocol_claim"
        ),
        raw_positive_expected_count=raw_positive,
        raw_negative_expected_count=raw_negative,
        field_prior_reliability=reliability64,
        field_prior_coverage=coverage64,
        field_prior_concentration=prior_concentration,
        residual_evidence_concentration=residual_concentration,
        positive_anchor_mask=positive_anchor,
        negative_anchor_mask=negative_anchor,
        positive_class_balance_scale=positive_scale,
        negative_class_balance_scale=negative_scale,
    )
    return evidence, diagnostics


def registered_observation_anchor_mask(
    observation: PrimitiveUnaryEvidence,
    *,
    anchor_threshold: float,
) -> torch.Tensor:
    """Return rows whose direct signed observation is also a hard seed.

    The threshold is intentionally shared with ``SupportSolverConfig`` rather
    than introducing another strength parameter.  Confidence must be positive
    so a zero-confidence row remains unobserved even when the shared threshold
    is zero.
    """

    threshold = float(anchor_threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("registered observation anchor_threshold must be in [0,1]")
    if observation.confidence is None:
        raise ValueError(
            "hard-seed anchored probability requires explicit observation confidence"
        )
    magnitude = observation.values.abs()
    return (observation.confidence > 0) & (magnitude > 0) & (
        magnitude >= threshold
    )


def registered_observation_effective_confidence(
    observation: PrimitiveUnaryEvidence,
    *,
    anchor_threshold: float,
) -> torch.Tensor:
    """Promote direct hard-seed rows to unit fusion confidence."""

    anchors = registered_observation_anchor_mask(
        observation,
        anchor_threshold=anchor_threshold,
    )
    assert observation.confidence is not None
    return torch.where(
        anchors,
        torch.ones_like(observation.confidence),
        observation.confidence,
    )


def fuse_registered_observation_unary(
    field_unary: torch.Tensor,
    observation: PrimitiveUnaryEvidence,
    *,
    unary_temperature: float,
    chunk_size: int = 262144,
    anchor_threshold: float | None = None,
) -> torch.Tensor:
    """Fuse registered evidence with field evidence in Bernoulli space.

    Let ``c`` be normalized joint prompt mass and ``q`` the foreground purity
    of that observation.  The fused probability is
    ``(1-c)*sigmoid(u/T) + c*q``.  Thus unobserved rows preserve the field
    exactly, a pure fully observed row overrides it, and conflicting
    foreground/background mass tends continuously toward an uninformative
    observation instead of winning an arbitrary tie.

    When ``anchor_threshold`` is supplied, rows that meet the solver's same
    direct signed hard-seed threshold use effective confidence one:
    ``a=1[c>0 and |s|>=tau]``, ``c_eff=a+(1-a)c``, and
    ``p=(1-c_eff)*p_field+c_eff*q``.  This makes an accepted native-raster adjoint
    seed explicit strong unary evidence while leaving weak footprint tails on
    the coverage-conditioned mixture.  No new task-tuned constant is added.
    """

    values = torch.as_tensor(field_unary)
    if values.ndim != 1 or not bool(torch.isfinite(values).all()):
        raise ValueError("field unary must be a finite vector")
    temperature = float(unary_temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("unary_temperature must be finite and positive")
    signed = observation.values.to(device=values.device, dtype=values.dtype)
    confidence = observation.confidence
    if confidence is None:
        raise ValueError(
            "probability mixture requires explicit observation confidence"
        )
    confidence = confidence.to(device=values.device, dtype=values.dtype)
    if signed.shape != values.shape or confidence.shape != values.shape:
        raise ValueError("registered observation does not align with field unary")
    if int(chunk_size) <= 0:
        raise ValueError("registered observation fusion chunk_size must be positive")
    effective_confidence: torch.Tensor | None = None
    if anchor_threshold is not None:
        effective_confidence = registered_observation_effective_confidence(
            observation,
            anchor_threshold=float(anchor_threshold),
        ).to(device=values.device, dtype=values.dtype)
    output = torch.empty_like(values)
    eps = float(torch.finfo(torch.float64).eps)
    for start in range(0, values.numel(), int(chunk_size)):
        stop = min(start + int(chunk_size), values.numel())
        field_chunk = values[start:stop].double()
        signed_chunk = signed[start:stop].double()
        confidence_chunk = confidence[start:stop].double()
        field_probability = torch.sigmoid(field_chunk / temperature)
        if effective_confidence is None:
            # Since signed=c*(2q-1), c*q=(c+signed)/2.  This closed form avoids
            # dividing tiny confidence values and keeps the v1/v2 calculation
            # bit-compatible in float64 before returning to field-unary dtype.
            fused_probability = (
                (1.0 - confidence_chunk) * field_probability
                + 0.5 * (confidence_chunk + signed_chunk)
            )
            effective_chunk = confidence_chunk
        else:
            effective_chunk = effective_confidence[start:stop].double()
            observation_probability = torch.where(
                confidence_chunk > 0,
                0.5
                * (
                    1.0
                    + signed_chunk / confidence_chunk.clamp_min(1e-300)
                ),
                torch.full_like(confidence_chunk, 0.5),
            ).clamp(0.0, 1.0)
            fused_probability = (
                (1.0 - effective_chunk) * field_probability
                + effective_chunk * observation_probability
            )
        fused_chunk = temperature * torch.logit(
            fused_probability.clamp(eps, 1.0 - eps)
        )
        unchanged = (effective_chunk == 0) | (
            fused_probability == field_probability
        )
        output[start:stop] = torch.where(
            unchanged,
            field_chunk,
            fused_chunk,
        ).to(dtype=values.dtype)
    return output


def _score_bank(
    field_features: torch.Tensor,
    evidence: PrototypeSet,
    *,
    temperature: float,
    calibration: SceneSpaceCalibration | None = None,
    background_negative_policy: str = "pooled_mean",
    chunk_size: int = 65536,
    explicit_negative_influence: torch.Tensor | None = None,
    positive_spatial_influence: torch.Tensor | None = None,
    explicit_negative_spatial: torch.Tensor | None = None,
    spatial_log_weight: float = 0.25,
    spatial_floor: float = 0.01,
) -> torch.Tensor:
    field = torch.as_tensor(field_features)
    if field.ndim != 2 or chunk_size <= 0:
        raise ValueError("field feature bank must be [N,D] and chunk_size positive")
    if background_negative_policy not in {"pooled_mean", "explicit_hard_max"}:
        raise ValueError(
            "background_negative_policy must be pooled_mean or explicit_hard_max"
        )
    if spatial_log_weight < 0 or not 0 < spatial_floor <= 1:
        raise ValueError("spatial_log_weight/spatial_floor are invalid")
    negative_influence = None
    if explicit_negative_influence is not None:
        negative_influence = torch.as_tensor(
            explicit_negative_influence, device=field.device
        ).float().reshape(-1)
        if (
            negative_influence.shape != (field.shape[0],)
            or not bool(torch.isfinite(negative_influence).all())
            or bool((negative_influence < 0).any())
            or bool((negative_influence > 1).any())
        ):
            raise ValueError(
                "explicit_negative_influence must be finite [N] values in [0,1]"
            )
    if calibration is None:
        prototypes = evidence.features.to(field.device)
    else:
        prototypes = calibration.transform(evidence.features)
    weights = evidence.weights.to(field.device).clamp_min(1e-8)
    explicit_negatives = None
    if evidence.negatives is not None:
        explicit_negatives = (
            evidence.negatives.to(field.device)
            if calibration is None
            else calibration.transform(evidence.negatives)
        )

    def prepare_spatial(
        values: torch.Tensor | None,
        prototype_count: int,
        *,
        name: str,
    ) -> torch.Tensor | None:
        if values is None:
            return None
        matrix = torch.as_tensor(values, device=field.device).float()
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        if (
            matrix.ndim != 2
            or matrix.shape[0] != field.shape[0]
            or matrix.shape[1] not in {1, int(prototype_count)}
            or not bool(torch.isfinite(matrix).all())
            or bool((matrix < 0).any())
            or bool((matrix > 1).any())
        ):
            raise ValueError(
                f"{name} must be finite [N,1] or [N,{prototype_count}] in [0,1]"
            )
        return (
            matrix.expand(-1, int(prototype_count))
            if matrix.shape[1] == 1 and int(prototype_count) > 1
            else matrix
        )

    positive_spatial = prepare_spatial(
        positive_spatial_influence,
        prototypes.shape[0],
        name="positive_spatial_influence",
    )
    negative_spatial = prepare_spatial(
        explicit_negative_spatial,
        0 if explicit_negatives is None else explicit_negatives.shape[0],
        name="explicit_negative_spatial",
    )
    has_background_bank = (
        calibration is not None and calibration.background_centroids is not None
    )
    if has_background_bank:
        assert calibration is not None
        centroids = calibration.background_centroids
    parts: list[torch.Tensor] = []
    for start in range(0, field.shape[0], int(chunk_size)):
        rows = field[start : start + int(chunk_size)].float()
        rows = (
            F.normalize(rows, dim=-1, eps=1e-8)
            if calibration is None
            else calibration.transform(rows)
        )
        logits = rows @ prototypes.T
        if positive_spatial is not None:
            local_positive_spatial = positive_spatial[
                start : start + rows.shape[0]
            ].clamp_min(float(spatial_floor))
            logits = (
                logits
                + float(spatial_log_weight) * local_positive_spatial.log()
            )
        positive = temperature * torch.logsumexp(
            logits / temperature + weights.log()[None], dim=1
        )
        if explicit_negatives is not None or has_background_bank:
            local_influence = (
                None
                if negative_influence is None
                else negative_influence[start : start + rows.shape[0]]
            )
            explicit_logits = (
                rows @ explicit_negatives.T
                if explicit_negatives is not None
                else None
            )
            local_negative_spatial = None
            if explicit_logits is not None and negative_spatial is not None:
                local_negative_spatial = negative_spatial[
                    start : start + rows.shape[0]
                ].clamp_min(float(spatial_floor))
                explicit_logits = (
                    explicit_logits
                    + float(spatial_log_weight) * local_negative_spatial.log()
                )
            explicit_negative = (
                explicit_logits.amax(dim=1)
                if explicit_logits is not None
                else None
            )
            background_negative = None
            if has_background_bank:
                assert calibration is not None and calibration.background_centroids is not None
                background_logits = rows @ calibration.background_centroids.T
                background_negative = temperature * (
                    torch.logsumexp(background_logits / temperature, dim=1)
                    - torch.log(
                        torch.tensor(
                            calibration.background_centroids.shape[0],
                            device=rows.device,
                            dtype=torch.float32,
                        )
                    )
                )
            if explicit_negative is None:
                assert background_negative is not None
                negative = background_negative
            elif background_negative is None:
                # Preserve the original explicit-negative readout exactly.
                negative = (
                    explicit_negative
                    if local_influence is None
                    else local_influence * explicit_negative
                )
            elif background_negative_policy == "pooled_mean":
                if local_influence is None:
                    # Historical behavior: pool user negatives and scene modes into
                    # one average background bank.
                    if local_negative_spatial is None:
                        negative_logits = rows @ torch.cat(
                            [
                                explicit_negatives,
                                calibration.background_centroids,
                            ],
                            dim=0,
                        ).T
                    else:
                        background_logits = (
                            rows @ calibration.background_centroids.T
                        )
                        negative_logits = torch.cat(
                            [explicit_logits, background_logits], dim=1
                        )
                    negative = temperature * (
                        torch.logsumexp(negative_logits / temperature, dim=1)
                        - torch.log(
                            torch.tensor(
                                negative_logits.shape[1],
                                device=rows.device,
                                dtype=torch.float32,
                            )
                        )
                    )
                else:
                    # Treat graph locality as the participation mass of the
                    # explicit click bank.  At zero influence the scene bank is
                    # unchanged; at one this is exactly the historical pooled
                    # log-mean-exp over both banks.
                    assert explicit_logits is not None
                    assert calibration is not None
                    assert calibration.background_centroids is not None
                    background_logits = rows @ calibration.background_centroids.T
                    log_gate = local_influence.clamp_min(1e-30).log()[:, None]
                    gated_explicit = (
                        explicit_logits / temperature + log_gate
                    ).masked_fill(local_influence[:, None] <= 0, float("-inf"))
                    joined = torch.cat(
                        [gated_explicit, background_logits / temperature], dim=1
                    )
                    count = (
                        local_influence * float(explicit_logits.shape[1])
                        + float(background_logits.shape[1])
                    )
                    negative = temperature * (
                        torch.logsumexp(joined, dim=1) - count.log()
                    )
            else:
                # Scene modes only provide a query-independent prior.  Once a
                # user supplies a concrete negative click, it must remain hard
                # contrastive evidence instead of being diluted by every mode.
                if local_influence is None:
                    localized_explicit = explicit_negative
                else:
                    localized_explicit = (
                        local_influence * explicit_negative
                        + (1.0 - local_influence) * (-1.0)
                    )
                negative = torch.maximum(localized_explicit, background_negative)
            positive = positive - negative
        parts.append(positive)
    return torch.cat(parts)


def score_query_evidence(
    query: QuerySpec,
    feature_banks: Mapping[str, torch.Tensor],
    *,
    config: EvidenceScoringConfig = EvidenceScoringConfig(),
    calibrations: Mapping[str, SceneSpaceCalibration] | None = None,
    num_nodes: int | None = None,
    explicit_negative_influence: torch.Tensor | None = None,
    positive_spatial_influence: torch.Tensor | None = None,
    explicit_negative_spatial: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    components: dict[str, torch.Tensor] = {}
    weighted: list[torch.Tensor] = []
    for name, evidence, weight in (
        ("semantic", query.semantic_evidence, config.semantic_weight),
        ("appearance", query.appearance_evidence, config.appearance_weight),
        ("boundary", query.boundary_evidence, config.boundary_weight),
    ):
        if evidence is None:
            continue
        if name not in feature_banks:
            raise KeyError(f"query requires missing {name} feature bank")
        score = _score_bank(
            feature_banks[name],
            evidence,
            temperature=config.prototype_temperature,
            calibration=(calibrations or {}).get(name),
            background_negative_policy=config.background_negative_policy,
            chunk_size=config.score_chunk_size,
            explicit_negative_influence=explicit_negative_influence,
            positive_spatial_influence=positive_spatial_influence,
            explicit_negative_spatial=explicit_negative_spatial,
            spatial_log_weight=config.spatial_log_weight,
            spatial_floor=config.spatial_floor,
        )
        if config.score_calibration != "none":
            score = robust_tanh_score_calibration(
                score,
                tanh_scale=config.score_tanh_scale,
                preserve_zero=config.score_calibration in {
                    "robust_tanh",
                    "robust_tanh_zero",
                },
            )
        components[name] = score
        weighted.append(score * float(weight))
    if not weighted:
        count = int(num_nodes) if num_nodes is not None else 0
        if count <= 0 and query.positive_seeds is not None:
            count = query.positive_seeds.weights.numel()
        if count == 0:
            raise ValueError("query has neither prototypes nor seeds")
        for seeds in (query.positive_seeds, query.negative_seeds):
            if seeds is not None and seeds.weights.numel() != count:
                raise ValueError("query seeds do not align with num_nodes")
        # A seed-only query has no global foreground evidence.  Use the
        # minimum cosine prior instead of an ambiguous 0-logit (p=0.5), then
        # let the same graph diffuse the registered/world-space seeds.
        unary = torch.full((count,), -1.0, dtype=torch.float32)
    else:
        unary = torch.stack(weighted).sum(dim=0) / max(
            1e-8,
            sum(
                weight
                for evidence, weight in (
                    (query.semantic_evidence, config.semantic_weight),
                    (query.appearance_evidence, config.appearance_weight),
                    (query.boundary_evidence, config.boundary_weight),
                )
                if evidence is not None
            ),
        )
    return unary, components
