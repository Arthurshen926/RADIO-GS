"""A small, modality-neutral head for calibrated primitive query likelihoods.

The head consumes sets of positive and negative observation affinities that
have already been registered to canonical primitive rows.  It deliberately
does not know whether those observations came from text, an image mask, a
scribble, or a world-space click.  Set statistics make the result invariant to
observation order, while non-negative weights make the evidence direction
auditable: stronger positive evidence cannot lower foreground likelihood and
stronger negative evidence cannot raise it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .query_spec import PrimitiveUnaryEvidence


def _validate_rows(name: str, values: torch.Tensor, row_count: int) -> torch.Tensor:
    tensor = torch.as_tensor(values).float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(-1)
    if tensor.ndim != 3 or tensor.shape[0] != row_count or tensor.shape[2] <= 0:
        raise ValueError(f"{name} must be [N,K] or [N,K,C]")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite")
    if bool(((tensor < 0) | (tensor > 1)).any()):
        raise ValueError(f"{name} must be in [0,1]")
    return tensor


def _validate_vector(name: str, values: torch.Tensor, row_count: int) -> torch.Tensor:
    tensor = torch.as_tensor(values).float().reshape(-1)
    if tensor.shape != (row_count,) or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be a finite [N] vector")
    if bool(((tensor < 0) | (tensor > 1)).any()):
        raise ValueError(f"{name} must be in [0,1]")
    return tensor


@dataclass(frozen=True)
class QueryLikelihoodInputs:
    """Registered observations shared by all query modalities.

    Each affinity matrix has one primitive per row and an unordered set of
    query observations per column.  Empty positive or negative sets are valid.
    ``coverage`` says where the query was actually observed; ``reliability``
    describes the query-independent primitive field authority.  They remain
    separate from likelihood so an unobserved row is not confused with a
    50/50 foreground observation.
    """

    positive_affinity: torch.Tensor
    negative_affinity: torch.Tensor
    prior_probability: torch.Tensor
    coverage: torch.Tensor
    reliability: torch.Tensor

    def validated(self) -> "QueryLikelihoodInputs":
        positive = torch.as_tensor(self.positive_affinity).float()
        if positive.ndim not in (2, 3) or positive.shape[0] <= 0:
            raise ValueError("positive_affinity must be [N,K] or [N,K,C]")
        rows = int(positive.shape[0])
        positive = _validate_rows("positive_affinity", positive, rows)
        negative = _validate_rows(
            "negative_affinity", self.negative_affinity, rows
        )
        if positive.shape[2] != negative.shape[2]:
            raise ValueError("positive and negative affinity channels must match")
        return QueryLikelihoodInputs(
            positive_affinity=positive,
            negative_affinity=negative,
            prior_probability=_validate_vector(
                "prior_probability", self.prior_probability, rows
            ),
            coverage=_validate_vector("coverage", self.coverage, rows),
            reliability=_validate_vector("reliability", self.reliability, rows),
        )


def _set_statistics(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return peak and set-density statistics for an unordered observation set."""

    if values.shape[1] == 0:
        zeros = values.new_zeros((values.shape[0], values.shape[2]))
        return zeros, zeros
    return values.amax(dim=1), values.mean(dim=1)


class MonotoneQueryLikelihoodHead(nn.Module):
    """Calibrate canonical observations into a bounded ``q,c`` unary.

    The learned scalar weights are transformed with ``softplus``.  Therefore
    positive affinity and foreground prior have non-negative derivatives, and
    negative affinity has a non-positive derivative.  Confidence is not a
    learned certainty heuristic: it is the explicit product of observation
    coverage and field reliability, preserving exact abstention.
    """

    schema_version = "monotone-query-likelihood-v1"

    def __init__(self, *, affinity_channel_count: int = 1) -> None:
        super().__init__()
        count = int(affinity_channel_count)
        if count <= 0 or count > 64:
            raise ValueError("affinity_channel_count must be in [1,64]")
        self.affinity_channel_count = count
        self.schema_version = (
            "monotone-query-likelihood-v1"
            if count == 1
            else "monotone-query-likelihood-multichannel-v2"
        )
        self.bias = nn.Parameter(torch.zeros(()))
        weight_shape = (2,) if count == 1 else (count, 2)
        self.raw_positive_weights = nn.Parameter(torch.zeros(weight_shape))
        self.raw_negative_weights = nn.Parameter(torch.zeros(weight_shape))
        self.raw_prior_weight = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        observations: QueryLikelihoodInputs,
        *,
        source: str,
    ) -> PrimitiveUnaryEvidence:
        inputs = observations.validated()
        if inputs.positive_affinity.shape[2] != self.affinity_channel_count:
            raise ValueError(
                "likelihood input affinity channels differ from the head contract"
            )
        positive_peak, positive_mean = _set_statistics(inputs.positive_affinity)
        negative_peak, negative_mean = _set_statistics(inputs.negative_affinity)
        positive = torch.stack((positive_peak, positive_mean), dim=-1)
        negative = torch.stack((negative_peak, negative_mean), dim=-1)

        positive_weight = F.softplus(self.raw_positive_weights)
        negative_weight = F.softplus(self.raw_negative_weights)
        prior_weight = F.softplus(self.raw_prior_weight)
        prior_logit = torch.logit(inputs.prior_probability.clamp(1e-4, 1.0 - 1e-4))
        if self.affinity_channel_count == 1:
            positive = positive[:, 0, :]
            negative = negative[:, 0, :]
            positive_logit = (positive * positive_weight).sum(dim=-1)
            negative_logit = (negative * negative_weight).sum(dim=-1)
        else:
            positive_logit = (positive * positive_weight).sum(dim=(-2, -1))
            negative_logit = (negative * negative_weight).sum(dim=(-2, -1))
        logit = (
            self.bias
            + positive_logit
            - negative_logit
            + prior_weight * prior_logit
        )
        probability = torch.sigmoid(logit)
        confidence = (inputs.coverage * inputs.reliability).clamp(0.0, 1.0)
        return PrimitiveUnaryEvidence.from_probability(
            probability,
            confidence=confidence,
            source=f"{source}:{self.schema_version}",
        )


class MonotoneLikelihoodRatioHead(nn.Module):
    """Prior-invariant monotone likelihood ratio over registered evidence.

    ``log_likelihood_ratio`` contains no empirical object prevalence.  During
    source training an ordinary posterior BCE must use
    ``sigmoid(ell + logit(pi_e))`` for that example's analytic class prior;
    inference discards ``pi_e`` and compiles ``sigmoid(ell)`` into the shared
    ``PrimitiveUnaryEvidence`` contract.  This prevents a balanced loss or a
    Dice surrogate from silently becoming a foreground-mass prior.
    """

    schema_version = "monotone-query-log-likelihood-ratio-multichannel-v3"

    def __init__(self, *, affinity_channel_count: int = 1) -> None:
        super().__init__()
        count = int(affinity_channel_count)
        if count <= 0 or count > 64:
            raise ValueError("affinity_channel_count must be in [1,64]")
        self.affinity_channel_count = count
        self.bias = nn.Parameter(torch.zeros(()))
        shape = (count, 2)
        self.raw_positive_weights = nn.Parameter(torch.zeros(shape))
        self.raw_negative_weights = nn.Parameter(torch.zeros(shape))

    def log_likelihood_ratio(
        self, observations: QueryLikelihoodInputs
    ) -> torch.Tensor:
        inputs = observations.validated()
        if inputs.positive_affinity.shape[2] != self.affinity_channel_count:
            raise ValueError(
                "likelihood-ratio input channels differ from the head contract"
            )
        positive_peak, positive_mean = _set_statistics(inputs.positive_affinity)
        negative_peak, negative_mean = _set_statistics(inputs.negative_affinity)
        positive = torch.stack((positive_peak, positive_mean), dim=-1)
        negative = torch.stack((negative_peak, negative_mean), dim=-1)
        return (
            self.bias
            + (positive * F.softplus(self.raw_positive_weights)).sum(dim=(-2, -1))
            - (negative * F.softplus(self.raw_negative_weights)).sum(dim=(-2, -1))
        )

    def posterior_probability(
        self,
        observations: QueryLikelihoodInputs,
        *,
        foreground_prevalence: float | torch.Tensor,
    ) -> torch.Tensor:
        prevalence = torch.as_tensor(
            foreground_prevalence,
            device=torch.as_tensor(observations.positive_affinity).device,
            dtype=torch.float32,
        )
        if prevalence.numel() != 1 or not bool(
            ((prevalence > 0) & (prevalence < 1)).all()
        ):
            raise ValueError("foreground prevalence must be one scalar in (0,1)")
        return torch.sigmoid(
            self.log_likelihood_ratio(observations)
            + torch.logit(prevalence.reshape(()))
        )

    def forward(
        self,
        observations: QueryLikelihoodInputs,
        *,
        source: str,
    ) -> PrimitiveUnaryEvidence:
        inputs = observations.validated()
        probability = torch.sigmoid(self.log_likelihood_ratio(inputs))
        confidence = (inputs.coverage * inputs.reliability).clamp(0.0, 1.0)
        return PrimitiveUnaryEvidence.from_probability(
            probability,
            confidence=confidence,
            source=f"{source}:{self.schema_version}",
        )


class MonotoneSignedLikelihoodRatioHead(MonotoneLikelihoodRatioHead):
    """Null-centered LLR over cosine affinities encoded in ``[0,1]``.

    Registered capability affinities use ``a=(cosine+1)/2`` for transport.
    The likelihood coordinate is instead ``s=2a-1=cosine``.  Consequently a
    query-free orthogonal/null match contributes exactly zero, dissimilarity
    to a positive observation is negative evidence, and similarity to a
    negative observation is suppressive evidence.  The zero buffer is not a
    trainable intercept: absence of all observations is strictly neutral.
    """

    schema_version = "monotone-signed-null-centered-likelihood-ratio-v4"

    def __init__(self, *, affinity_channel_count: int = 1) -> None:
        nn.Module.__init__(self)
        count = int(affinity_channel_count)
        if count <= 0 or count > 64:
            raise ValueError("affinity_channel_count must be in [1,64]")
        self.affinity_channel_count = count
        shape = (count, 2)
        self.raw_positive_weights = nn.Parameter(torch.zeros(shape))
        self.raw_negative_weights = nn.Parameter(torch.zeros(shape))
        self.register_buffer("bias", torch.zeros(()))

    def log_likelihood_ratio(
        self, observations: QueryLikelihoodInputs
    ) -> torch.Tensor:
        inputs = observations.validated()
        if inputs.positive_affinity.shape[2] != self.affinity_channel_count:
            raise ValueError(
                "signed likelihood-ratio input channels differ from the head contract"
            )
        positive_signed = 2.0 * inputs.positive_affinity - 1.0
        negative_signed = 2.0 * inputs.negative_affinity - 1.0
        positive_peak, positive_mean = _set_statistics(positive_signed)
        negative_peak, negative_mean = _set_statistics(negative_signed)
        positive = torch.stack((positive_peak, positive_mean), dim=-1)
        negative = torch.stack((negative_peak, negative_mean), dim=-1)
        return (
            (positive * F.softplus(self.raw_positive_weights)).sum(dim=(-2, -1))
            - (negative * F.softplus(self.raw_negative_weights)).sum(dim=(-2, -1))
        )


class MonotoneChannelDensityRatioHead(nn.Module):
    """Per-channel one-dimensional click-similarity density ratios.

    Each calibrated observation contributes ``ell_c(s)=a_c*s+b_c`` with
    ``a_c=softplus(raw_slope_c)>=0`` and signed cosine ``s=2a-1``.  Positive
    click ratios are added and negative click ratios are subtracted.  There is
    deliberately no aggregate bias, so an empty observation set has exact
    total log-likelihood ratio zero while common/null similarities can have a
    learned negative ratio through ``b_c``.
    """

    schema_version = "monotone-channel-density-ratio-v5"

    def __init__(self, *, affinity_channel_count: int = 1) -> None:
        super().__init__()
        count = int(affinity_channel_count)
        if count <= 0 or count > 64:
            raise ValueError("affinity_channel_count must be in [1,64]")
        self.affinity_channel_count = count
        self.raw_slopes = nn.Parameter(torch.zeros(count))
        self.intercepts = nn.Parameter(torch.zeros(count))

    def per_observation_log_likelihood_ratio(
        self, affinity: torch.Tensor
    ) -> torch.Tensor:
        values = torch.as_tensor(affinity).float()
        if values.ndim != 3 or values.shape[2] != self.affinity_channel_count:
            raise ValueError("density-ratio affinity must be [N,K,C]")
        if not bool(torch.isfinite(values).all()) or bool(
            ((values < 0) | (values > 1)).any()
        ):
            raise ValueError("density-ratio affinity must be finite in [0,1]")
        signed_cosine = 2.0 * values - 1.0
        return (
            signed_cosine * F.softplus(self.raw_slopes)[None, None, :]
            + self.intercepts[None, None, :]
        )

    def log_likelihood_ratio(
        self, observations: QueryLikelihoodInputs
    ) -> torch.Tensor:
        inputs = observations.validated()
        if inputs.positive_affinity.shape[2] != self.affinity_channel_count:
            raise ValueError("density-ratio input channels differ from head contract")
        positive = self.per_observation_log_likelihood_ratio(
            inputs.positive_affinity
        ).sum(dim=(1, 2))
        negative = self.per_observation_log_likelihood_ratio(
            inputs.negative_affinity
        ).sum(dim=(1, 2))
        return positive - negative

    def forward(
        self,
        observations: QueryLikelihoodInputs,
        *,
        source: str,
    ) -> PrimitiveUnaryEvidence:
        inputs = observations.validated()
        probability = torch.sigmoid(self.log_likelihood_ratio(inputs))
        confidence = (inputs.coverage * inputs.reliability).clamp(0.0, 1.0)
        return PrimitiveUnaryEvidence.from_probability(
            probability,
            confidence=confidence,
            source=f"{source}:{self.schema_version}",
        )


class MonotoneOneSidedDensityRatioHead(MonotoneChannelDensityRatioHead):
    """Density-ratio head whose negative clicks can only suppress.

    A negative observation contributes ``-relu(ell_c(s))``.  Similarity above
    the learned same-instance midpoint suppresses a primitive, while generic
    dissimilarity contributes exact zero rather than becoming positive target
    evidence.  Adding any negative observation therefore cannot increase the
    aggregate unary of any primitive.
    """

    schema_version = "monotone-one-sided-channel-density-ratio-v6"

    def log_likelihood_ratio(
        self, observations: QueryLikelihoodInputs
    ) -> torch.Tensor:
        inputs = observations.validated()
        if inputs.positive_affinity.shape[2] != self.affinity_channel_count:
            raise ValueError("one-sided density-ratio channels differ")
        positive = self.per_observation_log_likelihood_ratio(
            inputs.positive_affinity
        ).sum(dim=(1, 2))
        negative = F.relu(
            self.per_observation_log_likelihood_ratio(inputs.negative_affinity)
        ).sum(dim=(1, 2))
        return positive - negative


REGISTERED_2D_SOURCE_RECONSTRUCTION_RECIPE = {
    "recipe_id": "registered-2d-balanced-source-reconstruction-adam64-v1",
    "optimizer": "Adam",
    "steps": 64,
    "learning_rate": 0.05,
    "weight_decay": 0.0,
    "objective": "equal-positive-negative-raster-adjoint-cross-entropy",
    "parameter_initialization": "monotone-query-likelihood-v1-default",
    "target_scope": "legal_reference_prompt_only",
    "target_rgb_or_mask_scope": "never_target_view",
}


def registered_2d_likelihood_inputs(
    observation: PrimitiveUnaryEvidence,
    *,
    prior_probability: torch.Tensor,
    reliability: torch.Tensor | None = None,
) -> QueryLikelihoodInputs:
    """Adapt one registered raster observation to the shared likelihood head.

    ``observation`` already carries the sufficient statistics of the exact
    raster adjoint: its foreground probability is positive versus negative
    prompt support, while its confidence is labeled/source-trusted footprint
    coverage.  Query-independent field reliability remains a separate input.
    This factorization preserves the original neutral element exactly: rows
    with no registered observation have zero affinities and zero coverage.
    """

    if observation.confidence is None:
        raise ValueError("registered 2D likelihood requires explicit confidence")
    confidence = torch.as_tensor(observation.confidence).detach().float().reshape(-1)
    prior = torch.as_tensor(prior_probability).detach().float().reshape(-1)
    if prior.shape != confidence.shape:
        raise ValueError("registered observation and field prior must align")
    observed = confidence > 0
    foreground = observation.foreground_probability.detach().float()
    positive = torch.where(observed, foreground, torch.zeros_like(foreground))
    negative = torch.where(
        observed, 1.0 - foreground, torch.zeros_like(foreground)
    )
    field_reliability = (
        torch.ones_like(confidence)
        if reliability is None
        else torch.as_tensor(reliability).detach().float().reshape(-1)
    )
    return QueryLikelihoodInputs(
        positive_affinity=positive[:, None],
        negative_affinity=negative[:, None],
        prior_probability=prior,
        coverage=confidence,
        reliability=field_reliability,
    ).validated()


def _balanced_source_reconstruction_loss(
    probability: torch.Tensor,
    positive_mass: torch.Tensor,
    negative_mass: torch.Tensor,
) -> torch.Tensor:
    q = torch.as_tensor(probability).float().reshape(-1).clamp(1e-6, 1.0 - 1e-6)
    positive = torch.as_tensor(positive_mass).float().reshape(-1)
    negative = torch.as_tensor(negative_mass).float().reshape(-1)
    if q.shape != positive.shape or q.shape != negative.shape:
        raise ValueError("source reconstruction tensors must align")
    if not bool(torch.isfinite(positive).all()) or not bool(
        torch.isfinite(negative).all()
    ):
        raise ValueError("source reconstruction masses must be finite")
    if bool((positive < 0).any()) or bool((negative < 0).any()):
        raise ValueError("source reconstruction masses must be non-negative")
    positive_total = positive.sum()
    negative_total = negative.sum()
    if float(positive_total) <= 0 or float(negative_total) <= 0:
        raise ValueError("source reconstruction requires both prompt signs")
    positive_loss = -(positive * q.log()).sum() / positive_total
    negative_loss = -(negative * torch.log1p(-q)).sum() / negative_total
    return 0.5 * (positive_loss + negative_loss)


def fit_registered_2d_source_reconstruction_head(
    observations: QueryLikelihoodInputs,
    *,
    positive_reference_mass: torch.Tensor,
    negative_reference_mass: torch.Tensor,
) -> tuple[MonotoneQueryLikelihoodHead, Mapping[str, object]]:
    """Fit the six-parameter head from the legal reference query only.

    The target is not a target-view mask.  It is the same positive/negative
    reference raster that registered the query, accumulated by the same
    responsibility operator.  Equal sign losses prevent a full background
    mask from overwhelming a sparse foreground scribble.  The recipe is
    fixed here rather than exposed as benchmark-tunable CLI arguments.
    """

    inputs = observations.validated()
    positive = torch.as_tensor(positive_reference_mass).detach().float().reshape(-1)
    negative = torch.as_tensor(negative_reference_mass).detach().float().reshape(-1)
    rows = inputs.coverage.numel()
    if positive.shape != (rows,) or negative.shape != (rows,):
        raise ValueError("reference masses must align with likelihood rows")
    calibration = (positive + negative) > 0
    if not bool(calibration.any()):
        raise ValueError("reference prompt has no registered calibration mass")

    fit_inputs = QueryLikelihoodInputs(
        positive_affinity=inputs.positive_affinity[calibration],
        negative_affinity=inputs.negative_affinity[calibration],
        prior_probability=inputs.prior_probability[calibration],
        coverage=inputs.coverage[calibration],
        reliability=inputs.reliability[calibration],
    )
    fit_positive = positive[calibration]
    fit_negative = negative[calibration]
    torch.manual_seed(0)
    head = MonotoneQueryLikelihoodHead().cpu()

    def evaluate_loss() -> torch.Tensor:
        evidence = head(fit_inputs, source="registered_2d_source_reconstruction")
        return _balanced_source_reconstruction_loss(
            evidence.foreground_probability,
            fit_positive,
            fit_negative,
        )

    with torch.no_grad():
        initial_loss = float(evaluate_loss())
    optimizer = torch.optim.Adam(
        head.parameters(),
        lr=float(REGISTERED_2D_SOURCE_RECONSTRUCTION_RECIPE["learning_rate"]),
        weight_decay=float(
            REGISTERED_2D_SOURCE_RECONSTRUCTION_RECIPE["weight_decay"]
        ),
    )
    losses: list[float] = []
    for _ in range(int(REGISTERED_2D_SOURCE_RECONSTRUCTION_RECIPE["steps"])):
        optimizer.zero_grad(set_to_none=True)
        loss = evaluate_loss()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    head.eval()
    with torch.no_grad():
        fitted = head(fit_inputs, source="registered_2d_source_reconstruction")
        final_loss = float(
            _balanced_source_reconstruction_loss(
                fitted.foreground_probability,
                fit_positive,
                fit_negative,
            )
        )
        positive_mean = float(
            (fitted.foreground_probability * fit_positive).sum()
            / fit_positive.sum()
        )
        negative_mean = float(
            (fitted.foreground_probability * fit_negative).sum()
            / fit_negative.sum()
        )
    if not final_loss < initial_loss or not positive_mean > negative_mean:
        raise RuntimeError("registered source reconstruction calibration failed")
    return head, {
        "schema_version": 1,
        "recipe": dict(REGISTERED_2D_SOURCE_RECONSTRUCTION_RECIPE),
        "calibration_rows": int(calibration.sum()),
        "positive_reference_mass": float(positive.sum()),
        "negative_reference_mass": float(negative.sum()),
        "initial_balanced_bce": initial_loss,
        "final_balanced_bce": final_loss,
        "positive_mass_weighted_probability": positive_mean,
        "negative_mass_weighted_probability": negative_mean,
        "loss_trace_sha256": __import__("hashlib").sha256(
            torch.tensor(losses, dtype=torch.float64).numpy().tobytes()
        ).hexdigest(),
        "parameters": {
            "bias": float(head.bias.detach()),
            "positive_weights": F.softplus(
                head.raw_positive_weights.detach()
            ).tolist(),
            "negative_weights": F.softplus(
                head.raw_negative_weights.detach()
            ).tolist(),
            "prior_weight": float(F.softplus(head.raw_prior_weight.detach())),
        },
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_opened": False,
    }
