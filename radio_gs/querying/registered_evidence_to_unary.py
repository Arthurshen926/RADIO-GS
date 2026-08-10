"""Global prompt-conditioned, graph-off primitive unary calibration.

The module deliberately separates *registered source evidence* from the
held-out target supervision used by a trainer.  The feature builder accepts
only source-raster sufficient statistics and frozen query-free primitive
capabilities.  In particular, no target image, target mask, scene identifier,
or per-scene parameter enters :func:`build_registered_evidence_features`.

``RegisteredEvidenceToUnaryV1`` is a low-capacity bounded residual over a
fixed analytic unary.  Its zero initialization returns that analytic unary
bit-for-bit (up to the explicit probability/logit round trip), and its output
is a continuous primitive probability.  It contains no graph propagation or
connected-component selection.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .anchor_preserving_transport import apply_anchor_preserving_logit_residual


FEATURE_NAMES: tuple[str, ...] = (
    "analytic_logit",
    "registered_signed_observation",
    "labeled_coverage",
    "foreground_purity",
    "dino_positive_negative_margin",
    "sam_positive_negative_margin",
    "directional_dispersion",
    "log_amplitude_std",
    "observation_evidence",
    "visibility_purity_value",
    "visibility_purity_known",
    "log1p_source_visible_mass",
    "log1p_source_view_support",
)


@dataclass(frozen=True)
class RegisteredEvidenceFeatures:
    """Source-only features and the frozen graph-off analytic comparator."""

    values: torch.Tensor
    analytic_probability: torch.Tensor
    registered_probability: torch.Tensor
    labeled_coverage: torch.Tensor
    capability_valid: torch.Tensor


@dataclass(frozen=True)
class RegisteredUnaryOutput:
    """Continuous graph-off primitive prediction."""

    foreground_probability: torch.Tensor
    confidence: torch.Tensor
    abstention: torch.Tensor
    analytic_probability: torch.Tensor
    bounded_logit_residual: torch.Tensor
    residual_gate: torch.Tensor | None = None


def _vector(
    value: torch.Tensor,
    *,
    label: str,
    rows: int | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().reshape(-1).to(dtype=dtype)
    if rows is not None and tensor.shape != (rows,):
        raise ValueError(f"{label} row count differs")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{label} must be finite")
    return tensor


def _nonnegative(value: torch.Tensor, *, label: str, rows: int) -> torch.Tensor:
    tensor = _vector(value, label=label, rows=rows)
    if bool((tensor < 0).any()):
        raise ValueError(f"{label} must be non-negative")
    return tensor


def _probability(value: torch.Tensor, *, label: str, rows: int) -> torch.Tensor:
    tensor = _vector(value, label=label, rows=rows)
    if bool(((tensor < 0) | (tensor > 1)).any()):
        raise ValueError(f"{label} must be in [0,1]")
    return tensor


def build_registered_evidence_features(
    *,
    foreground_mass: torch.Tensor,
    background_mass: torch.Tensor,
    visible_mass: torch.Tensor,
    dino_margin: torch.Tensor,
    sam_margin: torch.Tensor,
    directional_dispersion: torch.Tensor,
    log_amplitude_std: torch.Tensor,
    observation_evidence: torch.Tensor,
    visibility_purity_value: torch.Tensor,
    visibility_purity_known: torch.Tensor,
    capability_valid: torch.Tensor,
    source_view_support: torch.Tensor | None = None,
    semantic_logit_scale: float = 4.0,
    eps: float = 1e-6,
) -> RegisteredEvidenceFeatures:
    """Build the fixed V1 source-evidence feature vector.

    The analytic comparator is a coverage-preserving mixture.  Frozen
    capability margins supply unobserved rows, while registered source purity
    replaces them in proportion to labeled footprint coverage::

        p_sem = sigmoid(4 * (m_dino + m_sam))
        p_base = coverage * p_registered + (1-coverage) * p_sem

    Invalid capability rows use the neutral semantic prior 0.5.  Unlabeled
    source pixels therefore never become background evidence.
    """

    foreground = torch.as_tensor(foreground_mass).detach().reshape(-1).float()
    if foreground.numel() == 0:
        raise ValueError("registered evidence must contain at least one row")
    rows = int(foreground.numel())
    foreground = _nonnegative(foreground, label="foreground_mass", rows=rows)
    background = _nonnegative(background_mass, label="background_mass", rows=rows)
    visible = _nonnegative(visible_mass, label="visible_mass", rows=rows)
    labeled = foreground + background
    tolerance = 1e-5 * torch.maximum(torch.ones_like(visible), visible)
    if bool((labeled > visible + tolerance).any()):
        raise ValueError("registered labeled mass exceeds source visible mass")

    valid = torch.as_tensor(capability_valid).detach().reshape(-1).bool()
    if valid.shape != (rows,):
        raise ValueError("capability_valid row count differs")
    dino = _vector(dino_margin, label="dino_margin", rows=rows)
    sam = _vector(sam_margin, label="sam_margin", rows=rows)
    dispersion = _nonnegative(
        directional_dispersion, label="directional_dispersion", rows=rows
    )
    amplitude_std = _nonnegative(
        log_amplitude_std, label="log_amplitude_std", rows=rows
    )
    evidence = _nonnegative(
        observation_evidence, label="observation_evidence", rows=rows
    )
    purity = _probability(
        visibility_purity_value, label="visibility_purity_value", rows=rows
    )
    purity_known = torch.as_tensor(visibility_purity_known).detach().reshape(-1).bool()
    if purity_known.shape != (rows,):
        raise ValueError("visibility_purity_known row count differs")
    if source_view_support is None:
        view_support = (visible > 0).float()
    else:
        view_support = _nonnegative(
            source_view_support, label="source_view_support", rows=rows
        )

    safe_visible = visible.clamp_min(float(eps))
    observed = visible > 0
    coverage = torch.where(observed, labeled / safe_visible, torch.zeros_like(visible))
    coverage = coverage.clamp(0.0, 1.0)
    foreground_purity = torch.where(
        labeled > 0,
        foreground / labeled.clamp_min(float(eps)),
        torch.full_like(labeled, 0.5),
    ).clamp(0.0, 1.0)
    signed = coverage * (2.0 * foreground_purity - 1.0)

    if not math.isfinite(float(semantic_logit_scale)) or semantic_logit_scale <= 0:
        raise ValueError("semantic_logit_scale must be finite and positive")
    semantic_logit = float(semantic_logit_scale) * (dino + sam)
    semantic_probability = torch.sigmoid(semantic_logit)
    semantic_probability = torch.where(
        valid, semantic_probability, torch.full_like(semantic_probability, 0.5)
    )
    analytic_probability = (
        coverage * foreground_purity + (1.0 - coverage) * semantic_probability
    ).clamp(float(eps), 1.0 - float(eps))
    analytic_logit = torch.logit(analytic_probability)

    values = torch.stack(
        (
            analytic_logit,
            signed,
            coverage,
            foreground_purity,
            dino,
            sam,
            torch.log1p(dispersion),
            torch.log1p(amplitude_std),
            torch.log1p(evidence),
            purity,
            purity_known.float(),
            torch.log1p(visible),
            torch.log1p(view_support),
        ),
        dim=-1,
    ).contiguous()
    if values.shape != (rows, len(FEATURE_NAMES)) or not bool(
        torch.isfinite(values).all()
    ):
        raise RuntimeError("registered evidence feature construction failed")
    return RegisteredEvidenceFeatures(
        values=values,
        analytic_probability=analytic_probability.contiguous(),
        registered_probability=foreground_purity.contiguous(),
        labeled_coverage=coverage.contiguous(),
        capability_valid=valid.contiguous(),
    )


class RegisteredEvidenceToUnaryV1(nn.Module):
    """Small globally shared bounded prompt-to-unary residual head."""

    def __init__(
        self,
        *,
        hidden_dim: int = 32,
        max_delta_logit: float = 4.0,
    ) -> None:
        super().__init__()
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if not math.isfinite(float(max_delta_logit)) or max_delta_logit <= 0:
            raise ValueError("max_delta_logit must be finite and positive")
        self.hidden_dim = int(hidden_dim)
        self.max_delta_logit = float(max_delta_logit)
        self.backbone = nn.Sequential(
            nn.Linear(len(FEATURE_NAMES), self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.output = nn.Linear(self.hidden_dim, 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: RegisteredEvidenceFeatures) -> RegisteredUnaryOutput:
        values = features.values
        if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
            raise ValueError("registered evidence values have the wrong shape")
        analytic = features.analytic_probability
        if analytic.shape != values.shape[:1]:
            raise ValueError("analytic probability row count differs")
        raw = self.output(self.backbone(values))
        confidence = torch.sigmoid(raw[:, 1])
        residual = self.max_delta_logit * confidence * torch.tanh(raw[:, 0])
        # Write the bounded update as a delta around an independently computed
        # round trip.  This keeps the zero-initialized output bitwise equal to
        # ``analytic`` without cutting the residual gradient at zero.
        analytic_logit = torch.logit(analytic)
        probability = analytic + (
            torch.sigmoid(analytic_logit + residual) - torch.sigmoid(analytic_logit)
        )
        return RegisteredUnaryOutput(
            foreground_probability=probability,
            confidence=confidence,
            abstention=1.0 - confidence,
            analytic_probability=analytic,
            bounded_logit_residual=residual,
        )


class RegisteredEvidenceToUnaryV2(RegisteredEvidenceToUnaryV1):
    """Observation-clamped residual that cannot rewrite complete evidence.

    The analytic unary already contains the exact registered source evidence.
    V2 therefore gates learned residual capacity by the unlabeled fraction of
    each primitive.  Rows with complete source coverage are strict identity;
    partially observed rows receive a proportional budget, and unobserved rows
    retain the full globally bounded V1 budget.

    This class deliberately does not load V1 checkpoints implicitly.  It is a
    new method family that must be trained and validated on source-only clean
    cohorts before any benchmark authority may name it.
    """

    fully_observed_tolerance = 1e-5

    def forward(self, features: RegisteredEvidenceFeatures) -> RegisteredUnaryOutput:
        values = features.values
        if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
            raise ValueError("registered evidence values have the wrong shape")
        analytic = features.analytic_probability
        coverage = features.labeled_coverage
        if analytic.shape != values.shape[:1] or coverage.shape != values.shape[:1]:
            raise ValueError("registered evidence row counts differ")
        if not bool(torch.isfinite(coverage).all()) or bool(
            ((coverage < 0) | (coverage > 1)).any()
        ):
            raise ValueError("labeled coverage must be finite in [0,1]")

        raw = self.output(self.backbone(values))
        confidence = torch.sigmoid(raw[:, 1])
        proposed_residual = self.max_delta_logit * confidence * torch.tanh(raw[:, 0])
        transported = apply_anchor_preserving_logit_residual(
            analytic,
            proposed_residual,
            coverage,
            active_domain=features.capability_valid,
            max_abs_logit_residual=self.max_delta_logit,
            fully_observed_tolerance=self.fully_observed_tolerance,
        )
        return RegisteredUnaryOutput(
            foreground_probability=transported.probability,
            confidence=confidence,
            abstention=1.0 - confidence,
            analytic_probability=analytic,
            bounded_logit_residual=transported.applied_logit_residual,
            residual_gate=transported.residual_gate,
        )
