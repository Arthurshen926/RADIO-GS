"""Commutative affine basis decoder for canonical RADIO features."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


BASIS_CONDITIONING_CONTRACT_VERSION = "affine-basis-conditioning-v1"
MAXIMUM_BASIS_CONDITION_NUMBER_V1 = 1_000_000.0
_BASIS_RANK_TOLERANCE_SEMANTICS = (
    "max(feature_dim,coefficient_dim)*float64_eps*largest_singular_value"
)


def _validate_feature_matrix(features: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(features)
    if values.ndim != 2 or min(values.shape) <= 0:
        raise ValueError(f"teacher features must be [N,C], got {tuple(values.shape)}")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("teacher features contain NaN or infinity")
    return values.float()


@dataclass(frozen=True)
class BasisFitReport:
    coefficient_dim: int
    feature_dim: int
    sample_count: int
    explained_variance_ratio: float
    reconstruction_cosine: float


@dataclass(frozen=True)
class BasisConditioningReport:
    """Numerical authority for one affine decoder basis.

    Diagnostics are computed in float64 on CPU so checkpoint acceptance does
    not depend on the requested inference device or mixed-precision policy.
    The v1 condition-number ceiling is deliberately permissive enough for the
    existing trained canonical assets while still rejecting effectively
    singular decoder coordinates before they reach an inverse or readout.
    """

    contract_version: str
    feature_dim: int
    coefficient_dim: int
    numerical_rank: int
    largest_singular_value: float
    smallest_singular_value: float
    rank_tolerance: float
    rank_tolerance_semantics: str
    condition_number: float
    maximum_condition_number: float

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "contract_version": self.contract_version,
            "feature_dim": self.feature_dim,
            "coefficient_dim": self.coefficient_dim,
            "numerical_rank": self.numerical_rank,
            "largest_singular_value": self.largest_singular_value,
            "smallest_singular_value": self.smallest_singular_value,
            "rank_tolerance": self.rank_tolerance,
            "rank_tolerance_semantics": self.rank_tolerance_semantics,
            "condition_number": self.condition_number,
            "maximum_condition_number": self.maximum_condition_number,
        }


@torch.no_grad()
def basis_conditioning_report(
    basis: torch.Tensor,
    *,
    maximum_condition_number: float = MAXIMUM_BASIS_CONDITION_NUMBER_V1,
) -> BasisConditioningReport:
    """Return deterministic rank and condition diagnostics for ``basis``."""

    values = torch.as_tensor(basis)
    if values.ndim != 2 or min(values.shape) <= 0:
        raise ValueError("basis conditioning requires a non-empty matrix")
    feature_dim, coefficient_dim = map(int, values.shape)
    if coefficient_dim > feature_dim:
        raise ValueError(
            "basis conditioning requires coefficient_dim <= feature_dim"
        )
    if not values.dtype.is_floating_point or not bool(
        torch.isfinite(values).all()
    ):
        raise ValueError("basis conditioning requires finite floating-point values")
    bound = float(maximum_condition_number)
    if not math.isfinite(bound) or bound < 1.0:
        raise ValueError("maximum basis condition number must be finite and >= 1")

    diagnostic = values.detach().to(device="cpu", dtype=torch.float64)
    try:
        singular_values = torch.linalg.svdvals(diagnostic)
    except RuntimeError as error:
        raise ValueError("basis singular-value diagnostics failed") from error
    largest = float(singular_values[0])
    smallest = float(singular_values[-1])
    tolerance = (
        max(feature_dim, coefficient_dim)
        * torch.finfo(torch.float64).eps
        * largest
    )
    numerical_rank = int((singular_values > tolerance).sum())
    condition_number = (
        largest / smallest if smallest > 0.0 else float("inf")
    )
    return BasisConditioningReport(
        contract_version=BASIS_CONDITIONING_CONTRACT_VERSION,
        feature_dim=feature_dim,
        coefficient_dim=coefficient_dim,
        numerical_rank=numerical_rank,
        largest_singular_value=largest,
        smallest_singular_value=smallest,
        rank_tolerance=float(tolerance),
        rank_tolerance_semantics=_BASIS_RANK_TOLERANCE_SEMANTICS,
        condition_number=float(condition_number),
        maximum_condition_number=bound,
    )


@torch.no_grad()
def validate_basis_conditioning(
    basis: torch.Tensor,
    *,
    maximum_condition_number: float = MAXIMUM_BASIS_CONDITION_NUMBER_V1,
) -> BasisConditioningReport:
    """Fail closed when an affine basis is rank deficient or ill-conditioned."""

    report = basis_conditioning_report(
        basis,
        maximum_condition_number=maximum_condition_number,
    )
    if report.numerical_rank != report.coefficient_dim:
        raise ValueError(
            f"basis is rank deficient under {report.contract_version}: "
            f"rank {report.numerical_rank} != {report.coefficient_dim}"
        )
    if (
        not math.isfinite(report.condition_number)
        or report.condition_number > report.maximum_condition_number
    ):
        raise ValueError(
            f"basis condition number {report.condition_number:.9g} exceeds "
            f"{report.contract_version} maximum "
            f"{report.maximum_condition_number:.9g}"
        )
    return report


class AffineBasisDecoder(nn.Module):
    """Decode coefficients using ``mu + sigma * (a @ B.T)``.

    The decoder has no normalization, activation, token mixing, or spatial
    state.  It is therefore point/batch invariant and exactly commutes with an
    alpha-normalized weighted average.
    """

    def __init__(
        self,
        feature_dim: int = 1280,
        coefficient_dim: int = 128,
        *,
        mean: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        basis: torch.Tensor | None = None,
        trainable_basis: bool = True,
        trainable_statistics: bool = False,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.coefficient_dim = int(coefficient_dim)
        if self.feature_dim <= 0 or self.coefficient_dim <= 0:
            raise ValueError("feature_dim and coefficient_dim must be positive")
        if self.coefficient_dim > self.feature_dim:
            raise ValueError("coefficient_dim cannot exceed feature_dim")

        mean_value = torch.zeros(self.feature_dim) if mean is None else torch.as_tensor(mean).float()
        scale_value = torch.ones(self.feature_dim) if scale is None else torch.as_tensor(scale).float()
        if mean_value.shape != (self.feature_dim,) or scale_value.shape != (self.feature_dim,):
            raise ValueError("mean/scale must have shape [feature_dim]")
        if bool((scale_value <= 0).any()):
            raise ValueError("scale must be strictly positive")

        if basis is None:
            basis_value = torch.empty(self.feature_dim, self.coefficient_dim)
            nn.init.orthogonal_(basis_value)
        else:
            basis_value = torch.as_tensor(basis).float()
        if basis_value.shape != (self.feature_dim, self.coefficient_dim):
            raise ValueError("basis must have shape [feature_dim,coefficient_dim]")
        if basis is not None:
            validate_basis_conditioning(basis_value)

        if trainable_statistics:
            self.mean = nn.Parameter(mean_value.clone())
            self.log_scale = nn.Parameter(scale_value.log())
        else:
            self.register_buffer("mean", mean_value.clone())
            self.register_buffer("log_scale", scale_value.log())
        self.basis = nn.Parameter(basis_value.clone(), requires_grad=trainable_basis)

    @property
    def scale(self) -> torch.Tensor:
        return self.log_scale.exp()

    def forward(self, coefficients: torch.Tensor) -> torch.Tensor:
        if coefficients.shape[-1] != self.coefficient_dim:
            raise ValueError(
                f"expected coefficient dim {self.coefficient_dim}, got {coefficients.shape[-1]}"
            )
        whitened = torch.matmul(coefficients, self.basis.transpose(0, 1))
        return self.mean + whitened * self.scale

    def decode_map(self, coefficient_map: torch.Tensor) -> torch.Tensor:
        """Decode ``[B,d,H,W]`` or ``[d,H,W]`` without token coupling."""

        squeeze = coefficient_map.ndim == 3
        values = coefficient_map.unsqueeze(0) if squeeze else coefficient_map
        if values.ndim != 4 or values.shape[1] != self.coefficient_dim:
            raise ValueError(
                f"coefficient map must be [B,{self.coefficient_dim},H,W], "
                f"got {tuple(values.shape)}"
            )
        tokens = values.permute(0, 2, 3, 1)
        decoded = self(tokens).permute(0, 3, 1, 2).contiguous()
        return decoded[0] if squeeze else decoded

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.feature_dim:
            raise ValueError(f"expected feature dim {self.feature_dim}")
        whitened = (features - self.mean) / self.scale
        return torch.matmul(whitened, self.encoding_projection())

    def encoding_projection(self) -> torch.Tensor:
        """Return the least-squares inverse of the current decoder basis.

        PCA initializes orthonormal columns, but the basis can subsequently be
        trained.  Multiplication by ``basis`` is then no longer the inverse of
        ``coefficients @ basis.T``.  Fail closed on rank/conditioning before
        solving the normal equations; the pseudoinverse remains only as a
        numerical fallback if Cholesky fails for an otherwise accepted basis.
        """

        validate_basis_conditioning(self.basis)
        gram = self.basis.transpose(0, 1) @ self.basis
        cholesky, info = torch.linalg.cholesky_ex(gram)
        if bool((info == 0).all()):
            return torch.cholesky_solve(
                self.basis.transpose(0, 1), cholesky
            ).transpose(0, 1)
        return torch.linalg.pinv(self.basis).transpose(0, 1)

    def orthogonality_loss(self) -> torch.Tensor:
        gram = self.basis.transpose(0, 1) @ self.basis
        identity = torch.eye(
            self.coefficient_dim, dtype=gram.dtype, device=gram.device
        )
        return (gram - identity).square().mean()


@torch.no_grad()
def fit_affine_basis(
    teacher_features: torch.Tensor,
    coefficient_dim: int,
    *,
    standardize: bool = True,
    max_samples: int = 200_000,
    seed: int = 0,
    trainable_basis: bool = True,
) -> tuple[AffineBasisDecoder, BasisFitReport]:
    """Fit a PCA-initialized affine decoder on training-only teacher tokens."""

    values = _validate_feature_matrix(teacher_features)
    if values.shape[0] > int(max_samples):
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        chosen = torch.randperm(values.shape[0], generator=generator)[: int(max_samples)]
        values = values[chosen]
    mean = values.mean(dim=0)
    if standardize:
        scale = values.std(dim=0, unbiased=False).clamp_min(1e-5)
    else:
        scale = torch.ones_like(mean)
    whitened = (values - mean) / scale
    q = min(int(coefficient_dim), whitened.shape[0], whitened.shape[1])
    if q != int(coefficient_dim):
        raise ValueError(
            f"coefficient_dim={coefficient_dim} exceeds PCA rank bound {q}"
        )
    _u, singular, right = torch.pca_lowrank(whitened, q=q, center=False)
    decoder = AffineBasisDecoder(
        feature_dim=values.shape[1],
        coefficient_dim=q,
        mean=mean,
        scale=scale,
        basis=right,
        trainable_basis=trainable_basis,
    )
    coefficients = decoder.encode(values)
    reconstruction = decoder(coefficients)
    cosine = torch.nn.functional.cosine_similarity(
        reconstruction, values, dim=-1, eps=1e-8
    ).mean()
    total_variance = whitened.square().sum().clamp_min(1e-12)
    explained = singular.square().sum() / total_variance
    return decoder, BasisFitReport(
        coefficient_dim=q,
        feature_dim=values.shape[1],
        sample_count=values.shape[0],
        explained_variance_ratio=float(explained),
        reconstruction_cosine=float(cosine),
    )
