"""Commutative affine basis decoder for canonical RADIO features."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


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
        return torch.matmul(whitened, self.basis)

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
