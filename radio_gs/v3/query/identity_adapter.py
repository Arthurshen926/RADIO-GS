"""Constant-size source-trained identity alignment for SUGM-v3 queries."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LowRankIdentityAdapter(nn.Module):
    """A bounded residual map in the shared D128 semantic coordinate system."""

    def __init__(self, dimension: int = 128, rank: int = 16) -> None:
        super().__init__()
        if dimension <= 0 or rank <= 0 or rank > dimension:
            raise ValueError("identity adapter dimensions differ")
        self.dimension = int(dimension)
        self.rank = int(rank)
        self.down = nn.Linear(self.dimension, self.rank, bias=False)
        self.up = nn.Linear(self.rank, self.dimension, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        source = F.normalize(torch.as_tensor(value).float(), dim=-1, eps=1e-8)
        residual = self.up(F.gelu(self.down(source)))
        # The fixed bound prevents a small source relation set from replacing
        # the frozen encoder geometry used by image and text packets.
        return F.normalize(source + 0.25 * residual, dim=-1, eps=1e-8)


class OrthogonalTextAlignment(nn.Module):
    """A source-fitted modality alignment that preserves text-space angles."""

    def __init__(self, matrix: torch.Tensor) -> None:
        super().__init__()
        value = torch.as_tensor(matrix).float()
        if value.shape != (128, 128):
            raise ValueError("text alignment axes differ")
        error = value.T @ value - torch.eye(128)
        if float(error.abs().max()) > 1e-4:
            raise ValueError("text alignment is not orthogonal")
        self.register_buffer("matrix", value)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.normalize(torch.as_tensor(value).float() @ self.matrix, dim=-1, eps=1e-8)


class AffineTextAlignment(nn.Module):
    """Regularized source-fitted affine map used only by text packets."""

    def __init__(self, matrix: torch.Tensor, bias: torch.Tensor) -> None:
        super().__init__()
        weight = torch.as_tensor(matrix).float()
        offset = torch.as_tensor(bias).float().reshape(-1)
        if weight.shape != (128, 128) or offset.shape != (128,):
            raise ValueError("affine text alignment axes differ")
        if not bool(torch.isfinite(weight).all()) or not bool(torch.isfinite(offset).all()):
            raise ValueError("affine text alignment contains non-finite values")
        self.register_buffer("matrix", weight)
        self.register_buffer("bias", offset)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        source = torch.as_tensor(value).float()
        return F.normalize(source @ self.matrix + self.bias, dim=-1, eps=1e-8)


class DirectTextProjection(nn.Module):
    """Project raw D1536 text without first discarding image-PCA nullspace."""

    def __init__(self, basis: torch.Tensor) -> None:
        super().__init__()
        value = torch.as_tensor(basis).float()
        if value.shape != (1536, 128) or not bool(torch.isfinite(value).all()):
            raise ValueError("direct text projection axes differ")
        self.register_buffer("basis", value)

    def project_raw(self, value: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        source = torch.as_tensor(value).float().reshape(-1, 1536)
        center = torch.as_tensor(mean, device=source.device).float().reshape(1536)
        return F.normalize((source - center) @ self.basis, dim=-1, eps=1e-8)


__all__ = [
    "AffineTextAlignment",
    "DirectTextProjection",
    "LowRankIdentityAdapter",
    "OrthogonalTextAlignment",
]
