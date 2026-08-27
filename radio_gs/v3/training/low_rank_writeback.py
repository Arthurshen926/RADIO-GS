"""Low-rank instance writeback that folds into the sole canonical D512."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GradientProjectionReport:
    dot_product: float
    cosine: float
    conflict: bool


class LowRankWritebackArm(nn.Module):
    """Train an N×r residual and fold it into one N×512 deployment field.

    The N×r codes exist only as an optimization parameterization.  They and
    the global r×512 basis are never a deployment sidecar: ``folded_latent``
    is the only Gaussian-indexed high-dimensional output.
    """

    deployment_eligible = True

    def __init__(
        self,
        latent: torch.Tensor,
        *,
        radio_basis: torch.Tensor,
        radio_mean: torch.Tensor,
        radio_scale: torch.Tensor,
        rank: int = 16,
        output_dim: int = 32,
        seed: int = 20260826,
    ) -> None:
        super().__init__()
        base = torch.as_tensor(latent).detach().float()
        basis = torch.as_tensor(radio_basis).detach().float()
        mean = torch.as_tensor(radio_mean).detach().float()
        scale = torch.as_tensor(radio_scale).detach().float()
        if base.ndim != 2 or base.shape[1] != 512:
            raise ValueError("low-rank writeback requires canonical D512")
        if basis.ndim != 2 or basis.shape[1] != 512:
            raise ValueError("RADIO decoder basis must be [F,512]")
        if mean.shape != (basis.shape[0],) or scale.shape != mean.shape:
            raise ValueError("RADIO decoder statistics differ")
        if not 0 < int(rank) <= 64 or int(output_dim) <= 0:
            raise ValueError("low-rank writeback dimensions differ")
        if bool((scale <= 0).any()):
            raise ValueError("RADIO decoder scale must be positive")

        self.register_buffer("base_latent", base, persistent=False)
        self.register_buffer("radio_basis", basis, persistent=False)
        self.register_buffer("radio_mean", mean, persistent=False)
        self.register_buffer("radio_scale", scale, persistent=False)
        self.residual_codes = nn.Parameter(torch.zeros(base.shape[0], int(rank)))
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        initial_basis = torch.randn(int(rank), 512, generator=generator) / 512**0.5
        self.residual_basis = nn.Parameter(initial_basis)
        self.projection = nn.Linear(512, int(output_dim), bias=False)
        self.scale_adapter = nn.Linear(2, 2 * int(output_dim))

    def coefficients(self, rows: torch.Tensor | None = None) -> torch.Tensor:
        if rows is None:
            return self.base_latent + self.residual_codes @ self.residual_basis
        indices = torch.as_tensor(rows, device=self.base_latent.device, dtype=torch.long)
        return self.base_latent[indices] + self.residual_codes[indices] @ self.residual_basis

    def projected_latent(self, rows: torch.Tensor | None = None) -> torch.Tensor:
        if rows is None:
            base = self.base_latent
            codes = self.residual_codes
        else:
            indices = torch.as_tensor(
                rows, device=self.base_latent.device, dtype=torch.long
            )
            base = self.base_latent[indices]
            codes = self.residual_codes[indices]
        # Exact associativity avoids retaining an N×512 residual graph for
        # every mask episode: P(z + UB) == Pz + U(BP^T).  The persistent
        # folded field remains unchanged.
        base_projected = F.linear(base, self.projection.weight)
        residual_projection = self.residual_basis @ self.projection.weight.T
        return base_projected + codes @ residual_projection

    def scale_embedding(
        self, projected: torch.Tensor, scale: float = 0.5
    ) -> torch.Tensor:
        phase = projected.new_tensor([float(scale)]).clamp(0, 1) * torch.pi
        gamma, beta = self.scale_adapter(
            torch.cat((phase.sin(), phase.cos()))
        ).chunk(2)
        value = projected * (1 + 0.1 * gamma.tanh()) + 0.1 * beta
        return F.normalize(value, dim=-1, eps=1e-8)

    def forward(
        self, scale: float = 0.5, rows: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.scale_embedding(self.projected_latent(rows), scale)

    def _decode_radio(self, coefficients: torch.Tensor) -> torch.Tensor:
        return self.radio_mean + (coefficients @ self.radio_basis.T) * self.radio_scale

    def decode_radio(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Decode through the frozen canonical RADIO affine basis."""

        return self._decode_radio(coefficients)

    def radio_anchor_loss(self, rows: torch.Tensor) -> torch.Tensor:
        indices = torch.as_tensor(rows, device=self.base_latent.device, dtype=torch.long)
        before = self._decode_radio(self.base_latent[indices]).detach()
        after = self._decode_radio(self.coefficients(indices))
        return (1.0 - F.cosine_similarity(after, before, dim=-1, eps=1e-8)).mean()

    @torch.no_grad()
    def radio_cosine(self, rows: torch.Tensor) -> torch.Tensor:
        indices = torch.as_tensor(rows, device=self.base_latent.device, dtype=torch.long)
        before = self._decode_radio(self.base_latent[indices])
        after = self._decode_radio(self.coefficients(indices))
        return F.cosine_similarity(after, before, dim=-1, eps=1e-8)

    @torch.no_grad()
    def folded_latent(self) -> torch.Tensor:
        return self.coefficients().detach().clone()


def pcgrad_backward(
    primary_loss: torch.Tensor,
    anchor_loss: torch.Tensor,
    parameters: Iterable[nn.Parameter],
    *,
    anchor_weight: float = 1.0,
) -> GradientProjectionReport:
    """Project the primary gradient off a conflicting frozen-visual anchor."""

    values = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if not values or anchor_weight < 0:
        raise ValueError("PCGrad parameters or anchor weight differ")
    primary = torch.autograd.grad(
        primary_loss, values, retain_graph=True, allow_unused=True
    )
    anchor = torch.autograd.grad(anchor_loss, values, allow_unused=True)
    paired = [
        (left, right)
        for left, right in zip(primary, anchor)
        if left is not None and right is not None
    ]
    if paired:
        # ``left * right`` materializes another full N×512 tensor for joint
        # mapping. BLAS dot products and in-place projection preserve exact
        # PCGrad semantics without that transient allocation.
        dot = sum(torch.dot(left.reshape(-1), right.reshape(-1)) for left, right in paired)
        primary_norm = sum(
            torch.dot(left.reshape(-1), left.reshape(-1)) for left, _ in paired
        ).sqrt()
        anchor_norm_squared = sum(
            torch.dot(right.reshape(-1), right.reshape(-1)) for _, right in paired
        )
        cosine = dot / (primary_norm * anchor_norm_squared.sqrt()).clamp_min(1e-12)
        conflict = bool(dot.detach() < 0)
        coefficient = (dot / anchor_norm_squared.clamp_min(1e-12)) if conflict else dot.new_zeros(())
    else:
        reference = primary_loss.detach()
        dot = reference.new_zeros(())
        cosine = reference.new_zeros(())
        conflict = False
        coefficient = reference.new_zeros(())
    for parameter, left, right in zip(values, primary, anchor):
        gradient = None if left is None else left
        if gradient is not None and right is not None and conflict:
            gradient.add_(right, alpha=-float(coefficient.detach()))
        if right is not None and anchor_weight:
            if gradient is None:
                gradient = right.detach().clone().mul_(float(anchor_weight))
            else:
                gradient.add_(right, alpha=float(anchor_weight))
        parameter.grad = None if gradient is None else gradient.detach()
    return GradientProjectionReport(
        dot_product=float(dot.detach()),
        cosine=float(cosine.detach()),
        conflict=conflict,
    )


def pcgrad_backward_sparse_anchor(
    primary_loss: torch.Tensor,
    anchor_loss: torch.Tensor,
    parameters: Iterable[nn.Parameter],
    *,
    anchor_parameter: nn.Parameter,
    anchor_rows: torch.Tensor,
    anchor_values: torch.Tensor,
    anchor_weight: float = 1.0,
) -> GradientProjectionReport:
    """Exact PCGrad when an anchor touches only selected rows of one table.

    Autograd normally expands an indexed anchor into a second dense gradient the
    size of ``anchor_parameter``.  Differentiating an explicit leaf row block and
    scattering it into the primary gradient is algebraically identical and keeps
    the transient anchor memory proportional to the sampled row budget.
    """

    values = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if not any(parameter is anchor_parameter for parameter in values) or anchor_weight < 0:
        raise ValueError("sparse PCGrad parameter or anchor weight differs")
    rows = torch.as_tensor(
        anchor_rows, device=anchor_parameter.device, dtype=torch.long
    )
    if rows.ndim != 1 or rows.numel() != torch.unique(rows).numel():
        raise ValueError("sparse PCGrad anchor rows must be unique")
    primary = torch.autograd.grad(primary_loss, values, allow_unused=True)
    (anchor,) = torch.autograd.grad(anchor_loss, (anchor_values,))
    parameter_index = next(
        index for index, parameter in enumerate(values) if parameter is anchor_parameter
    )
    table_primary = primary[parameter_index]
    if table_primary is None:
        dot = anchor.new_zeros(())
        primary_norm = anchor.new_zeros(())
    else:
        dot = torch.dot(table_primary[rows].reshape(-1), anchor.reshape(-1))
        primary_norm = torch.dot(
            table_primary.reshape(-1), table_primary.reshape(-1)
        ).sqrt()
    anchor_norm_squared = torch.dot(anchor.reshape(-1), anchor.reshape(-1))
    cosine = dot / (primary_norm * anchor_norm_squared.sqrt()).clamp_min(1e-12)
    conflict = bool(dot.detach() < 0)
    coefficient = (
        dot / anchor_norm_squared.clamp_min(1e-12)
        if conflict
        else dot.new_zeros(())
    )
    for index, (parameter, gradient) in enumerate(zip(values, primary)):
        if gradient is None:
            parameter.grad = None
            continue
        if index == parameter_index:
            if conflict:
                gradient.index_add_(
                    0, rows, anchor, alpha=-float(coefficient.detach())
                )
            if anchor_weight:
                gradient.index_add_(0, rows, anchor, alpha=float(anchor_weight))
        parameter.grad = gradient.detach()
    return GradientProjectionReport(
        dot_product=float(dot.detach()),
        cosine=float(cosine.detach()),
        conflict=conflict,
    )


__all__ = [
    "GradientProjectionReport",
    "LowRankWritebackArm",
    "pcgrad_backward",
    "pcgrad_backward_sparse_anchor",
]
