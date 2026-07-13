"""Alpha-normalized coefficient rendering for affine canonical fields."""

from __future__ import annotations

import torch

from radio_gs.field.basis_decoder import AffineBasisDecoder


def alpha_normalize(
    premultiplied: torch.Tensor,
    alpha: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Convert premultiplied splat output into a weighted feature average."""

    if alpha.ndim == premultiplied.ndim - 1:
        alpha = alpha.unsqueeze(-1)
    if alpha.ndim != premultiplied.ndim or alpha.shape[:-1] != premultiplied.shape[:-1]:
        raise ValueError("alpha must align with premultiplied feature rows")
    visible = alpha > float(eps)
    return torch.where(
        visible,
        premultiplied / alpha.clamp_min(float(eps)),
        torch.zeros_like(premultiplied),
    )


def normalized_weighted_sum(
    weights: torch.Tensor,
    rows: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference normalized splat for correctness tests and CPU diagnostics."""

    if weights.ndim != 2 or rows.ndim != 2 or weights.shape[1] != rows.shape[0]:
        raise ValueError("weights must be [P,N] and rows [N,D]")
    alpha = weights.sum(dim=-1, keepdim=True)
    premultiplied = weights @ rows
    return alpha_normalize(premultiplied, alpha, eps=eps), alpha


def decode_normalized_coefficients(
    premultiplied_coefficients: torch.Tensor,
    alpha: torch.Tensor,
    decoder: AffineBasisDecoder,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    coefficients = alpha_normalize(premultiplied_coefficients, alpha, eps=eps)
    return decoder(coefficients)


@torch.no_grad()
def affine_commutation_error(
    weights: torch.Tensor,
    coefficients: torch.Tensor,
    decoder: AffineBasisDecoder,
) -> torch.Tensor:
    """Maximum error of decode-after-splat versus splat-after-decode."""

    compact, _alpha = normalized_weighted_sum(weights, coefficients)
    decoded_after = decoder(compact)
    decoded_rows = decoder(coefficients)
    decoded_before, _ = normalized_weighted_sum(weights, decoded_rows)
    return (decoded_after - decoded_before).abs().max()
