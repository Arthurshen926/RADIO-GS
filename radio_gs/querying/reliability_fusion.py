"""Target-blind consensus operators for independent primitive posteriors."""

from __future__ import annotations

import math

import torch


# Provenance identifiers are deliberately exact strings: the dual-registration
# engine rejects a generic decoupled prototype query unless its two seed banks
# were produced by these independently frozen operators.
DUAL_PROTOTYPE_SEED_PROVENANCE = (
    "frozen_legacy_alpha_depth_alpha0.02_deterministic_cpu_v1"
)
DUAL_SOLVER_SEED_PROVENANCE = (
    "native_front_to_back_raster_adjoint_alpha0_v1"
)


def symmetric_bernoulli_product_of_experts(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    """Pool two aligned Bernoulli posteriors without a learned gate.

    For foreground probabilities ``p`` and ``q`` the normalized product is

    ``r = p*q / (p*q + (1-p)*(1-q))``.

    This is symmetric, monotone in either expert, and treats ``0.5`` as the
    exact neutral expert.  Equal certain experts remain certain.  The only
    undefined Bernoulli product is an exactly contradictory certain pair
    (``0`` versus ``1``); that pair is mapped to neutral ``0.5`` instead of
    allowing floating-point order to choose a class.

    Inputs must already be calibrated probabilities in the same primitive
    domain.  Values are validated rather than silently clipped.
    """

    left = torch.as_tensor(first)
    right = torch.as_tensor(second)
    if left.shape != right.shape:
        raise ValueError("Bernoulli experts must have matching shapes")
    if left.device != right.device:
        raise ValueError("Bernoulli experts must be on the same device")
    if not left.dtype.is_floating_point or not right.dtype.is_floating_point:
        raise ValueError("Bernoulli experts must have floating-point dtype")
    if not bool(torch.isfinite(left).all()) or not bool(
        torch.isfinite(right).all()
    ):
        raise ValueError("Bernoulli experts contain NaN or infinity")
    if bool((left < 0).any()) or bool((left > 1).any()):
        raise ValueError("first Bernoulli expert must be in [0,1]")
    if bool((right < 0).any()) or bool((right > 1).any()):
        raise ValueError("second Bernoulli expert must be in [0,1]")

    output_dtype = torch.promote_types(left.dtype, right.dtype)
    work_dtype = (
        torch.float64 if output_dtype == torch.float64 else torch.float32
    )
    left_work = left.to(dtype=work_dtype)
    right_work = right.to(dtype=work_dtype)
    foreground = left_work * right_work
    background = (1.0 - left_work) * (1.0 - right_work)
    normalizer = foreground + background
    pooled = torch.where(
        normalizer > 0,
        foreground / normalizer.clamp_min(torch.finfo(work_dtype).tiny),
        torch.full_like(normalizer, 0.5),
    )
    return pooled.to(dtype=output_dtype)


def geometric_consensus_unary(
    field_unary: torch.Tensor,
    raster_adjoint_unary: torch.Tensor,
    *,
    unary_temperature: float,
    chunk_size: int = 262144,
) -> torch.Tensor:
    """Pool prototype and exact-adjoint primitive unaries as Bernoulli experts.

    Both unaries are converted with the solver's *same* temperature before the
    symmetric normalized product.  Thus a zero exact-adjoint unary is the
    neutral posterior ``0.5`` and preserves the field unary bit-for-bit.  For
    finite nonsaturated values the pooled logit is exactly the sum of the two
    input logits; the explicit probability implementation documents and
    stabilizes the Bernoulli boundary behavior.
    """

    field = torch.as_tensor(field_unary)
    direct = torch.as_tensor(raster_adjoint_unary)
    if field.ndim != 1 or direct.shape != field.shape:
        raise ValueError("consensus unaries must align as vectors")
    if field.device != direct.device:
        raise ValueError("consensus unaries must be on the same device")
    if not field.dtype.is_floating_point or not direct.dtype.is_floating_point:
        raise ValueError("consensus unaries must have floating-point dtype")
    if not bool(torch.isfinite(field).all()) or not bool(
        torch.isfinite(direct).all()
    ):
        raise ValueError("consensus unaries contain NaN or infinity")
    if bool((direct < -1).any()) or bool((direct > 1).any()):
        raise ValueError("raster-adjoint unary must be in [-1,1]")
    temperature = float(unary_temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("unary_temperature must be finite and positive")
    if int(chunk_size) <= 0:
        raise ValueError("consensus chunk_size must be positive")

    output_dtype = torch.promote_types(field.dtype, direct.dtype)
    output = torch.empty(
        field.shape,
        dtype=output_dtype,
        device=field.device,
    )
    eps = float(torch.finfo(torch.float64).eps)
    for start in range(0, field.numel(), int(chunk_size)):
        stop = min(start + int(chunk_size), field.numel())
        field_chunk = field[start:stop].double()
        direct_chunk = direct[start:stop].double()
        field_probability = torch.sigmoid(field_chunk / temperature)
        direct_probability = torch.sigmoid(direct_chunk / temperature)
        pooled_probability = symmetric_bernoulli_product_of_experts(
            field_probability,
            direct_probability,
        )
        pooled_unary = temperature * torch.logit(
            pooled_probability.clamp(eps, 1.0 - eps)
        )
        # Avoid a sigmoid/logit round trip on prompt-invisible rows.
        pooled_unary = torch.where(
            direct_chunk == 0,
            field_chunk,
            pooled_unary,
        )
        output[start:stop] = pooled_unary.to(dtype=output_dtype)
    return output
