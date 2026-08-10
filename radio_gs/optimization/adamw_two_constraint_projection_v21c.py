"""Auditable AdamW descent direction and two-halfspace projection for V2.1C.

The vector called ``u`` is a *descent displacement*: parameters are updated as
``theta <- theta - u``.  Consequently the first-order non-regression
constraints for losses with gradients ``g_abs`` and ``g_pair`` are
``g_abs @ u >= 0`` and ``g_pair @ u >= 0``.

The optimizer moments are never projected.  AdamW first advances its moments
and constructs its ordinary candidate displacement (including decoupled
weight decay); only that candidate displacement is projected before it is
committed to the parameters.  This preserves AdamW's state-transition
semantics while making the intervention explicit and reproducible.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import torch

from radio_gs.utils.immutable_artifacts import canonical_json_sha256


GRADIENT_ORDER = ("combined", "absolute", "pairwise")
PARAMETER_SUBSET_SELECTION = "all_trainable_named_parameters_sorted_v1"
FEASIBILITY_ABSOLUTE_TOLERANCE = 1e-10


def trainable_named_parameters(
    model: torch.nn.Module,
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    """Return the fixed lexicographically ordered trainable parameter subset."""

    result = tuple(
        sorted(
            (
                (name, parameter)
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            ),
            key=lambda item: item[0],
        )
    )
    if not result or len({name for name, _ in result}) != len(result):
        raise ValueError("V2.1C requires a nonempty unique trainable subset")
    devices = {parameter.device for _, parameter in result}
    dtypes = {parameter.dtype for _, parameter in result}
    if len(devices) != 1 or len(dtypes) != 1:
        raise ValueError("V2.1C trainable subset must share one device and dtype")
    if not next(iter(dtypes)).is_floating_point:
        raise ValueError("V2.1C trainable subset must be floating point")
    return result


def parameter_subset_manifest(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
) -> dict[str, Any]:
    records = [
        {
            "name": str(name),
            "shape": list(parameter.shape),
            "numel": int(parameter.numel()),
            "dtype": str(parameter.dtype),
        }
        for name, parameter in named_parameters
    ]
    if not records or [item["name"] for item in records] != sorted(
        item["name"] for item in records
    ):
        raise ValueError("V2.1C parameter subset must be sorted and nonempty")
    return {
        "selection": PARAMETER_SUBSET_SELECTION,
        "parameter_count": len(records),
        "vector_numel": sum(int(item["numel"]) for item in records),
        "parameter_records_sha256": canonical_json_sha256(records),
        "parameter_records": records,
    }


def _flat_tensors(
    tensors: Sequence[torch.Tensor | None],
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
) -> torch.Tensor:
    if len(tensors) != len(named_parameters):
        raise ValueError("V2.1C flat tensor axis differs")
    chunks = []
    for value, (_name, parameter) in zip(tensors, named_parameters):
        tensor = torch.zeros_like(parameter) if value is None else value
        if tensor.shape != parameter.shape or tensor.device != parameter.device:
            raise ValueError("V2.1C tensor differs from parameter subset")
        chunks.append(tensor.detach().reshape(-1))
    return torch.cat(chunks)


def flatten_parameter_values(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
) -> torch.Tensor:
    return _flat_tensors(
        [parameter.detach() for _, parameter in named_parameters],
        named_parameters,
    )


def flatten_parameter_gradients(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
) -> torch.Tensor:
    return _flat_tensors(
        [parameter.grad for _, parameter in named_parameters], named_parameters
    )


def objective_gradient(
    loss: torch.Tensor,
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    *,
    retain_graph: bool,
) -> torch.Tensor:
    if loss.ndim != 0 or not bool(torch.isfinite(loss.detach())):
        raise ValueError("V2.1C diagnostic loss must be a finite scalar")
    gradients = torch.autograd.grad(
        loss,
        tuple(parameter for _, parameter in named_parameters),
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return _flat_tensors(gradients, named_parameters)


def gradient_geometry(
    gradients: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    if tuple(gradients) != GRADIENT_ORDER:
        raise ValueError("V2.1C gradient geometry axis differs")
    vectors = [gradients[name].detach().double() for name in GRADIENT_ORDER]
    if len({tuple(vector.shape) for vector in vectors}) != 1:
        raise ValueError("V2.1C gradient vectors must share one shape")
    gram = torch.stack(
        [torch.stack([left @ right for right in vectors]) for left in vectors]
    )
    if not bool(torch.isfinite(gram).all()):
        raise RuntimeError("V2.1C gradient Gram matrix is nonfinite")
    norms = torch.sqrt(torch.diagonal(gram).clamp_min(0.0))
    denominator = norms[:, None] * norms[None, :]
    cosine = torch.where(
        denominator > 0,
        gram / denominator.clamp_min(torch.finfo(torch.float64).tiny),
        torch.zeros_like(gram),
    )
    return {
        "gradient_order": list(GRADIENT_ORDER),
        "gram": gram.cpu().tolist(),
        "cosine": cosine.cpu().tolist(),
        "norm": {
            name: float(norms[index]) for index, name in enumerate(GRADIENT_ORDER)
        },
    }


def _optimizer_group_for_parameter(
    optimizer: torch.optim.Optimizer,
) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) in result:
                raise ValueError("V2.1C optimizer contains a parameter twice")
            result[id(parameter)] = group
    return result


def predict_adamw_descent_direction(
    optimizer: torch.optim.AdamW,
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
) -> torch.Tensor:
    """Predict the next PyTorch AdamW candidate displacement before ``step``."""

    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("V2.1C projection requires torch.optim.AdamW")
    groups = _optimizer_group_for_parameter(optimizer)
    chunks: list[torch.Tensor] = []
    for _name, parameter in named_parameters:
        group = groups.get(id(parameter))
        if group is None:
            raise ValueError("V2.1C parameter is absent from AdamW")
        gradient = parameter.grad
        if gradient is None:
            chunks.append(torch.zeros_like(parameter).reshape(-1))
            continue
        if gradient.is_sparse:
            raise ValueError("V2.1C does not support sparse AdamW gradients")
        if group.get("differentiable", False) or group.get("capturable", False):
            raise ValueError("V2.1C requires ordinary non-capturable AdamW")
        beta1, beta2 = group["betas"]
        eps = float(group["eps"])
        learning_rate = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        effective_gradient = -gradient if group.get("maximize", False) else gradient
        state = optimizer.state.get(parameter, {})
        old_step = state.get("step", 0)
        old_step_value = (
            float(old_step.detach().cpu())
            if torch.is_tensor(old_step)
            else float(old_step)
        )
        next_step = int(old_step_value) + 1
        old_mean = state.get("exp_avg", torch.zeros_like(parameter))
        old_square = state.get("exp_avg_sq", torch.zeros_like(parameter))
        next_mean = old_mean * beta1 + effective_gradient * (1.0 - beta1)
        next_square = old_square * beta2 + effective_gradient.square() * (1.0 - beta2)
        if group.get("amsgrad", False):
            old_maximum = state.get("max_exp_avg_sq", torch.zeros_like(parameter))
            denominator_square = torch.maximum(old_maximum, next_square)
        else:
            denominator_square = next_square
        bias_correction1 = 1.0 - beta1**next_step
        bias_correction2 = 1.0 - beta2**next_step
        adaptive = (
            (learning_rate / bias_correction1)
            * next_mean
            / (denominator_square.sqrt() / math.sqrt(bias_correction2) + eps)
        )
        # AdamW applies weight decay only to parameters with a gradient.
        displacement = learning_rate * weight_decay * parameter.detach() + adaptive
        chunks.append(displacement.detach().reshape(-1))
    result = torch.cat(chunks)
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("V2.1C AdamW candidate displacement is nonfinite")
    return result


def _feasible(
    dots: torch.Tensor,
    *,
    tolerance: float,
) -> bool:
    return bool((dots >= -float(tolerance)).all())


def project_two_halfspaces(
    candidate: torch.Tensor,
    absolute_gradient: torch.Tensor,
    pairwise_gradient: torch.Tensor,
    *,
    tolerance: float = FEASIBILITY_ABSOLUTE_TOLERANCE,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Project ``candidate`` onto two homogeneous gradient halfspaces.

    The active-set enumeration is exact for two constraints.  Singular Gram
    matrices are handled by the Moore-Penrose solution and an explicit primal,
    dual, and stationarity audit.  The all-zero displacement is always a
    feasible fallback, so failure to find a candidate is an implementation
    error rather than an optimization ambiguity.
    """

    u0 = torch.as_tensor(candidate)
    g_abs = torch.as_tensor(absolute_gradient, device=u0.device, dtype=u0.dtype)
    g_pair = torch.as_tensor(pairwise_gradient, device=u0.device, dtype=u0.dtype)
    if (
        u0.ndim != 1
        or g_abs.shape != u0.shape
        or g_pair.shape != u0.shape
        or not u0.is_floating_point()
        or not bool(torch.isfinite(torch.stack((u0, g_abs, g_pair))).all())
        or not math.isfinite(float(tolerance))
        or tolerance < 0
    ):
        raise ValueError("V2.1C projection inputs differ")
    work = torch.stack((g_abs, g_pair)).double()
    base = u0.double()
    initial_dots = work @ base
    gram = work @ work.T

    candidates: list[tuple[float, tuple[int, ...], torch.Tensor, torch.Tensor]] = []
    for active in ((), (0,), (1,), (0, 1)):
        multipliers = torch.zeros(2, dtype=torch.float64, device=u0.device)
        if active:
            index = torch.tensor(active, dtype=torch.int64, device=u0.device)
            matrix = gram.index_select(0, index).index_select(1, index)
            rhs = -initial_dots.index_select(0, index)
            solution = torch.linalg.pinv(matrix, rtol=1e-12, atol=1e-15) @ rhs
            multipliers[index] = solution
        projected = base + work.T @ multipliers
        dots = work @ projected
        stationarity = projected - base - work.T @ multipliers
        active_residual = (
            torch.tensor(0.0, dtype=torch.float64, device=u0.device)
            if not active
            else torch.max(
                torch.abs(
                    dots[
                        torch.tensor(active, dtype=torch.int64, device=u0.device)
                    ]
                )
            )
        )
        if (
            bool((multipliers >= -float(tolerance)).all())
            and _feasible(dots, tolerance=tolerance)
            and float(torch.linalg.vector_norm(stationarity)) <= tolerance
            and float(active_residual) <= max(tolerance, 1e-12)
        ):
            distance2 = float(((projected - base) ** 2).sum())
            candidates.append((distance2, active, projected, multipliers))

    # In very ill-conditioned cases, u=0 remains an exact feasible point but
    # is generally not an active-set optimum.  We deliberately fail rather
    # than silently returning a non-optimal fallback.
    if not candidates:
        raise RuntimeError("V2.1C two-halfspace active-set solver found no KKT point")
    _distance2, active, projected64, multipliers = min(
        candidates, key=lambda item: (item[0], len(item[1]), item[1])
    )
    projected = projected64.to(dtype=u0.dtype)
    final_dots = work @ projected.double()
    stationarity = projected.double() - base - work.T @ multipliers
    complementarity = multipliers * final_dots
    diagnostics = {
        "active_constraints": ["absolute" if item == 0 else "pairwise" for item in active],
        "multipliers": {
            "absolute": float(multipliers[0]),
            "pairwise": float(multipliers[1]),
        },
        "candidate_dot": {
            "absolute": float(initial_dots[0]),
            "pairwise": float(initial_dots[1]),
        },
        "projected_dot": {
            "absolute": float(final_dots[0]),
            "pairwise": float(final_dots[1]),
        },
        "candidate_norm": float(torch.linalg.vector_norm(base)),
        "projected_norm": float(torch.linalg.vector_norm(projected.double())),
        "projection_distance": float(
            torch.linalg.vector_norm(projected.double() - base)
        ),
        "kkt": {
            "primal_minimum": float(final_dots.min()),
            "dual_minimum": float(multipliers.min()),
            "stationarity_norm": float(torch.linalg.vector_norm(stationarity)),
            "complementarity_max_abs": float(torch.abs(complementarity).max()),
            "passed": bool(
                _feasible(final_dots, tolerance=max(tolerance, 2e-7))
                and bool((multipliers >= -max(tolerance, 2e-7)).all())
                and float(torch.linalg.vector_norm(stationarity)) <= 2e-7
                and float(torch.abs(complementarity).max()) <= 2e-7
            ),
        },
    }
    if diagnostics["kkt"]["passed"] is not True:
        raise RuntimeError("V2.1C projected displacement failed KKT audit")
    return projected, diagnostics


@torch.no_grad()
def commit_projected_adamw_step(
    optimizer: torch.optim.AdamW,
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    absolute_gradient: torch.Tensor,
    pairwise_gradient: torch.Tensor,
) -> dict[str, Any]:
    """Advance AdamW moments, project its candidate, and commit the projection."""

    before = flatten_parameter_values(named_parameters).clone()
    predicted = predict_adamw_descent_direction(optimizer, named_parameters)
    optimizer.step()
    ordinary_after = flatten_parameter_values(named_parameters)
    actual = before - ordinary_after
    reconstruction = float((predicted - actual).abs().max().detach().cpu())
    projected, projection = project_two_halfspaces(
        actual,
        absolute_gradient,
        pairwise_gradient,
    )
    offset = 0
    for _name, parameter in named_parameters:
        count = parameter.numel()
        replacement = (before[offset : offset + count] - projected[offset : offset + count]).view_as(parameter)
        parameter.copy_(replacement)
        offset += count
    if offset != projected.numel():
        raise RuntimeError("V2.1C projected commit consumed another vector axis")
    return {
        "adamw_candidate_reconstruction_max_abs_error": reconstruction,
        "adamw_moments_advanced_before_projection": True,
        "decoupled_weight_decay_in_candidate": True,
        **projection,
    }


__all__ = [
    "FEASIBILITY_ABSOLUTE_TOLERANCE",
    "GRADIENT_ORDER",
    "PARAMETER_SUBSET_SELECTION",
    "commit_projected_adamw_step",
    "flatten_parameter_gradients",
    "flatten_parameter_values",
    "gradient_geometry",
    "objective_gradient",
    "parameter_subset_manifest",
    "predict_adamw_descent_direction",
    "project_two_halfspaces",
    "trainable_named_parameters",
]
