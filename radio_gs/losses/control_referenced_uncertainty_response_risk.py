"""Control-referenced exact-hinge risk for fit-only response adaptation.

The validation selector judges *paired changes* from the frozen identity
adapter, whereas the v2 pilot optimized absolute scene-query losses.  This
module makes the fit objective use the same dimensionless paired units:

.. math::

   d_{s,q} = (U_{s,q}(\theta)-U_{s,q}(0)) / \bar U(0).

The primary objective is the global mean of ``d``.  Pre-registered global
CVaR, worst-scene mean/CVaR, and independent-unary constraints are enforced
with an L1 exact-hinge penalty.  Control tensors and validity are detached;
autograd is retained only through the candidate units and candidate unary
loss.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from radio_gs.losses.direct_point_query_logit_distill_loss import (
    fractional_upper_cvar,
)


def compute_control_referenced_exact_hinge_risk(
    candidate_unit_loss: torch.Tensor,
    candidate_valid: torch.Tensor,
    control_unit_loss: torch.Tensor,
    control_valid: torch.Tensor,
    candidate_unary_loss: torch.Tensor,
    control_unary_loss: torch.Tensor | float,
    *,
    cvar_tail_fraction: float = 0.10,
    global_cvar_tolerance: float = 0.005,
    worst_scene_mean_tolerance: float = 0.010,
    worst_scene_cvar_tolerance: float = 0.010,
    unary_delta_tolerance: float = 0.0,
    exact_penalty_weight: float = 1.0,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return paired-mean plus pre-registered exact-hinge constraints.

    ``candidate_unit_loss`` and ``control_unit_loss`` must be aligned ``[S,Q]``
    tensors with identical boolean validity.  Every scene must contain at
    least one valid query.  The unary delta is normalized independently as
    ``candidate_unary_loss / control_unary_loss - 1``.

    The returned statistics are detached audit values.  ``normalized_delta``
    is included so an artifact can replay the optimized fit statistic without
    reconstructing it from rounded scalars.
    """

    if not isinstance(candidate_unit_loss, torch.Tensor) or not isinstance(
        control_unit_loss, torch.Tensor
    ):
        raise TypeError("candidate/control scene-query losses must be tensors")
    if not isinstance(candidate_valid, torch.Tensor) or not isinstance(
        control_valid, torch.Tensor
    ):
        raise TypeError("candidate/control validity must be tensors")
    candidate = candidate_unit_loss
    control = control_unit_loss.detach().to(
        device=candidate.device, dtype=candidate.dtype
    )
    candidate_mask = candidate_valid.detach().to(device=candidate.device)
    control_mask = control_valid.detach().to(device=candidate.device)
    if (
        candidate.ndim != 2
        or candidate.numel() == 0
        or not candidate.is_floating_point()
        or control.shape != candidate.shape
        or candidate_mask.dtype != torch.bool
        or control_mask.dtype != torch.bool
        or candidate_mask.shape != candidate.shape
        or control_mask.shape != candidate.shape
        or not torch.equal(candidate_mask, control_mask)
        or not bool(candidate_mask.any(dim=1).all())
        or not bool(torch.isfinite(candidate).all())
        or not bool(torch.isfinite(control).all())
    ):
        raise ValueError("candidate/control scene-query inputs are invalid")

    if not isinstance(candidate_unary_loss, torch.Tensor):
        raise TypeError("candidate_unary_loss must be a tensor")
    if (
        candidate_unary_loss.numel() != 1
        or not candidate_unary_loss.is_floating_point()
        or not bool(torch.isfinite(candidate_unary_loss).all())
    ):
        raise ValueError("candidate_unary_loss must be one finite scalar")
    candidate_unary = candidate_unary_loss.reshape(()).to(
        device=candidate.device, dtype=candidate.dtype
    )
    control_unary = torch.as_tensor(
        control_unary_loss, device=candidate.device, dtype=candidate.dtype
    ).detach()
    if (
        control_unary.numel() != 1
        or not bool(torch.isfinite(control_unary).all())
        or float(control_unary) <= float(eps)
    ):
        raise ValueError("control_unary_loss must be one positive finite scalar")
    control_unary = control_unary.reshape(())

    scalars = {
        "cvar_tail_fraction": cvar_tail_fraction,
        "global_cvar_tolerance": global_cvar_tolerance,
        "worst_scene_mean_tolerance": worst_scene_mean_tolerance,
        "worst_scene_cvar_tolerance": worst_scene_cvar_tolerance,
        "unary_delta_tolerance": unary_delta_tolerance,
        "exact_penalty_weight": exact_penalty_weight,
        "eps": eps,
    }
    if any(not math.isfinite(float(value)) for value in scalars.values()):
        raise ValueError("paired-risk scalar parameters must be finite")
    if (
        not 0.0 < float(cvar_tail_fraction) <= 1.0
        or float(global_cvar_tolerance) < 0.0
        or float(worst_scene_mean_tolerance) < 0.0
        or float(worst_scene_cvar_tolerance) < 0.0
        or float(exact_penalty_weight) <= 0.0
        or float(eps) <= 0.0
    ):
        raise ValueError("paired-risk scalar parameters are outside their domain")

    control_scale = control[control_mask].mean()
    if not bool(torch.isfinite(control_scale)) or float(control_scale) <= float(eps):
        raise ValueError("control scene-query mean is degenerate")
    normalized_delta = (candidate - control) / control_scale
    active_delta = normalized_delta[candidate_mask]
    global_mean = active_delta.mean()
    global_cvar = fractional_upper_cvar(active_delta, float(cvar_tail_fraction))

    scene_means: list[torch.Tensor] = []
    scene_cvars: list[torch.Tensor] = []
    for scene_index in range(candidate.shape[0]):
        values = normalized_delta[scene_index][candidate_mask[scene_index]]
        scene_means.append(values.mean())
        scene_cvars.append(fractional_upper_cvar(values, float(cvar_tail_fraction)))
    scene_mean = torch.stack(scene_means)
    scene_cvar = torch.stack(scene_cvars)
    worst_scene_mean = scene_mean.max()
    worst_scene_cvar = scene_cvar.max()
    unary_delta = candidate_unary / control_unary - 1.0

    zero = global_mean.new_zeros(())
    violations = {
        "global_cvar": torch.relu(global_cvar - float(global_cvar_tolerance)),
        "worst_scene_mean": torch.relu(
            worst_scene_mean - float(worst_scene_mean_tolerance)
        ),
        "worst_scene_cvar": torch.relu(
            worst_scene_cvar - float(worst_scene_cvar_tolerance)
        ),
        "independent_unary": torch.relu(unary_delta - float(unary_delta_tolerance)),
    }
    exact_hinge_penalty = sum(violations.values(), zero)
    total = global_mean + float(exact_penalty_weight) * exact_hinge_penalty

    return total, {
        "control_scale_mean_unit_loss": control_scale.detach(),
        "normalized_delta": normalized_delta.detach(),
        "global_mean_delta": global_mean.detach(),
        "global_upper_fractional_cvar_delta": global_cvar.detach(),
        "scene_mean_delta": scene_mean.detach(),
        "scene_upper_fractional_cvar_delta": scene_cvar.detach(),
        "worst_scene_mean_delta": worst_scene_mean.detach(),
        "worst_scene_upper_fractional_cvar_delta": worst_scene_cvar.detach(),
        "independent_unary_delta": unary_delta.detach(),
        "exact_hinge_violations": {
            name: value.detach() for name, value in violations.items()
        },
        "exact_hinge_penalty": exact_hinge_penalty.detach(),
        "objective": total.detach(),
        "valid_scene_query_count": candidate_mask.sum().detach(),
        "scene_count": candidate.new_tensor(candidate.shape[0]).detach(),
    }
