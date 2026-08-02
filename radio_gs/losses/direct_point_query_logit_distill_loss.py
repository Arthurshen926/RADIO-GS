"""Text-query logit distillation for direct primitive summary heads."""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from typing import Optional

import torch
import torch.nn.functional as F


def _validate_cosine_response_inputs(
    student_descriptors: torch.Tensor,
    teacher_descriptors: torch.Tensor,
    frozen_text_bank: torch.Tensor,
) -> None:
    tensors = {
        "student_descriptors": student_descriptors,
        "teacher_descriptors": teacher_descriptors,
        "frozen_text_bank": frozen_text_bank,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim != 2:
            raise ValueError(f"{name} must have rank 2, got shape {tuple(tensor.shape)}")
        if not tensor.is_floating_point():
            raise ValueError(f"{name} must have a floating-point dtype")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name} must contain only finite values")

    if student_descriptors.shape != teacher_descriptors.shape:
        raise ValueError(
            "student_descriptors and teacher_descriptors must have the same [B,D] shape, got "
            f"{tuple(student_descriptors.shape)} vs {tuple(teacher_descriptors.shape)}"
        )
    batch_size, descriptor_dim = student_descriptors.shape
    query_count, text_dim = frozen_text_bank.shape
    if batch_size == 0:
        raise ValueError("student_descriptors and teacher_descriptors must have B > 0")
    if query_count == 0:
        raise ValueError("frozen_text_bank must have Q > 0")
    if descriptor_dim == 0:
        raise ValueError("descriptor dimension D must be positive")
    if descriptor_dim != text_dim:
        raise ValueError(
            "descriptor/text dimension mismatch: "
            f"{descriptor_dim} vs {text_dim}"
        )


def compute_independent_normalized_cosine_response_smooth_l1_loss(
    student_descriptors: torch.Tensor,
    teacher_descriptors: torch.Tensor,
    frozen_text_bank: torch.Tensor,
) -> torch.Tensor:
    """Regress independent cosine responses to a frozen text-query bank.

    The returned scalar is the mean SmoothL1 distance between the ``[B,Q]``
    student and teacher cosine-response matrices. Unlike a softmax/KL
    objective, one query's response does not change the target assigned to
    another query. Teacher descriptors and text embeddings are detached so
    gradients flow only through ``student_descriptors``.
    """
    _validate_cosine_response_inputs(
        student_descriptors,
        teacher_descriptors,
        frozen_text_bank,
    )

    student = F.normalize(student_descriptors.float(), dim=-1)
    with torch.no_grad():
        teacher = F.normalize(
            teacher_descriptors.detach().to(device=student.device, dtype=torch.float32),
            dim=-1,
        )
        text = F.normalize(
            frozen_text_bank.detach().to(device=student.device, dtype=torch.float32),
            dim=-1,
        )
        teacher_responses = teacher @ text.T
    student_responses = student @ text.T
    return F.smooth_l1_loss(student_responses, teacher_responses, reduction="mean")


def _scene_group_indices(
    scene_ids: Sequence[Hashable] | torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> list[torch.Tensor]:
    if isinstance(scene_ids, torch.Tensor):
        if scene_ids.ndim != 1 or scene_ids.shape[0] != batch_size:
            raise ValueError(
                f"scene_ids must have shape [{batch_size}], got {tuple(scene_ids.shape)}"
            )
        if scene_ids.is_floating_point() and not torch.isfinite(scene_ids).all().item():
            raise ValueError("scene_ids must contain only finite values")
        labels = scene_ids.detach().cpu().tolist()
    else:
        if isinstance(scene_ids, (str, bytes)) or not isinstance(scene_ids, Sequence):
            raise TypeError("scene_ids must be a one-dimensional tensor or a sequence")
        if len(scene_ids) != batch_size:
            raise ValueError(f"Expected {batch_size} scene_ids, got {len(scene_ids)}")
        labels = list(scene_ids)

    grouped: dict[Hashable, list[int]] = {}
    for row_index, label in enumerate(labels):
        if not isinstance(label, Hashable):
            raise TypeError(f"scene_ids[{row_index}] must be hashable")
        if isinstance(label, float) and not math.isfinite(label):
            raise ValueError("scene_ids must contain only finite values")
        grouped.setdefault(label, []).append(row_index)
    return [
        torch.tensor(indices, dtype=torch.long, device=device)
        for indices in grouped.values()
    ]


def compute_scene_wise_text_response_profile_ranking_loss(
    student_descriptors: torch.Tensor,
    teacher_descriptors: torch.Tensor,
    frozen_text_bank: torch.Tensor,
    scene_ids: Sequence[Hashable] | torch.Tensor,
    *,
    profile_weight: float = 1.0,
    ranking_weight: float = 1.0,
    ranking_temperature: float = 0.1,
    tie_tolerance: float = 1e-6,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Preserve query-response shape and ranking separately inside each scene.

    For every scene and text query, the profile term compares the centered
    student and teacher response vectors by cosine distance. Centering makes
    this term insensitive to a scene/query-wide logit offset. Constant teacher
    profiles are excluded from this angular term because they contain no
    ranking direction.

    The ranking term distills a teacher softmax over the regions in each scene.
    It is listwise, shift invariant, and treats teacher ties as equal target
    probability instead of imposing an arbitrary pair order. Teacher
    descriptors and the text bank are detached; gradients flow only through
    ``student_descriptors``. Scenes containing one row are skipped because no
    within-scene ordering is observable.
    """
    _validate_cosine_response_inputs(
        student_descriptors,
        teacher_descriptors,
        frozen_text_bank,
    )
    scalar_parameters = {
        "profile_weight": profile_weight,
        "ranking_weight": ranking_weight,
        "ranking_temperature": ranking_temperature,
        "tie_tolerance": tie_tolerance,
        "eps": eps,
    }
    for name, value in scalar_parameters.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if profile_weight < 0 or ranking_weight < 0:
        raise ValueError("profile_weight and ranking_weight must be non-negative")
    if profile_weight == 0 and ranking_weight == 0:
        raise ValueError("At least one loss weight must be positive")
    if ranking_temperature <= 0:
        raise ValueError("ranking_temperature must be positive")
    if tie_tolerance < 0:
        raise ValueError("tie_tolerance must be non-negative")
    if eps <= 0:
        raise ValueError("eps must be positive")

    student = F.normalize(student_descriptors.float(), dim=-1)
    with torch.no_grad():
        teacher = F.normalize(
            teacher_descriptors.detach().to(device=student.device, dtype=torch.float32),
            dim=-1,
        )
        text = F.normalize(
            frozen_text_bank.detach().to(device=student.device, dtype=torch.float32),
            dim=-1,
        )
        teacher_responses = teacher @ text.T
    student_responses = student @ text.T
    groups = _scene_group_indices(
        scene_ids,
        batch_size=student_descriptors.shape[0],
        device=student.device,
    )

    zero = student_descriptors.sum() * 0.0
    profile_units: list[torch.Tensor] = []
    ranking_units: list[torch.Tensor] = []
    valid_scene_count = 0
    valid_profile_count = 0
    for indices in groups:
        if indices.numel() < 2:
            continue
        valid_scene_count += 1
        student_scene = student_responses.index_select(0, indices)
        teacher_scene = teacher_responses.index_select(0, indices)

        student_centered = student_scene - student_scene.mean(dim=0, keepdim=True)
        teacher_centered = teacher_scene - teacher_scene.mean(dim=0, keepdim=True)
        with torch.no_grad():
            teacher_span = teacher_scene.amax(dim=0) - teacher_scene.amin(dim=0)
            profile_valid = teacher_span > float(tie_tolerance)
        if profile_valid.any():
            student_profile = F.normalize(
                student_centered[:, profile_valid],
                dim=0,
                eps=float(eps),
            )
            teacher_profile = F.normalize(
                teacher_centered[:, profile_valid],
                dim=0,
                eps=float(eps),
            )
            # Half the squared distance of unit vectors equals cosine distance.
            profile_units.append(
                0.5 * (student_profile - teacher_profile).square().sum(dim=0)
            )
            valid_profile_count += int(profile_valid.sum().item())

        student_prob = F.softmax(
            student_scene.T / float(ranking_temperature),
            dim=-1,
        )
        with torch.no_grad():
            teacher_prob = F.softmax(
                teacher_scene.T / float(ranking_temperature),
                dim=-1,
            )
        # A listwise Brier divergence is exactly zero at equality, remains
        # bounded for sharp targets, and avoids an arbitrary ordering for ties.
        ranking_units.append(
            0.5 * (student_prob - teacher_prob).square().sum(dim=-1)
        )

    profile_loss = torch.cat(profile_units).mean() if profile_units else zero
    ranking_loss = torch.cat(ranking_units).mean() if ranking_units else zero
    loss = float(profile_weight) * profile_loss + float(ranking_weight) * ranking_loss
    query_count = frozen_text_bank.shape[0]
    ranking_unit_count = valid_scene_count * query_count
    return loss, {
        "profile_loss": profile_loss.detach(),
        "ranking_loss": ranking_loss.detach(),
        "valid_scene_count": zero.new_tensor(valid_scene_count).detach(),
        "valid_profile_count": zero.new_tensor(valid_profile_count).detach(),
        "ranking_unit_count": zero.new_tensor(ranking_unit_count).detach(),
    }


def compute_scene_wise_text_response_pairwise_gap_smooth_l1_loss(
    student_descriptors: torch.Tensor,
    teacher_descriptors: torch.Tensor,
    frozen_text_bank: torch.Tensor,
    scene_ids: Sequence[Hashable] | torch.Tensor,
    *,
    tie_tolerance: float = 1e-6,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Distill the complete within-scene region order for every text query.

    For every unordered region pair ``(i,j)`` and query, this loss regresses
    ``response_i-response_j`` after normalizing by the teacher response span
    for that scene/query.  The objective is shift invariant, weights the full
    ordering axis instead of only softmax peaks, and excludes teacher near-ties
    whose order is not identifiable.  Teacher descriptors and text embeddings
    are detached; gradients flow only through the student descriptors.
    """

    _validate_cosine_response_inputs(
        student_descriptors,
        teacher_descriptors,
        frozen_text_bank,
    )
    for name, value in {"tie_tolerance": tie_tolerance, "eps": eps}.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if tie_tolerance < 0:
        raise ValueError("tie_tolerance must be non-negative")
    if eps <= 0:
        raise ValueError("eps must be positive")

    student = F.normalize(student_descriptors.float(), dim=-1)
    with torch.no_grad():
        teacher = F.normalize(
            teacher_descriptors.detach().to(
                device=student.device, dtype=torch.float32
            ),
            dim=-1,
        )
        text = F.normalize(
            frozen_text_bank.detach().to(
                device=student.device, dtype=torch.float32
            ),
            dim=-1,
        )
        teacher_responses = teacher @ text.T
    student_responses = student @ text.T
    groups = _scene_group_indices(
        scene_ids,
        batch_size=student_descriptors.shape[0],
        device=student.device,
    )

    zero = student_descriptors.sum() * 0.0
    losses: list[torch.Tensor] = []
    valid_scene_count = 0
    valid_query_count = 0
    valid_pair_query_count = 0
    for indices in groups:
        region_count = int(indices.numel())
        if region_count < 2:
            continue
        student_scene = student_responses.index_select(0, indices)
        teacher_scene = teacher_responses.index_select(0, indices)
        pairs = torch.triu_indices(
            region_count,
            region_count,
            offset=1,
            device=student.device,
        )
        student_gaps = student_scene[pairs[0]] - student_scene[pairs[1]]
        with torch.no_grad():
            teacher_gaps = teacher_scene[pairs[0]] - teacher_scene[pairs[1]]
            teacher_span = teacher_scene.amax(dim=0) - teacher_scene.amin(dim=0)
            valid_queries = teacher_span > float(tie_tolerance)
            valid = (
                teacher_gaps.abs() > float(tie_tolerance)
            ) & valid_queries.unsqueeze(0)
            scale = teacher_span.clamp_min(float(eps)).unsqueeze(0)
        if not bool(valid.any()):
            continue
        valid_scene_count += 1
        valid_query_count += int(valid_queries.sum().item())
        valid_pair_query_count += int(valid.sum().item())
        normalized_student = student_gaps / scale
        normalized_teacher = teacher_gaps / scale
        losses.append(
            F.smooth_l1_loss(
                normalized_student[valid],
                normalized_teacher[valid],
                reduction="none",
            )
        )

    loss = torch.cat(losses).mean() if losses else zero
    return loss, {
        "valid_scene_count": zero.new_tensor(valid_scene_count).detach(),
        "valid_query_count": zero.new_tensor(valid_query_count).detach(),
        "valid_pair_query_count": zero.new_tensor(
            valid_pair_query_count
        ).detach(),
    }


def compute_multiview_teacher_response_uncertainty(
    teacher_view_descriptors: torch.Tensor,
    teacher_mask: torch.Tensor,
    frozen_text_bank: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Return detached per-region text-response disagreement statistics.

    ``teacher_view_descriptors`` contains the independently re-encoded teacher
    crops for every region.  Their response variance is an unbiased sample
    variance across valid views; it is a deterministic disagreement proxy, not
    a claim that the selected views are IID samples.  At least two valid views
    are required so that a finite variance is identifiable.

    All returned tensors are detached even when a caller accidentally supplies
    differentiable teacher descriptors or text embeddings.
    """

    if not isinstance(teacher_view_descriptors, torch.Tensor):
        raise TypeError("teacher_view_descriptors must be a torch.Tensor")
    if not isinstance(teacher_mask, torch.Tensor):
        raise TypeError("teacher_mask must be a torch.Tensor")
    if not isinstance(frozen_text_bank, torch.Tensor):
        raise TypeError("frozen_text_bank must be a torch.Tensor")
    if teacher_view_descriptors.ndim != 3:
        raise ValueError(
            "teacher_view_descriptors must have shape [B,V,D], got "
            f"{tuple(teacher_view_descriptors.shape)}"
        )
    if not teacher_view_descriptors.is_floating_point():
        raise ValueError("teacher_view_descriptors must have a floating-point dtype")
    if teacher_mask.dtype != torch.bool:
        raise ValueError("teacher_mask must have dtype bool")
    if teacher_mask.shape != teacher_view_descriptors.shape[:2]:
        raise ValueError("teacher_mask must align with the [B,V] teacher views")
    if frozen_text_bank.ndim != 2 or not frozen_text_bank.is_floating_point():
        raise ValueError("frozen_text_bank must be floating point [Q,D]")
    if (
        teacher_view_descriptors.shape[0] == 0
        or teacher_view_descriptors.shape[1] == 0
        or teacher_view_descriptors.shape[2] == 0
        or frozen_text_bank.shape[0] == 0
        or frozen_text_bank.shape[1] != teacher_view_descriptors.shape[2]
    ):
        raise ValueError("teacher views and text bank have empty or mismatched shapes")
    if not bool(torch.isfinite(teacher_view_descriptors).all().item()) or not bool(
        torch.isfinite(frozen_text_bank).all().item()
    ):
        raise ValueError("teacher views and text bank must contain only finite values")
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("eps must be finite and positive")

    with torch.no_grad():
        views = F.normalize(
            teacher_view_descriptors.detach().to(dtype=torch.float32),
            dim=-1,
            eps=float(eps),
        )
        text = F.normalize(
            frozen_text_bank.detach().to(
                device=views.device,
                dtype=torch.float32,
            ),
            dim=-1,
            eps=float(eps),
        )
        mask = teacher_mask.detach().to(device=views.device)
        view_counts = mask.sum(dim=1)
        if not bool((view_counts >= 2).all().item()):
            raise ValueError(
                "multiview response uncertainty requires at least two valid "
                "teacher views per region"
            )
        responses = torch.einsum("bvd,qd->bvq", views, text)
        active = mask[..., None].to(dtype=responses.dtype)
        counts = view_counts.to(dtype=responses.dtype)[..., None]
        response_mean = (responses * active).sum(dim=1) / counts
        centered = responses - response_mean[:, None, :]
        response_variance = (
            centered.square() * active
        ).sum(dim=1) / (counts - 1.0)
        response_standard_error = torch.sqrt(
            response_variance.clamp_min(0.0) / counts
        )

    return {
        "response_mean": response_mean.detach(),
        "response_variance": response_variance.detach(),
        "response_standard_error": response_standard_error.detach(),
        "view_counts": view_counts.detach(),
    }


def compute_scene_wise_uncertainty_weighted_text_response_pairwise_gap_smooth_l1_loss(
    student_descriptors: torch.Tensor,
    teacher_descriptors: torch.Tensor,
    teacher_view_descriptors: torch.Tensor,
    teacher_mask: torch.Tensor,
    frozen_text_bank: torch.Tensor,
    scene_ids: Sequence[Hashable] | torch.Tensor,
    *,
    standard_error_multiplier: float = 2.0,
    tie_tolerance: float = 1e-6,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Regress scene-wise response gaps with multiview confidence weights.

    For a region pair and text query, the detached confidence is

    ``abs(gap) / (abs(gap) + multiplier * pair_standard_error + eps)``.

    Each scene/query is first reduced over all valid region pairs, then the
    scalar loss averages valid scene/query units.  This prevents a scene with
    more region pairs from silently dominating the objective.  Teacher
    descriptors, per-view variance, confidence weights, and the text bank are
    all excluded from autograd; only ``student_descriptors`` receives gradient.
    """

    _validate_cosine_response_inputs(
        student_descriptors,
        teacher_descriptors,
        frozen_text_bank,
    )
    if (
        not math.isfinite(float(standard_error_multiplier))
        or float(standard_error_multiplier) < 0.0
    ):
        raise ValueError("standard_error_multiplier must be finite and non-negative")
    if not math.isfinite(float(tie_tolerance)) or float(tie_tolerance) < 0.0:
        raise ValueError("tie_tolerance must be finite and non-negative")
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("eps must be finite and positive")
    if (
        teacher_view_descriptors.ndim != 3
        or teacher_view_descriptors.shape[0] != student_descriptors.shape[0]
        or teacher_view_descriptors.shape[2] != student_descriptors.shape[1]
        or teacher_mask.shape != teacher_view_descriptors.shape[:2]
    ):
        raise ValueError("teacher view descriptors/mask must align with [B,V,D]")

    student = F.normalize(student_descriptors.float(), dim=-1, eps=float(eps))
    uncertainty = compute_multiview_teacher_response_uncertainty(
        teacher_view_descriptors.detach().to(device=student.device),
        teacher_mask.detach().to(device=student.device),
        frozen_text_bank.detach().to(device=student.device),
        eps=float(eps),
    )
    with torch.no_grad():
        teacher = F.normalize(
            teacher_descriptors.detach().to(
                device=student.device,
                dtype=torch.float32,
            ),
            dim=-1,
            eps=float(eps),
        )
        text = F.normalize(
            frozen_text_bank.detach().to(
                device=student.device,
                dtype=torch.float32,
            ),
            dim=-1,
            eps=float(eps),
        )
        teacher_responses = teacher @ text.T
        response_standard_error = uncertainty["response_standard_error"]
    student_responses = student @ text.T
    groups = _scene_group_indices(
        scene_ids,
        batch_size=student_descriptors.shape[0],
        device=student.device,
    )

    zero = student_descriptors.sum() * 0.0
    scene_query_losses: list[torch.Tensor] = []
    scene_query_validity: list[torch.Tensor] = []
    scene_query_weight_sums: list[torch.Tensor] = []
    valid_weights: list[torch.Tensor] = []
    valid_scene_count = 0
    valid_pair_query_count = 0
    for indices in groups:
        region_count = int(indices.numel())
        if region_count < 2:
            continue
        student_scene = student_responses.index_select(0, indices)
        teacher_scene = teacher_responses.index_select(0, indices)
        standard_error_scene = response_standard_error.index_select(0, indices)
        pairs = torch.triu_indices(
            region_count,
            region_count,
            offset=1,
            device=student.device,
        )
        student_gaps = student_scene[pairs[0]] - student_scene[pairs[1]]
        with torch.no_grad():
            teacher_gaps = teacher_scene[pairs[0]] - teacher_scene[pairs[1]]
            teacher_span = teacher_scene.amax(dim=0) - teacher_scene.amin(dim=0)
            pair_standard_error = torch.sqrt(
                standard_error_scene[pairs[0]].square()
                + standard_error_scene[pairs[1]].square()
            )
            confidence = teacher_gaps.abs() / (
                teacher_gaps.abs()
                + float(standard_error_multiplier) * pair_standard_error
                + float(eps)
            )
            valid_queries = teacher_span > float(tie_tolerance)
            valid = (
                teacher_gaps.abs() > float(tie_tolerance)
            ) & valid_queries.unsqueeze(0)
            weights = confidence * valid.to(dtype=confidence.dtype)
            weight_sum = weights.sum(dim=0)
            unit_valid = weight_sum > float(eps)
            scale = teacher_span.clamp_min(float(eps)).unsqueeze(0)
        normalized_student = student_gaps / scale
        normalized_teacher = teacher_gaps / scale
        pair_losses = F.smooth_l1_loss(
            normalized_student,
            normalized_teacher,
            reduction="none",
        )
        unit_losses = (pair_losses * weights).sum(dim=0) / weight_sum.clamp_min(
            float(eps)
        )
        unit_losses = unit_losses.masked_fill(~unit_valid, 0.0)
        scene_query_losses.append(unit_losses)
        scene_query_validity.append(unit_valid)
        scene_query_weight_sums.append(weight_sum)
        if bool(valid.any().item()):
            valid_scene_count += 1
            valid_pair_query_count += int(valid.sum().item())
            valid_weights.append(confidence[valid])

    if scene_query_losses:
        unit_loss_tensor = torch.stack(scene_query_losses)
        unit_valid_tensor = torch.stack(scene_query_validity)
        unit_weight_tensor = torch.stack(scene_query_weight_sums)
        loss = (
            unit_loss_tensor[unit_valid_tensor].mean()
            if bool(unit_valid_tensor.any().item())
            else zero
        )
    else:
        query_count = int(frozen_text_bank.shape[0])
        unit_loss_tensor = zero.new_zeros((0, query_count))
        unit_valid_tensor = torch.zeros(
            (0, query_count),
            dtype=torch.bool,
            device=zero.device,
        )
        unit_weight_tensor = zero.new_zeros((0, query_count))
        loss = zero
    weight_mean = (
        torch.cat(valid_weights).mean()
        if valid_weights
        else zero.detach()
    )
    return loss, {
        "valid_scene_count": zero.new_tensor(valid_scene_count).detach(),
        "valid_query_count": unit_valid_tensor.sum().detach(),
        "valid_pair_query_count": zero.new_tensor(
            valid_pair_query_count
        ).detach(),
        "uncertainty_weight_mean": weight_mean.detach(),
        "scene_query_loss": unit_loss_tensor.detach(),
        "scene_query_valid": unit_valid_tensor.detach(),
        "scene_query_weight_sum": unit_weight_tensor.detach(),
        "teacher_response_variance_mean": uncertainty[
            "response_variance"
        ].mean().detach(),
    }


def fractional_upper_cvar(
    values: torch.Tensor,
    tail_fraction: float = 0.1,
    *,
    dim: int | None = None,
) -> torch.Tensor:
    """Return a fractional empirical upper-tail conditional value at risk.

    The final order statistic receives a fractional coefficient when
    ``tail_fraction * N`` is non-integral.  Unlike rounding the tail to an
    integer top-k, this preserves the requested empirical tail mass.  The
    operation remains differentiable with respect to the selected values.
    """

    if not isinstance(values, torch.Tensor):
        raise TypeError("values must be a torch.Tensor")
    if not values.is_floating_point() or values.numel() == 0:
        raise ValueError("values must be a non-empty floating-point tensor")
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("values must contain only finite values")
    if (
        not math.isfinite(float(tail_fraction))
        or not 0.0 < float(tail_fraction) <= 1.0
    ):
        raise ValueError("tail_fraction must lie in (0,1]")
    if dim is None:
        ordered = torch.sort(values.reshape(-1), descending=True).values
        axis = 0
    else:
        axis = int(dim)
        if axis < -values.ndim or axis >= values.ndim:
            raise ValueError("dim is outside the input rank")
        axis %= values.ndim
        ordered = torch.sort(values, dim=axis, descending=True).values

    count = int(ordered.shape[axis])
    tail_mass = float(tail_fraction) * count
    full_count = min(count, int(math.floor(tail_mass + 1e-12)))
    remainder = tail_mass - full_count
    if remainder < 1e-12:
        remainder = 0.0
    if full_count:
        full = ordered.narrow(axis, 0, full_count).sum(dim=axis)
    else:
        full = ordered.select(axis, 0) * 0.0
    if remainder > 0.0 and full_count < count:
        full = full + remainder * ordered.select(axis, full_count)
    return full / tail_mass


def _weighted_mean(values: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
    if weights is None:
        return values.mean()
    active_weights = weights.to(device=values.device, dtype=values.dtype).reshape(-1)
    if active_weights.shape[0] != values.shape[0]:
        raise ValueError(f"Expected {values.shape[0]} weights, got {active_weights.shape[0]}")
    return (values * active_weights).sum() / active_weights.sum().clamp_min(1e-6)


def _normalise_support_logits(logits: torch.Tensor, mode: str) -> torch.Tensor:
    mode = str(mode or "none").lower()
    if mode == "none":
        return logits
    centered = logits - logits.mean(dim=-1, keepdim=True)
    if mode == "center":
        return centered
    if mode == "zscore":
        scale = centered.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return centered / scale
    raise ValueError("support_logit_norm must be one of: none, center, zscore")


def compute_direct_point_query_logit_distill_loss(
    student_summary: torch.Tensor,
    teacher_summary: torch.Tensor,
    text_embeddings: torch.Tensor,
    *,
    temperature: float = 1.0,
    confidence_threshold: float = 0.0,
    sample_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match teacher query distributions for arbitrary text banks.

    This complements feature cosine distillation: small vector errors can still
    flip open-vocabulary query rankings, so the primitive head also preserves
    the VPR teacher's logits over the evaluation query bank.
    """
    if student_summary.shape != teacher_summary.shape:
        raise ValueError(
            "student_summary and teacher_summary must have the same shape, got "
            f"{tuple(student_summary.shape)} vs {tuple(teacher_summary.shape)}"
        )
    if student_summary.ndim != 2 or text_embeddings.ndim != 2:
        raise ValueError("Expected [N,C] summaries and [Q,C] text embeddings")
    if student_summary.shape[1] != text_embeddings.shape[1]:
        raise ValueError(
            "summary/text dimension mismatch: "
            f"{student_summary.shape[1]} vs {text_embeddings.shape[1]}"
        )
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    text = F.normalize(text_embeddings.float().to(student_summary.device), dim=-1)
    student = F.normalize(student_summary.float(), dim=-1)
    teacher = F.normalize(teacher_summary.float(), dim=-1)
    student_logits = student @ text.T
    with torch.no_grad():
        teacher_logits = teacher @ text.T
        teacher_prob = F.softmax(teacher_logits / temperature, dim=-1)
        teacher_conf, teacher_label = teacher_prob.max(dim=-1)
        valid = teacher_conf >= float(confidence_threshold)
    zero = student_summary.sum() * 0.0
    valid_ratio = valid.float().mean()
    if not valid.any():
        return zero, {
            "valid_ratio": valid_ratio.detach(),
            "teacher_conf": zero.detach(),
            "agreement": zero.detach(),
        }

    log_prob = F.log_softmax(student_logits[valid] / temperature, dim=-1)
    per_point = F.kl_div(log_prob, teacher_prob[valid], reduction="none").sum(dim=-1)
    loss = _weighted_mean(
        per_point,
        sample_weights[valid] if sample_weights is not None else None,
    )
    if temperature != 1.0:
        loss = loss * (temperature ** 2)
    with torch.no_grad():
        agreement = (student_logits[valid].argmax(dim=-1) == teacher_label[valid]).float().mean()
        mean_conf = teacher_conf[valid].mean()
    return loss, {
        "valid_ratio": valid_ratio.detach(),
        "teacher_conf": mean_conf.detach(),
        "agreement": agreement.detach(),
    }


def compute_direct_point_query_support_distill_loss(
    student_summary: torch.Tensor,
    teacher_summary: torch.Tensor,
    text_embeddings: torch.Tensor,
    *,
    temperature: float = 1.0,
    confidence_threshold: float = 0.0,
    sample_weights: Optional[torch.Tensor] = None,
    support_logit_norm: str = "none",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match teacher primitive-support distributions for each text query.

    ``compute_direct_point_query_logit_distill_loss`` preserves the text ranking
    inside each primitive. Direct 3D object selection, however, thresholds or
    ranks primitives for one query at a time. This loss distills that evaluation
    axis directly by matching the teacher distribution over sampled primitives
    for every text query.
    """
    if student_summary.shape != teacher_summary.shape:
        raise ValueError(
            "student_summary and teacher_summary must have the same shape, got "
            f"{tuple(student_summary.shape)} vs {tuple(teacher_summary.shape)}"
        )
    if student_summary.ndim != 2 or text_embeddings.ndim != 2:
        raise ValueError("Expected [N,C] summaries and [Q,C] text embeddings")
    if student_summary.shape[1] != text_embeddings.shape[1]:
        raise ValueError(
            "summary/text dimension mismatch: "
            f"{student_summary.shape[1]} vs {text_embeddings.shape[1]}"
        )
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    text = F.normalize(text_embeddings.float().to(student_summary.device), dim=-1)
    student = F.normalize(student_summary.float(), dim=-1)
    teacher = F.normalize(teacher_summary.float(), dim=-1)
    student_logits = student @ text.T
    with torch.no_grad():
        teacher_logits = teacher @ text.T
        teacher_conf, teacher_argmax = teacher_logits.max(dim=0)
        valid_queries = teacher_conf >= float(confidence_threshold)

    zero = student_summary.sum() * 0.0
    valid_ratio = valid_queries.float().mean()
    if not valid_queries.any():
        return zero, {
            "valid_ratio": valid_ratio.detach(),
            "teacher_conf": zero.detach(),
            "top1_agreement": zero.detach(),
        }

    # Transpose to [Q, N], so each row is one query's primitive support.
    teacher_support_logits = _normalise_support_logits(
        teacher_logits.T[valid_queries],
        support_logit_norm,
    ) / float(temperature)
    student_support_logits = _normalise_support_logits(
        student_logits.T[valid_queries],
        support_logit_norm,
    ) / float(temperature)
    if sample_weights is not None:
        weights = sample_weights.to(device=student_summary.device, dtype=student_support_logits.dtype)
        if weights.ndim != 1 or weights.shape[0] != student_summary.shape[0]:
            raise ValueError(
                f"Expected sample_weights [{student_summary.shape[0]}], got {tuple(weights.shape)}"
            )
        teacher_support_logits = teacher_support_logits + weights.clamp_min(1e-6).log().unsqueeze(0)

    with torch.no_grad():
        teacher_prob = F.softmax(teacher_support_logits, dim=-1)
    log_prob = F.log_softmax(student_support_logits, dim=-1)
    loss = F.kl_div(log_prob, teacher_prob, reduction="batchmean")
    with torch.no_grad():
        student_argmax = student_logits[:, valid_queries].argmax(dim=0)
        top1_agreement = (student_argmax == teacher_argmax[valid_queries]).float().mean()
        mean_conf = teacher_conf[valid_queries].mean()
    return loss, {
        "valid_ratio": valid_ratio.detach(),
        "teacher_conf": mean_conf.detach(),
        "top1_agreement": top1_agreement.detach(),
    }
