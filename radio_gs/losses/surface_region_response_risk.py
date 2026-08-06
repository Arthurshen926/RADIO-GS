"""Train-only visual contrast directions and scene-balanced response risk.

This module deliberately has no text, validation, or benchmark-vocabulary
input.  Its fixed response directions are selected only from frozen training
teacher views.  Selection is deterministic and parameter free:

1. L2-normalize valid teacher views.
2. Compute a view/region/scene-balanced global centre.
3. Centre and renormalize every valid view, remove exact duplicates, and sort
   candidates by a SHA-256 content key.
4. Run spherical farthest-first traversal.  The first candidate has the
   largest pre-normalization residual; later candidates maximize distance to
   the selected set.  Content order breaks exact numerical ties.

The loss preserves responses along those directions.  Valid-view disagreement
weights both within-scene pairwise gaps and a listwise regional distribution.
Each scene/query is one risk unit, scenes are averaged equally, and a
fractional upper CVaR exposes the worst response directions without rounded
top-k behaviour.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Hashable, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from radio_gs.losses.direct_point_query_logit_distill_loss import (
    _scene_group_indices,
    compute_multiview_teacher_response_uncertainty,
)
from radio_gs.losses.uncertainty_response_risk import (
    compute_equal_scene_mean_fractional_cvar_risk,
)


VISUAL_CONTRAST_BANK_CONTRACT = (
    "train_teacher_visual_contrast_bank_v1"
)
SCENE_RESPONSE_RISK_CONTRACT = (
    "equal_scene_uncertainty_gap_listwise_fractional_cvar_v1"
)


def _sha256_tensor(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()


def _canonical_scene_id(value: Hashable) -> str:
    """Return a stable type-aware representation for ordinary scene ids."""

    if isinstance(value, str):
        return "str:" + json.dumps(value, ensure_ascii=False)
    if isinstance(value, bytes):
        return "bytes:" + value.hex()
    if isinstance(value, bool):
        return f"bool:{str(value).lower()}"
    if isinstance(value, int):
        return f"int:{value}"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("scene_ids must be finite")
        return f"float:{value.hex()}"
    if isinstance(value, tuple):
        return "tuple:[" + ",".join(_canonical_scene_id(item) for item in value) + "]"
    raise TypeError(
        "scene_ids must use stable scalar or tuple values, got "
        f"{type(value).__name__}"
    )


def _scene_labels(
    scene_ids: Sequence[Hashable] | torch.Tensor,
    *,
    batch_size: int,
) -> list[Hashable]:
    if isinstance(scene_ids, torch.Tensor):
        if scene_ids.ndim != 1 or int(scene_ids.shape[0]) != batch_size:
            raise ValueError(f"scene_ids must have shape [{batch_size}]")
        labels = scene_ids.detach().cpu().tolist()
    else:
        if isinstance(scene_ids, (str, bytes)) or not isinstance(scene_ids, Sequence):
            raise TypeError("scene_ids must be a one-dimensional tensor or sequence")
        if len(scene_ids) != batch_size:
            raise ValueError(f"Expected {batch_size} scene_ids, got {len(scene_ids)}")
        labels = list(scene_ids)
    # Validate before any grouping so provenance never depends on object repr.
    for value in labels:
        _canonical_scene_id(value)
    return labels


def _content_key(vector: torch.Tensor) -> tuple[str, bytes]:
    raw = vector.detach().cpu().contiguous().numpy().tobytes(order="C")
    return hashlib.sha256(raw).hexdigest(), raw


def _stable_tensor_mean(values: list[torch.Tensor]) -> torch.Tensor:
    if not values:
        raise ValueError("cannot average an empty tensor collection")
    ordered = sorted(values, key=_content_key)
    return torch.stack(ordered, dim=0).mean(dim=0)


def _validate_teacher_views(
    teacher_view_descriptors: torch.Tensor,
    teacher_mask: torch.Tensor,
    *,
    require_two_views: bool,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(teacher_view_descriptors, torch.Tensor) or not isinstance(
        teacher_mask, torch.Tensor
    ):
        raise TypeError("teacher view descriptors and mask must be tensors")
    if (
        teacher_view_descriptors.ndim != 3
        or not teacher_view_descriptors.is_floating_point()
        or teacher_view_descriptors.numel() == 0
        or teacher_mask.dtype != torch.bool
        or teacher_mask.shape != teacher_view_descriptors.shape[:2]
        or not bool(torch.isfinite(teacher_view_descriptors).all())
    ):
        raise ValueError("teacher views/mask must be finite aligned [B,V,D] tensors")
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("eps must be finite and positive")
    mask = teacher_mask.detach().cpu()
    counts = mask.sum(dim=1)
    minimum = 2 if require_two_views else 1
    if not bool((counts >= minimum).all()):
        raise ValueError(
            f"every region requires at least {minimum} valid teacher view(s)"
        )
    views = F.normalize(
        teacher_view_descriptors.detach().cpu().to(dtype=torch.float64),
        dim=-1,
        eps=float(eps),
    )
    return views, mask


def build_train_only_visual_contrast_bank(
    teacher_view_descriptors: torch.Tensor,
    teacher_mask: torch.Tensor,
    scene_ids: Sequence[Hashable] | torch.Tensor,
    *,
    direction_count: int,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build a fixed visual direction bank from frozen training teachers.

    The returned bank is detached CPU float32 with shape ``[Q,D]``.  The
    centre gives every valid view equal weight inside its region, every region
    equal weight inside its scene, and every scene equal weight globally.
    Candidate coverage, unlike a frequency-based clustering objective, is not
    dominated by scenes containing more regions.

    ``direction_count`` is exact: the function raises instead of silently
    padding or reducing the bank when too few distinct non-degenerate visual
    residuals exist.
    """

    if isinstance(direction_count, bool) or not isinstance(direction_count, int):
        raise TypeError("direction_count must be an integer")
    if direction_count <= 0:
        raise ValueError("direction_count must be positive")
    views, mask = _validate_teacher_views(
        teacher_view_descriptors,
        teacher_mask,
        require_two_views=False,
        eps=eps,
    )
    batch_size, view_capacity, descriptor_dim = views.shape
    labels = _scene_labels(scene_ids, batch_size=batch_size)
    canonical_labels = [_canonical_scene_id(value) for value in labels]

    region_means: list[torch.Tensor] = []
    scene_regions: dict[str, list[torch.Tensor]] = {}
    valid_views: list[torch.Tensor] = []
    for row in range(batch_size):
        active = [views[row, column] for column in torch.where(mask[row])[0].tolist()]
        region_mean = _stable_tensor_mean(active)
        region_means.append(region_mean)
        scene_regions.setdefault(canonical_labels[row], []).append(region_mean)
        valid_views.extend(active)
    scene_means = [
        _stable_tensor_mean(scene_regions[label])
        for label in sorted(scene_regions)
    ]
    centre = torch.stack(scene_means, dim=0).mean(dim=0)

    # Exact duplicate removal is performed after centering/normalization.  For
    # a duplicate direction, retain the largest residual norm for deterministic
    # first-point initialization.
    unique: dict[bytes, tuple[str, torch.Tensor, float]] = {}
    nondegenerate_count = 0
    for view in valid_views:
        residual = view - centre
        residual_norm = float(torch.linalg.vector_norm(residual))
        if residual_norm <= float(eps):
            continue
        nondegenerate_count += 1
        direction = residual / residual_norm
        digest, raw = _content_key(direction)
        previous = unique.get(raw)
        if previous is None or residual_norm > previous[2]:
            unique[raw] = (digest, direction, residual_norm)
    candidates = sorted(unique.values(), key=lambda item: (item[0], item[1].numpy().tobytes()))
    if len(candidates) < direction_count:
        raise ValueError(
            f"requested {direction_count} directions but only "
            f"{len(candidates)} unique non-degenerate train visual residuals exist"
        )

    candidate_tensor = torch.stack([item[1] for item in candidates], dim=0)
    residual_norms = torch.tensor(
        [item[2] for item in candidates], dtype=torch.float64
    )
    selected: list[int] = [int(torch.argmax(residual_norms))]
    selected_mask = torch.zeros(len(candidates), dtype=torch.bool)
    selected_mask[selected[0]] = True
    nearest_distance = 1.0 - candidate_tensor @ candidate_tensor[selected[0]]
    nearest_distance[selected_mask] = -torch.inf
    while len(selected) < direction_count:
        next_index = int(torch.argmax(nearest_distance))
        selected.append(next_index)
        selected_mask[next_index] = True
        distance = 1.0 - candidate_tensor @ candidate_tensor[next_index]
        nearest_distance = torch.minimum(nearest_distance, distance)
        nearest_distance[selected_mask] = -torch.inf

    bank64 = candidate_tensor[selected]
    bank = F.normalize(bank64.float(), dim=-1, eps=float(eps)).detach()
    final_coverage_distance = (
        1.0 - candidate_tensor @ bank64.T
    ).amin(dim=1).clamp_min(0.0)
    canonical_scene_counts = sorted(
        (label, len(rows)) for label, rows in scene_regions.items()
    )
    multiset_hash = hashlib.sha256()
    for view in sorted(valid_views, key=_content_key):
        multiset_hash.update(_content_key(view)[1])
    source_hash = hashlib.sha256()
    source_hash.update(
        teacher_view_descriptors.detach().cpu().float().contiguous().numpy().tobytes(
            order="C"
        )
    )
    source_hash.update(mask.contiguous().numpy().tobytes(order="C"))
    source_hash.update(
        json.dumps(canonical_labels, separators=(",", ":")).encode("utf-8")
    )
    provenance: dict[str, Any] = {
        "contract": VISUAL_CONTRAST_BANK_CONTRACT,
        "source_scope": "frozen_train_teacher_visual_descriptors_only",
        "uses_text_or_vocabulary": False,
        "learned_parameters": 0,
        "random_seed": None,
        "selection_algorithm": (
            "equal_scene_centre_exact_dedup_spherical_farthest_first"
        ),
        "initial_point": "largest_centred_residual_norm",
        "tie_breaker": "ascending_sha256_then_float64_content_bytes",
        "direction_count_requested": direction_count,
        "direction_count": int(bank.shape[0]),
        "descriptor_dim": int(descriptor_dim),
        "scene_count": len(scene_regions),
        "region_count": int(batch_size),
        "view_capacity": int(view_capacity),
        "valid_view_count": int(mask.sum()),
        "valid_views_per_region_min": int(mask.sum(dim=1).min()),
        "valid_views_per_region_max": int(mask.sum(dim=1).max()),
        "valid_views_per_region_mean": float(mask.sum(dim=1).double().mean()),
        "nondegenerate_candidate_count": nondegenerate_count,
        "unique_candidate_count": len(candidates),
        "exact_duplicate_count": nondegenerate_count - len(candidates),
        "centering": "equal_view_within_region_equal_region_within_scene_equal_scene",
        "centre_l2_norm": float(torch.linalg.vector_norm(centre)),
        "centre_sha256_float64": _sha256_tensor(centre),
        "source_tensor_mask_scene_sha256": source_hash.hexdigest(),
        "normalized_valid_view_multiset_sha256": multiset_hash.hexdigest(),
        "canonical_scene_region_counts": canonical_scene_counts,
        "bank_sha256_float32": _sha256_tensor(bank),
        "candidate_covering_radius_cosine": float(final_coverage_distance.max()),
        "candidate_covering_distance_mean": float(final_coverage_distance.mean()),
        "selected_candidate_indices_in_content_order": selected,
    }
    return bank, provenance


def _validate_response_inputs(
    student_descriptors: torch.Tensor,
    teacher_view_descriptors: torch.Tensor,
    teacher_mask: torch.Tensor,
    visual_contrast_bank: torch.Tensor,
    *,
    eps: float,
) -> None:
    if not isinstance(student_descriptors, torch.Tensor) or not isinstance(
        visual_contrast_bank, torch.Tensor
    ):
        raise TypeError("student descriptors and visual contrast bank must be tensors")
    if (
        student_descriptors.ndim != 2
        or not student_descriptors.is_floating_point()
        or student_descriptors.numel() == 0
        or not bool(torch.isfinite(student_descriptors).all())
        or visual_contrast_bank.ndim != 2
        or not visual_contrast_bank.is_floating_point()
        or visual_contrast_bank.shape[0] == 0
        or not bool(torch.isfinite(visual_contrast_bank).all())
        or visual_contrast_bank.shape[1] != student_descriptors.shape[1]
    ):
        raise ValueError("student [B,D] and visual bank [Q,D] must be finite and align")
    if (
        not isinstance(teacher_view_descriptors, torch.Tensor)
        or teacher_view_descriptors.ndim != 3
        or teacher_view_descriptors.shape[0] != student_descriptors.shape[0]
        or teacher_view_descriptors.shape[2] != student_descriptors.shape[1]
        or not isinstance(teacher_mask, torch.Tensor)
        or teacher_mask.shape != teacher_view_descriptors.shape[:2]
    ):
        raise ValueError("teacher views/mask must align with student [B,D]")
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("eps must be finite and positive")


def compute_visual_contrast_scene_response_units(
    student_descriptors: torch.Tensor,
    teacher_view_descriptors: torch.Tensor,
    teacher_mask: torch.Tensor,
    visual_contrast_bank: torch.Tensor,
    scene_ids: Sequence[Hashable] | torch.Tensor,
    *,
    gap_weight: float = 0.5,
    listwise_weight: float = 0.5,
    listwise_temperature: float = 0.25,
    standard_error_multiplier: float = 2.0,
    tie_tolerance: float = 1e-6,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Return uncertainty-weighted ``[scene,direction]`` response-risk units.

    Pairwise gap losses assign every region the same number of comparisons in
    a complete scene.  The listwise Brier divergence starts from a uniform
    per-region mean.  Detached valid-view uncertainty then downweights only
    teacher evidence that is inconsistent across views; all such weights are
    renormalized inside their scene/direction unit.
    """

    _validate_response_inputs(
        student_descriptors,
        teacher_view_descriptors,
        teacher_mask,
        visual_contrast_bank,
        eps=eps,
    )
    scalars = {
        "gap_weight": gap_weight,
        "listwise_weight": listwise_weight,
        "listwise_temperature": listwise_temperature,
        "standard_error_multiplier": standard_error_multiplier,
        "tie_tolerance": tie_tolerance,
        "eps": eps,
    }
    if any(not math.isfinite(float(value)) for value in scalars.values()):
        raise ValueError("response-risk scalar parameters must be finite")
    if (
        gap_weight < 0.0
        or listwise_weight < 0.0
        or not math.isclose(gap_weight + listwise_weight, 1.0, abs_tol=1e-12)
        or listwise_temperature <= 0.0
        or standard_error_multiplier < 0.0
        or tie_tolerance < 0.0
        or eps <= 0.0
    ):
        raise ValueError("response-risk scalar parameters are outside their domain")

    labels = _scene_labels(scene_ids, batch_size=student_descriptors.shape[0])
    student = F.normalize(student_descriptors.float(), dim=-1, eps=float(eps))
    bank = F.normalize(
        visual_contrast_bank.detach().to(device=student.device, dtype=torch.float32),
        dim=-1,
        eps=float(eps),
    )
    uncertainty = compute_multiview_teacher_response_uncertainty(
        teacher_view_descriptors.detach().to(device=student.device),
        teacher_mask.detach().to(device=student.device),
        bank,
        eps=float(eps),
    )
    teacher_response = uncertainty["response_mean"]
    response_standard_error = uncertainty["response_standard_error"]
    student_response = student @ bank.T
    groups = _scene_group_indices(
        labels,
        batch_size=student.shape[0],
        device=student.device,
    )

    combined_units: list[torch.Tensor] = []
    combined_validity: list[torch.Tensor] = []
    gap_units: list[torch.Tensor] = []
    listwise_units: list[torch.Tensor] = []
    gap_confidences: list[torch.Tensor] = []
    region_confidences: list[torch.Tensor] = []
    valid_pair_query_count = 0
    for indices in groups:
        region_count = int(indices.numel())
        if region_count < 2:
            raise ValueError(
                "response risk requires complete scenes with at least two regions"
            )
        student_scene = student_response.index_select(0, indices)
        teacher_scene = teacher_response.index_select(0, indices)
        standard_error_scene = response_standard_error.index_select(0, indices)
        with torch.no_grad():
            teacher_span = teacher_scene.amax(dim=0) - teacher_scene.amin(dim=0)
            query_valid = teacher_span > float(tie_tolerance)
            scale = teacher_span.clamp_min(float(eps))

        pairs = torch.triu_indices(
            region_count, region_count, offset=1, device=student.device
        )
        student_gap = student_scene[pairs[0]] - student_scene[pairs[1]]
        with torch.no_grad():
            teacher_gap = teacher_scene[pairs[0]] - teacher_scene[pairs[1]]
            pair_standard_error = torch.sqrt(
                standard_error_scene[pairs[0]].square()
                + standard_error_scene[pairs[1]].square()
            )
            pair_valid = (
                teacher_gap.abs() > float(tie_tolerance)
            ) & query_valid.unsqueeze(0)
            pair_confidence = teacher_gap.abs() / (
                teacher_gap.abs()
                + float(standard_error_multiplier) * pair_standard_error
                + float(eps)
            )
            pair_weight = pair_confidence * pair_valid.to(pair_confidence.dtype)
            pair_weight_sum = pair_weight.sum(dim=0)
            gap_valid = pair_weight_sum > float(eps)
        pair_loss = F.smooth_l1_loss(
            student_gap / scale.unsqueeze(0),
            teacher_gap / scale.unsqueeze(0),
            reduction="none",
        )
        scene_gap = (pair_loss * pair_weight).sum(dim=0) / pair_weight_sum.clamp_min(
            float(eps)
        )
        scene_gap = scene_gap * gap_valid.to(scene_gap.dtype)

        student_logits = (
            student_scene - student_scene.mean(dim=0, keepdim=True)
        ) / scale.unsqueeze(0)
        with torch.no_grad():
            teacher_logits = (
                teacher_scene - teacher_scene.mean(dim=0, keepdim=True)
            ) / scale.unsqueeze(0)
            # Standard error is made dimensionless by the observed scene span.
            # A zero-disagreement view set therefore has unit confidence.
            region_confidence = 1.0 / (
                1.0
                + float(standard_error_multiplier)
                * standard_error_scene
                / scale.unsqueeze(0)
            )
            region_weight = region_confidence * query_valid.unsqueeze(0).to(
                region_confidence.dtype
            )
            region_weight_sum = region_weight.sum(dim=0)
            listwise_valid = region_weight_sum > float(eps)
            teacher_probability = F.softmax(
                teacher_logits / float(listwise_temperature), dim=0
            )
        student_probability = F.softmax(
            student_logits / float(listwise_temperature), dim=0
        )
        region_brier = 0.5 * (
            student_probability - teacher_probability
        ).square()
        scene_listwise = (
            region_brier * region_weight
        ).sum(dim=0) / region_weight_sum.clamp_min(float(eps))
        scene_listwise = scene_listwise * listwise_valid.to(scene_listwise.dtype)

        active_validity = (
            (gap_valid if gap_weight > 0.0 else query_valid)
            & (listwise_valid if listwise_weight > 0.0 else query_valid)
        )
        if not bool(active_validity.any()):
            raise ValueError("a scene contains no non-degenerate response direction")
        combined = (
            float(gap_weight) * scene_gap
            + float(listwise_weight) * scene_listwise
        )
        combined = combined * active_validity.to(combined.dtype)
        combined_units.append(combined)
        combined_validity.append(active_validity)
        gap_units.append(scene_gap)
        listwise_units.append(scene_listwise)
        if bool(pair_valid.any()):
            gap_confidences.append(pair_confidence[pair_valid])
            valid_pair_query_count += int(pair_valid.sum())
        region_confidences.append(region_confidence[query_valid.unsqueeze(0).expand_as(region_confidence)])

    units = torch.stack(combined_units)
    validity = torch.stack(combined_validity).detach()
    gap_tensor = torch.stack(gap_units)
    listwise_tensor = torch.stack(listwise_units)
    gap_confidence_values = torch.cat(gap_confidences)
    region_confidence_values = torch.cat(region_confidences)
    return units, validity, {
        "contract": SCENE_RESPONSE_RISK_CONTRACT,
        "scene_count": units.new_tensor(len(groups)).detach(),
        "direction_count": units.new_tensor(bank.shape[0]).detach(),
        "valid_scene_direction_count": validity.sum().detach(),
        "valid_pair_direction_count": units.new_tensor(
            valid_pair_query_count
        ).detach(),
        "gap_unit_loss": gap_tensor.detach(),
        "listwise_unit_loss": listwise_tensor.detach(),
        "gap_uncertainty_confidence_mean": gap_confidence_values.mean().detach(),
        "region_uncertainty_confidence_mean": (
            region_confidence_values.mean().detach()
        ),
        "teacher_response_standard_error_mean": (
            response_standard_error.mean().detach()
        ),
        "valid_views_per_region": uncertainty["view_counts"].detach(),
    }


def compute_visual_contrast_scene_response_risk(
    student_descriptors: torch.Tensor,
    teacher_view_descriptors: torch.Tensor,
    teacher_mask: torch.Tensor,
    visual_contrast_bank: torch.Tensor,
    scene_ids: Sequence[Hashable] | torch.Tensor,
    *,
    gap_weight: float = 0.5,
    listwise_weight: float = 0.5,
    listwise_temperature: float = 0.25,
    standard_error_multiplier: float = 2.0,
    tie_tolerance: float = 1e-6,
    eps: float = 1e-6,
    mean_weight: float = 0.5,
    cvar_weight: float = 0.5,
    cvar_tail_fraction: float = 0.10,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compose train-only response units with equal-scene fractional CVaR."""

    units, validity, unit_statistics = compute_visual_contrast_scene_response_units(
        student_descriptors,
        teacher_view_descriptors,
        teacher_mask,
        visual_contrast_bank,
        scene_ids,
        gap_weight=gap_weight,
        listwise_weight=listwise_weight,
        listwise_temperature=listwise_temperature,
        standard_error_multiplier=standard_error_multiplier,
        tie_tolerance=tie_tolerance,
        eps=eps,
    )
    risk, risk_statistics = compute_equal_scene_mean_fractional_cvar_risk(
        units,
        validity,
        mean_weight=mean_weight,
        cvar_weight=cvar_weight,
        cvar_tail_fraction=cvar_tail_fraction,
    )
    return risk, {
        **unit_statistics,
        **risk_statistics,
        "scene_direction_unit_loss": units.detach(),
        "scene_direction_valid": validity.detach(),
    }
