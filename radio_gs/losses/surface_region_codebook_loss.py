"""Losses for a query-independent multi-hypothesis SurfaceRegion readout."""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import torch
import torch.nn.functional as F


def permutation_set_matching_loss(
    predicted_tokens: torch.Tensor,
    predicted_descriptors: torch.Tensor,
    teacher_tokens: torch.Tensor,
    teacher_descriptors: torch.Tensor,
    teacher_mask: torch.Tensor,
    *,
    token_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match latent slots to the unordered valid teacher-view set.

    Slot assignment is a discrete target construction.  The selected costs
    retain gradients to the predicted token and descriptor tensors.
    """

    predicted_tokens = torch.as_tensor(predicted_tokens).float()
    predicted_descriptors = F.normalize(
        torch.as_tensor(predicted_descriptors).float(), dim=-1, eps=1e-8
    )
    teacher_tokens = torch.as_tensor(
        teacher_tokens, device=predicted_tokens.device
    ).float()
    teacher_descriptors = F.normalize(
        torch.as_tensor(
            teacher_descriptors, device=predicted_descriptors.device
        ).float(),
        dim=-1,
        eps=1e-8,
    )
    teacher_mask = torch.as_tensor(
        teacher_mask, device=predicted_tokens.device
    ).bool()
    if predicted_tokens.ndim != 3 or predicted_descriptors.ndim != 3:
        raise ValueError("predicted codebook tensors must be [B,K,D]")
    batch, slots = predicted_tokens.shape[:2]
    views = int(teacher_tokens.shape[1])
    if (
        predicted_descriptors.shape[:2] != (batch, slots)
        or teacher_tokens.shape[:2] != (batch, views)
        or teacher_descriptors.shape[:2] != (batch, views)
        or teacher_mask.shape != (batch, views)
        or not bool(teacher_mask.any(dim=1).all())
        or int(teacher_mask.sum(dim=1).max()) > slots
    ):
        raise ValueError("teacher and predicted codebook shapes are incompatible")
    if not 0.0 <= float(token_weight) <= 100.0:
        raise ValueError("token_weight is outside the supported range")

    descriptor_cost = 1.0 - torch.einsum(
        "bkd,bvd->bkv", predicted_descriptors, teacher_descriptors
    )
    normalized_predicted_tokens = F.normalize(
        predicted_tokens, dim=-1, eps=1e-8
    )
    normalized_teacher_tokens = F.normalize(
        teacher_tokens, dim=-1, eps=1e-8
    )
    token_cost = 1.0 - torch.einsum(
        "bkd,bvd->bkv", normalized_predicted_tokens, normalized_teacher_tokens
    )
    cost = descriptor_cost + float(token_weight) * token_cost
    losses: list[torch.Tensor] = []
    assignments = torch.full(
        (batch, views), -1, dtype=torch.long, device=predicted_tokens.device
    )
    for row in range(batch):
        valid_views = torch.where(teacher_mask[row])[0]
        candidates = tuple(itertools.permutations(range(slots), len(valid_views)))
        candidate_costs = torch.stack(
            [
                cost[
                    row,
                    torch.tensor(candidate, device=cost.device),
                    valid_views,
                ].mean()
                for candidate in candidates
            ]
        )
        selected = int(candidate_costs.detach().argmin())
        slot_indices = torch.tensor(candidates[selected], device=cost.device)
        assignments[row, valid_views] = slot_indices
        losses.append(candidate_costs[selected])
    return torch.stack(losses).mean(), assignments


def gauge_aware_permutation_set_matching_loss(
    predicted_tokens: torch.Tensor,
    predicted_descriptors: torch.Tensor,
    teacher_tokens: torch.Tensor,
    teacher_descriptors: torch.Tensor,
    teacher_mask: torch.Tensor,
    *,
    token_direction_weight: float,
    token_log_norm_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match unordered hypotheses while preserving token direction and gauge.

    The official summary head is nonlinear in the 1280-D token, so matching
    only cosine direction leaves an otherwise invisible scale degree of
    freedom.  Log norm is dimensionless and makes that gauge explicit.
    """

    predicted_tokens = torch.as_tensor(predicted_tokens).float()
    predicted_descriptors = F.normalize(
        torch.as_tensor(predicted_descriptors).float(), dim=-1, eps=1e-8
    )
    teacher_tokens = torch.as_tensor(
        teacher_tokens, device=predicted_tokens.device
    ).float()
    teacher_descriptors = F.normalize(
        torch.as_tensor(
            teacher_descriptors, device=predicted_descriptors.device
        ).float(),
        dim=-1,
        eps=1e-8,
    )
    teacher_mask = torch.as_tensor(
        teacher_mask, device=predicted_tokens.device
    ).bool()
    if predicted_tokens.ndim != 3 or predicted_descriptors.ndim != 3:
        raise ValueError("predicted codebook tensors must be [B,K,D]")
    batch, slots = predicted_tokens.shape[:2]
    views = int(teacher_tokens.shape[1])
    if (
        predicted_descriptors.shape[:2] != (batch, slots)
        or teacher_tokens.shape[:2] != (batch, views)
        or teacher_descriptors.shape[:2] != (batch, views)
        or teacher_mask.shape != (batch, views)
        or not bool(teacher_mask.any(dim=1).all())
        or int(teacher_mask.sum(dim=1).max()) > slots
    ):
        raise ValueError("teacher and predicted codebook shapes are incompatible")
    if not 0.0 <= float(token_direction_weight) <= 100.0:
        raise ValueError("token_direction_weight is outside the supported range")
    if not 0.0 <= float(token_log_norm_weight) <= 100.0:
        raise ValueError("token_log_norm_weight is outside the supported range")

    descriptor_cost = 1.0 - torch.einsum(
        "bkd,bvd->bkv", predicted_descriptors, teacher_descriptors
    )
    predicted_direction = F.normalize(predicted_tokens, dim=-1, eps=1e-8)
    teacher_direction = F.normalize(teacher_tokens, dim=-1, eps=1e-8)
    direction_cost = 1.0 - torch.einsum(
        "bkd,bvd->bkv", predicted_direction, teacher_direction
    )
    predicted_log_norm = predicted_tokens.norm(dim=-1).clamp_min(1e-8).log()
    teacher_log_norm = teacher_tokens.norm(dim=-1).clamp_min(1e-8).log()
    log_norm_cost = F.smooth_l1_loss(
        predicted_log_norm[:, :, None].expand(-1, -1, views),
        teacher_log_norm[:, None, :].expand(-1, slots, -1),
        reduction="none",
    )
    cost = (
        descriptor_cost
        + float(token_direction_weight) * direction_cost
        + float(token_log_norm_weight) * log_norm_cost
    )
    losses: list[torch.Tensor] = []
    assignments = torch.full(
        (batch, views), -1, dtype=torch.long, device=predicted_tokens.device
    )
    for row in range(batch):
        valid_views = torch.where(teacher_mask[row])[0]
        candidates = tuple(itertools.permutations(range(slots), len(valid_views)))
        candidate_costs = torch.stack(
            [
                cost[
                    row,
                    torch.tensor(candidate, device=cost.device),
                    valid_views,
                ].mean()
                for candidate in candidates
            ]
        )
        selected = int(candidate_costs.detach().argmin())
        slot_indices = torch.tensor(candidates[selected], device=cost.device)
        assignments[row, valid_views] = slot_indices
        losses.append(candidate_costs[selected])
    return torch.stack(losses).mean(), assignments


def latent_query_responses(
    descriptors: torch.Tensor,
    text_embeddings: torch.Tensor,
    *,
    priors: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Smoothly marginalize query responses over latent region hypotheses."""

    values = F.normalize(torch.as_tensor(descriptors).float(), dim=-1, eps=1e-8)
    text = F.normalize(
        torch.as_tensor(text_embeddings, device=values.device).float(),
        dim=-1,
        eps=1e-8,
    )
    if values.ndim != 3 or text.ndim != 2 or values.shape[-1] != text.shape[-1]:
        raise ValueError("descriptors [B,K,D] and text [Q,D] must align")
    if not 0.0 < float(temperature) <= 1.0:
        raise ValueError("temperature must be in (0,1]")
    batch, slots = values.shape[:2]
    active = (
        torch.ones(batch, slots, dtype=torch.bool, device=values.device)
        if mask is None
        else torch.as_tensor(mask, device=values.device).bool()
    )
    if active.shape != (batch, slots) or not bool(active.any(dim=1).all()):
        raise ValueError("latent mask must keep a hypothesis in every row")
    if priors is None:
        weights = active.float() / active.sum(dim=1, keepdim=True)
    else:
        weights = torch.as_tensor(priors, device=values.device).float()
        if weights.shape != (batch, slots) or not bool(torch.isfinite(weights).all()):
            raise ValueError("latent priors must be finite [B,K]")
        if bool((weights < 0).any()) or bool((weights[~active] != 0).any()):
            raise ValueError("latent priors must be non-negative and masked")
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    cosine = torch.einsum("bkd,qd->bkq", values, text)
    log_weight = weights.clamp_min(1e-12).log()[..., None]
    logits = cosine / float(temperature) + log_weight
    logits = logits.masked_fill(~active[..., None], torch.finfo(logits.dtype).min)
    return float(temperature) * torch.logsumexp(logits, dim=1)


def latent_query_max_responses(
    descriptors: torch.Tensor,
    text_embeddings: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Read a persistent hypothesis set with a masked hard maximum."""

    values = F.normalize(torch.as_tensor(descriptors).float(), dim=-1, eps=1e-8)
    text = F.normalize(
        torch.as_tensor(text_embeddings, device=values.device).float(),
        dim=-1,
        eps=1e-8,
    )
    if values.ndim != 3 or text.ndim != 2 or values.shape[-1] != text.shape[-1]:
        raise ValueError("descriptors [B,K,D] and text [Q,D] must align")
    batch, slots = values.shape[:2]
    active = (
        torch.ones(batch, slots, dtype=torch.bool, device=values.device)
        if mask is None
        else torch.as_tensor(mask, device=values.device).bool()
    )
    if active.shape != (batch, slots) or not bool(active.any(dim=1).all()):
        raise ValueError("latent mask must keep a hypothesis in every row")
    cosine = torch.einsum("bkd,qd->bkq", values, text)
    cosine = cosine.masked_fill(
        ~active[..., None], torch.finfo(cosine.dtype).min
    )
    return cosine.amax(dim=1)


def scene_listwise_and_hard_negative_loss(
    student_responses: torch.Tensor,
    teacher_responses: torch.Tensor,
    scene_ids: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve within-scene region ranking and the top/runner-up gap."""

    student = torch.as_tensor(student_responses).float()
    teacher = torch.as_tensor(teacher_responses, device=student.device).float()
    if student.ndim != 2 or teacher.shape != student.shape:
        raise ValueError("student and teacher responses must align as [B,Q]")
    if len(scene_ids) != student.shape[0] or any(not str(value) for value in scene_ids):
        raise ValueError("scene_ids must bind every response row")
    listwise_losses: list[torch.Tensor] = []
    margin_losses: list[torch.Tensor] = []
    for scene in sorted(set(str(value) for value in scene_ids)):
        rows = torch.tensor(
            [index for index, value in enumerate(scene_ids) if str(value) == scene],
            dtype=torch.long,
            device=student.device,
        )
        if rows.numel() < 2:
            continue
        target = teacher[rows]
        prediction = student[rows]
        scale = target.std(dim=0, unbiased=False)
        valid = scale > 1e-6
        if not bool(valid.any()):
            continue
        target_logits = (target[:, valid] - target[:, valid].mean(0)) / scale[valid]
        predicted_logits = (
            prediction[:, valid] - prediction[:, valid].mean(0)
        ) / scale[valid]
        target_probability = torch.softmax(target_logits, dim=0)
        listwise_losses.append(
            (
                target_probability
                * (
                    torch.log_softmax(target_logits, dim=0)
                    - torch.log_softmax(predicted_logits, dim=0)
                )
            ).sum(dim=0).mean()
        )
        teacher_valid = target[:, valid]
        prediction_valid = prediction[:, valid]
        top = teacher_valid.topk(k=2, dim=0).indices
        query = torch.arange(top.shape[1], device=student.device)
        teacher_gap = teacher_valid[top[0], query] - teacher_valid[top[1], query]
        predicted_gap = (
            prediction_valid[top[0], query]
            - prediction_valid[top[1], query]
        )
        margin_losses.append(F.smooth_l1_loss(predicted_gap, teacher_gap))
    if not listwise_losses:
        raise ValueError("scene batches must contain a non-degenerate multi-row scene")
    return torch.stack(listwise_losses).mean(), torch.stack(margin_losses).mean()


def balanced_latent_relation_loss(
    student_descriptors: torch.Tensor,
    teacher_descriptors: torch.Tensor,
    teacher_mask: torch.Tensor,
    scene_ids: Sequence[str],
) -> torch.Tensor:
    """Balance low/mid/high within-scene set relations equally.

    This is a semantic-boundary proxy available in the current cache.  It is
    deliberately not described as a physical SAM boundary target.
    """

    student = F.normalize(
        torch.as_tensor(student_descriptors).float(), dim=-1, eps=1e-8
    )
    teacher = F.normalize(
        torch.as_tensor(teacher_descriptors, device=student.device).float(),
        dim=-1,
        eps=1e-8,
    )
    mask = torch.as_tensor(teacher_mask, device=student.device).bool()
    if (
        student.ndim != 3
        or teacher.ndim != 3
        or student.shape[0] != teacher.shape[0]
        or student.shape[-1] != teacher.shape[-1]
        or mask.shape != teacher.shape[:2]
        or len(scene_ids) != student.shape[0]
    ):
        raise ValueError("latent relation inputs do not align")
    losses: list[torch.Tensor] = []
    for scene in sorted(set(str(value) for value in scene_ids)):
        rows = torch.tensor(
            [index for index, value in enumerate(scene_ids) if str(value) == scene],
            dtype=torch.long,
            device=student.device,
        )
        if rows.numel() < 2:
            continue
        student_scene = student[rows]
        teacher_scene = teacher[rows]
        teacher_scene_mask = mask[rows]
        student_pair = torch.einsum(
            "ikd,jld->ijkl", student_scene, student_scene
        ).amax(dim=(-1, -2))
        teacher_modes = torch.einsum(
            "ivd,jwd->ijvw", teacher_scene, teacher_scene
        )
        pair_mask = (
            teacher_scene_mask[:, None, :, None]
            & teacher_scene_mask[None, :, None, :]
        )
        teacher_pair = teacher_modes.masked_fill(
            ~pair_mask, torch.finfo(teacher_modes.dtype).min
        ).amax(dim=(-1, -2))
        off_diagonal = ~torch.eye(
            len(rows), dtype=torch.bool, device=student.device
        )
        target = teacher_pair[off_diagonal]
        predicted = student_pair[off_diagonal]
        if target.numel() < 3:
            losses.append(F.smooth_l1_loss(predicted, target))
            continue
        boundaries = torch.quantile(
            target.detach(),
            torch.tensor([1.0 / 3.0, 2.0 / 3.0], device=student.device),
        )
        buckets = (
            target <= boundaries[0],
            (target > boundaries[0]) & (target <= boundaries[1]),
            target > boundaries[1],
        )
        bucket_losses = [
            F.smooth_l1_loss(predicted[selection], target[selection])
            for selection in buckets
            if bool(selection.any())
        ]
        losses.append(torch.stack(bucket_losses).mean())
    if not losses:
        raise ValueError("relation batches must contain a multi-row scene")
    return torch.stack(losses).mean()


def uniform_slot_prior_loss(priors: torch.Tensor) -> torch.Tensor:
    """Keep every globally trained hypothesis alive without forcing its output."""

    values = torch.as_tensor(priors).float()
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("slot priors must be [B,K] with K >= 2")
    normalized = values / values.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return -(normalized.clamp_min(1e-12).log().mean()) - torch.log(
        torch.tensor(float(values.shape[1]), device=values.device)
    )
