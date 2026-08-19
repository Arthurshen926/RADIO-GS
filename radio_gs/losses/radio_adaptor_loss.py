"""Frozen RADIO adaptor consistency losses."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from radio_gs.models.radio_adaptors import project_feature_map_with_adaptor
from radio_gs.training.gauge_separated_capability import gauge_separated_radio


def _zero_like_features(features: torch.Tensor) -> torch.Tensor:
    return features.sum() * 0.0


def _maybe_downsample_projected(
    features: torch.Tensor,
    downsample: int,
) -> torch.Tensor:
    if downsample <= 1:
        return features
    if features.shape[-2] < downsample or features.shape[-1] < downsample:
        return features
    pooled = F.avg_pool2d(features, kernel_size=downsample, stride=downsample)
    return F.normalize(pooled, dim=1)


def _flatten_projected_tokens(
    features: torch.Tensor,
    *,
    downsample: int,
    max_tokens: int,
) -> torch.Tensor:
    features = _maybe_downsample_projected(features, downsample)
    tokens = features.flatten(2).transpose(1, 2)
    if max_tokens > 0 and tokens.shape[1] > max_tokens:
        indices = torch.linspace(
            0,
            tokens.shape[1] - 1,
            steps=max_tokens,
            device=tokens.device,
        ).round().long()
        tokens = tokens.index_select(1, indices)
    return F.normalize(tokens, dim=-1)


def _select_cross_view_anchor_indices(
    ref_source: torch.Tensor,
    ref_target: torch.Tensor,
    *,
    num_anchors: int,
    temperature: float,
    strategy: str,
) -> torch.Tensor:
    anchors = min(int(num_anchors), int(ref_source.shape[0]))
    if anchors <= 0:
        raise ValueError("num_anchors must be positive")
    if strategy == "linspace":
        return torch.linspace(
            0,
            ref_source.shape[0] - 1,
            steps=anchors,
            device=ref_source.device,
        ).round().long()
    if strategy != "distinctive":
        raise ValueError("anchor_strategy must be 'linspace' or 'distinctive'")

    with torch.no_grad():
        logits = (ref_source @ ref_target.transpose(0, 1)) / temperature
        prob = F.softmax(logits, dim=-1)
        topk = prob.topk(k=min(2, prob.shape[-1]), dim=-1).values
        if topk.shape[-1] == 1:
            score = topk[:, 0]
        else:
            score = topk[:, 0] - topk[:, 1]
        _, indices = score.topk(k=anchors, largest=True, sorted=False)
    return indices.sort().values


def _select_batched_self_anchor_indices(
    ref_tokens: torch.Tensor,
    *,
    num_anchors: int,
    temperature: float,
    strategy: str,
) -> torch.Tensor:
    anchors = min(int(num_anchors), int(ref_tokens.shape[1]))
    if anchors <= 0:
        raise ValueError("num_anchors must be positive")
    if strategy == "linspace":
        return torch.linspace(
            0,
            ref_tokens.shape[1] - 1,
            steps=anchors,
            device=ref_tokens.device,
        ).round().long()
    if strategy != "distinctive":
        raise ValueError("anchor_strategy must be 'linspace' or 'distinctive'")

    with torch.no_grad():
        logits = torch.matmul(ref_tokens, ref_tokens.transpose(1, 2)) / temperature
        prob = F.softmax(logits, dim=-1)
        topk = prob.topk(k=min(2, prob.shape[-1]), dim=-1).values
        if topk.shape[-1] == 1:
            score = topk[..., 0]
        else:
            score = topk[..., 0] - topk[..., 1]
        score = score.mean(dim=0)
        _, indices = score.topk(k=anchors, largest=True, sorted=False)
    return indices.sort().values


def compute_radio_adaptor_alignment_loss(
    decoded: torch.Tensor,
    target: torch.Tensor,
    adaptors: Mapping[str, nn.Module],
    *,
    projection_amp: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match decoded and teacher RADIO features in frozen adaptor spaces.

    Returns an unweighted mean cosine-distance loss and per-adaptor scalar
    losses.  Callers are responsible for multiplying the configured weight.
    """
    if not adaptors:
        return decoded.sum() * 0.0, {}

    losses: dict[str, torch.Tensor] = {}
    for name, adaptor in adaptors.items():
        pred = project_feature_map_with_adaptor(
            decoded, adaptor, amp=bool(projection_amp)
        )
        with torch.no_grad():
            ref = project_feature_map_with_adaptor(
                target, adaptor, amp=bool(projection_amp)
            )
        losses[name] = 1.0 - (pred * ref).sum(dim=1).mean()

    total = torch.stack(list(losses.values())).mean()
    return total, losses


def _masked_local_affinity_values(
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    radius: int,
) -> torch.Tensor:
    """Return local cosine affinities whose two endpoints are visible."""

    if radius <= 0:
        raise ValueError("radius must be positive")
    if features.ndim != 4 or valid_mask.shape != (
        features.shape[0],
        features.shape[2],
        features.shape[3],
    ):
        raise ValueError("features/valid_mask must align as [B,C,H,W]/[B,H,W]")
    values = F.normalize(features.float(), dim=1, eps=1e-8)
    _, _, height, width = values.shape
    relations: list[torch.Tensor] = []
    for dy in range(0, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx <= 0:
                continue
            if dy * dy + dx * dx > radius * radius:
                continue
            src_y0 = max(0, -dy)
            src_y1 = min(height, height - dy)
            dst_y0 = max(0, dy)
            dst_y1 = min(height, height + dy)
            src_x0 = max(0, -dx)
            src_x1 = min(width, width - dx)
            dst_x0 = max(0, dx)
            dst_x1 = min(width, width + dx)
            if src_y1 <= src_y0 or src_x1 <= src_x0:
                continue
            pair_valid = (
                valid_mask[:, src_y0:src_y1, src_x0:src_x1]
                & valid_mask[:, dst_y0:dst_y1, dst_x0:dst_x1]
            )
            relation = (
                values[:, :, src_y0:src_y1, src_x0:src_x1]
                * values[:, :, dst_y0:dst_y1, dst_x0:dst_x1]
            ).sum(dim=1)
            if bool(pair_valid.any()):
                relations.append(relation[pair_valid])
    if not relations:
        return features.new_empty(0, dtype=torch.float32)
    return torch.cat(relations)


def compute_radio_adaptor_masked_render_losses(
    decoded: torch.Tensor,
    target: torch.Tensor,
    adaptors: Mapping[str, nn.Module],
    valid_mask: torch.Tensor,
    *,
    adaptor_weights: Mapping[str, float] | None = None,
    local_radius: int = 1,
    local_balance_quantile: float = 0.0,
    teacher_capability_maps: Mapping[str, torch.Tensor] | None = None,
    gauge_separated: bool = True,
    projection_amp: bool = False,
    projection_checkpoint: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, dict[str, torch.Tensor]]]:
    """Preserve official dense capability and local relations after rendering.

    The loss is query-free and only supervises pixels supported by rendered
    alpha.  By default frozen official adaptor maps are evaluated on the
    complete 2-D RADIO grid before visible pixels or local pairs are selected.
    ``teacher_capability_maps`` optionally supplies those already-evaluated
    native official maps, which avoids incorrectly applying an adaptor a
    second time after raw-map interpolation.
    """

    if decoded.shape != target.shape or decoded.ndim != 4:
        raise ValueError("decoded/target must be matching [B,C,H,W]")
    valid = torch.as_tensor(valid_mask, device=decoded.device).bool()
    if valid.ndim == 4 and valid.shape[1] == 1:
        valid = valid[:, 0]
    if valid.shape != (decoded.shape[0], decoded.shape[2], decoded.shape[3]):
        raise ValueError("valid_mask must align as [B,H,W] or [B,1,H,W]")
    requested = {
        str(name): float(weight)
        for name, weight in dict(adaptor_weights or {}).items()
        if float(weight) > 0
    }
    if not requested:
        requested = {str(name): 1.0 for name in adaptors}
    missing = sorted(set(requested) - set(adaptors))
    if missing:
        raise ValueError(f"missing requested official adaptors: {missing}")
    if teacher_capability_maps is not None:
        missing_teacher = sorted(set(requested) - set(teacher_capability_maps))
        if missing_teacher:
            raise ValueError(
                f"missing requested precomputed official capability maps: {missing_teacher}"
            )
    if not requested or not bool(valid.any()):
        zero = _zero_like_features(decoded)
        return zero, zero, {}

    predicted_radio = (
        gauge_separated_radio(decoded, feature_dim=1)
        if bool(gauge_separated)
        else decoded
    )
    total_weight = sum(requested.values())
    alignment_total = _zero_like_features(decoded)
    local_total = _zero_like_features(decoded)
    details: dict[str, dict[str, torch.Tensor]] = {}
    for name, weight in requested.items():
        adaptor = adaptors[name]
        predicted = project_feature_map_with_adaptor(
            predicted_radio,
            adaptor,
            amp=bool(projection_amp),
            checkpoint_adaptor=bool(projection_checkpoint),
        )
        if teacher_capability_maps is None:
            with torch.no_grad():
                teacher = project_feature_map_with_adaptor(
                    target, adaptor, amp=bool(projection_amp)
                )
        else:
            with torch.no_grad():
                teacher = F.normalize(
                    torch.as_tensor(teacher_capability_maps[name])
                    .to(device=predicted.device, dtype=predicted.dtype),
                    dim=1,
                    eps=1e-8,
                )
            if teacher.shape != predicted.shape:
                raise ValueError(
                    f"precomputed {name} capability map does not match rendered map: "
                    f"{tuple(teacher.shape)} vs {tuple(predicted.shape)}"
                )
        cosine = (predicted.float() * teacher.float()).sum(dim=1)
        alignment = 1.0 - cosine[valid].mean()
        predicted_relation = _masked_local_affinity_values(
            predicted, valid, radius=int(local_radius)
        )
        with torch.no_grad():
            teacher_relation = _masked_local_affinity_values(
                teacher, valid, radius=int(local_radius)
            )
        balance_quantile = float(local_balance_quantile)
        if not 0.0 <= balance_quantile < 0.5:
            raise ValueError("local_balance_quantile must be in [0, 0.5)")
        if predicted_relation.numel() and balance_quantile > 0.0:
            # Interior pairs vastly outnumber true discontinuities.  Balance
            # the two teacher-defined tails so the loss cannot minimize its
            # average by smoothing away SAM boundaries.  The selection is
            # query/label-free and the teacher branch remains frozen.
            lower = torch.quantile(teacher_relation, balance_quantile)
            upper = torch.quantile(teacher_relation, 1.0 - balance_quantile)
            boundary = teacher_relation <= lower
            interior = teacher_relation >= upper
            local = 0.5 * (
                F.mse_loss(predicted_relation[boundary], teacher_relation[boundary])
                + F.mse_loss(predicted_relation[interior], teacher_relation[interior])
            )
        else:
            local = (
                F.mse_loss(predicted_relation, teacher_relation)
                if predicted_relation.numel()
                else _zero_like_features(predicted)
            )
        normalized_weight = float(weight) / total_weight
        alignment_total = alignment_total + normalized_weight * alignment
        local_total = local_total + normalized_weight * local
        details[name] = {
            "alignment": alignment,
            "local_affinity": local,
            "visible_pixels": torch.as_tensor(
                int(valid.sum()), device=decoded.device
            ),
            "visible_pairs": torch.as_tensor(
                int(predicted_relation.numel()), device=decoded.device
            ),
            "local_balance_quantile": torch.as_tensor(
                balance_quantile, device=decoded.device
            ),
        }
    return alignment_total, local_total, details


def compute_radio_adaptor_cross_view_propagation_loss(
    decoded: torch.Tensor,
    target: torch.Tensor,
    adaptors: Mapping[str, nn.Module],
    *,
    downsample: int = 1,
    max_tokens: int = 256,
    num_anchors: int = 16,
    temperature: float = 0.2,
    anchor_strategy: str = "linspace",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Distill DINO-style cross-view soft mask propagation.

    The frozen teacher picks source-view anchor tokens and propagates each
    anchor to the paired target view as a soft token map.  ``linspace`` keeps
    the historical deterministic anchors; ``distinctive`` selects teacher
    tokens with the clearest cross-view top-1 margin so the loss focuses on
    reliable DINO correspondences instead of arbitrary background patches.
    """
    if not adaptors:
        return _zero_like_features(decoded), {}
    if decoded.shape[0] < 2:
        return _zero_like_features(decoded), {}
    if num_anchors <= 0:
        raise ValueError("num_anchors must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    pair_count = decoded.shape[0] // 2
    losses: dict[str, torch.Tensor] = {}
    for name, adaptor in adaptors.items():
        pred = project_feature_map_with_adaptor(decoded, adaptor)
        pred_tokens = _flatten_projected_tokens(
            pred, downsample=downsample, max_tokens=max_tokens
        )
        with torch.no_grad():
            ref = project_feature_map_with_adaptor(target, adaptor)
            ref_tokens = _flatten_projected_tokens(
                ref, downsample=downsample, max_tokens=max_tokens
            )

        per_pair: list[torch.Tensor] = []
        for pair_idx in range(pair_count):
            a = 2 * pair_idx
            b = a + 1
            ref_source = ref_tokens[a]
            ref_target = ref_tokens[b]
            anchor_idx = _select_cross_view_anchor_indices(
                ref_source,
                ref_target,
                num_anchors=num_anchors,
                temperature=temperature,
                strategy=anchor_strategy,
            )

            pred_anchor = pred_tokens[a].index_select(0, anchor_idx)
            pred_target = pred_tokens[b]
            pred_logits = (pred_anchor @ pred_target.transpose(0, 1)) / temperature
            pred_log_prob = F.log_softmax(pred_logits, dim=-1)

            with torch.no_grad():
                ref_anchor = ref_source.index_select(0, anchor_idx)
                ref_logits = (ref_anchor @ ref_target.transpose(0, 1)) / temperature
                ref_log_prob = F.log_softmax(ref_logits, dim=-1)
                ref_prob = ref_log_prob.exp()

            per_pair.append((ref_prob * (ref_log_prob - pred_log_prob)).sum(dim=-1).mean())
        losses[name] = torch.stack(per_pair).mean()

    total = torch.stack(list(losses.values())).mean()
    return total, losses


def compute_radio_adaptor_cross_view_mask_propagation_loss(
    decoded: torch.Tensor,
    target: torch.Tensor,
    adaptors: Mapping[str, nn.Module],
    *,
    downsample: int = 1,
    max_tokens: int = 256,
    num_anchors: int = 16,
    temperature: float = 0.2,
    anchor_strategy: str = "linspace",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Distill teacher soft-mask transport across paired views.

    Unlike point-anchor propagation, the frozen teacher first builds a soft
    source mask around each source anchor, transports that mask to the paired
    target view with teacher DINO affinities, and then supervises the student's
    target transport distribution.  ``distinctive`` anchors focus this objective
    on teacher-stable correspondences.
    """
    if not adaptors:
        return _zero_like_features(decoded), {}
    if decoded.shape[0] < 2:
        return _zero_like_features(decoded), {}
    if num_anchors <= 0:
        raise ValueError("num_anchors must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    pair_count = decoded.shape[0] // 2
    losses: dict[str, torch.Tensor] = {}
    for name, adaptor in adaptors.items():
        pred = project_feature_map_with_adaptor(decoded, adaptor)
        pred_tokens = _flatten_projected_tokens(
            pred, downsample=downsample, max_tokens=max_tokens
        )
        with torch.no_grad():
            ref = project_feature_map_with_adaptor(target, adaptor)
            ref_tokens = _flatten_projected_tokens(
                ref, downsample=downsample, max_tokens=max_tokens
            )

        per_pair: list[torch.Tensor] = []
        for pair_idx in range(pair_count):
            a = 2 * pair_idx
            b = a + 1

            pred_source = pred_tokens[a]
            pred_target = pred_tokens[b]
            pred_transport = F.softmax(
                (pred_source @ pred_target.transpose(0, 1)) / temperature,
                dim=-1,
            )
            pred_target_prob = pred_transport

            with torch.no_grad():
                ref_source = ref_tokens[a]
                ref_target = ref_tokens[b]
                anchor_idx = _select_cross_view_anchor_indices(
                    ref_source,
                    ref_target,
                    num_anchors=num_anchors,
                    temperature=temperature,
                    strategy=anchor_strategy,
                )
                ref_anchors = ref_source.index_select(0, anchor_idx)
                source_mask = F.softmax(
                    (ref_anchors @ ref_source.transpose(0, 1)) / temperature,
                    dim=-1,
                )
                ref_transport = F.softmax(
                    (ref_source @ ref_target.transpose(0, 1)) / temperature,
                    dim=-1,
                )
                ref_target_prob = source_mask @ ref_transport
                ref_target_prob = ref_target_prob / ref_target_prob.sum(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-8)

            pred_mask_prob = source_mask @ pred_target_prob
            pred_mask_prob = pred_mask_prob / pred_mask_prob.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-8)
            per_pair.append(
                (
                    ref_target_prob
                    * (
                        ref_target_prob.clamp_min(1e-8).log()
                        - pred_mask_prob.clamp_min(1e-8).log()
                    )
                )
                .sum(dim=-1)
                .mean()
            )
        losses[name] = torch.stack(per_pair).mean()

    total = torch.stack(list(losses.values())).mean()
    return total, losses


def compute_radio_adaptor_relation_loss(
    decoded: torch.Tensor,
    target: torch.Tensor,
    adaptors: Mapping[str, nn.Module],
    *,
    downsample: int = 1,
    max_tokens: int = 512,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match FMGS-style pixel relations in frozen adaptor spaces.

    This loss compares pairwise token similarities after adaptor projection,
    which preserves DINO-like neighborhood/part structure beyond per-pixel
    cosine matching.
    """
    if not adaptors:
        return _zero_like_features(decoded), {}
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    losses: dict[str, torch.Tensor] = {}
    for name, adaptor in adaptors.items():
        pred = project_feature_map_with_adaptor(decoded, adaptor)
        pred_tokens = _flatten_projected_tokens(
            pred, downsample=downsample, max_tokens=max_tokens
        )
        with torch.no_grad():
            ref = project_feature_map_with_adaptor(target, adaptor)
            ref_tokens = _flatten_projected_tokens(
                ref, downsample=downsample, max_tokens=max_tokens
            )
            ref_sim = torch.matmul(ref_tokens, ref_tokens.transpose(1, 2)) / temperature
        pred_sim = torch.matmul(pred_tokens, pred_tokens.transpose(1, 2)) / temperature
        losses[name] = F.mse_loss(pred_sim, ref_sim)

    total = torch.stack(list(losses.values())).mean()
    return total, losses


def _local_affinity_values(features: torch.Tensor, *, radius: int) -> torch.Tensor:
    if radius <= 0:
        raise ValueError("radius must be positive")
    features = F.normalize(features, dim=1)
    _, _, height, width = features.shape
    values: list[torch.Tensor] = []
    for dy in range(0, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx <= 0:
                continue
            if dy * dy + dx * dx > radius * radius:
                continue

            src_y0 = max(0, -dy)
            src_y1 = min(height, height - dy)
            dst_y0 = max(0, dy)
            dst_y1 = min(height, height + dy)
            src_x0 = max(0, -dx)
            src_x1 = min(width, width - dx)
            dst_x0 = max(0, dx)
            dst_x1 = min(width, width + dx)
            if src_y1 <= src_y0 or src_x1 <= src_x0:
                continue

            src = features[:, :, src_y0:src_y1, src_x0:src_x1]
            dst = features[:, :, dst_y0:dst_y1, dst_x0:dst_x1]
            values.append((src * dst).sum(dim=1).flatten(1))
    if not values:
        return features.sum(dim=(1, 2, 3), keepdim=False)[:, None] * 0.0
    return torch.cat(values, dim=1)


def compute_radio_adaptor_local_affinity_loss(
    decoded: torch.Tensor,
    target: torch.Tensor,
    adaptors: Mapping[str, nn.Module],
    *,
    downsample: int = 1,
    radius: int = 1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Preserve local DINO-style neighborhood topology.

    Full pairwise relation loss captures global token structure, but DINO mask
    propagation is especially sensitive to local patch neighborhoods.  This
    loss matches frozen-adaptor cosine affinities for nearby spatial offsets,
    keeping local part boundaries and same-object continuity intact.
    """
    if not adaptors:
        return _zero_like_features(decoded), {}
    if radius <= 0:
        raise ValueError("radius must be positive")

    losses: dict[str, torch.Tensor] = {}
    for name, adaptor in adaptors.items():
        pred = project_feature_map_with_adaptor(decoded, adaptor)
        pred = _maybe_downsample_projected(pred, downsample)
        with torch.no_grad():
            ref = project_feature_map_with_adaptor(target, adaptor)
            ref = _maybe_downsample_projected(ref, downsample)
            ref_affinity = _local_affinity_values(ref, radius=radius)

        pred_affinity = _local_affinity_values(pred, radius=radius)
        losses[name] = F.mse_loss(pred_affinity, ref_affinity)

    total = torch.stack(list(losses.values())).mean()
    return total, losses


def compute_radio_adaptor_token_contrast_loss(
    decoded: torch.Tensor,
    target: torch.Tensor,
    adaptors: Mapping[str, nn.Module],
    *,
    downsample: int = 1,
    max_tokens: int = 512,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Distill token-level hard-negative structure in frozen adaptor spaces.

    Each predicted token is contrasted against all teacher tokens from the same
    view, with its same-location teacher token as the positive.  The loss is
    teacher-relative: subtracting the frozen teacher self-contrast makes
    identical predicted/teacher features exactly zero even when teacher tokens
    are visually ambiguous.
    """
    if not adaptors:
        return _zero_like_features(decoded), {}
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    losses: dict[str, torch.Tensor] = {}
    for name, adaptor in adaptors.items():
        pred = project_feature_map_with_adaptor(decoded, adaptor)
        pred_tokens = _flatten_projected_tokens(
            pred, downsample=downsample, max_tokens=max_tokens
        )
        with torch.no_grad():
            ref = project_feature_map_with_adaptor(target, adaptor)
            ref_tokens = _flatten_projected_tokens(
                ref, downsample=downsample, max_tokens=max_tokens
            )
            labels = torch.arange(ref_tokens.shape[1], device=ref_tokens.device)
            ref_logits = torch.matmul(
                ref_tokens, ref_tokens.transpose(1, 2)
            ) / temperature
            ref_loss = F.cross_entropy(
                ref_logits.reshape(-1, ref_logits.shape[-1]),
                labels.repeat(ref_logits.shape[0]),
            )

        pred_logits = torch.matmul(
            pred_tokens, ref_tokens.transpose(1, 2)
        ) / temperature
        pred_loss = F.cross_entropy(
            pred_logits.reshape(-1, pred_logits.shape[-1]),
            labels.repeat(pred_logits.shape[0]),
        )
        losses[name] = (pred_loss - ref_loss).clamp_min(0.0)

    total = torch.stack(list(losses.values())).mean()
    return total, losses


def compute_radio_adaptor_peak_background_loss(
    decoded: torch.Tensor,
    target: torch.Tensor,
    adaptors: Mapping[str, nn.Module],
    *,
    downsample: int = 1,
    max_tokens: int = 512,
    num_anchors: int = 16,
    temperature: float = 0.2,
    anchor_strategy: str = "linspace",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Preserve DINO-style peak/background margins around teacher anchors.

    Dense propagation fails when a rendered feature keeps the coarse match but
    flattens the target peak or raises visually similar background tokens.  For
    deterministic teacher anchors, this loss compares the teacher's strongest
    token against its hardest non-peak token and only penalizes student margins
    that fall below the frozen teacher margin.
    """
    if not adaptors:
        return _zero_like_features(decoded), {}
    if num_anchors <= 0:
        raise ValueError("num_anchors must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    losses: dict[str, torch.Tensor] = {}
    for name, adaptor in adaptors.items():
        pred = project_feature_map_with_adaptor(decoded, adaptor)
        pred_tokens = _flatten_projected_tokens(
            pred, downsample=downsample, max_tokens=max_tokens
        )
        with torch.no_grad():
            ref = project_feature_map_with_adaptor(target, adaptor)
            ref_tokens = _flatten_projected_tokens(
                ref, downsample=downsample, max_tokens=max_tokens
            )
            anchor_idx = _select_batched_self_anchor_indices(
                num_anchors=int(num_anchors),
                temperature=temperature,
                strategy=anchor_strategy,
                ref_tokens=ref_tokens,
            )
            ref_anchor = ref_tokens.index_select(1, anchor_idx)
            ref_logits = torch.matmul(ref_anchor, ref_tokens.transpose(1, 2)) / temperature
            peak_idx = ref_logits.argmax(dim=-1)
            if ref_logits.shape[-1] > 1:
                hard_negative_logits = ref_logits.masked_fill(
                    F.one_hot(peak_idx, num_classes=ref_logits.shape[-1]).bool(),
                    -torch.inf,
                )
                background_idx = hard_negative_logits.argmax(dim=-1)
            else:
                background_idx = peak_idx
            ref_peak = ref_logits.gather(-1, peak_idx.unsqueeze(-1)).squeeze(-1)
            ref_background = ref_logits.gather(-1, background_idx.unsqueeze(-1)).squeeze(-1)
            ref_margin_loss = F.softplus(-(ref_peak - ref_background))

        pred_anchor = pred_tokens.index_select(1, anchor_idx)
        pred_logits = torch.matmul(pred_anchor, pred_tokens.transpose(1, 2)) / temperature
        pred_peak = pred_logits.gather(-1, peak_idx.unsqueeze(-1)).squeeze(-1)
        pred_background = pred_logits.gather(-1, background_idx.unsqueeze(-1)).squeeze(-1)
        pred_margin_loss = F.softplus(-(pred_peak - pred_background))
        losses[name] = (pred_margin_loss - ref_margin_loss).clamp_min(0.0).mean()

    total = torch.stack(list(losses.values())).mean()
    return total, losses


def compute_radio_adaptor_region_loss(
    decoded: torch.Tensor,
    target: torch.Tensor,
    adaptors: Mapping[str, nn.Module],
    *,
    downsample: int = 1,
    max_tokens: int = 512,
    num_anchors: int = 16,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match SAM-style soft region prototypes in frozen adaptor spaces.

    Without an external SAM mask cache, teacher SAM3 adaptor tokens define soft
    regions by similarity to deterministic anchor tokens.  Predicted RADIO-GS
    features are then encouraged to produce the same region prototypes.
    """
    if not adaptors:
        return _zero_like_features(decoded), {}
    if num_anchors <= 0:
        raise ValueError("num_anchors must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    losses: dict[str, torch.Tensor] = {}
    for name, adaptor in adaptors.items():
        pred = project_feature_map_with_adaptor(decoded, adaptor)
        pred_tokens = _flatten_projected_tokens(
            pred, downsample=downsample, max_tokens=max_tokens
        )
        with torch.no_grad():
            ref = project_feature_map_with_adaptor(target, adaptor)
            ref_tokens = _flatten_projected_tokens(
                ref, downsample=downsample, max_tokens=max_tokens
            )
            anchors = min(num_anchors, ref_tokens.shape[1])
            anchor_idx = torch.linspace(
                0,
                ref_tokens.shape[1] - 1,
                steps=anchors,
                device=ref_tokens.device,
            ).round().long()
            anchor_tokens = ref_tokens.index_select(1, anchor_idx)
            logits = torch.matmul(ref_tokens, anchor_tokens.transpose(1, 2)) / temperature
            weights = F.softmax(logits, dim=-1).transpose(1, 2)
            denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            ref_proto = torch.matmul(weights, ref_tokens) / denom
            ref_proto = F.normalize(ref_proto, dim=-1)

        pred_proto = torch.matmul(weights, pred_tokens) / denom
        pred_proto = F.normalize(pred_proto, dim=-1)
        losses[name] = 1.0 - (pred_proto * ref_proto).sum(dim=-1).mean()

    total = torch.stack(list(losses.values())).mean()
    return total, losses


def compute_radio_adaptor_mask_logit_loss(
    decoded: torch.Tensor,
    target: torch.Tensor,
    adaptors: Mapping[str, nn.Module],
    *,
    downsample: int = 1,
    max_tokens: int = 512,
    num_anchors: int = 16,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Distill SAM-style soft mask logits in frozen adaptor space.

    This is the mask-logit fallback used when an external frozen SAM3 decoder
    is unavailable. Teacher adaptor tokens define deterministic mask anchors;
    the loss matches the per-token soft assignment logits from decoded features
    to those teacher anchors. It is stricter than prototype-only region loss
    because every spatial token must preserve its teacher mask distribution.
    """
    if not adaptors:
        return _zero_like_features(decoded), {}
    if num_anchors <= 0:
        raise ValueError("num_anchors must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    losses: dict[str, torch.Tensor] = {}
    for name, adaptor in adaptors.items():
        pred = project_feature_map_with_adaptor(decoded, adaptor)
        pred_tokens = _flatten_projected_tokens(
            pred, downsample=downsample, max_tokens=max_tokens
        )
        with torch.no_grad():
            ref = project_feature_map_with_adaptor(target, adaptor)
            ref_tokens = _flatten_projected_tokens(
                ref, downsample=downsample, max_tokens=max_tokens
            )
            anchors = min(num_anchors, ref_tokens.shape[1])
            anchor_idx = torch.linspace(
                0,
                ref_tokens.shape[1] - 1,
                steps=anchors,
                device=ref_tokens.device,
            ).round().long()
            anchor_tokens = ref_tokens.index_select(1, anchor_idx)
            ref_logits = torch.matmul(ref_tokens, anchor_tokens.transpose(1, 2)) / temperature
            ref_prob = F.softmax(ref_logits, dim=-1)

        pred_logits = torch.matmul(pred_tokens, anchor_tokens.transpose(1, 2)) / temperature
        pred_log_prob = F.log_softmax(pred_logits, dim=-1)
        losses[name] = -(ref_prob * pred_log_prob).sum(dim=-1).mean() + (
            ref_prob * ref_prob.clamp_min(1e-8).log()
        ).sum(dim=-1).mean()

    total = torch.stack(list(losses.values())).mean()
    return total, losses


def compute_radio_adaptor_cross_view_loss(
    decoded: torch.Tensor,
    target: torch.Tensor,
    adaptors: Mapping[str, nn.Module],
    *,
    downsample: int = 1,
    max_tokens: int = 256,
    temperature: float = 1.0,
    objective: str = "mse",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match ProFuse-style cross-view DINO token relations.

    Consecutive views in the batch are paired.  The frozen teacher adaptor
    defines the cross-view token similarity matrix for each pair, and rendered
    features are trained to reproduce the same matrix after adaptor projection.
    This gives a view-registration/context signal without changing the 1280d
    RADIO feature interface.
    """
    if not adaptors:
        return _zero_like_features(decoded), {}
    if decoded.shape[0] < 2:
        return _zero_like_features(decoded), {}
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if objective not in {"mse", "transport_cycle"}:
        raise ValueError("objective must be 'mse' or 'transport_cycle'")

    pair_count = decoded.shape[0] // 2
    losses: dict[str, torch.Tensor] = {}
    for name, adaptor in adaptors.items():
        pred = project_feature_map_with_adaptor(decoded, adaptor)
        pred_tokens = _flatten_projected_tokens(
            pred, downsample=downsample, max_tokens=max_tokens
        )
        with torch.no_grad():
            ref = project_feature_map_with_adaptor(target, adaptor)
            ref_tokens = _flatten_projected_tokens(
                ref, downsample=downsample, max_tokens=max_tokens
            )

        per_pair: list[torch.Tensor] = []
        for pair_idx in range(pair_count):
            a = 2 * pair_idx
            b = a + 1
            pred_sim = (
                pred_tokens[a] @ pred_tokens[b].transpose(0, 1)
            ) / temperature
            with torch.no_grad():
                ref_sim = (
                    ref_tokens[a] @ ref_tokens[b].transpose(0, 1)
                ) / temperature
            if objective == "mse":
                per_pair.append(F.mse_loss(pred_sim, ref_sim))
                continue

            pred_ab_log = F.log_softmax(pred_sim, dim=-1)
            pred_ba_log = F.log_softmax(pred_sim.transpose(0, 1), dim=-1)
            pred_ab = pred_ab_log.exp()
            pred_ba = pred_ba_log.exp()
            with torch.no_grad():
                ref_ab_log = F.log_softmax(ref_sim, dim=-1)
                ref_ba_log = F.log_softmax(ref_sim.transpose(0, 1), dim=-1)
                ref_ab = ref_ab_log.exp()
                ref_ba = ref_ba_log.exp()
                ref_cycle = ref_ab @ ref_ba
            kl_ab = (ref_ab * (ref_ab_log - pred_ab_log)).sum(dim=-1).mean()
            kl_ba = (ref_ba * (ref_ba_log - pred_ba_log)).sum(dim=-1).mean()
            cycle = F.mse_loss(pred_ab @ pred_ba, ref_cycle)
            per_pair.append(0.5 * (kl_ab + kl_ba) + 0.1 * cycle)
        losses[name] = torch.stack(per_pair).mean()

    total = torch.stack(list(losses.values())).mean()
    return total, losses
