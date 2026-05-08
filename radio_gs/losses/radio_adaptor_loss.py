"""Frozen RADIO adaptor consistency losses."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from radio_gs.models.radio_adaptors import project_feature_map_with_adaptor


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


def compute_radio_adaptor_alignment_loss(
    decoded: torch.Tensor,
    target: torch.Tensor,
    adaptors: Mapping[str, nn.Module],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match decoded and teacher RADIO features in frozen adaptor spaces.

    Returns an unweighted mean cosine-distance loss and per-adaptor scalar
    losses.  Callers are responsible for multiplying the configured weight.
    """
    if not adaptors:
        return decoded.sum() * 0.0, {}

    losses: dict[str, torch.Tensor] = {}
    for name, adaptor in adaptors.items():
        pred = project_feature_map_with_adaptor(decoded, adaptor)
        with torch.no_grad():
            ref = project_feature_map_with_adaptor(target, adaptor)
        losses[name] = 1.0 - (pred * ref).sum(dim=1).mean()

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


def compute_radio_adaptor_cross_view_loss(
    decoded: torch.Tensor,
    target: torch.Tensor,
    adaptors: Mapping[str, nn.Module],
    *,
    downsample: int = 1,
    max_tokens: int = 256,
    temperature: float = 1.0,
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
            per_pair.append(F.mse_loss(pred_sim, ref_sim))
        losses[name] = torch.stack(per_pair).mean()

    total = torch.stack(list(losses.values())).mean()
    return total, losses
