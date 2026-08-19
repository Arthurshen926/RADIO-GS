"""Typed loader and losses for optional official foundation-model caches."""

from __future__ import annotations

from collections.abc import Iterable as IterableABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FoundationCacheProducer:
    """Metadata describing how one foundation cache head was produced."""

    official: bool = False
    backend: str = ""
    decoder: str = ""
    source: str = ""


@dataclass(frozen=True)
class FoundationHeadCache:
    """Payload for one official foundation head cache."""

    mask_logits: torch.Tensor | None = None
    mask_tensor_semantics: str | None = None
    tokens: torch.Tensor | None = None
    feature_map: torch.Tensor | None = None
    queries: tuple[str, ...] = ()
    scores: torch.Tensor | None = None
    boxes_xyxy: torch.Tensor | None = None
    mask_query_indices: torch.Tensor | None = None
    mask_query_ranks: torch.Tensor | None = None
    producer: FoundationCacheProducer | None = None


@dataclass(frozen=True)
class FoundationCache:
    """Typed foundation cache payload for one frame."""

    version: int
    frame_id: int | str
    heads: dict[str, FoundationHeadCache]


def _as_mapping(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("foundation cache must be a mapping")
    return payload


def _require_tensor(head: str, key: str, value: Any, ndim: int) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise ValueError(f"{head}.{key} must be a torch.Tensor")
    if value.dim() != ndim:
        raise ValueError(f"{head}.{key} must have {ndim} dimensions")
    return value.detach()


def _parse_producer(name: str, payload: Any, *, require_official: bool) -> FoundationCacheProducer | None:
    if payload is None:
        if require_official:
            raise ValueError(f"{name} cache requires official producer metadata")
        return None
    data = _as_mapping(payload)
    producer = FoundationCacheProducer(
        official=bool(data.get("official", False)),
        backend=str(data.get("backend", "") or ""),
        decoder=str(data.get("decoder", "") or ""),
        source=str(data.get("source", "") or ""),
    )
    if require_official and not producer.official:
        raise ValueError(f"{name} cache requires official producer metadata")
    if producer.official and not producer.backend:
        raise ValueError(f"{name} official cache producer must include backend")
    return producer


def _parse_head_cache(
    name: str,
    payload: Any,
    *,
    require_official: bool,
) -> FoundationHeadCache:
    data = _as_mapping(payload)
    mask_logits = None
    tokens = None
    feature_map = None
    scores = None
    boxes_xyxy = None
    mask_query_indices = None
    mask_query_ranks = None
    mask_tensor_semantics = None
    queries: tuple[str, ...] = ()
    if "mask_logits" in data:
        mask_logits = _require_tensor(name, "mask_logits", data["mask_logits"], 3)
    if "mask_tensor_semantics" in data:
        mask_tensor_semantics = str(data["mask_tensor_semantics"] or "") or None
    if "tokens" in data:
        tokens = _require_tensor(name, "tokens", data["tokens"], 2)
    if "feature_map" in data:
        feature_map = _require_tensor(name, "feature_map", data["feature_map"], 3)
    if "scores" in data:
        scores = _require_tensor(name, "scores", data["scores"], 1)
    if "boxes_xyxy" in data:
        boxes_xyxy = _require_tensor(name, "boxes_xyxy", data["boxes_xyxy"], 2)
        if boxes_xyxy.shape[-1] != 4:
            raise ValueError(f"{name}.boxes_xyxy must have shape [N,4]")
    if "mask_query_indices" in data:
        mask_query_indices = _require_tensor(
            name,
            "mask_query_indices",
            data["mask_query_indices"],
            1,
        ).long()
    if "mask_query_ranks" in data:
        mask_query_ranks = _require_tensor(
            name,
            "mask_query_ranks",
            data["mask_query_ranks"],
            1,
        ).long()
    if "queries" in data:
        queries_payload = data["queries"]
        if queries_payload is None:
            queries = ()
        elif isinstance(queries_payload, IterableABC) and not isinstance(
            queries_payload, (str, bytes)
        ):
            queries = tuple(str(query) for query in queries_payload)
        else:
            raise ValueError(f"{name}.queries must be a sequence of strings")
    if scores is not None and mask_logits is not None and scores.shape[0] != mask_logits.shape[0]:
        raise ValueError(f"{name}.scores must match mask_logits mask count")
    if (
        boxes_xyxy is not None
        and mask_logits is not None
        and boxes_xyxy.shape[0] != mask_logits.shape[0]
    ):
        raise ValueError(f"{name}.boxes_xyxy must match mask_logits mask count")
    if (
        mask_query_indices is not None
        and mask_logits is not None
        and mask_query_indices.shape[0] != mask_logits.shape[0]
    ):
        raise ValueError(f"{name}.mask_query_indices must match mask_logits mask count")
    if (
        mask_query_ranks is not None
        and mask_logits is not None
        and mask_query_ranks.shape[0] != mask_logits.shape[0]
    ):
        raise ValueError(f"{name}.mask_query_ranks must match mask_logits mask count")
    if mask_query_indices is not None and mask_query_indices.numel() > 0 and queries:
        if int(mask_query_indices.min().item()) < 0 or int(mask_query_indices.max().item()) >= len(queries):
            raise ValueError(f"{name}.mask_query_indices contains query ids outside queries")
    if mask_logits is None and tokens is None and feature_map is None:
        raise ValueError(f"{name} cache must include mask_logits, tokens, or feature_map")
    producer = _parse_producer(
        name,
        data.get("producer"),
        require_official=require_official,
    )
    return FoundationHeadCache(
        mask_logits=mask_logits,
        mask_tensor_semantics=mask_tensor_semantics,
        tokens=tokens,
        feature_map=feature_map,
        queries=queries,
        scores=scores,
        boxes_xyxy=boxes_xyxy,
        mask_query_indices=mask_query_indices,
        mask_query_ranks=mask_query_ranks,
        producer=producer,
    )


def load_foundation_cache(
    path_or_payload: str | Path | Mapping[str, Any],
    *,
    require_official: bool = False,
) -> FoundationCache:
    """Load and validate a version-1 official foundation cache payload."""

    if isinstance(path_or_payload, (str, Path)):
        payload = torch.load(Path(path_or_payload), map_location="cpu")
    else:
        payload = path_or_payload
    data = _as_mapping(payload)

    version = int(data.get("version", -1))
    if version != 1:
        raise ValueError(f"unsupported foundation cache version: {version}")
    if "frame_id" not in data:
        raise ValueError("foundation cache is missing frame_id")

    heads_payload = _as_mapping(data.get("heads"))
    heads = {
        str(name): _parse_head_cache(
            str(name),
            head_payload,
            require_official=require_official,
        )
        for name, head_payload in heads_payload.items()
    }
    if not heads:
        raise ValueError("foundation cache must contain at least one head")
    return FoundationCache(
        version=version,
        frame_id=data["frame_id"],
        heads=heads,
    )


def _zero_like(decoded_features: torch.Tensor) -> torch.Tensor:
    return decoded_features.sum() * 0.0


def _normalise_heads(heads: str | Iterable[str] | None) -> set[str] | None:
    if heads is None:
        return None
    if isinstance(heads, str):
        values = [part.strip() for part in heads.split(",")]
    else:
        values = [str(part).strip() for part in heads]
    selected = {value for value in values if value}
    return selected or None


def _project(decoded_features: torch.Tensor, projector: Any) -> torch.Tensor:
    if projector is None:
        return decoded_features
    return projector(decoded_features)


def _to_batched_tokens(features: torch.Tensor) -> torch.Tensor:
    if features.dim() == 2:
        return features.unsqueeze(0)
    if features.dim() == 3:
        return features
    if features.dim() == 4:
        return features.flatten(2).transpose(1, 2)
    raise ValueError("projected token features must be [N,C], [B,N,C], or [B,C,H,W]")


def _target_tokens_from_cache(head_cache: FoundationHeadCache) -> torch.Tensor | None:
    if head_cache.tokens is not None:
        return head_cache.tokens
    if head_cache.feature_map is not None:
        return head_cache.feature_map.flatten(1).transpose(0, 1)
    return None


def _align_token_count(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    count = min(pred.shape[1], target.shape[1])
    if count <= 0:
        raise ValueError("token supervision requires at least one token")
    return pred[:, :count], target[:, :count]


def _mask_logits_from_projection(projected: torch.Tensor) -> torch.Tensor:
    if projected.dim() == 3:
        return projected.unsqueeze(0)
    if projected.dim() == 4:
        return projected
    raise ValueError("projected mask logits must be [M,H,W] or [B,M,H,W]")


def _mask_boundary_response(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits.float())
    dx = (probs[:, :, :, 1:] - probs[:, :, :, :-1]).abs()
    dy = (probs[:, :, 1:, :] - probs[:, :, :-1, :]).abs()
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return (dx.square() + dy.square() + 1e-8).sqrt()


def _normalise_map(response: torch.Tensor) -> torch.Tensor:
    denom = response.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return response / denom


def _feature_boundary_response(features: torch.Tensor) -> torch.Tensor:
    normalised = F.normalize(features.float(), dim=1)
    dx = (normalised[:, :, :, 1:] - normalised[:, :, :, :-1]).square().sum(
        dim=1,
        keepdim=True,
    )
    dy = (normalised[:, :, 1:, :] - normalised[:, :, :-1, :]).square().sum(
        dim=1,
        keepdim=True,
    )
    dx = F.pad(dx.sqrt(), (0, 1, 0, 0))
    dy = F.pad(dy.sqrt(), (0, 0, 0, 1))
    return (dx.square() + dy.square() + 1e-8).sqrt()


def _prepare_sam_region_targets(
    head_cache: FoundationHeadCache,
    *,
    spatial_size: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
    score_threshold: float,
    max_masks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_cache.mask_logits is None:
        empty_logits = torch.empty(1, 0, *spatial_size, device=device, dtype=dtype)
        empty_weights = torch.empty(0, device=device, dtype=dtype)
        return empty_logits, empty_weights

    logits = head_cache.mask_logits.to(device=device, dtype=dtype).unsqueeze(0)
    if logits.shape[-2:] != spatial_size:
        logits = F.interpolate(
            logits,
            size=spatial_size,
            mode="bilinear",
            align_corners=False,
        )
    mask_count = logits.shape[1]
    if head_cache.scores is not None:
        scores = head_cache.scores.to(device=device, dtype=dtype)
        scores = scores[:mask_count]
    else:
        scores = torch.ones(mask_count, device=device, dtype=dtype)

    keep = torch.ones(mask_count, device=device, dtype=torch.bool)
    if score_threshold > 0:
        keep &= scores >= score_threshold
    if not bool(keep.any()):
        empty_logits = logits[:, :0]
        empty_weights = scores[:0]
        return empty_logits, empty_weights

    indices = torch.nonzero(keep, as_tuple=False).flatten()
    if head_cache.scores is not None:
        order = torch.argsort(scores[indices], descending=True)
        indices = indices[order]
    if max_masks > 0:
        indices = indices[:max_masks]

    logits = logits[:, indices]
    weights = scores[indices].clamp_min(0.0)
    if weights.numel() > 0:
        weights = weights / weights.mean().clamp_min(1e-6)
    return logits, weights


def _sam_region_compactness_loss(
    decoded_features: torch.Tensor,
    target_logits: torch.Tensor,
    mask_weights: torch.Tensor,
) -> torch.Tensor:
    if target_logits.shape[1] == 0:
        return _zero_like(decoded_features)
    features = F.normalize(decoded_features.float(), dim=1)
    masks = torch.sigmoid(target_logits.float()).clamp(0.0, 1.0)
    area = masks.sum(dim=(-2, -1)).clamp_min(1e-6)
    means = torch.einsum("bmhw,bchw->bmc", masks, features) / area.unsqueeze(-1)
    means = F.normalize(means, dim=-1)
    similarity = torch.einsum("bchw,bmc->bmhw", features, means)
    per_mask = ((1.0 - similarity).clamp_min(0.0) * masks).sum(dim=(-2, -1)) / area
    weights = mask_weights.to(device=per_mask.device, dtype=per_mask.dtype).view(1, -1)
    return (per_mask * weights).sum() / weights.sum().clamp_min(1e-6)


def _sam_region_separation_loss(
    decoded_features: torch.Tensor,
    target_logits: torch.Tensor,
    mask_weights: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    if target_logits.shape[1] < 2:
        return _zero_like(decoded_features)
    features = F.normalize(decoded_features.float(), dim=1)
    masks = torch.sigmoid(target_logits.float()).clamp(0.0, 1.0)
    area = masks.sum(dim=(-2, -1)).clamp_min(1e-6)
    means = torch.einsum("bmhw,bchw->bmc", masks, features) / area.unsqueeze(-1)
    means = F.normalize(means, dim=-1)

    flat_masks = masks.flatten(2)
    inter = torch.einsum("bmi,bni->bmn", flat_masks, flat_masks)
    flat_area = flat_masks.sum(dim=-1).clamp_min(1e-6)
    union = flat_area[:, :, None] + flat_area[:, None, :] - inter
    iou = inter / union.clamp_min(1e-6)
    pair_mask = torch.triu(
        torch.ones_like(iou, dtype=torch.bool),
        diagonal=1,
    ) & (iou < 0.05)
    if not bool(pair_mask.any()):
        return _zero_like(decoded_features)

    similarity = torch.einsum("bmc,bnc->bmn", means, means)
    pair_loss = F.relu(similarity - margin)
    weights = mask_weights.to(device=pair_loss.device, dtype=pair_loss.dtype).view(1, -1)
    pair_weights = weights[:, :, None] * weights[:, None, :]
    weighted = pair_loss * pair_weights * pair_mask.to(dtype=pair_loss.dtype)
    return weighted.sum() / (pair_weights * pair_mask.to(dtype=pair_loss.dtype)).sum().clamp_min(1e-6)


def _sam_feature_boundary_loss(
    decoded_features: torch.Tensor,
    target_logits: torch.Tensor,
) -> torch.Tensor:
    if target_logits.shape[1] == 0:
        return _zero_like(decoded_features)
    predicted = _normalise_map(_feature_boundary_response(decoded_features))
    target = _normalise_map(_mask_boundary_response(target_logits).amax(dim=1, keepdim=True))
    return F.mse_loss(predicted, target.to(device=predicted.device, dtype=predicted.dtype))


def compute_foundation_cache_supervision_loss(
    *,
    decoded_features: torch.Tensor,
    cache: FoundationCache | None,
    projectors: Mapping[str, Any],
    heads: str | Iterable[str] | None = None,
    mask_logit_weight: float = 1.0,
    mask_boundary_weight: float = 0.0,
    token_weight: float = 1.0,
    region_consistency_weight: float = 0.0,
    region_separation_weight: float = 0.0,
    feature_boundary_weight: float = 0.0,
    region_score_threshold: float = 0.0,
    region_max_masks: int = 16,
    region_separation_margin: float = 0.25,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Compute optional official-cache supervision.

    The helper is intentionally zero-safe: absent caches, unmatched heads, and
    incompatible projector outputs return a decoded-feature-connected zero loss
    rather than changing the legacy RADIO-adaptor path.
    """

    stats = {
        "enabled": 0,
        "heads": 0,
        "mask_logit_heads": 0,
        "mask_boundary_heads": 0,
        "region_consistency_heads": 0,
        "region_separation_heads": 0,
        "feature_boundary_heads": 0,
        "token_heads": 0,
        "skipped_heads": 0,
    }
    if cache is None:
        return _zero_like(decoded_features), stats

    selected_heads = _normalise_heads(heads)
    losses: list[torch.Tensor] = []
    for name, head_cache in cache.heads.items():
        if selected_heads is not None and name not in selected_heads:
            continue
        stats["heads"] += 1
        projector = projectors[name] if name in projectors else None
        used_head = False
        try:
            if projector is not None:
                projected = _project(decoded_features.float(), projector)
                target_tokens = _target_tokens_from_cache(head_cache)
                if target_tokens is not None and token_weight > 0:
                    pred_tokens = _to_batched_tokens(projected)
                    target = target_tokens.to(
                        device=pred_tokens.device,
                        dtype=pred_tokens.dtype,
                    ).unsqueeze(0)
                    target = target.expand(pred_tokens.shape[0], -1, -1)
                    pred_tokens, target = _align_token_count(pred_tokens, target)
                    losses.append(token_weight * F.mse_loss(pred_tokens, target))
                    stats["token_heads"] += 1
                    used_head = True

                if head_cache.mask_logits is not None and mask_logit_weight > 0:
                    pred_logits = _mask_logits_from_projection(projected)
                    target_logits = head_cache.mask_logits.to(
                        device=pred_logits.device,
                        dtype=pred_logits.dtype,
                    ).unsqueeze(0)
                    if pred_logits.shape[-2:] != target_logits.shape[-2:]:
                        target_logits = F.interpolate(
                            target_logits,
                            size=pred_logits.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    mask_count = min(pred_logits.shape[1], target_logits.shape[1])
                    pred_mask_logits = pred_logits[:, :mask_count]
                    target_mask_logits = target_logits[:, :mask_count].expand_as(pred_mask_logits)
                    losses.append(
                        mask_logit_weight
                        * F.mse_loss(pred_mask_logits, target_mask_logits)
                    )
                    stats["mask_logit_heads"] += 1
                    used_head = True

                if head_cache.mask_logits is not None and mask_boundary_weight > 0:
                    pred_logits = _mask_logits_from_projection(projected)
                    target_logits = head_cache.mask_logits.to(
                        device=pred_logits.device,
                        dtype=pred_logits.dtype,
                    ).unsqueeze(0)
                    if pred_logits.shape[-2:] != target_logits.shape[-2:]:
                        target_logits = F.interpolate(
                            target_logits,
                            size=pred_logits.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    mask_count = min(pred_logits.shape[1], target_logits.shape[1])
                    pred_boundary = _mask_boundary_response(pred_logits[:, :mask_count])
                    target_boundary = _mask_boundary_response(
                        target_logits[:, :mask_count].expand_as(pred_logits[:, :mask_count])
                    )
                    losses.append(mask_boundary_weight * F.mse_loss(pred_boundary, target_boundary))
                    stats["mask_boundary_heads"] += 1
                    used_head = True

            if head_cache.mask_logits is not None and (
                region_consistency_weight > 0
                or region_separation_weight > 0
                or feature_boundary_weight > 0
            ):
                region_logits, mask_weights = _prepare_sam_region_targets(
                    head_cache,
                    spatial_size=decoded_features.shape[-2:],
                    device=decoded_features.device,
                    dtype=decoded_features.dtype,
                    score_threshold=region_score_threshold,
                    max_masks=region_max_masks,
                )
                if region_logits.shape[1] > 0:
                    if region_consistency_weight > 0:
                        losses.append(
                            region_consistency_weight
                            * _sam_region_compactness_loss(
                                decoded_features,
                                region_logits,
                                mask_weights,
                            )
                        )
                        stats["region_consistency_heads"] += 1
                        used_head = True
                    if region_separation_weight > 0:
                        losses.append(
                            region_separation_weight
                            * _sam_region_separation_loss(
                                decoded_features,
                                region_logits,
                                mask_weights,
                                margin=region_separation_margin,
                            )
                        )
                        stats["region_separation_heads"] += 1
                        used_head = True
                    if feature_boundary_weight > 0:
                        losses.append(
                            feature_boundary_weight
                            * _sam_feature_boundary_loss(decoded_features, region_logits)
                        )
                        stats["feature_boundary_heads"] += 1
                        used_head = True
        except (RuntimeError, ValueError):
            stats["skipped_heads"] += 1
            continue
        if not used_head:
            stats["skipped_heads"] += 1

    if not losses:
        return _zero_like(decoded_features), stats
    stats["enabled"] = 1
    return torch.stack(losses).mean(), stats
