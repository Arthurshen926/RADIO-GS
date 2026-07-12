"""Mask-level supervision for LangSplat SAM-CLIP feature caches."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SamClipMaskEntry:
    """Raw SAM-CLIP cache paths for one frame."""

    frame_id: int
    stem: str
    feature_path: Path
    segments_path: Path


def load_samclip_mask_manifest(level_root: str | Path) -> Dict[int, SamClipMaskEntry]:
    """Load a converted SAM-CLIP manifest and index raw cache paths by frame id."""
    level_root = Path(level_root)
    manifest_path = level_root / "samclip_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"SAM-CLIP manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = payload.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError(f"Invalid SAM-CLIP manifest without outputs list: {manifest_path}")

    entries: Dict[int, SamClipMaskEntry] = {}
    for item in outputs:
        if not isinstance(item, dict):
            continue
        frame_id = int(item["frame_id"])
        entries[frame_id] = SamClipMaskEntry(
            frame_id=frame_id,
            stem=str(item.get("stem", frame_id)),
            feature_path=Path(str(item["feature"])),
            segments_path=Path(str(item["segments"])),
        )
    return entries


def _zero_losses(
    reference: torch.Tensor,
    *,
    background_loss: torch.Tensor | None = None,
    background_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    zero = reference.sum() * 0.0
    background = background_loss if background_loss is not None else zero
    return {
        "prototype_loss": zero,
        "contrastive_loss": zero,
        "background_loss": background,
        "total_loss": float(background_weight) * background,
        "valid_regions": torch.zeros((), device=reference.device, dtype=torch.float32),
    }


def _select_largest_regions(
    region_ids: torch.Tensor,
    counts: torch.Tensor,
    *,
    max_regions: int,
) -> torch.Tensor:
    if max_regions <= 0 or int(region_ids.numel()) <= max_regions:
        return torch.arange(region_ids.numel(), device=region_ids.device)
    _, order = torch.topk(counts.float(), k=max_regions, largest=True, sorted=True)
    return order.sort().values


def compute_samclip_mask_losses(
    pred_features: torch.Tensor,
    target_prototypes: torch.Tensor,
    segment_map: torch.Tensor,
    *,
    min_pixels: int = 16,
    max_regions: int = 64,
    contrastive_temperature: float = 0.07,
    prototype_weight: float = 1.0,
    contrastive_weight: float = 1.0,
    background_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Pool predicted descriptors over SAM masks and match cached CLIP prototypes.

    Args:
        pred_features: Rendered dense feature map with shape ``[C,H,W]``.
        target_prototypes: LangSplat cached CLIP prototypes with shape ``[N,C]``.
        segment_map: Integer SAM segment ids with shape ``[H,W]``.
    """
    if pred_features.dim() != 3:
        raise ValueError(f"Expected pred_features [C,H,W], got {tuple(pred_features.shape)}")
    if target_prototypes.dim() != 2:
        raise ValueError(
            f"Expected target_prototypes [N,C], got {tuple(target_prototypes.shape)}"
        )
    if segment_map.dim() != 2:
        raise ValueError(f"Expected segment_map [H,W], got {tuple(segment_map.shape)}")

    pred = pred_features.float()
    targets = target_prototypes.to(device=pred.device, dtype=torch.float32)
    if pred.shape[0] != targets.shape[1]:
        raise ValueError(
            "Feature dimension mismatch: "
            f"pred has {pred.shape[0]} channels, prototypes have {targets.shape[1]}"
        )

    if segment_map.shape != pred.shape[-2:]:
        seg = F.interpolate(
            segment_map.to(device=pred.device, dtype=torch.float32).view(1, 1, *segment_map.shape),
            size=pred.shape[-2:],
            mode="nearest",
        ).view(*pred.shape[-2:]).long()
    else:
        seg = segment_map.to(device=pred.device, dtype=torch.long)

    background_mask = (seg < 0) | (seg >= targets.shape[0])
    if bool(background_mask.any()):
        pred_hw_c_for_bg = pred.permute(1, 2, 0)
        background_loss = pred_hw_c_for_bg[background_mask].square().sum(dim=-1).mean()
    else:
        background_loss = pred.sum() * 0.0

    seg_flat = seg.reshape(-1)
    valid = (seg_flat >= 0) & (seg_flat < targets.shape[0])
    if not bool(valid.any()):
        return _zero_losses(
            pred,
            background_loss=background_loss,
            background_weight=background_weight,
        )

    valid_ids = seg_flat[valid]
    unique_ids, inverse = torch.unique(valid_ids, sorted=True, return_inverse=True)
    counts = torch.bincount(inverse, minlength=int(unique_ids.numel())).to(pred.device)
    keep = counts >= max(1, int(min_pixels))
    if not bool(keep.any()):
        return _zero_losses(
            pred,
            background_loss=background_loss,
            background_weight=background_weight,
        )

    unique_ids = unique_ids[keep]
    keep_lookup = torch.zeros(
        counts.shape[0], device=pred.device, dtype=torch.bool
    )
    keep_lookup[keep.nonzero(as_tuple=False).flatten()] = True
    valid_keep = keep_lookup[inverse]
    kept_inverse = torch.cumsum(keep.to(torch.long), dim=0)[inverse[valid_keep]] - 1
    counts = counts[keep].float()

    selected = _select_largest_regions(unique_ids, counts, max_regions=int(max_regions))
    if selected.numel() == 0:
        return _zero_losses(
            pred,
            background_loss=background_loss,
            background_weight=background_weight,
        )

    selected_lookup = torch.zeros(
        unique_ids.shape[0], device=pred.device, dtype=torch.bool
    )
    selected_lookup[selected] = True
    selected_pixel_mask = selected_lookup[kept_inverse]
    if not bool(selected_pixel_mask.any()):
        return _zero_losses(
            pred,
            background_loss=background_loss,
            background_weight=background_weight,
        )

    selected_inverse = kept_inverse[selected_pixel_mask]
    remap = torch.full(
        (unique_ids.shape[0],),
        -1,
        device=pred.device,
        dtype=torch.long,
    )
    remap[selected] = torch.arange(selected.numel(), device=pred.device)
    pooled_inverse = remap[selected_inverse]
    region_ids = unique_ids[selected]

    pred_hw_c = pred.permute(1, 2, 0).reshape(-1, pred.shape[0])
    selected_pred = pred_hw_c[valid][valid_keep][selected_pixel_mask]
    pooled = torch.zeros(
        selected.numel(), pred.shape[0], device=pred.device, dtype=torch.float32
    )
    pooled.index_add_(0, pooled_inverse, selected_pred)
    pooled_counts = torch.bincount(
        pooled_inverse, minlength=int(selected.numel())
    ).to(device=pred.device, dtype=torch.float32)
    pooled = pooled / pooled_counts.clamp_min(1.0).unsqueeze(1)

    pooled = F.normalize(pooled, dim=-1, eps=1e-8)
    target = F.normalize(targets[region_ids], dim=-1, eps=1e-8)
    cosine = (pooled * target).sum(dim=-1)
    prototype_loss = (1.0 - cosine).mean()

    if selected.numel() >= 2:
        temperature = max(float(contrastive_temperature), 1e-6)
        logits = pooled @ target.t() / temperature
        labels = torch.arange(selected.numel(), device=pred.device)
        contrastive_loss = F.cross_entropy(logits, labels)
    else:
        contrastive_loss = prototype_loss * 0.0

    total = (
        float(prototype_weight) * prototype_loss
        + float(contrastive_weight) * contrastive_loss
        + float(background_weight) * background_loss
    )
    return {
        "prototype_loss": prototype_loss,
        "contrastive_loss": contrastive_loss,
        "background_loss": background_loss,
        "total_loss": total,
        "valid_regions": torch.as_tensor(
            float(selected.numel()), device=pred.device, dtype=torch.float32
        ),
    }
