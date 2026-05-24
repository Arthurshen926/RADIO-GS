#!/usr/bin/env python3
"""Train a prompt-conditioned SAM-adaptor mask head from official SAM3 caches.

The head is feature-only at inference time: rendered CTF-GS/RADIO features,
text-prompt embeddings, and a coarse mask are used to predict a refined mask.
Official SAM3 RGB masks are used only as training-view pseudo labels.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

sys.path.insert(0, ".")

from radio_gs.data.lerf_dataset import LERFDataset
from radio_gs.models.foundation_cache import FoundationHeadCache, load_foundation_cache
from radio_gs.models.prompt_conditioned_mask_head import PromptConditionedMaskHead
from radio_gs.scripts.eval_lerf_adaptor_downstream import _load_teacher_feature, _render_feature
from radio_gs.scripts.eval_lerf_grounding import (
    DEFAULT_GT_FEATURE_ROOT,
    DEFAULT_LABEL_DIR,
    build_gt_masks,
    load_lerf_ovs_labels,
    load_render_pipeline,
    resolve_lerf_label_dir,
    resolve_lerf_scene_root,
)


@dataclass(frozen=True)
class PromptMaskTargets:
    categories: list[str]
    prompts: torch.Tensor
    targets: torch.Tensor
    source_indices: torch.Tensor


@dataclass
class PromptMaskTrainingBatch:
    frame_id: int
    categories: list[str]
    prompts: torch.Tensor
    targets: torch.Tensor
    coarse: torch.Tensor
    feature: torch.Tensor | None = None


def _normalise_query(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _resize_targets(masks: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    if masks.ndim != 3:
        raise ValueError(f"targets must be [Q,H,W], got {tuple(masks.shape)}")
    height, width = int(target_size[0]), int(target_size[1])
    if masks.shape[-2:] == (height, width):
        return masks.float()
    return F.interpolate(
        masks.float().unsqueeze(1),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)


def select_prompt_mask_targets(
    head_cache: FoundationHeadCache,
    *,
    categories: Iterable[str],
    text_embeddings: Mapping[str, torch.Tensor],
    target_size: tuple[int, int],
    score_threshold: float = -float("inf"),
) -> PromptMaskTargets:
    """Select the best official SAM3 pseudo mask for each requested category.

    ``mask_query_indices`` is required because SAM3 may emit a variable number of
    candidate masks per text query. Without this provenance, a mask-logit tensor
    cannot safely supervise a prompt-conditioned head.
    """

    if head_cache.mask_logits is None:
        raise ValueError("SAM3 cache is missing mask_logits")
    if head_cache.mask_query_indices is None:
        raise ValueError("SAM3 cache is missing mask_query_indices")
    if not head_cache.queries:
        raise ValueError("SAM3 cache is missing queries")

    query_lookup = {
        _normalise_query(query): idx for idx, query in enumerate(head_cache.queries)
    }
    scores = head_cache.scores
    if scores is None:
        scores = torch.ones(head_cache.mask_logits.shape[0], dtype=torch.float32)
    scores = scores.float().cpu()
    query_indices = head_cache.mask_query_indices.long().cpu()

    selected_categories: list[str] = []
    selected_prompts: list[torch.Tensor] = []
    selected_masks: list[torch.Tensor] = []
    selected_indices: list[int] = []
    for category in categories:
        key = _normalise_query(category)
        if key not in query_lookup or category not in text_embeddings:
            continue
        query_idx = int(query_lookup[key])
        candidate_indices = torch.nonzero(query_indices == query_idx, as_tuple=False).flatten()
        if candidate_indices.numel() == 0:
            continue
        candidate_scores = scores[candidate_indices]
        best_pos = int(candidate_scores.argmax().item())
        best_idx = int(candidate_indices[best_pos].item())
        if float(scores[best_idx].item()) < float(score_threshold):
            continue
        selected_categories.append(str(category))
        selected_prompts.append(text_embeddings[category].detach().float().cpu())
        selected_masks.append(head_cache.mask_logits[best_idx].detach().float().cpu())
        selected_indices.append(best_idx)

    if not selected_categories:
        prompt_dim = next(iter(text_embeddings.values())).numel() if text_embeddings else 0
        return PromptMaskTargets(
            categories=[],
            prompts=torch.empty(0, prompt_dim),
            targets=torch.empty(0, int(target_size[0]), int(target_size[1])),
            source_indices=torch.empty(0, dtype=torch.long),
        )

    prompts = torch.stack(selected_prompts, dim=0)
    targets = _resize_targets(torch.stack(selected_masks, dim=0), target_size)
    return PromptMaskTargets(
        categories=selected_categories,
        prompts=prompts,
        targets=targets,
        source_indices=torch.tensor(selected_indices, dtype=torch.long),
    )


def build_coarse_prompt_from_target(
    target: torch.Tensor,
    *,
    dilate: int = 0,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Build a spatial coarse-mask prompt from pseudo targets."""

    if target.ndim != 3:
        raise ValueError(f"target must be [Q,H,W], got {tuple(target.shape)}")
    coarse = (target.float() > float(threshold)).float()
    radius = int(dilate)
    if radius > 0 and coarse.numel() > 0:
        kernel = radius * 2 + 1
        coarse = F.max_pool2d(
            coarse.unsqueeze(1),
            kernel_size=kernel,
            stride=1,
            padding=radius,
        ).squeeze(1)
    return coarse


def _mask_file_category(category: str) -> str:
    return str(category).replace("/", "_")


def load_coarse_prompt_mask(
    coarse_mask_dir: str | Path,
    *,
    frame_id: int,
    category: str,
    target_size: tuple[int, int],
) -> torch.Tensor:
    """Load a saved direct-3D coarse mask as the training spatial prompt."""

    root = Path(coarse_mask_dir)
    path = root / f"frame_{int(frame_id):05d}_{_mask_file_category(category)}.png"
    if not path.exists():
        raise FileNotFoundError(path)
    image = Image.open(path).convert("L")
    mask = torch.from_numpy(np.array(image, dtype="float32") / 255.0)
    mask = (mask > 0.5).float()
    height, width = int(target_size[0]), int(target_size[1])
    if mask.shape != (height, width):
        mask = F.interpolate(
            mask.view(1, 1, *mask.shape),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).view(height, width)
        mask = (mask > 0.0).float()
    return mask.float()


def categories_for_training_frame(
    frame_annotations: Mapping[int, list[dict]],
    frame_id: int,
    scene_categories: Iterable[str],
) -> list[str]:
    """Return active labelled categories or all scene queries for unlabelled views."""

    objects = frame_annotations.get(int(frame_id))
    if objects:
        return sorted({str(obj["category"]) for obj in objects})
    return list(scene_categories)


def _soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    target = target.float()
    inter = (prob * target).flatten(1).sum(dim=1)
    denom = prob.flatten(1).sum(dim=1) + target.flatten(1).sum(dim=1)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def _boundary_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    grad_pred_h = torch.abs(prob[..., 1:, :] - prob[..., :-1, :]).mean()
    grad_pred_w = torch.abs(prob[..., :, 1:] - prob[..., :, :-1]).mean()
    grad_tgt_h = torch.abs(target[..., 1:, :] - target[..., :-1, :]).mean()
    grad_tgt_w = torch.abs(target[..., :, 1:] - target[..., :, :-1]).mean()
    return torch.abs(grad_pred_h - grad_tgt_h) + torch.abs(grad_pred_w - grad_tgt_w)


def _target_for_loss(targets: torch.Tensor, mode: str, *, threshold: float = 0.5) -> torch.Tensor:
    mode = str(mode).lower()
    if mode == "raw":
        return targets.float().clamp(0.0, 1.0)
    if mode == "sigmoid":
        return torch.sigmoid(targets.float())
    if mode == "binary":
        return (targets.float() > float(threshold)).float()
    raise ValueError("target_activation must be one of: binary, sigmoid, raw")


def _frame_id_from_cache_path(path: Path) -> int | None:
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else None


def _cache_path(cache_root: Path, frame_id: int) -> Path:
    candidates = [
        cache_root / f"frame_{frame_id:05d}.pt",
        cache_root / f"frame_{frame_id:06d}.pt",
        cache_root / f"rgb_{frame_id}.pt",
        cache_root / f"{frame_id}.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _available_cache_frames(cache_root: Path) -> list[int]:
    frames: list[int] = []
    for path in sorted(cache_root.glob("*.pt")):
        frame_id = _frame_id_from_cache_path(path)
        if frame_id is not None:
            frames.append(frame_id)
    return sorted(set(frames))


def _load_text_embedding_map(path: str | Path, device: torch.device | None = None) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, Mapping) and "queries" in payload and "embeddings" in payload:
        queries = [str(query) for query in payload["queries"]]
        embeddings = payload["embeddings"].float()
        return {
            query: F.normalize(embeddings[idx], dim=0).to(device or torch.device("cpu"))
            for idx, query in enumerate(queries)
        }
    if isinstance(payload, Mapping):
        return {
            str(key): F.normalize(value.float(), dim=0).to(device or torch.device("cpu"))
            for key, value in payload.items()
            if torch.is_tensor(value)
        }
    raise ValueError(f"unsupported text embedding cache format: {path}")


def resolve_feature_dir_for_scene(root_or_scene_dir: str | Path, scene: str) -> Path:
    root = Path(root_or_scene_dir)
    if (root / "backbone").exists() or list(root.glob("rgb_*.pt")):
        return root
    scene_dir = root / scene
    if (scene_dir / "backbone").exists() or list(scene_dir.glob("rgb_*.pt")):
        return scene_dir
    return scene_dir if scene_dir.exists() else root


def _build_lerf_dataset_for_scene(
    *,
    scene: str,
    config: object,
    feature_dir_root: str | Path,
    label_dir: str | Path,
) -> LERFDataset:
    feature_height = int(getattr(config, "feature_height", 30))
    feature_width = int(getattr(config, "feature_width", 40))
    scene_root = resolve_lerf_scene_root(scene, getattr(config, "scene_root", ""))
    feature_dir = resolve_feature_dir_for_scene(feature_dir_root, scene)
    if not feature_dir.exists():
        feature_dir = Path(DEFAULT_GT_FEATURE_ROOT) / scene
    return LERFDataset(
        scene_root=str(scene_root),
        feature_dir=str(feature_dir),
        annotation_dir=str(Path(label_dir) / scene),
        feature_height=feature_height,
        feature_width=feature_width,
    )


def _load_source_feature(
    *,
    source: str,
    scene: str,
    frame_id: int,
    device: torch.device,
    radio_feature_dir: Path,
    render_pipeline: tuple | None,
    lerf_dataset: LERFDataset | None,
) -> torch.Tensor:
    if source == "teacher":
        return _load_teacher_feature(radio_feature_dir, frame_id, device).float()
    if source == "rendered":
        if render_pipeline is None or lerf_dataset is None:
            raise ValueError("rendered source requires --config and --checkpoint")
        return _render_feature(scene, frame_id, render_pipeline, lerf_dataset, device).float()
    raise ValueError(f"unsupported feature source: {source}")


def _resize_feature(feature: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if feature.shape[-2:] == size:
        return feature
    return F.interpolate(feature.float(), size=size, mode="bilinear", align_corners=False)


def train_prompt_mask_head(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    cache_root = Path(args.sam3_cache_root)
    text_embeddings = _load_text_embedding_map(args.text_embedding_cache)
    prompt_dim = int(next(iter(text_embeddings.values())).numel())

    label_dir = resolve_lerf_label_dir(args.label_dir)
    frame_annotations, scene_categories, img_h, img_w = load_lerf_ovs_labels(label_dir, args.scene)
    label_frames = sorted(frame_annotations)
    cache_frames = _available_cache_frames(cache_root)
    train_frames = list(cache_frames)
    if args.frame_ids:
        requested = {int(part) for part in str(args.frame_ids).replace("\n", ",").split(",") if part.strip()}
        train_frames = [frame for frame in train_frames if frame in requested]
    if not train_frames:
        raise RuntimeError(f"No labelled SAM3 cache frames found under {cache_root}")
    if args.max_frames > 0:
        train_frames = train_frames[: int(args.max_frames)]

    render_pipeline = None
    lerf_dataset = None
    if args.source == "rendered":
        render_pipeline = load_render_pipeline(args.config, args.checkpoint, device)
        lerf_dataset = _build_lerf_dataset_for_scene(
            scene=args.scene,
            config=render_pipeline[5],
            feature_dir_root=args.radio_feature_dir,
            label_dir=args.label_dir,
        )

    radio_feature_dir = resolve_feature_dir_for_scene(args.radio_feature_dir, args.scene)
    target_size = tuple(int(v) for v in args.train_size)
    head = PromptConditionedMaskHead(
        feature_dim=int(args.feature_dim),
        prompt_dim=prompt_dim,
        hidden_dim=int(args.hidden_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    feature_cache_dtype = torch.float16 if args.feature_cache_dtype == "float16" else torch.float32
    batches: list[PromptMaskTrainingBatch] = []
    for frame_id in train_frames:
        cache = load_foundation_cache(_cache_path(cache_root, frame_id), require_official=True)
        head_cache = cache.heads["sam3"]
        categories = categories_for_training_frame(
            frame_annotations,
            frame_id,
            scene_categories,
        )
        selected = select_prompt_mask_targets(
            head_cache,
            categories=categories,
            text_embeddings=text_embeddings,
            target_size=target_size,
            score_threshold=float(args.score_threshold),
        )
        if not selected.categories:
            continue
        coarse_from_direct: torch.Tensor | None = None
        if args.coarse_mask_dir:
            keep_indices: list[int] = []
            coarse_masks: list[torch.Tensor] = []
            for idx, category in enumerate(selected.categories):
                try:
                    coarse_masks.append(
                        load_coarse_prompt_mask(
                            args.coarse_mask_dir,
                            frame_id=frame_id,
                            category=category,
                            target_size=target_size,
                        )
                    )
                    keep_indices.append(idx)
                except FileNotFoundError:
                    continue
            if not keep_indices:
                continue
            keep = torch.tensor(keep_indices, dtype=torch.long)
            selected = PromptMaskTargets(
                categories=[selected.categories[idx] for idx in keep_indices],
                prompts=selected.prompts.index_select(0, keep),
                targets=selected.targets.index_select(0, keep),
                source_indices=selected.source_indices.index_select(0, keep),
            )
            coarse_from_direct = torch.stack(coarse_masks, dim=0)

        targets = _target_for_loss(
            selected.targets,
            args.target_activation,
            threshold=float(args.target_threshold),
        ).detach().cpu()
        if coarse_from_direct is not None:
            coarse = build_coarse_prompt_from_target(
                coarse_from_direct,
                dilate=int(args.coarse_dilate),
                threshold=float(args.coarse_threshold),
            ).detach().cpu()
        else:
            coarse = build_coarse_prompt_from_target(
                targets,
                dilate=int(args.coarse_dilate),
                threshold=float(args.coarse_threshold),
            ).detach().cpu()

        cached_feature: torch.Tensor | None = None
        if args.cache_source_features:
            feature = _load_source_feature(
                source=args.source,
                scene=args.scene,
                frame_id=frame_id,
                device=device,
                radio_feature_dir=radio_feature_dir,
                render_pipeline=render_pipeline,
                lerf_dataset=lerf_dataset,
            )
            feature = _resize_feature(feature, target_size).detach().to("cpu", dtype=feature_cache_dtype)
            cached_feature = feature
            del feature
            if device.type == "cuda":
                torch.cuda.empty_cache()

        batches.append(
            PromptMaskTrainingBatch(
                frame_id=int(frame_id),
                categories=list(selected.categories),
                prompts=selected.prompts.detach().cpu(),
                targets=targets,
                coarse=coarse,
                feature=cached_feature,
            )
        )
    if not batches:
        raise RuntimeError(f"No trainable SAM3 prompt-mask samples found under {cache_root}")

    history: list[dict[str, float]] = []
    head.train()
    for epoch in range(int(args.epochs)):
        for batch in batches:
            if batch.feature is None:
                feature = _load_source_feature(
                    source=args.source,
                    scene=args.scene,
                    frame_id=batch.frame_id,
                    device=device,
                    radio_feature_dir=radio_feature_dir,
                    render_pipeline=render_pipeline,
                    lerf_dataset=lerf_dataset,
                )
                feature = _resize_feature(feature, target_size).to(device)
            else:
                feature = batch.feature.to(device=device, dtype=torch.float32, non_blocking=True)
            prompts = batch.prompts.to(device=device, non_blocking=True)
            targets = batch.targets.to(device=device, non_blocking=True)
            coarse = batch.coarse.to(device=device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = head(feature, prompts.unsqueeze(0), coarse.unsqueeze(0)).squeeze(0)
            bce = F.binary_cross_entropy_with_logits(logits, targets)
            dice = _soft_dice_loss(logits, targets)
            boundary = _boundary_loss(logits, targets)
            loss = bce + float(args.dice_weight) * dice + float(args.boundary_weight) * boundary
            loss.backward()
            optimizer.step()
            history.append(
                {
                    "epoch": float(epoch),
                    "frame_id": float(batch.frame_id),
                    "loss": float(loss.detach().cpu()),
                    "bce": float(bce.detach().cpu()),
                    "dice": float(dice.detach().cpu()),
                    "boundary": float(boundary.detach().cpu()),
                    "queries": float(len(batch.categories)),
                }
            )
            del feature, prompts, targets, coarse, logits, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "prompt_mask_head_state_dict": head.state_dict(),
        "scene": args.scene,
        "source": args.source,
        "feature_dim": int(args.feature_dim),
        "prompt_dim": prompt_dim,
        "hidden_dim": int(args.hidden_dim),
        "target_size": target_size,
        "train_frames": [batch.frame_id for batch in batches],
        "loss_history": history,
        "config": vars(args),
    }
    checkpoint_path = out_dir / "prompt_conditioned_sam3_mask_head.pth"
    torch.save(checkpoint, checkpoint_path)

    summary = {
        "scene": args.scene,
        "source": args.source,
        "checkpoint": str(checkpoint_path),
        "train_frames": [batch.frame_id for batch in batches],
        "n_steps": len(history),
        "final_loss": history[-1]["loss"] if history else None,
        "cached_source_features": bool(args.cache_source_features),
        "n_pseudo_masks": int(sum(len(batch.categories) for batch in batches)),
        "img_h": img_h,
        "img_w": img_w,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--sam3_cache_root", required=True)
    parser.add_argument("--text_embedding_cache", default="checkpoints/siglip2_lerf_text_embeddings.pt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source", choices=("rendered", "teacher"), default="rendered")
    parser.add_argument("--config", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--radio_feature_dir", default=DEFAULT_GT_FEATURE_ROOT)
    parser.add_argument("--label_dir", default=DEFAULT_LABEL_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature_dim", type=int, default=1280)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--train_size", type=int, nargs=2, default=(240, 320))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dice_weight", type=float, default=0.5)
    parser.add_argument("--boundary_weight", type=float, default=0.25)
    parser.add_argument("--coarse_dilate", type=int, default=3)
    parser.add_argument("--coarse_threshold", type=float, default=0.5)
    parser.add_argument("--score_threshold", type=float, default=-float("inf"))
    parser.add_argument("--target_activation", choices=("binary", "sigmoid", "raw"), default="binary")
    parser.add_argument("--target_threshold", type=float, default=0.5, help="Probability/logit threshold for binary pseudo masks")
    parser.add_argument("--frame_ids", default="")
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--coarse_mask_dir", default="", help="Optional saved direct-3D coarse mask directory used as spatial prompts")
    parser.add_argument("--cache_source_features", action="store_true", help="Render/load source features once before optimisation")
    parser.add_argument("--feature_cache_dtype", choices=("float16", "float32"), default="float16")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    summary = train_prompt_mask_head(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
