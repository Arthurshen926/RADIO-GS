#!/usr/bin/env python3
"""Train a point-summary adapter on cached ScanNet RADIO teacher features.

This is a point-native fine-tune path: the RADIO-GS field is loaded from an
existing checkpoint and kept frozen, compact point features are queried directly
at ScanNet label vertices, and only a small compact→SigLIP-summary adapter is
optimized.  The saved checkpoint preserves the original field/codec state and
adds ``point_summary_adapter_state_dict`` for direct point-cloud evaluation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from radio_gs.config import config_to_dict, load_config
from radio_gs.models.point_summary_adapter import CompactToSummaryAdapter
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.scannet_constants import NYU40_ID_TO_NAME, OPENGAUSSIAN_NYU40_CLASS_SPLITS
from radio_gs.scripts.eval_lerf_grounding import (
    load_or_generate_prompt_ensemble_embeddings,
    parse_prompt_templates,
)
from radio_gs.scripts.eval_scannet_pointcloud_radio_gs import _build_hybrid_model
from radio_gs.utils.checkpoint_io import load_trusted_checkpoint


def _weighted_mean(values: torch.Tensor, sample_weights: torch.Tensor | None) -> torch.Tensor:
    if sample_weights is None:
        return values.mean()
    weights = sample_weights.to(device=values.device, dtype=values.dtype).clamp_min(0.0)
    denom = weights.sum().clamp_min(1e-6)
    return (values * weights).sum() / denom


def compute_text_rank_distillation_loss(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
    margin: float = 0.1,
    topk: int = 0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Penalize student text-score orderings that violate teacher rankings."""
    student_scores = student_scores.float()
    teacher_scores = teacher_scores.to(device=student_scores.device, dtype=student_scores.dtype)
    zero = student_scores.sum() * 0.0
    if student_scores.ndim != 2 or teacher_scores.shape != student_scores.shape:
        raise ValueError("student_scores and teacher_scores must have matching [N, C] shapes")
    if student_scores.shape[1] < 2:
        return zero, {"rank_pairs": zero.detach()}

    k = int(topk) if int(topk) > 0 else int(student_scores.shape[1])
    k = min(k, int(student_scores.shape[1]))
    top_indices = teacher_scores.topk(k, dim=-1).indices
    ranked_teacher = teacher_scores.gather(1, top_indices)
    ranked_student = student_scores.gather(1, top_indices)

    teacher_diff = ranked_teacher.unsqueeze(2) - ranked_teacher.unsqueeze(1)
    student_diff = ranked_student.unsqueeze(2) - ranked_student.unsqueeze(1)
    pair_mask = teacher_diff > 0
    rank_pairs = pair_mask.float().sum()
    if not pair_mask.any():
        return zero, {"rank_pairs": rank_pairs.detach()}

    per_pair = F.relu(float(margin) - student_diff)[pair_mask]
    if sample_weights is None:
        loss = per_pair.mean()
    else:
        point_weights = sample_weights.to(device=student_scores.device, dtype=student_scores.dtype).clamp_min(0.0)
        pair_weights = point_weights[:, None, None].expand_as(student_diff)[pair_mask]
        loss = (per_pair * pair_weights).sum() / pair_weights.sum().clamp_min(1e-6)
    return loss, {"rank_pairs": rank_pairs.detach()}


def compute_adapter_training_loss(
    compact: torch.Tensor,
    teacher_summary: torch.Tensor,
    adapter: torch.nn.Module,
    *,
    text_embeddings: torch.Tensor | None = None,
    summary_weight: float = 1.0,
    text_distill_weight: float = 0.0,
    text_distill_temperature: float = 1.0,
    text_confidence_threshold: float = 0.0,
    text_pseudo_ce_weight: float = 0.0,
    text_pseudo_ce_confidence_threshold: float = 0.0,
    text_pseudo_ce_logit_scale: float = 1.0,
    text_rank_distill_weight: float = 0.0,
    text_rank_distill_margin: float = 0.1,
    text_rank_distill_topk: int = 0,
    decoder_anchor_summary: torch.Tensor | None = None,
    decoder_anchor_weight: float = 0.0,
    sample_weights: torch.Tensor | None = None,
    detach_compact: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute adapter-only summary and text-distillation losses."""
    compact_in = compact.detach() if detach_compact else compact
    pred_summary = F.normalize(adapter(compact_in.float()).float(), dim=-1)
    teacher_summary = F.normalize(teacher_summary.float(), dim=-1)
    if sample_weights is not None:
        sample_weights = sample_weights.to(device=pred_summary.device, dtype=pred_summary.dtype)

    zero = pred_summary.sum() * 0.0
    summary_loss = zero
    if summary_weight > 0:
        summary_loss = _weighted_mean(
            1.0 - (pred_summary * teacher_summary).sum(dim=-1),
            sample_weights,
        )

    text_distill_loss = zero
    text_valid_ratio = zero.detach()
    text_teacher_conf = zero.detach()
    text_agreement = zero.detach()
    text_pseudo_ce_loss = zero
    text_pseudo_ce_valid_ratio = zero.detach()
    text_pseudo_ce_teacher_conf = zero.detach()
    text_pseudo_ce_agreement = zero.detach()
    text_rank_distill_loss = zero
    text_rank_pairs = zero.detach()
    decoder_anchor_loss = zero
    if text_distill_weight > 0 and text_embeddings is not None:
        text_embeddings = F.normalize(text_embeddings.to(pred_summary.device).float(), dim=-1)
        temperature = max(float(text_distill_temperature), 1e-6)
        student_logits = pred_summary @ text_embeddings.T
        with torch.no_grad():
            teacher_logits = teacher_summary @ text_embeddings.T
            teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
            teacher_conf = teacher_probs.max(dim=-1).values
            valid = teacher_conf >= float(text_confidence_threshold)
        text_valid_ratio = valid.float().mean()
        if valid.any():
            per_point_kl = F.kl_div(
                F.log_softmax(student_logits[valid] / temperature, dim=-1),
                teacher_probs[valid],
                reduction="none",
            ).sum(dim=-1)
            text_weights = sample_weights[valid] if sample_weights is not None else None
            text_distill_loss = _weighted_mean(per_point_kl, text_weights)
            if temperature != 1.0:
                text_distill_loss = text_distill_loss * (temperature**2)
            with torch.no_grad():
                student_pred = student_logits[valid].argmax(dim=-1)
                teacher_pred = teacher_probs[valid].argmax(dim=-1)
                text_agreement = (student_pred == teacher_pred).float().mean()
                text_teacher_conf = teacher_conf[valid].mean()

    if text_rank_distill_weight > 0 and text_embeddings is not None:
        text_embeddings = F.normalize(text_embeddings.to(pred_summary.device).float(), dim=-1)
        student_logits = pred_summary @ text_embeddings.T
        with torch.no_grad():
            teacher_logits = teacher_summary @ text_embeddings.T
        text_rank_distill_loss, rank_stats = compute_text_rank_distillation_loss(
            student_logits,
            teacher_logits,
            sample_weights=sample_weights,
            margin=text_rank_distill_margin,
            topk=text_rank_distill_topk,
        )
        text_rank_pairs = rank_stats["rank_pairs"]

    if text_pseudo_ce_weight > 0 and text_embeddings is not None:
        text_embeddings = F.normalize(text_embeddings.to(pred_summary.device).float(), dim=-1)
        logit_scale = float(text_pseudo_ce_logit_scale)
        student_logits = (pred_summary @ text_embeddings.T) * logit_scale
        with torch.no_grad():
            teacher_logits = (teacher_summary @ text_embeddings.T) * logit_scale
            teacher_probs = F.softmax(teacher_logits, dim=-1)
            teacher_conf, teacher_targets = teacher_probs.max(dim=-1)
            valid = teacher_conf >= float(text_pseudo_ce_confidence_threshold)
        text_pseudo_ce_valid_ratio = valid.float().mean()
        if valid.any():
            per_point_ce = F.cross_entropy(
                student_logits[valid],
                teacher_targets[valid],
                reduction="none",
            )
            ce_weights = sample_weights[valid] if sample_weights is not None else None
            text_pseudo_ce_loss = _weighted_mean(per_point_ce, ce_weights)
            with torch.no_grad():
                student_pred = student_logits[valid].argmax(dim=-1)
                text_pseudo_ce_agreement = (student_pred == teacher_targets[valid]).float().mean()
                text_pseudo_ce_teacher_conf = teacher_conf[valid].mean()

    total = (
        float(summary_weight) * summary_loss
        + float(text_distill_weight) * text_distill_loss
        + float(text_pseudo_ce_weight) * text_pseudo_ce_loss
        + float(text_rank_distill_weight) * text_rank_distill_loss
    )
    if decoder_anchor_weight > 0 and decoder_anchor_summary is not None:
        decoder_anchor_summary = F.normalize(decoder_anchor_summary.float(), dim=-1)
        decoder_anchor_loss = _weighted_mean(
            1.0 - (pred_summary * decoder_anchor_summary).sum(dim=-1),
            sample_weights,
        )
        total = total + float(decoder_anchor_weight) * decoder_anchor_loss
    sample_weight_mean = (
        sample_weights.detach().float().mean()
        if sample_weights is not None
        else torch.ones((), device=pred_summary.device)
    )
    return total, {
        "loss": total.detach(),
        "summary_loss": summary_loss.detach(),
        "text_distill_loss": text_distill_loss.detach(),
        "text_distill_valid_ratio": text_valid_ratio.detach(),
        "text_distill_teacher_conf": text_teacher_conf.detach(),
        "text_distill_agreement": text_agreement.detach(),
        "text_pseudo_ce_loss": text_pseudo_ce_loss.detach(),
        "text_pseudo_ce_valid_ratio": text_pseudo_ce_valid_ratio.detach(),
        "text_pseudo_ce_teacher_conf": text_pseudo_ce_teacher_conf.detach(),
        "text_pseudo_ce_agreement": text_pseudo_ce_agreement.detach(),
        "text_rank_distill_loss": text_rank_distill_loss.detach(),
        "text_rank_pairs": text_rank_pairs.detach(),
        "decoder_anchor_loss": decoder_anchor_loss.detach(),
        "sample_weight_mean": sample_weight_mean.detach(),
    }


def merge_adapter_checkpoint(
    base_checkpoint: dict[str, Any],
    adapter_state_dict: dict[str, torch.Tensor],
    *,
    metadata: dict[str, Any],
    epoch: int,
    best_metric: float,
) -> dict[str, Any]:
    """Return a checkpoint with the base field preserved and adapter attached."""
    merged = dict(base_checkpoint)
    merged["point_summary_adapter_state_dict"] = adapter_state_dict
    merged["point_summary_adapter_metadata"] = dict(metadata)
    merged["point_summary_adapter_epoch"] = int(epoch)
    merged["point_summary_adapter_best_metric"] = float(best_metric)
    return merged


def _split_cache_path(raw_cache: str, split: str) -> str | None:
    if not raw_cache:
        return None
    base = Path(raw_cache)
    return str(base.with_name(f"{base.stem}_split{split}.pt"))


def _load_text_embeddings(
    split: str,
    device: torch.device,
    cache_path: str,
    prompt_templates: str,
) -> torch.Tensor:
    class_names = [
        NYU40_ID_TO_NAME[class_id]
        for class_id in OPENGAUSSIAN_NYU40_CLASS_SPLITS[split]
    ]
    return load_or_generate_prompt_ensemble_embeddings(
        class_names,
        device,
        cache_path=_split_cache_path(cache_path, split),
        prompt_templates=parse_prompt_templates(prompt_templates),
    )


def _load_teacher_cache_class_names(path: str) -> list[str]:
    payload = torch.load(path, map_location="cpu")
    metadata = payload.get("metadata") or {}
    categories = metadata.get("categories") or metadata.get("class_names")
    if not categories:
        raise KeyError(
            "teacher cache metadata must contain 'categories' or 'class_names' "
            "when --text_class_source=teacher_cache"
        )
    return [str(item) for item in categories]


def _load_teacher_cache_text_embeddings(
    teacher_cache: str,
    device: torch.device,
    cache_path: str,
    prompt_templates: str,
) -> torch.Tensor:
    class_names = _load_teacher_cache_class_names(teacher_cache)
    return load_or_generate_prompt_ensemble_embeddings(
        class_names,
        device,
        cache_path=cache_path or None,
        prompt_templates=parse_prompt_templates(prompt_templates),
    )


def _load_summary_head(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    head_path = Path(args.summary_head_weights)
    if head_path.exists():
        head = SigLIP2SummaryHead.from_extracted_weights(str(head_path))
    else:
        head = SigLIP2SummaryHead.from_radio_checkpoint(args.radio_checkpoint)
    return head.to(device).eval()


def _load_teacher_cache(path: str, device: torch.device, *, valid_only: bool) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    required = {"xyz", "valid"}
    missing = sorted(required.difference(payload.keys()))
    if missing:
        raise KeyError(f"teacher cache missing keys: {missing}")
    if "features" not in payload and "summary_features" not in payload:
        raise KeyError("teacher cache must contain either 'features' or 'summary_features'")
    valid = payload["valid"].bool()
    if valid_only:
        indices = torch.nonzero(valid, as_tuple=False).reshape(-1)
    else:
        indices = torch.arange(int(valid.numel()))
    result = {
        "indices": indices.to(device=device, dtype=torch.long),
        "xyz": payload["xyz"][indices].to(device=device, dtype=torch.float32),
        "valid": valid[indices].to(device=device),
    }
    if "features" in payload:
        result["features"] = payload["features"][indices].to(device=device, dtype=torch.float32)
    if "summary_features" in payload:
        result["summary_features"] = payload["summary_features"][indices].to(
            device=device,
            dtype=torch.float32,
        )
    if "view_counts" in payload:
        result["view_counts"] = payload["view_counts"][indices].to(
            device=device,
            dtype=torch.float32,
        )
    if "labels" in payload and payload["labels"] is not None:
        result["labels"] = payload["labels"][indices].to(device=device, dtype=torch.long)
    return result


def _build_teacher_sample_weights(
    teacher: dict[str, torch.Tensor],
    *,
    mode: str,
    min_weight: float,
    percentile_low: float = 0.0,
    percentile_high: float = 100.0,
) -> torch.Tensor | None:
    """Build normalized per-point weights from GT-free registration support."""
    if mode == "none":
        return None
    if "view_counts" not in teacher:
        raise KeyError("teacher cache does not contain view_counts for weighted VPR-to-field training")
    counts = teacher["view_counts"].float().clamp_min(0.0)
    if mode == "linear":
        weights = counts
    elif mode == "sqrt":
        weights = torch.sqrt(counts)
    elif mode == "log":
        weights = torch.log1p(counts)
    elif mode == "clipped_log":
        valid = teacher.get("valid", counts > 0).to(device=counts.device).bool()
        weights = torch.zeros_like(counts)
        positive = valid & (counts > 0)
        if not positive.any():
            return weights
        low = max(0.0, min(100.0, float(percentile_low)))
        high = max(0.0, min(100.0, float(percentile_high)))
        if low > high:
            raise ValueError("percentile_low must be <= percentile_high")
        logged = torch.log1p(counts)
        support = logged[positive]
        q_low = torch.quantile(support, low / 100.0)
        q_high = torch.quantile(support, high / 100.0)
        denom = (q_high - q_low).clamp_min(1e-6)
        scaled = ((logged.clamp(min=q_low.item(), max=q_high.item()) - q_low) / denom).clamp(0.0, 1.0)
        if min_weight > 0:
            scaled = scaled.clamp_min(float(min_weight))
        weights = torch.where(positive, scaled, weights)
        return weights
    else:
        raise ValueError(f"Unsupported teacher sample weight mode: {mode}")
    positive = weights > 0
    if min_weight > 0:
        weights = torch.where(positive, weights.clamp_min(float(min_weight)), weights)
    mean = weights[positive].mean().clamp_min(1e-6) if positive.any() else weights.mean().clamp_min(1e-6)
    return weights / mean


@torch.no_grad()
def _project_teacher_summary(
    features: torch.Tensor,
    summary_head: torch.nn.Module,
    *,
    chunk_size: int,
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for start in tqdm(range(0, features.shape[0], chunk_size), desc="teacher summary"):
        end = min(start + chunk_size, features.shape[0])
        summary = summary_head(features[start:end].unsqueeze(0))
        parts.append(F.normalize(summary.squeeze(0).float(), dim=-1).cpu())
    return torch.cat(parts, dim=0)


def _prepare_teacher_summary(
    teacher: dict[str, torch.Tensor],
    summary_head: torch.nn.Module,
    *,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return normalized teacher summaries from RADIO or summary-space caches."""
    if "summary_features" in teacher:
        summary = F.normalize(teacher["summary_features"].float(), dim=-1).to(device)
        del teacher["summary_features"]
        return summary
    if "features" not in teacher:
        raise KeyError("teacher cache has neither 'features' nor 'summary_features'")
    teacher_summary = _project_teacher_summary(
        teacher["features"],
        summary_head,
        chunk_size=chunk_size,
    ).to(device)
    del teacher["features"]
    return teacher_summary


@torch.no_grad()
def _project_decoder_anchor_summary(
    compact: torch.Tensor,
    codec: torch.nn.Module,
    summary_head: torch.nn.Module,
) -> torch.Tensor:
    if hasattr(codec, "decode_points"):
        decoded_points = codec.decode_points(compact.float())
    else:
        compact_map = compact.T.reshape(1, compact.shape[1], compact.shape[0], 1)
        decoded = codec.decode(compact_map.float())
        decoded_points = decoded.squeeze(0).squeeze(-1).T.contiguous()
    summary = summary_head(decoded_points.unsqueeze(0))
    return F.normalize(summary.squeeze(0).float(), dim=-1)


def _query_compact(
    model: torch.nn.Module,
    indices: torch.Tensor,
    points: torch.Tensor,
    *,
    query_mode: str,
    gaussian_index_position_mode: str,
    compact_feature_key: str,
    k: int,
    candidate_k: int,
    detach: bool = True,
) -> torch.Tensor:
    if query_mode == "gaussian_index":
        points_xyz = points if gaussian_index_position_mode == "label_point" else None
        result = model.query_gaussian_points(indices, points_xyz=points_xyz, return_aux=True)
    else:
        query_k = 1 if query_mode == "nearest" else k
        query_kwargs: dict[str, Any] = {"k": query_k, "return_aux": True}
        if query_mode != "nearest" and candidate_k > 0:
            query_kwargs["candidate_k"] = candidate_k
        result = model.query_compact_points(points, **query_kwargs)
    if compact_feature_key not in result:
        raise KeyError(
            f"Requested compact feature branch {compact_feature_key!r}, "
            f"available={sorted(result.keys())}"
        )
    compact = result[compact_feature_key]
    return compact.detach() if detach else compact


def train_adapter(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    config = load_config(args.config)
    model, codec = _build_hybrid_model(config, args.checkpoint, device)
    for param in model.parameters():
        param.requires_grad_(bool(args.train_field))
    for param in codec.parameters():
        param.requires_grad_(False)

    summary_head = _load_summary_head(args, device)
    teacher = _load_teacher_cache(args.teacher_cache, device, valid_only=not args.include_invalid)
    if args.max_points and teacher["indices"].numel() > args.max_points:
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)
        order = torch.randperm(teacher["indices"].numel(), generator=generator, device=device)[: args.max_points]
        teacher = {key: value[order] for key, value in teacher.items()}

    teacher_summary = _prepare_teacher_summary(
        teacher,
        summary_head,
        chunk_size=args.teacher_chunk_size,
        device=device,
    )
    sample_weights = _build_teacher_sample_weights(
        teacher,
        mode=args.teacher_sample_weight_mode,
        min_weight=args.teacher_sample_weight_min,
        percentile_low=args.teacher_sample_weight_percentile_low,
        percentile_high=args.teacher_sample_weight_percentile_high,
    )

    text_embeddings = None
    if args.text_distill_weight > 0 or args.text_pseudo_ce_weight > 0 or args.text_rank_distill_weight > 0:
        if args.text_class_source == "teacher_cache":
            text_embeddings = _load_teacher_cache_text_embeddings(
                args.teacher_cache,
                device,
                args.text_embedding_cache,
                args.prompt_templates,
            )
        else:
            text_embeddings = _load_text_embeddings(
                args.class_split,
                device,
                args.text_embedding_cache,
                args.prompt_templates,
            )

    adapter = CompactToSummaryAdapter(
        input_dim=getattr(config, "bottleneck_dim", getattr(config, "hybrid_output_dim", 128)),
        output_dim=teacher_summary.shape[1],
        hidden_dim=getattr(config, "point_summary_adapter_hidden_dim", 512),
        num_layers=getattr(config, "point_summary_adapter_num_layers", 2),
        dropout=getattr(config, "point_summary_adapter_dropout", 0.0),
    ).to(device)
    base_ckpt = load_trusted_checkpoint(args.checkpoint, map_location="cpu")
    if args.init_from_checkpoint and "point_summary_adapter_state_dict" in base_ckpt:
        adapter.load_state_dict(base_ckpt["point_summary_adapter_state_dict"], strict=False)

    optimizer_groups: list[dict[str, Any]] = [
        {"params": adapter.parameters(), "lr": args.lr, "weight_decay": args.weight_decay}
    ]
    if args.train_field:
        field_params = [param for param in model.parameters() if param.requires_grad]
        if field_params:
            optimizer_groups.append(
                {
                    "params": field_params,
                    "lr": args.field_lr,
                    "weight_decay": args.field_weight_decay,
                }
            )
    optimizer = torch.optim.AdamW(optimizer_groups)
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    num_points = int(teacher["indices"].numel())
    best_loss = float("inf")
    best_epoch = 0
    best_adapter_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        order = torch.randperm(num_points, generator=generator, device=device)
        accum = {
            "loss": 0.0,
            "summary_loss": 0.0,
            "text_distill_loss": 0.0,
            "text_distill_valid_ratio": 0.0,
            "text_distill_teacher_conf": 0.0,
            "text_distill_agreement": 0.0,
            "text_pseudo_ce_loss": 0.0,
            "text_pseudo_ce_valid_ratio": 0.0,
            "text_pseudo_ce_teacher_conf": 0.0,
            "text_pseudo_ce_agreement": 0.0,
            "text_rank_distill_loss": 0.0,
            "text_rank_pairs": 0.0,
            "decoder_anchor_loss": 0.0,
            "sample_weight_mean": 0.0,
        }
        steps = 0
        for start in tqdm(range(0, num_points, args.batch_size), desc=f"adapter E{epoch:03d}"):
            batch_ids = order[start : start + args.batch_size]
            compact = _query_compact(
                model,
                teacher["indices"][batch_ids],
                teacher["xyz"][batch_ids],
                query_mode=args.query_mode,
                gaussian_index_position_mode=args.gaussian_index_position_mode,
                compact_feature_key=args.compact_feature_key,
                k=args.k,
                candidate_k=args.candidate_k,
                detach=not args.train_field,
            )
            decoder_anchor_summary = None
            if args.decoder_anchor_weight > 0:
                decoder_anchor_summary = _project_decoder_anchor_summary(
                    compact,
                    codec,
                    summary_head,
                )
            loss, stats = compute_adapter_training_loss(
                compact,
                teacher_summary[batch_ids],
                adapter,
                text_embeddings=text_embeddings,
                summary_weight=args.summary_weight,
                text_distill_weight=args.text_distill_weight,
                text_distill_temperature=args.text_distill_temperature,
                text_confidence_threshold=args.text_confidence_threshold,
                text_pseudo_ce_weight=args.text_pseudo_ce_weight,
                text_pseudo_ce_confidence_threshold=args.text_pseudo_ce_confidence_threshold,
                text_pseudo_ce_logit_scale=args.text_pseudo_ce_logit_scale,
                text_rank_distill_weight=args.text_rank_distill_weight,
                text_rank_distill_margin=args.text_rank_distill_margin,
                text_rank_distill_topk=args.text_rank_distill_topk,
                decoder_anchor_summary=decoder_anchor_summary,
                decoder_anchor_weight=args.decoder_anchor_weight,
                sample_weights=sample_weights[batch_ids] if sample_weights is not None else None,
                detach_compact=not args.train_field,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.grad_clip)
            optimizer.step()
            for key in accum:
                accum[key] += float(stats[key].detach().cpu())
            steps += 1

        epoch_stats = {key: value / max(steps, 1) for key, value in accum.items()}
        epoch_stats["epoch"] = float(epoch)
        history.append(epoch_stats)
        print(
            f"[adapter E{epoch:03d}] loss={epoch_stats['loss']:.4f} "
            f"summary={epoch_stats['summary_loss']:.4f} "
            f"text_kl={epoch_stats['text_distill_loss']:.4f} "
            f"text_ce={epoch_stats['text_pseudo_ce_loss']:.4f} "
            f"text_rank={epoch_stats['text_rank_distill_loss']:.4f} "
            f"anchor={epoch_stats['decoder_anchor_loss']:.4f} "
            f"w={epoch_stats['sample_weight_mean']:.4f} "
            f"agree={epoch_stats['text_distill_agreement']:.4f}/"
            f"{epoch_stats['text_pseudo_ce_agreement']:.4f}"
        )

        if epoch_stats["loss"] < best_loss:
            best_loss = epoch_stats["loss"]
            best_epoch = epoch
            best_adapter_state = {
                key: value.detach().cpu().clone()
                for key, value in adapter.state_dict().items()
            }

    metadata = {
        "script": "train_scannet_point_summary_adapter.py",
        "config": args.config,
        "base_checkpoint": args.checkpoint,
        "teacher_cache": args.teacher_cache,
        "num_points": num_points,
        "query_mode": args.query_mode,
        "gaussian_index_position_mode": args.gaussian_index_position_mode,
        "compact_feature_key": args.compact_feature_key,
        "summary_weight": args.summary_weight,
        "text_distill_weight": args.text_distill_weight,
        "text_distill_temperature": args.text_distill_temperature,
        "text_pseudo_ce_weight": args.text_pseudo_ce_weight,
        "text_pseudo_ce_confidence_threshold": args.text_pseudo_ce_confidence_threshold,
        "text_pseudo_ce_logit_scale": args.text_pseudo_ce_logit_scale,
        "text_rank_distill_weight": args.text_rank_distill_weight,
        "text_rank_distill_margin": args.text_rank_distill_margin,
        "text_rank_distill_topk": args.text_rank_distill_topk,
        "decoder_anchor_weight": args.decoder_anchor_weight,
        "teacher_sample_weight_mode": args.teacher_sample_weight_mode,
        "teacher_sample_weight_min": args.teacher_sample_weight_min,
        "teacher_sample_weight_percentile_low": args.teacher_sample_weight_percentile_low,
        "teacher_sample_weight_percentile_high": args.teacher_sample_weight_percentile_high,
        "text_class_source": args.text_class_source,
        "train_field": bool(args.train_field),
        "field_lr": args.field_lr,
        "field_weight_decay": args.field_weight_decay,
        "class_split": args.class_split,
        "prompt_templates": parse_prompt_templates(args.prompt_templates),
        "config_snapshot": config_to_dict(config),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    latest_adapter_state = {
        key: value.detach().cpu()
        for key, value in adapter.state_dict().items()
    }
    if args.train_field:
        base_ckpt = dict(base_ckpt)
        base_ckpt["model_state_dict"] = {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        }
    latest = merge_adapter_checkpoint(
        base_ckpt,
        latest_adapter_state,
        metadata=metadata,
        epoch=args.epochs,
        best_metric=history[-1]["loss"] if history else float("inf"),
    )
    torch.save(latest, ckpt_dir / "latest.pth")
    if best_adapter_state is None:
        best_adapter_state = latest_adapter_state
        best_epoch = args.epochs
        best_loss = history[-1]["loss"] if history else float("inf")
    best = merge_adapter_checkpoint(
        base_ckpt,
        best_adapter_state,
        metadata=metadata,
        epoch=best_epoch,
        best_metric=best_loss,
    )
    torch.save(best, ckpt_dir / "best.pth")

    summary = {
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "num_points": num_points,
        "history": history,
        "best_checkpoint": str(ckpt_dir / "best.pth"),
        "latest_checkpoint": str(ckpt_dir / "latest.pth"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "adapter_training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--teacher_cache", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32768)
    parser.add_argument("--teacher_chunk_size", type=int, default=8192)
    parser.add_argument("--max_points", type=int, default=0)
    parser.add_argument("--include_invalid", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_field", action="store_true", help="Also update the RADIO-GS field with the VPR teacher loss")
    parser.add_argument("--field_lr", type=float, default=1e-4)
    parser.add_argument("--field_weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--summary_weight", type=float, default=1.0)
    parser.add_argument("--text_distill_weight", type=float, default=1.0)
    parser.add_argument("--text_distill_temperature", type=float, default=0.7)
    parser.add_argument("--text_confidence_threshold", type=float, default=0.0)
    parser.add_argument("--text_pseudo_ce_weight", type=float, default=0.0)
    parser.add_argument("--text_pseudo_ce_confidence_threshold", type=float, default=0.0)
    parser.add_argument("--text_pseudo_ce_logit_scale", type=float, default=1.0)
    parser.add_argument("--text_rank_distill_weight", type=float, default=0.0)
    parser.add_argument("--text_rank_distill_margin", type=float, default=0.1)
    parser.add_argument("--text_rank_distill_topk", type=int, default=0)
    parser.add_argument("--decoder_anchor_weight", type=float, default=0.0)
    parser.add_argument(
        "--teacher_sample_weight_mode",
        choices=("none", "log", "sqrt", "linear", "clipped_log"),
        default="none",
        help="Use VPR registration view_counts as GT-free sample weights for VPR-to-field training",
    )
    parser.add_argument("--teacher_sample_weight_min", type=float, default=0.0)
    parser.add_argument("--teacher_sample_weight_percentile_low", type=float, default=0.0)
    parser.add_argument("--teacher_sample_weight_percentile_high", type=float, default=100.0)
    parser.add_argument("--text_embedding_cache", default="checkpoints/siglip2_scannet_og_text_embeddings_ens5.pt")
    parser.add_argument("--text_class_source", choices=("scannet", "teacher_cache"), default="scannet")
    parser.add_argument(
        "--prompt_templates",
        default="{query}|a photo of a {query}|a 3d scan of a {query}|a point cloud of a {query}|an indoor scene containing a {query}",
    )
    parser.add_argument("--class_split", choices=sorted(OPENGAUSSIAN_NYU40_CLASS_SPLITS), default="19")
    parser.add_argument("--query_mode", choices=("gaussian_index", "knn", "nearest"), default="gaussian_index")
    parser.add_argument(
        "--gaussian_index_position_mode",
        choices=("gaussian_center", "label_point"),
        default="label_point",
    )
    parser.add_argument("--compact_feature_key", choices=("features", "fused", "semantic", "geometry"), default="features")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--candidate_k", type=int, default=0)
    parser.add_argument("--init_from_checkpoint", action="store_true")
    parser.add_argument("--summary_head_weights", default="checkpoints/siglip2_summary_head.pth")
    parser.add_argument("--radio_checkpoint", default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar")
    return parser.parse_args()


def main() -> None:
    train_adapter(_parse_args())


if __name__ == "__main__":
    main()
