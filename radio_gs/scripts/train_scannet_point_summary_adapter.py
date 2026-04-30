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
    decoder_anchor_summary: torch.Tensor | None = None,
    decoder_anchor_weight: float = 0.0,
    detach_compact: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute adapter-only summary and text-distillation losses."""
    compact_in = compact.detach() if detach_compact else compact
    pred_summary = F.normalize(adapter(compact_in.float()).float(), dim=-1)
    teacher_summary = F.normalize(teacher_summary.float(), dim=-1)

    zero = pred_summary.sum() * 0.0
    summary_loss = zero
    if summary_weight > 0:
        summary_loss = 1.0 - (pred_summary * teacher_summary).sum(dim=-1).mean()

    text_distill_loss = zero
    text_valid_ratio = zero.detach()
    text_teacher_conf = zero.detach()
    text_agreement = zero.detach()
    text_pseudo_ce_loss = zero
    text_pseudo_ce_valid_ratio = zero.detach()
    text_pseudo_ce_teacher_conf = zero.detach()
    text_pseudo_ce_agreement = zero.detach()
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
            text_distill_loss = F.kl_div(
                F.log_softmax(student_logits[valid] / temperature, dim=-1),
                teacher_probs[valid],
                reduction="batchmean",
            )
            if temperature != 1.0:
                text_distill_loss = text_distill_loss * (temperature**2)
            with torch.no_grad():
                student_pred = student_logits[valid].argmax(dim=-1)
                teacher_pred = teacher_probs[valid].argmax(dim=-1)
                text_agreement = (student_pred == teacher_pred).float().mean()
                text_teacher_conf = teacher_conf[valid].mean()

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
            text_pseudo_ce_loss = F.cross_entropy(student_logits[valid], teacher_targets[valid])
            with torch.no_grad():
                student_pred = student_logits[valid].argmax(dim=-1)
                text_pseudo_ce_agreement = (student_pred == teacher_targets[valid]).float().mean()
                text_pseudo_ce_teacher_conf = teacher_conf[valid].mean()

    total = (
        float(summary_weight) * summary_loss
        + float(text_distill_weight) * text_distill_loss
        + float(text_pseudo_ce_weight) * text_pseudo_ce_loss
    )
    if decoder_anchor_weight > 0 and decoder_anchor_summary is not None:
        decoder_anchor_summary = F.normalize(decoder_anchor_summary.float(), dim=-1)
        decoder_anchor_loss = 1.0 - (pred_summary * decoder_anchor_summary).sum(dim=-1).mean()
        total = total + float(decoder_anchor_weight) * decoder_anchor_loss
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
        "decoder_anchor_loss": decoder_anchor_loss.detach(),
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


def _load_summary_head(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    head_path = Path(args.summary_head_weights)
    if head_path.exists():
        head = SigLIP2SummaryHead.from_extracted_weights(str(head_path))
    else:
        head = SigLIP2SummaryHead.from_radio_checkpoint(args.radio_checkpoint)
    return head.to(device).eval()


def _load_teacher_cache(path: str, device: torch.device, *, valid_only: bool) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    required = {"xyz", "features", "valid"}
    missing = sorted(required.difference(payload.keys()))
    if missing:
        raise KeyError(f"teacher cache missing keys: {missing}")
    valid = payload["valid"].bool()
    if valid_only:
        indices = torch.nonzero(valid, as_tuple=False).reshape(-1)
    else:
        indices = torch.arange(int(valid.numel()))
    result = {
        "indices": indices.to(device=device, dtype=torch.long),
        "xyz": payload["xyz"][indices].to(device=device, dtype=torch.float32),
        "features": payload["features"][indices].to(device=device, dtype=torch.float32),
        "valid": valid[indices].to(device=device),
    }
    if "labels" in payload and payload["labels"] is not None:
        result["labels"] = payload["labels"][indices].to(device=device, dtype=torch.long)
    return result


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


@torch.no_grad()
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
    return result[compact_feature_key].detach()


def train_adapter(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    config = load_config(args.config)
    model, codec = _build_hybrid_model(config, args.checkpoint, device)
    for param in model.parameters():
        param.requires_grad_(False)
    for param in codec.parameters():
        param.requires_grad_(False)

    summary_head = _load_summary_head(args, device)
    teacher = _load_teacher_cache(args.teacher_cache, device, valid_only=not args.include_invalid)
    if args.max_points and teacher["indices"].numel() > args.max_points:
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)
        order = torch.randperm(teacher["indices"].numel(), generator=generator, device=device)[: args.max_points]
        teacher = {key: value[order] for key, value in teacher.items()}

    teacher_summary = _project_teacher_summary(
        teacher["features"],
        summary_head,
        chunk_size=args.teacher_chunk_size,
    ).to(device)
    del teacher["features"]

    text_embeddings = None
    if args.text_distill_weight > 0 or args.text_pseudo_ce_weight > 0:
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
    base_ckpt = torch.load(args.checkpoint, map_location="cpu")
    if args.init_from_checkpoint and "point_summary_adapter_state_dict" in base_ckpt:
        adapter.load_state_dict(base_ckpt["point_summary_adapter_state_dict"], strict=False)

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
            "decoder_anchor_loss": 0.0,
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
                decoder_anchor_summary=decoder_anchor_summary,
                decoder_anchor_weight=args.decoder_anchor_weight,
                detach_compact=True,
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
            f"anchor={epoch_stats['decoder_anchor_loss']:.4f} "
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
        "decoder_anchor_weight": args.decoder_anchor_weight,
        "class_split": args.class_split,
        "prompt_templates": parse_prompt_templates(args.prompt_templates),
        "config_snapshot": config_to_dict(config),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    latest_adapter_state = {
        key: value.detach().cpu()
        for key, value in adapter.state_dict().items()
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
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--summary_weight", type=float, default=1.0)
    parser.add_argument("--text_distill_weight", type=float, default=1.0)
    parser.add_argument("--text_distill_temperature", type=float, default=0.7)
    parser.add_argument("--text_confidence_threshold", type=float, default=0.0)
    parser.add_argument("--text_pseudo_ce_weight", type=float, default=0.0)
    parser.add_argument("--text_pseudo_ce_confidence_threshold", type=float, default=0.0)
    parser.add_argument("--text_pseudo_ce_logit_scale", type=float, default=1.0)
    parser.add_argument("--decoder_anchor_weight", type=float, default=0.0)
    parser.add_argument("--text_embedding_cache", default="checkpoints/siglip2_scannet_og_text_embeddings_ens5.pt")
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
