#!/usr/bin/env python3
"""Train a frozen global crop-context bridge in official DINO feature space.

The bridge is trained only on scene-disjoint RGB crop/full-image pairs.  Its
input is the official DINO crop-centre descriptor plus the crop's official
spatial global mean; its target is the official full-image DINO descriptor at
the same pixel.  It never opens a test field, PFPR anchor, depth, pose, label,
instance, mask, text prompt, or metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from radio_gs.interfaces.crop_spatial_alignment import (
    CropContextAlignmentManifest,
    GlobalCropContextAdapter,
    checkpoint_sha256,
)


_SCOPE = "global_cross_scene_crop_context_to_spatial_dino"


def _physical_space(scene: str) -> str:
    return str(scene).split("_", 1)[0]


def _load(
    path: str | Path,
    *,
    role: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    source = Path(path)
    payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, Mapping) or int(payload.get("schema_version", -1)) != 1:
        raise ValueError(f"invalid crop-context cache: {source}")
    metadata = dict(payload.get("metadata", {}))
    required = {
        "training_scope": "global_cross_scene_crop_to_spatial_dino",
        "split_role": role,
        "uses_benchmark_scenes": False,
        "uses_benchmark_labels": False,
        "uses_depth": False,
        "uses_pose": False,
        "uses_instances": False,
        "uses_text": False,
        "physical_space_disjoint": True,
        "benchmark_exclusion_declared": True,
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(f"{source} violates crop-context cache contract: {key}")
    center = F.normalize(torch.as_tensor(payload.get("crop_descriptors")).float(), dim=-1, eps=1e-8)
    context = F.normalize(
        torch.as_tensor(payload.get("crop_context_descriptors")).float(), dim=-1, eps=1e-8
    )
    target = F.normalize(
        torch.as_tensor(payload.get("full_image_anchor_descriptors")).float(),
        dim=-1,
        eps=1e-8,
    )
    if (
        center.ndim != 2
        or context.shape != center.shape
        or target.shape != center.shape
        or not len(center)
    ):
        raise ValueError("crop-context cache tensors must be nonempty aligned [N,D]")
    scenes = [str(value) for value in metadata.get("scene_names", [])]
    if not scenes or len(set(scenes)) != len(scenes):
        raise ValueError("crop-context cache needs a unique nonempty scene list")
    metadata["path"] = str(source.resolve())
    metadata["sha256"] = checkpoint_sha256(source)
    return center, context, target, metadata


def _load_many(
    paths: list[str], *, role: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    if not paths:
        raise ValueError(f"at least one {role} crop-context cache is required")
    parts = [_load(path, role=role) for path in paths]
    seen_spaces: set[str] = set()
    exclusions: set[str] | None = None
    for _center, _context, _target, metadata in parts:
        spaces = {_physical_space(scene) for scene in metadata["scene_names"]}
        if seen_spaces & spaces:
            raise ValueError(f"duplicate physical space across {role} crop-context shards")
        seen_spaces |= spaces
        current = set(metadata.get("excluded_physical_spaces", []))
        if exclusions is None:
            exclusions = current
        elif exclusions != current:
            raise ValueError(f"{role} crop-context exclusion contracts differ")
    return (
        torch.cat([center for center, _context, _target, _metadata in parts]),
        torch.cat([context for _center, context, _target, _metadata in parts]),
        torch.cat([target for _center, _context, target, _metadata in parts]),
        {
            "paths": [metadata["path"] for _center, _context, _target, metadata in parts],
            "sha256s": [metadata["sha256"] for _center, _context, _target, metadata in parts],
            "scene_names": sorted(
                scene
                for _center, _context, _target, metadata in parts
                for scene in metadata["scene_names"]
            ),
            "excluded_physical_spaces": sorted(exclusions or set()),
        },
    )


@torch.inference_mode()
def _evaluate(
    model: GlobalCropContextAdapter,
    center: torch.Tensor,
    context: torch.Tensor,
    target: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> float:
    values: list[torch.Tensor] = []
    for start in range(0, len(center), int(batch_size)):
        stop = start + int(batch_size)
        prediction = model(center[start:stop].to(device), context[start:stop].to(device))
        values.append(F.cosine_similarity(prediction.cpu(), target[start:stop], dim=-1))
    return float(torch.cat(values).mean())


def _symmetric_contrastive_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Preserve one-to-one local discrimination in the frozen DINO space.

    The crop-context bridge previously optimized only positive cosine.  That
    can improve a full-image-teacher alignment score while collapsing nearby
    repeated appearances, precisely the failure that a pose-free point
    retrieval query must avoid.  In-batch targets are RGB-only examples from
    physically disjoint training scenes, so the negatives carry no benchmark
    identity, pose, depth, field, or evaluator information.
    """

    if float(temperature) <= 0:
        raise ValueError("contrastive temperature must be positive")
    query = F.normalize(torch.as_tensor(prediction).float(), dim=-1, eps=1e-8)
    keys = F.normalize(torch.as_tensor(target, device=query.device).float(), dim=-1, eps=1e-8)
    if query.ndim != 2 or keys.shape != query.shape or query.shape[0] <= 1:
        raise ValueError("contrastive crop-context pairs must be aligned [B,D] with B > 1")
    logits = query @ keys.T / float(temperature)
    labels = torch.arange(query.shape[0], device=query.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


@torch.inference_mode()
def _evaluate_retrieval(
    model: GlobalCropContextAdapter,
    center: torch.Tensor,
    context: torch.Tensor,
    target: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float]:
    """Measure generic full-image-token retrieval on the disjoint RGB-only set.

    This does not use PFPR anchors or candidate geometry.  It merely asks
    whether each crop query ranks its paired full-image teacher token above
    other RGB-only examples in the held-out generic cache.  The scalar MRR is
    used to choose a contrastive bridge before PFPR is opened.
    """

    targets = F.normalize(target.to(device).float(), dim=-1, eps=1e-8)
    top1: list[torch.Tensor] = []
    reciprocal_ranks: list[torch.Tensor] = []
    for start in range(0, len(center), int(batch_size)):
        stop = min(start + int(batch_size), len(center))
        prediction = model(center[start:stop].to(device), context[start:stop].to(device))
        logits = prediction @ targets.T
        correct = torch.arange(start, stop, device=device)
        positive = logits[torch.arange(stop - start, device=device), correct]
        ranks = 1 + (logits > positive[:, None]).sum(dim=1)
        top1.append((ranks == 1).float().cpu())
        reciprocal_ranks.append(ranks.reciprocal().float().cpu())
    return float(torch.cat(top1).mean()), float(torch.cat(reciprocal_ranks).mean())


def train(args: argparse.Namespace) -> dict[str, Any]:
    center_train, context_train, target_train, train_meta = _load_many(
        args.train_caches, role="train"
    )
    center_val, context_val, target_val, val_meta = _load_many(
        args.validation_caches, role="validation"
    )
    train_spaces = {_physical_space(scene) for scene in train_meta["scene_names"]}
    val_spaces = {_physical_space(scene) for scene in val_meta["scene_names"]}
    if train_spaces & val_spaces:
        raise ValueError("crop-context train/validation physical spaces overlap")
    forbidden = set(train_meta["excluded_physical_spaces"]) | set(
        val_meta["excluded_physical_spaces"]
    )
    if (train_spaces | val_spaces) & forbidden:
        raise ValueError("excluded benchmark physical space leaked into crop-context training")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    contrastive_weight = float(getattr(args, "contrastive_weight", 0.0))
    contrastive_temperature = float(getattr(args, "contrastive_temperature", 0.07))
    if contrastive_weight < 0 or contrastive_temperature <= 0:
        raise ValueError("contrastive weight must be non-negative and temperature positive")
    model = GlobalCropContextAdapter(
        feature_dim=int(center_train.shape[1]), hidden_dim=int(args.hidden_dim)
    ).to(device)
    baseline = _evaluate(
        model, center_val, context_val, target_val, device=device, batch_size=int(args.batch_size)
    )
    baseline_top1, baseline_mrr = _evaluate_retrieval(
        model,
        center_val,
        context_val,
        target_val,
        device=device,
        batch_size=int(args.batch_size),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    generator = torch.Generator().manual_seed(int(args.seed))
    selection_metric = (
        "validation_retrieval_mrr" if contrastive_weight > 0 else "validation_cosine"
    )
    best_score, best_epoch, stale = (
        baseline_mrr if contrastive_weight > 0 else baseline,
        0,
        0,
    )
    best_validation_cosine = baseline
    best_validation_top1 = baseline_top1
    best_validation_mrr = baseline_mrr
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        order = torch.randperm(len(center_train), generator=generator)
        losses: list[float] = []
        cosine_losses: list[float] = []
        contrastive_losses: list[float] = []
        for start in range(0, len(order), int(args.batch_size)):
            rows = order[start : start + int(args.batch_size)]
            prediction = model(center_train[rows].to(device), context_train[rows].to(device))
            target = target_train[rows].to(device)
            cosine_loss = (
                1.0 - F.cosine_similarity(prediction, target, dim=-1)
            ).mean()
            contrastive_loss = (
                _symmetric_contrastive_loss(
                    prediction, target, temperature=contrastive_temperature
                )
                if contrastive_weight > 0 and int(rows.numel()) > 1
                else cosine_loss.new_zeros(())
            )
            loss = cosine_loss + contrastive_weight * contrastive_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            cosine_losses.append(float(cosine_loss.detach()))
            contrastive_losses.append(float(contrastive_loss.detach()))
        model.eval()
        validation_cosine = _evaluate(
            model, center_val, context_val, target_val, device=device, batch_size=int(args.batch_size)
        )
        validation_top1, validation_mrr = _evaluate_retrieval(
            model,
            center_val,
            context_val,
            target_val,
            device=device,
            batch_size=int(args.batch_size),
        )
        score = validation_mrr if contrastive_weight > 0 else validation_cosine
        record = {
            "epoch": epoch,
            "loss": float(sum(losses) / len(losses)),
            "cosine_loss": float(sum(cosine_losses) / len(cosine_losses)),
            "contrastive_loss": float(sum(contrastive_losses) / len(contrastive_losses)),
            "validation_cosine": validation_cosine,
            "validation_retrieval_top1": validation_top1,
            "validation_retrieval_mrr": validation_mrr,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_validation_cosine = validation_cosine
            best_validation_top1 = validation_top1
            best_validation_mrr = validation_mrr
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if int(args.patience) and stale >= int(args.patience):
            break
    model.load_state_dict(best_state)
    provenance = {
        "training_scope": _SCOPE,
        "frozen": True,
        "uses_benchmark_scenes": False,
        "uses_benchmark_labels": False,
        "scene_disjoint": True,
        "train_caches": train_meta["paths"],
        "train_cache_sha256s": train_meta["sha256s"],
        "validation_caches": val_meta["paths"],
        "validation_cache_sha256s": val_meta["sha256s"],
        "train_scenes": train_meta["scene_names"],
        "validation_scenes": val_meta["scene_names"],
        "excluded_physical_spaces": sorted(forbidden),
        "teacher": "official_c_radio_v4_dino_v3_7b_spatial",
        "adapter_input": "official_dino_crop_center_3x3_plus_spatial_global_mean",
        "adapter_target": "official_dino_full_image_anchor_spatial",
        "training_loss": {
            "positive_cosine_weight": 1.0,
            "symmetric_inbatch_contrastive_weight": contrastive_weight,
            "symmetric_inbatch_contrastive_temperature": contrastive_temperature,
            "selection_metric": selection_metric,
        },
    }
    training_manifest_sha256 = hashlib.sha256(
        json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "architecture": model.architecture(),
            "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
            "provenance": provenance,
            "history": history,
            "best_epoch": best_epoch,
            "baseline_validation_cosine": baseline,
            "baseline_validation_retrieval_top1": baseline_top1,
            "baseline_validation_retrieval_mrr": baseline_mrr,
            "best_validation_cosine": best_validation_cosine,
            "best_validation_retrieval_top1": best_validation_top1,
            "best_validation_retrieval_mrr": best_validation_mrr,
            "best_selection_score": best_score,
            "selection_metric": selection_metric,
        },
        output,
    )
    manifest = CropContextAlignmentManifest(
        checkpoint_sha256=checkpoint_sha256(output),
        training_scope=_SCOPE,
        frozen=True,
        uses_benchmark_scenes=False,
        uses_benchmark_labels=False,
        scene_disjoint=True,
        training_manifest_sha256=training_manifest_sha256,
    )
    manifest.validate()
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest.__dict__, indent=2), encoding="utf-8"
    )
    report = {
        "output": str(output.resolve()),
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "baseline_validation_cosine": baseline,
        "baseline_validation_retrieval_top1": baseline_top1,
        "baseline_validation_retrieval_mrr": baseline_mrr,
        "best_validation_cosine": best_validation_cosine,
        "best_validation_retrieval_top1": best_validation_top1,
        "best_validation_retrieval_mrr": best_validation_mrr,
        "best_selection_score": best_score,
        "selection_metric": selection_metric,
        "best_epoch": best_epoch,
        "train_pairs": len(center_train),
        "validation_pairs": len(center_val),
        "provenance": provenance,
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-caches", nargs="+", required=True)
    parser.add_argument("--validation-caches", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--contrastive-weight",
        type=float,
        default=0.0,
        help="generic RGB-only in-batch discrimination regularizer; zero preserves cosine-only v1",
    )
    parser.add_argument("--contrastive-temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if min(args.hidden_dim, args.epochs, args.batch_size) <= 0:
        parser.error("hidden dimension, epochs, and batch size must be positive")
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
