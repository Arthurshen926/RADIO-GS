#!/usr/bin/env python3
"""Train the optional global region aligner on generic crop pairs only."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from radio_gs.interfaces.semantic_alignment import (
    GlobalRegionSummaryBridge,
    GlobalSemanticBridgeManifest,
)
from radio_gs.models.siglip_projection import SigLIP2SummaryHead


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split_rows(
    metadata: dict,
    num_rows: int,
    *,
    validation_fraction: float,
    seed: int,
    split_unit: str,
    validation_excluded_images: set[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    fraction = float(validation_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation-fraction must be strictly between zero and one")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    excluded = {str(value) for value in (validation_excluded_images or set())}
    if split_unit == "crop":
        if excluded:
            raise ValueError("validation image exclusions require --validation-unit image")
        order = torch.randperm(int(num_rows), generator=generator)
        validation_count = max(1, int(round(int(num_rows) * fraction)))
        validation_rows = order[:validation_count]
        training_rows = order[validation_count:]
        split = {
            "unit": "crop",
            "seed": int(seed),
            "validation_fraction": fraction,
            "validation_rows": int(validation_rows.numel()),
        }
    elif split_unit == "image":
        records = metadata.get("crop_records", [])
        if not isinstance(records, list) or len(records) != int(num_rows):
            raise ValueError("image-level validation requires one crop_record per cache row")
        row_images = [str(record.get("image", "")) for record in records]
        if any(not value for value in row_images):
            raise ValueError("every crop_record must identify its source image")
        unique_images = sorted(set(row_images))
        validation_image_count = max(1, int(round(len(unique_images) * fraction)))
        candidates = [value for value in unique_images if value not in excluded]
        if len(candidates) < validation_image_count:
            raise ValueError("too few unseen images remain for the requested validation split")
        order = torch.randperm(len(candidates), generator=generator).tolist()
        validation_images = {
            candidates[index] for index in order[:validation_image_count]
        }
        validation_mask = torch.tensor(
            [value in validation_images for value in row_images], dtype=torch.bool
        )
        validation_rows = torch.where(validation_mask)[0]
        training_rows = torch.where(~validation_mask)[0]
        image_manifest = "\n".join(sorted(validation_images)).encode("utf-8")
        excluded_manifest = "\n".join(sorted(excluded)).encode("utf-8")
        split = {
            "unit": "image",
            "seed": int(seed),
            "validation_fraction": fraction,
            "training_images": len(unique_images) - len(validation_images),
            "validation_images": len(validation_images),
            "validation_rows": int(validation_rows.numel()),
            "validation_image_manifest_sha256": hashlib.sha256(image_manifest).hexdigest(),
            "validation_excluded_images": len(excluded),
            "validation_exclusion_manifest_sha256": hashlib.sha256(
                excluded_manifest
            ).hexdigest(),
        }
    else:
        raise ValueError(f"unsupported validation unit: {split_unit}")
    if training_rows.numel() == 0 or validation_rows.numel() == 0:
        raise ValueError("generic crop cache is too small for a train/validation split")
    return training_rows, validation_rows, split


def _images_from_cache(path: str | Path) -> set[str]:
    cache = torch.load(Path(path), map_location="cpu")
    if not isinstance(cache, dict) or not isinstance(cache.get("metadata"), dict):
        raise ValueError("validation exclusion cache must contain metadata")
    records = cache["metadata"].get("crop_records", [])
    if not isinstance(records, list) or not records:
        raise ValueError("validation exclusion cache must contain crop_records")
    images = {str(record.get("image", "")) for record in records}
    if "" in images:
        raise ValueError("validation exclusion cache contains an unnamed image")
    return images


def _parse_token_grid_sizes(metadata: dict, raw: str) -> tuple[int, tuple[int, ...]]:
    """Resolve square source/augmentation grids for density-robust training."""

    source_shape = metadata.get("region_token_grid", [])
    if (
        not isinstance(source_shape, (list, tuple))
        or len(source_shape) != 2
        or int(source_shape[0]) != int(source_shape[1])
        or int(source_shape[0]) <= 0
    ):
        raise ValueError("generic crop cache must declare a square region_token_grid")
    source_grid = int(source_shape[0])
    values = str(raw).replace(",", " ").split()
    grids = (
        tuple(sorted({int(value) for value in values}))
        if values
        else (source_grid,)
    )
    if not grids or grids[0] <= 0 or grids[-1] > source_grid:
        raise ValueError("token grid sizes must lie in [1, source token grid]")
    return source_grid, grids


def _pool_region_tokens(
    tokens: torch.Tensor,
    *,
    source_grid: int,
    target_grid: int,
) -> torch.Tensor:
    """Spatially pool one square token grid without changing its region target."""

    values = torch.as_tensor(tokens)
    if values.ndim != 3 or values.shape[1] != int(source_grid) ** 2:
        raise ValueError("tokens must be [B,source_grid**2,C]")
    if not 0 < int(target_grid) <= int(source_grid):
        raise ValueError("target_grid must lie in [1, source_grid]")
    if int(target_grid) == int(source_grid):
        return values
    maps = values.transpose(1, 2).reshape(
        values.shape[0], values.shape[2], int(source_grid), int(source_grid)
    )
    return F.adaptive_avg_pool2d(
        maps.float(), (int(target_grid), int(target_grid))
    ).flatten(2).transpose(1, 2).to(values.dtype)


@torch.no_grad()
def _metrics(
    bridge,
    summary_head,
    tokens,
    target_token,
    target_descriptor,
    rows,
    device,
    *,
    source_grid: int,
    token_grid_sizes: tuple[int, ...],
):
    per_grid = {}
    target_rows = target_descriptor[rows].to(device)
    target_centered = target_rows - target_rows.mean(dim=0, keepdim=True)
    for grid in token_grid_sizes:
        inputs = _pool_region_tokens(
            tokens[rows], source_grid=source_grid, target_grid=grid
        ).to(device)
        predicted_token = bridge(inputs)
        predicted_descriptor = F.normalize(
            summary_head(predicted_token[:, None])[:, 0].float(), dim=-1, eps=1e-8
        )
        predicted_centered = predicted_descriptor - predicted_descriptor.mean(
            dim=0, keepdim=True
        )
        per_grid[str(grid)] = {
            "summary_token_cosine": float(
                F.cosine_similarity(
                    predicted_token.cpu(), target_token[rows], dim=-1
                ).mean()
            ),
            "semantic_descriptor_cosine": float(
                F.cosine_similarity(
                    predicted_descriptor, target_rows, dim=-1
                ).mean()
            ),
            "semantic_descriptor_centered_cosine": float(
                F.cosine_similarity(
                    predicted_centered, target_centered, dim=-1, eps=1e-8
                ).mean()
            ),
        }
    keys = (
        "summary_token_cosine",
        "semantic_descriptor_cosine",
        "semantic_descriptor_centered_cosine",
    )
    return {
        key: sum(value[key] for value in per_grid.values()) / len(per_grid)
        for key in keys
    } | {"per_token_grid": per_grid}


def train(args: argparse.Namespace) -> dict:
    if int(args.early_stopping_patience) < 0:
        raise ValueError("early-stopping-patience must be non-negative")
    cache_path = Path(args.training_cache)
    cache = torch.load(cache_path, map_location="cpu")
    required = {
        "radio_region_tokens",
        "official_summary_tokens",
        "official_crop_summaries",
        "metadata",
    }
    if not isinstance(cache, dict) or not required.issubset(cache):
        raise ValueError(f"generic crop cache must contain {sorted(required)}")
    metadata = dict(cache["metadata"])
    if metadata.get("uses_benchmark_test_vocabulary", True):
        raise ValueError("global bridge cache cannot use benchmark test vocabulary")
    if metadata.get("uses_benchmark_scenes", True):
        raise ValueError("global bridge cache cannot use benchmark scenes")
    if metadata.get("training_scope") != "global_cross_scene":
        raise ValueError("global bridge cache must declare global_cross_scene")

    tokens = torch.as_tensor(cache["radio_region_tokens"]).float().cpu()
    target_token = torch.as_tensor(cache["official_summary_tokens"]).float().cpu()
    target_descriptor = F.normalize(
        torch.as_tensor(cache["official_crop_summaries"]).float().cpu(), dim=-1
    )
    if tokens.ndim != 3 or tokens.shape[-1] != 1280:
        raise ValueError("radio_region_tokens must be [N,T,1280]")
    if target_token.shape != (tokens.shape[0], 1280):
        raise ValueError("official_summary_tokens must be [N,1280]")
    if target_descriptor.ndim != 2 or target_descriptor.shape[0] != tokens.shape[0]:
        raise ValueError("official_crop_summaries must align as [N,D]")
    source_grid, token_grid_sizes = _parse_token_grid_sizes(
        metadata, args.token_grid_sizes
    )
    if tokens.shape[1] != source_grid**2:
        raise ValueError("region tokens disagree with metadata region_token_grid")

    device = torch.device(args.device)
    bridge = GlobalRegionSummaryBridge(
        input_dim=1280,
        output_dim=1280,
        hidden_dim=int(args.hidden_dim),
    ).to(device)
    summary_head = SigLIP2SummaryHead.from_radio_checkpoint(args.radio_checkpoint).to(device)
    summary_head.eval()
    for parameter in summary_head.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        bridge.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    excluded_images = (
        _images_from_cache(args.validation_exclusion_cache)
        if args.validation_exclusion_cache
        else set()
    )
    training_rows, validation_rows, validation_split = _split_rows(
        metadata,
        tokens.shape[0],
        validation_fraction=float(args.validation_fraction),
        seed=int(args.seed),
        split_unit=str(args.validation_unit),
        validation_excluded_images=excluded_images,
    )
    baseline = None
    if args.baseline_checkpoint:
        baseline_bridge, baseline_manifest = GlobalRegionSummaryBridge.from_checkpoint(
            args.baseline_checkpoint, map_location="cpu"
        )
        baseline_bridge = baseline_bridge.to(device).eval()
        baseline_metrics = _metrics(
            baseline_bridge,
            summary_head,
            tokens,
            target_token,
            target_descriptor,
            validation_rows,
            device,
            source_grid=source_grid,
            token_grid_sizes=token_grid_sizes,
        )
        baseline = {
            "checkpoint": str(Path(args.baseline_checkpoint).resolve()),
            "checkpoint_sha256": baseline_manifest.checkpoint_sha256,
            **baseline_metrics,
            "semantic_descriptor_selection_score": 0.5
            * (
                baseline_metrics["semantic_descriptor_cosine"]
                + baseline_metrics["semantic_descriptor_centered_cosine"]
            ),
        }
        del baseline_bridge
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps({"baseline": baseline, "validation_split": validation_split}), flush=True)
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    history = []
    best_descriptor_score = -1.0
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    early_stopped = False
    for epoch in range(int(args.epochs)):
        epoch_rows = training_rows[
            torch.randperm(training_rows.numel(), generator=generator)
        ]
        losses = []
        bridge.train()
        for batch_index, start in enumerate(
            range(0, epoch_rows.numel(), int(args.batch_size))
        ):
            rows = epoch_rows[start : start + int(args.batch_size)]
            target_grid = token_grid_sizes[
                (epoch * max(1, len(token_grid_sizes)) + batch_index)
                % len(token_grid_sizes)
            ]
            inputs = _pool_region_tokens(
                tokens[rows], source_grid=source_grid, target_grid=target_grid
            ).to(device)
            teacher_token = target_token[rows].to(device)
            teacher_descriptor = target_descriptor[rows].to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted_token = bridge(inputs)
            predicted_descriptor = F.normalize(
                summary_head(predicted_token[:, None])[:, 0].float(), dim=-1, eps=1e-8
            )
            token_cosine = (
                1.0 - F.cosine_similarity(predicted_token, teacher_token, dim=-1)
            ).mean()
            token_huber = F.huber_loss(predicted_token, teacher_token, delta=0.1)
            semantic_cosine = (1.0 - (predicted_descriptor * teacher_descriptor).sum(dim=-1)).mean()
            predicted_centered = predicted_descriptor - predicted_descriptor.mean(
                dim=0, keepdim=True
            )
            teacher_centered = teacher_descriptor - teacher_descriptor.mean(
                dim=0, keepdim=True
            )
            semantic_centered = (
                1.0
                - F.cosine_similarity(
                    predicted_centered, teacher_centered, dim=-1, eps=1e-8
                )
            ).mean()
            sample_count = min(predicted_descriptor.shape[0], int(args.relation_samples))
            pred_relation = predicted_descriptor[:sample_count] @ predicted_descriptor[:sample_count].T
            teacher_relation = teacher_descriptor[:sample_count] @ teacher_descriptor[:sample_count].T
            relation = F.smooth_l1_loss(pred_relation, teacher_relation)
            loss = (
                float(args.token_cosine_weight) * token_cosine
                + float(args.token_huber_weight) * token_huber
                + float(args.semantic_weight) * semantic_cosine
                + float(args.semantic_centered_weight) * semantic_centered
                + float(args.relation_weight) * relation
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        bridge.eval()
        validation = _metrics(
            bridge,
            summary_head,
            tokens,
            target_token,
            target_descriptor,
            validation_rows,
            device,
            source_grid=source_grid,
            token_grid_sizes=token_grid_sizes,
        )
        record = {
            "epoch": epoch + 1,
            "loss": sum(losses) / max(1, len(losses)),
            **validation,
        }
        descriptor_score = 0.5 * (
            validation["semantic_descriptor_cosine"]
            + validation["semantic_descriptor_centered_cosine"]
        )
        record["semantic_descriptor_selection_score"] = descriptor_score
        improved = descriptor_score > best_descriptor_score
        record["selected_as_best"] = improved
        history.append(record)
        if descriptor_score > best_descriptor_score:
            best_descriptor_score = descriptor_score
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in bridge.state_dict().items()
            }
        else:
            epochs_without_improvement += 1
        print(json.dumps(record), flush=True)
        if (
            int(args.early_stopping_patience) > 0
            and epochs_without_improvement >= int(args.early_stopping_patience)
        ):
            early_stopped = True
            print(
                json.dumps(
                    {
                        "early_stopping": True,
                        "completed_epochs": epoch + 1,
                        "best_epoch": best_epoch,
                        "patience": int(args.early_stopping_patience),
                    }
                ),
                flush=True,
            )
            break
    assert best_state is not None
    bridge.load_state_dict(best_state, strict=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    training_manifest_hash = str(metadata.get("dataset_manifest_sha256", "")) or _sha256(cache_path)
    provisional_manifest = GlobalSemanticBridgeManifest(
        checkpoint_sha256="pending",
        training_scope="global_cross_scene",
        frozen=True,
        uses_benchmark_test_vocabulary=False,
        uses_benchmark_scenes=False,
        training_dataset_manifest_sha256=training_manifest_hash,
    )
    payload = {
        "schema_version": 1,
        "architecture": {
            "input_dim": bridge.input_dim,
            "output_dim": bridge.output_dim,
            "hidden_dim": bridge.hidden_dim,
        },
        "state_dict": bridge.cpu().eval().state_dict(),
        "official_summary_head": {
            "name": "siglip2-g visual summary head",
            "radio_checkpoint_sha256": metadata.get("radio_checkpoint_sha256", ""),
            "custom_text_projection": False,
        },
        "manifest": asdict(provisional_manifest),
        "training_cache": str(cache_path.resolve()),
        "training_cache_metadata": metadata,
        "training_config": {key: value for key, value in vars(args).items()},
        "validation_split": validation_split,
        "source_token_grid": source_grid,
        "training_token_grid_sizes": list(token_grid_sizes),
        "baseline": baseline,
        "history": history,
        "best_epoch": best_epoch,
        "best_semantic_descriptor_score": best_descriptor_score,
        "completed_epochs": len(history),
        "early_stopped": early_stopped,
    }
    torch.save(payload, output)
    checkpoint_hash = _sha256(output)
    manifest = GlobalSemanticBridgeManifest(
        checkpoint_sha256=checkpoint_hash,
        training_scope="global_cross_scene",
        frozen=True,
        uses_benchmark_test_vocabulary=False,
        uses_benchmark_scenes=False,
        training_dataset_manifest_sha256=training_manifest_hash,
    )
    manifest.validate()
    sidecar = output.with_suffix(output.suffix + ".manifest.json")
    sidecar.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    report = {
        "output": str(output),
        "manifest": str(sidecar),
        "checkpoint_sha256": checkpoint_hash,
        "best_semantic_descriptor_score": best_descriptor_score,
        "best_epoch": best_epoch,
        "completed_epochs": len(history),
        "early_stopped": early_stopped,
        "baseline_semantic_descriptor_score": (
            None if baseline is None else baseline["semantic_descriptor_selection_score"]
        ),
        "semantic_descriptor_score_delta": (
            None
            if baseline is None
            else best_descriptor_score - baseline["semantic_descriptor_selection_score"]
        ),
        "num_pairs": tokens.shape[0],
        "validation_pairs": validation_rows.numel(),
        "validation_split": validation_split,
        "source_token_grid": source_grid,
        "training_token_grid_sizes": list(token_grid_sizes),
        "training_config": {key: value for key, value in vars(args).items()},
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--radio-checkpoint",
        default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many non-improving validation epochs; zero disables it.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--token-cosine-weight", type=float, default=0.25)
    parser.add_argument("--token-huber-weight", type=float, default=0.05)
    parser.add_argument("--semantic-weight", type=float, default=1.0)
    parser.add_argument("--semantic-centered-weight", type=float, default=1.0)
    parser.add_argument("--relation-weight", type=float, default=0.1)
    parser.add_argument("--relation-samples", type=int, default=64)
    parser.add_argument(
        "--token-grid-sizes",
        default="",
        help=(
            "Square token grids cycled during training and averaged for validation; "
            "empty keeps the cache's native grid only."
        ),
    )
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument(
        "--validation-unit", choices=["crop", "image"], default="crop"
    )
    parser.add_argument(
        "--validation-exclusion-cache",
        default="",
        help="Keep images present in this generic cache out of image-level validation.",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        default="",
        help="Evaluate a frozen prior bridge on the exact same validation rows.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
