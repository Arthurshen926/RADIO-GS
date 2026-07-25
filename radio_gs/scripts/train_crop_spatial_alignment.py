#!/usr/bin/env python3
"""Train a frozen global crop-context adapter in official DINO feature space."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from radio_gs.interfaces.crop_spatial_alignment import (
    CropSpatialAlignmentManifest,
    GlobalCropSpatialAdapter,
    checkpoint_sha256,
)


def _physical_space(scene: str) -> str:
    return str(scene).split("_", 1)[0]


def _load(path: str | Path, *, role: str) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    source = Path(path)
    payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, Mapping) or int(payload.get("schema_version", -1)) != 1:
        raise ValueError(f"invalid crop-spatial cache: {source}")
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
            raise ValueError(f"{source} violates crop-spatial cache contract: {key}")
    crop = F.normalize(torch.as_tensor(payload.get("crop_descriptors")).float(), dim=-1, eps=1e-8)
    target = F.normalize(torch.as_tensor(payload.get("full_image_anchor_descriptors")).float(), dim=-1, eps=1e-8)
    if crop.ndim != 2 or target.shape != crop.shape or not len(crop):
        raise ValueError("crop-spatial cache descriptors must be nonempty aligned [N,D]")
    scenes = [str(value) for value in metadata.get("scene_names", [])]
    if not scenes or len(set(scenes)) != len(scenes):
        raise ValueError("crop-spatial cache needs a unique nonempty scene list")
    metadata["path"] = str(source.resolve())
    metadata["sha256"] = checkpoint_sha256(source)
    return crop, target, metadata


def _load_many(paths: list[str], *, role: str) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if not paths:
        raise ValueError(f"at least one {role} crop-spatial cache is required")
    parts = [_load(path, role=role) for path in paths]
    seen_spaces: set[str] = set()
    exclusions: set[str] | None = None
    for _crop, _target, metadata in parts:
        spaces = {_physical_space(scene) for scene in metadata["scene_names"]}
        if seen_spaces & spaces:
            raise ValueError(f"duplicate physical space across {role} crop-spatial shards")
        seen_spaces |= spaces
        current = set(metadata.get("excluded_physical_spaces", []))
        if exclusions is None:
            exclusions = current
        elif exclusions != current:
            raise ValueError(f"{role} crop-spatial exclusion contracts differ")
    return (
        torch.cat([crop for crop, _target, _metadata in parts]),
        torch.cat([target for _crop, target, _metadata in parts]),
        {
            "paths": [metadata["path"] for _crop, _target, metadata in parts],
            "sha256s": [metadata["sha256"] for _crop, _target, metadata in parts],
            "scene_names": sorted(
                scene for _crop, _target, metadata in parts for scene in metadata["scene_names"]
            ),
            "excluded_physical_spaces": sorted(exclusions or set()),
        },
    )


@torch.inference_mode()
def _evaluate(
    model: GlobalCropSpatialAdapter,
    crop: torch.Tensor,
    target: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> float:
    values: list[torch.Tensor] = []
    for start in range(0, len(crop), int(batch_size)):
        prediction = model(crop[start : start + int(batch_size)].to(device))
        values.append(F.cosine_similarity(prediction.cpu(), target[start : start + int(batch_size)], dim=-1))
    return float(torch.cat(values).mean())


def train(args: argparse.Namespace) -> dict[str, Any]:
    crop_train, target_train, train_meta = _load_many(args.train_caches, role="train")
    crop_val, target_val, val_meta = _load_many(args.validation_caches, role="validation")
    train_spaces = {_physical_space(scene) for scene in train_meta["scene_names"]}
    val_spaces = {_physical_space(scene) for scene in val_meta["scene_names"]}
    if train_spaces & val_spaces:
        raise ValueError("crop-spatial train/validation physical spaces overlap")
    forbidden = set(train_meta.get("excluded_physical_spaces", [])) | set(
        val_meta.get("excluded_physical_spaces", [])
    )
    if (train_spaces | val_spaces) & forbidden:
        raise ValueError("excluded benchmark physical space leaked into crop alignment")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = GlobalCropSpatialAdapter(
        feature_dim=int(crop_train.shape[1]), hidden_dim=int(args.hidden_dim)
    ).to(device)
    baseline = _evaluate(model, crop_val, target_val, device=device, batch_size=int(args.batch_size))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    generator = torch.Generator().manual_seed(int(args.seed))
    best_score, best_epoch, best_state, stale = baseline, 0, {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }, 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        order = torch.randperm(len(crop_train), generator=generator)
        losses: list[float] = []
        for start in range(0, len(order), int(args.batch_size)):
            rows = order[start : start + int(args.batch_size)]
            prediction = model(crop_train[rows].to(device))
            target = target_train[rows].to(device)
            loss = (1.0 - F.cosine_similarity(prediction, target, dim=-1)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        score = _evaluate(model, crop_val, target_val, device=device, batch_size=int(args.batch_size))
        record = {"epoch": epoch, "loss": float(sum(losses) / len(losses)), "validation_cosine": score}
        history.append(record)
        print(json.dumps(record), flush=True)
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if int(args.patience) and stale >= int(args.patience):
            break
    model.load_state_dict(best_state)
    provenance = {
        "training_scope": "global_cross_scene_crop_to_spatial_dino",
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
        "adapter_input": "official_dino_center_3x3_on_128px_crop",
        "adapter_target": "official_dino_full_image_anchor_spatial",
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
            "best_validation_cosine": best_score,
        },
        output,
    )
    manifest = CropSpatialAlignmentManifest(
        checkpoint_sha256=checkpoint_sha256(output),
        training_scope="global_cross_scene_crop_to_spatial_dino",
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
        "best_validation_cosine": best_score,
        "best_epoch": best_epoch,
        "train_pairs": len(crop_train),
        "validation_pairs": len(crop_val),
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    print(json.dumps(train(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
