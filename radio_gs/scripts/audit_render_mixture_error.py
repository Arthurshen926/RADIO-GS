#!/usr/bin/env python3
"""Bucket held-out reconstruction error by exact 3DGS mixture statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from radio_gs.config import load_config
from radio_gs.field import load_canonical_field_checkpoint
from radio_gs.models.radio_adaptors import (
    load_radio_adaptor_from_checkpoint,
    project_feature_map_with_adaptor,
)
from radio_gs.rendering.coefficient_renderer import render_canonical_radio
from radio_gs.rendering.contribution_compositor import (
    contribution_rank,
    rasterize_single_view_contributions,
)
from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline
from radio_gs.training.feature_training_utils import SimpleRadioDataset


def _parse_ids(value: str) -> list[int]:
    return [int(token) for token in value.replace(",", " ").split()]


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float() - left.float().mean()
    right = right.float() - right.float().mean()
    denom = left.norm() * right.norm()
    return float((left * right).sum() / denom) if float(denom) > 0 else 0.0


def _quartiles(variable: torch.Tensor, error: torch.Tensor) -> list[dict]:
    edges = torch.quantile(variable, torch.linspace(0, 1, 5, device=variable.device))
    rows = []
    for index in range(4):
        mask = (variable >= edges[index]) & (
            variable <= edges[index + 1] if index == 3 else variable < edges[index + 1]
        )
        rows.append(
            {
                "minimum": float(edges[index]),
                "maximum": float(edges[index + 1]),
                "pixels": int(mask.sum()),
                "mean_cosine_error": float(error[mask].mean()) if bool(mask.any()) else None,
            }
        )
    return rows


def audit(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    config = load_config(args.config)
    model, _codec, renderer, _sharpener, _refiner, _cfg, _hybrid = load_render_pipeline(
        args.config, args.geometry_checkpoint, device, strict_checkpoint_contract=True,
        load_ply_rgb_features=False,
    )
    field, payload = load_canonical_field_checkpoint(args.field_checkpoint, map_location="cpu")
    field = field.to(device).eval()
    feature_dir = Path(str(config.feature_dir))
    pose_dir = str(getattr(config, "pose_dir", "") or "")
    if not pose_dir or not Path(pose_dir).is_dir():
        pose_dir = str(feature_dir / "poses_w2c")
    dataset = SimpleRadioDataset(
        feature_dir=str(feature_dir), pose_file=None, pose_dir=pose_dir,
        feature_size=(int(config.feature_height), int(config.feature_width)),
        split="validation", dataset_type=str(getattr(config, "dataset_type", "lerf")),
    )
    frame_to_index = {int(frame): index for index, frame in enumerate(dataset.frame_indices)}
    adaptors = {
        name: load_radio_adaptor_from_checkpoint(
            args.radio_checkpoint, name, kind="feature_projection"
        ).to(device).eval()
        for name in ("dino_v3", "sam3")
    }
    values: dict[str, list[torch.Tensor]] = {
        key: [] for key in (
            "entropy", "top2_ratio", "depth_gap", "sam_boundary",
            "raw_error", "dino_error", "sam_error",
        )
    }
    height, width = int(config.feature_height), int(config.feature_width)
    with torch.inference_mode():
        for frame in _parse_ids(args.frame_ids):
            sample = dataset[frame_to_index[frame]]
            pose = sample["pose_w2c"].to(device)
            hits = rasterize_single_view_contributions(
                model, renderer, pose, height=height, width=width
            )
            pids, weights = hits["pixel_ids"], hits["weights"].float()
            rank = contribution_rank(pids, weights)
            mass = torch.zeros(height * width, device=device).index_add_(0, pids, weights)
            normalized = weights / mass[pids].clamp_min(1e-8)
            entropy = torch.zeros(height * width, device=device).index_add_(
                0, pids, -normalized * normalized.clamp_min(1e-8).log()
            )
            top1 = torch.zeros(height * width, device=device)
            top2 = torch.zeros(height * width, device=device)
            depth1 = torch.zeros(height * width, device=device)
            depth2 = torch.zeros(height * width, device=device)
            for target_rank, target_weight, target_depth in (
                (0, top1, depth1), (1, top2, depth2)
            ):
                chosen = rank == target_rank
                target_weight[pids[chosen]] = normalized[chosen]
                target_depth[pids[chosen]] = hits["depths"][chosen]
            rendered = render_canonical_radio(
                renderer, model, field, pose, feature_height=height, feature_width=width
            )
            prediction = rendered["feature_map"][None].float()
            teacher = sample["radio_features"].to(device)[None].float()
            valid = rendered["alpha_map"].reshape(-1) >= float(args.alpha_threshold)
            errors = {
                "raw_error": 1.0 - F.cosine_similarity(prediction, teacher, dim=1),
            }
            for name, adaptor in adaptors.items():
                predicted_cap = project_feature_map_with_adaptor(prediction, adaptor)
                teacher_cap = project_feature_map_with_adaptor(teacher, adaptor)
                errors[f"{'dino' if name == 'dino_v3' else 'sam'}_error"] = (
                    1.0 - F.cosine_similarity(predicted_cap, teacher_cap, dim=1)
                )
            teacher_sam = project_feature_map_with_adaptor(teacher, adaptors["sam3"])[0]
            horizontal = 1.0 - (teacher_sam[:, :, 1:] * teacher_sam[:, :, :-1]).sum(0)
            vertical = 1.0 - (teacher_sam[:, 1:, :] * teacher_sam[:, :-1, :]).sum(0)
            boundary = torch.zeros(height, width, device=device)
            boundary[:, 1:] = torch.maximum(boundary[:, 1:], horizontal)
            boundary[:, :-1] = torch.maximum(boundary[:, :-1], horizontal)
            boundary[1:, :] = torch.maximum(boundary[1:, :], vertical)
            boundary[:-1, :] = torch.maximum(boundary[:-1, :], vertical)
            values["entropy"].append(entropy[valid].cpu())
            values["top2_ratio"].append((top2 / top1.clamp_min(1e-8))[valid].cpu())
            values["depth_gap"].append((depth2 - depth1).abs()[valid].cpu())
            values["sam_boundary"].append(boundary.reshape(-1)[valid].cpu())
            for name, error in errors.items():
                values[name].append(error.reshape(-1)[valid].cpu())
    merged = {name: torch.cat(parts) for name, parts in values.items()}
    report = {
        "schema_version": 1,
        "field_checkpoint": str(Path(args.field_checkpoint).resolve()),
        "frame_ids": _parse_ids(args.frame_ids),
        "query_free": True,
        "variables": {},
    }
    for variable in ("entropy", "top2_ratio", "depth_gap", "sam_boundary"):
        report["variables"][variable] = {
            error_name: {
                "pearson": _pearson(merged[variable], merged[error_name]),
                "quartiles": _quartiles(merged[variable], merged[error_name]),
            }
            for error_name in ("raw_error", "dino_error", "sam_error")
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--geometry-checkpoint", required=True)
    parser.add_argument("--field-checkpoint", required=True)
    parser.add_argument("--radio-checkpoint", required=True)
    parser.add_argument("--frame-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--alpha-threshold", type=float, default=0.02)
    args = parser.parse_args()
    print(json.dumps(audit(args), indent=2))


if __name__ == "__main__":
    main()
