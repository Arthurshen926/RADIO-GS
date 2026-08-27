"""Uncompressed native-teacher exact-MPR ceiling for SUGM-v3.1."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from radio_gs.utils.immutable_artifacts import write_frozen_json
from radio_gs.v3.evaluation.structured_source_capability import _same_pixel_retrieval
from radio_gs.v3.training.instance_upper_bound import sha256_file


def _teacher_path(root: Path, kind: str, frame: int) -> Path:
    if kind == "radio":
        return root / "backbone" / f"rgb_{frame}.pt"
    return root / f"frame_{frame:05d}.pt"


def _load_teacher(root: Path, kind: str, frame: int) -> torch.Tensor:
    value = torch.load(_teacher_path(root, kind, frame), map_location="cpu")
    if kind == "dinov2":
        if value.get("schema") != "radio_gs.native_dinov2_exact_mpr_teacher.v1":
            raise ValueError("native DINOv2 frame contract differs")
        value = value["feature"]
    feature = torch.as_tensor(value).float()
    if feature.ndim != 3 or not bool(torch.isfinite(feature).all()):
        raise ValueError("native source teacher axes differ")
    return feature


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    membership_path = Path(args.membership).resolve(strict=True)
    root = Path(args.teacher_root).resolve(strict=True)
    membership = torch.load(membership_path, map_location="cpu")
    records = membership["metadata"]["source_records"]
    train = [record for record in records if int(record["source_view_index"]) % 4 in (1, 2)]
    dev = [record for record in records if int(record["source_view_index"]) % 4 == 3]
    if len(train) != 16 or len(dev) != 8:
        raise ValueError("uncompressed ceiling requires the frozen source32 split")
    first = _load_teacher(root, args.teacher_kind, int(train[0]["frame_id"]))
    channels, height, width = first.shape
    rows = int(membership["num_rows"])
    feature_sum = torch.zeros(rows, channels)
    mass_sum = torch.zeros(rows)
    for record in train:
        feature = _load_teacher(root, args.teacher_kind, int(record["frame_id"]))
        if tuple(feature.shape) != (channels, height, width):
            raise ValueError("native source teacher shape changed")
        pixels = F.normalize(
            feature.permute(1, 2, 0).reshape(-1, channels), dim=-1, eps=1e-8
        )
        shard = torch.load(Path(record["responsibility_view"]), map_location="cpu")
        gaussian_ids = torch.as_tensor(shard["gaussian_ids"]).long()
        pixel_ids = torch.as_tensor(shard["pixel_ids"]).long()
        weights = torch.as_tensor(shard["base_weights"]).float()
        for start in range(0, gaussian_ids.numel(), args.hit_chunk):
            stop = min(start + args.hit_chunk, gaussian_ids.numel())
            ids = gaussian_ids[start:stop]
            contribution = pixels[pixel_ids[start:stop]] * weights[start:stop, None]
            feature_sum.index_add_(0, ids, contribution)
            mass_sum.index_add_(0, ids, weights[start:stop])
    observed = mass_sum > 0
    memory = feature_sum / mass_sum.clamp_min(1e-8)[:, None]
    memory[observed] = F.normalize(memory[observed], dim=-1, eps=1e-8)
    del feature_sum

    if args.robust_cosine_delta > 0:
        robust_sum = torch.zeros(rows, channels)
        robust_mass = torch.zeros(rows)
        for record in train:
            feature = _load_teacher(root, args.teacher_kind, int(record["frame_id"]))
            pixels = F.normalize(
                feature.permute(1, 2, 0).reshape(-1, channels), dim=-1, eps=1e-8
            )
            shard = torch.load(Path(record["responsibility_view"]), map_location="cpu")
            gaussian_ids = torch.as_tensor(shard["gaussian_ids"]).long()
            pixel_ids = torch.as_tensor(shard["pixel_ids"]).long()
            weights = torch.as_tensor(shard["base_weights"]).float()
            for start in range(0, gaussian_ids.numel(), args.hit_chunk):
                stop = min(start + args.hit_chunk, gaussian_ids.numel())
                ids = gaussian_ids[start:stop]
                values = pixels[pixel_ids[start:stop]]
                residual = 1.0 - (values * memory[ids]).sum(-1).clamp(-1.0, 1.0)
                robust = (
                    float(args.robust_cosine_delta)
                    / residual.clamp_min(float(args.robust_cosine_delta))
                ).clamp_min(float(args.robust_weight_floor))
                robust_weight = weights[start:stop] * robust
                robust_sum.index_add_(0, ids, values * robust_weight[:, None])
                robust_mass.index_add_(0, ids, robust_weight)
        robust_observed = robust_mass > 0
        memory = robust_sum / robust_mass.clamp_min(1e-8)[:, None]
        memory[robust_observed] = F.normalize(
            memory[robust_observed], dim=-1, eps=1e-8
        )
        del robust_sum
    memory = memory.half()

    device = torch.device(args.device)
    cosine_values, retrieval_values, native_values = [], [], []
    valid_pixels = 0
    for record in dev:
        feature = _load_teacher(root, args.teacher_kind, int(record["frame_id"]))
        target = F.normalize(
            feature.permute(1, 2, 0).reshape(-1, channels), dim=-1, eps=1e-8
        ).to(device)
        shard = torch.load(Path(record["responsibility_view"]), map_location="cpu")
        gaussian_ids = torch.as_tensor(shard["gaussian_ids"]).long()
        pixel_ids = torch.as_tensor(shard["pixel_ids"]).long()
        weights = torch.as_tensor(shard["base_weights"]).float()
        rendered = torch.zeros(height * width, channels, device=device)
        alpha = torch.zeros(height * width, device=device)
        for start in range(0, gaussian_ids.numel(), args.hit_chunk):
            stop = min(start + args.hit_chunk, gaussian_ids.numel())
            ids = gaussian_ids[start:stop]
            target_pixels = pixel_ids[start:stop].to(device)
            weight = weights[start:stop].to(device)
            source = memory[ids].to(device).float()
            rendered.index_add_(0, target_pixels, source * weight[:, None])
            alpha.index_add_(0, target_pixels, weight)
        valid = alpha >= args.alpha_threshold
        cosine_values.append(F.cosine_similarity(rendered[valid], target[valid], dim=-1))
        retrieval_values.append(_same_pixel_retrieval(
            rendered[valid], target[valid], args.retrieval_samples_per_view
        ))
        native_values.append(_same_pixel_retrieval(
            target[valid], target[valid], args.retrieval_samples_per_view
        ))
        valid_pixels += int(valid.sum())

    def mean_retrieval(values):
        return {
            "top1": sum(value[0] for value in values) / len(values),
            "top5": sum(value[1] for value in values) / len(values),
            "positive_margin": sum(value[2] for value in values) / len(values),
        }

    report = {
        "schema": "radio_gs.sugm_v3.uncompressed_exact_mpr_ceiling.v1",
        "teacher": args.teacher_kind,
        "split": "source_train_residues_1_2_to_dev_residue_3",
        "feature_dim": channels,
        "observed_gaussian_rows": int(observed.sum()),
        "valid_dev_pixels": valid_pixels,
        "render_cosine": float(torch.cat(cosine_values).mean()),
        "same_view_same_pixel_retrieval": mean_retrieval(retrieval_values),
        "native_self_ceiling": mean_retrieval(native_values),
        "aggregation": (
            "one_step_cosine_huber_irls_exact_mpr"
            if args.robust_cosine_delta > 0 else
            "per_pixel_unit_direction_exact_mpr_weighted_mean_then_row_normalize"
        ),
        "robust_observation_weight": (
            {
                "cosine_residual_delta": args.robust_cosine_delta,
                "minimum_weight_factor": args.robust_weight_floor,
            }
            if args.robust_cosine_delta > 0 else None
        ),
        "historical_field_opened": False,
        "target_rgb_opened": False,
        "benchmark_metrics_opened": False,
        "inputs": {
            "membership": {"path": str(membership_path), "sha256": sha256_file(membership_path)},
            "teacher_root": str(root),
            "train_frames": [int(record["frame_id"]) for record in train],
            "dev_frames": [int(record["frame_id"]) for record in dev],
        },
    }
    write_frozen_json(Path(args.output).resolve(), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--membership", required=True)
    parser.add_argument("--teacher-root", required=True)
    parser.add_argument("--teacher-kind", choices=("radio", "dinov2"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hit-chunk", type=int, default=8192)
    parser.add_argument("--alpha-threshold", type=float, default=0.02)
    parser.add_argument("--retrieval-samples-per-view", type=int, default=512)
    parser.add_argument("--robust-cosine-delta", type=float, default=0.0)
    parser.add_argument("--robust-weight-floor", type=float, default=0.05)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.robust_cosine_delta < 0 or not 0 < args.robust_weight_floor <= 1:
        raise ValueError("robust observation parameters are invalid")
    print(run(args))


if __name__ == "__main__":
    main()
