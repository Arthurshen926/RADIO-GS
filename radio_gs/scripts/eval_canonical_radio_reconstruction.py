#!/usr/bin/env python3
"""Evaluate held-out 2-D reconstruction of a canonical primitive RADIO field."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from radio_gs.config import load_config
from radio_gs.field import (
    load_canonical_field_checkpoint,
    load_view_residual_checkpoint,
)
from radio_gs.rendering.coefficient_renderer import (
    render_canonical_radio,
    render_view_conditioned_radio,
)
from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline
from radio_gs.training.feature_training_utils import SimpleRadioDataset


def _sha256_tensor_rows(values: torch.Tensor) -> str:
    array = values.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_frame_ids(raw: str) -> list[int]:
    value = str(raw or "").strip()
    if not value:
        return []
    path = Path(value)
    if path.is_file():
        value = path.read_text(encoding="utf-8")
    tokens: list[str] = []
    for line in value.splitlines():
        tokens.extend(line.split("#", 1)[0].replace(",", " ").split())
    return sorted({int(token) for token in tokens})


def evaluate(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    config = load_config(args.config)
    model, _codec, renderer, _sharpener, _refiner, _cfg, _hybrid = load_render_pipeline(
        args.config,
        args.geometry_checkpoint,
        device,
        strict_checkpoint_contract=True,
        load_ply_rgb_features=False,
    )
    field, field_payload = load_canonical_field_checkpoint(
        args.field_checkpoint, map_location="cpu"
    )
    field = field.to(device).eval()
    xyz_hash = _sha256_tensor_rows(model.get_xyz())
    expected_hash = str(
        field_payload.get("geometry_fingerprint", {}).get("xyz_sha256", "")
    )
    if not expected_hash or xyz_hash != expected_hash:
        raise ValueError("canonical field/geometry row fingerprint mismatch")

    view_residual = None
    view_residual_payload = None
    if str(args.view_residual_checkpoint).strip():
        view_residual, view_residual_payload = load_view_residual_checkpoint(
            args.view_residual_checkpoint, map_location="cpu"
        )
        residual_fingerprint = dict(
            view_residual_payload.get("geometry_fingerprint", {})
        )
        if str(residual_fingerprint.get("xyz_sha256", "")) != xyz_hash:
            raise ValueError("view residual/geometry row fingerprint mismatch")
        if int(view_residual.num_gaussians) != int(model.get_xyz().shape[0]):
            raise ValueError("view residual row count mismatch")
        if int(view_residual.coefficient_dim) != int(field.decoder.coefficient_dim):
            raise ValueError("view residual coefficient dimension mismatch")
        expected_field_hash = str(view_residual_payload.get("base_field_sha256", ""))
        if not expected_field_hash or expected_field_hash != _sha256_file(args.field_checkpoint):
            raise ValueError("view residual was trained over a different canonical field")
        if not bool(
            view_residual_payload.get("invariants", {}).get(
                "residual_used_only_for_view_rendering", False
            )
        ):
            raise ValueError("view residual checkpoint lacks rendering-only invariant")
        view_residual = view_residual.to(device).eval()

    feature_dir = Path(str(getattr(config, "feature_dir", "")))
    raw_pose_file = str(getattr(config, "pose_file", "") or "").strip()
    pose_file = raw_pose_file if raw_pose_file and Path(raw_pose_file).is_file() else None
    raw_pose_dir = str(getattr(config, "pose_dir", "") or "").strip()
    if raw_pose_dir and Path(raw_pose_dir).is_dir():
        pose_dir = raw_pose_dir
    else:
        fallback = feature_dir / "poses_w2c"
        pose_dir = str(fallback) if fallback.is_dir() else None
    feature_height = int(getattr(config, "feature_height", renderer.image_height))
    feature_width = int(getattr(config, "feature_width", renderer.image_width))
    dataset = SimpleRadioDataset(
        feature_dir=str(feature_dir),
        pose_file=pose_file,
        pose_dir=pose_dir,
        feature_size=(feature_height, feature_width),
        split="validation",
        dataset_type=str(getattr(config, "dataset_type", "lerf")),
    )
    training_frames = set(
        int(value)
        for value in field_payload.get("mpr_cache_metadata", {}).get(
            "selected_frame_indices", []
        )
    )
    mpr_excluded_frames = set(
        int(value)
        for value in field_payload.get("mpr_cache_metadata", {}).get(
            "excluded_frame_ids", []
        )
    )
    render_metadata = dict(field_payload.get("render_optimization", {}))
    explicit_frames = _parse_frame_ids(args.frame_ids)
    if explicit_frames:
        requested_frames = explicit_frames
        frame_policy = "explicit"
    elif args.frame_policy == "render_validation":
        requested_frames = [int(value) for value in render_metadata.get("validation_frames", [])]
        frame_policy = "render_validation"
    elif args.frame_policy == "benchmark":
        requested_frames = [
            int(value) for value in render_metadata.get("excluded_benchmark_frames", [])
        ]
        if not requested_frames:
            requested_frames = sorted(mpr_excluded_frames)
        frame_policy = "benchmark"
    else:
        requested_frames = []
        frame_policy = "mpr_heldout"
    if requested_frames:
        requested = set(requested_frames)
        candidates = [
            index
            for index, frame in enumerate(dataset.frame_indices)
            if int(frame) in requested
        ]
        missing = requested - {int(dataset.frame_indices[index]) for index in candidates}
        if missing:
            raise ValueError(f"requested reconstruction frames are unavailable: {sorted(missing)}")
    else:
        candidates = [
            index
            for index, frame in enumerate(dataset.frame_indices)
            if int(frame) not in training_frames and int(frame) not in mpr_excluded_frames
        ]
    if not candidates:
        raise RuntimeError(f"no frame is available for policy {frame_policy}")
    if args.max_views > 0 and len(candidates) > args.max_views:
        positions = torch.linspace(0, len(candidates) - 1, args.max_views).round().long()
        candidates = [candidates[int(index)] for index in positions]

    per_frame = []
    with torch.inference_mode():
        for index in candidates:
            sample = dataset[index]
            if view_residual is None:
                rendered = render_canonical_radio(
                    renderer,
                    model,
                    field,
                    sample["pose_w2c"].to(device),
                    feature_height=feature_height,
                    feature_width=feature_width,
                    use_reliability=bool(args.reliability_splat),
                )
            else:
                if bool(args.reliability_splat):
                    raise ValueError(
                        "reliability splatting is not part of the frozen view-residual path"
                    )
                rendered = render_view_conditioned_radio(
                    renderer,
                    model,
                    field,
                    view_residual,
                    sample["pose_w2c"].to(device),
                    feature_height=feature_height,
                    feature_width=feature_width,
                )
            predicted = rendered["feature_map"].permute(1, 2, 0).float()
            teacher = sample["radio_features"].to(device).permute(1, 2, 0).float()
            valid = rendered["alpha_map"] >= float(args.alpha_threshold)
            cosine = F.cosine_similarity(predicted[valid], teacher[valid], dim=-1, eps=1e-8)
            rmse = (predicted[valid] - teacher[valid]).square().mean(dim=-1).sqrt()
            per_frame.append(
                {
                    "frame_id": int(dataset.frame_indices[index]),
                    "valid_pixels": int(valid.sum()),
                    "mean_cosine": float(cosine.mean()) if cosine.numel() else 0.0,
                    "mean_rmse": float(rmse.mean()) if rmse.numel() else 0.0,
                }
            )
    total_pixels = sum(item["valid_pixels"] for item in per_frame)
    if total_pixels <= 0:
        raise RuntimeError("canonical field rendered no valid held-out pixels")
    report = {
        "schema_version": 1,
        "field_checkpoint": str(Path(args.field_checkpoint).resolve()),
        "view_residual_checkpoint": (
            str(Path(args.view_residual_checkpoint).resolve())
            if view_residual is not None
            else ""
        ),
        "view_conditioned_rendering": view_residual is not None,
        "primitive_query_remains_canonical": view_residual is not None,
        "geometry_checkpoint": str(Path(args.geometry_checkpoint).resolve()),
        "held_out_from_mpr": True,
        "frame_policy": frame_policy,
        "benchmark_frames_excluded": sorted(
            int(value)
            for value in render_metadata.get(
                "excluded_benchmark_frames", sorted(mpr_excluded_frames)
            )
        ),
        "num_views": len(per_frame),
        "valid_pixels": total_pixels,
        "mean_cosine": sum(
            item["mean_cosine"] * item["valid_pixels"] for item in per_frame
        )
        / total_pixels,
        "mean_rmse": sum(
            item["mean_rmse"] * item["valid_pixels"] for item in per_frame
        )
        / total_pixels,
        "alpha_threshold": float(args.alpha_threshold),
        "reliability_splat": bool(args.reliability_splat),
        "per_frame": per_frame,
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
    parser.add_argument("--view-residual-checkpoint", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-views", type=int, default=16)
    parser.add_argument(
        "--frame-policy",
        choices=["render_validation", "benchmark", "mpr_heldout"],
        default="render_validation",
    )
    parser.add_argument("--frame-ids", default="")
    parser.add_argument("--alpha-threshold", type=float, default=0.02)
    parser.add_argument(
        "--reliability-splat",
        action="store_true",
        help="Optional ablation: multiply fixed geometry opacity by MPR reliability.",
    )
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
