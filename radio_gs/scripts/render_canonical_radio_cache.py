#!/usr/bin/env python3
"""Render query-free canonical RADIO maps into the standard teacher-cache layout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

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
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    config = load_config(args.config)
    model, _codec, renderer, _sharpener, _refiner, _cfg, _hybrid = load_render_pipeline(
        args.config,
        args.geometry_checkpoint,
        device,
        strict_checkpoint_contract=True,
        load_ply_rgb_features=False,
    )
    field, payload = load_canonical_field_checkpoint(args.field_checkpoint, map_location="cpu")
    xyz_hash = _sha256_tensor_rows(model.get_xyz())
    if xyz_hash != str(
        payload.get("geometry_fingerprint", {}).get("xyz_sha256", "")
    ):
        raise ValueError("canonical field and geometry rows differ")
    field = field.to(device).eval()
    view_residual = None
    if str(args.view_residual_checkpoint).strip():
        view_residual, residual_payload = load_view_residual_checkpoint(
            args.view_residual_checkpoint, map_location="cpu"
        )
        residual_fingerprint = dict(residual_payload.get("geometry_fingerprint", {}))
        if str(residual_fingerprint.get("xyz_sha256", "")) != xyz_hash:
            raise ValueError("view residual and geometry rows differ")
        if int(view_residual.num_gaussians) != int(model.get_xyz().shape[0]):
            raise ValueError("view residual row count differs from geometry")
        if int(view_residual.coefficient_dim) != int(field.decoder.coefficient_dim):
            raise ValueError("view residual coefficient dimension differs from field")
        if str(residual_payload.get("base_field_sha256", "")) != _sha256_file(
            args.field_checkpoint
        ):
            raise ValueError("view residual was trained over a different field")
        if not bool(
            residual_payload.get("invariants", {}).get(
                "residual_used_only_for_view_rendering", False
            )
        ):
            raise ValueError("view residual lacks the rendering-only invariant")
        view_residual = view_residual.to(device).eval()

    feature_dir = Path(str(getattr(config, "feature_dir", "")))
    fallback = feature_dir / "poses_w2c"
    dataset = SimpleRadioDataset(
        feature_dir=str(feature_dir),
        pose_file=None,
        pose_dir=str(fallback),
        feature_size=(int(config.feature_height), int(config.feature_width)),
        split="validation",
        dataset_type=str(config.dataset_type),
    )
    metadata = dict(payload.get("render_optimization", {}))
    if args.frame_policy == "benchmark":
        frames = [int(value) for value in metadata.get("excluded_benchmark_frames", [])]
    else:
        frames = [int(value) for value in metadata.get("validation_frames", [])]
    if not frames:
        raise RuntimeError(f"field checkpoint has no {args.frame_policy} frame manifest")
    frame_to_index = {int(frame): index for index, frame in enumerate(dataset.frame_indices)}
    output_scene = Path(args.output_root) / str(args.scene)
    output_dir = output_scene / "backbone"
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_frames: list[int] = []
    with torch.inference_mode():
        for frame in frames:
            if frame not in frame_to_index:
                raise ValueError(f"frame {frame} is unavailable in the RADIO cache")
            sample = dataset[frame_to_index[frame]]
            if view_residual is None:
                result = render_canonical_radio(
                    renderer,
                    model,
                    field,
                    sample["pose_w2c"].to(device),
                    feature_height=int(config.feature_height),
                    feature_width=int(config.feature_width),
                    use_reliability=False,
                )
            else:
                result = render_view_conditioned_radio(
                    renderer,
                    model,
                    field,
                    view_residual,
                    sample["pose_w2c"].to(device),
                    feature_height=int(config.feature_height),
                    feature_width=int(config.feature_width),
                )
            torch.save(result["feature_map"].half().cpu(), output_dir / f"rgb_{frame}.pt")
            rendered_frames.append(frame)
    report = {
        "schema_version": 1,
        "scene": str(args.scene),
        "field_checkpoint": str(Path(args.field_checkpoint).resolve()),
        "view_residual_checkpoint": (
            str(Path(args.view_residual_checkpoint).resolve())
            if view_residual is not None
            else ""
        ),
        "view_conditioned_rendering": view_residual is not None,
        "primitive_query_remains_canonical": view_residual is not None,
        "geometry_checkpoint": str(Path(args.geometry_checkpoint).resolve()),
        "frame_policy": args.frame_policy,
        "frames": rendered_frames,
        "feature_dim": int(field.decoder.feature_dim),
        "feature_grid": [int(config.feature_height), int(config.feature_width)],
        "normalized_splat": True,
        "affine_decode_after_splat": True,
        "reliability_splat": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }
    (output_scene / "canonical_render_manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--geometry-checkpoint", required=True)
    parser.add_argument("--field-checkpoint", required=True)
    parser.add_argument("--view-residual-checkpoint", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument(
        "--frame-policy", choices=["benchmark", "render_validation"], default="benchmark"
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(render(args), indent=2))


if __name__ == "__main__":
    main()
