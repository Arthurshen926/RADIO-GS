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
    load_factorized_canonical_field_checkpoint,
    load_view_residual_checkpoint,
)
from radio_gs.data.benchmark_paths import load_w2c_from_pose_dir
from radio_gs.interfaces.semantic_alignment import (
    GlobalRegionSummaryBridge,
    project_dense_region_semantics,
)
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.rendering.coefficient_renderer import (
    render_canonical_radio,
    render_view_conditioned_radio,
)
from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline


def _sha256_tensor_rows(values: torch.Tensor) -> str:
    array = values.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _benchmark_frames_from_payload(payload: dict) -> list[int]:
    render_optimization = payload.get("render_optimization", {})
    if isinstance(render_optimization, dict):
        direct = render_optimization.get("excluded_benchmark_frames")
        if direct:
            return sorted({int(frame) for frame in direct})
    mpr_metadata = payload.get("mpr_cache_metadata", {})
    if not isinstance(mpr_metadata, dict):
        return []
    direct = mpr_metadata.get("excluded_frame_ids")
    registration = mpr_metadata.get("registration_responsibility_contract", {})
    nested = (
        registration.get("excluded_frame_ids")
        if isinstance(registration, dict)
        else None
    )
    declared = direct if direct is not None else nested
    return sorted({int(frame) for frame in (declared or [])})


def _parse_region_kernel_sizes(raw: str) -> tuple[int, ...]:
    values = tuple(int(value) for value in str(raw).split(",") if value.strip())
    if not values or any(value <= 0 or value % 2 == 0 for value in values):
        raise ValueError("region kernel sizes must be positive odd integers")
    return values


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
    field, payload, _factorized_signature = (
        load_factorized_canonical_field_checkpoint(
            args.field_checkpoint, map_location="cpu"
        )
    )
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

    region_bridge = None
    region_summary_head = None
    region_kernel_sizes: tuple[int, ...] = ()
    if args.readout == "region_summary":
        if view_residual is not None:
            raise ValueError(
                "region-summary readout is defined only on the canonical field"
            )
        if not str(args.semantic_bridge_checkpoint).strip():
            raise ValueError(
                "region-summary readout requires --semantic-bridge-checkpoint"
            )
        region_bridge, _bridge_manifest = GlobalRegionSummaryBridge.from_checkpoint(
            args.semantic_bridge_checkpoint, map_location="cpu"
        )
        region_bridge = region_bridge.to(device).eval()
        region_summary_head = SigLIP2SummaryHead.from_radio_checkpoint(
            args.radio_checkpoint
        ).to(device).eval()
        region_kernel_sizes = _parse_region_kernel_sizes(
            args.semantic_kernel_sizes
        )

    metadata = dict(payload.get("render_optimization", {}))
    if args.frame_policy == "benchmark":
        frames = _benchmark_frames_from_payload(dict(payload))
    else:
        frames = [int(value) for value in metadata.get("validation_frames", [])]
    if not frames:
        raise RuntimeError(f"field checkpoint has no {args.frame_policy} frame manifest")
    feature_dir = Path(str(getattr(config, "feature_dir", "")))
    declared_pose_dir = str(getattr(config, "pose_dir", "") or "").strip()
    pose_dir = Path(declared_pose_dir) if declared_pose_dir else feature_dir / "poses_w2c"
    if not pose_dir.is_dir():
        raise FileNotFoundError(f"registered pose directory is missing: {pose_dir}")
    poses_w2c = load_w2c_from_pose_dir(pose_dir, frames)
    output_scene = Path(args.output_root) / str(args.scene)
    output_dir = output_scene / "backbone"
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_frames: list[int] = []
    with torch.inference_mode():
        for frame, pose_w2c in zip(frames, poses_w2c):
            pose_tensor = torch.as_tensor(pose_w2c).to(device)
            if view_residual is None:
                result = render_canonical_radio(
                    renderer,
                    model,
                    field,
                    pose_tensor,
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
                    pose_tensor,
                    feature_height=int(config.feature_height),
                    feature_width=int(config.feature_width),
                )
            feature_map = result["feature_map"]
            if region_bridge is not None:
                feature_map = project_dense_region_semantics(
                    region_bridge,
                    region_summary_head,
                    feature_map[None],
                    kernel_sizes=region_kernel_sizes,
                    projection_batch_size=int(args.semantic_projection_batch_size),
                )[0]
            torch.save(feature_map.half().cpu(), output_dir / f"rgb_{frame}.pt")
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
        "pose_dir": str(pose_dir.resolve()),
        "readout": args.readout,
        "feature_dim": (
            1536 if region_bridge is not None else int(field.decoder.feature_dim)
        ),
        "feature_grid": [int(config.feature_height), int(config.feature_width)],
        "semantic_bridge_checkpoint": (
            str(Path(args.semantic_bridge_checkpoint).resolve())
            if region_bridge is not None
            else ""
        ),
        "semantic_kernel_sizes": list(region_kernel_sizes),
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
    parser.add_argument(
        "--readout",
        choices=["raw_radio", "region_summary"],
        default="raw_radio",
    )
    parser.add_argument("--semantic-bridge-checkpoint", default="")
    parser.add_argument(
        "--radio-checkpoint",
        default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar",
    )
    parser.add_argument("--semantic-kernel-sizes", default="3,7,15")
    parser.add_argument("--semantic-projection-batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(render(args), indent=2))


if __name__ == "__main__":
    main()
