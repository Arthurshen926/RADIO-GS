#!/usr/bin/env python3
"""Render a native LUDVIG scalar and exact-adjoint it to the current carrier."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from radio_gs.config import load_config
from radio_gs.interfaces.streaming_prompt_adjoint import (
    streaming_prompt_adjoint,
    streaming_prompt_cache_metadata,
)
from radio_gs.interfaces.prompt_responsibility_cache import tensor_sha256
from radio_gs.models.explicit_gaussian import ExplicitFeatureGaussian
from radio_gs.rendering.feature_renderer import FeatureFieldRenderer
from radio_gs.scripts.render_promptable_nvs_features import resolve_protocol_views


ARTIFACT_TYPE = "nvos_ludvig_carrier_native_exact_adjoint_bridge_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_bound(path: str | Path, expected: str, label: str) -> Path:
    source = Path(path).expanduser().resolve(strict=True)
    if len(str(expected)) != 64 or _sha256(source) != str(expected):
        raise ValueError(f"{label} SHA-256 differs")
    return source


def _write_numpy(path: Path, value: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.asarray(value, dtype=np.float32), allow_pickle=False)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(path)


def _write_torch(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(dict(value), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(
            json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@torch.inference_mode()
def materialize(args: argparse.Namespace) -> Mapping[str, Any]:
    native_ply = _require_bound(args.native_ply, args.native_ply_sha256, "native PLY")
    native_scalar = _require_bound(
        args.native_scalar, args.native_scalar_sha256, "native scalar"
    )
    native_ply_load = native_ply
    native_scalar_load = native_scalar
    if args.native_ply_local_copy:
        native_ply_load = _require_bound(
            args.native_ply_local_copy, args.native_ply_sha256, "local native PLY copy"
        )
    if args.native_scalar_local_copy:
        native_scalar_load = _require_bound(
            args.native_scalar_local_copy,
            args.native_scalar_sha256,
            "local native scalar copy",
        )
    dataset_path = _require_bound(
        args.dataset_manifest, args.dataset_manifest_sha256, "dataset manifest"
    )
    config_path = _require_bound(args.current_config, args.current_config_sha256, "config")
    camera_map_path = _require_bound(
        args.camera_mapping, args.camera_mapping_sha256, "camera mapping"
    )
    responsibility = _require_bound(
        args.current_responsibility,
        args.current_responsibility_sha256,
        "current responsibility",
    )
    metadata = streaming_prompt_cache_metadata(responsibility)
    authority = metadata.get("authority")
    if not isinstance(authority, Mapping):
        raise ValueError("responsibility authority is absent")
    if (
        authority.get("scene_id") != args.scene_id
        or authority.get("frame_id") != args.reference_frame
        or authority.get("target_rgb_opened") is not False
        or authority.get("target_mask_opened") is not False
    ):
        raise ValueError("responsibility scene/frame/information authority differs")
    height, width = int(authority["height"]), int(authority["width"])
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    camera_mapping = json.loads(camera_map_path.read_text(encoding="utf-8"))
    config = load_config(str(config_path))
    views = resolve_protocol_views(
        dataset,
        scene_id=args.scene_id,
        scene_root=Path(str(config.scene_root)).resolve(),
        camera_mapping=camera_mapping,
    )
    selected = [view for view in views if str(view["frame_id"]) == args.reference_frame]
    if len(selected) != 1 or selected[0].get("role") != "prompt":
        raise ValueError("registered reference view authority differs")
    pose_array = np.asarray(selected[0]["w2c"], dtype="<f4")
    pose_cpu = torch.from_numpy(pose_array.copy()).contiguous()
    pose_sha = tensor_sha256(pose_cpu)
    if pose_sha != authority.get("pose_sha256"):
        raise ValueError("registered reference pose differs from exact W authority")

    scalar = np.load(native_scalar_load, allow_pickle=False)
    if scalar.ndim != 2 or scalar.shape[1] != 1 or scalar.dtype != np.float32:
        raise ValueError("native LUDVIG scalar must be float32 [N,1]")
    tolerance = 8 * np.finfo(np.float32).eps
    if not np.isfinite(scalar).all() or scalar.min() < -tolerance or scalar.max() > 1 + tolerance:
        raise ValueError("native LUDVIG scalar leaves [0,1]")
    scalar = np.clip(scalar, 0.0, 1.0)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("native scalar rendering requires CUDA")
    model = ExplicitFeatureGaussian(latent_dim=1)
    model.load_from_ply(str(native_ply_load))
    if model.num_gaussians != scalar.shape[0]:
        raise ValueError("native scalar and native PLY rows differ")
    model = model.to(device).eval()
    # Reproduce the renderer construction used by the exact-W exporter.  Its
    # canonical K lives at the configured feature raster, then
    # ``scaled_intrinsics`` maps it to the requested responsibility raster.
    # Constructing K directly at RGB resolution is geometrically equivalent,
    # but differs by float32 rounding and therefore must fail the hash bind.
    feature_height = int(getattr(config, "feature_height", 30))
    feature_width = int(getattr(config, "feature_width", 40))
    renderer = FeatureFieldRenderer(
        image_height=feature_height,
        image_width=feature_width,
        fx=float(config.fx) * feature_width / int(config.image_width),
        fy=float(config.fy) * feature_height / int(config.image_height),
        cx=float(config.cx) * feature_width / int(config.image_width),
        cy=float(config.cy) * feature_height / int(config.image_height),
        max_channels_per_chunk=1,
    ).to(device)
    intrinsics = renderer.scaled_intrinsics(width, height).detach().float().cpu().contiguous()
    intrinsics_sha = tensor_sha256(intrinsics)
    if intrinsics_sha != authority.get("intrinsics_sha256"):
        raise ValueError("render intrinsics differ from exact W authority")
    rendered = renderer.render_feature_rows(
        model,
        pose_cpu.to(device),
        torch.from_numpy(scalar).to(device),
        feature_height=height,
        feature_width=width,
        alpha_normalize=True,
    )
    pixel_probability = rendered["feature_map"][0].float().clamp(0, 1).cpu()
    alpha = rendered["alpha_map"].float().cpu()
    if tuple(pixel_probability.shape) != (height, width) or tuple(alpha.shape) != (
        height, width
    ):
        raise ValueError("native scalar render dimensions differ from exact W")
    alpha_tolerance = 8 * torch.finfo(torch.float32).eps
    if (
        not bool(torch.isfinite(alpha).all())
        or bool((alpha < -alpha_tolerance).any())
        or bool((alpha > 1.0 + alpha_tolerance).any())
        or not bool((alpha > 0).any())
    ):
        raise ValueError("native scalar render alpha is invalid")
    output = Path(args.output_dir).resolve()
    render_path = output / "native_scalar_reference_probability.npy"
    render_sha = _write_numpy(render_path, pixel_probability.numpy())
    alpha_supported_fraction = float((alpha > 0).float().mean())
    alpha_min, alpha_max = float(alpha.min()), float(alpha.max())
    del rendered, alpha, renderer, model
    gc.collect()
    torch.cuda.empty_cache()

    adjoint, cache_payload = streaming_prompt_adjoint(
        responsibility,
        pixel_probability,
        expected_file_sha256=args.current_responsibility_sha256,
        chunk_hits=int(args.chunk_hits),
    )
    state_path = output / "current_carrier_primitive_probability.pt"
    state_sha = _write_torch(
        state_path,
        {
            "artifact_type": ARTIFACT_TYPE,
            "scene_id": args.scene_id,
            "reference_frame": args.reference_frame,
            "primitive_probability": adjoint.primitive_probability.float(),
            "visible_mass": adjoint.visible_mass,
            "geometry_xyz_sha256": authority["geometry_xyz_sha256"],
            "responsibility_authority_sha256": cache_payload["authority_sha256"],
            "native_scalar_sha256": args.native_scalar_sha256,
            "native_ply_sha256": args.native_ply_sha256,
            "target_mask_opened": False,
            "target_metric_opened": False,
            "nearest_neighbor_transfer": False,
        },
    )
    visible = adjoint.visible_mass > 0
    receipt = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "scene_id": args.scene_id,
        "reference_frame": args.reference_frame,
        "inputs": {
            "native_ply": {"path": str(native_ply), "sha256": args.native_ply_sha256},
            "native_scalar": {"path": str(native_scalar), "sha256": args.native_scalar_sha256},
            "current_responsibility": {"path": str(responsibility), "sha256": args.current_responsibility_sha256},
            "dataset_manifest": {"path": str(dataset_path), "sha256": args.dataset_manifest_sha256},
            "current_config": {"path": str(config_path), "sha256": args.current_config_sha256},
            "camera_mapping": {"path": str(camera_map_path), "sha256": args.camera_mapping_sha256},
        },
        "render": {
            "height": height,
            "width": width,
            "alpha_normalized": True,
            "probability_min": float(pixel_probability.min()),
            "probability_max": float(pixel_probability.max()),
            "probability_mean": float(pixel_probability.mean()),
            "pose_sha256": pose_sha,
            "intrinsics_sha256": intrinsics_sha,
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
            "alpha_supported_fraction": alpha_supported_fraction,
        },
        "exact_adjoint": {
            "hit_count": adjoint.hit_count,
            "chunk_hits": adjoint.chunk_hits,
            "visible_gaussians": int(visible.sum()),
            "visible_mass_max_abs_error": adjoint.visible_mass_max_abs_error,
            "constant_conservation_max_abs_error": adjoint.constant_conservation_max_abs_error,
            "operator": "normalized_exact_W_transpose",
        },
        "outputs": {
            "reference_probability": {"path": str(render_path), "sha256": render_sha},
            "primitive_state": {"path": str(state_path), "sha256": state_sha},
        },
        "safety": {
            "target_mask_opened": False,
            "target_metric_opened": False,
            "nearest_neighbor_transfer": False,
            "carrier_modified": False,
            "second_persistent_scene_field_created": False,
        },
    }
    receipt_path = output / "receipt.json"
    _write_json(receipt_path, receipt)
    return {**receipt, "receipt": str(receipt_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--reference-frame", required=True)
    parser.add_argument("--native-ply", required=True)
    parser.add_argument("--native-ply-sha256", required=True)
    parser.add_argument("--native-ply-local-copy")
    parser.add_argument("--native-scalar", required=True)
    parser.add_argument("--native-scalar-sha256", required=True)
    parser.add_argument("--native-scalar-local-copy")
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--current-config", required=True)
    parser.add_argument("--current-config-sha256", required=True)
    parser.add_argument("--camera-mapping", required=True)
    parser.add_argument("--camera-mapping-sha256", required=True)
    parser.add_argument("--current-responsibility", required=True)
    parser.add_argument("--current-responsibility-sha256", required=True)
    parser.add_argument("--chunk-hits", type=int, default=1_000_000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    report = materialize(parser.parse_args(argv))
    print(json.dumps({"receipt": report["receipt"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
