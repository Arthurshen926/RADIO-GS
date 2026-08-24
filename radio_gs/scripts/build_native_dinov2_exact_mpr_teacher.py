#!/usr/bin/env python3
"""Build a source-only native DINOv2 teacher on an exact-MPR Gaussian domain.

This is deliberately independent of C-RADIO and its adaptor heads.  Frozen
official DINOv2 patch tokens are extracted from registered source RGB, aligned
to the immutable exact-marginal grid, and accumulated into disjoint train and
held-out source-view sufficient statistics.  The held-out split is required
so a downstream shared-latent pilot cannot pass by merely memorizing its
training-view primitive targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from tqdm import tqdm

from radio_gs.rendering.sparse_marginal_authority import (
    load_sparse_exact_marginal_authority,
)
from radio_gs.scripts.build_gaussian_multiview_teacher_cache import (
    accumulate_contribution_mean_channel_chunked,
    finalize_registered_mean_chunked,
)
from radio_gs.utils.immutable_artifacts import sha256_file


SCHEMA = "radio_gs.native_dinov2_exact_mpr_teacher.v1"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be one regular JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain one JSON object")
    return dict(value)


def _atomic_torch_save(value: object, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        if output.exists():
            raise FileExistsError(f"refusing to overwrite immutable output: {output}")
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _source_rgb_by_frame(
    manifest: Mapping[str, Any], exact_frames: list[int]
) -> dict[int, tuple[Path, str]]:
    image_dir = Path(str(manifest.get("image_dir", ""))).expanduser().resolve()
    if not image_dir.is_dir():
        raise ValueError(f"source image directory is absent: {image_dir}")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("source frame manifest has no frames")
    excluded_names = {str(value) for value in manifest.get("excluded_image_names", [])}
    excluded_stems = {str(value) for value in manifest.get("excluded_image_stems", [])}
    indexed: dict[int, tuple[Path, str]] = {}
    for raw in frames:
        if not isinstance(raw, Mapping):
            raise ValueError("source frame record is not an object")
        frame = int(raw.get("frame_idx", -1))
        relative = str(raw.get("source_file", ""))
        digest = str(raw.get("source_sha256", ""))
        path = (image_dir / relative).resolve()
        if (
            frame < 0
            or frame in indexed
            or not relative
            or relative in excluded_names
            or Path(relative).stem in excluded_stems
            or path.parent != image_dir
            or not path.is_file()
            or path.is_symlink()
            or len(digest) != 64
        ):
            raise ValueError(f"invalid source frame identity: {frame}")
        indexed[frame] = (path, digest)
    missing = sorted(set(exact_frames) - set(indexed))
    if missing:
        raise ValueError(f"exact-MPR frames are absent from source manifest: {missing}")
    return {frame: indexed[frame] for frame in exact_frames}


def _load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    repository = Path(args.torchhub_repo).expanduser().resolve(strict=True)
    checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    model = torch.hub.load(
        str(repository), str(args.model_name), source="local", pretrained=False
    )
    state = torch.load(checkpoint, map_location="cpu")
    if not isinstance(state, Mapping):
        raise ValueError("native DINOv2 checkpoint is not a state dictionary")
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model


def _image_tensor(path: Path, *, height: int, width: int) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.float32) / np.float32(255.0)
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)[None]
    tensor = F.interpolate(
        tensor,
        size=(int(height), int(width)),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )[0]
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32)[:, None, None]
    return (tensor - mean) / std


def _extract_feature_map(
    model: torch.nn.Module,
    image_path: Path,
    *,
    device: torch.device,
    grid_height: int,
    grid_width: int,
    patch_size: int,
) -> torch.Tensor:
    image = _image_tensor(
        image_path,
        height=int(grid_height) * int(patch_size),
        width=int(grid_width) * int(patch_size),
    ).to(device)
    with torch.inference_mode():
        context = (
            torch.cuda.amp.autocast(enabled=True, dtype=torch.float16)
            if device.type == "cuda"
            else torch.autocast(device_type="cpu", enabled=False)
        )
        with context:
            outputs = model.get_intermediate_layers(image[None], n=1, reshape=True)
    if len(outputs) != 1:
        raise ValueError("native DINOv2 returned an unexpected layer count")
    feature = outputs[0]
    if isinstance(feature, (tuple, list)):
        feature = feature[0]
    feature = torch.as_tensor(feature)
    if feature.ndim == 4 and feature.shape[0] == 1:
        feature = feature[0]
    if feature.ndim != 3 or tuple(feature.shape[1:]) != (
        int(grid_height),
        int(grid_width),
    ):
        raise ValueError(
            "native DINOv2 grid differs: "
            f"found {tuple(feature.shape)}, expected [C,{grid_height},{grid_width}]"
        )
    feature = F.normalize(feature.float(), dim=0, eps=1e-8)
    if not bool(torch.isfinite(feature).all()):
        raise ValueError("native DINOv2 feature map contains non-finite values")
    return feature


def _frame_cache_path(root: Path, frame: int) -> Path:
    return root / f"frame_{int(frame):05d}.pt"


def _load_or_extract_frame(
    *,
    model: torch.nn.Module,
    path: Path,
    source_sha256: str,
    frame: int,
    cache_root: Path,
    device: torch.device,
    height: int,
    width: int,
    patch_size: int,
    checkpoint_sha256: str,
) -> torch.Tensor:
    if sha256_file(path) != source_sha256:
        raise ValueError(f"source image SHA-256 differs for frame {frame}")
    cached = _frame_cache_path(cache_root, frame)
    if cached.is_file():
        value = torch.load(cached, map_location="cpu")
        if not isinstance(value, Mapping) or value.get("schema") != SCHEMA:
            raise ValueError(f"native DINOv2 frame cache differs: {cached}")
        feature = torch.as_tensor(value.get("feature"))
        if (
            value.get("frame_index") != int(frame)
            or value.get("source_sha256") != source_sha256
            or value.get("checkpoint_sha256") != checkpoint_sha256
            or feature.ndim != 3
            or tuple(feature.shape[1:]) != (height, width)
            or not bool(torch.isfinite(feature.float()).all())
        ):
            raise ValueError(f"native DINOv2 cached frame lineage differs: {cached}")
        return feature.to(device=device, dtype=torch.float32)
    feature = _extract_feature_map(
        model,
        path,
        device=device,
        grid_height=height,
        grid_width=width,
        patch_size=patch_size,
    )
    cached.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{cached.name}.", suffix=".tmp", dir=cached.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(
            {
                "schema": SCHEMA,
                "frame_index": int(frame),
                "source_path": str(path),
                "source_sha256": source_sha256,
                "checkpoint_sha256": checkpoint_sha256,
                "feature": feature.detach().cpu().half(),
            },
            temporary,
        )
        os.replace(temporary, cached)
    finally:
        temporary.unlink(missing_ok=True)
    return feature


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite native teacher: {output}")
    exact_path = Path(args.exact_mpr_authority).expanduser().resolve(strict=True)
    source_manifest_path = Path(args.source_frame_manifest).expanduser().resolve(strict=True)
    exact = _load_json(exact_path, label="exact-MPR authority")
    source_manifest = _load_json(source_manifest_path, label="source frame manifest")
    metadata = exact.get("metadata")
    frames = [int(value) for value in exact.get("frame_indices", [])]
    if (
        exact.get("schema")
        != "radio_gs.sparse_exact_marginal_responsibility_authority.v1"
        or exact.get("schema_version") != 1
        or not isinstance(metadata, Mapping)
        or not frames
        or bool(metadata.get("benchmark_images_opened", True))
        or bool(metadata.get("benchmark_masks_opened", True))
    ):
        raise ValueError("exact-MPR source-only authority contract differs")
    height = int(metadata.get("feature_height", -1))
    width = int(metadata.get("feature_width", -1))
    num_gaussians = int(exact.get("num_gaussians", -1))
    if min(height, width, num_gaussians) <= 0 or int(exact.get("num_pixels", -1)) != height * width:
        raise ValueError("exact-MPR dimensions differ")
    exact_sha256 = sha256_file(exact_path)
    assignments, verified_sha256, _ = load_sparse_exact_marginal_authority(
        exact_path,
        expected_metadata=metadata,
        expected_frame_indices=frames,
        num_gaussians=num_gaussians,
        num_pixels=height * width,
        expected_sha256=exact_sha256,
    )
    if verified_sha256 != exact_sha256 or len(assignments) != len(frames):
        raise ValueError("exact-MPR authority verification differs")
    source_rgb = _source_rgb_by_frame(source_manifest, frames)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    checkpoint_sha256 = sha256_file(checkpoint)
    model = _load_model(args, device)
    cache_root = Path(args.resume_root).expanduser().resolve()
    train_sum: torch.Tensor | None = None
    validation_sum: torch.Tensor | None = None
    train_mass = torch.zeros(num_gaussians, dtype=torch.float32)
    validation_mass = torch.zeros(num_gaussians, dtype=torch.float32)
    train_frames: list[int] = []
    validation_frames: list[int] = []
    staging_sum: torch.Tensor | None = None
    staging_mass = torch.empty(num_gaussians, dtype=torch.float32)
    feature_dim = -1
    for view_index, (frame, assignment) in enumerate(
        tqdm(list(zip(frames, assignments)), desc="native DINOv2 exact MPR")
    ):
        image_path, image_sha256 = source_rgb[frame]
        feature = _load_or_extract_frame(
            model=model,
            path=image_path,
            source_sha256=image_sha256,
            frame=frame,
            cache_root=cache_root,
            device=device,
            height=height,
            width=width,
            patch_size=int(args.patch_size),
            checkpoint_sha256=checkpoint_sha256,
        )
        if train_sum is None:
            feature_dim = int(feature.shape[0])
            train_sum = torch.zeros(num_gaussians, feature_dim, dtype=torch.float32)
            validation_sum = torch.zeros_like(train_sum)
            staging_sum = torch.empty(
                num_gaussians,
                min(feature_dim, int(args.channel_chunk_size)),
                dtype=torch.float32,
            )
        if int(feature.shape[0]) != feature_dim:
            raise ValueError("native DINOv2 feature dimension changed between frames")
        held_out = view_index % int(args.validation_stride) == int(args.validation_offset)
        destination_sum = validation_sum if held_out else train_sum
        destination_mass = validation_mass if held_out else train_mass
        assert destination_sum is not None and staging_sum is not None
        accumulate_contribution_mean_channel_chunked(
            feature,
            assignment["gaussian_ids"],
            assignment["pixel_ids"],
            assignment["marginal_weights"],
            destination_sum,
            destination_mass,
            channel_chunk_size=int(args.channel_chunk_size),
            cpu_sum_staging=staging_sum,
            cpu_count_staging=staging_mass,
        )
        (validation_frames if held_out else train_frames).append(frame)
        del feature
    if train_sum is None or validation_sum is None or feature_dim <= 0:
        raise ValueError("native DINOv2 extraction produced no features")
    train_feature, train_valid = finalize_registered_mean_chunked(
        train_sum, train_mass, row_chunk_size=int(args.row_chunk_size)
    )
    validation_feature, validation_valid = finalize_registered_mean_chunked(
        validation_sum, validation_mass, row_chunk_size=int(args.row_chunk_size)
    )
    # Direction, rather than amplitude, is the native DINO semantic authority.
    train_feature = F.normalize(train_feature.float(), dim=-1, eps=1e-8).half()
    validation_feature = F.normalize(
        validation_feature.float(), dim=-1, eps=1e-8
    ).half()
    overlap = train_valid & validation_valid
    if not bool(overlap.any()):
        raise ValueError("native DINOv2 train/held-out Gaussian overlap is empty")
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "scene": str(source_manifest.get("scene", "")),
        "features_train": train_feature,
        "features_validation": validation_feature,
        "valid_train": train_valid,
        "valid_validation": validation_valid,
        "mass_train": train_mass,
        "mass_validation": validation_mass,
        "metadata": {
            "construction": (
                "independent_native_dinov2_patch_tokens_then_disjoint_source_view_"
                "exact_front_to_back_marginal_mpr"
            ),
            "model_name": str(args.model_name),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "torchhub_repo": str(Path(args.torchhub_repo).expanduser().resolve()),
            "source_frame_manifest": str(source_manifest_path),
            "source_frame_manifest_sha256": sha256_file(source_manifest_path),
            "exact_mpr_authority": str(exact_path),
            "exact_mpr_authority_sha256": exact_sha256,
            "grid": [height, width],
            "patch_size": int(args.patch_size),
            "feature_dim": feature_dim,
            "num_gaussians": num_gaussians,
            "train_frames": train_frames,
            "validation_frames": validation_frames,
            "validation_stride": int(args.validation_stride),
            "validation_offset": int(args.validation_offset),
            "train_valid_rows": int(train_valid.sum()),
            "validation_valid_rows": int(validation_valid.sum()),
            "overlap_valid_rows": int(overlap.sum()),
            "query_free": True,
            "source_only": True,
            "benchmark_queries_opened": False,
            "benchmark_ground_truth_opened": False,
            "c_radio_or_radio_adaptor_used": False,
        },
    }
    _atomic_torch_save(payload, output)
    return payload["metadata"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-frame-manifest", required=True)
    parser.add_argument("--exact-mpr-authority", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--torchhub-repo",
        default="/root/.cache/torch/hub/facebookresearch_dinov2_main",
    )
    parser.add_argument("--model-name", default="dinov2_vitb14")
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--validation-stride", type=int, default=5)
    parser.add_argument("--validation-offset", type=int, default=0)
    parser.add_argument("--channel-chunk-size", type=int, default=96)
    parser.add_argument("--row-chunk-size", type=int, default=8192)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
