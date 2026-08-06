#!/usr/bin/env python3
"""Freeze source-only official SAM3 completions for NVOS prompts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from PIL import Image
from skimage import __version__ as skimage_version
from skimage.filters import threshold_li

from radio_gs.scripts.build_sam3_foundation_cache import (
    _load_sam3_model,
    set_requested_cuda_device,
)


def _load_source_completion_helpers():
    """Load the leaf helper without importing the heavyweight query package."""

    helper_path = (
        Path(__file__).resolve().parents[1]
        / "querying"
        / "sam3_reference_completion.py"
    )
    spec = importlib.util.spec_from_file_location(
        "radio_gs_sam3_reference_completion_leaf", helper_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source-completion helper {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SOURCE_COMPLETION_HELPERS = _load_source_completion_helpers()
aggregate_completed_positive = (
    _SOURCE_COMPLETION_HELPERS.aggregate_completed_positive
)
deterministic_positive_points = (
    _SOURCE_COMPLETION_HELPERS.deterministic_positive_points
)


WIDTH = 1008
HEIGHT = 756
TRIALS = 10
POINTS_PER_TRIAL = 3
FROZEN_MANIFEST_SHA256 = (
    "bafc48ce30a0a637f5ea4d81a196ea240f80c153c41a3e257b6a2fd45fa3f2ea"
)
FROZEN_FULL_COHORT = (
    "fern",
    "flower",
    "fortress",
    "horns_center",
    "horns_left",
    "leaves",
    "orchids",
    "trex",
)
LEGACY_SENTINEL_ORDER = ("horns_left", "fern")
LEGACY_SENTINEL_REGISTRATION_SHA256 = (
    "a30db1d31a6ffc2d651af37c50675cd12a9fb38b448bd3d96c5eb54740428eca"
)
FULL8_EXPANSION_ORDER = (
    "flower",
    "fortress",
    "horns_center",
    "leaves",
    "orchids",
    "trex",
)
FULL8_EXPANSION_REGISTRATION_SHA256 = (
    "d5e8521d037f4f3baca9ac260d196505d9dd7777aa8422e357e5fe2dc12aa280"
)
_REGISTERED_EXECUTION_AUTHORITIES = {
    LEGACY_SENTINEL_ORDER: {
        "name": "legacy_two_task_sentinel_v1",
        "registration_sha256": LEGACY_SENTINEL_REGISTRATION_SHA256,
    },
    FULL8_EXPANSION_ORDER: {
        "name": "remaining_six_full8_expansion_v1",
        "registration_sha256": FULL8_EXPANSION_REGISTRATION_SHA256,
    },
}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    descriptor = {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
    digest.update(
        json.dumps(
            descriptor, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    )
    digest.update(b"\0")
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def validate_registered_execution_authority(
    *,
    scene_ids: list[str] | tuple[str, ...],
    manifest_sha256: str,
    registration_sha256: str,
    manifest_cohort: list[str] | tuple[str, ...],
) -> dict[str, str]:
    """Fail closed before CUDA unless the request is an exact frozen phase."""

    if str(manifest_sha256) != FROZEN_MANIFEST_SHA256:
        raise ValueError("NVOS source-completion manifest SHA256 differs")
    cohort = tuple(str(value) for value in manifest_cohort)
    if cohort != FROZEN_FULL_COHORT:
        raise ValueError("NVOS source-completion manifest cohort differs")
    requested = tuple(str(value) for value in scene_ids)
    authority = _REGISTERED_EXECUTION_AUTHORITIES.get(requested)
    if authority is None:
        raise ValueError(
            "source-completion scene order is not a registered execution authority"
        )
    if str(registration_sha256) != str(authority["registration_sha256"]):
        raise ValueError("source-completion registration SHA256 differs")
    return dict(authority)


def _atomic_torch_save(value: object, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, output)
        temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(value: object, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, output)
        temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def _source_authority(manifest: dict, scene_id: str) -> dict:
    scenes = [scene for scene in manifest["scenes"] if scene["scene_id"] == scene_id]
    if len(scenes) != 1:
        raise ValueError(f"manifest must contain exactly one scene {scene_id!r}")
    scene = scenes[0]
    prompt = scene["prompt"]
    frame_id = prompt["frame_id"]
    frames = [frame for frame in scene["frames"] if frame["frame_id"] == frame_id]
    if len(frames) != 1:
        raise ValueError(f"scene {scene_id} has ambiguous prompt frame {frame_id}")
    frame = frames[0]
    return {
        "scene_id": scene_id,
        "frame_id": frame_id,
        "source_rgb_path": frame["rgb_path"],
        "annotation_rgb_path": frame["annotation_rgb_path"],
        "positive_scribble_path": prompt["positive_path"],
        "negative_scribble_path": prompt["negative_path"],
    }


def _load_binary_mask(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != (WIDTH, HEIGHT):
        raise ValueError(f"{path} must be exactly {WIDTH}x{HEIGHT}, got {image.size}")
    return np.asarray(image) > 0


@torch.inference_mode()
def _complete_one(processor, authority: dict, *, output_root: Path) -> dict:
    scene_id = authority["scene_id"]
    source_path = Path(authority["source_rgb_path"]).resolve()
    annotation_path = Path(authority["annotation_rgb_path"]).resolve()
    positive_path = Path(authority["positive_scribble_path"]).resolve()
    negative_path = Path(authority["negative_scribble_path"]).resolve()
    source_sha256 = sha256_file(source_path)
    if sha256_file(annotation_path) != source_sha256:
        raise ValueError(f"{scene_id} source and annotation reference RGB differ")

    raw_positive = _load_binary_mask(positive_path)
    raw_negative = _load_binary_mask(negative_path)
    if bool(np.logical_and(raw_positive, raw_negative).any()):
        raise ValueError(f"{scene_id} raw positive and negative scribbles overlap")
    points = deterministic_positive_points(
        raw_positive, count=TRIALS * POINTS_PER_TRIAL
    ).reshape(TRIALS, POINTS_PER_TRIAL, 2)

    source = Image.open(source_path).convert("RGB")
    source_size = source.size
    source = source.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    resized_rgb = np.asarray(source).copy()
    resized_rgb_sha256 = hashlib.sha256(
        resized_rgb.tobytes(order="C")
    ).hexdigest()

    started = time.time()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = processor.set_image(source)
        trial_masks: list[np.ndarray] = []
        qualities: list[float] = []
        low_resolution_shapes: list[list[int]] = []
        for trial_points in points:
            masks, quality, low_resolution = processor.model.predict_inst(
                state,
                point_coords=trial_points.astype(np.float32, copy=False),
                point_labels=np.ones(POINTS_PER_TRIAL, dtype=np.int32),
                multimask_output=False,
            )
            masks = np.asarray(masks).astype(bool, copy=False)
            quality = np.asarray(quality, dtype=np.float32).reshape(-1)
            low_resolution = np.asarray(low_resolution)
            if masks.shape != (1, HEIGHT, WIDTH):
                raise ValueError(f"{scene_id} unexpected mask shape {masks.shape}")
            if quality.shape != (1,) or not np.isfinite(quality).all():
                raise ValueError(f"{scene_id} malformed official quality output")
            if low_resolution.ndim != 3 or low_resolution.shape[0] != 1:
                raise ValueError(
                    f"{scene_id} malformed low-resolution logits {low_resolution.shape}"
                )
            trial_masks.append(masks[0])
            qualities.append(float(quality[0]))
            low_resolution_shapes.append(list(low_resolution.shape))
    torch.cuda.synchronize()
    stacked_masks = np.stack(trial_masks, axis=0)
    preliminary = stacked_masks.astype(np.float32).mean(axis=0, dtype=np.float32)
    if not np.isfinite(preliminary).all() or np.unique(preliminary).size < 2:
        raise ValueError(f"{scene_id} aggregate is nonfinite or constant")
    li_threshold = float(threshold_li(preliminary))
    aggregate, completed_positive = aggregate_completed_positive(
        stacked_masks,
        raw_positive,
        raw_negative,
        threshold=li_threshold,
    )

    tensors = {
        "trial_masks": torch.from_numpy(stacked_masks.copy()),
        "aggregate_probability": torch.from_numpy(aggregate.copy()),
        "completed_positive": torch.from_numpy(completed_positive.copy()),
        "raw_positive": torch.from_numpy(raw_positive.copy()),
        "raw_negative": torch.from_numpy(raw_negative.copy()),
        "point_coordinates_xy": torch.from_numpy(points.copy()),
        "quality": torch.tensor(qualities, dtype=torch.float32),
    }
    tensor_digests = {
        name: tensor_sha256(value) for name, value in sorted(tensors.items())
    }
    bundle_digest = _canonical_sha256(tensor_digests)
    payload = {
        "artifact_type": "radio_gs.nvos_sam3_reference_completion",
        "schema_version": 1,
        "authority": {
            **authority,
            "source_rgb_sha256": source_sha256,
            "annotation_rgb_sha256": source_sha256,
            "positive_scribble_sha256": sha256_file(positive_path),
            "negative_scribble_sha256": sha256_file(negative_path),
            "source_size": list(source_size),
            "decoder_source_size": [WIDTH, HEIGHT],
            "resized_rgb_tensor_sha256": resized_rgb_sha256,
            "target_rgb_opened": False,
            "target_mask_opened": False,
        },
        "method": {
            "trials": TRIALS,
            "positive_points_per_trial": POINTS_PER_TRIAL,
            "negative_points_sent_to_sam3": 0,
            "multimask_output": False,
            "aggregation": "mean of ten official binary masks",
            "threshold": "skimage.filters.threshold_li then aggregate >= threshold",
            "li_threshold": li_threshold,
            "scribble_overwrite": "raw positive true then raw negative false",
            "negative_observation": "raw negative scribble only",
        },
        "tensor_sha256": tensor_digests,
        "tensor_bundle_sha256": bundle_digest,
        "tensors": tensors,
    }
    artifact_path = output_root / "masks" / f"{scene_id}.pt"
    png_path = output_root / "masks" / f"{scene_id}.png"
    receipt_path = output_root / "receipts" / f"{scene_id}.json"
    _atomic_torch_save(payload, artifact_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    if png_path.exists():
        raise FileExistsError(png_path)
    Image.fromarray(completed_positive.astype(np.uint8) * 255, mode="L").save(
        png_path
    )
    reloaded = torch.load(artifact_path, map_location="cpu", weights_only=True)
    reloaded_digests = {
        name: tensor_sha256(value)
        for name, value in sorted(reloaded["tensors"].items())
    }
    if reloaded_digests != tensor_digests:
        raise RuntimeError(f"{scene_id} frozen tensor hashes failed reload")

    receipt = {
        "schema_version": "nvos_sam3_reference_completion_receipt_v1",
        "scene_id": scene_id,
        "frame_id": authority["frame_id"],
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": sha256_file(artifact_path),
        "mask_png_path": str(png_path.resolve()),
        "mask_png_sha256": sha256_file(png_path),
        "tensor_sha256": tensor_digests,
        "tensor_bundle_sha256": bundle_digest,
        "source": payload["authority"],
        "method": payload["method"],
        "positive_pixel_count": int(raw_positive.sum()),
        "negative_pixel_count": int(raw_negative.sum()),
        "completed_positive_pixels": int(completed_positive.sum()),
        "completed_positive_fraction": float(completed_positive.mean()),
        "sam_before_overwrite_positive_coverage": float(
            (aggregate[raw_positive] >= li_threshold).mean()
        ),
        "sam_before_overwrite_negative_overlap": float(
            (aggregate[raw_negative] >= li_threshold).mean()
        ),
        "quality": qualities,
        "low_resolution_shapes": low_resolution_shapes,
        "scene_elapsed_seconds": float(time.time() - started),
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_opened": False,
    }
    _atomic_json(receipt, receipt_path)
    return receipt


def run(args: argparse.Namespace) -> dict:
    manifest_path = Path(args.manifest).resolve()
    registration_path = Path(args.registration).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_root = Path(args.output_root).resolve()
    manifest_sha256 = sha256_file(manifest_path)
    registration_sha256 = sha256_file(registration_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scene_ids = [value for value in args.scene_ids.replace(",", " ").split() if value]
    execution_authority = validate_registered_execution_authority(
        scene_ids=scene_ids,
        manifest_sha256=manifest_sha256,
        registration_sha256=registration_sha256,
        manifest_cohort=manifest.get("protocol", {}).get("cohort", []),
    )
    for scene_id in scene_ids:
        if (output_root / "receipts" / f"{scene_id}.json").exists():
            raise FileExistsError(output_root / "receipts" / f"{scene_id}.json")

    set_requested_cuda_device(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    processor = _load_sam3_model(
        checkpoint_path=str(checkpoint_path),
        device=args.device,
        confidence_threshold=0.0,
        dtype="bfloat16",
        resolution=1008,
        point_only=True,
    )
    reports = [
        _complete_one(
            processor,
            _source_authority(manifest, scene_id),
            output_root=output_root,
        )
        for scene_id in scene_ids
    ]
    summary = {
        "schema_version": "nvos_sam3_reference_completion_phase1_v1",
        "status": "all_registered_source_reference_receipts_frozen",
        "execution_authority": execution_authority,
        "scene_order": scene_ids,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "registration_path": str(registration_path),
        "registration_sha256": registration_sha256,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "official_sam3_source": "/root/external/sam3",
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "numpy_version": np.__version__,
        "skimage_version": skimage_version,
        "device_name": torch.cuda.get_device_name(),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "receipts": {
            report["scene_id"]: {
                "path": str(
                    (output_root / "receipts" / f"{report['scene_id']}.json").resolve()
                ),
                "sha256": sha256_file(
                    output_root / "receipts" / f"{report['scene_id']}.json"
                ),
                "artifact_sha256": report["artifact_sha256"],
                "tensor_bundle_sha256": report["tensor_bundle_sha256"],
            }
            for report in reports
        },
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_opened": False,
    }
    _atomic_json(summary, output_root / "phase1_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scene-ids", default="horns_left fern")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
