#!/usr/bin/env python3
"""One-image smoke test for the official SAM3 interactive point decoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from radio_gs.scripts.build_sam3_foundation_cache import (
    _load_sam3_model,
    set_requested_cuda_device,
    sha256_file,
)
from radio_gs.querying.sam3_reference_completion import (
    deterministic_positive_points,
)


MASK_WIDTH = 1008
MASK_HEIGHT = 756
def deterministic_positive_triplet(mask: np.ndarray) -> np.ndarray:
    return deterministic_positive_points(mask, count=3)


def run(args: argparse.Namespace) -> dict:
    image_path = Path(args.image).resolve()
    scribble_path = Path(args.positive_scribble).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_path = Path(args.output).resolve()

    source = Image.open(image_path).convert("RGB")
    source_size = source.size
    source = source.resize((MASK_WIDTH, MASK_HEIGHT), Image.Resampling.LANCZOS)
    source_array = np.asarray(source).copy()
    resized_rgb_sha256 = hashlib.sha256(source_array.tobytes(order="C")).hexdigest()

    scribble_image = Image.open(scribble_path).convert("L")
    if scribble_image.size != (MASK_WIDTH, MASK_HEIGHT):
        raise ValueError(
            f"scribble must be exactly {MASK_WIDTH}x{MASK_HEIGHT}, got {scribble_image.size}"
        )
    scribble = np.asarray(scribble_image) > 0
    points = deterministic_positive_triplet(scribble)

    set_requested_cuda_device(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable after selecting the requested device")
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    processor = _load_sam3_model(
        checkpoint_path=str(checkpoint_path),
        device=args.device,
        confidence_threshold=0.0,
        dtype="bfloat16",
        resolution=1008,
        point_only=True,
    )
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        state = processor.set_image(source)
        masks, quality, low_resolution = processor.model.predict_inst(
            state,
            point_coords=points,
            point_labels=np.ones(3, dtype=np.int32),
            multimask_output=False,
        )
    torch.cuda.synchronize()

    masks = np.asarray(masks).astype(bool, copy=False)
    quality = np.asarray(quality, dtype=np.float32).reshape(-1)
    low_resolution = np.asarray(low_resolution)
    if masks.shape != (1, MASK_HEIGHT, MASK_WIDTH):
        raise ValueError(f"unexpected official mask shape: {masks.shape}")
    if quality.shape != (1,) or not np.isfinite(quality).all():
        raise ValueError(f"unexpected official quality: shape={quality.shape}")
    if low_resolution.ndim != 3 or low_resolution.shape[0] != 1:
        raise ValueError(
            f"unexpected official low-resolution logits shape: {low_resolution.shape}"
        )

    result = {
        "schema_version": "sam3_point_decoder_smoke_v1",
        "status": "ok",
        "official_sam3_source": "/root/external/sam3",
        "image_path": str(image_path),
        "image_sha256": sha256_file(image_path),
        "source_size": list(source_size),
        "decoder_source_size": [MASK_WIDTH, MASK_HEIGHT],
        "resized_rgb_tensor_sha256": resized_rgb_sha256,
        "positive_scribble_path": str(scribble_path),
        "positive_scribble_sha256": sha256_file(scribble_path),
        "positive_pixel_count": int(scribble.sum()),
        "point_coordinates_xy": points.tolist(),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(),
        "multimask_output": False,
        "mask_shape": list(masks.shape),
        "mask_area_pixels": int(masks[0].sum()),
        "mask_area_fraction": float(masks[0].mean()),
        "quality": float(quality[0]),
        "low_resolution_shape": list(low_resolution.shape),
        "elapsed_seconds": float(time.time() - started),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--positive-scribble", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
