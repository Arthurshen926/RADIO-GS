#!/usr/bin/env python3
"""Build query-free automatic-region caches with the official SAM3 decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from radio_gs.scripts.build_sam3_foundation_cache import (
    IMAGE_SUFFIXES,
    _load_sam3_model,
    sha256_file,
    set_requested_cuda_device,
)
def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.logical_and(left, right).sum(dtype=np.int64)
    union = np.logical_or(left, right).sum(dtype=np.int64)
    return float(intersection / union) if union else 0.0


def choose_official_sam3_point_mask(
    masks: np.ndarray, quality: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    candidates = np.asarray(masks)
    scores = np.asarray(quality, dtype=np.float32).reshape(-1)
    if candidates.ndim != 3 or candidates.shape[0] != scores.size or scores.size == 0:
        raise ValueError("SAM3 masks/scores must be aligned non-empty [M,H,W]/[M]")
    index = int(np.argmax(scores))
    return candidates[index].astype(bool), index, float(scores[index])


def mask_nms(
    masks: list[np.ndarray], scores: list[float], *, threshold: float, maximum: int,
) -> list[int]:
    order = sorted(range(len(masks)), key=lambda index: (-scores[index], index))
    kept: list[int] = []
    for index in order:
        if all(mask_iou(masks[index], masks[other]) <= float(threshold) for other in kept):
            kept.append(index)
            if len(kept) >= int(maximum):
                break
    return kept


def pack_masks(masks: np.ndarray) -> torch.Tensor:
    values = np.asarray(masks, dtype=np.uint8)
    if values.ndim != 3:
        raise ValueError("masks must be [M,H,W]")
    return torch.from_numpy(np.packbits(values, axis=-1, bitorder="little"))


def unpack_masks(packed: torch.Tensor, width: int) -> np.ndarray:
    values = np.unpackbits(
        torch.as_tensor(packed).cpu().numpy(), axis=-1, bitorder="little"
    )
    return values[..., : int(width)].astype(bool)


def _images(root: Path, pattern: str) -> list[Path]:
    if root.is_file():
        return [root]
    result = sorted(
        path for path in root.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not result:
        raise FileNotFoundError(f"no images under {root} matching {pattern!r}")
    return result


@torch.inference_mode()
def automatic_masks(processor, image: Image.Image, args: argparse.Namespace) -> dict:
    state = processor.set_image(image)
    masks: list[np.ndarray] = []
    scores: list[float] = []
    seeds: list[tuple[float, float]] = []
    grid = int(args.grid_size)
    for row in range(grid):
        for column in range(grid):
            x = (column + 0.5) * image.width / grid
            y = (row + 0.5) * image.height / grid
            candidates, quality, _low_resolution = processor.model.predict_inst(
                state,
                point_coords=np.asarray([[x, y]], dtype=np.float32),
                point_labels=np.ones(1, dtype=np.int32),
                multimask_output=True,
            )
            mask, _candidate, score = choose_official_sam3_point_mask(candidates, quality)
            area_fraction = float(mask.mean())
            if score < float(args.minimum_quality):
                continue
            if not float(args.minimum_area_fraction) <= area_fraction <= float(args.maximum_area_fraction):
                continue
            masks.append(mask); scores.append(score); seeds.append((x, y))
    kept = mask_nms(
        masks, scores, threshold=float(args.nms_iou), maximum=int(args.maximum_masks)
    )
    if kept:
        selected = np.stack([masks[index] for index in kept])
    else:
        selected = np.empty((0, image.height, image.width), dtype=bool)
    boxes = []
    for mask in selected:
        y, x = np.where(mask)
        boxes.append([int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1])
    return {
        "packed_masks": pack_masks(selected),
        "mask_shape": [int(image.height), int(image.width)],
        "scores": torch.tensor([scores[index] for index in kept], dtype=torch.float32),
        "seed_xy": torch.tensor([seeds[index] for index in kept], dtype=torch.float32).reshape(-1, 2),
        "boxes_xyxy": torch.tensor(boxes, dtype=torch.int32).reshape(-1, 4),
    }


def run(args: argparse.Namespace) -> dict:
    image_paths = _images(Path(args.image_root), str(args.image_glob))
    if int(args.maximum_images) > 0 and len(image_paths) > int(args.maximum_images):
        indices = np.linspace(
            0, len(image_paths) - 1, int(args.maximum_images)
        ).round().astype(int)
        image_paths = [image_paths[int(index)] for index in indices]
    output_root = Path(args.output_root); output_root.mkdir(parents=True, exist_ok=True)
    set_requested_cuda_device(args.device)
    processor = _load_sam3_model(
        checkpoint_path=args.checkpoint_path, device=args.device,
        confidence_threshold=0.0, dtype=args.dtype,
        resolution=int(args.resolution), point_only=True,
    )
    checkpoint_sha256 = sha256_file(args.checkpoint_path)
    reports = []
    for image_path in image_paths:
        output = output_root / f"{image_path.stem}.pt"
        if output.exists() and args.skip_existing:
            continue
        image = Image.open(image_path).convert("RGB")
        payload = automatic_masks(processor, image, args)
        payload["metadata"] = {
            "schema_version": 1,
            "source": "official_sam3_interactive_grid_automatic_masks",
            "official_decoder": True, "query_free": True,
            "image": str(image_path.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "grid_size": int(args.grid_size),
            "minimum_quality": float(args.minimum_quality),
            "minimum_area_fraction": float(args.minimum_area_fraction),
            "maximum_area_fraction": float(args.maximum_area_fraction),
            "nms_iou": float(args.nms_iou),
        }
        torch.save(payload, output)
        reports.append({"image": str(image_path), "output": str(output),
                        "masks": int(payload["scores"].numel())})
    report = {"output_root": str(output_root.resolve()), "images": reports}
    (output_root / "manifest.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--image-glob", default="*")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint-path", default="checkpoints/sam3_modelscope/sam3.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--grid-size", type=int, default=12)
    parser.add_argument("--minimum-quality", type=float, default=0.70)
    parser.add_argument("--minimum-area-fraction", type=float, default=0.001)
    parser.add_argument("--maximum-area-fraction", type=float, default=0.80)
    parser.add_argument("--nms-iou", type=float, default=0.85)
    parser.add_argument("--maximum-masks", type=int, default=64)
    parser.add_argument("--maximum-images", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(); print(json.dumps(run(args), indent=2))


if __name__ == "__main__": main()
