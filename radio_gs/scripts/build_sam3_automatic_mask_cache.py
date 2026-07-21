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


def _as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


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


def mask_stability(
    low_resolution_logits: np.ndarray | None,
    masks: np.ndarray,
    *,
    offset: float,
) -> np.ndarray:
    """Compute the standard two-threshold mask stability when logits exist.

    SAM3's public interactive API exposes low-resolution decoder logits on
    supported checkpoints.  Older wrappers may not, so the cache records a
    neutral value rather than inventing a proxy.  The caller stores the
    provenance flag needed to distinguish the two cases.
    """

    candidates = _as_numpy(masks).astype(bool, copy=False)
    logits = None if low_resolution_logits is None else _as_numpy(low_resolution_logits)
    # Decoder logits normally live on a lower-resolution raster than the
    # returned masks, so only the multimask axis—not H/W—must align.
    if (
        logits is None or logits.ndim != 3 or logits.shape[0] != candidates.shape[0]
        or not np.issubdtype(logits.dtype, np.number)
    ):
        return np.ones(candidates.shape[0], dtype=np.float32)
    high = logits > float(offset)
    low = logits > -float(offset)
    intersection = np.logical_and(high, low).sum(axis=(1, 2), dtype=np.int64)
    union = np.logical_or(high, low).sum(axis=(1, 2), dtype=np.int64)
    return (intersection / np.maximum(union, 1)).astype(np.float32)


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


def containment_aware_deduplicate(
    masks: list[np.ndarray],
    scores: list[float],
    *,
    iou_threshold: float,
    minimum_area_ratio: float,
    maximum: int,
) -> list[int]:
    """Remove only near-identical proposals, preserving nested mask scales.

    Ordinary IoU NMS discards a small part mask whenever it substantially
    overlaps an object mask.  For scale-ordered supervision that distinction
    is evidence, not redundancy.  A proposal is therefore a duplicate only
    when it has both high IoU *and* nearly the same 2-D support area as an
    already kept proposal.  ``maximum=0`` means no cap.
    """

    if not 0.0 <= float(iou_threshold) <= 1.0:
        raise ValueError("iou_threshold must lie in [0,1]")
    if not 0.0 < float(minimum_area_ratio) <= 1.0:
        raise ValueError("minimum_area_ratio must lie in (0,1]")
    area = [int(np.asarray(mask, dtype=bool).sum()) for mask in masks]
    order = sorted(range(len(masks)), key=lambda index: (-float(scores[index]), index))
    kept: list[int] = []
    for index in order:
        if area[index] == 0:
            continue
        duplicate = False
        for other in kept:
            area_ratio = min(area[index], area[other]) / max(area[index], area[other])
            if (
                area_ratio >= float(minimum_area_ratio)
                and mask_iou(masks[index], masks[other]) > float(iou_threshold)
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(index)
            if int(maximum) > 0 and len(kept) >= int(maximum):
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
    stability: list[float] = []
    seeds: list[tuple[float, float]] = []
    prompt_indices: list[int] = []
    candidate_indices: list[int] = []
    logits_available = True
    grid = int(args.grid_size)
    prompt_index = 0
    for row in range(grid):
        for column in range(grid):
            x = (column + 0.5) * image.width / grid
            y = (row + 0.5) * image.height / grid
            candidates, quality, low_resolution = processor.model.predict_inst(
                state,
                point_coords=np.asarray([[x, y]], dtype=np.float32),
                point_labels=np.ones(1, dtype=np.int32),
                multimask_output=True,
            )
            candidates = _as_numpy(candidates).astype(bool, copy=False)
            quality = _as_numpy(quality).astype(np.float32, copy=False).reshape(-1)
            if candidates.ndim != 3 or candidates.shape[0] != quality.size:
                raise ValueError("official SAM3 multimask candidates and quality do not align")
            candidate_stability = mask_stability(
                low_resolution, candidates, offset=float(args.stability_offset)
            )
            logits_available &= (
                low_resolution is not None
                and _as_numpy(low_resolution).ndim == 3
                and _as_numpy(low_resolution).shape[0] == candidates.shape[0]
            )
            for candidate_index, (mask, score, stable) in enumerate(
                zip(candidates, quality, candidate_stability)
            ):
                area_fraction = float(mask.mean())
                if score < float(args.minimum_quality) or stable < float(args.minimum_stability):
                    continue
                if not float(args.minimum_area_fraction) <= area_fraction <= float(args.maximum_area_fraction):
                    continue
                masks.append(mask); scores.append(float(score)); stability.append(float(stable))
                seeds.append((x, y)); prompt_indices.append(prompt_index)
                candidate_indices.append(candidate_index)
            prompt_index += 1
    kept = containment_aware_deduplicate(
        masks, scores, iou_threshold=float(args.nms_iou),
        minimum_area_ratio=float(args.duplicate_minimum_area_ratio),
        maximum=int(args.maximum_masks),
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
        "stability": torch.tensor([stability[index] for index in kept], dtype=torch.float32),
        "seed_xy": torch.tensor([seeds[index] for index in kept], dtype=torch.float32).reshape(-1, 2),
        "prompt_index": torch.tensor([prompt_indices[index] for index in kept], dtype=torch.int32),
        "candidate_index": torch.tensor([candidate_indices[index] for index in kept], dtype=torch.int8),
        "boxes_xyxy": torch.tensor(boxes, dtype=torch.int32).reshape(-1, 4),
        "proposal_area_fraction": torch.tensor(
            [float(masks[index].mean()) for index in kept], dtype=torch.float32
        ),
        "proposal_count_before_deduplication": int(len(masks)),
        "decoder_logits_available": bool(logits_available),
    }


def run(args: argparse.Namespace) -> dict:
    image_paths = _images(Path(args.image_root), str(args.image_glob))
    requested_stems = {
        value for value in str(args.image_stems).replace(",", " ").split() if value
    }
    if requested_stems:
        by_stem = {path.stem: path for path in image_paths}
        missing = sorted(requested_stems - set(by_stem))
        if missing:
            raise ValueError(f"requested image stems are absent: {missing}")
        image_paths = [by_stem[stem] for stem in sorted(requested_stems)]
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
            "schema_version": 2,
            "source": "official_sam3_interactive_grid_multimask_hierarchy",
            "official_decoder": True, "query_free": True,
            "image": str(image_path.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "grid_size": int(args.grid_size),
            "minimum_quality": float(args.minimum_quality),
            "minimum_area_fraction": float(args.minimum_area_fraction),
            "maximum_area_fraction": float(args.maximum_area_fraction),
            "nms_iou": float(args.nms_iou),
            "minimum_stability": float(args.minimum_stability),
            "stability_offset": float(args.stability_offset),
            "deduplication": "containment_aware_near_duplicate_only",
            "duplicate_minimum_area_ratio": float(args.duplicate_minimum_area_ratio),
            "multimask_candidates_retained_before_deduplication": int(
                payload["proposal_count_before_deduplication"]
            ),
            "decoder_logits_available": bool(payload["decoder_logits_available"]),
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
    parser.add_argument(
        "--image-stems", default="",
        help="Optional deterministic subset of frame stems for a scene split.",
    )
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
    parser.add_argument(
        "--duplicate-minimum-area-ratio", type=float, default=0.90,
        help="Only suppress high-IoU masks with near-equal 2-D support area.",
    )
    parser.add_argument("--minimum-stability", type=float, default=0.0)
    parser.add_argument("--stability-offset", type=float, default=1.0)
    parser.add_argument(
        "--maximum-masks", type=int, default=0,
        help="Optional cap after hierarchy-preserving deduplication; zero keeps all.",
    )
    parser.add_argument("--maximum-images", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(); print(json.dumps(run(args), indent=2))


if __name__ == "__main__": main()
