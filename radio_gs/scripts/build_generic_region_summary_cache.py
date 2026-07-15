#!/usr/bin/env python3
"""Build query-free generic crop pairs for the global region-summary aligner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

from radio_gs.interfaces.frozen_radio_views import OfficialCropSummaryRuntime


def _image_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        paths.extend(root.rglob(pattern))
    return sorted(set(paths))


def _select_shard(paths: list[Path], shard_count: int, shard_index: int) -> list[Path]:
    shard_count = int(shard_count)
    shard_index = int(shard_index)
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("shard-count must be positive and shard-index must be in range")
    selected = paths[shard_index::shard_count]
    if not selected:
        raise ValueError("generic region-summary shard is empty")
    return selected


def _parse_crop_scales(raw: str) -> tuple[float, ...]:
    scales = tuple(float(value) for value in str(raw).replace(",", " ").split())
    if not scales or any(not 0.0 < value <= 1.0 for value in scales):
        raise ValueError("crop scales must be non-empty and lie in (0, 1]")
    return scales


def _crop(
    image: torch.Tensor,
    rng: random.Random,
    crop_index: int,
    *,
    scales: tuple[float, ...] = (1.0, 0.80, 0.67, 0.50),
    scale_policy: str = "legacy_first_then_random",
) -> tuple[torch.Tensor, list[int]]:
    _, height, width = image.shape
    if scale_policy == "legacy_first_then_random":
        scale = scales[0] if crop_index == 0 else rng.choice(scales)
    elif scale_policy == "cycle":
        scale = scales[crop_index % len(scales)]
    else:
        raise ValueError(f"unsupported crop scale policy: {scale_policy}")
    crop_h = max(16, min(height, int(round(height * scale))))
    crop_w = max(16, min(width, int(round(width * scale))))
    top = 0 if crop_h == height else rng.randint(0, height - crop_h)
    left = 0 if crop_w == width else rng.randint(0, width - crop_w)
    return image[:, top : top + crop_h, left : left + crop_w], [top, left, crop_h, crop_w]


def _pool_spatial_region(
    spatial: torch.Tensor,
    box_tl_hw: list[int],
    *,
    image_height: int,
    image_width: int,
    token_grid: int,
) -> torch.Tensor:
    """Pool one image-coordinate box from a full-image-context spatial map."""

    values = torch.as_tensor(spatial)
    if values.ndim == 4 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 3 or image_height <= 0 or image_width <= 0 or token_grid <= 0:
        raise ValueError("spatial and image dimensions must be positive [C,H,W]")
    top, left, crop_h, crop_w = (int(value) for value in box_tl_hw)
    if min(top, left, crop_h, crop_w) < 0 or crop_h <= 0 or crop_w <= 0:
        raise ValueError("crop box must be a valid top/left/height/width tuple")
    spatial_h, spatial_w = values.shape[-2:]
    y0 = min(spatial_h - 1, max(0, int(top * spatial_h // image_height)))
    x0 = min(spatial_w - 1, max(0, int(left * spatial_w // image_width)))
    y1 = min(
        spatial_h,
        max(y0 + 1, int((top + crop_h) * spatial_h + image_height - 1) // image_height),
    )
    x1 = min(
        spatial_w,
        max(x0 + 1, int((left + crop_w) * spatial_w + image_width - 1) // image_width),
    )
    pooled = F.adaptive_avg_pool2d(
        values[:, y0:y1, x0:x1].float(), (int(token_grid), int(token_grid))
    )
    return pooled.flatten(1).transpose(0, 1).contiguous()


def build(args: argparse.Namespace) -> dict:
    dataset_id = str(args.dataset_id).strip()
    forbidden = ("lerf", "scannet", "nvos", "spin-nerf", "spin_nerf")
    if not dataset_id or any(value in dataset_id.lower() for value in forbidden):
        raise ValueError("region-summary training requires an explicitly generic non-benchmark dataset-id")
    image_dir = Path(args.image_dir)
    paths = _image_paths(image_dir)
    if not paths:
        raise FileNotFoundError(f"no generic images found under {image_dir}")
    rng = random.Random(int(args.seed))
    crop_scales = _parse_crop_scales(args.crop_scales)
    rng.shuffle(paths)
    paths = paths[: min(len(paths), int(args.max_images))]
    global_selection_payload = {
        "dataset_id": dataset_id,
        "images": [str(path.relative_to(image_dir)) for path in paths],
        "max_images": int(args.max_images),
        "crops_per_image": int(args.crops_per_image),
        "crop_resolution": int(args.crop_resolution),
        "token_grid": int(args.token_grid),
        "crop_scales": list(crop_scales),
        "crop_scale_policy": str(args.crop_scale_policy),
        "source_token_context": str(args.source_token_context),
        "seed": int(args.seed),
    }
    global_selection_hash = hashlib.sha256(
        json.dumps(global_selection_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    paths = _select_shard(paths, int(args.shard_count), int(args.shard_index))
    device = torch.device(args.device)
    runtime = OfficialCropSummaryRuntime.load(
        checkpoint_path=args.radio_checkpoint,
        radio_repo=args.radio_repo,
        version=args.radio_version,
        device=device,
    )
    pending: list[torch.Tensor] = []
    pending_source_tokens: list[torch.Tensor] = []
    pending_meta: list[dict] = []
    region_parts: list[torch.Tensor] = []
    summary_parts: list[torch.Tensor] = []
    descriptor_parts: list[torch.Tensor] = []
    crop_records: list[dict] = []

    def flush() -> None:
        if not pending:
            return
        batch = torch.stack(
            [
                F.interpolate(
                    crop[None],
                    size=(int(args.crop_resolution), int(args.crop_resolution)),
                    mode="bilinear",
                    align_corners=False,
                )[0]
                for crop in pending
            ]
        ).to(device)
        spatial, summary_token, descriptor = runtime.encode_training_pair(batch)
        if pending_source_tokens:
            if len(pending_source_tokens) != len(pending):
                raise RuntimeError("full-image source tokens do not align with crops")
            tokens = torch.stack(pending_source_tokens).float()
        else:
            pooled = F.adaptive_avg_pool2d(
                spatial.float(), (int(args.token_grid), int(args.token_grid))
            )
            tokens = pooled.flatten(2).transpose(1, 2).contiguous()
        region_parts.append(tokens.half().cpu())
        summary_parts.append(summary_token.half().cpu())
        descriptor_parts.append(descriptor.half().cpu())
        crop_records.extend(pending_meta)
        pending.clear()
        pending_source_tokens.clear()
        pending_meta.clear()

    progress = tqdm(total=len(paths), desc="generic region-summary pairs")
    for group_start in range(0, len(paths), int(args.batch_size)):
        group_paths = paths[group_start : group_start + int(args.batch_size)]
        group_images = [
            pil_to_tensor(Image.open(path).convert("RGB")).float().div_(255.0)
            for path in group_paths
        ]
        full_spatial_batch = None
        if args.source_token_context == "full_image":
            full_inputs = torch.stack(
                [
                    F.interpolate(
                        image[None],
                        size=(int(args.crop_resolution), int(args.crop_resolution)),
                        mode="bilinear",
                        align_corners=False,
                    )[0]
                    for image in group_images
                ]
            ).to(device)
            full_spatial_batch = runtime.encode_training_pair(full_inputs)[0]
        for group_index, (path, image) in enumerate(zip(group_paths, group_images)):
            full_spatial = (
                None
                if full_spatial_batch is None
                else full_spatial_batch[group_index]
            )
            for crop_index in range(int(args.crops_per_image)):
                crop, box = _crop(
                    image,
                    rng,
                    crop_index,
                    scales=crop_scales,
                    scale_policy=str(args.crop_scale_policy),
                )
                pending.append(crop)
                if full_spatial is not None:
                    pending_source_tokens.append(
                        _pool_spatial_region(
                            full_spatial,
                            box,
                            image_height=int(image.shape[1]),
                            image_width=int(image.shape[2]),
                            token_grid=int(args.token_grid),
                        ).half().cpu()
                    )
                pending_meta.append(
                    {
                        "image": str(path.relative_to(image_dir)),
                        "crop_box_tl_hw": box,
                        "source_token_context": str(args.source_token_context),
                    }
                )
                if len(pending) >= int(args.batch_size):
                    flush()
        progress.update(len(group_paths))
    progress.close()
    flush()
    region_tokens = torch.cat(region_parts, dim=0)
    summary_tokens = torch.cat(summary_parts, dim=0)
    descriptors = torch.cat(descriptor_parts, dim=0)
    manifest_payload = {
        "dataset_id": dataset_id,
        "images": [str(path.relative_to(image_dir)) for path in paths],
        "crops_per_image": int(args.crops_per_image),
        "crop_resolution": int(args.crop_resolution),
        "token_grid": int(args.token_grid),
        "crop_scales": list(crop_scales),
        "crop_scale_policy": str(args.crop_scale_policy),
        "source_token_context": str(args.source_token_context),
        "seed": int(args.seed),
        "shard_count": int(args.shard_count),
        "shard_index": int(args.shard_index),
        "global_selection_manifest_sha256": global_selection_hash,
    }
    manifest_hash = hashlib.sha256(
        json.dumps(manifest_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    metadata = {
        "schema_version": 1,
        "training_scope": "global_cross_scene",
        "dataset_id": dataset_id,
        "dataset_manifest_sha256": manifest_hash,
        "uses_benchmark_test_vocabulary": False,
        "uses_benchmark_scenes": False,
        "annotations_opened": False,
        "text_opened": False,
        "radio_version": runtime.version,
        "radio_checkpoint_sha256": runtime.radio_checkpoint_sha256,
        "region_token_grid": [int(args.token_grid), int(args.token_grid)],
        "region_token_dim": 1280,
        "summary_token_dim": 1280,
        "official_descriptor_dim": int(descriptors.shape[1]),
        "official_summary_head_used_for_target": True,
        "custom_text_projection": False,
        "source_token_context": str(args.source_token_context),
        "crop_scales": list(crop_scales),
        "crop_scale_policy": str(args.crop_scale_policy),
        "source_region_sampling": (
            "image_box_to_spatial_slice_adaptive_pool"
            if args.source_token_context == "full_image"
            else "crop_reencode_adaptive_pool"
        ),
        "num_images": len(paths),
        "num_crops": int(region_tokens.shape[0]),
        "crop_records": crop_records,
        "shard_count": int(args.shard_count),
        "shard_index": int(args.shard_index),
        "global_selection_manifest_sha256": global_selection_hash,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "radio_region_tokens": region_tokens,
            "official_summary_tokens": summary_tokens,
            "official_crop_summaries": descriptors,
            "metadata": metadata,
        },
        output,
    )
    report = {
        "output": str(output),
        "dataset_id": dataset_id,
        "num_images": len(paths),
        "num_crops": int(region_tokens.shape[0]),
        "region_tokens": list(region_tokens.shape),
        "summary_tokens": list(summary_tokens.shape),
        "official_crop_summaries": list(descriptors.shape),
        "dataset_manifest_sha256": manifest_hash,
        "shard_count": int(args.shard_count),
        "shard_index": int(args.shard_index),
        "global_selection_manifest_sha256": global_selection_hash,
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--crops-per-image", type=int, default=2)
    parser.add_argument("--crop-resolution", type=int, default=384)
    parser.add_argument("--token-grid", type=int, default=8)
    parser.add_argument(
        "--crop-scales",
        default="1.0,0.8,0.67,0.5",
        help="Relative height/width scales used to construct target regions.",
    )
    parser.add_argument(
        "--crop-scale-policy",
        choices=["legacy_first_then_random", "cycle"],
        default="legacy_first_then_random",
        help="Legacy sampling or deterministic coverage of every requested scale.",
    )
    parser.add_argument(
        "--source-token-context",
        choices=["crop", "full_image"],
        default="crop",
        help=(
            "Use spatial tokens from the re-encoded crop or from the matching "
            "region of the full-image spatial map; official crop summaries remain targets."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--radio-checkpoint",
        default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar",
    )
    parser.add_argument("--radio-repo", default="/root/RADIO")
    parser.add_argument("--radio-version", default="c-radio_v4-h")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()
