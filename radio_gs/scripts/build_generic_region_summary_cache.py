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


def _crop(image: torch.Tensor, rng: random.Random, crop_index: int) -> tuple[torch.Tensor, list[int]]:
    _, height, width = image.shape
    scales = (1.0, 0.80, 0.67, 0.50)
    scale = scales[crop_index % len(scales)] if crop_index == 0 else rng.choice(scales)
    crop_h = max(16, min(height, int(round(height * scale))))
    crop_w = max(16, min(width, int(round(width * scale))))
    top = 0 if crop_h == height else rng.randint(0, height - crop_h)
    left = 0 if crop_w == width else rng.randint(0, width - crop_w)
    return image[:, top : top + crop_h, left : left + crop_w], [top, left, crop_h, crop_w]


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
    rng.shuffle(paths)
    paths = paths[: min(len(paths), int(args.max_images))]
    device = torch.device(args.device)
    runtime = OfficialCropSummaryRuntime.load(
        checkpoint_path=args.radio_checkpoint,
        radio_repo=args.radio_repo,
        version=args.radio_version,
        device=device,
    )
    pending: list[torch.Tensor] = []
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
        pooled = F.adaptive_avg_pool2d(
            spatial.float(), (int(args.token_grid), int(args.token_grid))
        )
        tokens = pooled.flatten(2).transpose(1, 2).contiguous()
        region_parts.append(tokens.half().cpu())
        summary_parts.append(summary_token.half().cpu())
        descriptor_parts.append(descriptor.half().cpu())
        crop_records.extend(pending_meta)
        pending.clear()
        pending_meta.clear()

    for path in tqdm(paths, desc="generic region-summary pairs"):
        image = pil_to_tensor(Image.open(path).convert("RGB")).float().div_(255.0)
        for crop_index in range(int(args.crops_per_image)):
            crop, box = _crop(image, rng, crop_index)
            pending.append(crop)
            pending_meta.append(
                {
                    "image": str(path.relative_to(image_dir)),
                    "crop_box_tl_hw": box,
                }
            )
            if len(pending) >= int(args.batch_size):
                flush()
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
        "seed": int(args.seed),
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
        "num_images": len(paths),
        "num_crops": int(region_tokens.shape[0]),
        "crop_records": crop_records,
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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
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
