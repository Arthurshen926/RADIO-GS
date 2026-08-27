"""Extract official DINOv2 maps for the sealed SUGM-v3 source32 views."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from radio_gs.v3.training.instance_upper_bound import sha256_file


FRAME_SCHEMA = "radio_gs.native_dinov2_exact_mpr_teacher.v1"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    repository = Path(args.torchhub_repo).resolve(strict=True)
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    model = torch.hub.load(
        str(repository), args.model_name, source="local", pretrained=False
    )
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model


def _image_tensor(path: Path, height: int, width: int) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / np.float32(255.0)
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)[None]
    tensor = F.interpolate(
        tensor, size=(height, width), mode="bicubic",
        align_corners=False, antialias=True,
    )[0]
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    return (tensor - mean) / std


@torch.no_grad()
def _extract_feature(
    model: torch.nn.Module,
    source: Path,
    device: torch.device,
    height: int,
    width: int,
    patch_size: int,
) -> torch.Tensor:
    image = _image_tensor(source, height * patch_size, width * patch_size)
    with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=torch.float16):
        output = model.get_intermediate_layers(
            image[None].to(device), n=1, reshape=True
        )[0]
    if isinstance(output, (tuple, list)):
        output = output[0]
    if output.ndim == 4 and output.shape[0] == 1:
        output = output[0]
    if tuple(output.shape[-2:]) != (height, width):
        raise ValueError("official DINOv2 output grid differs")
    output = F.normalize(output.float(), dim=0, eps=1e-8)
    if not bool(torch.isfinite(output).all()):
        raise ValueError("official DINOv2 feature contains non-finite values")
    return output


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
    cached = cache_root / f"frame_{frame:05d}.pt"
    if cached.exists():
        value = torch.load(cached, map_location="cpu")
        if (
            value.get("schema") != FRAME_SCHEMA
            or int(value.get("frame_index", -1)) != frame
            or value.get("source_sha256") != source_sha256
            or value.get("checkpoint_sha256") != checkpoint_sha256
        ):
            raise ValueError("native DINOv2 cached frame lineage differs")
        return torch.as_tensor(value["feature"]).float()
    feature = _extract_feature(model, path, device, height, width, patch_size)
    cached.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{cached.name}.", suffix=".tmp", dir=cached.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save({
            "schema": FRAME_SCHEMA,
            "frame_index": frame,
            "source_path": str(path),
            "source_sha256": source_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "feature": feature.cpu().half(),
        }, temporary)
        os.replace(temporary, cached)
    finally:
        temporary.unlink(missing_ok=True)
    return feature


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, object]:
    membership_path = Path(args.membership).resolve(strict=True)
    membership = torch.load(membership_path, map_location="cpu")
    metadata = membership["metadata"]
    if (
        metadata.get("benchmark_images_opened") is not False
        or metadata.get("benchmark_masks_opened") is not False
        or metadata.get("evaluation_rgb_opened") is not False
        or int(metadata.get("source_view_count", -1)) != 32
    ):
        raise ValueError("DINO extraction requires the sealed source32 authority")
    height = int(metadata["feature_height"])
    width = int(metadata["feature_width"])
    device = torch.device(args.device)
    model = _load_model(args, device)
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    checkpoint_sha256 = sha256_file(checkpoint)
    cache_root = Path(args.output_root).resolve()
    frames = []
    for record in metadata["source_records"]:
        frame = int(record["frame_id"])
        source = Path(record["source_image"]).resolve(strict=True)
        source_sha256 = sha256_file(source)
        if source_sha256 != record["source_image_sha256"]:
            raise ValueError("source32 RGB hash mismatch")
        feature = _load_or_extract_frame(
            model=model,
            path=source,
            source_sha256=source_sha256,
            frame=frame,
            cache_root=cache_root,
            device=device,
            height=height,
            width=width,
            patch_size=args.patch_size,
            checkpoint_sha256=checkpoint_sha256,
        )
        if tuple(feature.shape) != (768, height, width):
            raise ValueError("official DINOv2 source feature axes differ")
        frames.append(frame)
        del feature
    return {
        "schema": "radio_gs.sugm_v3.source32_native_dinov2_frames.v1",
        "membership": {"path": str(membership_path), "sha256": sha256_file(membership_path)},
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha256},
        "output_root": str(cache_root),
        "frames": frames,
        "source_only": True,
        "benchmark_metrics_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--membership", required=True)
    parser.add_argument(
        "--checkpoint",
        default="/root/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth",
    )
    parser.add_argument(
        "--torchhub-repo",
        default="/root/.cache/torch/hub/facebookresearch_dinov2_main",
    )
    parser.add_argument("--model-name", default="dinov2_vitb14")
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", required=True)
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
