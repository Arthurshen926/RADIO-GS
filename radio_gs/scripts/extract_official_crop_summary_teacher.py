#!/usr/bin/env python3
"""Extract dense level-2 teachers from official multiscale C-RADIO crop summaries."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

from radio_gs.data.view_split import (
    load_excluded_image_stems,
    select_image_indices,
)
from radio_gs.interfaces.frozen_radio_views import OfficialCropSummaryRuntime
from radio_gs.utils.immutable_artifacts import load_torch_payload


FRAME_ID_MODES = ("numeric_suffix", "source_rank")


def _atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_reusable_dense(path: Path, output_size: tuple[int, int]) -> torch.Tensor:
    value, _digest, _source = load_torch_payload(
        path,
        map_location="cpu",
        label="partial official crop-summary tensor",
    )
    expected_shape = (1536, *output_size)
    if (
        not torch.is_tensor(value)
        or tuple(value.shape) != expected_shape
        or value.dtype != torch.float16
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"partial crop-summary tensor differs: {path}")
    return value


def _window_starts(length: int, window: int, stride_ratio: float) -> list[int]:
    if window >= length:
        return [0]
    stride = max(1, int(round(window * stride_ratio)))
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


def _boxes(height: int, width: int, scales: list[float], stride_ratio: float):
    result: list[tuple[int, int, int, int, float]] = []
    for scale in scales:
        crop_h = min(height, max(2, int(round(height * scale))))
        crop_w = min(width, max(2, int(round(width * scale))))
        for top in _window_starts(height, crop_h, stride_ratio):
            for left in _window_starts(width, crop_w, stride_ratio):
                result.append((top, left, top + crop_h, left + crop_w, scale))
    return result


def _blend_window(height: int, width: int, device: torch.device) -> torch.Tensor:
    if height <= 2 or width <= 2:
        return torch.ones(height, width, device=device)
    y = torch.hann_window(height, periodic=False, device=device).clamp_min(0.10)
    x = torch.hann_window(width, periodic=False, device=device).clamp_min(0.10)
    return y[:, None] * x[None, :]


def _resolve_frames(
    dataset_root: Path,
    label_dir: Path,
    scene: str,
    mode: str,
) -> list[Path]:
    image_dir = dataset_root / scene / "images"
    if mode == "all":
        frames = sorted(image_dir.glob("frame_*.jpg"))
    else:
        frames = []
        for label_path in sorted((label_dir / scene).glob("frame_*.json")):
            image_path = image_dir / f"{label_path.stem}.jpg"
            if not image_path.exists():
                image_path = image_dir / f"{label_path.stem}.png"
            if image_path.exists():
                frames.append(image_path)
    if not frames:
        raise FileNotFoundError(f"no {mode} images resolved for {scene}")
    return frames


def _generic_frame_records(
    image_dir: Path,
    *,
    frame_id_mode: str,
    excluded_image_stems: tuple[str, ...] = (),
) -> tuple[list[tuple[int, Path]], list[str]]:
    """Resolve generic RGBs with the same post-exclusion IDs as extraction.

    Promptable benchmarks build their pose/config authority after excluding a
    target RGB.  ``source_rank`` therefore assigns dense IDs only after the
    exact basename exclusion, matching ``extract_radio_features.py``.
    """

    if frame_id_mode not in FRAME_ID_MODES:
        raise ValueError(f"unsupported frame-id mode: {frame_id_mode}")
    paths = sorted(
        (
            path
            for path in image_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ),
        key=lambda path: (int(path.stem.split("_")[-1]), path.name),
    )
    retained_indices, excluded_names = select_image_indices(
        paths,
        excluded_image_stems,
    )
    retained = [paths[index] for index in retained_indices]
    if frame_id_mode == "source_rank":
        records = list(enumerate(retained))
    else:
        records = [
            (int(path.stem.split("_")[-1]), path)
            for path in retained
        ]
    return records, excluded_names


@torch.no_grad()
def _extract_frame(
    image_path: Path,
    runtime: OfficialCropSummaryRuntime,
    *,
    output_size: tuple[int, int],
    scales: list[float],
    stride_ratio: float,
    crop_resolution: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    image = pil_to_tensor(Image.open(image_path).convert("RGB")).float().div_(255.0)
    _, source_h, source_w = image.shape
    boxes = _boxes(source_h, source_w, scales, stride_ratio)
    accumulator = torch.zeros(1536, *output_size, device=device)
    weight_sum = torch.zeros(*output_size, device=device)
    for start in range(0, len(boxes), batch_size):
        batch_boxes = boxes[start : start + batch_size]
        crops = []
        for top, left, bottom, right, _scale in batch_boxes:
            crop = image[:, top:bottom, left:right][None]
            crop = F.interpolate(
                crop,
                size=(crop_resolution, crop_resolution),
                mode="bilinear",
                align_corners=False,
            )[0]
            crops.append(crop)
        descriptors = runtime.encode(torch.stack(crops).to(device))
        for descriptor, (top, left, bottom, right, _scale) in zip(
            descriptors, batch_boxes
        ):
            target_top = int(math.floor(top * output_size[0] / source_h))
            target_left = int(math.floor(left * output_size[1] / source_w))
            target_bottom = int(math.ceil(bottom * output_size[0] / source_h))
            target_right = int(math.ceil(right * output_size[1] / source_w))
            target_bottom = max(target_top + 1, min(output_size[0], target_bottom))
            target_right = max(target_left + 1, min(output_size[1], target_right))
            blend = _blend_window(
                target_bottom - target_top, target_right - target_left, device
            )
            accumulator[:, target_top:target_bottom, target_left:target_right].add_(
                descriptor[:, None, None] * blend[None]
            )
            weight_sum[target_top:target_bottom, target_left:target_right].add_(blend)
    dense = accumulator / weight_sum.clamp_min(1e-8)[None]
    dense = F.normalize(dense.float(), dim=0, eps=1e-8).half().cpu()
    return dense, {
        "source_image": str(image_path),
        "source_size": [source_h, source_w],
        "output_size": list(output_size),
        "num_crops": len(boxes),
        "scales": scales,
        "stride_ratio": stride_ratio,
        "crop_resolution": crop_resolution,
    }


def extract(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    scenes = [value.strip() for value in args.scenes.split(",") if value.strip()]
    scales = sorted({float(value) for value in args.scales.split(",")})
    if not scales or min(scales) <= 0 or max(scales) > 1:
        raise ValueError("crop scales must be in (0,1]")
    output_size = tuple(int(value) for value in args.output_size.split("x"))
    if len(output_size) != 2 or min(output_size) <= 0:
        raise ValueError("output-size must be HxW")
    runtime = OfficialCropSummaryRuntime.load(
        checkpoint_path=args.radio_checkpoint,
        radio_repo=args.radio_repo,
        version=args.radio_version,
        device=device,
    )
    dataset_root = Path(args.dataset_root)
    label_dir = Path(args.label_dir)
    output_root = Path(args.output_root)
    excluded_image_stems = load_excluded_image_stems(
        args.exclude_image_stem,
        args.exclude_image_stems_file,
    )
    excluded = {
        int(value)
        for value in str(args.exclude_frame_ids).replace(",", " ").split()
        if value.strip()
    }
    included = {
        int(value)
        for value in str(args.include_frame_ids).replace(",", " ").split()
        if value.strip()
    }
    scene_reports = {}
    excluded_image_names: list[str] = []
    for scene in scenes:
        if args.image_dir:
            if len(scenes) != 1:
                raise ValueError("--image-dir requires exactly one --scenes entry")
            image_dir = Path(args.image_dir)
            frame_records, excluded_image_names = _generic_frame_records(
                image_dir,
                frame_id_mode=args.frame_id_mode,
                excluded_image_stems=excluded_image_stems,
            )
        else:
            frames = _resolve_frames(dataset_root, label_dir, scene, args.frames)
            retained_indices, excluded_image_names = select_image_indices(
                frames,
                excluded_image_stems,
            )
            retained = [frames[index] for index in retained_indices]
            frame_records = [
                (int(path.stem.split("_")[-1]), path)
                for path in retained
            ]
        frame_records = [
            (frame_id, path)
            for frame_id, path in frame_records
            if frame_id not in excluded
            and (not included or frame_id in included)
        ]
        if not frame_records:
            raise RuntimeError(f"all frames excluded for {scene}")
        scene_output = output_root / scene
        scene_output.mkdir(parents=True, exist_ok=True)
        frame_reports = []
        for frame_id, image_path in tqdm(
            frame_records,
            desc=f"official crop summaries/{scene}",
        ):
            tensor_path = scene_output / f"rgb_{frame_id}.pt"
            if bool(args.resume_partial) and tensor_path.is_file():
                _load_reusable_dense(tensor_path, output_size)
                with Image.open(image_path) as source_image:
                    source_w, source_h = source_image.size
                report = {
                    "source_image": str(image_path),
                    "source_size": [source_h, source_w],
                    "output_size": list(output_size),
                    "num_crops": len(
                        _boxes(source_h, source_w, scales, float(args.stride_ratio))
                    ),
                    "scales": scales,
                    "stride_ratio": float(args.stride_ratio),
                    "crop_resolution": int(args.crop_resolution),
                }
            else:
                dense, report = _extract_frame(
                    image_path,
                    runtime,
                    output_size=output_size,
                    scales=scales,
                    stride_ratio=float(args.stride_ratio),
                    crop_resolution=int(args.crop_resolution),
                    batch_size=int(args.batch_size),
                    device=device,
                )
                _atomic_torch_save(dense, tensor_path)
            frame_reports.append({"frame_id": frame_id, **report})
        scene_reports[scene] = {
            "num_frames": len(frame_reports),
            "frames": frame_reports,
        }
        # Each scene owns an immutable provenance record.  This avoids
        # concurrent single-scene extraction jobs racing on the aggregate
        # root manifest while retaining the root report for serial runs.
        scene_report = {
            "schema_version": 1,
            "teacher_space": "official_siglip2_crop_summary",
            "radio_version": runtime.version,
            "radio_checkpoint_sha256": runtime.radio_checkpoint_sha256,
            "dataset_root": str(dataset_root.resolve()),
            "label_dir": str(label_dir.resolve()),
            "frame_selection": args.frames,
            "frame_id_mode": args.frame_id_mode,
            "excluded_frame_ids": sorted(excluded),
            "excluded_image_stems": list(excluded_image_stems),
            "excluded_image_names": excluded_image_names,
            "label_content_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
            "scales": scales,
            "stride_ratio": float(args.stride_ratio),
            "output_size": list(output_size),
            "scenes": {scene: scene_reports[scene]},
        }
        _atomic_json_write(scene_output / "manifest.json", scene_report)
    report = {
        "schema_version": 1,
        "teacher_space": "official_siglip2_crop_summary",
        "radio_version": runtime.version,
        "radio_checkpoint_sha256": runtime.radio_checkpoint_sha256,
        "dataset_root": str(dataset_root.resolve()),
        "label_dir": str(label_dir.resolve()),
        "frame_selection": args.frames,
        "frame_id_mode": args.frame_id_mode,
        "excluded_frame_ids": sorted(excluded),
        "excluded_image_stems": list(excluded_image_stems),
        "excluded_image_names": excluded_image_names,
        "label_content_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
        "scales": scales,
        "stride_ratio": float(args.stride_ratio),
        "output_size": list(output_size),
        "scenes": scene_reports,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json_write(output_root / "manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument(
        "--image-dir",
        default="",
        help="Optional generic frame-id RGB directory for one declared scene.",
    )
    parser.add_argument(
        "--frame-id-mode",
        choices=FRAME_ID_MODES,
        default="numeric_suffix",
        help=(
            "Teacher tensor ID policy. source_rank assigns dense IDs after "
            "exact image-stem exclusion, matching promptable field configs."
        ),
    )
    parser.add_argument(
        "--exclude-image-stem",
        action="append",
        default=[],
        help="Exact case-sensitive RGB basename stem to exclude before opening it.",
    )
    parser.add_argument(
        "--exclude-image-stems-file",
        default="",
        help="Optional JSON/text exact-stem exclusion authority.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scenes", default="figurines,ramen,teatime,waldo_kitchen")
    parser.add_argument("--frames", choices=["all", "labeled"], default="all")
    parser.add_argument(
        "--exclude-frame-ids",
        default="",
        help="Comma list of benchmark/evaluation frame IDs never opened by extraction.",
    )
    parser.add_argument(
        "--include-frame-ids",
        default="",
        help="Optional comma/space list restricting extraction to fixed frame IDs.",
    )
    parser.add_argument("--scales", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--stride-ratio", type=float, default=0.5)
    parser.add_argument("--output-size", default="46x62")
    parser.add_argument("--crop-resolution", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--resume-partial", action="store_true")
    parser.add_argument(
        "--radio-checkpoint",
        default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar",
    )
    parser.add_argument("--radio-repo", default="/root/RADIO")
    parser.add_argument("--radio-version", default="c-radio_v4-h")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(extract(args), indent=2))


if __name__ == "__main__":
    main()
