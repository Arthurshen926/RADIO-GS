#!/usr/bin/env python3
"""Measure the crop-context shift before changing the PFPR query compiler.

This is intentionally an evaluator-private diagnostic: it compares the
method-visible held-out crop with the frozen official DINO token at the same
private source-image pixel.  It produces no field scores or submission
artifacts, so it cannot be used as a PFPR result.  Its sole purpose is to
decide whether a global crop-context bridge has enough signal to justify
training it on scene-disjoint, label-free RGB data.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from radio_gs.interfaces import (
    GlobalCropContextAdapter,
    OfficialRadioRuntime,
    crop_spatial_adapter_sha256,
)

from .audit_teacher_oracle import (
    _color_pixel_from_private_depth,
    _load_records,
)
from .score_dino_center import center_spatial_descriptor, sample_spatial_descriptor_at_pixels


def crop_context_descriptors(
    summary: torch.Tensor | None,
    spatial: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return query-visible official DINO descriptors at three context scales.

    ``center3x3`` is the frozen PFPR v1 baseline.  ``spatial_global`` and the
    released adaptor summary retain crop context and are reported only as
    oracle diagnostics here; they never inspect a 3-D field, pose, or label.
    """

    values = torch.as_tensor(spatial).float()
    if values.ndim != 4 or values.shape[1] <= 0 or min(values.shape[-2:]) <= 0:
        raise ValueError("official DINO spatial map must be [B,D,H,W]")
    result = {
        "center3x3": torch.from_numpy(center_spatial_descriptor(values)).to(values.device),
        "spatial_global": F.normalize(values.mean(dim=(-2, -1)), dim=-1, eps=1e-8),
    }
    if summary is not None:
        pooled = torch.as_tensor(summary, device=values.device).float()
        if pooled.ndim == 3 and pooled.shape[1] == 1:
            pooled = pooled[:, 0]
        if pooled.ndim == 2 and pooled.shape[0] == values.shape[0]:
            result["official_crop_summary"] = F.normalize(pooled, dim=-1, eps=1e-8)
    return result


def _image(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as handle:
        values = np.asarray(handle.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(values).permute(2, 0, 1).unsqueeze(0).to(device)


def _crop_batch(records: Sequence[Mapping[str, Any]], device: torch.device) -> torch.Tensor:
    images = [_image(Path(str(record["crop_rgb_path"])), device)[0] for record in records]
    if not images or any(image.shape != images[0].shape for image in images):
        raise ValueError("PFPR crop diagnostic needs a nonempty fixed crop shape")
    return torch.stack(images)


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _median(values: Sequence[float]) -> float | None:
    return float(np.median(values)) if values else None


def run(args: argparse.Namespace) -> dict[str, Any]:
    benchmark_dir = Path(args.benchmark_dir)
    frames_root = Path(args.frames_root)
    output = Path(args.output)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    records_by_scene, _candidates, _config = _load_records(benchmark_dir)
    requested = set(str(args.scene_names).replace(",", " ").split())
    if requested:
        unknown = requested - set(records_by_scene)
        if unknown:
            raise ValueError(f"unknown PFPR diagnostic scenes: {sorted(unknown)}")
        records_by_scene = {scene: records_by_scene[scene] for scene in sorted(requested)}
    context_adapter = None
    context_adapter_manifest = None
    context_adapter_path = ""
    if str(args.crop_context_adapter_checkpoint).strip():
        context_adapter_path = str(
            Path(args.crop_context_adapter_checkpoint).resolve()
        )
        context_adapter, context_adapter_manifest = (
            GlobalCropContextAdapter.from_checkpoint(context_adapter_path)
        )
        context_adapter = context_adapter.to(device).eval()
    runtime = OfficialRadioRuntime.load(
        radio_repo=args.radio_repo,
        version=args.radio_version,
        adaptor_names=("dino_v3_7b",),
        device=device,
    )
    rows: list[dict[str, object]] = []
    usable: dict[str, list[float]] = {
        "center3x3": [],
        "spatial_global": [],
        "official_crop_summary": [],
    }
    if context_adapter is not None:
        usable["frozen_global_context_bridge"] = []
    try:
        with torch.inference_mode():
            for scene_id in sorted(records_by_scene):
                records = records_by_scene[scene_id]
                query_views: dict[str, list[torch.Tensor]] = {}
                for start in range(0, len(records), int(args.crop_batch_size)):
                    batch_records = records[start : start + int(args.crop_batch_size)]
                    summary, spatial = runtime.encode_adaptor_images(
                        _crop_batch(batch_records, device),
                        "dino_v3_7b",
                        feature_fmt="NCHW",
                    )
                    for name, values in crop_context_descriptors(summary, spatial).items():
                        query_views.setdefault(name, []).append(values.detach())
                query_views = {
                    name: F.normalize(torch.cat(values), dim=-1, eps=1e-8)
                    for name, values in query_views.items()
                }
                if context_adapter is not None:
                    query_views["frozen_global_context_bridge"] = context_adapter(
                        query_views["center3x3"], query_views["spatial_global"]
                    )

                targets: list[torch.Tensor] = []
                cached: dict[str, tuple[torch.Tensor, int, int]] = {}
                for record in records:
                    frame_id = str(record["source_frame_id"])
                    if frame_id not in cached:
                        image_path = frames_root / scene_id / "color" / f"{frame_id}.jpg"
                        image = _image(image_path, device)
                        _summary, spatial = runtime.encode_adaptor_images(
                            image, "dino_v3_7b", feature_fmt="NCHW"
                        )
                        cached[frame_id] = (
                            spatial.detach(),
                            int(image.shape[-1]),
                            int(image.shape[-2]),
                        )
                    spatial, width, height = cached[frame_id]
                    target = sample_spatial_descriptor_at_pixels(
                        spatial,
                        np.asarray(
                            _color_pixel_from_private_depth(
                                frames_root / scene_id, record
                            ),
                            dtype=np.float32,
                        ),
                        image_width=width,
                        image_height=height,
                    )
                    targets.append(target[0])
                target_matrix = F.normalize(torch.stack(targets), dim=-1, eps=1e-8)
                for local_index, record in enumerate(records):
                    row: dict[str, object] = {
                        "query_id": str(record["query_id"]),
                        "scene_id": scene_id,
                    }
                    for name, values in query_views.items():
                        if values.shape[1] != target_matrix.shape[1]:
                            row[f"{name}_available"] = False
                            continue
                        cosine = float((values[local_index] * target_matrix[local_index]).sum())
                        row[f"{name}_available"] = True
                        row[f"{name}_to_full_anchor_cosine"] = cosine
                        usable[name].append(cosine)
                    rows.append(row)
    finally:
        del runtime
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = {
        "diagnostic": "ScanNet-PFPR crop-context-to-full-image-DINO alignment",
        "formal_prediction_eligible": False,
        "reason": (
            "opens evaluator-private source frames, depth pixels, and poses only to "
            "measure crop-context shift; it never reads a field, candidate ranking, "
            "target instance, mask, or PFPR metric"
        ),
        "protocol": {
            "benchmark_dir": str(benchmark_dir.resolve()),
            "frames_root": str(frames_root.resolve()),
            "teacher": "official_c_radio_v4_dino_v3_7b",
            "query_input": "held_out_rgb_crop",
            "private_target": "full_source_image_spatial_descriptor_at_private_anchor_pixel",
            "uses_instances_or_masks": False,
            "uses_field_or_candidate_geometry": False,
            "uses_private_pose_depth_pixel": True,
            "crop_context_adapter": (
                {
                    "checkpoint": context_adapter_path,
                    "checkpoint_sha256": crop_spatial_adapter_sha256(
                        context_adapter_path
                    ),
                    "manifest": context_adapter_manifest.__dict__,
                    "selected_before_private_diagnostic": True,
                }
                if context_adapter is not None and context_adapter_manifest is not None
                else None
            ),
        },
        "summary": {
            name: {
                "count": len(values),
                "mean_cosine": _mean(values),
                "median_cosine": _median(values),
            }
            for name, values in usable.items()
        },
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scene-names", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--radio-repo", default="/root/RADIO")
    parser.add_argument("--radio-version", default="c-radio_v4-h")
    parser.add_argument(
        "--crop-context-adapter-checkpoint",
        default="",
        help=(
            "optional frozen global RGB-only bridge; this remains a private "
            "alignment diagnostic rather than a PFPR prediction"
        ),
    )
    parser.add_argument("--crop-batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.crop_batch_size <= 0:
        parser.error("--crop-batch-size must be positive")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
