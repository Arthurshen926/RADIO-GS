#!/usr/bin/env python3
"""Build label-free crop-to-full-image DINO pairs for a global query adapter.

Each pair contains a 128px RGB crop descriptor and the frozen official DINO
descriptor at the same original-image pixel in the uncropped RGB frame.  No
depth, pose, semantic label, instance, benchmark query, or 3-D field is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from radio_gs.interfaces import OfficialRadioRuntime
from radio_gs.benchmarks.scannet_pfpr.score_dino_center import (
    center_spatial_descriptor,
    sample_spatial_descriptor_at_pixels,
)


def _physical_space(scene: str) -> str:
    return str(scene).split("_", 1)[0]


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_names(value: str) -> list[str]:
    names = [item for item in str(value).replace(",", " ").split() if item]
    if not names or len(set(names)) != len(names):
        raise ValueError("scene names must be a non-empty duplicate-free list")
    return names


def _excluded_scenes(
    *,
    raw_names: str,
    manifest_path: str,
) -> tuple[set[str], dict[str, Any] | None]:
    names = set(_parse_names(raw_names)) if str(raw_names).strip() else set()
    manifest = None
    if str(manifest_path).strip():
        path = Path(manifest_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", -1)) != 1 or payload.get("purpose") != "global_crop_spatial_adapter_scene_exclusion":
            raise ValueError("crop adapter exclusion manifest is invalid")
        if any(payload.get(key) is not False for key in ("uses_labels", "uses_masks", "uses_clicks", "uses_metrics")):
            raise ValueError("crop adapter exclusion manifest opens evaluator information")
        manifest_names = {str(item) for item in payload.get("scene_names", [])}
        if not manifest_names:
            raise ValueError("crop adapter exclusion manifest has no scenes")
        names |= manifest_names
        manifest = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "scene_count": len(manifest_names),
        }
    if not names:
        raise ValueError("a nonempty benchmark-scene exclusion declaration is required")
    return names, manifest


def _evenly(values: Sequence[Path], count: int) -> list[Path]:
    if count <= 0:
        raise ValueError("frames-per-scene must be positive")
    if len(values) <= count:
        return list(values)
    rows = np.linspace(0, len(values) - 1, int(count)).round().astype(np.int64)
    return [values[int(row)] for row in rows]


def _image(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as handle:
        array = np.asarray(handle.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def _centers(
    *,
    scene: str,
    frame: str,
    width: int,
    height: int,
    crop_size: int,
    count: int,
    seed: int,
) -> np.ndarray:
    if crop_size <= 0 or crop_size % 2:
        raise ValueError("crop-size must be a positive even integer")
    if count <= 0 or width < crop_size or height < crop_size:
        raise ValueError("image/crop geometry cannot provide a training crop")
    digest = hashlib.sha256(
        f"crop-spatial-v1:{seed}:{scene}:{frame}".encode("utf-8")
    ).digest()
    generator = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    half = crop_size // 2
    # The high endpoint is exclusive, so every [y-half:y+half,x-half:x+half]
    # crop has exactly crop_size pixels without padding.
    x = generator.integers(half, width - half + 1, size=int(count))
    y = generator.integers(half, height - half + 1, size=int(count))
    return np.column_stack([x, y]).astype(np.int64)


def build(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.dataset_root)
    scenes = _parse_names(args.scene_names)
    excluded_names, exclusion_manifest = _excluded_scenes(
        raw_names=args.exclude_scene_names,
        manifest_path=args.exclude_manifest,
    )
    excluded = {_physical_space(item) for item in excluded_names}
    leaked = sorted({_physical_space(scene) for scene in scenes} & excluded)
    if leaked:
        raise ValueError(f"benchmark physical spaces leaked into crop adapter cache: {leaked}")
    role = str(args.split_role)
    if role not in {"train", "validation"}:
        raise ValueError("split-role must be train or validation")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    runtime = OfficialRadioRuntime.load(
        radio_repo=args.radio_repo,
        version=args.radio_version,
        adaptor_names=("dino_v3_7b",),
        device=device,
    )
    crop_rows: list[torch.Tensor] = []
    crop_context_rows: list[torch.Tensor] = []
    full_rows: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    scene_reports: list[dict[str, Any]] = []
    try:
        for scene in scenes:
            scene_dir = root / scene
            colors = sorted((scene_dir / "color").glob("*.jpg"))
            selected = _evenly(colors, int(args.frames_per_scene))
            if not selected:
                raise FileNotFoundError(f"no RGB frames for crop adapter scene: {scene_dir}")
            frame_count = 0
            for color_path in selected:
                image = _image(color_path, device)
                height, width = (int(image.shape[-2]), int(image.shape[-1]))
                centres = _centers(
                    scene=scene,
                    frame=color_path.stem,
                    width=width,
                    height=height,
                    crop_size=int(args.crop_size),
                    count=int(args.crops_per_frame),
                    seed=int(args.seed),
                )
                _summary, full_spatial = runtime.encode_adaptor_images(
                    image, "dino_v3_7b", feature_fmt="NCHW"
                )
                target = sample_spatial_descriptor_at_pixels(
                    full_spatial,
                    centres.astype(np.float32),
                    image_width=width,
                    image_height=height,
                )
                half = int(args.crop_size) // 2
                crop_batch = torch.stack(
                    [
                        image[0, :, int(y) - half : int(y) + half, int(x) - half : int(x) + half]
                        for x, y in centres
                    ]
                )
                descriptor_parts: list[torch.Tensor] = []
                context_parts: list[torch.Tensor] = []
                for start in range(0, len(crop_batch), int(args.crop_batch_size)):
                    _summary, crop_spatial = runtime.encode_adaptor_images(
                        crop_batch[start : start + int(args.crop_batch_size)],
                        "dino_v3_7b",
                        feature_fmt="NCHW",
                    )
                    descriptor_parts.append(
                        torch.from_numpy(center_spatial_descriptor(crop_spatial)).to(device)
                    )
                    context_parts.append(
                        F.normalize(crop_spatial.mean(dim=(-2, -1)), dim=-1, eps=1e-8)
                    )
                crop_descriptor = F.normalize(torch.cat(descriptor_parts), dim=-1, eps=1e-8)
                crop_context = torch.cat(context_parts)
                if crop_descriptor.shape != target.shape:
                    raise RuntimeError("crop and full-image official DINO dimensions differ")
                if crop_context.shape != target.shape:
                    raise RuntimeError("crop context and full-image official DINO dimensions differ")
                crop_rows.append(crop_descriptor.cpu().to(torch.float16))
                crop_context_rows.append(crop_context.cpu().to(torch.float16))
                full_rows.append(target.cpu().to(torch.float16))
                records.extend(
                    {
                        "scene_id": scene,
                        "frame_id": color_path.stem,
                        "center_xy": [int(x), int(y)],
                        "crop_size_px": int(args.crop_size),
                    }
                    for x, y in centres
                )
                frame_count += 1
            scene_reports.append(
                {"scene_id": scene, "frames": frame_count, "pairs": frame_count * int(args.crops_per_frame)}
            )
    finally:
        del runtime
        if device.type == "cuda":
            torch.cuda.empty_cache()
    crops = torch.cat(crop_rows)
    crop_context = torch.cat(crop_context_rows)
    full = torch.cat(full_rows)
    metadata = {
        "schema_version": 1,
        "training_scope": "global_cross_scene_crop_to_spatial_dino",
        "split_role": role,
        "dataset": "ScanNet_frames_25k_rgb_only",
        "scene_names": scenes,
        "excluded_physical_spaces": sorted(excluded),
        "benchmark_exclusion_declared": True,
        "benchmark_exclusion_manifest": exclusion_manifest,
        "physical_space_disjoint": True,
        "uses_benchmark_scenes": False,
        "uses_benchmark_labels": False,
        "uses_depth": False,
        "uses_pose": False,
        "uses_instances": False,
        "uses_text": False,
        "teacher": "official_c_radio_v4_dino_v3_7b_spatial",
        "query_descriptor": "official_dino_center_3x3_on_128px_crop",
        "query_context_descriptor": "official_dino_spatial_global_mean_on_128px_crop",
        "target_descriptor": "official_dino_bilinear_full_image_spatial_at_same_pixel",
        "records": records,
        "scene_reports": scene_reports,
        "config": {
            "frames_per_scene": int(args.frames_per_scene),
            "crops_per_frame": int(args.crops_per_frame),
            "crop_size": int(args.crop_size),
            "seed": int(args.seed),
            "radio_version": str(args.radio_version),
        },
    }
    metadata["record_manifest_sha256"] = _sha256_json(records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "crop_descriptors": crops,
            "crop_context_descriptors": crop_context,
            "full_image_anchor_descriptors": full,
            "metadata": metadata,
        },
        output,
    )
    report = {
        "output": str(output.resolve()),
        "pairs": int(len(crops)),
        "feature_dim": int(crops.shape[1]),
        "metadata": {key: value for key, value in metadata.items() if key != "records"},
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--scene-names", required=True)
    parser.add_argument(
        "--exclude-scene-names",
        default="",
        help="optional target scenes; their physical ScanNet spaces are excluded",
    )
    parser.add_argument(
        "--exclude-manifest",
        default="",
        help="label-free evaluation-scene exclusion manifest",
    )
    parser.add_argument("--split-role", choices=("train", "validation"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames-per-scene", type=int, default=4)
    parser.add_argument("--crops-per-frame", type=int, default=64)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--crop-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--radio-repo", default="/root/RADIO")
    parser.add_argument("--radio-version", default="c-radio_v4-h")
    print(json.dumps(build(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
