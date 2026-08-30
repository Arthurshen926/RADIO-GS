"""Resumable, sharded exact MoGe-3 inference for sealed source views."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from radio_gs.v4.contracts.geometry_receipt import sha256_file
from radio_gs.v4.geometry.moge3_inference import OfficialMoge3Runner


def _frame_index(path: Path) -> int:
    values = re.findall(r"\d+", path.stem)
    if not values:
        raise ValueError(f"cannot extract frame index from {path.name}")
    return int(values[-1])


def _load_source_frames(authority_path: Path, scene_root: Path) -> list[tuple[int, Path]]:
    authority = json.loads(authority_path.read_text())
    metadata = authority.get("metadata", {})
    for key in ("benchmark_images_opened", "benchmark_masks_opened", "text_queries_opened"):
        if metadata.get(key) is not False:
            raise ValueError(f"source authority is not sealed: {key}={metadata.get(key)!r}")
    requested = [int(value) for value in authority["frame_indices"]]
    available: dict[int, Path] = {}
    for path in sorted((scene_root / "images").glob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        index = _frame_index(path)
        if index in available:
            raise ValueError(f"duplicate source image frame index {index}")
        available[index] = path
    missing = sorted(set(requested) - set(available))
    if missing:
        raise FileNotFoundError(f"authority source frames missing from RGB directory: {missing[:8]}")
    return [(index, available[index]) for index in requested]


def _existing_output_is_valid(path: Path, expected: dict[str, object]) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception:
        return False
    return all(payload.get(key) == value for key, value in expected.items())


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, object]:
    if args.num_shards <= 0 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard id must be in [0, num_shards)")
    authority_path = Path(args.source_authority).resolve(strict=True)
    scene_root = Path(args.scene_root).resolve(strict=True)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_frames = _load_source_frames(authority_path, scene_root)
    frames = all_frames[args.shard_id :: args.num_shards]
    revision = OfficialMoge3Runner.RELEASES[args.model_id]
    runner = OfficialMoge3Runner.from_huggingface(
        args.model_id,
        revision=revision,
        device=args.device,
        resolution_level=args.resolution_level,
        refine_steps=args.refine_steps,
        use_fp16=not args.disable_fp16,
    )
    records: list[dict[str, object]] = []
    started = time.time()
    for ordinal, (frame_index, image_path) in enumerate(frames, start=1):
        image_sha256 = sha256_file(image_path)
        destination = output_dir / f"frame_{frame_index:05d}.pt"
        expected = {
            "schema": "radio_gs.surface_object_memory_v4.moge3_prediction.v1",
            "frame_index": frame_index,
            "source_image_sha256": image_sha256,
            "model_id": args.model_id,
            "model_revision": revision,
            "model_checkpoint_sha256": runner.checkpoint_sha256,
            "fov_x_degrees": float(args.fov_x_degrees),
        }
        frame_started = time.time()
        if not _existing_output_is_valid(destination, expected):
            rgb = torch.from_numpy(np.asarray(Image.open(image_path).convert("RGB")).copy()).permute(2, 0, 1)
            rgb = rgb.float().div_(255)
            prediction = runner.predict(rgb, fov_x=float(args.fov_x_degrees))
            payload = {
                **expected,
                "source_image": str(image_path),
                "point_map": prediction.point_map.to(torch.float16),
                "normals": prediction.normals.to(torch.float16),
                "validity": prediction.validity,
                "intrinsics": prediction.intrinsics,
                "confidence_semantics": "binary MoGe-3 geometry-validity observation weight",
                "resolution_level": int(args.resolution_level),
                "refine_steps": int(args.refine_steps),
                "use_fp16_inference": not args.disable_fp16,
            }
            temporary = destination.with_suffix(f".pt.tmp.{os.getpid()}")
            torch.save(payload, temporary)
            temporary.replace(destination)
            reused = False
        else:
            reused = True
        payload = torch.load(destination, map_location="cpu")
        validity = torch.as_tensor(payload["validity"], dtype=torch.bool)
        points = torch.as_tensor(payload["point_map"])
        records.append(
            {
                "frame_index": frame_index,
                "source_image": str(image_path),
                "source_image_sha256": image_sha256,
                "prediction": str(destination),
                "prediction_sha256": sha256_file(destination),
                "valid_fraction": float(validity.float().mean()),
                "median_depth": float(points[..., 2][validity].float().median()),
                "elapsed_seconds": time.time() - frame_started,
                "reused": reused,
            }
        )
        print(
            f"shard {args.shard_id}/{args.num_shards} {ordinal}/{len(frames)} "
            f"frame={frame_index} reused={reused}",
            flush=True,
        )
    manifest = {
        "schema": "radio_gs.surface_object_memory_v4.moge3_shard_manifest.v1",
        "scene_label": args.scene_label,
        "scene_root": str(scene_root),
        "source_authority": str(authority_path),
        "source_authority_sha256": sha256_file(authority_path),
        "source_rgb_opened": True,
        "target_rgb_opened": False,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
        "model_family": "MoGe-3",
        "model_id": args.model_id,
        "model_revision": revision,
        "model_checkpoint": str(runner.checkpoint),
        "model_checkpoint_sha256": runner.checkpoint_sha256,
        "fov_x_degrees": float(args.fov_x_degrees),
        "resolution_level": int(args.resolution_level),
        "refine_steps": int(args.refine_steps),
        "shard_id": int(args.shard_id),
        "num_shards": int(args.num_shards),
        "total_authority_frames": len(all_frames),
        "records": records,
        "elapsed_seconds": time.time() - started,
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    manifest_path = output_dir / f"shard_{args.shard_id:03d}_of_{args.num_shards:03d}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-root", required=True)
    parser.add_argument("--scene-label", required=True)
    parser.add_argument("--source-authority", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fov-x-degrees", type=float, required=True)
    parser.add_argument("--model-id", default="Ruicheng/moge-3-vitl", choices=sorted(OfficialMoge3Runner.RELEASES))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolution-level", type=int, default=9)
    parser.add_argument("--refine-steps", type=int, default=3)
    parser.add_argument("--disable-fp16", action="store_true")
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.fov_x_degrees) or not 1 < args.fov_x_degrees < 179:
        parser.error("fov-x-degrees must be finite and in (1, 179)")
    result = run(args)
    print(json.dumps({"records": len(result["records"]), "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
