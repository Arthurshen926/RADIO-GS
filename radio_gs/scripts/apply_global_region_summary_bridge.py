#!/usr/bin/env python3
"""Apply the frozen region aligner and official summary head to RADIO maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from radio_gs.interfaces.semantic_alignment import (
    GlobalRegionSummaryBridge,
    project_dense_region_semantics,
)
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.training.tensor_cache_io import load_training_tensor_cache


def _frame_ids(raw: str) -> set[int]:
    value = str(raw or "").strip()
    if not value:
        return set()
    path = Path(value)
    if path.is_file():
        value = path.read_text(encoding="utf-8")
    tokens: list[str] = []
    for line in value.splitlines():
        tokens.extend(line.split("#", 1)[0].replace(",", " ").split())
    return {int(token) for token in tokens}


def _feature_paths(root: Path, scene: str) -> list[Path]:
    scene_root = root / scene
    direct = sorted(scene_root.glob("rgb_*.pt"))
    return direct or sorted((scene_root / "backbone").glob("rgb_*.pt"))


def apply(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    bridge, manifest = GlobalRegionSummaryBridge.from_checkpoint(
        args.bridge_checkpoint, map_location="cpu"
    )
    bridge = bridge.to(device).eval()
    summary_head = SigLIP2SummaryHead.from_radio_checkpoint(args.radio_checkpoint).to(device)
    summary_head.eval()
    for parameter in summary_head.parameters():
        parameter.requires_grad_(False)
    kernels = tuple(int(value) for value in args.kernel_sizes.split(","))
    requested = _frame_ids(args.frame_ids)
    excluded = _frame_ids(args.exclude_frame_ids)
    output_root = Path(args.output_root)
    reports = {}
    with torch.inference_mode():
        for scene in [value.strip() for value in args.scenes.split(",") if value.strip()]:
            paths = _feature_paths(Path(args.feature_root), scene)
            if requested:
                paths = [
                    path
                    for path in paths
                    if int(path.stem.split("_")[-1]) in requested
                ]
            if excluded:
                paths = [
                    path
                    for path in paths
                    if int(path.stem.split("_")[-1]) not in excluded
                ]
            if not paths:
                raise FileNotFoundError(f"no RADIO maps found for {scene}")
            scene_output = output_root / scene
            scene_output.mkdir(parents=True, exist_ok=True)
            frames = []
            for path in paths:
                feature_map = load_training_tensor_cache(
                    path, map_location="cpu", purpose="RADIO region-summary source"
                ).float()
                if feature_map.ndim == 4:
                    feature_map = feature_map.squeeze(0)
                descriptor = project_dense_region_semantics(
                    bridge,
                    summary_head,
                    feature_map[None].to(device),
                    kernel_sizes=kernels,
                    projection_batch_size=int(args.projection_batch_size),
                )
                dense = descriptor[0].half().cpu()
                frame = int(path.stem.split("_")[-1])
                torch.save(dense, scene_output / f"rgb_{frame}.pt")
                frames.append(frame)
            reports[scene] = {"frames": frames, "num_frames": len(frames)}
    report = {
        "schema_version": 1,
        "feature_space": "global_region_summary_then_official_siglip2_summary",
        "bridge_checkpoint": str(Path(args.bridge_checkpoint).resolve()),
        "bridge_checkpoint_sha256": manifest.checkpoint_sha256,
        "bridge_training_scope": manifest.training_scope,
        "official_summary_head": True,
        "custom_text_projection": False,
        "kernel_sizes": list(kernels),
        "scale_fusion": "normalized_descriptor_mean",
        "feature_root": str(Path(args.feature_root).resolve()),
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
        "scenes": reports,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--bridge-checkpoint", required=True)
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--frame-ids", default="")
    parser.add_argument("--exclude-frame-ids", default="")
    parser.add_argument("--kernel-sizes", default="3,7,15")
    parser.add_argument("--projection-batch-size", type=int, default=2048)
    parser.add_argument(
        "--radio-checkpoint",
        default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(apply(args), indent=2))


if __name__ == "__main__":
    main()
