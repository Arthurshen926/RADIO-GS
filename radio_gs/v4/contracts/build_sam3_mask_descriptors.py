"""Build query-free SAM3 visual descriptors for already sealed source masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F

from radio_gs.v4.contracts.geometry_receipt import sha256_file
from radio_gs.v4.evaluation.lerf_source_mask_gate import _load_sam_records, _masks


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict:
    from radio_gs.scripts.build_sam3_foundation_cache import (
        _load_sam3_model,
        set_requested_cuda_device,
    )

    authority_path = Path(args.source_rgb_authority).resolve(strict=True)
    authority = json.loads(authority_path.read_text())
    if authority.get("information_policy", {}).get("benchmark_ground_truth_used") is not False:
        raise ValueError("source authority is not label-free")
    image_records = authority["images"][: args.maximum_view_count]
    selected = [record for index, record in enumerate(image_records) if index % args.shard_count == args.shard_index]
    manifests = [Path(value).resolve(strict=True) for value in args.sam_manifest]
    sam_records = _load_sam_records(manifests)
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    set_requested_cuda_device(args.device)
    processor = _load_sam3_model(
        checkpoint_path=str(checkpoint),
        device=args.device,
        confidence_threshold=0.0,
        dtype="bfloat16",
        resolution=1008,
        point_only=True,
    )
    rows = []
    for record in selected:
        frame_id = int(str(record["image_id"]).removeprefix("frame_"))
        image_path = Path(record["path"]).resolve(strict=True)
        mask_path = Path(sam_records[frame_id]["output"]).resolve(strict=True)
        masks = _masks(mask_path, args.mask_height, args.mask_width).to(args.device)
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                state = processor.set_image(image)
        feature = state["backbone_out"]["vision_features"][0].float()
        resized = F.interpolate(
            masks[:, None], size=feature.shape[-2:], mode="bilinear", align_corners=False
        )[:, 0]
        mass = resized.sum((1, 2)).clamp_min(1e-8)
        descriptor = torch.einsum("mhw,chw->mc", resized, feature) / mass[:, None]
        descriptor = F.normalize(descriptor, dim=-1, eps=1e-8).half().cpu()
        output_path = output_root / f"frame_{frame_id:05d}.pt"
        torch.save({
            "schema": "radio_gs.surface_object_memory_v4.sam3_mask_descriptor.v1",
            "frame_id": frame_id,
            "descriptor": descriptor,
            "feature_grid": list(feature.shape[-2:]),
            "source_rgb_sha256": record["sha256"],
            "sam_mask_cache_sha256": sha256_file(mask_path),
            "sam3_checkpoint_sha256": sha256_file(checkpoint),
            "query_or_label_used": False,
        }, output_path)
        rows.append({
            "frame_id": frame_id,
            "output": str(output_path),
            "sha256": sha256_file(output_path),
            "proposal_count": int(descriptor.shape[0]),
        })
    report = {
        "schema": "radio_gs.surface_object_memory_v4.sam3_mask_descriptor_manifest.v1",
        "source_rgb_authority": {"path": str(authority_path), "sha256": sha256_file(authority_path)},
        "sam_manifests": [{"path": str(path), "sha256": sha256_file(path)} for path in manifests],
        "sam3_checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        "information_policy": {
            "source_rgb_used": True,
            "query_text_used": False,
            "benchmark_labels_used": False,
            "target_or_evaluation_rgb_used": False,
        },
        "descriptor": "masked_mean_official_sam3_vision_features_l2_normalized",
        "shard": {"index": args.shard_index, "count": args.shard_count},
        "records": rows,
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    output_manifest = Path(args.output_manifest).resolve()
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-rgb-authority", required=True)
    parser.add_argument("--sam-manifest", action="append", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--maximum-view-count", type=int, default=64)
    parser.add_argument("--mask-height", type=int, default=60)
    parser.add_argument("--mask-width", type=int, default=81)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid descriptor shard")
    report = run(args)
    print(json.dumps({"record_count": len(report["records"]), "shard": report["shard"]}, indent=2))


if __name__ == "__main__":
    main()
