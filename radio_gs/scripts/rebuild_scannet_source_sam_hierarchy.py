#!/usr/bin/env python3
"""Rebuild SHA-bound official-SAM masks for one exact-MPR frame shard."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from radio_gs.scripts.build_sam3_automatic_mask_cache import run as build_masks
from radio_gs.utils.immutable_artifacts import sha256_file


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-mpr-authority", required=True)
    parser.add_argument("--expected-exact-mpr-sha256", required=True)
    parser.add_argument("--current-feature-manifest", required=True)
    parser.add_argument("--expected-feature-manifest-sha256", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--checkpoint-path", default="checkpoints/sam3_modelscope/sam3.pt"
    )
    parser.add_argument(
        "--expected-checkpoint-sha256",
        default="9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e",
    )
    args = parser.parse_args()
    authority_path = Path(args.exact_mpr_authority).resolve()
    manifest_path = Path(args.current_feature_manifest).resolve()
    checkpoint_path = Path(args.checkpoint_path).resolve()
    if sha256_file(authority_path) != args.expected_exact_mpr_sha256:
        raise ValueError("current exact-MPR authority SHA-256 mismatch")
    if sha256_file(manifest_path) != args.expected_feature_manifest_sha256:
        raise ValueError("current feature manifest SHA-256 mismatch")
    if sha256_file(checkpoint_path) != args.expected_checkpoint_sha256:
        raise ValueError("official SAM checkpoint SHA-256 mismatch")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    feature_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        authority.get("schema")
        != "radio_gs.sparse_exact_marginal_responsibility_authority.v1"
        or authority.get("metadata", {}).get("query_independent") is not True
    ):
        raise ValueError("exact-MPR authority contract differs")
    frames = [int(value) for value in authority["frame_indices"]]
    if frames != [int(value) for value in authority["metadata"]["selected_frame_indices"]]:
        raise ValueError("exact-MPR authority frame axes differ")
    selected = [
        frame
        for position, frame in enumerate(frames)
        if position % args.shard_count == args.shard_index
    ]
    manifest_rows = {
        int(row["frame_idx"]): row for row in feature_manifest.get("frames", [])
    }
    image_root = Path(args.image_root).resolve()
    image_records: list[dict[str, object]] = []
    for frame in selected:
        image = image_root / f"{frame}.jpg"
        row = manifest_rows.get(frame)
        if not image.is_file() or row is None or row.get("source_file") != image.name:
            raise ValueError(f"current source image inventory lacks frame {frame}")
        actual = _sha256(image)
        if actual != row.get("source_sha256"):
            raise ValueError(f"current source image SHA differs for frame {frame}")
        image_records.append({"frame": frame, "path": str(image), "sha256": actual})
    shard_root = Path(args.output_root).resolve() / f"shard{args.shard_index}"
    receipt_path = shard_root / "rebuild_receipt.json"
    if receipt_path.exists():
        raise FileExistsError(f"rebuild receipt already exists: {receipt_path}")
    mask_args = argparse.Namespace(
        image_root=str(image_root),
        image_glob="*.jpg",
        image_stems=" ".join(str(frame) for frame in selected),
        output_root=str(shard_root),
        manifest_name="manifest.json",
        checkpoint_path=str(checkpoint_path),
        device=args.device,
        dtype="bfloat16",
        resolution=1008,
        grid_size=12,
        minimum_quality=0.70,
        minimum_area_fraction=0.001,
        maximum_area_fraction=0.80,
        nms_iou=0.85,
        duplicate_minimum_area_ratio=0.90,
        minimum_stability=0.0,
        stability_offset=1.0,
        maximum_masks=0,
        maximum_images=0,
        # A complete per-frame payload is already source/checkpoint/contract
        # bound and validated by the common cache builder.  Reuse it after an
        # interrupted multi-scene run instead of recomputing official SAM;
        # the shard receipt is still committed only after every frame passes.
        skip_existing=True,
    )
    report = build_masks(mask_args)
    outputs = []
    for row in report["images"]:
        path = Path(row["output"]).resolve()
        outputs.append(
            {
                "frame": int(path.stem),
                "path": str(path),
                "sha256": sha256_file(path),
                "num_masks": int(row["masks"]),
            }
        )
    receipt = {
        "artifact_type": "radio_gs_scannet_source_sam_hierarchy_rebuild_shard",
        "status": "complete",
        "scene_id": image_root.parent.name,
        "query_free": True,
        "benchmark_queries_opened": False,
        "benchmark_labels_or_masks_opened": False,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "exact_mpr_authority": {
            "path": str(authority_path),
            "sha256": args.expected_exact_mpr_sha256,
        },
        "current_feature_manifest": {
            "path": str(manifest_path),
            "sha256": args.expected_feature_manifest_sha256,
        },
        "official_sam_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": args.expected_checkpoint_sha256,
        },
        "source_images": image_records,
        "mask_caches": outputs,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "receipt": str(receipt_path), "frames": len(selected)}, indent=2))


if __name__ == "__main__":
    main()
