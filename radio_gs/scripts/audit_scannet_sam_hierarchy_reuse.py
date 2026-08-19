#!/usr/bin/env python3
"""Fail-closed audit of an old official-SAM hierarchy against current exact MPR."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from radio_gs.utils.immutable_artifacts import sha256_file


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(args: argparse.Namespace) -> dict[str, object]:
    hierarchy_root = Path(args.hierarchy_root).resolve()
    authority_path = Path(args.exact_mpr_authority).resolve()
    feature_manifest_path = Path(args.current_feature_manifest).resolve()
    image_root = Path(args.current_image_root).resolve()
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    if (
        authority.get("schema")
        != "radio_gs.sparse_exact_marginal_responsibility_authority.v1"
        or int(authority.get("schema_version", -1)) != 1
        or authority.get("metadata", {}).get("query_independent") is not True
    ):
        raise ValueError("current exact-MPR authority contract differs")
    current_frames = [int(value) for value in authority["frame_indices"]]
    if current_frames != [
        int(value) for value in authority["metadata"]["selected_frame_indices"]
    ]:
        raise ValueError("current exact-MPR frame axes differ internally")
    current_manifest = {
        int(row["frame_idx"]): row for row in feature_manifest.get("frames", [])
    }
    current_image_records: list[dict[str, object]] = []
    for frame in current_frames:
        row = current_manifest.get(frame)
        image = image_root / f"{frame}.jpg"
        if row is None or row.get("source_file") != image.name or not image.is_file():
            raise ValueError(f"current source image inventory lacks frame {frame}")
        actual = _sha256(image)
        if actual != row.get("source_sha256"):
            raise ValueError(f"current source image SHA differs for frame {frame}")
        current_image_records.append(
            {"frame": frame, "path": str(image), "sha256": actual}
        )

    cache_paths = sorted(
        hierarchy_root.glob("shard*/*.pt"), key=lambda path: int(path.stem)
    )
    if not cache_paths:
        raise FileNotFoundError("old hierarchy contains no packed-mask caches")
    old_frames: list[int] = []
    old_records: list[dict[str, object]] = []
    checkpoints: set[str] = set()
    parameter_contracts: set[str] = set()
    missing_old_image_paths = 0
    missing_image_sha_bindings = 0
    total_masks = 0
    for cache_path in cache_paths:
        frame = int(cache_path.stem)
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata")
        packed = payload.get("packed_masks")
        mask_shape = payload.get("mask_shape")
        if (
            not isinstance(metadata, dict)
            or metadata.get("schema_version") != 2
            or metadata.get("source")
            != "official_sam3_interactive_grid_multimask_hierarchy"
            or metadata.get("official_decoder") is not True
            or metadata.get("query_free") is not True
            or not isinstance(packed, torch.Tensor)
            or packed.ndim != 3
            or list(mask_shape or []) != [968, 1296]
            or Path(str(metadata.get("image", ""))).stem != str(frame)
        ):
            raise ValueError(f"old official-SAM cache contract differs: {cache_path}")
        old_frames.append(frame)
        checkpoints.add(str(metadata.get("checkpoint_sha256", "")))
        parameter_contract = {
            key: metadata.get(key)
            for key in (
                "grid_size",
                "minimum_quality",
                "minimum_area_fraction",
                "maximum_area_fraction",
                "nms_iou",
                "minimum_stability",
                "stability_offset",
                "deduplication",
                "duplicate_minimum_area_ratio",
            )
        }
        parameter_contracts.add(json.dumps(parameter_contract, sort_keys=True))
        old_image = Path(str(metadata["image"]))
        missing_old_image_paths += int(not old_image.is_file())
        bound_sha = next(
            (
                str(metadata[key])
                for key in (
                    "image_sha256",
                    "source_image_sha256",
                    "source_sha256",
                )
                if len(str(metadata.get(key, ""))) == 64
            ),
            "",
        )
        missing_image_sha_bindings += int(not bound_sha)
        total_masks += int(packed.shape[0])
        old_records.append(
            {
                "frame": frame,
                "cache_path": str(cache_path),
                "cache_sha256": sha256_file(cache_path),
                "declared_image": str(old_image),
                "declared_image_exists": old_image.is_file(),
                "declared_image_sha256": bound_sha,
                "num_masks": int(packed.shape[0]),
            }
        )
    if len(old_frames) != len(set(old_frames)):
        raise ValueError("old hierarchy contains duplicate frame caches")
    if len(checkpoints) != 1 or any(len(value) != 64 for value in checkpoints):
        raise ValueError("old hierarchy SAM checkpoint identity differs")
    if len(parameter_contracts) != 1:
        raise ValueError("old hierarchy generation parameters differ across frames")

    old_set, current_set = set(old_frames), set(current_frames)
    shared = sorted(old_set & current_set)
    old_only = sorted(old_set - current_set)
    current_only = sorted(current_set - old_set)
    frame_axis_exact = old_frames == current_frames
    image_hash_binding_complete = missing_image_sha_bindings == 0
    reusable = frame_axis_exact and image_hash_binding_complete
    return {
        "artifact_type": "radio_gs_scannet_official_sam_hierarchy_reuse_audit",
        "status": "compatible" if reusable else "incompatible_fail_closed",
        "scene_id": "scene0000_00",
        "reusable_for_current_exact_mpr": reusable,
        "reason": (
            "exact frame axis and every old mask cache is source-image SHA bound"
            if reusable
            else "old hierarchy frame axis differs and/or lacks cryptographic source-image binding"
        ),
        "old_hierarchy": {
            "root": str(hierarchy_root),
            "frames": len(old_frames),
            "total_masks": total_masks,
            "official_sam_checkpoint_sha256": next(iter(checkpoints)),
            "parameter_contract": json.loads(next(iter(parameter_contracts))),
            "missing_declared_image_paths": missing_old_image_paths,
            "missing_source_image_sha_bindings": missing_image_sha_bindings,
            "records": old_records,
        },
        "current_exact_mpr": {
            "authority": {
                "path": str(authority_path),
                "sha256": sha256_file(authority_path),
            },
            "num_gaussians": int(authority["num_gaussians"]),
            "frames": len(current_frames),
            "excluded_frames": authority["metadata"]["excluded_frame_ids"],
            "formula_sha256": authority["formula_sha256"],
            "current_source_images_all_match_manifest_sha256": True,
            "current_source_images": current_image_records,
        },
        "alignment": {
            "frame_axis_exact": frame_axis_exact,
            "shared_frames": len(shared),
            "shared_frame_ids": shared,
            "old_only_frames": len(old_only),
            "old_only_frame_ids": old_only,
            "current_only_frames": len(current_only),
            "current_only_frame_ids": current_only,
            "old_cache_image_hash_binding_complete": image_hash_binding_complete,
        },
        "required_action": (
            "lift old masks through current exact MPR"
            if reusable
            else "regenerate all current-frame official-SAM masks from SHA-bound current source images"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hierarchy-root", required=True)
    parser.add_argument("--exact-mpr-authority", required=True)
    parser.add_argument("--current-feature-manifest", required=True)
    parser.add_argument("--current-image-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"audit output already exists: {args.output}")
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "reusable_for_current_exact_mpr", "reason", "alignment", "required_action")}, indent=2))


if __name__ == "__main__":
    main()
