#!/usr/bin/env python3
"""Bind legal LERF source RGBs to an exact-MPR-compatible SAM authority.

This builder intersects the source-only feature manifest with a sealed exact
marginal responsibility authority.  It never accepts arbitrary image paths,
queries, benchmark masks, or evaluation RGB.  Optional subsampling is uniform
over the ordered compatible source cohort so a small development closure keeps
camera coverage without becoming query dependent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, *, label: str) -> dict:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be one regular JSON file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain one JSON object")
    return dict(payload)


def uniform_indices(length: int, maximum: int) -> tuple[int, ...]:
    """Return deterministic edge-preserving uniform indices without duplicates."""

    length, maximum = int(length), int(maximum)
    if length <= 0:
        raise ValueError("compatible source cohort is empty")
    if maximum <= 0 or maximum >= length:
        return tuple(range(length))
    if maximum == 1:
        return (length // 2,)
    indices = tuple(
        ((rank * (length - 1)) + ((maximum - 1) // 2)) // (maximum - 1)
        for rank in range(maximum)
    )
    if len(set(indices)) != maximum:
        raise AssertionError("uniform subset unexpectedly contains duplicates")
    return indices


def build_authority(
    *,
    scene: str,
    frame_manifest_path: Path,
    exact_mpr_authority_path: Path,
    maximum_images: int,
) -> dict:
    manifest = _load_json_object(frame_manifest_path, label="source frame manifest")
    exact = _load_json_object(exact_mpr_authority_path, label="exact-MPR authority")
    if str(manifest.get("scene", "")) != str(scene):
        raise ValueError("source frame manifest scene differs")
    if (
        exact.get("schema")
        != "radio_gs.sparse_exact_marginal_responsibility_authority.v1"
        or exact.get("schema_version") != 1
    ):
        raise ValueError("exact-MPR authority schema differs")
    metadata = exact.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("exact-MPR metadata is absent")
    forbidden_flags = (
        "benchmark_images_opened",
        "benchmark_masks_opened",
        "text_queries_opened",
    )
    if any(bool(metadata.get(flag, True)) for flag in forbidden_flags):
        raise ValueError("exact-MPR authority opened forbidden benchmark information")
    if metadata.get("assignment_mode") != "exact_front_to_back_sparse_marginal":
        raise ValueError("responsibility assignment is not exact front-to-back marginal")

    image_dir = Path(str(manifest.get("image_dir", ""))).expanduser().resolve()
    if not image_dir.is_dir():
        raise ValueError(f"source image directory is absent: {image_dir}")
    raw_frames = manifest.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise ValueError("source frame manifest has no frames")
    excluded_names = {str(value) for value in manifest.get("excluded_image_names", [])}
    excluded_stems = {str(value) for value in manifest.get("excluded_image_stems", [])}
    by_index: dict[int, dict] = {}
    for raw in raw_frames:
        if not isinstance(raw, Mapping):
            raise ValueError("source frame record is not one object")
        frame_index = int(raw.get("frame_idx", -1))
        source_file = str(raw.get("source_file", ""))
        source_sha = str(raw.get("source_sha256", ""))
        if (
            frame_index < 0
            or frame_index in by_index
            or not source_file
            or source_file in excluded_names
            or Path(source_file).stem in excluded_stems
            or len(source_sha) != 64
        ):
            raise ValueError("source frame identity or exclusion contract differs")
        by_index[frame_index] = dict(raw)

    raw_exact_indices = exact.get("frame_indices")
    views = exact.get("views")
    if not isinstance(raw_exact_indices, list) or not isinstance(views, list):
        raise ValueError("exact-MPR authority has no frame/view index")
    exact_indices = tuple(int(value) for value in raw_exact_indices)
    view_indices = tuple(int(view.get("frame_index", -1)) for view in views if isinstance(view, Mapping))
    if len(view_indices) != len(views) or exact_indices != view_indices or len(set(exact_indices)) != len(exact_indices):
        raise ValueError("exact-MPR frame and view indices differ")
    compatible = [frame_index for frame_index in exact_indices if frame_index in by_index]
    if len(compatible) != len(exact_indices):
        missing = sorted(set(exact_indices) - set(compatible))
        raise ValueError(f"exact-MPR views are absent from source-only manifest: {missing}")
    selected = [compatible[index] for index in uniform_indices(len(compatible), maximum_images)]

    images: list[dict[str, str]] = []
    for frame_index in selected:
        record = by_index[frame_index]
        image_path = (image_dir / str(record["source_file"])).resolve()
        if image_path.parent != image_dir or not image_path.is_file() or image_path.is_symlink():
            raise ValueError(f"source image escaped or is absent: {image_path}")
        actual_sha = sha256_file(image_path)
        if actual_sha != str(record["source_sha256"]):
            raise ValueError(f"source image SHA-256 differs for frame {frame_index}")
        images.append(
            {
                "image_id": f"frame_{frame_index:05d}",
                "path": str(image_path),
                "sha256": actual_sha,
                "rgb_role": "registered_source_or_mapping_view",
            }
        )

    return {
        "schema_version": 1,
        "contract": "sam3-query-free-source-rgb-authority-v1",
        "scene": str(scene),
        "cohort": "lerf-source-only-exact-mpr-compatible",
        "information_policy": {
            "registered_source_rgb_only": True,
            "query_text_used": False,
            "benchmark_ground_truth_used": False,
            "target_or_evaluation_rgb_used": False,
        },
        "construction": {
            "contract": "lerf-source-only-exact-mpr-intersection-uniform-v1",
            "frame_manifest": {
                "path": str(frame_manifest_path),
                "sha256": sha256_file(frame_manifest_path),
                "source_only_count": len(raw_frames),
            },
            "exact_mpr_authority": {
                "path": str(exact_mpr_authority_path),
                "sha256": sha256_file(exact_mpr_authority_path),
                "compatible_count": len(compatible),
            },
            "selection": {
                "type": "uniform_ordered_compatible_cohort_edge_preserving",
                "maximum_images": int(maximum_images),
                "selected_count": len(selected),
                "selected_frame_indices": selected,
            },
        },
        "images": images,
    }


def atomic_json_write(payload: object, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
    )
    temporary = Path(handle.name)
    try:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()
    try:
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output authority exists: {output}")
    payload = build_authority(
        scene=args.scene,
        frame_manifest_path=Path(args.frame_manifest).expanduser().resolve(),
        exact_mpr_authority_path=Path(args.exact_mpr_authority).expanduser().resolve(),
        maximum_images=int(args.maximum_images),
    )
    atomic_json_write(payload, output)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--frame-manifest", required=True)
    parser.add_argument("--exact-mpr-authority", required=True)
    parser.add_argument("--maximum-images", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
