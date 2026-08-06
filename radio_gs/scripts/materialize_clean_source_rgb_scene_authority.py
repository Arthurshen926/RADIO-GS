#!/usr/bin/env python3
"""Seal clean field RGB frames as an immutable scene authority.

This CPU-only prerequisite binds the query-free field source contract, the
sealed RADIO extraction frame manifest, and the exact source RGB bytes.  It
does not run RADIO and never reads benchmark images, masks, labels, or text.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from PIL import Image

from radio_gs.scripts import extract_radio_features as radio_feature_extraction
from radio_gs.scripts.materialize_official_multiview_siglip2_teacher_authority import (
    UPSTREAM_CHAIN,
    build_source_rgb_scene_authority,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    sha256_file,
    write_frozen_json,
)


def _required_file(path: str | Path, expected_sha256: str, *, label: str) -> Path:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"{label} is missing: {source}. Required chain: {UPSTREAM_CHAIN}"
        )
    if sha256_file(source) != str(expected_sha256):
        raise ValueError(f"{label} SHA-256 differs")
    return source


def _clean_field_contract(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("field source contract must be a mapping")
    contract = dict(value)
    frames = contract.get("selected_frame_indices")
    scene = str(contract.get("scene_id", ""))
    if (
        not scene
        or not isinstance(frames, list)
        or not frames
        or any(not isinstance(frame, int) or frame < 0 for frame in frames)
        or frames != sorted(frames)
        or len(set(frames)) != len(frames)
        or int(contract.get("field_frame_count", -1)) != len(frames)
        or len(str(contract.get("field_frame_manifest_sha256", ""))) != 64
        or int(contract.get("excluded_query_source_frame_count", -1)) != 0
        or contract.get("uses_private_anchor") is not False
        or contract.get("uses_private_depth_pixel") is not False
        or contract.get("uses_instances_or_semantic_labels") is not False
        or contract.get("contains_instance_or_label_directories") is not False
    ):
        raise ValueError("field source contract is not clean/query-free")
    return contract


def _sealed_feature_manifest(
    value: object,
    *,
    scene_id: str,
    selected_frames: list[int],
    manifest_path: str | Path | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("feature frame manifest must be a mapping")
    manifest = dict(value)
    records = manifest.get("frames")
    execution = manifest.get("execution")
    strict_resume = (
        isinstance(execution, Mapping)
        and execution.get("resume_contract")
        == radio_feature_extraction.RESUME_CONTRACT_FILENAME
    )
    if (
        str(manifest.get("scene", "")) != scene_id
        or not isinstance(records, list)
        or int(manifest.get("num_frames", -1)) != len(records)
        or not isinstance(execution, Mapping)
        or manifest.get("excluded_image_names") != []
        or manifest.get("excluded_image_stems") != []
    ):
        raise ValueError("feature frame manifest is not a sealed clean source")
    if strict_resume:
        if execution.get("resume_partial") is not True:
            raise ValueError("strict-resume feature manifest is not complete")
        if manifest_path is None or expected_manifest_sha256 is None:
            raise ValueError("strict-resume feature manifest lacks caller authority")
        path = Path(manifest_path).expanduser().resolve()
        if path.name != "frame_manifest.json":
            raise ValueError("strict-resume feature manifest path differs")
        validation = radio_feature_extraction._validate_final_output_bundle(
            path.parent,
            manifest=manifest,
            verify_source_images=True,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        if int(validation.get("num_frames", -1)) != len(records):
            raise ValueError("strict-resume feature bundle frame count differs")
    elif (
        bool(execution.get("resume_partial"))
        or bool(str(execution.get("resume_contract", "")))
        or bool(str(execution.get("resume_contract_sha256", "")))
        or bool(str(execution.get("resume_contract_file_sha256", "")))
    ):
        raise ValueError("feature frame manifest has unsupported resume provenance")
    elif (
        execution.get("benchmark_masks_opened") is not False
        or execution.get("text_queries_opened") is not False
    ):
        # Preserve the historical resealed-manifest contract.  Native
        # strict-resume outputs instead prove their complete bundle through
        # the extractor's validator above and do not claim target-access
        # fields that the extractor never consumes.
        raise ValueError("feature frame manifest is not a sealed clean source")
    frame_ids: list[int] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("feature frame record must be a mapping")
        required = {"frame_idx", "source_file", "source_sha256"}
        if not required.issubset(record):
            raise ValueError("feature frame record lacks source identity")
        frame = int(record["frame_idx"])
        relative = Path(str(record["source_file"]))
        stem = relative.stem
        if (
            frame < 0
            or relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or not stem.isdecimal()
            or stem != f"{frame:0{len(stem)}d}"
            or len(str(record["source_sha256"])) != 64
        ):
            raise ValueError("feature frame source identity/path differs")
        frame_ids.append(frame)
    if frame_ids != selected_frames:
        raise ValueError("field contract and feature frame manifest differ")
    return manifest


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    field_path = _required_file(
        args.field_source_contract,
        args.expected_field_source_contract_sha256,
        label="field source contract",
    )
    feature_path = _required_file(
        args.feature_frame_manifest,
        args.expected_feature_frame_manifest_sha256,
        label="feature frame manifest",
    )
    field_value, _, _ = load_json_object(
        field_path,
        expected_sha256=args.expected_field_source_contract_sha256,
        label="field source contract",
    )
    field = _clean_field_contract(field_value)
    feature_value, _, _ = load_json_object(
        feature_path,
        expected_sha256=args.expected_feature_frame_manifest_sha256,
        label="feature frame manifest",
    )
    feature = _sealed_feature_manifest(
        feature_value,
        scene_id=str(field["scene_id"]),
        selected_frames=list(field["selected_frame_indices"]),
        manifest_path=feature_path,
        expected_manifest_sha256=args.expected_feature_frame_manifest_sha256,
    )
    root = Path(args.source_rgb_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"source RGB root is missing: {root}")
    execution = feature["execution"]
    if (
        execution.get("resume_contract")
        == radio_feature_extraction.RESUME_CONTRACT_FILENAME
        and Path(str(feature.get("image_dir", ""))).expanduser().resolve() != root
    ):
        raise ValueError("strict-resume feature image directory differs from source RGB root")
    records: list[dict[str, Any]] = []
    for raw in feature["frames"]:
        relative = Path(str(raw["source_file"]))
        image_path = (root / relative).resolve()
        if root not in image_path.parents or not image_path.is_file():
            raise FileNotFoundError(f"source RGB frame is missing/unsafe: {image_path}")
        if sha256_file(image_path) != str(raw["source_sha256"]):
            raise ValueError("source RGB frame SHA-256 differs")
        with Image.open(image_path) as image:
            width, height = image.size
            image.verify()
        records.append(
            {
                "frame_id": relative.stem,
                "source_relative_path": relative.as_posix(),
                "source_image_sha256": str(raw["source_sha256"]),
                "source_image_height": int(height),
                "source_image_width": int(width),
            }
        )
    authority = build_source_rgb_scene_authority(
        scene_id=str(field["scene_id"]),
        field_source_contract_file_sha256=sha256_file(field_path),
        field_frame_manifest_sha256=str(field["field_frame_manifest_sha256"]),
        feature_frame_manifest_file_sha256=sha256_file(feature_path),
        frame_records=records,
    )
    return {
        "field_path": field_path,
        "feature_path": feature_path,
        "source_root": root,
        "authority": authority,
    }


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if not bool(args.preflight_only) and (output.exists() or output.is_symlink()):
        raise FileExistsError(f"refuses to clobber source RGB authority: {output}")
    prepared = preflight(args)
    authority = prepared["authority"]
    result = {
        "status": "ready" if bool(args.preflight_only) else "materialized",
        "scene_id": authority["scene_id"],
        "frames": len(authority["frame_records"]),
        "authority_sha256": authority["authority_sha256"],
        "outputs_written": False,
    }
    if bool(args.preflight_only):
        return result
    write_frozen_json(output, authority)
    return {**result, "output": file_record(output), "outputs_written": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-source-contract", required=True)
    parser.add_argument("--expected-field-source-contract-sha256", required=True)
    parser.add_argument("--feature-frame-manifest", required=True)
    parser.add_argument("--expected-feature-frame-manifest-sha256", required=True)
    parser.add_argument("--source-rgb-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    print(json.dumps(materialize(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
