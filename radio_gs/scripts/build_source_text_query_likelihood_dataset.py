#!/usr/bin/env python3
"""Build and seal source-only ScanNet text-likelihood training shards.

The input is a pre-aligned scene bundle.  Producing that bundle is an explicit
data-adapter step because an RGB/field row cannot be assigned a semantic label
without a separately audited ScanNet coordinate/label authority.  This script
never guesses that alignment and never opens evaluation scenes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from radio_gs.querying.source_text_query_likelihood import (
    SOURCE_TEXT_DATASET_MANIFEST_SCHEMA,
    SOURCE_TEXT_TRAINING_SHARD_SCHEMA,
    build_source_text_training_shard,
    sha256_file,
    source_text_likelihood_contract,
    validate_source_text_training_shard,
)


RECEIPT_SCHEMA = "radio_gs.source_text_query_likelihood_training_shard_receipt.v1"


def _canonical_output(path: str | Path) -> Path:
    raw = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    raw.parent.mkdir(parents=True, exist_ok=True)
    return raw.parent.resolve(strict=True) / raw.name


def _write_json_noclobber(path: str | Path, value: Mapping[str, Any]) -> Path:
    output = _canonical_output(path)
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, output)
    except FileExistsError:
        if output.read_bytes() != encoded:
            raise ValueError(f"refusing to replace different artifact: {output}")
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _write_torch_noclobber(path: str | Path, value: Mapping[str, Any]) -> Path:
    output = _canonical_output(path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable output already exists: {output}")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(value), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def build_scene_shard(
    *,
    scene_input: str | Path,
    output_shard: str | Path,
    receipt: str | Path,
) -> tuple[Path, Path, dict[str, Any]]:
    source_path = Path(scene_input).expanduser().resolve(strict=True)
    source_sha = sha256_file(source_path)
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    if not isinstance(source, Mapping):
        raise ValueError("source text scene input must be a torch mapping")
    shard = build_source_text_training_shard(source)
    output = _write_torch_noclobber(output_shard, shard)
    output_sha = sha256_file(output)
    receipt_payload = {
        "schema": RECEIPT_SCHEMA,
        "schema_version": 1,
        "status": "complete_source_train_text_likelihood_shard",
        "scene_id": shard["scene_id"],
        "physical_space_id": shard["physical_space_id"],
        "partition": shard["partition"],
        "scene_input": {"path": str(source_path), "sha256": source_sha},
        "shard": {"path": str(output), "sha256": output_sha},
        "row_count": int(shard["positive_affinity"].shape[0]),
        "scale_count": int(shard["scale_count"]),
        "class_count": len(shard["class_ids"]),
        "present_class_ids": list(shard["present_class_ids"]),
        "source_access": dict(shard["source_access"]),
    }
    receipt_path = _write_json_noclobber(receipt, receipt_payload)
    return output, receipt_path, receipt_payload


def seal_dataset(
    *,
    shards: Sequence[str | Path],
    output: str | Path,
) -> tuple[Path, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    scenes: set[str] = set()
    spaces: set[str] = set()
    vocabulary: tuple[tuple[int, str], ...] | None = None
    for raw_path in shards:
        shard_path = Path(raw_path).expanduser().resolve(strict=True)
        shard_sha = sha256_file(shard_path)
        payload = validate_source_text_training_shard(
            torch.load(shard_path, map_location="cpu", weights_only=True)
        )
        scene = str(payload["scene_id"])
        space = str(payload["physical_space_id"])
        if scene in scenes or space in spaces:
            raise ValueError("source text dataset repeats a scene or physical space")
        scenes.add(scene)
        spaces.add(space)
        current_vocabulary = tuple(zip(payload["class_ids"], payload["class_names"]))
        if vocabulary is None:
            vocabulary = current_vocabulary
        elif current_vocabulary != vocabulary:
            raise ValueError("source text shards use different class vocabulary/order")
        records.append(
            {
                "scene_id": scene,
                "physical_space_id": space,
                "partition": "source_train",
                "shard": {"path": str(shard_path), "sha256": shard_sha},
                "row_count": int(payload["positive_affinity"].shape[0]),
                "scale_count": int(payload["scale_count"]),
                "present_class_ids": list(payload["present_class_ids"]),
                "source_access": dict(payload["source_access"]),
            }
        )
    if not records or vocabulary is None:
        raise ValueError("cannot seal an empty source text dataset")
    manifest = {
        "schema": SOURCE_TEXT_DATASET_MANIFEST_SCHEMA,
        "schema_version": 1,
        "status": "sealed_ready_for_source_train_only_text_calibration",
        "contract": source_text_likelihood_contract(),
        "class_ids": [class_id for class_id, _name in vocabulary],
        "class_names": [name for _class_id, name in vocabulary],
        "scene_count": len(records),
        "records": sorted(records, key=lambda value: value["scene_id"]),
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "source_access": {
            "official_scannet_train_scenes_only": True,
            "source_train_semantic_labels_opened": True,
            "development_labels_opened": False,
            "test_labels_opened": False,
            "lerf_queries_or_ground_truth_opened": False,
            "target_rgb_or_mask_opened": False,
            "benchmark_predictions_or_metrics_opened": False,
            "full_benchmark_evaluation_authorized": False,
        },
    }
    manifest_path = _write_json_noclobber(output, manifest)
    return manifest_path, manifest


def validate_dataset_manifest(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = Path(path).expanduser().resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SOURCE_TEXT_DATASET_MANIFEST_SCHEMA:
        raise ValueError("unexpected source text dataset manifest schema")
    if manifest.get("schema_version") != 1:
        raise ValueError("unexpected source text dataset manifest schema_version")
    if manifest.get("contract") != source_text_likelihood_contract():
        raise ValueError("source text dataset contract differs")
    source_access = manifest.get("source_access", {})
    required = {
        "official_scannet_train_scenes_only": True,
        "source_train_semantic_labels_opened": True,
        "development_labels_opened": False,
        "test_labels_opened": False,
        "lerf_queries_or_ground_truth_opened": False,
        "target_rgb_or_mask_opened": False,
        "benchmark_predictions_or_metrics_opened": False,
        "full_benchmark_evaluation_authorized": False,
    }
    for key, expected in required.items():
        if source_access.get(key) is not expected:
            raise PermissionError(f"source text dataset manifest violates {key}")
    payloads = []
    seen_scenes: set[str] = set()
    seen_spaces: set[str] = set()
    for record in manifest.get("records", []):
        if record.get("partition") != "source_train":
            raise PermissionError("source text training record is not source_train")
        scene = str(record.get("scene_id", ""))
        space = str(record.get("physical_space_id", ""))
        if not scene or not space or scene in seen_scenes or space in seen_spaces:
            raise ValueError("source text manifest scene authority differs")
        seen_scenes.add(scene)
        seen_spaces.add(space)
        shard = record.get("shard")
        if not isinstance(shard, Mapping):
            raise ValueError("source text manifest lacks a shard record")
        shard_path = Path(str(shard.get("path", ""))).expanduser().resolve(strict=True)
        if sha256_file(shard_path) != shard.get("sha256"):
            raise ValueError("sealed source text shard changed")
        payload = validate_source_text_training_shard(
            torch.load(shard_path, map_location="cpu", weights_only=True)
        )
        if payload["scene_id"] != scene or payload["physical_space_id"] != space:
            raise ValueError("source text manifest/shard identity differs")
        payloads.append(payload)
    if len(payloads) != manifest.get("scene_count") or not payloads:
        raise ValueError("source text dataset scene count differs")
    return manifest, payloads


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-shard")
    build.add_argument("--scene-input", required=True)
    build.add_argument("--output-shard", required=True)
    build.add_argument("--receipt", required=True)
    seal = commands.add_parser("seal")
    seal.add_argument("--shard", action="append", required=True)
    seal.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "build-shard":
        shard, receipt, payload = build_scene_shard(
            scene_input=args.scene_input,
            output_shard=args.output_shard,
            receipt=args.receipt,
        )
        print(json.dumps({"shard": str(shard), "receipt": str(receipt), **payload}))
    else:
        path, manifest = seal_dataset(shards=args.shard, output=args.output)
        print(json.dumps({"manifest": str(path), "scene_count": manifest["scene_count"]}))


if __name__ == "__main__":
    main()

