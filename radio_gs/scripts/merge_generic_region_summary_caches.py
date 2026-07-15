#!/usr/bin/env python3
"""Merge deterministic shards of one generic region-summary teacher cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


_TENSOR_KEYS = (
    "radio_region_tokens",
    "official_summary_tokens",
    "official_crop_summaries",
)


def _validate_disjoint_shard_images(metadata: list[dict]) -> list[set[str]]:
    image_sets: list[set[str]] = []
    for value in metadata:
        records = value.get("crop_records", [])
        if not isinstance(records, list) or not records:
            raise ValueError("every shard must contain crop_records")
        images = {str(record.get("image", "")) for record in records}
        if "" in images:
            raise ValueError("every crop record must identify its source image")
        if len(images) != int(value.get("num_images", -1)):
            raise ValueError("shard num_images disagrees with its crop records")
        if len(records) != int(value.get("num_crops", -1)):
            raise ValueError("shard num_crops disagrees with its crop records")
        image_sets.append(images)
    for index, left in enumerate(image_sets):
        for right in image_sets[index + 1 :]:
            if not left.isdisjoint(right):
                raise ValueError("merged shards contain duplicate source images")
    return image_sets


def merge(args: argparse.Namespace) -> dict:
    input_paths = [Path(value) for value in args.inputs]
    if not input_paths:
        raise ValueError("at least one shard cache is required")
    payloads = [torch.load(path, map_location="cpu") for path in input_paths]
    required = {*_TENSOR_KEYS, "metadata"}
    if any(not isinstance(value, dict) or not required.issubset(value) for value in payloads):
        raise ValueError("every shard must be a generic region-summary cache")

    metadata = [dict(value["metadata"]) for value in payloads]
    selection_hashes = {
        str(value.get("global_selection_manifest_sha256", "")) for value in metadata
    }
    if len(selection_hashes) != 1 or not next(iter(selection_hashes)):
        raise ValueError("shards do not share one deterministic global selection")
    shard_counts = {int(value.get("shard_count", -1)) for value in metadata}
    if len(shard_counts) != 1:
        raise ValueError("shards disagree on shard_count")
    shard_count = next(iter(shard_counts))
    shard_indices = sorted(int(value.get("shard_index", -1)) for value in metadata)
    if shard_indices != list(range(shard_count)) or len(payloads) != shard_count:
        raise ValueError("a complete, unique shard set is required")

    contract_keys = (
        "training_scope",
        "uses_benchmark_test_vocabulary",
        "uses_benchmark_scenes",
        "annotations_opened",
        "text_opened",
        "radio_version",
        "radio_checkpoint_sha256",
        "region_token_grid",
        "region_token_dim",
        "summary_token_dim",
        "official_descriptor_dim",
        "official_summary_head_used_for_target",
        "custom_text_projection",
        "source_token_context",
        "crop_scales",
        "crop_scale_policy",
        "source_region_sampling",
    )
    for key in contract_keys:
        values = [value.get(key) for value in metadata]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"shards disagree on {key}")
    if metadata[0].get("uses_benchmark_test_vocabulary") is not False:
        raise ValueError("merged bridge data cannot use benchmark vocabulary")
    if metadata[0].get("uses_benchmark_scenes") is not False:
        raise ValueError("merged bridge data cannot use benchmark scenes")

    _validate_disjoint_shard_images(metadata)
    records = [record for value in metadata for record in value.get("crop_records", [])]
    tensors = {
        key: torch.cat([torch.as_tensor(value[key]) for value in payloads], dim=0)
        for key in _TENSOR_KEYS
    }
    row_counts = {int(value.shape[0]) for value in tensors.values()}
    if len(row_counts) != 1:
        raise ValueError("merged teacher tensors do not align")

    manifest_payload = {
        "dataset_id": str(args.dataset_id),
        "global_selection_manifest_sha256": next(iter(selection_hashes)),
        "shard_manifests": sorted(
            str(value.get("dataset_manifest_sha256", "")) for value in metadata
        ),
        "num_crops": next(iter(row_counts)),
    }
    manifest_hash = hashlib.sha256(
        json.dumps(manifest_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    merged_metadata = {
        **metadata[0],
        "dataset_id": str(args.dataset_id),
        "dataset_manifest_sha256": manifest_hash,
        "num_images": sum(int(value.get("num_images", 0)) for value in metadata),
        "num_crops": next(iter(row_counts)),
        "crop_records": records,
        "shard_count": shard_count,
        "shard_index": "merged",
        "source_shards": [str(path.resolve()) for path in input_paths],
        "source_shard_manifest_sha256": manifest_payload["shard_manifests"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**tensors, "metadata": merged_metadata}, output)
    report = {
        "output": str(output),
        "dataset_id": str(args.dataset_id),
        "num_images": merged_metadata["num_images"],
        "num_crops": merged_metadata["num_crops"],
        "region_tokens": list(tensors["radio_region_tokens"].shape),
        "summary_tokens": list(tensors["official_summary_tokens"].shape),
        "official_crop_summaries": list(tensors["official_crop_summaries"].shape),
        "dataset_manifest_sha256": manifest_hash,
        "source_shards": merged_metadata["source_shards"],
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(merge(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
