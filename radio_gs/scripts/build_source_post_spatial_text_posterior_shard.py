#!/usr/bin/env python3
"""Build a source-only post-spatial TextPosteriorV2 training shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from radio_gs.interfaces.factorized_primitive_state import (
    load_factorized_primitive_state,
)
from radio_gs.querying.source_post_spatial_text_posterior import (
    EXTENT_FEATURE_NAMES,
    SOURCE_POST_SPATIAL_TEXT_SHARD_SCHEMA,
    aggregate_region_reliability,
    build_post_spatial_channels,
    validate_source_post_spatial_shard,
)
from radio_gs.querying.source_spatial_text_likelihood import (
    sha256_file,
    tensor_sha256,
    validate_source_spatial_shard,
)
from radio_gs.querying.source_text_query_likelihood import (
    validate_source_text_training_shard,
)


RECEIPT_SCHEMA = "radio_gs.source_post_spatial_text_posterior_shard_receipt.v1"


def _record(path: str | Path) -> dict[str, str]:
    source = Path(path).expanduser().resolve(strict=True)
    return {"path": str(source), "sha256": sha256_file(source)}


def _write_torch_noclobber(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    torch.save(dict(payload), output)
    return output


def _write_json_noclobber(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    return output


def build_shard(
    *,
    source_spatial_shard: str | Path,
    factorized_primitive_state: str | Path,
    expected_factorized_state_sha256: str,
) -> dict[str, Any]:
    spatial_path = Path(source_spatial_shard).expanduser().resolve(strict=True)
    spatial = validate_source_spatial_shard(
        torch.load(spatial_path, map_location="cpu", weights_only=False)
    )
    text_path = Path(
        spatial["lineage"]["source_text_training_shard"]["path"]
    ).resolve(strict=True)
    accepted_path = Path(
        spatial["lineage"]["accepted_region_authority"]["path"]
    ).resolve(strict=True)
    text = validate_source_text_training_shard(
        torch.load(text_path, map_location="cpu", weights_only=False)
    )
    accepted = torch.load(accepted_path, map_location="cpu", weights_only=False)
    state = load_factorized_primitive_state(
        factorized_primitive_state,
        expected_sha256=expected_factorized_state_sha256,
    )
    if (
        text["scene_id"] != spatial["scene_id"]
        or accepted.get("scene_id") != spatial["scene_id"]
        or accepted.get("schema")
        != "radio_gs.surface_region_accepted_v2_canonical_region_authority.v2"
        or state.metadata.get("query_independent") is not True
        or state.metadata.get("benchmark_masks_opened") is not False
        or state.metadata.get("text_queries_opened") is not False
    ):
        raise ValueError("post-spatial source authorities differ")
    reliability = aggregate_region_reliability(
        state, accepted, region_valid=spatial["valid"]
    )
    base, raw, extent = build_post_spatial_channels(
        text["positive_affinity"],
        spatial["neighbor_indices"],
        spatial["region_xyz"],
        valid=spatial["valid"],
    )
    tensors = {
        "base_probability": base,
        "raw_positive_cosine": raw,
        "extent_features": extent,
        "semantic_class_distribution": spatial[
            "semantic_class_distribution"
        ].float().contiguous(),
        "reliability": reliability,
        "valid": spatial["valid"].bool().contiguous(),
        "training_label_weight": spatial["training_label_weight"].float().contiguous(),
        "coverage": spatial["coverage"].float().contiguous(),
        "neighbor_indices": spatial["neighbor_indices"].long().contiguous(),
    }
    payload = {
        "schema": SOURCE_POST_SPATIAL_TEXT_SHARD_SCHEMA,
        "schema_version": 1,
        "scene_id": spatial["scene_id"],
        "physical_space_id": spatial["physical_space_id"],
        "partition": "source_train",
        "neighbor_count": 10,
        "spatial_base": "positive_cosine_knn10_per_query_scene_minmax_clip_2u_minus_1",
        "extent_feature_names": list(EXTENT_FEATURE_NAMES),
        "class_ids": list(spatial["class_ids"]),
        "class_names": list(spatial["class_names"]),
        **tensors,
        "channel_sha256": {
            name: tensor_sha256(tensor) for name, tensor in tensors.items()
        },
        "lineage": {
            "source_spatial_shard": _record(spatial_path),
            "source_text_training_shard": _record(text_path),
            "accepted_region_authority": _record(accepted_path),
            "factorized_primitive_state": _record(factorized_primitive_state),
        },
        "source_access": dict(spatial["source_access"]),
    }
    validate_source_post_spatial_shard(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-spatial-shard", required=True)
    parser.add_argument("--factorized-primitive-state", required=True)
    parser.add_argument("--expected-factorized-state-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    payload = build_shard(
        source_spatial_shard=args.source_spatial_shard,
        factorized_primitive_state=args.factorized_primitive_state,
        expected_factorized_state_sha256=args.expected_factorized_state_sha256,
    )
    output = _write_torch_noclobber(args.output, payload)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "schema_version": 1,
        "status": "complete_source_only_post_spatial_text_posterior_shard",
        "scene_id": payload["scene_id"],
        "shard": {"path": str(output), "sha256": sha256_file(output)},
        "row_count": int(payload["valid"].numel()),
        "valid_row_count": int(payload["valid"].sum()),
        "extent_feature_names": list(payload["extent_feature_names"]),
        "lineage": dict(payload["lineage"]),
        "source_access": dict(payload["source_access"]),
    }
    receipt_path = _write_json_noclobber(args.receipt, receipt)
    print(json.dumps({"receipt": str(receipt_path), **receipt}, sort_keys=True))


if __name__ == "__main__":
    main()
