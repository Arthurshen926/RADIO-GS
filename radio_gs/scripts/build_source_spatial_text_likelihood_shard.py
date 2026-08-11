#!/usr/bin/env python3
"""Build an immutable source-fit spatial text-likelihood shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from radio_gs.querying.source_spatial_text_likelihood import (
    FIXED_NEIGHBOR_COUNT,
    SOURCE_SPATIAL_SHARD_SCHEMA,
    fixed_knn_indices,
    fixed_spatial_logit_statistics,
    sha256_file,
    tensor_sha256,
    validate_source_spatial_shard,
)
from radio_gs.querying.source_text_query_likelihood import (
    validate_source_text_training_shard,
)


RECEIPT_SCHEMA = "radio_gs.source_spatial_text_likelihood_shard_receipt.v1"


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


def canonical_region_xyz(
    accepted: Mapping[str, Any], support_graph: Mapping[str, Any]
) -> torch.Tensor:
    region_rows = torch.as_tensor(accepted.get("region_rows")).long()
    token_mask = torch.as_tensor(accepted.get("token_mask"))
    global_rows = torch.as_tensor(support_graph.get("global_rows")).long()
    xyz = torch.as_tensor(support_graph.get("xyz")).float()
    num_global_rows = int(support_graph.get("num_global_rows", -1))
    if (
        region_rows.ndim != 2
        or token_mask.shape != region_rows.shape
        or token_mask.dtype != torch.bool
        or global_rows.ndim != 1
        or xyz.shape != (global_rows.numel(), 3)
        or num_global_rows <= int(global_rows.max())
        or bool(((region_rows < 0) & token_mask).any())
        or bool(((region_rows >= num_global_rows) & token_mask).any())
    ):
        raise ValueError("MPR region membership and support graph axes differ")
    mapping = torch.full((num_global_rows,), -1, dtype=torch.long)
    mapping[global_rows] = torch.arange(global_rows.numel(), dtype=torch.long)
    local = mapping[region_rows.clamp(0, num_global_rows - 1)]
    keep = token_mask & (local >= 0)
    counts = keep.sum(dim=1, keepdim=True)
    if bool((counts == 0).any()):
        raise ValueError("canonical region lacks support-graph geometry")
    gathered = xyz[local.clamp_min(0)]
    centroid = (gathered * keep.unsqueeze(-1)).sum(dim=1) / counts
    if not bool(torch.isfinite(centroid).all()):
        raise ValueError("canonical region centroid is nonfinite")
    return centroid.float().contiguous()


def build_spatial_shard(
    *,
    source_text_training_shard: str | Path,
    accepted_region_authority: str | Path,
    query_independent_support_graph: str | Path,
) -> dict[str, Any]:
    paths = {
        "source_text_training_shard": Path(source_text_training_shard)
        .expanduser()
        .resolve(strict=True),
        "accepted_region_authority": Path(accepted_region_authority)
        .expanduser()
        .resolve(strict=True),
        "query_independent_support_graph": Path(query_independent_support_graph)
        .expanduser()
        .resolve(strict=True),
    }
    base = validate_source_text_training_shard(
        torch.load(paths["source_text_training_shard"], map_location="cpu", weights_only=False)
    )
    accepted = torch.load(
        paths["accepted_region_authority"], map_location="cpu", weights_only=False
    )
    graph = torch.load(
        paths["query_independent_support_graph"], map_location="cpu", weights_only=False
    )
    scene_id = str(base["scene_id"])
    if (
        accepted.get("scene_id") != scene_id
        or accepted.get("schema")
        != "radio_gs.surface_region_accepted_v2_canonical_region_authority.v2"
        or graph.get("metadata", {}).get("benchmark_masks_opened") is not False
        or graph.get("metadata", {}).get("text_queries_opened") is not False
    ):
        raise ValueError("source MPR/graph authority differs or crossed query boundary")
    if int(base["scale_count"]) != 1:
        raise ValueError("current source MPR shard must expose one canonical scale per row")
    rows = int(base["valid"].numel())
    if torch.as_tensor(accepted.get("accepted_v2_e0")).shape[0] != rows:
        raise ValueError("source text and MPR row axes differ")
    region_xyz = canonical_region_xyz(accepted, graph)
    neighbors = fixed_knn_indices(region_xyz)
    raw_logit = torch.logit(
        torch.as_tensor(base["field_prior_probability"])
        .float()
        .clamp(1.0e-6, 1.0 - 1.0e-6)
    )[:, None, :].contiguous()
    mean, maximum, contrast = fixed_spatial_logit_statistics(
        raw_logit,
        neighbors,
        valid=base["valid"],
    )
    tensors = {
        "raw_logit": raw_logit,
        "neighbor_mean_logit": mean,
        "neighbor_max_logit": maximum,
        "neighbor_contrast_logit": contrast,
        "region_xyz": region_xyz,
        "neighbor_indices": neighbors,
        "semantic_class_distribution": torch.as_tensor(
            base["semantic_class_distribution"]
        ).float().contiguous(),
        "valid": torch.as_tensor(base["valid"]).bool().contiguous(),
        "coverage": torch.as_tensor(base["coverage"]).float().contiguous(),
        "reliability": torch.as_tensor(base["reliability"]).float().contiguous(),
        "training_label_weight": torch.as_tensor(
            base["training_label_weight"]
        ).float().contiguous(),
    }
    payload = {
        "schema": SOURCE_SPATIAL_SHARD_SCHEMA,
        "schema_version": 1,
        "scene_id": scene_id,
        "physical_space_id": str(base["physical_space_id"]),
        "partition": "source_train",
        "neighbor_count": FIXED_NEIGHBOR_COUNT,
        "scale_count": 1,
        "canonical_region_scale_indices": torch.as_tensor(
            accepted.get("scale_indices")
        ).long().contiguous(),
        "class_ids": list(base["class_ids"]),
        "class_names": list(base["class_names"]),
        **tensors,
        "channel_sha256": {
            name: tensor_sha256(tensor) for name, tensor in tensors.items()
        },
        "lineage": {name: _record(path) for name, path in paths.items()},
        "source_access": {
            "official_scannet_train_scene": True,
            "source_train_semantic_labels_opened": True,
            "development_labels_opened": False,
            "test_labels_opened": False,
            "lerf_queries_or_ground_truth_opened": False,
            "benchmark_predictions_or_metrics_opened": False,
            "per_scene_or_per_query_metric_tuning": False,
        },
    }
    validate_source_spatial_shard(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-text-training-shard", required=True)
    parser.add_argument("--accepted-region-authority", required=True)
    parser.add_argument("--query-independent-support-graph", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    payload = build_spatial_shard(
        source_text_training_shard=args.source_text_training_shard,
        accepted_region_authority=args.accepted_region_authority,
        query_independent_support_graph=args.query_independent_support_graph,
    )
    output = _write_torch_noclobber(args.output, payload)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "schema_version": 1,
        "status": "complete_source_fit_query_independent_spatial_shard",
        "scene_id": payload["scene_id"],
        "spatial_shard": {"path": str(output), "sha256": sha256_file(output)},
        "row_count": int(payload["valid"].numel()),
        "valid_row_count": int(payload["valid"].sum()),
        "neighbor_count": FIXED_NEIGHBOR_COUNT,
        "lineage": dict(payload["lineage"]),
        "source_access": dict(payload["source_access"]),
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    receipt_path = _write_json_noclobber(args.receipt, receipt)
    print(json.dumps({"receipt": str(receipt_path), **receipt}, sort_keys=True))


if __name__ == "__main__":
    main()
