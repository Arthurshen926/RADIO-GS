#!/usr/bin/env python3
"""Lift typed MPR primitive edges into an immutable source-fit region shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from radio_gs.querying.source_spatial_text_likelihood import sha256_file, tensor_sha256
from radio_gs.querying.source_text_query_likelihood import (
    validate_source_text_training_shard,
)
from radio_gs.querying.source_typed_spatial_text_likelihood import (
    FROZEN_EDGE_TYPES,
    SOURCE_TYPED_SPATIAL_SHARD_SCHEMA,
    normalize_typed_region_edges,
    typed_spatial_logit_statistics,
    validate_source_typed_spatial_shard,
)


RECEIPT_SCHEMA = "radio_gs.source_typed_spatial_likelihood_shard_receipt.v1"


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


def aggregate_typed_primitive_edges_to_regions(
    accepted: Mapping[str, Any],
    support_graph: Mapping[str, Any],
    *,
    edge_types: tuple[str, ...] = FROZEN_EDGE_TYPES,
) -> dict[str, dict[str, torch.Tensor]]:
    """Compute same-scale fractional-membership ``M.T @ A_type @ M``.

    A primitive that belongs to ``d`` regions at a scale contributes ``1/d``
    to each membership.  This retains overlapping MPR support without allowing
    high-overlap primitives to have disproportionate mass.
    """

    from scipy import sparse

    region_rows = torch.as_tensor(accepted.get("region_rows")).long().cpu()
    token_mask = torch.as_tensor(accepted.get("token_mask")).cpu()
    scale_indices = torch.as_tensor(accepted.get("scale_indices")).long().cpu()
    global_rows = torch.as_tensor(support_graph.get("global_rows")).long().cpu()
    primitive_edges = torch.as_tensor(support_graph.get("edge_index")).long().cpu()
    num_global_rows = int(support_graph.get("num_global_rows", -1))
    channels = support_graph.get("edge_channels")
    metadata_types = tuple(sorted(support_graph.get("metadata", {}).get("edge_channels", ())))
    if (
        region_rows.ndim != 2
        or token_mask.shape != region_rows.shape
        or token_mask.dtype != torch.bool
        or scale_indices.shape != (region_rows.shape[0],)
        or global_rows.ndim != 1
        or primitive_edges.ndim != 2
        or primitive_edges.shape[0] != 2
        or num_global_rows <= 0
        or not isinstance(channels, Mapping)
        or tuple(sorted(channels)) != tuple(edge_types)
        or metadata_types != tuple(edge_types)
        or bool(((region_rows < 0) & token_mask).any())
        or bool(((region_rows >= num_global_rows) & token_mask).any())
        or bool((primitive_edges < 0).any())
        or bool((primitive_edges >= global_rows.numel()).any())
    ):
        raise ValueError("typed MPR membership/support graph authority differs")
    for edge_type in edge_types:
        value = torch.as_tensor(channels[edge_type]).float().reshape(-1)
        if value.numel() != primitive_edges.shape[1] or not bool(torch.isfinite(value).all()):
            raise ValueError(f"primitive edge channel differs: {edge_type}")
        if bool((value < 0).any()):
            raise ValueError(f"primitive edge channel is negative: {edge_type}")

    global_to_local = np.full(num_global_rows, -1, dtype=np.int64)
    global_to_local[global_rows.numpy()] = np.arange(global_rows.numel(), dtype=np.int64)
    primitive_edge_numpy = primitive_edges.numpy()
    row_count = int(region_rows.shape[0])
    accumulated: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        edge_type: [] for edge_type in edge_types
    }
    for scale in sorted(int(v) for v in torch.unique(scale_indices).tolist()):
        canonical_rows = torch.nonzero(scale_indices == scale, as_tuple=False).reshape(-1)
        memberships = region_rows[canonical_rows]
        membership_mask = token_mask[canonical_rows]
        local_region = np.repeat(
            np.arange(canonical_rows.numel(), dtype=np.int64),
            membership_mask.sum(dim=1).numpy(),
        )
        primitive_global = memberships[membership_mask].numpy()
        primitive_local = global_to_local[primitive_global]
        present = primitive_local >= 0
        local_region = local_region[present]
        primitive_local = primitive_local[present]
        membership_count = np.bincount(
            primitive_local, minlength=global_rows.numel()
        ).astype(np.float32)
        membership_weight = 1.0 / membership_count[primitive_local]
        membership = sparse.csr_matrix(
            (membership_weight, (primitive_local, local_region)),
            shape=(global_rows.numel(), canonical_rows.numel()),
            dtype=np.float32,
        )
        if membership.nnz == 0:
            raise ValueError("canonical scale has no support-graph membership")
        for edge_type in edge_types:
            primitive_weight = torch.as_tensor(channels[edge_type]).float().numpy()
            primitive_graph = sparse.csr_matrix(
                (
                    primitive_weight,
                    (primitive_edge_numpy[0], primitive_edge_numpy[1]),
                ),
                shape=(global_rows.numel(), global_rows.numel()),
                dtype=np.float32,
            )
            lifted = (membership.T @ primitive_graph @ membership).tocsr()
            lifted.setdiag(0)
            lifted.eliminate_zeros()
            lifted = lifted.tocoo()
            positive = np.isfinite(lifted.data) & (lifted.data > 0)
            accumulated[edge_type].append(
                (
                    canonical_rows[lifted.row[positive]].numpy(),
                    canonical_rows[lifted.col[positive]].numpy(),
                    lifted.data[positive].astype(np.float32, copy=False),
                )
            )
    result: dict[str, dict[str, torch.Tensor]] = {}
    for edge_type in edge_types:
        receiver = np.concatenate([value[0] for value in accumulated[edge_type]])
        neighbor = np.concatenate([value[1] for value in accumulated[edge_type]])
        weight = np.concatenate([value[2] for value in accumulated[edge_type]])
        index, normalized = normalize_typed_region_edges(
            torch.from_numpy(np.stack((receiver, neighbor), axis=0)),
            torch.from_numpy(weight),
            row_count=row_count,
        )
        result[edge_type] = {
            "edge_index": index,
            "edge_weight": normalized,
        }
    return result


def build_typed_spatial_shard(
    *,
    source_text_training_shard: str | Path,
    accepted_region_authority: str | Path,
    query_independent_support_graph: str | Path,
) -> dict[str, Any]:
    paths = {
        "source_text_training_shard": Path(source_text_training_shard).expanduser().resolve(strict=True),
        "accepted_region_authority": Path(accepted_region_authority).expanduser().resolve(strict=True),
        "query_independent_support_graph": Path(query_independent_support_graph).expanduser().resolve(strict=True),
    }
    base = validate_source_text_training_shard(
        torch.load(paths["source_text_training_shard"], map_location="cpu", weights_only=False)
    )
    accepted = torch.load(paths["accepted_region_authority"], map_location="cpu", weights_only=False)
    graph = torch.load(paths["query_independent_support_graph"], map_location="cpu", weights_only=False)
    scene_id = str(base["scene_id"])
    graph_metadata = graph.get("metadata", {})
    if (
        accepted.get("scene_id") != scene_id
        or accepted.get("schema")
        != "radio_gs.surface_region_accepted_v2_canonical_region_authority.v2"
        or graph_metadata.get("benchmark_masks_opened") is not False
        or graph_metadata.get("text_queries_opened") is not False
        or tuple(sorted(graph_metadata.get("edge_channels", ()))) != FROZEN_EDGE_TYPES
        or graph_metadata.get("covisibility_relation", {}).get("mode") != "none"
    ):
        raise ValueError("typed source MPR graph differs or crossed query boundary")
    rows = int(base["valid"].numel())
    if torch.as_tensor(accepted.get("accepted_v2_e0")).shape[0] != rows:
        raise ValueError("source text and typed MPR row axes differ")
    typed_edges = aggregate_typed_primitive_edges_to_regions(accepted, graph)
    raw_logit = torch.logit(
        torch.as_tensor(base["field_prior_probability"])
        .float()
        .clamp(1.0e-6, 1.0 - 1.0e-6)
    )[:, None, :].contiguous()
    typed_statistics = typed_spatial_logit_statistics(
        raw_logit, typed_edges, valid=base["valid"]
    )
    tensors: dict[str, torch.Tensor] = {
        "raw_logit": raw_logit,
        "semantic_class_distribution": torch.as_tensor(base["semantic_class_distribution"]).float().contiguous(),
        "valid": torch.as_tensor(base["valid"]).bool().contiguous(),
        "coverage": torch.as_tensor(base["coverage"]).float().contiguous(),
        "reliability": torch.as_tensor(base["reliability"]).float().contiguous(),
        "training_label_weight": torch.as_tensor(base["training_label_weight"]).float().contiguous(),
        "canonical_region_scale_indices": torch.as_tensor(accepted["scale_indices"]).long().contiguous(),
    }
    for edge_type in FROZEN_EDGE_TYPES:
        tensors[f"{edge_type}.edge_index"] = typed_edges[edge_type]["edge_index"]
        tensors[f"{edge_type}.edge_weight"] = typed_edges[edge_type]["edge_weight"]
        for name, tensor in typed_statistics[edge_type].items():
            tensors[f"{edge_type}.{name}"] = tensor
    payload = {
        "schema": SOURCE_TYPED_SPATIAL_SHARD_SCHEMA,
        "schema_version": 1,
        "scene_id": scene_id,
        "physical_space_id": str(base["physical_space_id"]),
        "partition": "source_train",
        "scale_count": 1,
        "edge_types": list(FROZEN_EDGE_TYPES),
        "primitive_to_region_aggregation": "same_scale_fractional_membership_MtAM",
        "class_ids": list(base["class_ids"]),
        "class_names": list(base["class_names"]),
        "raw_logit": tensors["raw_logit"],
        "semantic_class_distribution": tensors["semantic_class_distribution"],
        "valid": tensors["valid"],
        "coverage": tensors["coverage"],
        "reliability": tensors["reliability"],
        "training_label_weight": tensors["training_label_weight"],
        "canonical_region_scale_indices": tensors["canonical_region_scale_indices"],
        "typed_region_edges": typed_edges,
        "typed_statistics": typed_statistics,
        "channel_sha256": {name: tensor_sha256(tensor) for name, tensor in tensors.items()},
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
    validate_source_typed_spatial_shard(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-text-training-shard", required=True)
    parser.add_argument("--accepted-region-authority", required=True)
    parser.add_argument("--query-independent-support-graph", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    payload = build_typed_spatial_shard(
        source_text_training_shard=args.source_text_training_shard,
        accepted_region_authority=args.accepted_region_authority,
        query_independent_support_graph=args.query_independent_support_graph,
    )
    output = _write_torch_noclobber(args.output, payload)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "schema_version": 1,
        "status": "complete_source_fit_query_independent_typed_mpr_shard",
        "scene_id": payload["scene_id"],
        "typed_spatial_shard": {"path": str(output), "sha256": sha256_file(output)},
        "row_count": int(payload["valid"].numel()),
        "valid_row_count": int(payload["valid"].sum()),
        "edge_types": list(payload["edge_types"]),
        "region_edge_counts": {
            name: int(payload["typed_region_edges"][name]["edge_weight"].numel())
            for name in payload["edge_types"]
        },
        "lineage": dict(payload["lineage"]),
        "source_access": dict(payload["source_access"]),
        "implementation": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
    }
    receipt_path = _write_json_noclobber(args.receipt, receipt)
    print(json.dumps({"receipt": str(receipt_path), **receipt}, sort_keys=True))


if __name__ == "__main__":
    main()
