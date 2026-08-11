"""Typed-MPR source-only spatial likelihood calibration.

Primitive support edges are lifted to canonical regions before text evidence is
seen.  Each actually available edge type is normalized independently and is
kept as a separate carrier.  The learned component remains a scene-shared,
bounded log-odds residual; a zero residual is exactly the legacy probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from radio_gs.querying.source_spatial_text_likelihood import (
    MAX_ABS_LOG_ODDS_RESIDUAL,
    RAW_LOGIT_FEATURE_SCALE,
    apply_bounded_log_odds_residual,
    sha256_file,
    tensor_sha256,
)


SOURCE_TYPED_SPATIAL_SHARD_SCHEMA = (
    "radio_gs.source_typed_spatial_text_likelihood_shard.v1"
)
SOURCE_TYPED_SPATIAL_CHECKPOINT_SCHEMA = (
    "radio_gs.source_typed_spatial_text_likelihood_checkpoint.v1"
)
SOURCE_TYPED_SPATIAL_HEAD_SCHEMA = "bounded-source-typed-spatial-residual-v1"
ALLOWED_EDGE_TYPES = ("appearance", "boundary", "geometry", "covisibility")
FROZEN_EDGE_TYPES = ("appearance", "boundary", "geometry")


def _validate_edge_types(edge_types: Sequence[str]) -> tuple[str, ...]:
    result = tuple(str(value) for value in edge_types)
    if result != tuple(sorted(result)) or len(result) != len(set(result)):
        raise ValueError("typed MPR edge types must be unique and sorted")
    if not result or any(value not in ALLOWED_EDGE_TYPES for value in result):
        raise ValueError("typed MPR edge type is not part of the frozen contract")
    return result


def normalize_typed_region_edges(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    *,
    row_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize a typed region graph independently over each receiver row."""

    index = torch.as_tensor(edge_index).detach().cpu().long().contiguous()
    weight = torch.as_tensor(edge_weight).detach().cpu().float().reshape(-1)
    if (
        index.ndim != 2
        or index.shape[0] != 2
        or index.shape[1] != weight.numel()
        or int(row_count) <= 0
        or bool((index < 0).any())
        or bool((index >= int(row_count)).any())
        or bool((index[0] == index[1]).any())
        or not bool(torch.isfinite(weight).all())
        or bool((weight <= 0).any())
    ):
        raise ValueError("typed region edge axes/weights differ")
    denominator = torch.zeros(int(row_count), dtype=torch.float32)
    denominator.scatter_add_(0, index[0], weight)
    active = denominator > 0
    # Preserve already-normalized immutable shards bit-for-bit.  Re-dividing by
    # a sum such as 0.99999994 changes the tensor hash despite representing the
    # same graph.
    if bool(
        torch.allclose(
            denominator[active],
            torch.ones_like(denominator[active]),
            atol=2e-6,
            rtol=0,
        )
    ):
        normalized = weight
    else:
        normalized = weight / denominator[index[0]].clamp_min(
            torch.finfo(weight.dtype).tiny
        )
    check = torch.zeros_like(denominator)
    check.scatter_add_(0, index[0], normalized)
    if not bool(torch.allclose(check[active], torch.ones_like(check[active]), atol=2e-6, rtol=0)):
        raise RuntimeError("typed region weights do not normalize per receiver")
    return index, normalized.contiguous()


def typed_spatial_logit_statistics(
    raw_logit: torch.Tensor,
    typed_edges: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    valid: torch.Tensor,
) -> dict[str, dict[str, torch.Tensor]]:
    """Compute per-type normalized neighbor mean/max/contrast.

    ``maximum`` is the maximum affinity-interpolated message
    ``center + w_ij * (neighbor - center)``.  Unlike ``w_ij * neighbor`` this
    definition is sign-safe for negative logits and reduces to the neighbor
    logit for a single authoritative edge.
    """

    raw = torch.as_tensor(raw_logit).detach().cpu().float().contiguous()
    if raw.ndim == 2:
        raw = raw[:, None, :]
    keep = torch.as_tensor(valid).detach().cpu()
    if raw.ndim != 3 or keep.shape != (raw.shape[0],) or keep.dtype != torch.bool:
        raise ValueError("typed spatial raw logits and valid rows differ")
    if not bool(torch.isfinite(raw).all()):
        raise ValueError("typed spatial raw logits are nonfinite")
    edge_types = _validate_edge_types(tuple(typed_edges))
    result: dict[str, dict[str, torch.Tensor]] = {}
    rows = int(raw.shape[0])
    for edge_type in edge_types:
        record = typed_edges[edge_type]
        index, weight = normalize_typed_region_edges(
            record["edge_index"], record["edge_weight"], row_count=rows
        )
        receiver, neighbor = index
        edge_keep = keep[receiver] & keep[neighbor]
        filtered_weight = weight * edge_keep.float()
        denominator = torch.zeros(rows, dtype=torch.float32)
        denominator.scatter_add_(0, receiver, filtered_weight)

        flat_channels = int(raw.shape[1] * raw.shape[2])
        raw_flat = raw.reshape(rows, flat_channels)
        weighted = filtered_weight[:, None] * raw_flat[neighbor]
        mean_flat = torch.zeros_like(raw_flat)
        mean_flat.scatter_add_(0, receiver[:, None].expand_as(weighted), weighted)
        mean_flat = mean_flat / denominator[:, None].clamp_min(1.0e-12)
        mean_flat = torch.where(denominator[:, None] > 0, mean_flat, raw_flat)

        message = raw_flat[receiver] + filtered_weight[:, None] * (
            raw_flat[neighbor] - raw_flat[receiver]
        )
        message = message.masked_fill(~edge_keep[:, None], float("-inf"))
        maximum_flat = torch.full_like(raw_flat, float("-inf"))
        maximum_flat.scatter_reduce_(
            0,
            receiver[:, None].expand_as(message),
            message,
            reduce="amax",
            include_self=True,
        )
        maximum_flat = torch.where(
            torch.isfinite(maximum_flat), maximum_flat, raw_flat
        )
        mean = mean_flat.reshape_as(raw).contiguous()
        maximum = maximum_flat.reshape_as(raw).contiguous()
        result[edge_type] = {
            "neighbor_mean_logit": mean,
            "neighbor_max_logit": maximum,
            "neighbor_contrast_logit": (raw - mean).contiguous(),
        }
    return result


@dataclass(frozen=True)
class SourceTypedSpatialLikelihoodInputs:
    raw_logit: torch.Tensor
    typed_statistics: Mapping[str, Mapping[str, torch.Tensor]]
    coverage: torch.Tensor
    reliability: torch.Tensor

    def validated(self) -> "SourceTypedSpatialLikelihoodInputs":
        raw = torch.as_tensor(self.raw_logit).float()
        if raw.ndim == 2:
            raw = raw[:, None, :]
        if raw.ndim != 3 or not bool(torch.isfinite(raw).all()):
            raise ValueError("typed spatial raw logit must be finite [N,S,Q]")
        edge_types = _validate_edge_types(tuple(self.typed_statistics))
        checked: dict[str, dict[str, torch.Tensor]] = {}
        expected = {
            "neighbor_mean_logit",
            "neighbor_max_logit",
            "neighbor_contrast_logit",
        }
        for edge_type in edge_types:
            values = self.typed_statistics[edge_type]
            if set(values) != expected:
                raise ValueError("typed spatial statistics channels differ")
            checked[edge_type] = {}
            for name in sorted(expected):
                tensor = torch.as_tensor(values[name]).float()
                if tensor.shape != raw.shape or not bool(torch.isfinite(tensor).all()):
                    raise ValueError("typed spatial statistic axes differ")
                checked[edge_type][name] = tensor
        coverage = torch.as_tensor(self.coverage).float().reshape(-1)
        reliability = torch.as_tensor(self.reliability).float().reshape(-1)
        if coverage.shape != (raw.shape[0],) or reliability.shape != coverage.shape:
            raise ValueError("typed coverage/reliability axes differ")
        for name, value in (("coverage", coverage), ("reliability", reliability)):
            if not bool(torch.isfinite(value).all()) or bool(((value < 0) | (value > 1)).any()):
                raise ValueError(f"{name} must be finite in [0,1]")
        return SourceTypedSpatialLikelihoodInputs(
            raw_logit=raw,
            typed_statistics=checked,
            coverage=coverage,
            reliability=reliability,
        )


class BoundedTypedSourceSpatialLikelihoodHead(nn.Module):
    schema_version = SOURCE_TYPED_SPATIAL_HEAD_SCHEMA

    def __init__(
        self,
        edge_types: Sequence[str] = FROZEN_EDGE_TYPES,
        hidden_dimension: int = 12,
    ) -> None:
        super().__init__()
        self.edge_types = _validate_edge_types(edge_types)
        if int(hidden_dimension) != 12:
            raise ValueError("typed spatial hidden dimension is frozen at 12")
        self.hidden_dimension = int(hidden_dimension)
        self.input_dimension = 1 + 3 * len(self.edge_types) + 2
        self.input = nn.Linear(self.input_dimension, self.hidden_dimension)
        self.output = nn.Linear(self.hidden_dimension, 1)

    def reset_parameters_deterministic(self, *, seed: int = 17) -> None:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        with torch.no_grad():
            self.input.weight.copy_(
                torch.randn(self.input.weight.shape, generator=generator) * 0.08
            )
            self.input.bias.zero_()
            self.output.weight.copy_(
                torch.randn(self.output.weight.shape, generator=generator) * 0.02
            )
            self.output.bias.zero_()

    def forward(
        self, inputs: SourceTypedSpatialLikelihoodInputs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value = inputs.validated()
        if tuple(value.typed_statistics) != self.edge_types:
            raise ValueError("typed head and input edge types differ")
        raw = value.raw_logit
        coverage = value.coverage[:, None, None].expand_as(raw)
        reliability = value.reliability[:, None, None].expand_as(raw)
        channels = [raw * RAW_LOGIT_FEATURE_SCALE]
        for edge_type in self.edge_types:
            record = value.typed_statistics[edge_type]
            channels.extend(
                record[name] * RAW_LOGIT_FEATURE_SCALE
                for name in (
                    "neighbor_mean_logit",
                    "neighbor_max_logit",
                    "neighbor_contrast_logit",
                )
            )
        channels.extend((coverage, reliability))
        features = torch.stack(channels, dim=-1)
        residual = MAX_ABS_LOG_ODDS_RESIDUAL * torch.tanh(
            self.output(torch.tanh(self.input(features))).squeeze(-1)
        )
        residual = residual * torch.sqrt((coverage * reliability).clamp_min(0.0))
        probability = apply_bounded_log_odds_residual(torch.sigmoid(raw), residual)
        return probability, residual.contiguous()


def validate_source_typed_spatial_shard(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("typed source spatial shard must be a mapping")
    payload = dict(value)
    if (
        payload.get("schema") != SOURCE_TYPED_SPATIAL_SHARD_SCHEMA
        or payload.get("schema_version") != 1
        or payload.get("partition") != "source_train"
        or payload.get("primitive_to_region_aggregation")
        != "same_scale_fractional_membership_MtAM"
    ):
        raise ValueError("typed source spatial shard contract differs")
    access = payload.get("source_access")
    required = {
        "official_scannet_train_scene": True,
        "source_train_semantic_labels_opened": True,
        "development_labels_opened": False,
        "test_labels_opened": False,
        "lerf_queries_or_ground_truth_opened": False,
        "benchmark_predictions_or_metrics_opened": False,
        "per_scene_or_per_query_metric_tuning": False,
    }
    if not isinstance(access, Mapping) or any(access.get(k) is not v for k, v in required.items()):
        raise PermissionError("typed spatial shard crossed its source-fit boundary")
    lineage = payload.get("lineage")
    required_lineage = {
        "source_text_training_shard",
        "accepted_region_authority",
        "query_independent_support_graph",
    }
    if not isinstance(lineage, Mapping) or set(lineage) != required_lineage:
        raise ValueError("typed spatial lineage differs")
    checked_lineage = {}
    for name, record in lineage.items():
        if not isinstance(record, Mapping):
            raise ValueError("typed spatial lineage record differs")
        path = Path(str(record.get("path", ""))).expanduser().resolve(strict=True)
        digest = str(record.get("sha256", ""))
        if sha256_file(path) != digest:
            raise ValueError(f"typed spatial lineage changed: {name}")
        checked_lineage[name] = {"path": str(path), "sha256": digest}

    edge_types = _validate_edge_types(payload.get("edge_types", ()))
    raw = torch.as_tensor(payload.get("raw_logit")).float()
    rows, scales, classes = raw.shape if raw.ndim == 3 else (-1, -1, -1)
    valid = torch.as_tensor(payload.get("valid"))
    coverage = torch.as_tensor(payload.get("coverage")).float().reshape(-1)
    reliability = torch.as_tensor(payload.get("reliability")).float().reshape(-1)
    target = torch.as_tensor(payload.get("semantic_class_distribution")).float()
    label_weight = torch.as_tensor(payload.get("training_label_weight")).float().reshape(-1)
    scale_indices = torch.as_tensor(payload.get("canonical_region_scale_indices")).long()
    if (
        rows <= 0
        or scales != 1
        or valid.shape != (rows,)
        or valid.dtype != torch.bool
        or coverage.shape != reliability.shape != (rows,)
        or coverage.shape != (rows,)
        or target.shape != (rows, classes)
        or label_weight.shape != (rows,)
        or scale_indices.shape != (rows,)
    ):
        raise ValueError("typed spatial row/class axes differ")
    typed_edges = payload.get("typed_region_edges")
    typed_statistics = payload.get("typed_statistics")
    if not isinstance(typed_edges, Mapping) or tuple(typed_edges) != edge_types:
        raise ValueError("typed region edge records differ")
    if not isinstance(typed_statistics, Mapping) or tuple(typed_statistics) != edge_types:
        raise ValueError("typed statistic records differ")
    checked_edges: dict[str, dict[str, torch.Tensor]] = {}
    for edge_type in edge_types:
        record = typed_edges[edge_type]
        if not isinstance(record, Mapping) or set(record) != {"edge_index", "edge_weight"}:
            raise ValueError("typed region edge record differs")
        index, weight = normalize_typed_region_edges(
            record["edge_index"], record["edge_weight"], row_count=rows
        )
        if bool((scale_indices[index[0]] != scale_indices[index[1]]).any()):
            raise ValueError("typed region edge crossed canonical scales")
        checked_edges[edge_type] = {"edge_index": index, "edge_weight": weight}
    inputs = SourceTypedSpatialLikelihoodInputs(
        raw, typed_statistics, coverage, reliability
    ).validated()
    class_ids = [int(v) for v in payload.get("class_ids", [])]
    class_names = [str(v) for v in payload.get("class_names", [])]
    if len(class_ids) != classes or len(class_names) != classes:
        raise ValueError("typed spatial class authority differs")
    tensors: dict[str, torch.Tensor] = {
        "raw_logit": inputs.raw_logit,
        "semantic_class_distribution": target,
        "valid": valid,
        "coverage": coverage,
        "reliability": reliability,
        "training_label_weight": label_weight,
        "canonical_region_scale_indices": scale_indices,
    }
    for edge_type in edge_types:
        tensors[f"{edge_type}.edge_index"] = checked_edges[edge_type]["edge_index"]
        tensors[f"{edge_type}.edge_weight"] = checked_edges[edge_type]["edge_weight"]
        for name, tensor in inputs.typed_statistics[edge_type].items():
            tensors[f"{edge_type}.{name}"] = tensor
    hashes = payload.get("channel_sha256")
    if not isinstance(hashes, Mapping) or set(hashes) != set(tensors):
        raise ValueError("typed spatial channel hashes differ")
    for name, tensor in tensors.items():
        if hashes.get(name) != tensor_sha256(tensor):
            raise ValueError(f"typed spatial channel changed: {name}")
    return {
        **payload,
        **{name: tensors[name] for name in (
            "raw_logit", "semantic_class_distribution", "valid", "coverage",
            "reliability", "training_label_weight", "canonical_region_scale_indices"
        )},
        "edge_types": edge_types,
        "typed_region_edges": checked_edges,
        "typed_statistics": inputs.typed_statistics,
        "class_ids": class_ids,
        "class_names": class_names,
        "lineage": checked_lineage,
    }


__all__ = [
    "ALLOWED_EDGE_TYPES",
    "BoundedTypedSourceSpatialLikelihoodHead",
    "FROZEN_EDGE_TYPES",
    "SOURCE_TYPED_SPATIAL_CHECKPOINT_SCHEMA",
    "SOURCE_TYPED_SPATIAL_HEAD_SCHEMA",
    "SOURCE_TYPED_SPATIAL_SHARD_SCHEMA",
    "SourceTypedSpatialLikelihoodInputs",
    "normalize_typed_region_edges",
    "typed_spatial_logit_statistics",
    "validate_source_typed_spatial_shard",
]
