"""Source-only spatial likelihood calibration shared by ScanNet and LERF.

The graph is query independent.  Text enters only through the raw
class-versus-canonical-negative logit.  A small scene-shared head predicts a
bounded log-odds residual from fixed local statistics and field support.  A
zero residual is an exact, storage-level identity with the legacy probability.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import torch
from torch import nn


SOURCE_SPATIAL_SHARD_SCHEMA = "radio_gs.source_spatial_text_likelihood_shard.v1"
SOURCE_SPATIAL_CHECKPOINT_SCHEMA = (
    "radio_gs.source_spatial_text_likelihood_checkpoint.v1"
)
SOURCE_SPATIAL_HEAD_SCHEMA = "bounded-source-spatial-residual-v1"
FIXED_NEIGHBOR_COUNT = 10
RAW_LOGIT_FEATURE_SCALE = 0.1
MAX_ABS_LOG_ODDS_RESIDUAL = 2.0
PROBABILITY_EPS = 1.0e-6
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: str | Path) -> str:
    source = Path(path).expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = torch.as_tensor(value).detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def fixed_knn_indices(xyz: torch.Tensor, *, k: int = FIXED_NEIGHBOR_COUNT) -> torch.Tensor:
    points = torch.as_tensor(xyz).detach().cpu().float().contiguous()
    if points.ndim != 2 or points.shape[1] != 3 or not bool(torch.isfinite(points).all()):
        raise ValueError("xyz must be finite [N,3]")
    if int(points.shape[0]) < 2 or int(k) != FIXED_NEIGHBOR_COUNT:
        raise ValueError(f"fixed spatial likelihood requires k={FIXED_NEIGHBOR_COUNT}")
    from sklearn.neighbors import NearestNeighbors

    indices = NearestNeighbors(n_neighbors=min(int(k), int(points.shape[0]))).fit(
        points.numpy()
    ).kneighbors(points.numpy(), return_distance=False)
    result = torch.from_numpy(indices).long().contiguous()
    if result.shape != (points.shape[0], min(int(k), int(points.shape[0]))):
        raise RuntimeError("fixed kNN returned unexpected axes")
    return result


def fixed_spatial_logit_statistics(
    raw_logit: torch.Tensor,
    neighbor_indices: torch.Tensor,
    *,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return fixed neighbor mean, max and center-minus-mean contrast."""

    raw = torch.as_tensor(raw_logit).detach().cpu().float().contiguous()
    if raw.ndim == 2:
        raw = raw[:, None, :]
    indices = torch.as_tensor(neighbor_indices).detach().cpu().long().contiguous()
    keep = torch.as_tensor(valid).detach().cpu()
    if (
        raw.ndim != 3
        or indices.ndim != 2
        or indices.shape[0] != raw.shape[0]
        or indices.shape[1] != min(FIXED_NEIGHBOR_COUNT, int(raw.shape[0]))
        or keep.shape != (raw.shape[0],)
        or keep.dtype != torch.bool
        or not bool(torch.isfinite(raw).all())
        or bool((indices < 0).any())
        or bool((indices >= raw.shape[0]).any())
    ):
        raise ValueError("raw logits, fixed neighbors, and valid mask do not align")
    neighbor = raw[indices]
    neighbor_valid = keep[indices]
    count = neighbor_valid.sum(dim=1, keepdim=True).unsqueeze(-1)
    mean = (neighbor * neighbor_valid[:, :, None, None]).sum(dim=1) / count.clamp_min(1)
    mean = torch.where(count > 0, mean, raw)
    masked = neighbor.masked_fill(~neighbor_valid[:, :, None, None], float("-inf"))
    maximum = masked.amax(dim=1)
    maximum = torch.where(torch.isfinite(maximum), maximum, raw)
    contrast = raw - mean
    return mean.contiguous(), maximum.contiguous(), contrast.contiguous()


def apply_bounded_log_odds_residual(
    legacy_probability: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    legacy = torch.as_tensor(legacy_probability).float()
    delta = torch.as_tensor(residual).float()
    if legacy.shape != delta.shape or legacy.ndim not in {2, 3}:
        raise ValueError("legacy probability and residual axes differ")
    if not bool(torch.isfinite(legacy).all()) or bool(
        ((legacy < 0) | (legacy > 1)).any()
    ):
        raise ValueError("legacy probability must be finite in [0,1]")
    if not bool(torch.isfinite(delta).all()) or bool(
        (delta.abs() > MAX_ABS_LOG_ODDS_RESIDUAL + 1.0e-6).any()
    ):
        raise ValueError("residual exceeds the frozen bound")
    clipped = legacy.clamp(PROBABILITY_EPS, 1.0 - PROBABILITY_EPS)
    transported = torch.sigmoid(torch.logit(clipped) + delta)
    identity = (delta == 0) | (legacy == 0) | (legacy == 1)
    transported = torch.where(identity, legacy, transported)
    if not torch.equal(transported[delta == 0], legacy[delta == 0]):
        raise RuntimeError("zero spatial residual changed legacy probability")
    return transported.contiguous()


@dataclass(frozen=True)
class SourceSpatialLikelihoodInputs:
    raw_logit: torch.Tensor
    neighbor_mean_logit: torch.Tensor
    neighbor_max_logit: torch.Tensor
    neighbor_contrast_logit: torch.Tensor
    coverage: torch.Tensor
    reliability: torch.Tensor

    def validated(self) -> "SourceSpatialLikelihoodInputs":
        raw = torch.as_tensor(self.raw_logit).float()
        mean = torch.as_tensor(self.neighbor_mean_logit).float()
        maximum = torch.as_tensor(self.neighbor_max_logit).float()
        contrast = torch.as_tensor(self.neighbor_contrast_logit).float()
        if raw.ndim == 2:
            raw = raw[:, None, :]
        values = [raw, mean, maximum, contrast]
        if any(value.shape != raw.shape for value in values) or raw.ndim != 3:
            raise ValueError("spatial logit channels must share [N,S,Q] axes")
        if any(not bool(torch.isfinite(value).all()) for value in values):
            raise ValueError("spatial logit channels contain NaN or infinity")
        coverage = torch.as_tensor(self.coverage).float().reshape(-1)
        reliability = torch.as_tensor(self.reliability).float().reshape(-1)
        if coverage.shape != (raw.shape[0],) or reliability.shape != coverage.shape:
            raise ValueError("coverage/reliability must align with rows")
        for name, value in (("coverage", coverage), ("reliability", reliability)):
            if not bool(torch.isfinite(value).all()) or bool(
                ((value < 0) | (value > 1)).any()
            ):
                raise ValueError(f"{name} must be finite in [0,1]")
        return SourceSpatialLikelihoodInputs(
            raw_logit=raw,
            neighbor_mean_logit=mean,
            neighbor_max_logit=maximum,
            neighbor_contrast_logit=contrast,
            coverage=coverage,
            reliability=reliability,
        )


class BoundedSourceSpatialLikelihoodHead(nn.Module):
    schema_version = SOURCE_SPATIAL_HEAD_SCHEMA

    def __init__(self, hidden_dimension: int = 12) -> None:
        super().__init__()
        if int(hidden_dimension) != 12:
            raise ValueError("source spatial head hidden dimension is frozen at 12")
        self.hidden_dimension = int(hidden_dimension)
        self.input = nn.Linear(6, self.hidden_dimension)
        self.output = nn.Linear(self.hidden_dimension, 1)

    def reset_parameters_deterministic(self, *, seed: int = 17) -> None:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        with torch.no_grad():
            self.input.weight.copy_(
                torch.randn(
                    self.input.weight.shape,
                    generator=generator,
                    dtype=self.input.weight.dtype,
                )
                * 0.08
            )
            self.input.bias.zero_()
            self.output.weight.copy_(
                torch.randn(
                    self.output.weight.shape,
                    generator=generator,
                    dtype=self.output.weight.dtype,
                )
                * 0.02
            )
            self.output.bias.zero_()

    def forward(self, inputs: SourceSpatialLikelihoodInputs) -> tuple[torch.Tensor, torch.Tensor]:
        value = inputs.validated()
        raw = value.raw_logit
        coverage = value.coverage[:, None, None].expand_as(raw)
        reliability = value.reliability[:, None, None].expand_as(raw)
        features = torch.stack(
            (
                raw * RAW_LOGIT_FEATURE_SCALE,
                value.neighbor_mean_logit * RAW_LOGIT_FEATURE_SCALE,
                value.neighbor_max_logit * RAW_LOGIT_FEATURE_SCALE,
                value.neighbor_contrast_logit * RAW_LOGIT_FEATURE_SCALE,
                coverage,
                reliability,
            ),
            dim=-1,
        )
        residual = MAX_ABS_LOG_ODDS_RESIDUAL * torch.tanh(
            self.output(torch.tanh(self.input(features))).squeeze(-1)
        )
        authority = torch.sqrt((coverage * reliability).clamp_min(0.0))
        residual = residual * authority
        legacy = torch.sigmoid(raw)
        probability = apply_bounded_log_odds_residual(legacy, residual)
        return probability, residual.contiguous()


def _require_file_record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a file record")
    path = Path(str(value.get("path", ""))).expanduser().resolve(strict=True)
    digest = str(value.get("sha256", ""))
    if _SHA256.fullmatch(digest) is None or sha256_file(path) != digest:
        raise ValueError(f"{label} changed")
    return {"path": str(path), "sha256": digest}


def validate_source_spatial_shard(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("source spatial shard must be a mapping")
    payload = dict(value)
    if (
        payload.get("schema") != SOURCE_SPATIAL_SHARD_SCHEMA
        or payload.get("schema_version") != 1
        or payload.get("partition") != "source_train"
        or payload.get("neighbor_count") != FIXED_NEIGHBOR_COUNT
    ):
        raise ValueError("source spatial shard contract differs")
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
        raise PermissionError("source spatial shard crossed its fit boundary")
    lineage = payload.get("lineage")
    if not isinstance(lineage, Mapping) or set(lineage) != {
        "source_text_training_shard",
        "accepted_region_authority",
        "query_independent_support_graph",
    }:
        raise ValueError("source spatial shard lineage differs")
    checked_lineage = {
        name: _require_file_record(record, label=name)
        for name, record in lineage.items()
    }
    raw = torch.as_tensor(payload.get("raw_logit")).float()
    mean = torch.as_tensor(payload.get("neighbor_mean_logit")).float()
    maximum = torch.as_tensor(payload.get("neighbor_max_logit")).float()
    contrast = torch.as_tensor(payload.get("neighbor_contrast_logit")).float()
    if raw.ndim != 3 or any(x.shape != raw.shape for x in (mean, maximum, contrast)):
        raise ValueError("source spatial logit axes differ")
    rows, _scales, classes = raw.shape
    xyz = torch.as_tensor(payload.get("region_xyz")).float()
    neighbors = torch.as_tensor(payload.get("neighbor_indices")).long()
    target = torch.as_tensor(payload.get("semantic_class_distribution")).float()
    valid = torch.as_tensor(payload.get("valid"))
    coverage = torch.as_tensor(payload.get("coverage")).float().reshape(-1)
    reliability = torch.as_tensor(payload.get("reliability")).float().reshape(-1)
    weight = torch.as_tensor(payload.get("training_label_weight")).float().reshape(-1)
    class_ids = [int(v) for v in payload.get("class_ids", [])]
    class_names = [str(v) for v in payload.get("class_names", [])]
    if (
        xyz.shape != (rows, 3)
        or neighbors.shape != (rows, min(FIXED_NEIGHBOR_COUNT, rows))
        or target.shape != (rows, classes)
        or valid.shape != (rows,)
        or valid.dtype != torch.bool
        or any(x.shape != (rows,) for x in (coverage, reliability, weight))
        or len(class_ids) != classes
        or len(class_names) != classes
    ):
        raise ValueError("source spatial shard row/class channels differ")
    inputs = SourceSpatialLikelihoodInputs(
        raw, mean, maximum, contrast, coverage, reliability
    ).validated()
    for name, tensor in (
        ("region_xyz", xyz),
        ("semantic_class_distribution", target),
        ("training_label_weight", weight),
    ):
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} contains NaN or infinity")
    if bool(((target < 0) | (target > 1)).any()) or bool(
        ((weight < 0) | (weight > 1)).any()
    ):
        raise ValueError("source spatial target/weight must lie in [0,1]")
    channels = payload.get("channel_sha256")
    tensors = {
        "raw_logit": inputs.raw_logit,
        "neighbor_mean_logit": inputs.neighbor_mean_logit,
        "neighbor_max_logit": inputs.neighbor_max_logit,
        "neighbor_contrast_logit": inputs.neighbor_contrast_logit,
        "region_xyz": xyz,
        "neighbor_indices": neighbors,
        "semantic_class_distribution": target,
        "valid": valid,
        "coverage": coverage,
        "reliability": reliability,
        "training_label_weight": weight,
    }
    if not isinstance(channels, Mapping) or set(channels) != set(tensors):
        raise ValueError("source spatial channel hashes differ")
    for name, tensor in tensors.items():
        if channels.get(name) != tensor_sha256(tensor):
            raise ValueError(f"source spatial channel changed: {name}")
    return {
        **payload,
        **tensors,
        "class_ids": class_ids,
        "class_names": class_names,
        "lineage": checked_lineage,
    }


__all__ = [
    "BoundedSourceSpatialLikelihoodHead",
    "FIXED_NEIGHBOR_COUNT",
    "MAX_ABS_LOG_ODDS_RESIDUAL",
    "PROBABILITY_EPS",
    "RAW_LOGIT_FEATURE_SCALE",
    "SOURCE_SPATIAL_CHECKPOINT_SCHEMA",
    "SOURCE_SPATIAL_HEAD_SCHEMA",
    "SOURCE_SPATIAL_SHARD_SCHEMA",
    "SourceSpatialLikelihoodInputs",
    "apply_bounded_log_odds_residual",
    "fixed_knn_indices",
    "fixed_spatial_logit_statistics",
    "sha256_file",
    "state_dict_sha256",
    "tensor_sha256",
    "validate_source_spatial_shard",
]
