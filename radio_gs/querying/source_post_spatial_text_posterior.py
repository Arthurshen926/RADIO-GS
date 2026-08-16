"""Source-only examples for the post-spatial TextPosteriorV2 readout."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping

import torch

from radio_gs.interfaces.factorized_primitive_state import FactorizedPrimitiveState
from radio_gs.querying.source_spatial_text_likelihood import (
    fixed_spatial_logit_statistics,
    sha256_file,
    tensor_sha256,
)
from radio_gs.querying.typed_posteriors import validate_reliability_state


SOURCE_POST_SPATIAL_TEXT_SHARD_SCHEMA = (
    "radio_gs.source_post_spatial_text_posterior_shard.v1"
)
SOURCE_POST_SPATIAL_TEXT_CHECKPOINT_SCHEMA = (
    "radio_gs.source_post_spatial_text_posterior_checkpoint.v1"
)
EXTENT_FEATURE_NAMES = (
    "raw_positive_cosine",
    "knn10_raw_positive_cosine_mean",
    "knn10_raw_positive_cosine_max",
    "raw_minus_knn10_mean",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def aggregate_region_reliability(
    state: FactorizedPrimitiveState,
    accepted_region_authority: Mapping[str, Any],
    *,
    region_valid: torch.Tensor,
) -> torch.Tensor:
    """Average the five exact-MPR scalars over each canonical region."""

    region_rows = torch.as_tensor(
        accepted_region_authority.get("region_rows")
    ).long()
    token_mask = torch.as_tensor(accepted_region_authority.get("token_mask"))
    valid = torch.as_tensor(region_valid).bool().reshape(-1)
    if (
        region_rows.ndim != 2
        or token_mask.shape != region_rows.shape
        or token_mask.dtype != torch.bool
        or valid.shape != (region_rows.shape[0],)
        or state.xyz.ndim != 2
        or bool(((region_rows < 0) & token_mask).any())
        or bool(((region_rows >= state.xyz.shape[0]) & token_mask).any())
    ):
        raise ValueError("canonical regions and factorized state do not align")

    full_to_compact = torch.full((state.xyz.shape[0],), -1, dtype=torch.long)
    full_to_compact[state.global_rows] = torch.arange(state.global_rows.numel())
    compact_rows = full_to_compact[region_rows.clamp(0, state.xyz.shape[0] - 1)]
    keep = token_mask & (compact_rows >= 0)
    compact = torch.stack(
        (
            1.0 - state.directional_dispersion,
            state.directional_dispersion,
            state.log_amplitude_std,
            state.observation_evidence,
            torch.where(
                state.visibility_purity_known,
                state.visibility_purity_value,
                torch.zeros_like(state.visibility_purity_value),
            ),
        ),
        dim=-1,
    ).float()
    gathered = compact[compact_rows.clamp_min(0)]
    count = keep.sum(dim=1, keepdim=True)
    result = (gathered * keep.unsqueeze(-1)).sum(dim=1) / count.clamp_min(1)
    result = torch.where((valid & count.squeeze(-1).gt(0))[:, None], result, 0.0)
    validate_reliability_state(result, valid)
    return result.contiguous()


def build_post_spatial_channels(
    positive_affinity: torch.Tensor,
    neighbor_indices: torch.Tensor,
    region_xyz: torch.Tensor,
    *,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduce the deployed positive-cosine kNN10/min-max base score."""

    affinity = torch.as_tensor(positive_affinity).float()
    if affinity.ndim != 3 or affinity.shape[1] != 1:
        raise ValueError("positive affinity must be [N,1,Q]")
    raw = (2.0 * affinity[:, 0] - 1.0).contiguous()
    keep = torch.as_tensor(valid).bool().reshape(-1)
    if keep.shape != (raw.shape[0],):
        raise ValueError("positive affinity and valid rows differ")
    from radio_gs.scripts.eval_lerf_direct_3d_selection import vala_knn_minmax_scores

    base = vala_knn_minmax_scores(
        raw,
        torch.as_tensor(region_xyz).float(),
        k=10,
        valid_mask=keep,
    ).contiguous()
    mean, maximum, contrast = fixed_spatial_logit_statistics(
        raw[:, None, :], neighbor_indices, valid=keep
    )
    extent = torch.stack(
        (raw, mean[:, 0], maximum[:, 0], contrast[:, 0]), dim=-1
    ).contiguous()
    if bool(base[~keep].ne(0).any()):
        raise RuntimeError("post-spatial invalid rows are not zero")
    return base, raw, extent


def _file_record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a file record")
    path = Path(str(value.get("path", ""))).expanduser().resolve(strict=True)
    digest = str(value.get("sha256", ""))
    if _SHA256.fullmatch(digest) is None or sha256_file(path) != digest:
        raise ValueError(f"{label} changed")
    return {"path": str(path), "sha256": digest}


def validate_source_post_spatial_shard(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("post-spatial source shard must be a mapping")
    payload = dict(value)
    if (
        payload.get("schema") != SOURCE_POST_SPATIAL_TEXT_SHARD_SCHEMA
        or payload.get("schema_version") != 1
        or payload.get("partition") != "source_train"
        or payload.get("neighbor_count") != 10
        or payload.get("extent_feature_names") != list(EXTENT_FEATURE_NAMES)
    ):
        raise ValueError("post-spatial source shard contract differs")
    access = payload.get("source_access")
    required_access = {
        "official_scannet_train_scene": True,
        "source_train_semantic_labels_opened": True,
        "development_labels_opened": False,
        "test_labels_opened": False,
        "lerf_queries_or_ground_truth_opened": False,
        "benchmark_predictions_or_metrics_opened": False,
        "per_scene_or_per_query_metric_tuning": False,
    }
    if not isinstance(access, Mapping) or any(
        access.get(name) is not expected
        for name, expected in required_access.items()
    ):
        raise PermissionError("post-spatial source shard crossed its fit boundary")
    lineage = payload.get("lineage")
    expected_lineage = {
        "source_spatial_shard",
        "source_text_training_shard",
        "accepted_region_authority",
        "factorized_primitive_state",
    }
    if not isinstance(lineage, Mapping) or set(lineage) != expected_lineage:
        raise ValueError("post-spatial source lineage differs")
    checked_lineage = {
        name: _file_record(record, label=name) for name, record in lineage.items()
    }

    base = torch.as_tensor(payload.get("base_probability")).float()
    raw = torch.as_tensor(payload.get("raw_positive_cosine")).float()
    extent = torch.as_tensor(payload.get("extent_features")).float()
    target = torch.as_tensor(payload.get("semantic_class_distribution")).float()
    reliability = torch.as_tensor(payload.get("reliability")).float()
    valid = torch.as_tensor(payload.get("valid"))
    weight = torch.as_tensor(payload.get("training_label_weight")).float()
    coverage = torch.as_tensor(payload.get("coverage")).float()
    neighbors = torch.as_tensor(payload.get("neighbor_indices")).long()
    rows, classes = base.shape if base.ndim == 2 else (-1, -1)
    if (
        rows <= 0
        or classes <= 1
        or raw.shape != base.shape
        or extent.shape != (rows, classes, len(EXTENT_FEATURE_NAMES))
        or target.shape != base.shape
        or reliability.shape != (rows, 5)
        or valid.shape != (rows,)
        or valid.dtype != torch.bool
        or weight.shape != (rows,)
        or coverage.shape != (rows,)
        or neighbors.shape != (rows, min(10, rows))
        or len(payload.get("class_ids", [])) != classes
        or len(payload.get("class_names", [])) != classes
    ):
        raise ValueError("post-spatial source shard axes differ")
    for name, tensor in (
        ("base_probability", base),
        ("raw_positive_cosine", raw),
        ("extent_features", extent),
        ("semantic_class_distribution", target),
        ("training_label_weight", weight),
        ("coverage", coverage),
    ):
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} is non-finite")
    if bool(((base < 0) | (base > 1)).any()) or bool(
        ((target < 0) | (target > 1)).any()
    ) or bool(((weight < 0) | (weight > 1)).any()) or bool(
        ((coverage < 0) | (coverage > 1)).any()
    ):
        raise ValueError("post-spatial probability/weight range differs")
    if bool(base[~valid].ne(0).any()):
        raise ValueError("post-spatial invalid base rows must be zero")
    validate_reliability_state(reliability, valid)
    channels = payload.get("channel_sha256")
    tensors = {
        "base_probability": base,
        "raw_positive_cosine": raw,
        "extent_features": extent,
        "semantic_class_distribution": target,
        "reliability": reliability,
        "valid": valid,
        "training_label_weight": weight,
        "coverage": coverage,
        "neighbor_indices": neighbors,
    }
    if not isinstance(channels, Mapping) or set(channels) != set(tensors):
        raise ValueError("post-spatial channel hashes differ")
    if any(channels[name] != tensor_sha256(tensor) for name, tensor in tensors.items()):
        raise ValueError("post-spatial channel changed")
    payload.update(tensors)
    payload["lineage"] = checked_lineage
    return payload


__all__ = [
    "EXTENT_FEATURE_NAMES",
    "SOURCE_POST_SPATIAL_TEXT_CHECKPOINT_SCHEMA",
    "SOURCE_POST_SPATIAL_TEXT_SHARD_SCHEMA",
    "aggregate_region_reliability",
    "build_post_spatial_channels",
    "validate_source_post_spatial_shard",
]
