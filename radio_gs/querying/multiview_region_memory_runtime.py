"""Fail-closed runtime adapter for sealed object-level region memories."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F

from radio_gs.interfaces.prompt_responsibility_cache import tensor_sha256
from radio_gs.utils.immutable_artifacts import (
    load_torch_mapping,
    validate_file_record,
)

from .multiview_region_memory import METHOD, method_contract
from .query_spec import PrimitiveUnaryEvidence, PrototypeSet, QuerySpec


ARTIFACT_TYPE = "multiview_region_memory_primitive_asset_v1"
RUNTIME_MODE = "source_only_object_multiview_region_memory_v1"


@dataclass(frozen=True)
class LoadedRegionMemory:
    probability: torch.Tensor
    confidence: torch.Tensor
    positive_mass_by_view: torch.Tensor
    view_reliability: torch.Tensor
    evidence: Mapping[str, object]


@dataclass(frozen=True)
class RegionMemoryCompletionDiagnostics:
    num_nodes: int
    base_observed_rows: int
    base_abstained_rows: int
    memory_observed_rows: int
    completed_rows: int
    completed_positive_rows: int
    completed_negative_rows: int
    completed_confidence_sum: float
    observed_values_bitwise_equal: bool
    observed_confidence_bitwise_equal: bool


def load_region_memory(
    path: str | Path,
    *,
    expected_sha256: str,
    scene_id: str,
    capability_path: str | Path,
    capability_sha256: str | None,
    global_rows: torch.Tensor,
    num_gaussians: int,
) -> LoadedRegionMemory:
    """Load one local, sealed memory without reopening source or target images."""

    payload, digest, source = load_torch_mapping(
        path,
        expected_sha256=expected_sha256,
        map_location="cpu",
        label="object multiview region memory",
    )
    access = payload.get("source_access")
    capability = payload.get("capability_cache")
    tensor_hashes = payload.get("tensor_sha256")
    views = payload.get("views")
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_type") != ARTIFACT_TYPE
        or payload.get("status") != "primitive_region_memory_sealed_before_target_access"
        or payload.get("scene_id") != str(scene_id)
        or payload.get("method") != METHOD
        or payload.get("method_contract") != method_contract()
        or int(payload.get("num_gaussians", -1)) != int(num_gaussians)
        or int(payload.get("view_count", -1)) != 3
        or not isinstance(views, list)
        or len(views) != 3
        or not isinstance(capability, Mapping)
        or Path(str(capability.get("path", ""))).expanduser().resolve()
        != Path(capability_path).expanduser().resolve()
        or (
            capability_sha256 is not None
            and capability.get("sha256") != str(capability_sha256)
        )
        or not isinstance(tensor_hashes, Mapping)
        or access
        != {
            "source_rgb_opened_by_upstream_sam3": True,
            "target_rgb_opened": False,
            "target_mask_opened": False,
            "target_metric_opened": False,
            "candidate_selected_with_gt": False,
        }
    ):
        raise ValueError("object multiview region-memory authority differs")
    validate_file_record(payload.get("implementation"), label="region-memory materializer")
    required = {
        "valid_rows",
        "membership_probability",
        "membership_confidence",
        "membership_observed",
        "positive_mass_by_view",
        "proposal_masks_feature",
        "observation_domains_feature",
        "view_reliability",
    }
    if set(tensor_hashes) != required:
        raise ValueError("region-memory tensor authority differs")
    tensors = {name: payload.get(name) for name in required}
    if any(not torch.is_tensor(value) for value in tensors.values()):
        raise ValueError("region-memory tensor is missing")
    for name, value in tensors.items():
        if value.device.type != "cpu" or tensor_sha256(value) != tensor_hashes[name]:
            raise ValueError(f"region-memory tensor {name} changed")
    rows = tensors["valid_rows"].long().reshape(-1)
    expected_rows = torch.as_tensor(global_rows).long().cpu().reshape(-1)
    probability = tensors["membership_probability"].float().reshape(-1)
    confidence = tensors["membership_confidence"].float().reshape(-1)
    observed = tensors["membership_observed"].bool().reshape(-1)
    positive_mass = tensors["positive_mass_by_view"].float()
    reliability = tensors["view_reliability"].float().reshape(-1)
    count = int(expected_rows.numel())
    if (
        not torch.equal(rows, expected_rows)
        or probability.shape != (count,)
        or confidence.shape != (count,)
        or observed.shape != (count,)
        or positive_mass.shape != (3, count)
        or reliability.shape != (3,)
        or not bool(torch.isfinite(probability).all())
        or not bool(torch.isfinite(confidence).all())
        or not bool(torch.isfinite(positive_mass).all())
        or not bool(torch.isfinite(reliability).all())
        or bool(((probability < 0) | (probability > 1)).any())
        or bool(((confidence < 0) | (confidence > 1)).any())
        or bool((positive_mass < 0).any())
        or bool(((reliability <= 0) | (reliability > 1)).any())
        or not torch.equal(observed, confidence > 0)
        or not bool((positive_mass.sum(dim=1) > 0).all())
    ):
        raise ValueError("region-memory primitive tensors differ")
    return LoadedRegionMemory(
        probability=probability.contiguous(),
        confidence=confidence.contiguous(),
        positive_mass_by_view=positive_mass.contiguous(),
        view_reliability=reliability.contiguous(),
        evidence={
            "mode": RUNTIME_MODE,
            "path": str(source),
            "sha256": digest,
            "method_contract": method_contract(),
            "view_count": 3,
            "views": views,
            "target_rgb_opened": False,
            "target_mask_opened": False,
            "target_metric_opened": False,
        },
    )


def complete_abstaining_observation(
    observation: PrimitiveUnaryEvidence,
    memory: LoadedRegionMemory,
) -> tuple[PrimitiveUnaryEvidence, torch.Tensor, RegionMemoryCompletionDiagnostics]:
    """Write only exact abstentions and preserve every observed row bitwise."""

    if observation.confidence is None:
        raise ValueError("region-memory completion requires explicit base confidence")
    base_values = observation.values.detach().float().cpu()
    base_confidence = observation.confidence.detach().float().cpu()
    if (
        memory.probability.shape != base_values.shape
        or memory.confidence.shape != base_values.shape
    ):
        raise ValueError("region memory and base observation do not align")
    changed = (base_confidence == 0) & (memory.confidence > 0)
    values = base_values.clone()
    confidence = base_confidence.clone()
    values[changed] = memory.confidence[changed] * (
        2.0 * memory.probability[changed] - 1.0
    )
    confidence[changed] = memory.confidence[changed]
    observed = base_confidence > 0
    values_equal = torch.equal(values[observed], base_values[observed])
    confidence_equal = torch.equal(confidence[observed], base_confidence[observed])
    if not values_equal or not confidence_equal:
        raise RuntimeError("region memory changed a base-observed prompt row")
    completed = PrimitiveUnaryEvidence(
        values,
        "source_only_object_multiview_region_memory_v1",
        confidence=confidence,
    )
    diagnostics = RegionMemoryCompletionDiagnostics(
        num_nodes=int(values.numel()),
        base_observed_rows=int(observed.sum()),
        base_abstained_rows=int((~observed).sum()),
        memory_observed_rows=int((memory.confidence > 0).sum()),
        completed_rows=int(changed.sum()),
        completed_positive_rows=int((changed & (memory.probability >= 0.5)).sum()),
        completed_negative_rows=int((changed & (memory.probability < 0.5)).sum()),
        completed_confidence_sum=float(memory.confidence[changed].sum()),
        observed_values_bitwise_equal=values_equal,
        observed_confidence_bitwise_equal=confidence_equal,
    )
    if diagnostics.completed_rows <= 0:
        raise ValueError("region memory does not complete any base abstention")
    return completed, changed, diagnostics


def _pool_tokens(
    features: torch.Tensor,
    positive_mass_by_view: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    values = torch.as_tensor(features)
    mass = torch.as_tensor(positive_mass_by_view).float().cpu()
    if (
        values.ndim != 2
        or mass.ndim != 2
        or mass.shape[1] != values.shape[0]
        or int(chunk_size) <= 0
    ):
        raise ValueError("region token features and mass do not align")
    tokens = torch.zeros(
        (mass.shape[0], values.shape[1]),
        device=values.device,
        dtype=torch.float32,
    )
    for start in range(0, values.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), values.shape[0])
        local_mass = mass[:, start:stop].to(values.device)
        tokens += local_mass @ values[start:stop].float()
    tokens /= mass.sum(dim=1).to(tokens.device)[:, None].clamp_min(1e-30)
    tokens = F.normalize(tokens, dim=-1, eps=1e-8)
    if not bool(torch.isfinite(tokens).all()) or bool(
        (torch.linalg.vector_norm(tokens, dim=1) <= 0).any()
    ):
        raise ValueError("region token is degenerate")
    return tokens


def augment_query_with_region_tokens(
    query: QuerySpec,
    feature_banks: Mapping[str, torch.Tensor],
    memory: LoadedRegionMemory,
    *,
    chunk_size: int = 8192,
) -> tuple[QuerySpec, dict[str, object]]:
    """Append three reliability-weighted source-object tokens per field bank."""

    if query.appearance_evidence is None or query.boundary_evidence is None:
        raise ValueError("region-token augmentation requires appearance and boundary evidence")
    if set(feature_banks) != {"appearance", "boundary"}:
        raise ValueError("region-token feature banks differ")
    source_raw_weights = memory.view_reliability / float(
        memory.view_reliability.numel()
    )

    def augment(name: str, evidence: PrototypeSet) -> PrototypeSet:
        tokens = _pool_tokens(
            feature_banks[name],
            memory.positive_mass_by_view,
            chunk_size=int(chunk_size),
        )
        features = torch.cat(
            [evidence.features.to(tokens.device), tokens], dim=0
        )
        weights = torch.cat(
            [
                evidence.weights.to(tokens.device),
                source_raw_weights.to(tokens.device),
            ]
        )
        return PrototypeSet(
            features,
            evidence.signature,
            weights=weights,
            negatives=evidence.negatives,
        )

    appearance = augment("appearance", query.appearance_evidence)
    boundary = augment("boundary", query.boundary_evidence)
    augmented = replace(
        query,
        appearance_evidence=appearance,
        boundary_evidence=boundary,
        metadata={
            **query.metadata,
            "object_multiview_region_tokens": {
                "view_count": 3,
                "source_raw_weight_sum": float(source_raw_weights.sum()),
                "source_raw_weight_formula": "view_reliability_divided_by_fixed_view_count",
            },
        },
    )
    return augmented, {
        "view_count": 3,
        "appearance_token_count": 3,
        "boundary_token_count": 3,
        "source_raw_weight_sum": float(source_raw_weights.sum()),
        "reference_appearance_prototype_count": int(
            query.appearance_evidence.features.shape[0]
        ),
        "reference_boundary_prototype_count": int(
            query.boundary_evidence.features.shape[0]
        ),
    }


__all__ = [
    "LoadedRegionMemory",
    "RegionMemoryCompletionDiagnostics",
    "RUNTIME_MODE",
    "augment_query_with_region_tokens",
    "complete_abstaining_observation",
    "load_region_memory",
]
