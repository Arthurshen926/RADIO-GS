"""Fail-closed training loader for canonical-factorized-radio-v1 caches."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import torch

from radio_gs.field.factorized_radio_contract import (
    CANONICAL_FACTORIZED_RADIO_CACHE_SCHEMA,
    CANONICAL_FACTORIZED_RADIO_CONTRACT_NAME,
    CANONICAL_FACTORIZED_RADIO_CONTRACT_SHA256,
    FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES,
    FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES_SHA256,
    canonical_factorized_radio_contract,
)
from radio_gs.rendering.sparse_marginal_authority import (
    SPARSE_EXACT_MARGINAL_FORMULA_SHA256,
)
from radio_gs.training.primitive_consensus import PrimitiveConsensus
from radio_gs.utils.immutable_artifacts import load_torch_mapping


CANONICAL_FACTORIZED_RADIO_BUILDER_CACHE_SCHEMA = (
    "radio_gs.canonical_factorized_radio_builder_cache.v1"
)
CANONICAL_FACTORIZED_RADIO_BUILDER_CACHE_SCHEMA_V2 = (
    "radio_gs.canonical_factorized_radio_builder_cache.v2"
)
FACTORIZED_RADIO_TOP1_PURITY_AUTHORITY = {
    "authority": "raster_gaussian_top1_sidecar_v1_missing_visible_mass",
    "measurement_available": False,
    "encoding": "exact_zero_unknown_sentinel",
    "consumer_policy": "must_not_treat_visibility_purity_as_a_measurement",
}
FACTORIZED_RADIO_EXACT_MARGINAL_PURITY_AUTHORITY = {
    "authority": "sparse_exact_marginal_responsibility_authority_v1",
    "measurement_available": True,
    "encoding": "positive_marginal_mass_over_exact_visible_mass",
    "consumer_policy": "measured_visibility_purity_may_weight_confidence_only",
    "formula_sha256": SPARSE_EXACT_MARGINAL_FORMULA_SHA256,
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_factorized_radio_builder_contract() -> dict[str, object]:
    """Mirror the immutable builder policy without importing the script layer."""

    return {
        "name": "canonical-factorized-radio-v1-builder",
        "schema_version": 1,
        "parent_contract_sha256": CANONICAL_FACTORIZED_RADIO_CONTRACT_SHA256,
        "feature_space": "radio",
        "input_feature_space": "radio_raw_full",
        "maximum_views": 120,
        "view_selection": "uniform_temporal_deterministic",
        "aggregation_mode": "raster_gaussian_top1",
        "raster_view_fusion": "contribution_mean",
        "registration_weight_mode": "alpha_depth",
        "normalize_each_view": False,
        "robust_mpr": False,
        "observation_unit": "positive_norm_raw_radio_pixel",
        "visibility_purity": "exact_zero_unknown_top1_sidecar_sentinel",
        "query_independent": True,
        "benchmark_masks_opened": False,
        "benchmark_images_opened": False,
        "text_queries_opened": False,
    }


def factorized_radio_builder_contract_sha256() -> str:
    encoded = json.dumps(
        canonical_factorized_radio_builder_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_factorized_radio_builder_contract_v2() -> dict[str, object]:
    return {
        "name": "canonical-factorized-radio-v1-builder-exact-marginal-v2",
        "schema_version": 2,
        "parent_contract_sha256": CANONICAL_FACTORIZED_RADIO_CONTRACT_SHA256,
        "feature_space": "radio",
        "input_feature_space": "radio_raw_full",
        "maximum_views": 120,
        "view_selection": "uniform_temporal_deterministic",
        "aggregation_mode": "raster_marginal_responsibility",
        "raster_view_fusion": "contribution_mean",
        "registration_weight_mode": (
            "exact_front_to_back_marginal_responsibility"
        ),
        "normalize_each_view": False,
        "robust_mpr": False,
        "observation_unit": "positive_norm_raw_radio_pixel",
        "semantic_weight": "exact_base_weight_times_pixel_marginal",
        "visibility_weight": "exact_base_weight",
        "visibility_purity": (
            "positive_amplitude_marginal_mass_over_exact_visible_mass"
        ),
        "sparse_authority_formula_sha256": (
            SPARSE_EXACT_MARGINAL_FORMULA_SHA256
        ),
        "query_independent": True,
        "benchmark_masks_opened": False,
        "benchmark_images_opened": False,
        "text_queries_opened": False,
    }


def factorized_radio_builder_contract_v2_sha256() -> str:
    encoded = json.dumps(
        canonical_factorized_radio_builder_contract_v2(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_tensor_rows(values: torch.Tensor) -> str:
    array = values.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _all_finite(values: torch.Tensor, *, row_chunk: int = 4096) -> bool:
    for start in range(0, int(values.shape[0]), int(row_chunk)):
        if not bool(torch.isfinite(values[start : start + row_chunk]).all()):
            return False
    return True


@dataclass(frozen=True)
class FactorizedRadioTrainingCache:
    source: Path
    sha256: str
    xyz: torch.Tensor
    geometry_fingerprint: dict[str, Any]
    canonical_feature: torch.Tensor
    log_amplitude: torch.Tensor
    valid: torch.Tensor
    view_counts: torch.Tensor
    reliability: torch.Tensor
    reliability_scalar_names: tuple[str, ...]
    reliability_scalar_names_sha256: str
    metadata: dict[str, Any]

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.canonical_feature.shape[0]), int(
            self.canonical_feature.shape[1]
        )

    def as_consensus(self) -> PrimitiveConsensus:
        """Expose canonical rows while keeping reliability out of the field."""

        return PrimitiveConsensus(
            targets=self.canonical_feature,
            valid=self.valid,
            observation_count=self.view_counts,
            reliability=self.reliability,
            per_view_agreement=torch.empty(0, int(self.valid.numel())),
        )

    def support_mapping(self) -> dict[str, Any]:
        return {
            "xyz": self.xyz,
            "valid": self.valid,
            "view_counts": self.view_counts,
            "geometry_fingerprint": dict(self.geometry_fingerprint),
            "metadata": dict(self.metadata),
        }

    def provenance(self) -> dict[str, Any]:
        builder_version = int(self.metadata.get("builder_contract", {}).get(
            "schema_version", 1
        ))
        return {
            "storage": (
                CANONICAL_FACTORIZED_RADIO_BUILDER_CACHE_SCHEMA_V2
                if builder_version == 2
                else CANONICAL_FACTORIZED_RADIO_BUILDER_CACHE_SCHEMA
            ),
            "path": str(self.source),
            "sha256": self.sha256,
            "contract_sha256": CANONICAL_FACTORIZED_RADIO_CONTRACT_SHA256,
            "builder_contract_sha256": (
                factorized_radio_builder_contract_v2_sha256()
                if builder_version == 2
                else factorized_radio_builder_contract_sha256()
            ),
        }


def validate_factorized_radio_training_payload(
    value: object,
    *,
    expected_feature_output_bundle_sha256: str,
) -> dict[str, Any]:
    """Validate the builder envelope without deriving a duplicate N x D tensor."""

    if not isinstance(value, Mapping):
        raise ValueError("factorized RADIO training cache must contain a mapping")
    payload = dict(value)
    required = {
        "schema",
        "schema_version",
        "xyz",
        "geometry_fingerprint",
        "factorized_radio",
        "view_counts",
        "metadata",
    }
    if set(payload) != required:
        raise ValueError("factorized RADIO builder cache schema differs")
    schema_pair = (payload.get("schema"), payload.get("schema_version"))
    if schema_pair == (CANONICAL_FACTORIZED_RADIO_BUILDER_CACHE_SCHEMA, 1):
        builder_version = 1
    elif schema_pair == (CANONICAL_FACTORIZED_RADIO_BUILDER_CACHE_SCHEMA_V2, 2):
        builder_version = 2
    else:
        raise ValueError("factorized RADIO builder cache schema differs")
    xyz = payload.get("xyz")
    geometry = payload.get("geometry_fingerprint")
    counts = payload.get("view_counts")
    core = payload.get("factorized_radio")
    metadata = payload.get("metadata")
    if (
        not torch.is_tensor(xyz)
        or xyz.dtype != torch.float32
        or xyz.ndim != 2
        or xyz.shape[1] != 3
        or int(xyz.shape[0]) <= 0
        or not torch.is_tensor(counts)
        or counts.dtype != torch.long
        or counts.shape != (xyz.shape[0],)
        or not isinstance(geometry, Mapping)
        or not isinstance(core, Mapping)
        or not isinstance(metadata, Mapping)
    ):
        raise ValueError("factorized RADIO builder support differs")
    expected_geometry = {
        "num_gaussians": int(xyz.shape[0]),
        "xyz_sha256": _sha256_tensor_rows(xyz),
    }
    if dict(geometry) != expected_geometry:
        raise ValueError("factorized RADIO geometry fingerprint differs")
    required_core = {
        "schema",
        "schema_version",
        "contract",
        "contract_sha256",
        "reliability_scalar_names",
        "reliability_scalar_names_sha256",
        "log_amplitude",
        "canonical_feature",
        "valid",
        "reliability",
    }
    if set(core) != required_core or (
        core.get("schema") != CANONICAL_FACTORIZED_RADIO_CACHE_SCHEMA
        or core.get("schema_version") != 1
        or core.get("contract") != canonical_factorized_radio_contract()
        or core.get("contract_sha256") != CANONICAL_FACTORIZED_RADIO_CONTRACT_SHA256
        or core.get("reliability_scalar_names")
        != list(FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES)
        or core.get("reliability_scalar_names_sha256")
        != FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES_SHA256
    ):
        raise ValueError("factorized RADIO core contract differs")
    canonical = core.get("canonical_feature")
    log_amplitude = core.get("log_amplitude")
    valid = core.get("valid")
    reliability = core.get("reliability")
    rows = int(xyz.shape[0])
    if (
        not torch.is_tensor(canonical)
        or canonical.dtype != torch.float16
        or canonical.shape != (rows, 1280)
        or not torch.is_tensor(log_amplitude)
        or log_amplitude.dtype != torch.float32
        or log_amplitude.shape != (rows,)
        or not torch.is_tensor(valid)
        or valid.dtype != torch.bool
        or valid.shape != (rows,)
        or not torch.is_tensor(reliability)
        or reliability.dtype != torch.float32
        or reliability.shape != (rows, len(FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES))
    ):
        raise ValueError("factorized RADIO output tensors differ")
    if not all(
        _all_finite(item) for item in (xyz, canonical, log_amplitude, reliability)
    ):
        raise ValueError("factorized RADIO cache contains non-finite values")
    invalid = ~valid
    if (
        bool((counts < 0).any())
        or not torch.equal(valid, counts > 0)
        or bool(canonical[invalid].ne(0).any())
        or bool(log_amplitude[invalid].ne(0).any())
        or bool(reliability[invalid].ne(0).any())
        or (builder_version == 1 and bool(reliability[:, 4].ne(0).any()))
        or (builder_version == 2 and bool((reliability[:, 4] < 0).any()))
        or (builder_version == 2 and bool((reliability[:, 4] > 1).any()))
    ):
        raise ValueError("factorized RADIO support or unknown-purity sentinel differs")
    expected_evidence = counts.float() / (counts.float() + 1.0)
    expected_evidence[invalid] = 0.0
    if not torch.equal(reliability[:, 3], expected_evidence):
        raise ValueError("factorized RADIO observation evidence differs")
    for start in range(0, rows, 4096):
        stop = min(start + 4096, rows)
        active = valid[start:stop]
        if not bool(active.any()):
            continue
        target = canonical[start:stop][active].float()
        target_log = log_amplitude[start:stop][active]
        rel = reliability[start:stop][active]
        if not torch.allclose(
            torch.linalg.vector_norm(target, dim=-1),
            torch.exp(target_log),
            atol=2e-3,
            rtol=2e-3,
        ):
            raise ValueError("factorized RADIO amplitude reconstruction differs")
        resultant, dispersion, amplitude_std, evidence, purity = rel.unbind(dim=1)
        if (
            bool(((resultant <= 0) | (resultant > 1)).any())
            or bool(((dispersion < 0) | (dispersion >= 1)).any())
            or bool((amplitude_std < 0).any())
            or bool(((evidence <= 0) | (evidence >= 1)).any())
            or (builder_version == 1 and bool(purity.ne(0).any()))
            or (builder_version == 2 and bool((purity < 0).any()))
            or (builder_version == 2 and bool((purity > 1).any()))
            or not torch.allclose(dispersion, 1.0 - resultant, atol=2e-6, rtol=2e-6)
        ):
            raise ValueError("factorized RADIO reliability semantics differ")
    metadata = dict(metadata)
    expected_builder_contract = (
        canonical_factorized_radio_builder_contract()
        if builder_version == 1
        else canonical_factorized_radio_builder_contract_v2()
    )
    expected_builder_contract_sha256 = (
        factorized_radio_builder_contract_sha256()
        if builder_version == 1
        else factorized_radio_builder_contract_v2_sha256()
    )
    if metadata.get("builder_contract") != expected_builder_contract or (
        metadata.get("builder_contract_sha256")
        != expected_builder_contract_sha256
    ):
        raise ValueError("factorized RADIO builder contract differs")
    fixed = {
        "construction": CANONICAL_FACTORIZED_RADIO_CONTRACT_NAME,
        "feature_space": "radio",
        "input_feature_space": "radio_raw_full",
        "feature_dim": 1280,
        "max_views_authority": 120,
        "aggregation_mode": (
            "raster_gaussian_top1"
            if builder_version == 1
            else "raster_marginal_responsibility"
        ),
        "raster_view_fusion": "contribution_mean",
        "registration_weight_mode": (
            "alpha_depth"
            if builder_version == 1
            else "exact_front_to_back_marginal_responsibility"
        ),
        "semantic_direction_storage": "derived_from_canonical_feature_not_persisted",
        "canonical_feature_dtype": "float16",
        "log_amplitude_dtype": "float32",
        "reliability_dtype": "float32",
        "robust_mpr": False,
        "benchmark_masks_opened": False,
        "benchmark_images_opened": False,
        "text_queries_opened": False,
        "query_independent": True,
    }
    mismatched = sorted(
        name for name, expected in fixed.items() if metadata.get(name) != expected
    )
    if mismatched:
        raise ValueError(f"factorized RADIO fixed metadata differs: {mismatched}")
    declared = int(metadata.get("num_declared_views", 0))
    selected_dataset = metadata.get("selected_dataset_indices")
    selected_frames = metadata.get("selected_frame_indices")
    if (
        declared <= 0
        or declared > 120
        or bool((counts > declared).any())
        or not isinstance(selected_dataset, list)
        or not isinstance(selected_frames, list)
        or len(selected_dataset) != declared
        or len(selected_frames) != declared
        or len(set(selected_dataset)) != declared
        or len(set(selected_frames)) != declared
    ):
        raise ValueError("factorized RADIO selected-view authority differs")
    expected_bundle = str(expected_feature_output_bundle_sha256)
    if (
        _SHA256.fullmatch(expected_bundle) is None
        or metadata.get("feature_output_bundle_sha256") != expected_bundle
    ):
        raise ValueError("factorized RADIO feature output bundle differs")
    for name in (
        "geometry_checkpoint_sha256",
        "feature_frame_manifest_sha256",
        "registration_responsibility_cache_sha256",
    ):
        if _SHA256.fullmatch(str(metadata.get(name, ""))) is None:
            raise ValueError(f"factorized RADIO {name} authority differs")
    expected_purity = {
        **(
            FACTORIZED_RADIO_TOP1_PURITY_AUTHORITY
            if builder_version == 1
            else FACTORIZED_RADIO_EXACT_MARGINAL_PURITY_AUTHORITY
        ),
        "registration_responsibility_cache_sha256": metadata[
            "registration_responsibility_cache_sha256"
        ],
    }
    if metadata.get("visibility_purity_authority") != expected_purity:
        raise ValueError("factorized RADIO visibility purity authority differs")
    return payload


def load_factorized_radio_training_cache(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_feature_output_bundle_sha256: str,
) -> FactorizedRadioTrainingCache:
    if _SHA256.fullmatch(str(expected_sha256)) is None:
        raise ValueError("factorized RADIO cache requires a caller-trusted SHA-256")
    payload, digest, source = load_torch_mapping(
        path,
        expected_sha256=expected_sha256,
        map_location="cpu",
        label="factorized RADIO training cache",
    )
    validated = validate_factorized_radio_training_payload(
        payload,
        expected_feature_output_bundle_sha256=expected_feature_output_bundle_sha256,
    )
    core = dict(validated["factorized_radio"])
    return FactorizedRadioTrainingCache(
        source=source,
        sha256=digest,
        xyz=validated["xyz"],
        geometry_fingerprint=dict(validated["geometry_fingerprint"]),
        canonical_feature=core["canonical_feature"],
        log_amplitude=core["log_amplitude"],
        valid=core["valid"],
        view_counts=validated["view_counts"],
        reliability=core["reliability"],
        reliability_scalar_names=tuple(core["reliability_scalar_names"]),
        reliability_scalar_names_sha256=str(core["reliability_scalar_names_sha256"]),
        metadata=dict(validated["metadata"]),
    )
