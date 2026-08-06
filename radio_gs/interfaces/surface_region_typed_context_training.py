"""Fail-closed authorities for cross-scene typed-context residual training.

This is a versioned overlay over the frozen AcceptedV2/full-scalar/adaptive
context artifacts.  It owns only train-split scalar normalization, OOD
routing, promotion evidence, and a small typed-context residual checkpoint.
It never embeds AcceptedV2 or teacher parameters and never permits target or
benchmark supervision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import torch

from radio_gs.interfaces.surface_region_full_scalar_contract import (
    SURFACE_REGION_FULL_SCALAR_DIM,
    SURFACE_REGION_FULL_SCALAR_NAMES,
    SURFACE_REGION_FULL_SCALAR_NAMES_SHA256,
)
from radio_gs.interfaces.surface_region_summary import (
    ACCEPTED_SURFACE_REGION_V2_ARCHITECTURE_SHA256,
    ACCEPTED_SURFACE_REGION_V2_CHECKPOINT_SHA256,
    ACCEPTED_SURFACE_REGION_V2_CONTRACT_SHA256,
    ACCEPTED_SURFACE_REGION_V2_PROVENANCE_SHA256,
    ACCEPTED_SURFACE_REGION_V2_STATE_DICT_SHA256,
    surface_region_state_dict_sha256,
)
from radio_gs.interfaces.surface_region_typed_context import (
    TYPED_CONTEXT_STATISTIC_DIM,
    TYPED_CONTEXT_STATISTIC_NAMES,
    TYPED_CONTEXT_STATISTIC_NAMES_SHA256,
)
from radio_gs.interfaces.surface_region_typed_context_adaptive import (
    ADAPTIVE_TYPED_CONTEXT_OVERLAY_CONTRACT_SHA256,
)
from radio_gs.models.surface_region_typed_context_residual import (
    SurfaceRegionAcceptedV2TypedContextResidualV1,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_json_object,
    load_torch_mapping,
    sha256_file,
    write_torch_noclobber,
)


TYPED_CONTEXT_TRAINING_SCHEMA_VERSION = 1
TYPED_CONTEXT_NORMALIZATION_SCHEMA = (
    "radio_gs.surface_region_typed_context_normalization.v1"
)
TYPED_CONTEXT_CERTIFICATE_SCHEMA = (
    "radio_gs.surface_region_typed_context_training_certificate.v1"
)
TYPED_CONTEXT_CHECKPOINT_SCHEMA = (
    "radio_gs.surface_region_accepted_v2_typed_context_residual_checkpoint.v1"
)
# Deliberately pinned like the existing full-scalar-v1 promotion boundary.
# A training-policy change requires a new certificate/checkpoint schema.
EXPECTED_TYPED_CONTEXT_TRAINING_CONTRACT_SHA256 = (
    "5cfb48867eba92bb4bb386c62da35d03c89a17941cb5c3511ae1e8558097fb58"
)
COMBINED_SCALAR_NAMES = tuple(SURFACE_REGION_FULL_SCALAR_NAMES) + tuple(
    f"typed_context.{name}" for name in TYPED_CONTEXT_STATISTIC_NAMES
)
COMBINED_SCALAR_DIM = SURFACE_REGION_FULL_SCALAR_DIM + TYPED_CONTEXT_STATISTIC_DIM
COMBINED_SCALAR_NAMES_SHA256 = canonical_json_sha256(list(COMBINED_SCALAR_NAMES))
_MAD_NORMAL_CONSISTENCY = 1.4826
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STATE_KEYS = frozenset(
    {
        "scalar_median",
        "scalar_robust_scale",
        "descriptor_projection.weight",
        "context_projection.weight",
        "scalar_projection.weight",
        "scalar_projection.bias",
        "fusion_projection.weight",
        "fusion_projection.bias",
        "residual_projection.weight",
        "residual_projection.bias",
    }
)


def accepted_v2_authority() -> dict[str, str]:
    return {
        "checkpoint_sha256": ACCEPTED_SURFACE_REGION_V2_CHECKPOINT_SHA256,
        "architecture_sha256": ACCEPTED_SURFACE_REGION_V2_ARCHITECTURE_SHA256,
        "state_dict_sha256": ACCEPTED_SURFACE_REGION_V2_STATE_DICT_SHA256,
        "provenance_sha256": ACCEPTED_SURFACE_REGION_V2_PROVENANCE_SHA256,
        "contract_sha256": ACCEPTED_SURFACE_REGION_V2_CONTRACT_SHA256,
    }


def typed_context_training_source_access() -> dict[str, bool]:
    return {
        "query_independent": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_queries_opened": False,
        "benchmark_labels_opened": False,
        "target_heldout_opened": False,
        "text_queries_opened": False,
        "per_scene_hyperparameters": False,
    }


def typed_context_normalization_contract() -> dict[str, Any]:
    return {
        "schema": TYPED_CONTEXT_NORMALIZATION_SCHEMA,
        "schema_version": TYPED_CONTEXT_TRAINING_SCHEMA_VERSION,
        "fit_split": "source_train_only",
        "fit_rows": "accepted_exact_overlap_and_adaptive_typed_context_valid",
        "dimension": COMBINED_SCALAR_DIM,
        "names": list(COMBINED_SCALAR_NAMES),
        "names_sha256": COMBINED_SCALAR_NAMES_SHA256,
        "full_scalar_names_sha256": SURFACE_REGION_FULL_SCALAR_NAMES_SHA256,
        "typed_context_statistic_names_sha256": (
            TYPED_CONTEXT_STATISTIC_NAMES_SHA256
        ),
        "location": "coordinatewise_lower_median",
        "dispersion": "coordinatewise_lower_median_absolute_deviation",
        "positive_mad_scale": "1.4826_times_mad",
        "zero_mad_scale": "one",
        "ood": {
            "score": "linf_absolute_train_robust_normalized_coordinate",
            "threshold": "maximum_source_train_fit_row_score",
            "constant_coordinate": "any_exact_deviation_is_ood",
            "comparison": "strict_greater_than",
            "fallback": "bitwise_immutable_accepted_v2_e0",
        },
        "boundary_balance": {
            "score": (
                "one_minus_mean_of_context_resultant_and_rescaled_"
                "context_to_anchor_cosine"
            ),
            "threshold": "source_train_fit_row_lower_median",
            "validation_contribution": False,
        },
        "validation_contribution": False,
        "source_access": typed_context_training_source_access(),
    }


TYPED_CONTEXT_NORMALIZATION_CONTRACT_SHA256 = canonical_json_sha256(
    typed_context_normalization_contract()
)


def _require_sha256(value: object, *, label: str) -> str:
    text = str(value)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _file_record_shape(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    path = str(value.get("path", ""))
    if not path:
        raise ValueError(f"{label} path is empty")
    return {"path": path, "sha256": _require_sha256(value["sha256"], label=label)}


@dataclass(frozen=True)
class TypedContextNormalizationResult:
    normalized: torch.Tensor
    ood_score: torch.Tensor
    ood_mask: torch.Tensor


def build_typed_context_normalization_authority(
    combined_source_train_scalars: torch.Tensor,
    source_train_fit_mask: torch.Tensor,
    *,
    source_state_cohort_authority_sha256: str,
    train_input_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    values = torch.as_tensor(combined_source_train_scalars).detach().float().cpu()
    mask = torch.as_tensor(source_train_fit_mask).detach().bool().cpu()
    if (
        values.ndim != 2
        or values.shape[1] != COMBINED_SCALAR_DIM
        or mask.shape != (values.shape[0],)
        or int(mask.sum()) < 2
        or not bool(torch.isfinite(values).all())
    ):
        raise ValueError("typed-context train-only normalization inputs differ")
    source_sha = _require_sha256(
        source_state_cohort_authority_sha256,
        label="typed-context source-state cohort authority",
    )
    records = [dict(item) for item in train_input_records]
    if len(records) != 24 or any(
        set(record) != {"scene_id", "training_shard", "adaptive_context"}
        for record in records
    ):
        raise ValueError("typed-context train normalization input records differ")
    frozen_records = []
    for record in records:
        scene = str(record["scene_id"])
        if not scene:
            raise ValueError("typed-context normalization scene is empty")
        frozen_records.append(
            {
                "scene_id": scene,
                "training_shard": _file_record_shape(
                    record["training_shard"], label="typed-context training shard"
                ),
                "adaptive_context": _file_record_shape(
                    record["adaptive_context"], label="adaptive context authority"
                ),
            }
        )
    if [item["scene_id"] for item in frozen_records] != sorted(
        item["scene_id"] for item in frozen_records
    ) or len({item["scene_id"] for item in frozen_records}) != len(frozen_records):
        raise ValueError("typed-context normalization scenes differ")
    selected = values[mask]
    median = selected.median(dim=0).values
    deviation = (selected - median).abs()
    mad = deviation.median(dim=0).values
    minimum = selected.min(dim=0).values
    maximum = selected.max(dim=0).values
    constant = minimum == maximum
    scale = torch.where(mad > 0, mad * _MAD_NORMAL_CONSISTENCY, torch.ones_like(mad))
    normalized = (selected - median) / scale
    variable = ~constant
    score = (
        normalized[:, variable].abs().amax(dim=1)
        if bool(variable.any())
        else torch.zeros(selected.shape[0])
    )
    resultant = selected[:, SURFACE_REGION_FULL_SCALAR_DIM + 7].clamp(0.0, 1.0)
    anchor_cosine = selected[:, SURFACE_REGION_FULL_SCALAR_DIM + 10].clamp(-1.0, 1.0)
    boundary_score = 1.0 - 0.5 * (resultant + 0.5 * (anchor_cosine + 1.0))
    return {
        "schema": TYPED_CONTEXT_NORMALIZATION_SCHEMA,
        "schema_version": TYPED_CONTEXT_TRAINING_SCHEMA_VERSION,
        "contract": typed_context_normalization_contract(),
        "contract_sha256": TYPED_CONTEXT_NORMALIZATION_CONTRACT_SHA256,
        "source_state_cohort_authority_sha256": source_sha,
        "train_input_records": frozen_records,
        "train_input_records_sha256": canonical_json_sha256(frozen_records),
        "source_combined_scalars_sha256": _tensor_sha256(values),
        "source_fit_mask_sha256": _tensor_sha256(mask),
        "source_count": int(mask.sum()),
        "median": median.contiguous(),
        "mad": mad.contiguous(),
        "robust_scale": scale.contiguous(),
        "constant_coordinate_mask": constant.contiguous(),
        "source_max_robust_linf": float(score.max()),
        "source_boundary_score_median": float(boundary_score.median()),
        "source_access": typed_context_training_source_access(),
    }


def validate_typed_context_normalization_authority(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("typed-context normalization authority must be a mapping")
    authority = dict(value)
    required = {
        "schema", "schema_version", "contract", "contract_sha256",
        "source_state_cohort_authority_sha256", "train_input_records",
        "train_input_records_sha256", "source_combined_scalars_sha256",
        "source_fit_mask_sha256", "source_count", "median", "mad",
        "robust_scale", "constant_coordinate_mask", "source_max_robust_linf",
        "source_boundary_score_median",
        "source_access",
    }
    if set(authority) != required:
        raise ValueError("typed-context normalization authority fields differ")
    if (
        authority.get("schema") != TYPED_CONTEXT_NORMALIZATION_SCHEMA
        or authority.get("schema_version") != TYPED_CONTEXT_TRAINING_SCHEMA_VERSION
        or authority.get("contract") != typed_context_normalization_contract()
        or authority.get("contract_sha256")
        != TYPED_CONTEXT_NORMALIZATION_CONTRACT_SHA256
        or authority.get("source_access") != typed_context_training_source_access()
    ):
        raise ValueError("typed-context normalization contract differs")
    for name in (
        "source_state_cohort_authority_sha256",
        "train_input_records_sha256",
        "source_combined_scalars_sha256",
        "source_fit_mask_sha256",
    ):
        _require_sha256(authority.get(name), label=name.replace("_", " "))
    records = authority.get("train_input_records")
    if not isinstance(records, list) or len(records) != 24:
        raise ValueError("typed-context normalization records differ")
    frozen_records = []
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "scene_id", "training_shard", "adaptive_context"
        }:
            raise ValueError("typed-context normalization record differs")
        scene = str(record.get("scene_id", ""))
        if not scene:
            raise ValueError("typed-context normalization scene is empty")
        frozen_records.append(
            {
                "scene_id": scene,
                "training_shard": _file_record_shape(
                    record["training_shard"], label="typed-context training shard"
                ),
                "adaptive_context": _file_record_shape(
                    record["adaptive_context"], label="adaptive context authority"
                ),
            }
        )
    if (
        frozen_records != sorted(frozen_records, key=lambda item: item["scene_id"])
        or len({item["scene_id"] for item in frozen_records}) != len(frozen_records)
        or canonical_json_sha256(frozen_records)
        != authority["train_input_records_sha256"]
    ):
        raise ValueError("typed-context normalization input authority differs")
    median = authority.get("median")
    mad = authority.get("mad")
    scale = authority.get("robust_scale")
    constant = authority.get("constant_coordinate_mask")
    if (
        not torch.is_tensor(median)
        or median.dtype != torch.float32
        or median.shape != (COMBINED_SCALAR_DIM,)
        or not torch.is_tensor(mad)
        or mad.dtype != torch.float32
        or mad.shape != median.shape
        or not torch.is_tensor(scale)
        or scale.dtype != torch.float32
        or scale.shape != median.shape
        or not torch.is_tensor(constant)
        or constant.dtype != torch.bool
        or constant.shape != median.shape
        or not all(bool(torch.isfinite(item).all()) for item in (median, mad, scale))
        or bool((mad < 0).any())
        or bool((scale <= 0).any())
        or not torch.equal(
            scale,
            torch.where(mad > 0, mad * _MAD_NORMAL_CONSISTENCY, torch.ones_like(mad)),
        )
        or int(authority.get("source_count", 0)) < 2
        or not isinstance(authority.get("source_max_robust_linf"), float)
        or not float(authority["source_max_robust_linf"]) >= 0.0
        or not isinstance(authority.get("source_boundary_score_median"), float)
        or not 0.0 <= float(authority["source_boundary_score_median"]) <= 1.0
    ):
        raise ValueError("typed-context normalization statistics differ")
    return {
        **authority,
        "train_input_records": frozen_records,
        "median": median.detach().cpu().contiguous(),
        "mad": mad.detach().cpu().contiguous(),
        "robust_scale": scale.detach().cpu().contiguous(),
        "constant_coordinate_mask": constant.detach().cpu().contiguous(),
    }


def typed_context_normalization_authority_sha256(
    value: Mapping[str, Any],
) -> str:
    """Hash normalized authority content independent of torch serialization."""

    frozen = validate_typed_context_normalization_authority(value)
    content = dict(frozen)
    for name in ("median", "mad", "robust_scale", "constant_coordinate_mask"):
        content[name] = {
            "tensor_channel_sha256": _tensor_sha256(frozen[name]),
        }
    return canonical_json_sha256(content)


def apply_typed_context_normalization(
    combined_scalars: torch.Tensor,
    authority: Mapping[str, Any],
) -> TypedContextNormalizationResult:
    frozen = validate_typed_context_normalization_authority(authority)
    values = torch.as_tensor(combined_scalars).detach().float().cpu()
    squeeze = values.ndim == 1
    if squeeze:
        values = values[None]
    if (
        values.ndim != 2
        or values.shape[1] != COMBINED_SCALAR_DIM
        or not bool(torch.isfinite(values).all())
    ):
        raise ValueError("typed-context normalization values differ")
    normalized = (values - frozen["median"]) / frozen["robust_scale"]
    variable = ~frozen["constant_coordinate_mask"]
    score = (
        normalized[:, variable].abs().amax(dim=1)
        if bool(variable.any())
        else torch.zeros(values.shape[0])
    )
    constant_deviation = (
        (values[:, frozen["constant_coordinate_mask"]]
         != frozen["median"][frozen["constant_coordinate_mask"]]).any(dim=1)
        if bool(frozen["constant_coordinate_mask"].any())
        else torch.zeros(values.shape[0], dtype=torch.bool)
    )
    ood = constant_deviation | (score > float(frozen["source_max_robust_linf"]))
    result = TypedContextNormalizationResult(normalized, score, ood)
    if not squeeze:
        return result
    return TypedContextNormalizationResult(
        result.normalized[0], result.ood_score[0], result.ood_mask[0]
    )


def typed_context_checkpoint_contract() -> dict[str, Any]:
    return {
        "schema": TYPED_CONTEXT_CHECKPOINT_SCHEMA,
        "schema_version": TYPED_CONTEXT_TRAINING_SCHEMA_VERSION,
        "model_class": "SurfaceRegionAcceptedV2TypedContextResidualV1",
        "accepted_v2_authority": accepted_v2_authority(),
        "adaptive_context_contract_sha256": (
            ADAPTIVE_TYPED_CONTEXT_OVERLAY_CONTRACT_SHA256
        ),
        "normalization_contract_sha256": (
            TYPED_CONTEXT_NORMALIZATION_CONTRACT_SHA256
        ),
        "bounds": {"max_angle_radians": 0.15, "max_alpha": 0.25},
        "write": "atomic_first_writer_wins",
        "load": "caller_sha_exact_keys_strict_state_no_fallback",
        "fallback": "inactive_or_ood_bitwise_immutable_accepted_v2_e0",
        "source_access": typed_context_training_source_access(),
    }


TYPED_CONTEXT_CHECKPOINT_CONTRACT_SHA256 = canonical_json_sha256(
    typed_context_checkpoint_contract()
)


def _copy_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().contiguous().clone()
        for name, value in model.state_dict().items()
    }


def build_typed_context_checkpoint_payload(
    model: SurfaceRegionAcceptedV2TypedContextResidualV1,
    *,
    normalization_authority: Mapping[str, Any],
    normalization_file_sha256: str,
    certificate: Mapping[str, Any],
    certificate_file_sha256: str,
) -> dict[str, Any]:
    if type(model) is not SurfaceRegionAcceptedV2TypedContextResidualV1:
        raise TypeError("typed-context checkpoint model class differs")
    normalization = validate_typed_context_normalization_authority(
        normalization_authority
    )
    cert = validate_typed_context_training_certificate(certificate)
    state = _copy_state(model)
    if set(state) != _STATE_KEYS:
        raise ValueError("typed-context checkpoint state keys differ")
    if not torch.equal(state["scalar_median"], normalization["median"]) or not torch.equal(
        state["scalar_robust_scale"], normalization["robust_scale"]
    ):
        raise ValueError("typed-context checkpoint normalization buffers differ")
    architecture = model.architecture()
    state_sha = surface_region_state_dict_sha256(state)
    norm_sha = _require_sha256(normalization_file_sha256, label="normalization file")
    cert_sha = _require_sha256(certificate_file_sha256, label="certificate file")
    if (
        cert["model_authority"]["architecture"] != architecture
        or cert["model_authority"]["state_dict_sha256"] != state_sha
        or cert["normalization_authority"]["sha256"] != norm_sha
    ):
        raise ValueError("typed-context certificate/checkpoint binding differs")
    payload = {
        "schema": TYPED_CONTEXT_CHECKPOINT_SCHEMA,
        "schema_version": TYPED_CONTEXT_TRAINING_SCHEMA_VERSION,
        "contract": typed_context_checkpoint_contract(),
        "contract_sha256": TYPED_CONTEXT_CHECKPOINT_CONTRACT_SHA256,
        "accepted_v2_authority": accepted_v2_authority(),
        "normalization_authority": {
            "sha256": norm_sha,
            "source_state_cohort_authority_sha256": normalization[
                "source_state_cohort_authority_sha256"
            ],
        },
        "training_certificate_sha256": cert_sha,
        "model_class": type(model).__name__,
        "model_architecture": architecture,
        "model_architecture_sha256": canonical_json_sha256(architecture),
        "model_state_dict": state,
        "model_state_dict_sha256": state_sha,
        "source_access": typed_context_training_source_access(),
    }
    validate_typed_context_checkpoint_payload(
        payload,
        normalization_authority=normalization,
        expected_normalization_file_sha256=norm_sha,
        certificate=cert,
        expected_certificate_file_sha256=cert_sha,
    )
    return payload


def validate_typed_context_checkpoint_payload(
    value: object,
    *,
    normalization_authority: Mapping[str, Any],
    expected_normalization_file_sha256: str,
    certificate: Mapping[str, Any],
    expected_certificate_file_sha256: str,
) -> tuple[dict[str, Any], SurfaceRegionAcceptedV2TypedContextResidualV1]:
    if not isinstance(value, Mapping):
        raise ValueError("typed-context checkpoint must be a mapping")
    payload = dict(value)
    required = {
        "schema", "schema_version", "contract", "contract_sha256",
        "accepted_v2_authority", "normalization_authority",
        "training_certificate_sha256", "model_class", "model_architecture",
        "model_architecture_sha256", "model_state_dict",
        "model_state_dict_sha256", "source_access",
    }
    if set(payload) != required:
        raise ValueError("typed-context checkpoint fields differ")
    if (
        payload.get("schema") != TYPED_CONTEXT_CHECKPOINT_SCHEMA
        or payload.get("schema_version") != TYPED_CONTEXT_TRAINING_SCHEMA_VERSION
        or payload.get("contract") != typed_context_checkpoint_contract()
        or payload.get("contract_sha256") != TYPED_CONTEXT_CHECKPOINT_CONTRACT_SHA256
        or payload.get("accepted_v2_authority") != accepted_v2_authority()
        or payload.get("model_class")
        != "SurfaceRegionAcceptedV2TypedContextResidualV1"
        or payload.get("source_access") != typed_context_training_source_access()
    ):
        raise ValueError("typed-context checkpoint contract differs")
    normalization = validate_typed_context_normalization_authority(
        normalization_authority
    )
    cert = validate_typed_context_training_certificate(certificate)
    norm_sha = _require_sha256(
        expected_normalization_file_sha256, label="normalization file"
    )
    cert_sha = _require_sha256(
        expected_certificate_file_sha256, label="certificate file"
    )
    if payload.get("normalization_authority") != {
        "sha256": norm_sha,
        "source_state_cohort_authority_sha256": normalization[
            "source_state_cohort_authority_sha256"
        ],
    } or payload.get("training_certificate_sha256") != cert_sha:
        raise ValueError("typed-context checkpoint caller authority differs")
    architecture = payload.get("model_architecture")
    if (
        not isinstance(architecture, Mapping)
        or canonical_json_sha256(dict(architecture))
        != payload.get("model_architecture_sha256")
    ):
        raise ValueError("typed-context checkpoint architecture differs")
    model = SurfaceRegionAcceptedV2TypedContextResidualV1(
        scalar_median=normalization["median"],
        scalar_robust_scale=normalization["robust_scale"],
        max_angle_radians=0.15,
        max_alpha=0.25,
    )
    if model.architecture() != dict(architecture):
        raise ValueError("typed-context checkpoint architecture is unsupported")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping) or set(state) != _STATE_KEYS or any(
        not torch.is_tensor(item) for item in state.values()
    ):
        raise ValueError("typed-context checkpoint state differs")
    frozen_state = {
        str(name): tensor.detach().cpu().contiguous().clone()
        for name, tensor in state.items()
    }
    state_sha = surface_region_state_dict_sha256(frozen_state)
    if (
        state_sha != payload.get("model_state_dict_sha256")
        or cert["model_authority"]["state_dict_sha256"] != state_sha
        or cert["model_authority"]["architecture"] != dict(architecture)
        or cert["normalization_authority"]["sha256"] != norm_sha
    ):
        raise ValueError("typed-context checkpoint state/certificate differs")
    model.load_state_dict(frozen_state, strict=True)
    if not torch.equal(model.scalar_median, normalization["median"]) or not torch.equal(
        model.scalar_robust_scale, normalization["robust_scale"]
    ):
        raise ValueError("typed-context checkpoint loaded buffers differ")
    return payload, model


def typed_context_certificate_contract() -> dict[str, Any]:
    return {
        "schema": TYPED_CONTEXT_CERTIFICATE_SCHEMA,
        "schema_version": TYPED_CONTEXT_TRAINING_SCHEMA_VERSION,
        "selection_split": "source_validation_only",
        "scene_macro": True,
        "base_vs_residual_delta_required": True,
        "automatic_fallback": "select_epoch_zero_if_no_trained_epoch_nonregresses",
        "validation_gradients": False,
        "input_files": "caller_sha_bound_training_shard_and_adaptive_context_pairs",
        "source_access": typed_context_training_source_access(),
    }


TYPED_CONTEXT_CERTIFICATE_CONTRACT_SHA256 = canonical_json_sha256(
    typed_context_certificate_contract()
)


def _certificate_content_sha256(value: Mapping[str, Any]) -> str:
    content = dict(value)
    content.pop("content_sha256", None)
    return canonical_json_sha256(content)


def build_typed_context_training_certificate(
    *,
    training_contract: Mapping[str, Any],
    model: SurfaceRegionAcceptedV2TypedContextResidualV1,
    normalization_authority_file: Mapping[str, str],
    cohort_authority: Mapping[str, Any],
    external_manifests: Mapping[str, Any],
    input_records_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    selected_epoch: int,
    selected_validation: Mapping[str, Any],
) -> dict[str, Any]:
    state = _copy_state(model)
    records = {
        split: [
            {
                "scene_id": str(item["scene_id"]),
                "training_shard": _file_record_shape(
                    item["training_shard"], label=f"{split} training shard"
                ),
                "adaptive_context": _file_record_shape(
                    item["adaptive_context"], label=f"{split} adaptive context"
                ),
            }
            for item in input_records_by_split[split]
        ]
        for split in ("source_train", "source_validation")
    }
    payload: dict[str, Any] = {
        "schema": TYPED_CONTEXT_CERTIFICATE_SCHEMA,
        "schema_version": TYPED_CONTEXT_TRAINING_SCHEMA_VERSION,
        "contract": typed_context_certificate_contract(),
        "contract_sha256": TYPED_CONTEXT_CERTIFICATE_CONTRACT_SHA256,
        "training_contract": dict(training_contract),
        "training_contract_sha256": canonical_json_sha256(dict(training_contract)),
        "model_authority": {
            "class": type(model).__name__,
            "architecture": model.architecture(),
            "architecture_sha256": canonical_json_sha256(model.architecture()),
            "state_dict_sha256": surface_region_state_dict_sha256(state),
        },
        "normalization_authority": _file_record_shape(
            normalization_authority_file, label="typed-context normalization"
        ),
        "cohort_authority": dict(cohort_authority),
        "external_manifests": dict(external_manifests),
        "input_records_by_split": records,
        "selected_epoch": int(selected_epoch),
        "selected_validation": dict(selected_validation),
        "source_access": typed_context_training_source_access(),
    }
    payload["content_sha256"] = _certificate_content_sha256(payload)
    return validate_typed_context_training_certificate(payload)


def validate_typed_context_training_certificate(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("typed-context certificate must be a mapping")
    cert = dict(value)
    required = {
        "schema", "schema_version", "contract", "contract_sha256",
        "training_contract", "training_contract_sha256", "model_authority",
        "normalization_authority", "cohort_authority", "external_manifests",
        "input_records_by_split", "selected_epoch", "selected_validation",
        "source_access", "content_sha256",
    }
    if set(cert) != required:
        raise ValueError("typed-context certificate fields differ")
    if (
        cert.get("schema") != TYPED_CONTEXT_CERTIFICATE_SCHEMA
        or cert.get("schema_version") != TYPED_CONTEXT_TRAINING_SCHEMA_VERSION
        or cert.get("contract") != typed_context_certificate_contract()
        or cert.get("contract_sha256") != TYPED_CONTEXT_CERTIFICATE_CONTRACT_SHA256
        or cert.get("source_access") != typed_context_training_source_access()
        or _certificate_content_sha256(cert) != cert.get("content_sha256")
    ):
        raise ValueError("typed-context certificate contract/content differs")
    training_contract = cert.get("training_contract")
    if not isinstance(training_contract, Mapping) or canonical_json_sha256(
        dict(training_contract)
    ) != cert.get("training_contract_sha256") or cert.get(
        "training_contract_sha256"
    ) != EXPECTED_TYPED_CONTEXT_TRAINING_CONTRACT_SHA256:
        raise ValueError("typed-context certificate training contract differs")
    model = cert.get("model_authority")
    if not isinstance(model, Mapping) or set(model) != {
        "class", "architecture", "architecture_sha256", "state_dict_sha256"
    } or model.get("class") != "SurfaceRegionAcceptedV2TypedContextResidualV1":
        raise ValueError("typed-context certificate model authority differs")
    if not isinstance(model.get("architecture"), Mapping) or canonical_json_sha256(
        dict(model["architecture"])
    ) != model.get("architecture_sha256"):
        raise ValueError("typed-context certificate architecture differs")
    _require_sha256(model.get("state_dict_sha256"), label="model state")
    cert["normalization_authority"] = _file_record_shape(
        cert.get("normalization_authority"), label="typed-context normalization"
    )
    cohort = cert.get("cohort_authority")
    if not isinstance(cohort, Mapping) or set(cohort) != {
        "file", "authority_sha256"
    }:
        raise ValueError("typed-context certificate cohort authority differs")
    frozen_cohort = {
        "file": _file_record_shape(
            cohort["file"], label="typed-context cohort authority"
        ),
        "authority_sha256": _require_sha256(
            cohort["authority_sha256"], label="typed-context cohort content"
        ),
    }
    manifests = cert.get("external_manifests")
    if not isinstance(manifests, Mapping) or set(manifests) != {
        "source_state", "teacher", "benchmark_exclusion"
    }:
        raise ValueError("typed-context certificate external manifests differ")
    frozen_manifests = {
        name: _file_record_shape(
            manifests[name], label=f"typed-context {name} manifest"
        )
        for name in ("source_state", "teacher", "benchmark_exclusion")
    }
    records_by_split = cert.get("input_records_by_split")
    if not isinstance(records_by_split, Mapping) or set(records_by_split) != {
        "source_train", "source_validation"
    }:
        raise ValueError("typed-context certificate input splits differ")
    frozen: dict[str, list[dict[str, Any]]] = {}
    for split, expected_count in (("source_train", 24), ("source_validation", 8)):
        rows = records_by_split[split]
        if not isinstance(rows, list) or len(rows) != expected_count:
            raise ValueError("typed-context certificate scene count differs")
        frozen[split] = []
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {
                "scene_id", "training_shard", "adaptive_context"
            }:
                raise ValueError("typed-context certificate input record differs")
            frozen[split].append(
                {
                    "scene_id": str(row["scene_id"]),
                    "training_shard": _file_record_shape(
                        row["training_shard"], label=f"{split} training shard"
                    ),
                    "adaptive_context": _file_record_shape(
                        row["adaptive_context"], label=f"{split} adaptive context"
                    ),
                }
            )
        scenes = [row["scene_id"] for row in frozen[split]]
        if scenes != sorted(scenes) or len(set(scenes)) != len(scenes):
            raise ValueError("typed-context certificate scene order differs")
    selected = cert.get("selected_validation")
    if (
        int(cert.get("selected_epoch", -1)) < 0
        or not isinstance(selected, Mapping)
        or selected.get("non_regression_passed") is not True
        or selected.get("validation_no_grad") is not True
        or not isinstance(selected.get("per_scene"), Mapping)
        or len(selected["per_scene"]) != 8
        or sorted(selected["per_scene"])
        != [row["scene_id"] for row in frozen["source_validation"]]
    ):
        raise ValueError("typed-context certificate validation gate differs")
    return {
        **cert,
        "cohort_authority": frozen_cohort,
        "external_manifests": frozen_manifests,
        "input_records_by_split": frozen,
    }


def write_typed_context_checkpoint(
    path: str | Path,
    model: SurfaceRegionAcceptedV2TypedContextResidualV1,
    **kwargs: Any,
) -> tuple[Path, str]:
    payload = build_typed_context_checkpoint_payload(model, **kwargs)
    output = write_torch_noclobber(path, payload)
    return output, sha256_file(output)


def load_typed_context_checkpoint(
    path: str | Path,
    *,
    expected_checkpoint_sha256: str,
    normalization_path: str | Path,
    expected_normalization_sha256: str,
    certificate_path: str | Path,
    expected_certificate_sha256: str,
) -> tuple[dict[str, Any], SurfaceRegionAcceptedV2TypedContextResidualV1]:
    normalization_raw, _, _ = load_torch_mapping(
        normalization_path,
        expected_sha256=_require_sha256(
            expected_normalization_sha256, label="normalization"
        ),
        map_location="cpu",
        label="typed-context normalization",
    )
    normalization = validate_typed_context_normalization_authority(normalization_raw)
    certificate_raw, _, _ = load_json_object(
        certificate_path,
        expected_sha256=_require_sha256(
            expected_certificate_sha256, label="certificate"
        ),
        label="typed-context training certificate",
    )
    certificate = validate_typed_context_training_certificate(certificate_raw)
    payload, _, _ = load_torch_mapping(
        path,
        expected_sha256=_require_sha256(
            expected_checkpoint_sha256, label="checkpoint"
        ),
        map_location="cpu",
        label="typed-context residual checkpoint",
    )
    return validate_typed_context_checkpoint_payload(
        payload,
        normalization_authority=normalization,
        expected_normalization_file_sha256=expected_normalization_sha256,
        certificate=certificate,
        expected_certificate_file_sha256=expected_certificate_sha256,
    )


__all__ = [
    "COMBINED_SCALAR_DIM",
    "COMBINED_SCALAR_NAMES",
    "COMBINED_SCALAR_NAMES_SHA256",
    "EXPECTED_TYPED_CONTEXT_TRAINING_CONTRACT_SHA256",
    "TYPED_CONTEXT_CERTIFICATE_CONTRACT_SHA256",
    "TYPED_CONTEXT_CHECKPOINT_CONTRACT_SHA256",
    "TYPED_CONTEXT_NORMALIZATION_CONTRACT_SHA256",
    "TypedContextNormalizationResult",
    "accepted_v2_authority",
    "apply_typed_context_normalization",
    "build_typed_context_checkpoint_payload",
    "build_typed_context_normalization_authority",
    "build_typed_context_training_certificate",
    "load_typed_context_checkpoint",
    "typed_context_checkpoint_contract",
    "typed_context_normalization_contract",
    "typed_context_training_source_access",
    "typed_context_normalization_authority_sha256",
    "validate_typed_context_checkpoint_payload",
    "validate_typed_context_normalization_authority",
    "validate_typed_context_training_certificate",
    "write_typed_context_checkpoint",
]
