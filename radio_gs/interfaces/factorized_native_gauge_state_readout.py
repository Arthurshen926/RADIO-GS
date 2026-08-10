"""Fail-closed interface for the factorized-native gauge/state readout."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from radio_gs.interfaces.factorized_primitive_state import (
    FACTORIZED_PRIMITIVE_STATE_CONTRACT_SHA256,
    FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES,
    FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES_SHA256,
    FactorizedPrimitiveState,
)
from radio_gs.models.factorized_native_gauge_state_readout import (
    FACTORIZED_NATIVE_READOUT_ARMS,
    FactorizedNativeGaugeStateReadout,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


INTERFACE_SCHEMA = "radio_gs.factorized_native_gauge_state_readout_interface.v1"
INTERFACE_SCHEMA_VERSION = 1
NORMALIZATION_SCHEMA = "radio_gs.factorized_native_gauge_state_normalization.v1"
NORMALIZATION_SCHEMA_VERSION = 1
STATE_DIM = len(FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES)
_MAD_SCALE = 1.4826


@dataclass(frozen=True)
class FactorizedNativeRegionInputs:
    unit_direction: torch.Tensor
    log_amplitude: torch.Tensor
    state: torch.Tensor
    state_known_mask: torch.Tensor
    token_mask: torch.Tensor
    anchor_index: torch.Tensor


def source_access() -> dict[str, bool]:
    return {
        "query_independent": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_queries_opened": False,
        "benchmark_labels_opened": False,
        "target_heldout_opened": False,
        "text_queries_opened": False,
        "runtime_query_strings_consumed": False,
        "scene_identifiers_consumed_by_model": False,
        "per_scene_hyperparameters": False,
    }


def interface_contract() -> dict[str, Any]:
    return {
        "schema": INTERFACE_SCHEMA,
        "schema_version": INTERFACE_SCHEMA_VERSION,
        "parent_factorized_state_contract_sha256": (
            FACTORIZED_PRIMITIVE_STATE_CONTRACT_SHA256
        ),
        "inputs": {
            "unit_direction": {
                "shape": "B,T,1280",
                "gauge": "unit_l2",
                "padding": "exact_zero",
            },
            "log_amplitude": {
                "shape": "B,T",
                "source": "FactorizedPrimitiveState.predicted_log_amplitude",
                "raw_vector_reconstruction_allowed": False,
            },
            "state": {
                "shape": "B,T,6",
                "names": list(FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES),
                "names_sha256": FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES_SHA256,
                "unknown_values": "exact_zero",
            },
            "state_known_mask": {
                "shape": "B,T,6",
                "dtype": "bool",
                "purity_value_availability_is_explicit": True,
            },
            "routing": ["token_mask", "anchor_index"],
        },
        "amplitude_uniqueness": {
            "state_column_zero_equals_separate_log_amplitude": True,
            "state_column_zero_excluded_from_full_state_encoder": True,
            "direction_times_amplitude_prohibited": True,
            "legacy_raw_radio_vector_prohibited": True,
        },
        "arms": list(FACTORIZED_NATIVE_READOUT_ARMS),
        "output": {
            "summary_token": "B,1280",
            "compatibility": (
                "frozen_official_c_radio_v4_siglip2_g_summary_head_then_"
                "unit_1536_descriptor"
            ),
        },
        "training": {
            "query_free": True,
            "global_scene_independent_parameters": True,
            "normalization_fit": "source_train_only_known_values",
            "validation_contributes_to_normalization": False,
        },
        "legacy_accepted_v2_default_changed": False,
        "source_access": source_access(),
    }


INTERFACE_CONTRACT_SHA256 = canonical_json_sha256(interface_contract())


def factorized_state_known_mask(state: FactorizedPrimitiveState) -> torch.Tensor:
    """Return one explicit availability bit for each of the six state columns."""

    if not isinstance(state, FactorizedPrimitiveState):
        raise TypeError("known-mask construction requires FactorizedPrimitiveState")
    count = int(state.global_rows.numel())
    known = torch.ones(count, STATE_DIM, dtype=torch.bool)
    known[:, 4] = state.visibility_purity_known.detach().bool().cpu()
    return known


def gather_factorized_native_region_inputs(
    state: FactorizedPrimitiveState,
    region_rows: torch.Tensor,
    token_mask: torch.Tensor,
    anchor_index: torch.Tensor | int,
) -> FactorizedNativeRegionInputs:
    """Gather padded region tokens without manufacturing a raw feature vector."""

    if not isinstance(state, FactorizedPrimitiveState):
        raise TypeError("region input gathering requires FactorizedPrimitiveState")
    rows = torch.as_tensor(region_rows).detach().long().cpu()
    declared = torch.as_tensor(token_mask).detach().bool().cpu()
    squeeze = rows.ndim == 1
    if squeeze:
        rows, declared = rows[None], declared[None]
    if (
        rows.ndim != 2
        or declared.shape != rows.shape
        or not bool(declared.any(dim=1).all())
        or bool(rows[~declared].ne(-1).any())
        or bool((rows[declared] < 0).any())
        or bool((rows[declared] >= state.valid.numel()).any())
    ):
        raise ValueError("factorized-native region rows/mask differ")
    anchor = torch.as_tensor(anchor_index).detach().long().cpu().reshape(-1)
    if anchor.numel() == 1:
        anchor = anchor.expand(rows.shape[0])
    batch = torch.arange(rows.shape[0])
    if (
        anchor.shape != (rows.shape[0],)
        or bool((anchor < 0).any())
        or bool((anchor >= rows.shape[1]).any())
        or not bool(declared[batch, anchor].all())
    ):
        raise ValueError("factorized-native region anchor differs")

    safe_rows = rows.clamp_min(0)
    exact_mask = declared & state.valid[safe_rows]
    if not bool(exact_mask[batch, anchor].all()):
        raise ValueError("factorized-native region anchor lacks exact state")
    global_to_compact = torch.full((state.valid.numel(),), -1, dtype=torch.long)
    global_to_compact[state.global_rows] = torch.arange(state.global_rows.numel())
    compact = global_to_compact[safe_rows]
    if bool((compact[exact_mask] < 0).any()):
        raise RuntimeError("factorized-native compact-row map is incomplete")
    safe_compact = compact.clamp_min(0)

    directions = state.semantic_direction.float().cpu()[safe_compact]
    scalars = state.scalar_encoding_input().float().cpu()[safe_compact]
    known_source = factorized_state_known_mask(state)
    known = known_source[safe_compact]
    directions = directions.masked_fill(~exact_mask[..., None], 0.0)
    scalars = scalars.masked_fill(~exact_mask[..., None], 0.0)
    known = known & exact_mask[..., None]
    amplitude = scalars[..., 0].clone()
    result = FactorizedNativeRegionInputs(
        unit_direction=directions,
        log_amplitude=amplitude,
        state=scalars,
        state_known_mask=known,
        token_mask=exact_mask,
        anchor_index=anchor,
    )
    if not squeeze:
        return result
    return FactorizedNativeRegionInputs(
        unit_direction=result.unit_direction[0],
        log_amplitude=result.log_amplitude[0],
        state=result.state[0],
        state_known_mask=result.state_known_mask[0],
        token_mask=result.token_mask[0],
        anchor_index=result.anchor_index[:1],
    )


def build_source_normalization(
    states: Sequence[FactorizedPrimitiveState],
    *,
    source_state_cohort_authority_sha256: str,
) -> dict[str, Any]:
    """Fit robust scalar normalization once from source-train compact rows."""

    if not states:
        raise ValueError("source normalization requires at least one state")
    scalar_rows: list[torch.Tensor] = []
    known_rows: list[torch.Tensor] = []
    for state in states:
        if not isinstance(state, FactorizedPrimitiveState):
            raise TypeError("source normalization states differ")
        scalar_rows.append(state.scalar_encoding_input().detach().float().cpu())
        known_rows.append(factorized_state_known_mask(state))
    values = torch.cat(scalar_rows)
    known = torch.cat(known_rows)
    if values.shape != known.shape or values.shape[1] != STATE_DIM:
        raise ValueError("source normalization state layout differs")
    medians = torch.zeros(STATE_DIM)
    scales = torch.ones(STATE_DIM)
    known_counts: list[int] = []
    for column in range(STATE_DIM):
        selected = values[:, column][known[:, column]]
        if selected.numel() <= 0:
            raise ValueError("source normalization has an entirely unknown column")
        median = selected.median()
        mad = (selected - median).abs().median()
        medians[column] = median
        scales[column] = mad * _MAD_SCALE if float(mad) > 0 else 1.0
        known_counts.append(int(selected.numel()))
    payload = {
        "schema": NORMALIZATION_SCHEMA,
        "schema_version": NORMALIZATION_SCHEMA_VERSION,
        "interface_contract_sha256": INTERFACE_CONTRACT_SHA256,
        "source_state_cohort_authority_sha256": str(
            source_state_cohort_authority_sha256
        ),
        "state_names": list(FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES),
        "state_names_sha256": FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES_SHA256,
        "state_median": medians,
        "state_robust_scale": scales,
        "log_amplitude_median": medians[:1].clone(),
        "log_amplitude_robust_scale": scales[:1].clone(),
        "known_count_by_state_column": known_counts,
        "fit_scene_count": len(states),
        "source_access": source_access(),
    }
    return validate_source_normalization(payload)


def validate_source_normalization(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("factorized-native normalization must be a mapping")
    payload = dict(value)
    required = {
        "schema",
        "schema_version",
        "interface_contract_sha256",
        "source_state_cohort_authority_sha256",
        "state_names",
        "state_names_sha256",
        "state_median",
        "state_robust_scale",
        "log_amplitude_median",
        "log_amplitude_robust_scale",
        "known_count_by_state_column",
        "fit_scene_count",
        "source_access",
    }
    state_median = payload.get("state_median")
    state_scale = payload.get("state_robust_scale")
    log_median = payload.get("log_amplitude_median")
    log_scale = payload.get("log_amplitude_robust_scale")
    counts = payload.get("known_count_by_state_column")
    if (
        set(payload) != required
        or payload.get("schema") != NORMALIZATION_SCHEMA
        or payload.get("schema_version") != NORMALIZATION_SCHEMA_VERSION
        or payload.get("interface_contract_sha256") != INTERFACE_CONTRACT_SHA256
        or payload.get("state_names")
        != list(FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES)
        or payload.get("state_names_sha256")
        != FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES_SHA256
        or payload.get("source_access") != source_access()
        or not torch.is_tensor(state_median)
        or state_median.shape != (STATE_DIM,)
        or not torch.is_tensor(state_scale)
        or state_scale.shape != (STATE_DIM,)
        or not torch.is_tensor(log_median)
        or log_median.shape != (1,)
        or not torch.is_tensor(log_scale)
        or log_scale.shape != (1,)
        or not bool(torch.isfinite(state_median).all())
        or not bool(torch.isfinite(state_scale).all())
        or not bool((state_scale > 0).all())
        or not torch.equal(log_median, state_median[:1])
        or not torch.equal(log_scale, state_scale[:1])
        or not isinstance(counts, list)
        or len(counts) != STATE_DIM
        or any(not isinstance(count, int) or count <= 0 for count in counts)
        or int(payload.get("fit_scene_count", 0)) <= 0
        or len(str(payload.get("source_state_cohort_authority_sha256", ""))) != 64
    ):
        raise ValueError("factorized-native normalization contract differs")
    return payload


def build_model(
    arm: str,
    normalization: Mapping[str, Any],
    *,
    hidden_dim: int = 128,
) -> FactorizedNativeGaugeStateReadout:
    frozen = validate_source_normalization(normalization)
    return FactorizedNativeGaugeStateReadout(
        arm=arm,
        hidden_dim=hidden_dim,
        log_amplitude_median=frozen["log_amplitude_median"],
        log_amplitude_robust_scale=frozen["log_amplitude_robust_scale"],
        state_median=frozen["state_median"],
        state_robust_scale=frozen["state_robust_scale"],
    )


__all__ = [
    "FactorizedNativeRegionInputs",
    "INTERFACE_CONTRACT_SHA256",
    "INTERFACE_SCHEMA",
    "INTERFACE_SCHEMA_VERSION",
    "NORMALIZATION_SCHEMA",
    "NORMALIZATION_SCHEMA_VERSION",
    "build_model",
    "build_source_normalization",
    "factorized_state_known_mask",
    "gather_factorized_native_region_inputs",
    "interface_contract",
    "source_access",
    "validate_source_normalization",
]
