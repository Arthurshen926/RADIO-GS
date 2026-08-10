#!/usr/bin/env python3
"""All-available O1/O2 with online query-free agreement and source LOO."""

from __future__ import annotations

from collections.abc import Mapping
import copy
from pathlib import Path
from typing import Any

import torch

from radio_gs.interfaces.lerf_reliability_geodesic_budget import (
    VIEW_AGREEMENT_SCALAR,
    VIEW_AGREEMENT_SHA256_FIELD,
)
from radio_gs.scripts import materialize_lerf_o1_o2_all_available_streaming as all_view
from radio_gs.scripts import materialize_lerf_o1_o2_streaming as legacy
from radio_gs.scripts import materialize_lerf_o1_o2_teacher_agreement_streaming_v2 as v2
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
)


AUTHORITY_SCHEMA = (
    "radio_gs.lerf_o1_o2_all_available_teacher_agreement_execution.v2"
)
MEAN_SCHEMA = "radio_gs.lerf_source_teacher_mean_siglip_all_available.v2"
RESULT_SCHEMA = "radio_gs.lerf_o1_o2_all_available_teacher_agreement_result.v2"
SCHEMA_VERSION = 2

BASE_IMPLEMENTATION = file_record(Path(all_view.__file__).resolve())
LOO_IMPLEMENTATION = file_record(Path(v2.__file__).resolve())
ENTRYPOINT_IMPLEMENTATION = file_record(Path(__file__).resolve())
_BASE_METHOD_CONTRACT = all_view.method_contract
_BASE_PREPARE_INPUTS = all_view.prepare_inputs
_BASE_MATERIALIZE = all_view.materialize
_BASE_CHUNKED_TEACHER_MEAN = all_view._chunked_teacher_mean
_BASE_RAW_CACHE = all_view._raw_cache
_CORE_WRITE_TORCH = legacy.write_torch_noclobber
_CORE_WRITE_JSON = legacy.write_frozen_json

_base_descriptor_state: torch.Tensor | None = None
_statistics_state: dict[str, Any] | None = None


def method_contract() -> dict[str, Any]:
    contract = copy.deepcopy(_BASE_METHOD_CONTRACT())
    contract.update(
        {
            "schema": AUTHORITY_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "all_available_base_implementation": dict(BASE_IMPLEMENTATION),
            "source_loo_numerical_implementation": dict(LOO_IMPLEMENTATION),
            "streaming_entrypoint_implementation": dict(ENTRYPOINT_IMPLEMENTATION),
            "teacher_payload": {
                "schema": MEAN_SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "additional_tensor": VIEW_AGREEMENT_SCALAR,
                "formula": (
                    "l2_norm(sum(unit(global_top4_teacher_descriptors)))"
                    "/retained_view_count"
                ),
                "query_independent": True,
                "top4_descriptors_durable": False,
            },
            "source_only_leave_one_view_out_audit": {
                "field": v2.LOO_AUDIT_FIELD,
                "hash_field": v2.LOO_AUDIT_SHA256_FIELD,
                "formula_reused_without_change": True,
                "evaluation_axis": "global_all_available_top4_source_views",
                "query_independent": True,
                "target_candidate_authorized": False,
            },
            "target_metric_execution_authorized": False,
        }
    )
    return contract


METHOD_CONTRACT_SHA256 = canonical_json_sha256(method_contract())


def _prepare_inputs(*args: Any, **kwargs: Any) -> dict[str, Any]:
    global _base_descriptor_state
    prepared = _BASE_PREPARE_INPUTS(*args, **kwargs)
    base = prepared.get("base")
    features = base.get("features_by_scale") if isinstance(base, Mapping) else None
    if torch.is_tensor(features):
        _base_descriptor_state = features
    return prepared


def _chunked_teacher_mean_with_source_audit(
    top_descriptors: torch.Tensor,
    top_frame_ids: torch.Tensor,
    *,
    chunk_rows: int = all_view.TEACHER_MEAN_CHUNK_ROWS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global _statistics_state
    if _base_descriptor_state is None:
        raise RuntimeError("all-available source LOO base descriptor was not captured")
    agreement, agreement_counts = v2.directional_resultant_from_canonical_top_views(
        top_descriptors, top_frame_ids
    )
    loo_audit = v2.source_only_leave_one_view_out_ceiling_audit(
        top_descriptors, top_frame_ids, _base_descriptor_state
    )
    mean, valid, retained = _BASE_CHUNKED_TEACHER_MEAN(
        top_descriptors, top_frame_ids, chunk_rows=chunk_rows
    )
    if not torch.equal(agreement_counts, retained):
        raise RuntimeError("all-available teacher agreement count differs")
    _statistics_state = {
        VIEW_AGREEMENT_SCALAR: agreement,
        v2.LOO_AUDIT_FIELD: loo_audit,
    }
    return mean, valid, retained


def _write_torch_with_source_audit(
    path: str | Path, payload: Mapping[str, Any]
) -> None:
    if payload.get("schema") != MEAN_SCHEMA:
        _CORE_WRITE_TORCH(path, payload)
        return
    if _statistics_state is None:
        raise RuntimeError("all-available source statistics were not captured")
    agreement = _statistics_state[VIEW_AGREEMENT_SCALAR]
    loo_audit = _statistics_state[v2.LOO_AUDIT_FIELD]
    output = dict(payload)
    output[VIEW_AGREEMENT_SCALAR] = agreement
    output[VIEW_AGREEMENT_SHA256_FIELD] = legacy.tensor_sha256_typed(agreement)
    output[v2.LOO_AUDIT_FIELD] = copy.deepcopy(loo_audit)
    output[v2.LOO_AUDIT_SHA256_FIELD] = canonical_json_sha256(loo_audit)
    _CORE_WRITE_TORCH(path, output)


def _write_json_with_source_audit(
    path: str | Path, payload: Mapping[str, Any]
) -> None:
    global _statistics_state
    if payload.get("schema") != RESULT_SCHEMA:
        _CORE_WRITE_JSON(path, payload)
        return
    if _statistics_state is None:
        raise RuntimeError("all-available source result statistics are missing")
    loo_audit = _statistics_state[v2.LOO_AUDIT_FIELD]
    output = dict(payload)
    output[v2.LOO_AUDIT_FIELD] = copy.deepcopy(loo_audit)
    output[v2.LOO_AUDIT_SHA256_FIELD] = canonical_json_sha256(loo_audit)
    output[VIEW_AGREEMENT_SHA256_FIELD] = legacy.tensor_sha256_typed(
        _statistics_state[VIEW_AGREEMENT_SCALAR]
    )
    _CORE_WRITE_JSON(path, output)
    _statistics_state = None


def _raw_cache(*args: Any, **kwargs: Any) -> dict[str, Any]:
    payload = _BASE_RAW_CACHE(*args, **kwargs)
    payload["authority"]["descriptor_axis"]["execution_representation"] = (
        "source_teacher_mean_all_available_with_directional_resultant_v2"
    )
    return payload


@torch.inference_mode()
def materialize(args: Any) -> dict[str, Any]:
    global _base_descriptor_state, _statistics_state
    _base_descriptor_state = None
    _statistics_state = None
    result = _BASE_MATERIALIZE(args)
    if _statistics_state is not None:
        raise RuntimeError("all-available source statistics were not persisted")
    _base_descriptor_state = None
    return result


def _install() -> None:
    if (
        ENTRYPOINT_IMPLEMENTATION != file_record(Path(__file__).resolve())
        or BASE_IMPLEMENTATION != file_record(Path(all_view.__file__).resolve())
        or LOO_IMPLEMENTATION != file_record(Path(v2.__file__).resolve())
        or METHOD_CONTRACT_SHA256 != canonical_json_sha256(method_contract())
    ):
        raise RuntimeError("all-available teacher-agreement binding differs")
    all_view.AUTHORITY_SCHEMA = AUTHORITY_SCHEMA
    all_view.MEAN_SCHEMA = MEAN_SCHEMA
    all_view.RESULT_SCHEMA = RESULT_SCHEMA
    all_view.SCHEMA_VERSION = SCHEMA_VERSION
    all_view.method_contract = method_contract
    all_view.METHOD_CONTRACT_SHA256 = METHOD_CONTRACT_SHA256
    all_view.prepare_inputs = _prepare_inputs
    all_view._chunked_teacher_mean = _chunked_teacher_mean_with_source_audit
    all_view._raw_cache = _raw_cache
    all_view.materialize = materialize
    all_view.__file__ = str(Path(__file__).resolve())
    legacy.write_torch_noclobber = _write_torch_with_source_audit
    legacy.write_frozen_json = _write_json_with_source_audit


def main() -> None:
    _install()
    all_view.main()


if __name__ == "__main__":
    main()


__all__ = [
    "AUTHORITY_SCHEMA",
    "MEAN_SCHEMA",
    "METHOD_CONTRACT_SHA256",
    "RESULT_SCHEMA",
    "materialize",
    "method_contract",
]
