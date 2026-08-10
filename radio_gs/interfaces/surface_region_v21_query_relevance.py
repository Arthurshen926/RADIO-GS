"""Formal query-time authority for source-calibrated V2.1 relevance."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces import surface_region_v21_target as target_formal
from radio_gs.interfaces.surface_region_v21_source_gate import (
    validate_source_pilot_chain,
)
from radio_gs.losses import source_global_response_listwise_loss_v21 as v21_loss
from radio_gs.querying import unified_query
from radio_gs.querying import v21_absolute_relevance_adapter as adapter
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
)


QUERY_EXECUTION_SCHEMA = "radio_gs.surface_region_v21_query_execution_authority.v1"
QUERY_RELEVANCE_SCHEMA = "radio_gs.surface_region_v21_absolute_relevance_authority.v1"
IMPLEMENTATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/materialize_surface_region_v21_query_relevance.py"
)
IMPLEMENTATION_DEPENDENCIES = {
    "query_authority": Path(__file__).resolve(),
    "target_descriptor_authority": Path(target_formal.__file__).resolve(),
    "absolute_relevance_adapter": Path(adapter.__file__).resolve(),
    "shared_relevance_function": Path(unified_query.__file__).resolve(),
    "v21_source_loss": Path(v21_loss.__file__).resolve(),
}
PREREGISTRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "paper/artifacts/lerf_v21_absolute_relevance_greedy_novelty_union_preregistration_20260807.json"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def query_relevance_access_audit() -> dict[str, bool]:
    return {
        "source_promotion_validated_before_target_or_query_files": True,
        "target_descriptor_opened": True,
        "benchmark_queries_opened": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "target_metrics_computed": False,
    }


def query_relevance_contract() -> dict[str, Any]:
    return {
        "schema": QUERY_RELEVANCE_SCHEMA,
        "schema_version": 1,
        "descriptor": "source_promoted_v21_target_unit_l2_siglip2",
        "positive_text": "official_siglip2_g_exact_query_template",
        "canonical_negative": "source_training_exact_frozen_four_row_bank",
        "formula": "binary_softmax_positive_vs_max_canonical_negative",
        "logit_scale": v21_loss.INFERENCE_LOGIT_SCALE,
        "assume_normalized": True,
        "postprocess": "none",
        "absolute_relevance_boundary": 0.5,
        "query_smoothing": False,
        "scene_minmax_remap": False,
        "query_ranking_normalization": False,
        "metric_access": False,
    }


QUERY_RELEVANCE_CONTRACT_SHA256 = canonical_json_sha256(query_relevance_contract())


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    path = str(value["path"])
    digest = str(value["sha256"])
    if not Path(path).is_absolute() or _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} file record differs")
    return {"path": path, "sha256": digest}


def _canonical_output(value: object) -> str:
    raw = str(value)
    resolved = str(Path(raw).expanduser().resolve())
    if raw != resolved:
        raise ValueError("V2.1 query relevance output must be absolute and canonical")
    return resolved


def validate_query_execution_authority(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_output: str | Path | None = None,
) -> dict[str, Any]:
    """Validate source promotion before opening target or query artifacts."""

    raw, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="V2.1 query relevance execution authority",
    )
    required = {
        "schema",
        "schema_version",
        "status",
        "scene_id",
        "physical_space_id",
        "source_pilot_result",
        "implementation",
        "implementation_dependencies",
        "preregistration",
        "target_descriptor",
        "positive_text_cache",
        "canonical_negative_bank",
        "query_relevance_output",
        "query_execution_authorized",
        "metric_execution_authorized",
        "access_audit",
    }
    authority = dict(raw)
    if (
        set(authority) != required
        or authority.get("schema") != QUERY_EXECUTION_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("status")
        != "authorized_after_v21_source_promotion_for_calibrated_query_relevance"
        or not isinstance(authority.get("scene_id"), str)
        or not authority["scene_id"]
        or not isinstance(authority.get("physical_space_id"), str)
        or not authority["physical_space_id"]
        or authority.get("query_execution_authorized") is not True
        or authority.get("metric_execution_authorized") is not False
        or authority.get("access_audit") != query_relevance_access_audit()
    ):
        raise ValueError("V2.1 query relevance execution header differs")
    source_result = _record(
        authority["source_pilot_result"], label="V2.1 source pilot result"
    )

    # Security/scientific ordering invariant: no target descriptor, code
    # implementation, or benchmark query cache is opened before promotion.
    source_gate = validate_source_pilot_chain(
        source_result["path"],
        expected_sha256=source_result["sha256"],
        require_promotion=True,
    )
    if source_gate.get("source_promotion_authorized") is not True:
        raise ValueError("V2.1 source promotion is not authorized")
    source_execution_raw, _, _ = load_json_object(
        source_gate["execution_authority"]["path"],
        expected_sha256=source_gate["execution_authority"]["sha256"],
        label="promoted V2.1 source execution authority",
    )
    promoted_negative = _record(
        source_execution_raw.get("canonical_negative_bank"),
        label="promoted V2.1 canonical-negative bank",
    )
    if (
        _record(
            authority["canonical_negative_bank"],
            label="V2.1 query canonical-negative bank",
        )
        != promoted_negative
    ):
        raise ValueError(
            "V2.1 query canonical-negative bank differs from source training"
        )

    implementation = validate_file_record(
        authority["implementation"], label="V2.1 query relevance implementation"
    )
    if implementation != IMPLEMENTATION_PATH:
        raise ValueError("V2.1 query relevance implementation differs")
    dependencies = authority.get("implementation_dependencies")
    if not isinstance(dependencies, Mapping) or set(dependencies) != set(
        IMPLEMENTATION_DEPENDENCIES
    ):
        raise ValueError("V2.1 query relevance implementation dependencies differ")
    verified_dependencies: dict[str, dict[str, str]] = {}
    for name, expected in IMPLEMENTATION_DEPENDENCIES.items():
        verified = validate_file_record(
            dependencies[name], label=f"V2.1 query relevance dependency {name}"
        )
        if verified != expected:
            raise ValueError(f"V2.1 query relevance dependency differs: {name}")
        verified_dependencies[name] = _record(
            dependencies[name], label=f"V2.1 query relevance dependency {name}"
        )
    preregistration = validate_file_record(
        authority["preregistration"], label="V2.1 query relevance preregistration"
    )
    if preregistration != PREREGISTRATION_PATH:
        raise ValueError("V2.1 query relevance preregistration differs")

    descriptor_record = _record(
        authority["target_descriptor"], label="V2.1 target descriptor"
    )
    descriptor_path = validate_file_record(
        descriptor_record, label="V2.1 target descriptor"
    )
    descriptor_raw, descriptor_sha, descriptor_source = load_torch_mapping(
        descriptor_path,
        expected_sha256=descriptor_record["sha256"],
        map_location="cpu",
        label="V2.1 target descriptor",
    )
    descriptor = target_formal.validate_target_descriptor_authority(descriptor_raw)
    target_execution_record = descriptor["target_execution_authority"]
    target_execution = target_formal.validate_target_execution_authority(
        target_execution_record["path"],
        expected_sha256=target_execution_record["sha256"],
        expected_scene_id=authority["scene_id"],
        expected_output=descriptor_source,
    )
    if (
        target_execution.get("source_pilot_result") != source_result
        or descriptor["scene_id"] != authority["scene_id"]
        or descriptor["physical_space_id"] != authority["physical_space_id"]
        or descriptor["input_authority"] != target_execution["target_inputs"]
    ):
        raise ValueError("V2.1 query relevance descriptor/source binding differs")

    # Query records are only path/hash validated here, after the complete
    # promoted target descriptor chain.  Their tensors are opened by the
    # materializer's strict model/canonicalization loaders.
    query_records: dict[str, dict[str, str]] = {}
    for name in ("positive_text_cache", "canonical_negative_bank"):
        shaped = _record(authority[name], label=f"V2.1 query {name}")
        verified = validate_file_record(shaped, label=f"V2.1 query {name}")
        query_records[name] = {"path": str(verified), "sha256": shaped["sha256"]}
    output = _canonical_output(authority["query_relevance_output"])
    if expected_output is not None and output != str(
        Path(expected_output).expanduser().resolve()
    ):
        raise ValueError("V2.1 query relevance output differs")
    authority["source_pilot_result"] = source_result
    authority["implementation"] = _record(
        authority["implementation"], label="V2.1 query relevance implementation"
    )
    authority["implementation_dependencies"] = verified_dependencies
    authority["preregistration"] = _record(
        authority["preregistration"], label="V2.1 query relevance preregistration"
    )
    authority["target_descriptor"] = {
        "path": str(descriptor_source),
        "sha256": descriptor_sha,
    }
    authority.update(query_records)
    authority["query_relevance_output"] = output
    authority["verified_record"] = {"path": str(source), "sha256": digest}
    authority["verified_source_gate"] = source_gate
    authority["verified_promoted_negative"] = promoted_negative
    authority["verified_descriptor"] = descriptor
    return authority


def query_relevance_channel_sha256(value: Mapping[str, Any]) -> dict[str, str]:
    return {
        "region_row_ids": canonical_json_sha256(value["region_row_ids"]),
        "canonical_region_indices": tensor_sha256(value["canonical_region_indices"]),
        "region_fingerprints": canonical_json_sha256(value["region_fingerprints"]),
        "query_ids": canonical_json_sha256(value["query_ids"]),
        "region_absolute_relevance": tensor_sha256(value["region_absolute_relevance"]),
    }


def validate_query_relevance_authority(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("V2.1 query relevance authority must be a mapping")
    payload = dict(value)
    required = {
        "schema",
        "schema_version",
        "contract",
        "contract_sha256",
        "scene_id",
        "physical_space_id",
        "producer",
        "query_execution_authority",
        "input_authority",
        "region_row_ids",
        "canonical_region_indices",
        "region_fingerprints",
        "query_ids",
        "region_absolute_relevance",
        "channel_sha256",
        "access_audit",
    }
    if (
        set(payload) != required
        or payload.get("schema") != QUERY_RELEVANCE_SCHEMA
        or payload.get("schema_version") != 1
        or payload.get("contract") != query_relevance_contract()
        or payload.get("contract_sha256") != QUERY_RELEVANCE_CONTRACT_SHA256
        or payload.get("access_audit") != query_relevance_access_audit()
        or not isinstance(payload.get("scene_id"), str)
        or not payload["scene_id"]
        or not isinstance(payload.get("physical_space_id"), str)
        or not payload["physical_space_id"]
    ):
        raise ValueError("V2.1 query relevance output contract differs")
    payload["producer"] = _record(payload["producer"], label="V2.1 relevance producer")
    payload["query_execution_authority"] = _record(
        payload["query_execution_authority"],
        label="V2.1 relevance execution authority",
    )
    inputs = payload["input_authority"]
    expected_inputs = {
        "target_descriptor",
        "positive_text_cache",
        "canonical_negative_bank",
    }
    if not isinstance(inputs, Mapping) or set(inputs) != expected_inputs:
        raise ValueError("V2.1 query relevance input authority differs")
    payload["input_authority"] = {
        name: _record(inputs[name], label=f"V2.1 relevance {name}")
        for name in sorted(expected_inputs)
    }
    rows = payload["region_row_ids"]
    canonical = payload["canonical_region_indices"]
    fingerprints = payload["region_fingerprints"]
    queries = payload["query_ids"]
    relevance = payload["region_absolute_relevance"]
    regions = len(rows) if isinstance(rows, list) else -1
    query_count = len(queries) if isinstance(queries, list) else -1
    if (
        regions <= 0
        or query_count <= 0
        or len(set(rows)) != regions
        or any(not isinstance(item, str) or not item for item in rows)
        or len(set(queries)) != query_count
        or any(not isinstance(item, str) or not item for item in queries)
        or not isinstance(fingerprints, list)
        or len(fingerprints) != regions
        or len(set(fingerprints)) != regions
        or any(_SHA256.fullmatch(str(item)) is None for item in fingerprints)
        or not torch.is_tensor(canonical)
        or canonical.dtype != torch.long
        or canonical.device.type != "cpu"
        or canonical.shape != (regions,)
        or (regions > 1 and not bool((canonical[1:] > canonical[:-1]).all()))
        or not torch.is_tensor(relevance)
        or relevance.dtype != torch.float32
        or relevance.device.type != "cpu"
        or relevance.shape != (regions, query_count)
        or not bool(torch.isfinite(relevance).all())
        or bool((relevance < 0).any())
        or bool((relevance > 1).any())
    ):
        raise ValueError("V2.1 query relevance tensor layout differs")
    if payload["channel_sha256"] != query_relevance_channel_sha256(payload):
        raise ValueError("V2.1 query relevance channel SHA-256 differs")
    return payload


__all__ = [
    "IMPLEMENTATION_DEPENDENCIES",
    "IMPLEMENTATION_PATH",
    "PREREGISTRATION_PATH",
    "QUERY_EXECUTION_SCHEMA",
    "QUERY_RELEVANCE_CONTRACT_SHA256",
    "QUERY_RELEVANCE_SCHEMA",
    "query_relevance_access_audit",
    "query_relevance_channel_sha256",
    "query_relevance_contract",
    "validate_query_execution_authority",
    "validate_query_relevance_authority",
]
