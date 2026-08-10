#!/usr/bin/env python3
"""Opt-in exact LERF query/readout/metric chain for factorized-native descriptors."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch

from radio_gs.interfaces import factorized_native_lerf_exact as formal
from radio_gs.interfaces import factorized_native_target_descriptor as target_formal
from radio_gs.interfaces import factorized_native_target_health as health_formal
from radio_gs.interfaces.factorized_primitive_state import load_factorized_primitive_state
from radio_gs.interfaces.surface_region_target_accepted_v2 import (
    validate_target_accepted_v2_authority,
)
from radio_gs.losses.source_global_response_listwise_loss_v21 import (
    CANONICAL_NEGATIVE_MODEL,
    load_frozen_canonical_negative_bank,
)
from radio_gs.models.factorized_native_gauge_state_readout import (
    FACTORIZED_NATIVE_READOUT_ARMS,
)
from radio_gs.querying.v21_absolute_relevance_adapter import (
    OFFICIAL_TEXT_CANONICALIZATION,
    calibrated_v21_absolute_relevance,
    load_v21_positive_text_bank,
)
from radio_gs.scripts import build_lerf_region_comembership_external_cache_v2 as v2_external
from radio_gs.scripts import build_lerf_region_comembership_external_cache_v21 as v21_external
from radio_gs.scripts import eval_lerf_direct_3d_selection as frozen_evaluator
from radio_gs.scripts.infer_region_comembership_v2 import validate_inference_authority
from radio_gs.scripts.materialize_region_comembership_features_v2 import (
    validate_feature_authority,
)
from radio_gs.scripts.materialize_lerf_multiscale_query_score_cache import (
    _renderer_checkpoint_xyz,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    load_sha_bound_project_checkpoint_mapping,
    load_torch_mapping,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


IMPLEMENTATION = Path(__file__).resolve()
QUERY_STATUS = "authorized_factorized_native_exact_query_only"
EXTERNAL_STATUS = "authorized_factorized_native_frozen_greedy_novelty_readout"
METRIC_STATUS = "authorized_factorized_native_single_frozen_lerf_metric"


def _new(value: object, *, label: str) -> Path:
    raw = str(value)
    path = Path(raw).expanduser().resolve()
    if raw != str(path):
        raise ValueError(f"{label} must be a canonical absolute path")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{label} already exists: {path}")
    return path


def _existing(value: object, *, label: str) -> Path:
    raw = str(value)
    path = Path(raw).expanduser().resolve()
    if raw != str(path) or not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be an existing canonical regular file")
    return path


def _source_records_from_args(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    return {
        arm: formal.record(
            {
                "path": str(Path(getattr(args, f"{arm}_result")).expanduser().resolve()),
                "sha256": str(getattr(args, f"{arm}_result_sha256")),
            },
            label=f"{arm} source result",
        )
        for arm in FACTORIZED_NATIVE_READOUT_ARMS
    }


def _validated_source_records(value: object) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping) or set(value) != set(
        FACTORIZED_NATIVE_READOUT_ARMS
    ):
        raise ValueError("factorized-native source arm result records differ")
    records = {
        arm: formal.record(value[arm], label=f"{arm} source result")
        for arm in FACTORIZED_NATIVE_READOUT_ARMS
    }
    return records


def _query_dependencies() -> dict[str, dict[str, str]]:
    return {
        "query_formal": file_record(Path(formal.__file__).resolve()),
        **{
            name: file_record(path)
            for name, path in formal.FROZEN_QUERY_DEPENDENCIES.items()
        },
    }


def _external_dependencies() -> dict[str, dict[str, str]]:
    return {
        "query_formal": file_record(Path(formal.__file__).resolve()),
        **{
            name: file_record(path)
            for name, path in formal.FROZEN_EXTERNAL_DEPENDENCIES.items()
        },
    }


def _validate_dependencies(
    value: object, *, expected: Mapping[str, Path], prefix: str
) -> dict[str, dict[str, str]]:
    combined = {"query_formal": Path(formal.__file__).resolve(), **dict(expected)}
    if not isinstance(value, Mapping) or set(value) != set(combined):
        raise ValueError(f"{prefix} implementation dependencies differ")
    records: dict[str, dict[str, str]] = {}
    for name, path in combined.items():
        verified = validate_file_record(value[name], label=f"{prefix} dependency {name}")
        if verified != path:
            raise ValueError(f"{prefix} dependency differs: {name}")
        records[name] = formal.record(value[name], label=f"{prefix} dependency {name}")
    return records


def _load_frozen_all_query_cache(record: Mapping[str, str]) -> tuple[list[str], torch.Tensor]:
    if dict(record) != formal.FROZEN_ALL_QUERY_CACHE:
        raise ValueError("all-query cache singleton differs")
    payload, _, _ = load_torch_mapping(
        record["path"], expected_sha256=record["sha256"], map_location="cpu",
        label="frozen all-query text cache",
    )
    required = {"queries", "prompt_templates", "text_encoder", "model_name", "embeddings"}
    queries, embeddings = payload.get("queries"), payload.get("embeddings")
    if (
        set(payload) != required
        or not isinstance(queries, list)
        or not queries
        or len(set(queries)) != len(queries)
        or any(not isinstance(item, str) or not item.strip() for item in queries)
        or payload.get("prompt_templates") != ["{query}"]
        or payload.get("text_encoder") != "siglip2"
        or payload.get("model_name") != CANONICAL_NEGATIVE_MODEL
        or not torch.is_tensor(embeddings)
        or embeddings.dtype != torch.float32
        or embeddings.device.type != "cpu"
        or embeddings.shape != (len(queries), 1536)
        or not bool(torch.isfinite(embeddings).all())
    ):
        raise ValueError("frozen all-query cache contract differs")
    norms = torch.linalg.vector_norm(embeddings, dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), rtol=0.0, atol=2e-4):
        raise ValueError("frozen all-query embeddings are not unit L2")
    return list(queries), embeddings.detach().contiguous()


def _validate_exact_text_protocol(
    *,
    scene_id: str,
    manifest_record: Mapping[str, str],
    positive_record: Mapping[str, str],
) -> tuple[dict[str, Any], Any]:
    manifest_raw, _, _ = load_json_object(
        manifest_record["path"], expected_sha256=manifest_record["sha256"],
        label="factorized-native exact scene query manifest",
    )
    manifest = formal.validate_exact_query_manifest(
        manifest_raw, scene_id=scene_id
    )
    if (
        manifest["frozen_all_query_cache"] != formal.FROZEN_ALL_QUERY_CACHE
        or manifest["frozen_evaluator"] != formal.FROZEN_EVALUATOR
    ):
        raise ValueError("exact query manifest differs from frozen protocol")
    validate_file_record(formal.FROZEN_EVALUATOR, label="frozen LERF evaluator")
    positive = load_v21_positive_text_bank(
        positive_record["path"], expected_file_sha256=positive_record["sha256"]
    )
    all_queries, all_embeddings = _load_frozen_all_query_cache(
        formal.FROZEN_ALL_QUERY_CACHE
    )
    positions = {query: index for index, query in enumerate(all_queries)}
    exact_ids = list(manifest["query_ids"])
    if (
        list(positive.query_ids) != exact_ids
        or any(query not in positions for query in exact_ids)
    ):
        raise ValueError("positive cache differs from exact scene query order")
    selected = all_embeddings[
        torch.tensor([positions[query] for query in exact_ids], dtype=torch.long)
    ]
    if not torch.equal(positive.embeddings, selected):
        raise ValueError("positive cache differs from frozen all-query subset")
    return manifest, positive


def _load_target_descriptor(record: Mapping[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, _, _ = load_torch_mapping(
        record["path"], expected_sha256=record["sha256"], map_location="cpu",
        label="factorized-native target descriptor",
    )
    descriptor = target_formal.validate_target_descriptor_authority(raw)
    execution = target_formal.validate_target_execution_authority(
        descriptor["target_execution_authority"]["path"],
        expected_sha256=descriptor["target_execution_authority"]["sha256"],
        expected_output=record["path"],
    )
    source = execution["verified_source_gate"]
    if (
        descriptor["producer"] != execution["implementation"]
        or descriptor["target_execution_authority"] != execution["verified_record"]
        or descriptor["input_authority"]["source_arm_results"]
        != execution["source_arm_results"]
        or descriptor["input_authority"]["target_accepted_v2"]
        != execution["target_inputs"]["target_accepted_v2"]
        or descriptor["input_authority"]["factorized_primitive_state"]
        != execution["target_inputs"]["factorized_primitive_state"]
        or descriptor["winner_arm"] != source["winner_arm"]
        or descriptor["winner_selected_step"]
        != source["winner_result"]["selected_step"]
    ):
        raise ValueError("factorized-native descriptor/target execution binding differs")
    return descriptor, execution


def _load_descriptor_health_gate(
    record: Mapping[str, str], *, descriptor_record: Mapping[str, str],
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    raw, _, _ = load_json_object(
        record["path"], expected_sha256=record["sha256"],
        label="factorized-native query-free descriptor health audit",
    )
    audit = health_formal.validate_health_audit(raw, require_pass=True)
    producer = validate_file_record(
        audit["producer"], label="descriptor health audit producer"
    )
    if (
        producer != health_formal.HEALTH_AUDIT_IMPLEMENTATION_PATH
        or audit["input_authority"]["target_descriptor"]
        != dict(descriptor_record)
        or audit["input_authority"]["accepted_v2_baseline"]
        != descriptor["input_authority"]["target_accepted_v2"]
        or audit["descriptor_channel_sha256"] != descriptor["channel_sha256"]
        or audit["scene_id"] != descriptor["scene_id"]
        or audit["physical_space_id"] != descriptor["physical_space_id"]
    ):
        raise ValueError("factorized-native descriptor health gate binding differs")
    return audit


def build_query_authority(args: argparse.Namespace) -> dict[str, Any]:
    source_records = _source_records_from_args(args)
    source = target_formal.validate_source_arm_winner(source_records)
    authority_output = _new(args.output_authority, label="query authority output")
    relevance_output = _new(args.query_relevance_output, label="query relevance output")
    descriptor_record = file_record(_existing(args.target_descriptor, label="target descriptor"))
    descriptor, target_execution = _load_target_descriptor(descriptor_record)
    if source_records != target_execution["source_arm_results"]:
        raise ValueError("query source arms differ from target descriptor")
    health_record = file_record(
        _existing(args.descriptor_health_audit, label="descriptor health audit")
    )
    _load_descriptor_health_gate(
        health_record, descriptor_record=descriptor_record, descriptor=descriptor
    )
    manifest_record = file_record(
        _existing(args.exact_query_manifest, label="exact query manifest")
    )
    positive_record = file_record(
        _existing(args.positive_text_cache, label="positive text cache")
    )
    _validate_exact_text_protocol(
        scene_id=descriptor["scene_id"], manifest_record=manifest_record,
        positive_record=positive_record,
    )
    for singleton, label in (
        (formal.FROZEN_CANONICAL_NEGATIVE_BANK, "canonical negative bank"),
        (formal.FROZEN_PREREGISTRATION, "frozen preregistration"),
    ):
        validate_file_record(singleton, label=label)
    authority = {
        "schema": formal.QUERY_EXECUTION_SCHEMA,
        "schema_version": 1,
        "status": QUERY_STATUS,
        "source_arm_results": source_records,
        "winner_arm": source["winner_arm"],
        "scene_id": descriptor["scene_id"],
        "physical_space_id": descriptor["physical_space_id"],
        "implementation": file_record(IMPLEMENTATION),
        "implementation_dependencies": _query_dependencies(),
        "preregistration": dict(formal.FROZEN_PREREGISTRATION),
        "target_descriptor": descriptor_record,
        "descriptor_health_audit": health_record,
        "exact_query_manifest": manifest_record,
        "positive_text_cache": positive_record,
        "all_query_text_cache": dict(formal.FROZEN_ALL_QUERY_CACHE),
        "canonical_negative_bank": dict(formal.FROZEN_CANONICAL_NEGATIVE_BANK),
        "query_relevance_output": str(relevance_output),
        "query_execution_authorized": True,
        "metric_execution_authorized": False,
        "access_audit": formal.query_access_audit(),
    }
    write_frozen_json(authority_output, authority)
    return {"status": "factorized_native_query_authority_built", "authority": file_record(authority_output)}


def validate_query_authority(
    path: str | Path, *, expected_sha256: str,
    expected_output: str | Path | None = None,
) -> dict[str, Any]:
    raw, digest, source_path = load_json_object(
        path, expected_sha256=expected_sha256,
        label="factorized-native query execution authority",
    )
    authority = dict(raw)
    required = {
        "schema", "schema_version", "status", "source_arm_results", "winner_arm",
        "scene_id", "physical_space_id", "implementation", "implementation_dependencies",
        "preregistration", "target_descriptor", "descriptor_health_audit",
        "exact_query_manifest",
        "positive_text_cache", "all_query_text_cache", "canonical_negative_bank",
        "query_relevance_output", "query_execution_authorized",
        "metric_execution_authorized", "access_audit",
    }
    if set(authority) != required:
        raise ValueError("factorized-native query authority fields differ")
    # Source gate must precede every target/query/code file access.
    source_records = _validated_source_records(authority["source_arm_results"])
    source_gate = target_formal.validate_source_arm_winner(source_records)
    if (
        authority["schema"] != formal.QUERY_EXECUTION_SCHEMA
        or authority["schema_version"] != 1
        or authority["status"] != QUERY_STATUS
        or authority["winner_arm"] != source_gate["winner_arm"]
        or authority["query_execution_authorized"] is not True
        or authority["metric_execution_authorized"] is not False
        or authority["access_audit"] != formal.query_access_audit()
    ):
        raise ValueError("factorized-native query authority header differs")
    if validate_file_record(authority["implementation"], label="query implementation") != IMPLEMENTATION:
        raise ValueError("factorized-native query implementation differs")
    dependencies = _validate_dependencies(
        authority["implementation_dependencies"],
        expected=formal.FROZEN_QUERY_DEPENDENCIES, prefix="query",
    )
    preregistration = formal.record(authority["preregistration"], label="preregistration")
    if preregistration != formal.FROZEN_PREREGISTRATION:
        raise ValueError("factorized-native query preregistration differs")
    validate_file_record(preregistration, label="query preregistration")
    records = {
        name: formal.record(authority[name], label=f"query {name}")
        for name in (
            "target_descriptor", "descriptor_health_audit", "exact_query_manifest",
            "positive_text_cache",
            "all_query_text_cache", "canonical_negative_bank",
        )
    }
    if (
        records["all_query_text_cache"] != formal.FROZEN_ALL_QUERY_CACHE
        or records["canonical_negative_bank"]
        != formal.FROZEN_CANONICAL_NEGATIVE_BANK
    ):
        raise ValueError("factorized-native frozen query singleton differs")
    descriptor, target_execution = _load_target_descriptor(records["target_descriptor"])
    if source_records != target_execution["source_arm_results"]:
        raise ValueError("query source arms differ from descriptor source arms")
    health_audit = _load_descriptor_health_gate(
        records["descriptor_health_audit"],
        descriptor_record=records["target_descriptor"], descriptor=descriptor,
    )
    manifest, positive = _validate_exact_text_protocol(
        scene_id=descriptor["scene_id"],
        manifest_record=records["exact_query_manifest"],
        positive_record=records["positive_text_cache"],
    )
    validate_file_record(records["canonical_negative_bank"], label="canonical negative bank")
    if (
        authority["scene_id"] != descriptor["scene_id"]
        or authority["physical_space_id"] != descriptor["physical_space_id"]
    ):
        raise ValueError("factorized-native query descriptor identity differs")
    output = formal.canonical_output(
        authority["query_relevance_output"], label="query relevance output"
    )
    if expected_output is not None and output != str(Path(expected_output).expanduser().resolve()):
        raise ValueError("factorized-native query relevance output differs")
    authority.update(records)
    authority.update({
        "source_arm_results": source_records,
        "implementation_dependencies": dependencies,
        "preregistration": preregistration,
        "query_relevance_output": output,
        "verified_source_gate": source_gate,
        "verified_descriptor": descriptor,
        "verified_descriptor_health_audit": health_audit,
        "verified_manifest": manifest,
        "verified_positive": positive,
        "verified_record": {"path": str(source_path), "sha256": digest},
    })
    return authority


def materialize_query(args: argparse.Namespace) -> dict[str, Any]:
    output = _new(args.output, label="query relevance output")
    execution = validate_query_authority(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
        expected_output=output,
    )
    negative = load_frozen_canonical_negative_bank(
        execution["canonical_negative_bank"]["path"],
        expected_file_sha256=execution["canonical_negative_bank"]["sha256"],
    )
    descriptor = target_formal.exact_query_descriptor_view(
        execution["verified_descriptor"]
    )
    relevance = calibrated_v21_absolute_relevance(
        descriptor["semantic_descriptor"],
        positive_bank=execution["verified_positive"],
        canonical_negative_bank=negative,
    )
    payload = {
        "schema": formal.QUERY_RELEVANCE_SCHEMA,
        "schema_version": 1,
        "contract": formal.query_contract(),
        "contract_sha256": formal.QUERY_CONTRACT_SHA256,
        "scene_id": descriptor["scene_id"],
        "physical_space_id": descriptor["physical_space_id"],
        "producer": file_record(IMPLEMENTATION),
        "query_execution_authority": dict(execution["verified_record"]),
        "input_authority": {
            name: dict(execution[name])
            for name in (
                "target_descriptor", "descriptor_health_audit",
                "exact_query_manifest", "positive_text_cache",
                "all_query_text_cache", "canonical_negative_bank",
            )
        },
        "region_row_ids": list(descriptor["region_row_ids"]),
        "canonical_region_indices": descriptor["canonical_region_indices"].clone(),
        "region_fingerprints": list(descriptor["region_fingerprints"]),
        "query_ids": list(execution["verified_positive"].query_ids),
        "region_absolute_relevance": relevance,
        "access_audit": formal.query_access_audit(),
    }
    payload["channel_sha256"] = formal.query_channel_sha256(payload)
    payload = formal.validate_query_relevance(payload)
    write_torch_noclobber(output, payload)
    return {"status": "factorized_native_exact_query_relevance_complete", "output": file_record(output)}


def build_external_authority(args: argparse.Namespace) -> dict[str, Any]:
    source_records = _source_records_from_args(args)
    target_formal.validate_source_arm_winner(source_records)
    authority_output = _new(args.output_authority, label="external authority output")
    cache_output = _new(args.output_cache, label="external cache output")
    report_output = _new(args.output_report, label="external report output")
    query_execution_record = file_record(
        _existing(args.query_execution_authority, label="query execution authority")
    )
    relevance_record = file_record(
        _existing(args.query_relevance, label="query relevance")
    )
    query = validate_query_authority(
        query_execution_record["path"], expected_sha256=query_execution_record["sha256"],
        expected_output=relevance_record["path"],
    )
    if source_records != query["source_arm_results"]:
        raise ValueError("external source arms differ from query source arms")
    authority = {
        "schema": formal.EXTERNAL_EXECUTION_SCHEMA,
        "schema_version": 1,
        "status": EXTERNAL_STATUS,
        "source_arm_results": source_records,
        "scene_id": query["scene_id"],
        "physical_space_id": query["physical_space_id"],
        "implementation": file_record(IMPLEMENTATION),
        "implementation_dependencies": _external_dependencies(),
        "preregistration": dict(formal.FROZEN_PREREGISTRATION),
        "query_execution_authority": query_execution_record,
        "query_relevance": relevance_record,
        "comembership_feature_authority": file_record(
            _existing(args.comembership_feature_authority, label="co-membership features")
        ),
        "comembership_inference_authority": file_record(
            _existing(args.comembership_inference_authority, label="co-membership inference")
        ),
        "renderer_geometry_checkpoint": file_record(
            _existing(args.renderer_geometry_checkpoint, label="renderer checkpoint")
        ),
        "output_cache": str(cache_output),
        "output_report": str(report_output),
        "query_readout_authorized": True,
        "target_metric_authorized": False,
        "access_audit": formal.external_access_audit(),
    }
    write_frozen_json(authority_output, authority)
    return {"status": "factorized_native_external_authority_built", "authority": file_record(authority_output)}


def validate_external_authority(path: str | Path, *, expected_sha256: str) -> dict[str, Any]:
    raw, digest, source_path = load_json_object(
        path, expected_sha256=expected_sha256,
        label="factorized-native external execution authority",
    )
    authority = dict(raw)
    required = {
        "schema", "schema_version", "status", "source_arm_results", "scene_id",
        "physical_space_id", "implementation", "implementation_dependencies",
        "preregistration", "query_execution_authority", "query_relevance",
        "comembership_feature_authority", "comembership_inference_authority",
        "renderer_geometry_checkpoint", "output_cache", "output_report",
        "query_readout_authorized", "target_metric_authorized", "access_audit",
    }
    if set(authority) != required:
        raise ValueError("factorized-native external authority fields differ")
    source_records = _validated_source_records(authority["source_arm_results"])
    source_gate = target_formal.validate_source_arm_winner(source_records)
    if (
        authority["schema"] != formal.EXTERNAL_EXECUTION_SCHEMA
        or authority["schema_version"] != 1
        or authority["status"] != EXTERNAL_STATUS
        or authority["query_readout_authorized"] is not True
        or authority["target_metric_authorized"] is not False
        or authority["access_audit"] != formal.external_access_audit()
    ):
        raise ValueError("factorized-native external authority header differs")
    if validate_file_record(authority["implementation"], label="external implementation") != IMPLEMENTATION:
        raise ValueError("factorized-native external implementation differs")
    dependencies = _validate_dependencies(
        authority["implementation_dependencies"],
        expected=formal.FROZEN_EXTERNAL_DEPENDENCIES, prefix="external",
    )
    preregistration = formal.record(authority["preregistration"], label="preregistration")
    if preregistration != formal.FROZEN_PREREGISTRATION:
        raise ValueError("factorized-native external preregistration differs")
    validate_file_record(preregistration, label="external preregistration")
    names = (
        "query_execution_authority", "query_relevance",
        "comembership_feature_authority", "comembership_inference_authority",
        "renderer_geometry_checkpoint",
    )
    records = {name: formal.record(authority[name], label=f"external {name}") for name in names}
    query = validate_query_authority(
        records["query_execution_authority"]["path"],
        expected_sha256=records["query_execution_authority"]["sha256"],
        expected_output=records["query_relevance"]["path"],
    )
    relevance_raw, _, _ = load_torch_mapping(
        records["query_relevance"]["path"],
        expected_sha256=records["query_relevance"]["sha256"], map_location="cpu",
        label="factorized-native query relevance",
    )
    relevance = formal.validate_query_relevance(relevance_raw)
    feature_raw, _, _ = load_torch_mapping(
        records["comembership_feature_authority"]["path"],
        expected_sha256=records["comembership_feature_authority"]["sha256"],
        map_location="cpu", label="co-membership feature authority",
    )
    inference_raw, _, _ = load_torch_mapping(
        records["comembership_inference_authority"]["path"],
        expected_sha256=records["comembership_inference_authority"]["sha256"],
        map_location="cpu", label="co-membership inference authority",
    )
    feature = validate_feature_authority(feature_raw)
    inference = validate_inference_authority(inference_raw)
    selected_rule = v2_external.validate_v2_authority_binding(
        feature=feature, inference=inference,
        feature_record=records["comembership_feature_authority"],
    )
    descriptor = query["verified_descriptor"]
    accepted_record = formal.record(
        feature.get("input_authority", {}).get("accepted_v2"),
        label="co-membership AcceptedV2",
    )
    accepted_raw, _, _ = load_torch_mapping(
        accepted_record["path"], expected_sha256=accepted_record["sha256"],
        map_location="cpu", label="external AcceptedV2",
    )
    accepted = validate_target_accepted_v2_authority(accepted_raw)
    renderer_raw, renderer_sha, _ = load_sha_bound_project_checkpoint_mapping(
        records["renderer_geometry_checkpoint"]["path"],
        expected_sha256=records["renderer_geometry_checkpoint"]["sha256"],
        map_location="cpu", label="renderer checkpoint",
    )
    renderer_xyz = _renderer_checkpoint_xyz(renderer_raw)
    v2_external.validate_renderer_geometry_binding(
        feature=feature, accepted=accepted, accepted_record=accepted_record,
        renderer_geometry_checkpoint_sha256=renderer_sha,
    )
    state_record = descriptor["input_authority"]["factorized_primitive_state"]
    state = load_factorized_primitive_state(
        state_record["path"], expected_sha256=state_record["sha256"]
    )
    expected_relevance_inputs = {
        name: query[name]
        for name in (
            "target_descriptor", "descriptor_health_audit", "exact_query_manifest",
            "positive_text_cache",
            "all_query_text_cache", "canonical_negative_bank",
        )
    }
    if (
        query["source_arm_results"] != source_records
        or relevance["query_execution_authority"]
        != records["query_execution_authority"]
        or relevance["producer"] != authority["implementation"]
        or relevance["input_authority"] != expected_relevance_inputs
        or relevance["query_ids"] != list(query["verified_positive"].query_ids)
        or relevance["scene_id"] != authority["scene_id"]
        or descriptor["scene_id"] != authority["scene_id"]
        or descriptor["physical_space_id"] != authority["physical_space_id"]
        or relevance["physical_space_id"] != authority["physical_space_id"]
        or accepted_record != descriptor["input_authority"]["target_accepted_v2"]
        or relevance["region_row_ids"] != descriptor["region_row_ids"]
        or relevance["region_fingerprints"] != feature["region_fingerprints"]
        or descriptor["region_fingerprints"] != feature["region_fingerprints"]
        or not torch.equal(
            relevance["canonical_region_indices"], descriptor["canonical_region_indices"]
        )
        or not torch.equal(
            descriptor["canonical_region_indices"], feature["canonical_region_indices"]
        )
        or state.xyz.shape != renderer_xyz.shape
        or not torch.equal(state.xyz.float(), renderer_xyz)
    ):
        raise ValueError("factorized-native external descriptor/graph/renderer binding differs")
    output = formal.canonical_output(authority["output_cache"], label="external cache output")
    report = formal.canonical_output(authority["output_report"], label="external report output")
    if output == report:
        raise ValueError("external cache and report outputs must differ")
    authority.update(records)
    authority.update({
        "source_arm_results": source_records,
        "implementation_dependencies": dependencies,
        "preregistration": preregistration,
        "output_cache": output, "output_report": report,
        "verified_source_gate": source_gate, "verified_query": query,
        "verified_descriptor": descriptor, "verified_relevance": relevance,
        "verified_feature": feature, "verified_inference": inference,
        "verified_state": state, "selected_rule": selected_rule,
        "verified_record": {"path": str(source_path), "sha256": digest},
    })
    return authority


def materialize_external(args: argparse.Namespace) -> dict[str, Any]:
    execution = validate_external_authority(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
    )
    output = _new(execution["output_cache"], label="external cache output")
    report_output = _new(execution["output_report"], label="external report output")
    state = execution["verified_state"]
    readout = v21_external.greedy_novelty_readout_from_v21(
        feature=execution["verified_feature"], inference=execution["verified_inference"],
        relevance=execution["verified_relevance"], num_primitives=int(state.xyz.shape[0]),
    )
    membership, removed = v21_external.mask_union_to_valid(
        readout.primitive_membership, state.valid
    )
    query_ids = list(execution["verified_relevance"]["query_ids"])
    selection = canonical_external_selection(
        readout, expected_query_count=len(query_ids)
    )
    selection["invalid_memberships_removed"] = removed
    cache = {
        "schema": formal.EXTERNAL_CACHE_SCHEMA,
        "schema_version": 1,
        "contract": formal.external_contract(),
        "contract_sha256": formal.EXTERNAL_CONTRACT_SHA256,
        "query_scores": membership,
        "valid": state.valid.bool().cpu().contiguous(),
        "xyz": state.xyz.float().cpu().contiguous(),
        "metadata": {
            "query_names": query_ids,
            "score_semantics": "binary_factorized_native_absolute_relevance_greedy_novelty_union",
            "producer": file_record(IMPLEMENTATION),
            "execution_authority": dict(execution["verified_record"]),
        },
        "selection": selection,
    }
    cache = formal.validate_external_cache(cache)
    write_torch_noclobber(output, cache)
    report = {
        "schema": formal.EXTERNAL_CACHE_SCHEMA,
        "status": "factorized_native_external_cache_complete",
        "cache": file_record(output),
        "query_ids": query_ids,
        "selected_region_counts": [len(value) for value in readout.selected_region_indices],
        "selected_primitive_counts": membership.sum(dim=0).int().tolist(),
        "invalid_memberships_removed": removed,
        "execution_authority": dict(execution["verified_record"]),
        "access_audit": formal.external_access_audit(),
    }
    write_frozen_json(report_output, report)
    return report


def canonical_external_selection(
    readout: object, *, expected_query_count: int
) -> dict[str, list[list[int]] | list[list[float]]]:
    """Serialize the frozen tuple readout into the cache's list contract.

    Conversion is positional only: it neither sorts nor filters a query or a
    selected region.  The strict checks make an upstream order/layout change
    fail before a cache can be published.
    """

    query_count = int(expected_query_count)
    indices = getattr(readout, "selected_region_indices", None)
    scores = getattr(readout, "selected_region_scores", None)
    core_rows = getattr(readout, "selected_marginal_core_rows", None)
    if (
        query_count <= 0
        or not isinstance(indices, tuple)
        or not isinstance(scores, tuple)
        or not isinstance(core_rows, tuple)
        or len(indices) != query_count
        or len(scores) != query_count
        or len(core_rows) != query_count
    ):
        raise ValueError("frozen greedy-novelty selection query order differs")
    canonical_indices: list[list[int]] = []
    canonical_scores: list[list[float]] = []
    canonical_core_rows: list[list[int]] = []
    for query_index in range(query_count):
        query_indices = indices[query_index]
        query_scores = scores[query_index]
        query_core_rows = core_rows[query_index]
        if (
            not isinstance(query_indices, tuple)
            or not isinstance(query_scores, tuple)
            or not isinstance(query_core_rows, tuple)
            or len(query_indices) != len(query_scores)
            or len(query_indices) != len(query_core_rows)
            or len(query_indices) > v21_external.V21_MAXIMUM_REGIONS
            or any(type(item) is not int or item < 0 for item in query_indices)
            or len(set(query_indices)) != len(query_indices)
            or any(
                type(item) is not float or not math.isfinite(item)
                or item < 0.0 or item > 1.0
                for item in query_scores
            )
            or any(type(item) is not int or item <= 0 for item in query_core_rows)
        ):
            raise ValueError("frozen greedy-novelty per-query selection differs")
        canonical_indices.append(list(query_indices))
        canonical_scores.append(list(query_scores))
        canonical_core_rows.append(list(query_core_rows))
    return {
        "region_indices": canonical_indices,
        "region_scores": canonical_scores,
        "marginal_core_rows": canonical_core_rows,
    }


def validate_external_report(
    value: object, *, expected_cache: Mapping[str, str],
    expected_execution: Mapping[str, str], expected_query_ids: list[str],
) -> dict[str, Any]:
    required = {
        "schema", "status", "cache", "query_ids", "selected_region_counts",
        "selected_primitive_counts", "invalid_memberships_removed",
        "execution_authority", "access_audit",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("factorized-native external report fields differ")
    report = dict(value)
    count = len(expected_query_ids)
    if (
        report["schema"] != formal.EXTERNAL_CACHE_SCHEMA
        or report["status"] != "factorized_native_external_cache_complete"
        or formal.record(report["cache"], label="external report cache")
        != dict(expected_cache)
        or formal.record(report["execution_authority"], label="external report execution")
        != dict(expected_execution)
        or report["query_ids"] != expected_query_ids
        or not isinstance(report["selected_region_counts"], list)
        or len(report["selected_region_counts"]) != count
        or any(not isinstance(item, int) or item < 0 for item in report["selected_region_counts"])
        or not isinstance(report["selected_primitive_counts"], list)
        or len(report["selected_primitive_counts"]) != count
        or any(not isinstance(item, int) or item < 0 for item in report["selected_primitive_counts"])
        or not isinstance(report["invalid_memberships_removed"], int)
        or report["invalid_memberships_removed"] < 0
        or report["access_audit"] != formal.external_access_audit()
    ):
        raise ValueError("factorized-native external report contract differs")
    return report


def build_metric_authority(args: argparse.Namespace) -> dict[str, Any]:
    source_records = _source_records_from_args(args)
    target_formal.validate_source_arm_winner(source_records)
    authority_output = _new(args.output_authority, label="metric authority output")
    output_dir = Path(formal.canonical_output(args.output_dir, label="metric output directory"))
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("metric output directory must be new")
    external_record = file_record(
        _existing(args.external_execution_authority, label="external execution authority")
    )
    external = validate_external_authority(
        external_record["path"], expected_sha256=external_record["sha256"]
    )
    if external["source_arm_results"] != source_records:
        raise ValueError("metric source arms differ from external chain")
    cache_record = file_record(_existing(args.external_cache, label="external cache"))
    report_record = file_record(_existing(args.external_report, label="external report"))
    if (
        cache_record["path"] != external["output_cache"]
        or report_record["path"] != external["output_report"]
    ):
        raise ValueError("metric external outputs differ from execution authority")
    frozen_inputs = {
        "config": file_record(_existing(args.config, label="frozen config")),
        "renderer_geometry_checkpoint": file_record(
            _existing(args.renderer_geometry_checkpoint, label="renderer checkpoint")
        ),
        "summary_head": file_record(_existing(args.summary_head, label="summary head")),
        "all_query_text_cache": dict(formal.FROZEN_ALL_QUERY_CACHE),
        "canonical_negative_text_cache": dict(formal.FROZEN_CANONICAL_NEGATIVE_BANK),
    }
    label_root = Path(args.label_root).expanduser().resolve()
    if str(label_root) != args.label_root or not label_root.is_dir() or label_root.is_symlink():
        raise ValueError("metric label root must be an existing canonical directory")
    authority = {
        "schema": formal.METRIC_EXECUTION_SCHEMA,
        "schema_version": 1,
        "status": METRIC_STATUS,
        "source_arm_results": source_records,
        "scene_id": external["scene_id"],
        "implementation": file_record(IMPLEMENTATION),
        "frozen_evaluator": dict(formal.FROZEN_EVALUATOR),
        "preregistration": dict(formal.FROZEN_PREREGISTRATION),
        "external_execution_authority": external_record,
        "external_cache": cache_record,
        "external_report": report_record,
        "frozen_inputs": frozen_inputs,
        "label_root": str(label_root),
        "output_dir": str(output_dir),
        "protocol": dict(formal.METRIC_PROTOCOL),
        "single_candidate_no_sweep": True,
        "target_metric_authorized": True,
    }
    write_frozen_json(authority_output, authority)
    return {"status": "factorized_native_metric_authority_built", "authority": file_record(authority_output)}


def validate_metric_authority(path: str | Path, *, expected_sha256: str) -> dict[str, Any]:
    raw, digest, source_path = load_json_object(
        path, expected_sha256=expected_sha256,
        label="factorized-native metric execution authority",
    )
    authority = dict(raw)
    required = {
        "schema", "schema_version", "status", "source_arm_results", "scene_id",
        "implementation", "frozen_evaluator", "preregistration",
        "external_execution_authority", "external_cache", "external_report",
        "frozen_inputs", "label_root", "output_dir", "protocol",
        "single_candidate_no_sweep", "target_metric_authorized",
    }
    if set(authority) != required:
        raise ValueError("factorized-native metric authority fields differ")
    source_records = _validated_source_records(authority["source_arm_results"])
    source_gate = target_formal.validate_source_arm_winner(source_records)
    if (
        authority["schema"] != formal.METRIC_EXECUTION_SCHEMA
        or authority["schema_version"] != 1
        or authority["status"] != METRIC_STATUS
        or authority["protocol"] != formal.METRIC_PROTOCOL
        or authority["single_candidate_no_sweep"] is not True
        or authority["target_metric_authorized"] is not True
    ):
        raise ValueError("factorized-native metric authority header differs")
    if validate_file_record(authority["implementation"], label="metric implementation") != IMPLEMENTATION:
        raise ValueError("factorized-native metric implementation differs")
    for name, singleton in (
        ("frozen_evaluator", formal.FROZEN_EVALUATOR),
        ("preregistration", formal.FROZEN_PREREGISTRATION),
    ):
        shaped = formal.record(authority[name], label=f"metric {name}")
        if shaped != singleton:
            raise ValueError(f"factorized-native metric {name} differs")
        validate_file_record(shaped, label=f"metric {name}")
    external_record = formal.record(
        authority["external_execution_authority"], label="metric external authority"
    )
    external = validate_external_authority(
        external_record["path"], expected_sha256=external_record["sha256"]
    )
    records = {
        name: formal.record(authority[name], label=f"metric {name}")
        for name in ("external_cache", "external_report")
    }
    inputs = authority["frozen_inputs"]
    input_names = {
        "config", "renderer_geometry_checkpoint", "summary_head",
        "all_query_text_cache", "canonical_negative_text_cache",
    }
    if not isinstance(inputs, Mapping) or set(inputs) != input_names:
        raise ValueError("factorized-native metric frozen inputs differ")
    inputs = {name: formal.record(inputs[name], label=f"metric input {name}") for name in sorted(input_names)}
    for item in (*records.values(), *inputs.values()):
        validate_file_record(item, label="metric bound file")
    cache_raw, _, _ = load_torch_mapping(
        records["external_cache"]["path"], expected_sha256=records["external_cache"]["sha256"],
        map_location="cpu", label="factorized-native external score cache",
    )
    cache = formal.validate_external_cache(cache_raw)
    report_raw, _, _ = load_json_object(
        records["external_report"]["path"],
        expected_sha256=records["external_report"]["sha256"],
        label="factorized-native external cache report",
    )
    report = validate_external_report(
        report_raw, expected_cache=records["external_cache"],
        expected_execution=external_record,
        expected_query_ids=external["verified_relevance"]["query_ids"],
    )
    if (
        external["source_arm_results"] != source_records
        or external["scene_id"] != authority["scene_id"]
        or records["external_cache"]["path"] != external["output_cache"]
        or records["external_report"]["path"] != external["output_report"]
        or inputs["renderer_geometry_checkpoint"]
        != external["renderer_geometry_checkpoint"]
        or inputs["all_query_text_cache"] != formal.FROZEN_ALL_QUERY_CACHE
        or inputs["canonical_negative_text_cache"]
        != formal.FROZEN_CANONICAL_NEGATIVE_BANK
        or cache["metadata"]["query_names"]
        != external["verified_relevance"]["query_ids"]
    ):
        raise ValueError("factorized-native metric nested binding differs")
    label_root = Path(authority["label_root"])
    output_dir = Path(formal.canonical_output(authority["output_dir"], label="metric output directory"))
    if not label_root.is_absolute() or not label_root.is_dir() or label_root.is_symlink():
        raise ValueError("factorized-native metric label root differs")
    authority.update(records)
    authority.update({
        "source_arm_results": source_records,
        "external_execution_authority": external_record,
        "frozen_inputs": inputs, "output_dir": str(output_dir),
        "verified_source_gate": source_gate, "verified_external": external,
        "verified_cache": cache, "verified_external_report": report,
        "verified_record": {"path": str(source_path), "sha256": digest},
    })
    return authority


def run_metric(args: argparse.Namespace) -> dict[str, Any]:
    authority = validate_metric_authority(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
    )
    output = _new(authority["output_dir"], label="metric output directory")
    inputs = authority["frozen_inputs"]
    command = [
        sys.executable, formal.FROZEN_EVALUATOR["path"],
        "--config", inputs["config"]["path"],
        "--checkpoint", inputs["renderer_geometry_checkpoint"]["path"],
        "--scene", authority["scene_id"],
        "--protocol_preset", "vala_paper_3d",
        "--label_dir", authority["label_root"],
        "--output_dir", str(output),
        "--summary_head_weights", inputs["summary_head"]["path"],
        "--text_embedding_cache", inputs["all_query_text_cache"]["path"],
        "--canonical_embedding_cache", inputs["canonical_negative_text_cache"]["path"],
        "--external_query_score_cache", authority["external_cache"]["path"],
        "--gpu", str(args.gpu),
    ]
    subprocess.run(command, check=True)
    result = output / authority["scene_id"] / "lerf_direct_3d_selection_results.json"
    if not result.is_file() or result.is_symlink():
        raise RuntimeError("frozen LERF evaluator did not produce its exact result")
    return {
        "status": "factorized_native_single_frozen_lerf_metric_complete",
        "result": file_record(result),
        "execution_authority": dict(authority["verified_record"]),
    }


def _add_source(parser: argparse.ArgumentParser) -> None:
    for arm in FACTORIZED_NATIVE_READOUT_ARMS:
        option = arm.replace("_", "-")
        parser.add_argument(f"--{option}-result", required=True)
        parser.add_argument(f"--{option}-result-sha256", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("build-query-authority"); _add_source(command)
    for name in (
        "target_descriptor", "descriptor_health_audit", "exact_query_manifest",
        "positive_text_cache",
        "query_relevance_output", "output_authority",
    ):
        command.add_argument("--" + name.replace("_", "-"), required=True)
    command.set_defaults(func=build_query_authority)
    command = sub.add_parser("materialize-query")
    command.add_argument("--execution-authority", required=True)
    command.add_argument("--expected-execution-authority-sha256", required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(func=materialize_query)
    command = sub.add_parser("build-external-authority"); _add_source(command)
    for name in (
        "query_execution_authority", "query_relevance",
        "comembership_feature_authority", "comembership_inference_authority",
        "renderer_geometry_checkpoint", "output_cache", "output_report", "output_authority",
    ):
        command.add_argument("--" + name.replace("_", "-"), required=True)
    command.set_defaults(func=build_external_authority)
    command = sub.add_parser("materialize-external")
    command.add_argument("--execution-authority", required=True)
    command.add_argument("--expected-execution-authority-sha256", required=True)
    command.set_defaults(func=materialize_external)
    command = sub.add_parser("build-metric-authority"); _add_source(command)
    for name in (
        "external_execution_authority", "external_cache", "external_report", "config",
        "renderer_geometry_checkpoint", "summary_head", "label_root", "output_dir",
        "output_authority",
    ):
        command.add_argument("--" + name.replace("_", "-"), required=True)
    command.set_defaults(func=build_metric_authority)
    command = sub.add_parser("run-metric")
    command.add_argument("--execution-authority", required=True)
    command.add_argument("--expected-execution-authority-sha256", required=True)
    command.add_argument("--gpu", type=int, default=0)
    command.set_defaults(func=run_metric)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(args.func(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "build_external_authority", "build_metric_authority", "build_query_authority",
    "build_parser", "canonical_external_selection", "materialize_external",
    "materialize_query", "run_metric",
    "validate_external_authority", "validate_external_report",
    "validate_metric_authority", "validate_query_authority",
]
