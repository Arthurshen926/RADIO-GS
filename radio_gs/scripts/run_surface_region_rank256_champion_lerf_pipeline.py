#!/usr/bin/env python3
"""Source-gated rank-256 target/query/cache and one-shot LERF pipeline.

No subcommand may inspect target or query artifacts before the selected V2.1B
or V2.1C source promotion chain passes.  Outputs are first-writer-only.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch

from radio_gs.interfaces.factorized_primitive_state import (
    load_factorized_primitive_state,
)
from radio_gs.interfaces.surface_region_full_scalar_contract import (
    aggregate_surface_region_full_scalars,
)
from radio_gs.interfaces.surface_region_target_accepted_v2 import (
    validate_target_accepted_v2_authority,
)
from radio_gs.interfaces.surface_region_target_adaptive_typed_context import (
    validate_target_adaptive_typed_context_authority,
)
from radio_gs.interfaces import surface_region_rank256_champion as formal
from radio_gs.losses.source_global_response_listwise_loss_v21 import (
    load_frozen_canonical_negative_bank,
)
from radio_gs.querying.v21_absolute_relevance_adapter import (
    CANONICAL_NEGATIVE_MODEL,
    OFFICIAL_TEXT_CANONICALIZATION,
    calibrated_v21_absolute_relevance,
    load_v21_positive_text_bank,
)
from radio_gs.scripts import (
    build_lerf_region_comembership_external_cache_v2 as v2_external,
)
from radio_gs.scripts import (
    build_lerf_region_comembership_external_cache_v21 as v21_external,
)
from radio_gs.scripts import eval_lerf_direct_3d_selection as frozen_evaluator
from radio_gs.scripts import (
    run_surface_region_rank256_champion_target as frozen_target,
)
from radio_gs.scripts import materialize_full_scalar_clean_training_shard as shard
from radio_gs.scripts import materialize_surface_region_v21_target_descriptor as v21_target
from radio_gs.scripts import train_surface_region_typed_context_response_listwise_v21_pilot as routing
from radio_gs.scripts.infer_region_comembership_v2 import (
    validate_inference_authority,
)
from radio_gs.scripts.materialize_region_comembership_features_v2 import (
    validate_feature_authority,
)
from radio_gs.scripts.materialize_lerf_multiscale_query_score_cache import (
    _renderer_checkpoint_xyz,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_sha_bound_project_checkpoint_mapping,
    load_torch_mapping,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


IMPLEMENTATION = Path(__file__).resolve()
PREREGISTRATION = (
    Path(__file__).resolve().parents[2]
    / "paper/artifacts/lerf_v21_absolute_relevance_greedy_novelty_union_preregistration_20260807.json"
)
TARGET_STATUS = "authorized_after_rank256_source_pass_for_query_free_target"
QUERY_STATUS = "authorized_after_rank256_source_pass_for_exact_query_relevance"
EXTERNAL_STATUS = "authorized_rank256_champion_frozen_lerf_readout"
METRIC_STATUS = "authorized_rank256_champion_single_frozen_lerf_metric"


def _existing(value: object, *, label: str) -> Path:
    raw = str(value)
    path = Path(raw).expanduser().resolve()
    if raw != str(path) or not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be an existing canonical regular file")
    return path


def _new(value: object, *, label: str) -> Path:
    raw = str(value)
    path = Path(raw).expanduser().resolve()
    if raw != str(path):
        raise ValueError(f"{label} must be a new canonical path")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{label} already exists: {path}")
    return path


def _source(args: argparse.Namespace) -> dict[str, Any]:
    return formal.validate_champion_source(
        args.source_variant,
        args.source_result,
        expected_sha256=args.expected_source_result_sha256,
    )


def _load_legacy_frozen_all_query_cache(
    path: str | Path,
    *,
    expected_sha256: str,
) -> tuple[list[str], torch.Tensor]:
    """Load the exact frozen pre-V2.1 five-field SigLIP2 query cache.

    This compatibility boundary is intentionally restricted to the all-query
    input.  The emitted scene subset uses the stricter six-field V2.1 contract.
    """

    payload, _, _ = load_torch_mapping(
        path,
        expected_sha256=expected_sha256,
        map_location="cpu",
        label="legacy frozen all-query cache",
    )
    required = {
        "queries", "prompt_templates", "text_encoder", "model_name",
        "embeddings",
    }
    queries = payload.get("queries")
    embeddings = payload.get("embeddings")
    if (
        set(payload) != required
        or not isinstance(queries, list)
        or not queries
        or any(not isinstance(item, str) or not item.strip() for item in queries)
        or len(set(queries)) != len(queries)
        or payload.get("prompt_templates") != ["{query}"]
        or payload.get("text_encoder") != "siglip2"
        or payload.get("model_name") != CANONICAL_NEGATIVE_MODEL
        or not torch.is_tensor(embeddings)
        or embeddings.dtype != torch.float32
        or embeddings.device.type != "cpu"
        or embeddings.shape != (len(queries), 1536)
        or not bool(torch.isfinite(embeddings).all())
    ):
        raise ValueError("legacy frozen all-query cache contract differs")
    normalized = embeddings.detach().contiguous()
    norms = torch.linalg.vector_norm(normalized, dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), rtol=0.0, atol=2e-4):
        raise ValueError("legacy frozen all-query embeddings are not unit L2")
    return list(queries), normalized


def _validate_frozen_target_binding(
    descriptor: Mapping[str, Any],
    target_execution: Mapping[str, Any],
) -> None:
    """Bind a descriptor to the exact frozen target producer and inputs."""

    if (
        descriptor["producer"] != target_execution["implementation"]
        or descriptor["target_execution_authority"]
        != target_execution["verified_record"]
        or descriptor["input_authority"] != target_execution["target_inputs"]
        or descriptor["scene_id"] != target_execution["scene_id"]
        or descriptor["physical_space_id"]
        != target_execution["physical_space_id"]
    ):
        raise ValueError("rank-256 descriptor/frozen-target binding differs")


def build_target_authority(args: argparse.Namespace) -> dict[str, Any]:
    gate = _source(args)  # Must remain the first artifact access.
    authority_output = _new(args.output_authority, label="target authority output")
    descriptor_output = _new(args.target_descriptor_output, label="target descriptor output")
    accepted = _existing(args.target_accepted_v2, label="target AcceptedV2")
    adaptive = _existing(args.target_adaptive_typed_context, label="target adaptive context")
    state = _existing(args.factorized_primitive_state, label="factorized primitive state")
    physical = formal.physical_authority(
        args.dataset_id, args.scene_id, args.geometry_checkpoint_sha256
    )
    authority = {
        "schema": formal.TARGET_EXECUTION_SCHEMA,
        "schema_version": 1,
        "status": TARGET_STATUS,
        "source_variant": args.source_variant,
        "source_result": dict(gate["source_result"]),
        "scene_id": physical["scene_id"],
        "physical_space_id": physical["physical_space_id"],
        "implementation": file_record(IMPLEMENTATION),
        "preregistration": file_record(PREREGISTRATION),
        "target_inputs": {
            "target_accepted_v2": file_record(accepted),
            "target_adaptive_typed_context": file_record(adaptive),
            "factorized_primitive_state": file_record(state),
            "champion_checkpoint": dict(gate["checkpoint"]),
            "champion_normalization": dict(gate["normalization_authority"]),
        },
        "target_descriptor_output": str(descriptor_output),
        "materialization_authorized": True,
        "query_execution_authorized": False,
        "metric_execution_authorized": False,
        "access_audit": formal.target_access_audit(),
    }
    written = write_frozen_json(authority_output, authority)
    return {"status": "rank256_target_authority_built", "authority": file_record(written)}


def validate_target_authority(
    path: str | Path, *, expected_sha256: str, expected_output: str | Path | None = None
) -> dict[str, Any]:
    raw, digest, source = load_json_object(
        path, expected_sha256=expected_sha256, label="rank-256 target authority"
    )
    required = {
        "schema", "schema_version", "status", "source_variant", "source_result",
        "scene_id", "physical_space_id", "implementation", "preregistration",
        "target_inputs", "target_descriptor_output", "materialization_authorized",
        "query_execution_authorized", "metric_execution_authorized", "access_audit",
    }
    authority = dict(raw)
    if not required == set(authority):
        raise ValueError("rank-256 target authority fields differ")
    source_record = formal._record(authority["source_result"], label="target source result")
    gate = formal.validate_champion_source(
        authority["source_variant"], source_record["path"], expected_sha256=source_record["sha256"]
    )
    if (
        authority["schema"] != formal.TARGET_EXECUTION_SCHEMA
        or authority["schema_version"] != 1 or authority["status"] != TARGET_STATUS
        or authority["source_variant"] not in formal.SOURCE_VARIANTS
        or authority["materialization_authorized"] is not True
        or authority["query_execution_authorized"] is not False
        or authority["metric_execution_authorized"] is not False
        or authority["access_audit"] != formal.target_access_audit()
        or source_record != gate["source_result"]
    ):
        raise ValueError("rank-256 target authority header differs")
    if validate_file_record(authority["implementation"], label="target implementation") != IMPLEMENTATION:
        raise ValueError("rank-256 target implementation differs")
    if validate_file_record(authority["preregistration"], label="target preregistration") != PREREGISTRATION:
        raise ValueError("rank-256 target preregistration differs")
    inputs = authority["target_inputs"]
    expected_names = {
        "target_accepted_v2", "target_adaptive_typed_context", "factorized_primitive_state",
        "champion_checkpoint", "champion_normalization",
    }
    if not isinstance(inputs, Mapping) or set(inputs) != expected_names:
        raise ValueError("rank-256 target inputs differ")
    authority["target_inputs"] = {
        name: formal._record(inputs[name], label=f"target {name}") for name in sorted(expected_names)
    }
    if (
        authority["target_inputs"]["champion_checkpoint"] != gate["checkpoint"]
        or authority["target_inputs"]["champion_normalization"] != gate["normalization_authority"]
    ):
        raise ValueError("rank-256 target source model binding differs")
    output = formal._output(authority["target_descriptor_output"], label="target descriptor output")
    if expected_output is not None and output != str(Path(expected_output).expanduser().resolve()):
        raise ValueError("rank-256 target output differs")
    authority.update({
        "source_result": source_record,
        "target_descriptor_output": output,
        "verified_source_gate": gate,
        "verified_record": {"path": str(source), "sha256": digest},
    })
    return authority


def _target_inputs(execution: Mapping[str, Any]) -> dict[str, Any]:
    records = execution["target_inputs"]
    accepted_raw, _, _ = load_torch_mapping(
        records["target_accepted_v2"]["path"], expected_sha256=records["target_accepted_v2"]["sha256"],
        map_location="cpu", label="rank-256 target AcceptedV2",
    )
    adaptive_raw, _, _ = load_torch_mapping(
        records["target_adaptive_typed_context"]["path"], expected_sha256=records["target_adaptive_typed_context"]["sha256"],
        map_location="cpu", label="rank-256 target adaptive context",
    )
    return {
        "records": records,
        "accepted": validate_target_accepted_v2_authority(accepted_raw),
        "adaptive": validate_target_adaptive_typed_context_authority(adaptive_raw),
        "state": load_factorized_primitive_state(
            records["factorized_primitive_state"]["path"],
            expected_sha256=records["factorized_primitive_state"]["sha256"],
        ),
    }


def materialize_target(args: argparse.Namespace) -> dict[str, Any]:
    output = _new(args.output, label="rank-256 target descriptor")
    execution = validate_target_authority(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
        expected_output=output,
    )
    inputs = _target_inputs(execution)
    v21_target._validate_alignment(inputs)
    accepted, adaptive, state = inputs["accepted"], inputs["adaptive"], inputs["state"]
    summary = aggregate_surface_region_full_scalars(
        state, accepted["accepted_base_valid"], accepted["region_rows"],
        accepted["token_mask"], accepted["anchor_index"],
    )
    model, normalization, _checkpoint = formal.load_champion_model(
        execution["verified_source_gate"], execution["source_variant"]
    )
    scene = {
        "raw_full_scalar_summary": summary.summary,
        "typed_context_statistics": adaptive["typed_context_statistics"],
        "eligible": summary.use_full_scalar_mask,
        "typed_context_valid": adaptive["typed_context_valid"],
    }
    declared, effective_ood, active = routing._pilot_routing(scene, normalization)
    regions = int(accepted["accepted_v2_e0"].shape[0])
    descriptor = torch.empty_like(accepted["accepted_v2_e0"].float().cpu())
    reliability = torch.empty(regions, dtype=torch.float32)
    budget = torch.empty(regions, dtype=torch.float32)
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise ValueError("rank-256 target batch size must be positive")
    with torch.inference_mode():
        for start in range(0, regions, batch_size):
            stop = min(start + batch_size, regions)
            diag = model.forward_with_diagnostics(
                accepted["accepted_v2_e0"][start:stop].float().cpu(),
                adaptive["pooled_context_radio_direction"][start:stop].float().cpu(),
                summary.summary[start:stop].float().cpu(),
                adaptive["typed_context_statistics"][start:stop].float().cpu(),
                active_mask=declared[start:stop], ood_mask=effective_ood[start:stop],
            )
            descriptor[start:stop] = diag.semantic_descriptor
            reliability[start:stop] = diag.reliability_score
            budget[start:stop] = diag.angular_budget_radians
    fallback = ~active
    fallback_equal = torch.equal(descriptor[fallback], accepted["accepted_v2_e0"][fallback].float().cpu())
    if not fallback_equal:
        raise RuntimeError("rank-256 target immutable fallback changed")
    masks = {
        "full_scalar_eligible_mask": summary.use_full_scalar_mask.bool().cpu().contiguous(),
        "typed_context_valid_mask": adaptive["typed_context_valid"].bool().cpu().contiguous(),
        "normalization_ood_mask": (effective_ood & summary.use_full_scalar_mask).bool().cpu().contiguous(),
        "effective_ood_mask": effective_ood.bool().cpu().contiguous(),
        "active_update_mask": active.bool().cpu().contiguous(),
        "immutable_fallback_mask": fallback.bool().cpu().contiguous(),
        "descriptor_changed_mask": (descriptor != accepted["accepted_v2_e0"].float().cpu()).any(dim=-1),
    }
    payload = {
        "schema": formal.TARGET_DESCRIPTOR_SCHEMA, "schema_version": 1,
        "contract": formal.target_contract(), "contract_sha256": formal.TARGET_CONTRACT_SHA256,
        "source_variant": execution["source_variant"], "scene_id": accepted["scene_id"],
        "physical_space_id": accepted["physical_space_id"],
        "physical_space_authority": dict(accepted["physical_space_authority"]),
        "producer": file_record(IMPLEMENTATION),
        "target_execution_authority": dict(execution["verified_record"]),
        "input_authority": dict(execution["target_inputs"]),
        "region_row_ids": list(adaptive["region_row_ids"]),
        "canonical_region_indices": accepted["canonical_region_indices"].clone(),
        "region_fingerprints": list(accepted["region_fingerprints"]),
        "semantic_descriptor": descriptor.float().cpu().contiguous(),
        "reliability_score": reliability.contiguous(),
        "angular_budget_radians": budget.contiguous(), **masks,
        "fallback_bitwise_equal": fallback_equal,
        "routing_audit": {
            "regions": regions, "active_update": int(active.sum()),
            "immutable_fallback": int(fallback.sum()),
            "descriptor_changed": int(masks["descriptor_changed_mask"].sum()),
        },
        "access_audit": formal.target_access_audit(),
    }
    payload["channel_sha256"] = formal.target_channel_sha256(payload)
    payload = formal.validate_target_descriptor(payload)
    write_torch_noclobber(output, payload)
    return {"status": "rank256_target_descriptor_complete", "output": file_record(output)}


def materialize_exact_query_subset(args: argparse.Namespace) -> dict[str, Any]:
    gate = _source(args)  # Source PASS before manifest/cache/output inspection.
    output = _new(args.output, label="exact-query cache output")
    receipt_output = _new(args.output_receipt, label="exact-query receipt output")
    if output == receipt_output:
        raise ValueError("exact-query cache and receipt outputs must differ")
    manifest_path = _existing(args.query_manifest, label="exact query manifest")
    all_path = _existing(args.all_query_cache, label="frozen all-query cache")
    manifest_raw, manifest_sha, manifest_source = load_json_object(
        manifest_path, expected_sha256=args.expected_query_manifest_sha256,
        label="exact scene query manifest",
    )
    manifest = formal.validate_exact_query_manifest(manifest_raw, scene_id=args.scene_id)
    all_record = file_record(all_path)
    if all_record["sha256"] != args.expected_all_query_cache_sha256 or manifest["frozen_all_query_cache"] != all_record:
        raise ValueError("exact manifest/all-query cache binding differs")
    evaluator_path = validate_file_record(
        manifest["frozen_evaluator"], label="exact-query evaluator"
    )
    if evaluator_path != Path(frozen_evaluator.__file__).resolve():
        raise ValueError("exact-query manifest evaluator differs from frozen evaluator")
    all_query_ids, all_embeddings = _load_legacy_frozen_all_query_cache(
        all_path, expected_sha256=all_record["sha256"]
    )
    manifest_ids = list(manifest["query_ids"])
    positions = {name: index for index, name in enumerate(all_query_ids)}
    if len(positions) != len(all_query_ids) or any(name not in positions for name in manifest_ids):
        raise ValueError("exact scene queries are not an exact subset of frozen all-query cache")
    indices = torch.tensor([positions[name] for name in manifest_ids], dtype=torch.long)
    exact = {
        "queries": manifest_ids,
        "prompt_templates": ["{query}"],
        "text_encoder": "siglip2",
        "model_name": CANONICAL_NEGATIVE_MODEL,
        "text_canonicalization": OFFICIAL_TEXT_CANONICALIZATION,
        "embeddings": all_embeddings.index_select(0, indices).contiguous(),
    }
    write_torch_noclobber(output, exact)
    receipt = {
        "schema": formal.EXACT_QUERY_RECEIPT_SCHEMA, "schema_version": 1,
        "status": "source_gated_exact_query_subset_complete",
        "source_variant": args.source_variant, "source_result": dict(gate["source_result"]),
        "scene_id": args.scene_id,
        "query_manifest": {"path": str(manifest_source), "sha256": manifest_sha},
        "all_query_cache": all_record, "output_cache": file_record(output),
        "query_ids": manifest_ids, "query_ids_sha256": canonical_json_sha256(manifest_ids),
        "access_audit": formal.query_access_audit(),
    }
    write_frozen_json(receipt_output, receipt)
    return {"status": "exact_query_subset_complete", "cache": file_record(output), "receipt": file_record(receipt_output)}


def build_query_authority(args: argparse.Namespace) -> dict[str, Any]:
    gate = _source(args)
    authority_output = _new(args.output_authority, label="query authority output")
    relevance_output = _new(args.query_relevance_output, label="query relevance output")
    descriptor_path = _existing(args.target_descriptor, label="rank-256 target descriptor")
    positive_path = _existing(args.positive_text_cache, label="exact positive cache")
    receipt_path = _existing(args.positive_text_receipt, label="exact positive receipt")
    descriptor_record = file_record(descriptor_path)
    descriptor_raw, _, _ = load_torch_mapping(descriptor_path, expected_sha256=descriptor_record["sha256"], map_location="cpu", label="rank-256 target descriptor")
    descriptor = formal.validate_target_descriptor(descriptor_raw)
    if descriptor["source_variant"] != args.source_variant:
        raise ValueError("query descriptor source variant differs")
    target_exec = frozen_target.validate_target_authority(
        descriptor["target_execution_authority"]["path"],
        expected_sha256=descriptor["target_execution_authority"]["sha256"],
        expected_output=descriptor_path,
    )
    _validate_frozen_target_binding(descriptor, target_exec)
    if target_exec["source_result"] != gate["source_result"]:
        raise ValueError("query descriptor uses another source promotion")
    receipt_record = file_record(receipt_path)
    receipt_raw, _, _ = load_json_object(receipt_path, expected_sha256=receipt_record["sha256"], label="exact positive receipt")
    receipt = formal.validate_exact_query_receipt(receipt_raw)
    if receipt["source_result"] != gate["source_result"] or receipt["output_cache"] != file_record(positive_path) or receipt["scene_id"] != descriptor["scene_id"]:
        raise ValueError("exact positive cache source/scene binding differs")
    authority = {
        "schema": formal.QUERY_EXECUTION_SCHEMA, "schema_version": 1,
        "status": QUERY_STATUS, "source_variant": args.source_variant,
        "source_result": dict(gate["source_result"]), "scene_id": descriptor["scene_id"],
        "physical_space_id": descriptor["physical_space_id"],
        "implementation": file_record(IMPLEMENTATION), "preregistration": file_record(PREREGISTRATION),
        "target_descriptor": descriptor_record, "positive_text_cache": file_record(positive_path),
        "positive_text_receipt": receipt_record,
        "canonical_negative_bank": formal.source_canonical_negative(gate, args.source_variant),
        "query_relevance_output": str(relevance_output),
        "query_execution_authorized": True, "metric_execution_authorized": False,
        "access_audit": formal.query_access_audit(),
    }
    written = write_frozen_json(authority_output, authority)
    return {"status": "rank256_query_authority_built", "authority": file_record(written)}


def validate_query_authority(path: str | Path, *, expected_sha256: str, expected_output: str | Path | None = None) -> dict[str, Any]:
    raw, digest, source = load_json_object(path, expected_sha256=expected_sha256, label="rank-256 query authority")
    required = {
        "schema", "schema_version", "status", "source_variant", "source_result",
        "scene_id", "physical_space_id", "implementation", "preregistration",
        "target_descriptor", "positive_text_cache", "positive_text_receipt",
        "canonical_negative_bank", "query_relevance_output",
        "query_execution_authorized", "metric_execution_authorized", "access_audit",
    }
    authority = dict(raw)
    if set(authority) != required:
        raise ValueError("rank-256 query authority fields differ")
    source_record = formal._record(authority["source_result"], label="query source result")
    gate = formal.validate_champion_source(authority["source_variant"], source_record["path"], expected_sha256=source_record["sha256"])
    if (
        authority["schema"] != formal.QUERY_EXECUTION_SCHEMA or authority["schema_version"] != 1
        or authority["status"] != QUERY_STATUS or authority["query_execution_authorized"] is not True
        or authority["metric_execution_authorized"] is not False
        or authority["access_audit"] != formal.query_access_audit()
        or source_record != gate["source_result"]
    ):
        raise ValueError("rank-256 query authority header differs")
    if validate_file_record(authority["implementation"], label="query implementation") != IMPLEMENTATION or validate_file_record(authority["preregistration"], label="query preregistration") != PREREGISTRATION:
        raise ValueError("rank-256 query implementation/preregistration differs")
    records = {name: formal._record(authority[name], label=f"query {name}") for name in ("target_descriptor", "positive_text_cache", "positive_text_receipt", "canonical_negative_bank")}
    if records["canonical_negative_bank"] != formal.source_canonical_negative(gate, authority["source_variant"]):
        raise ValueError("rank-256 query negative bank differs from source")
    descriptor_raw, _, _ = load_torch_mapping(records["target_descriptor"]["path"], expected_sha256=records["target_descriptor"]["sha256"], map_location="cpu", label="rank-256 target descriptor")
    descriptor = formal.validate_target_descriptor(descriptor_raw)
    target_exec = frozen_target.validate_target_authority(
        descriptor["target_execution_authority"]["path"],
        expected_sha256=descriptor["target_execution_authority"]["sha256"],
        expected_output=records["target_descriptor"]["path"],
    )
    _validate_frozen_target_binding(descriptor, target_exec)
    receipt_raw, _, _ = load_json_object(records["positive_text_receipt"]["path"], expected_sha256=records["positive_text_receipt"]["sha256"], label="exact positive receipt")
    receipt = formal.validate_exact_query_receipt(receipt_raw)
    if (
        descriptor["source_variant"] != authority["source_variant"]
        or target_exec["source_result"] != source_record
        or descriptor["scene_id"] != authority["scene_id"]
        or descriptor["physical_space_id"] != authority["physical_space_id"]
        or receipt["source_result"] != source_record or receipt["scene_id"] != authority["scene_id"]
        or receipt["output_cache"] != records["positive_text_cache"]
    ):
        raise ValueError("rank-256 query nested binding differs")
    output = formal._output(authority["query_relevance_output"], label="query relevance output")
    if expected_output is not None and output != str(Path(expected_output).expanduser().resolve()):
        raise ValueError("rank-256 query output differs")
    authority.update(records)
    authority.update({"source_result": source_record, "query_relevance_output": output, "verified_source_gate": gate, "verified_descriptor": descriptor, "verified_record": {"path": str(source), "sha256": digest}})
    return authority


def materialize_query(args: argparse.Namespace) -> dict[str, Any]:
    output = _new(args.output, label="rank-256 query relevance")
    execution = validate_query_authority(args.execution_authority, expected_sha256=args.expected_execution_authority_sha256, expected_output=output)
    positive = load_v21_positive_text_bank(execution["positive_text_cache"]["path"], expected_file_sha256=execution["positive_text_cache"]["sha256"])
    negative = load_frozen_canonical_negative_bank(execution["canonical_negative_bank"]["path"], expected_file_sha256=execution["canonical_negative_bank"]["sha256"])
    descriptor = execution["verified_descriptor"]
    relevance = calibrated_v21_absolute_relevance(descriptor["semantic_descriptor"], positive_bank=positive, canonical_negative_bank=negative)
    payload = {
        "schema": formal.QUERY_RELEVANCE_SCHEMA, "schema_version": 1,
        "contract": formal.query_contract(), "contract_sha256": formal.QUERY_CONTRACT_SHA256,
        "source_variant": execution["source_variant"], "scene_id": descriptor["scene_id"],
        "physical_space_id": descriptor["physical_space_id"], "producer": file_record(IMPLEMENTATION),
        "query_execution_authority": dict(execution["verified_record"]),
        "input_authority": {name: dict(execution[name]) for name in ("target_descriptor", "positive_text_cache", "positive_text_receipt", "canonical_negative_bank")},
        "region_row_ids": list(descriptor["region_row_ids"]),
        "canonical_region_indices": descriptor["canonical_region_indices"].clone(),
        "region_fingerprints": list(descriptor["region_fingerprints"]),
        "query_ids": list(positive.query_ids), "region_absolute_relevance": relevance,
        "access_audit": formal.query_access_audit(),
    }
    payload["channel_sha256"] = formal.query_channel_sha256(payload)
    payload = formal.validate_query_relevance(payload)
    write_torch_noclobber(output, payload)
    return {"status": "rank256_query_relevance_complete", "output": file_record(output)}


def build_external_authority(args: argparse.Namespace) -> dict[str, Any]:
    gate = _source(args)
    authority_output = _new(args.output_authority, label="external authority output")
    cache_output = _new(args.output_cache, label="external cache output")
    report_output = _new(args.output_report, label="external report output")
    query_exec_path = _existing(args.query_execution_authority, label="query execution authority")
    relevance_path = _existing(args.query_relevance, label="query relevance")
    feature_path = _existing(args.comembership_feature_authority, label="co-membership features")
    inference_path = _existing(args.comembership_inference_authority, label="co-membership inference")
    renderer_path = _existing(args.renderer_geometry_checkpoint, label="renderer checkpoint")
    query_exec_record = file_record(query_exec_path)
    query_exec = validate_query_authority(query_exec_path, expected_sha256=query_exec_record["sha256"], expected_output=relevance_path)
    if query_exec["source_result"] != gate["source_result"]:
        raise ValueError("external cache query chain uses another source")
    authority = {
        "schema": formal.EXTERNAL_EXECUTION_SCHEMA, "schema_version": 1,
        "status": EXTERNAL_STATUS, "source_variant": args.source_variant,
        "source_result": dict(gate["source_result"]), "scene_id": query_exec["scene_id"],
        "physical_space_id": query_exec["physical_space_id"], "implementation": file_record(IMPLEMENTATION),
        "preregistration": file_record(PREREGISTRATION), "query_execution_authority": query_exec_record,
        "query_relevance": file_record(relevance_path), "comembership_feature_authority": file_record(feature_path),
        "comembership_inference_authority": file_record(inference_path), "renderer_geometry_checkpoint": file_record(renderer_path),
        "output_cache": str(cache_output), "output_report": str(report_output),
        "query_readout_authorized": True, "target_metric_authorized": False,
        "access_audit": formal.query_access_audit(),
    }
    written = write_frozen_json(authority_output, authority)
    return {"status": "rank256_external_authority_built", "authority": file_record(written)}


def validate_external_authority(path: str | Path, *, expected_sha256: str) -> dict[str, Any]:
    raw, digest, source = load_json_object(path, expected_sha256=expected_sha256, label="rank-256 external authority")
    required = {
        "schema", "schema_version", "status", "source_variant", "source_result", "scene_id",
        "physical_space_id", "implementation", "preregistration", "query_execution_authority",
        "query_relevance", "comembership_feature_authority", "comembership_inference_authority",
        "renderer_geometry_checkpoint", "output_cache", "output_report", "query_readout_authorized",
        "target_metric_authorized", "access_audit",
    }
    authority = dict(raw)
    if set(authority) != required:
        raise ValueError("rank-256 external authority fields differ")
    source_record = formal._record(authority["source_result"], label="external source result")
    gate = formal.validate_champion_source(authority["source_variant"], source_record["path"], expected_sha256=source_record["sha256"])
    if (
        authority["schema"] != formal.EXTERNAL_EXECUTION_SCHEMA or authority["schema_version"] != 1
        or authority["status"] != EXTERNAL_STATUS or authority["query_readout_authorized"] is not True
        or authority["target_metric_authorized"] is not False or authority["access_audit"] != formal.query_access_audit()
        or source_record != gate["source_result"]
    ):
        raise ValueError("rank-256 external authority header differs")
    if validate_file_record(authority["implementation"], label="external implementation") != IMPLEMENTATION or validate_file_record(authority["preregistration"], label="external preregistration") != PREREGISTRATION:
        raise ValueError("rank-256 external implementation/preregistration differs")
    names = ("query_execution_authority", "query_relevance", "comembership_feature_authority", "comembership_inference_authority", "renderer_geometry_checkpoint")
    records = {name: formal._record(authority[name], label=f"external {name}") for name in names}
    query_exec = validate_query_authority(records["query_execution_authority"]["path"], expected_sha256=records["query_execution_authority"]["sha256"], expected_output=records["query_relevance"]["path"])
    relevance_raw, _, _ = load_torch_mapping(records["query_relevance"]["path"], expected_sha256=records["query_relevance"]["sha256"], map_location="cpu", label="rank-256 query relevance")
    relevance = formal.validate_query_relevance(relevance_raw)
    feature_raw, _, _ = load_torch_mapping(records["comembership_feature_authority"]["path"], expected_sha256=records["comembership_feature_authority"]["sha256"], map_location="cpu", label="co-membership features")
    inference_raw, _, _ = load_torch_mapping(records["comembership_inference_authority"]["path"], expected_sha256=records["comembership_inference_authority"]["sha256"], map_location="cpu", label="co-membership inference")
    feature = validate_feature_authority(feature_raw)
    inference = validate_inference_authority(inference_raw)
    selected_rule = v2_external.validate_v2_authority_binding(feature=feature, inference=inference, feature_record=records["comembership_feature_authority"])
    descriptor = query_exec["verified_descriptor"]
    receipt_raw, _, _ = load_json_object(
        query_exec["positive_text_receipt"]["path"],
        expected_sha256=query_exec["positive_text_receipt"]["sha256"],
        label="exact positive receipt",
    )
    receipt = formal.validate_exact_query_receipt(receipt_raw)
    accepted_record = formal._record(feature.get("input_authority", {}).get("accepted_v2"), label="external AcceptedV2")
    accepted_raw, _, _ = load_torch_mapping(accepted_record["path"], expected_sha256=accepted_record["sha256"], map_location="cpu", label="external AcceptedV2")
    accepted = validate_target_accepted_v2_authority(accepted_raw)
    renderer_raw, renderer_sha, _ = load_sha_bound_project_checkpoint_mapping(records["renderer_geometry_checkpoint"]["path"], expected_sha256=records["renderer_geometry_checkpoint"]["sha256"], map_location="cpu", label="renderer checkpoint")
    renderer_xyz = _renderer_checkpoint_xyz(renderer_raw)
    v2_external.validate_renderer_geometry_binding(feature=feature, accepted=accepted, accepted_record=accepted_record, renderer_geometry_checkpoint_sha256=renderer_sha)
    state_record = descriptor["input_authority"]["factorized_primitive_state"]
    state = load_factorized_primitive_state(state_record["path"], expected_sha256=state_record["sha256"])
    if (
        query_exec["source_result"] != source_record or relevance["query_execution_authority"] != records["query_execution_authority"]
        or relevance["producer"] != authority["implementation"]
        or relevance["input_authority"]
        != {
            name: query_exec[name]
            for name in (
                "target_descriptor", "positive_text_cache",
                "positive_text_receipt", "canonical_negative_bank",
            )
        }
        or relevance["source_variant"] != authority["source_variant"]
        or descriptor["scene_id"] != authority["scene_id"] or relevance["scene_id"] != authority["scene_id"]
        or descriptor["physical_space_id"] != authority["physical_space_id"]
        or relevance["physical_space_id"] != authority["physical_space_id"]
        or accepted_record != descriptor["input_authority"]["target_accepted_v2"]
        or relevance["region_row_ids"] != descriptor["region_row_ids"]
        or descriptor["region_fingerprints"] != feature["region_fingerprints"]
        or relevance["region_fingerprints"] != descriptor["region_fingerprints"]
        or relevance["query_ids"] != receipt["query_ids"]
        or not torch.equal(descriptor["canonical_region_indices"], feature["canonical_region_indices"])
        or not torch.equal(
            relevance["canonical_region_indices"],
            descriptor["canonical_region_indices"],
        )
        or state.xyz.shape != renderer_xyz.shape or not torch.equal(state.xyz.float(), renderer_xyz)
    ):
        raise ValueError("rank-256 external descriptor/graph/renderer binding differs")
    output = formal._output(authority["output_cache"], label="external cache output")
    report = formal._output(authority["output_report"], label="external report output")
    if output == report:
        raise ValueError("external cache and report outputs must differ")
    authority.update(records)
    authority.update({"source_result": source_record, "output_cache": output, "output_report": report, "verified_source_gate": gate, "verified_descriptor": descriptor, "verified_relevance": relevance, "verified_feature": feature, "verified_inference": inference, "verified_state": state, "selected_rule": selected_rule, "verified_record": {"path": str(source), "sha256": digest}})
    return authority


def materialize_external(args: argparse.Namespace) -> dict[str, Any]:
    execution = validate_external_authority(args.execution_authority, expected_sha256=args.expected_execution_authority_sha256)
    output = _new(execution["output_cache"], label="external cache output")
    report_output = _new(execution["output_report"], label="external report output")
    state = execution["verified_state"]
    readout = v21_external.greedy_novelty_readout_from_v21(feature=execution["verified_feature"], inference=execution["verified_inference"], relevance=execution["verified_relevance"], num_primitives=int(state.xyz.shape[0]))
    membership, removed = v21_external.mask_union_to_valid(readout.primitive_membership, state.valid)
    query_ids = list(execution["verified_relevance"]["query_ids"])
    cache = {
        "schema": formal.EXTERNAL_CACHE_SCHEMA, "query_scores": membership,
        "valid": state.valid.bool().cpu().contiguous(), "xyz": state.xyz.float().cpu().contiguous(),
        "metadata": {
            "query_names": query_ids, "score_semantics": "binary_rank256_absolute_relevance_greedy_novelty_union_membership",
            "semantic_boundary": v21_external.V21_ABSOLUTE_RELEVANCE_BOUNDARY,
            "maximum_regions": v21_external.V21_MAXIMUM_REGIONS, "postprocess_before_region_readout": "none",
            "producer": file_record(IMPLEMENTATION), "execution_authority": dict(execution["verified_record"]),
        },
        "selection": {"region_indices": readout.selected_region_indices, "region_scores": readout.selected_region_scores, "marginal_core_rows": readout.selected_marginal_core_rows, "invalid_memberships_removed": removed},
    }
    write_torch_noclobber(output, cache)
    report = {
        "schema": formal.EXTERNAL_CACHE_SCHEMA, "status": "rank256_champion_external_cache_complete",
        "cache": file_record(output), "query_ids": query_ids,
        "selected_region_counts": [len(value) for value in readout.selected_region_indices],
        "selected_primitive_counts": membership.sum(dim=0).int().tolist(),
        "invalid_memberships_removed": removed, "execution_authority": dict(execution["verified_record"]),
        "access_audit": formal.query_access_audit(),
    }
    write_frozen_json(report_output, report)
    return report


def build_metric_authority(args: argparse.Namespace) -> dict[str, Any]:
    gate = _source(args)
    authority_output = _new(args.output_authority, label="metric authority output")
    output_dir = Path(formal._output(args.output_dir, label="metric output directory"))
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("metric output directory must be new")
    external_exec_path = _existing(args.external_execution_authority, label="external execution authority")
    external_exec_record = file_record(external_exec_path)
    external = validate_external_authority(external_exec_path, expected_sha256=external_exec_record["sha256"])
    cache = _existing(args.external_cache, label="external cache")
    report = _existing(args.external_report, label="external report")
    if file_record(cache)["path"] != external["output_cache"] or file_record(report)["path"] != external["output_report"] or external["source_result"] != gate["source_result"]:
        raise ValueError("metric external-cache chain differs")
    inputs = {
        "config": file_record(_existing(args.config, label="frozen config")),
        "renderer_geometry_checkpoint": file_record(_existing(args.renderer_geometry_checkpoint, label="renderer checkpoint")),
        "summary_head": file_record(_existing(args.summary_head, label="summary head")),
        "all_query_text_cache": file_record(_existing(args.all_query_text_cache, label="all-query text cache")),
        "canonical_negative_text_cache": file_record(_existing(args.canonical_negative_text_cache, label="canonical-negative text cache")),
    }
    label_root = Path(args.label_root).expanduser().resolve()
    if str(label_root) != args.label_root or not label_root.is_dir() or label_root.is_symlink():
        raise ValueError("metric label root must be an existing canonical directory")
    authority = {
        "schema": formal.METRIC_EXECUTION_SCHEMA, "schema_version": 1,
        "status": METRIC_STATUS, "source_variant": args.source_variant,
        "source_result": dict(gate["source_result"]), "scene_id": external["scene_id"],
        "implementation": file_record(IMPLEMENTATION), "frozen_evaluator": file_record(Path(frozen_evaluator.__file__).resolve()),
        "preregistration": file_record(PREREGISTRATION), "external_execution_authority": external_exec_record,
        "external_cache": file_record(cache), "external_report": file_record(report), "frozen_inputs": inputs,
        "label_root": str(label_root), "output_dir": str(output_dir),
        "protocol": {
            "protocol_preset": "vala_paper_3d", "selection_mode": "score_threshold",
            "score_threshold": 0.6, "score_postprocess": "none", "projection_mode": "selected_only_alpha",
            "silhouette_threshold": 10.0 / 255.0, "alpha_binarization": "png_uint8_gt10",
            "mask_refinement": "none", "official_frames_only": True,
        },
        "single_candidate_no_sweep": True, "target_metric_authorized": True,
    }
    written = write_frozen_json(authority_output, authority)
    return {"status": "rank256_one_shot_metric_authority_built", "authority": file_record(written)}


def validate_metric_authority(path: str | Path, *, expected_sha256: str) -> dict[str, Any]:
    raw, digest, source = load_json_object(path, expected_sha256=expected_sha256, label="rank-256 metric authority")
    required = {
        "schema", "schema_version", "status", "source_variant", "source_result", "scene_id",
        "implementation", "frozen_evaluator", "preregistration", "external_execution_authority",
        "external_cache", "external_report", "frozen_inputs", "label_root", "output_dir", "protocol",
        "single_candidate_no_sweep", "target_metric_authorized",
    }
    authority = dict(raw)
    if set(authority) != required:
        raise ValueError("rank-256 metric authority fields differ")
    source_record = formal._record(authority["source_result"], label="metric source result")
    gate = formal.validate_champion_source(authority["source_variant"], source_record["path"], expected_sha256=source_record["sha256"])
    expected_protocol = {"protocol_preset": "vala_paper_3d", "selection_mode": "score_threshold", "score_threshold": 0.6, "score_postprocess": "none", "projection_mode": "selected_only_alpha", "silhouette_threshold": 10.0 / 255.0, "alpha_binarization": "png_uint8_gt10", "mask_refinement": "none", "official_frames_only": True}
    if (
        authority["schema"] != formal.METRIC_EXECUTION_SCHEMA or authority["schema_version"] != 1
        or authority["status"] != METRIC_STATUS or source_record != gate["source_result"]
        or authority["protocol"] != expected_protocol or authority["single_candidate_no_sweep"] is not True
        or authority["target_metric_authorized"] is not True
    ):
        raise ValueError("rank-256 metric authority header differs")
    if validate_file_record(authority["implementation"], label="metric implementation") != IMPLEMENTATION or validate_file_record(authority["frozen_evaluator"], label="frozen evaluator") != Path(frozen_evaluator.__file__).resolve() or validate_file_record(authority["preregistration"], label="metric preregistration") != PREREGISTRATION:
        raise ValueError("rank-256 metric implementation/evaluator differs")
    external_record = formal._record(authority["external_execution_authority"], label="metric external authority")
    external = validate_external_authority(external_record["path"], expected_sha256=external_record["sha256"])
    records = {name: formal._record(authority[name], label=f"metric {name}") for name in ("external_cache", "external_report")}
    inputs = authority["frozen_inputs"]
    expected_inputs = {"config", "renderer_geometry_checkpoint", "summary_head", "all_query_text_cache", "canonical_negative_text_cache"}
    if not isinstance(inputs, Mapping) or set(inputs) != expected_inputs:
        raise ValueError("rank-256 metric frozen inputs differ")
    inputs = {name: formal._record(inputs[name], label=f"metric input {name}") for name in sorted(expected_inputs)}
    for record in (*records.values(), *inputs.values()):
        validate_file_record(record, label="metric bound file")
    if records["external_cache"]["path"] != external["output_cache"] or records["external_report"]["path"] != external["output_report"] or external["source_result"] != source_record or inputs["renderer_geometry_checkpoint"] != external["renderer_geometry_checkpoint"]:
        raise ValueError("rank-256 metric nested binding differs")
    label_root = Path(authority["label_root"])
    output_dir = Path(formal._output(authority["output_dir"], label="metric output directory"))
    if not label_root.is_absolute() or not label_root.is_dir() or label_root.is_symlink():
        raise ValueError("rank-256 metric label root differs")
    authority.update(records)
    authority.update({"source_result": source_record, "external_execution_authority": external_record, "frozen_inputs": inputs, "output_dir": str(output_dir), "verified_source_gate": gate, "verified_external": external, "verified_record": {"path": str(source), "sha256": digest}})
    return authority


def run_metric(args: argparse.Namespace) -> dict[str, Any]:
    authority = validate_metric_authority(args.execution_authority, expected_sha256=args.expected_execution_authority_sha256)
    output = _new(authority["output_dir"], label="metric output directory")
    inputs = authority["frozen_inputs"]
    command = [
        sys.executable, authority["frozen_evaluator"]["path"],
        "--config", inputs["config"]["path"], "--checkpoint", inputs["renderer_geometry_checkpoint"]["path"],
        "--scene", authority["scene_id"], "--protocol_preset", "vala_paper_3d",
        "--label_dir", authority["label_root"], "--output_dir", str(output),
        "--summary_head_weights", inputs["summary_head"]["path"],
        "--text_embedding_cache", inputs["all_query_text_cache"]["path"],
        "--canonical_embedding_cache", inputs["canonical_negative_text_cache"]["path"],
        "--external_query_score_cache", authority["external_cache"]["path"], "--gpu", str(args.gpu),
    ]
    subprocess.run(command, check=True)
    result = output / authority["scene_id"] / "lerf_direct_3d_selection_results.json"
    if not result.is_file() or result.is_symlink():
        raise RuntimeError("frozen LERF evaluator did not produce its exact result")
    return {"status": "rank256_champion_one_shot_metric_complete", "result": file_record(result), "execution_authority": dict(authority["verified_record"])}


def _add_source(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-variant", choices=formal.SOURCE_VARIANTS, required=True)
    parser.add_argument("--source-result", required=True)
    parser.add_argument("--expected-source-result-sha256", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build-target-authority"); _add_source(p)
    for name in ("dataset_id", "scene_id", "geometry_checkpoint_sha256", "target_accepted_v2", "target_adaptive_typed_context", "factorized_primitive_state", "target_descriptor_output", "output_authority"):
        p.add_argument("--" + name.replace("_", "-"), required=True)
    p.set_defaults(func=build_target_authority)
    p = sub.add_parser("materialize-target")
    p.add_argument("--execution-authority", required=True); p.add_argument("--expected-execution-authority-sha256", required=True); p.add_argument("--output", required=True); p.add_argument("--batch-size", type=int, default=256); p.set_defaults(func=materialize_target)
    p = sub.add_parser("materialize-exact-query-subset"); _add_source(p)
    p.add_argument("--scene-id", required=True); p.add_argument("--query-manifest", required=True); p.add_argument("--expected-query-manifest-sha256", required=True); p.add_argument("--all-query-cache", required=True); p.add_argument("--expected-all-query-cache-sha256", required=True); p.add_argument("--output", required=True); p.add_argument("--output-receipt", required=True); p.set_defaults(func=materialize_exact_query_subset)
    p = sub.add_parser("build-query-authority"); _add_source(p)
    p.add_argument("--target-descriptor", required=True); p.add_argument("--positive-text-cache", required=True); p.add_argument("--positive-text-receipt", required=True); p.add_argument("--query-relevance-output", required=True); p.add_argument("--output-authority", required=True); p.set_defaults(func=build_query_authority)
    p = sub.add_parser("materialize-query")
    p.add_argument("--execution-authority", required=True); p.add_argument("--expected-execution-authority-sha256", required=True); p.add_argument("--output", required=True); p.set_defaults(func=materialize_query)
    p = sub.add_parser("build-external-authority"); _add_source(p)
    for name in ("query_execution_authority", "query_relevance", "comembership_feature_authority", "comembership_inference_authority", "renderer_geometry_checkpoint", "output_cache", "output_report", "output_authority"):
        p.add_argument("--" + name.replace("_", "-"), required=True)
    p.set_defaults(func=build_external_authority)
    p = sub.add_parser("materialize-external")
    p.add_argument("--execution-authority", required=True); p.add_argument("--expected-execution-authority-sha256", required=True); p.set_defaults(func=materialize_external)
    p = sub.add_parser("build-metric-authority"); _add_source(p)
    for name in ("external_execution_authority", "external_cache", "external_report", "config", "renderer_geometry_checkpoint", "summary_head", "all_query_text_cache", "canonical_negative_text_cache", "label_root", "output_dir", "output_authority"):
        p.add_argument("--" + name.replace("_", "-"), required=True)
    p.set_defaults(func=build_metric_authority)
    p = sub.add_parser("run-metric")
    p.add_argument("--execution-authority", required=True); p.add_argument("--expected-execution-authority-sha256", required=True); p.add_argument("--gpu", type=int, default=0); p.set_defaults(func=run_metric)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(args.func(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "build_external_authority", "build_metric_authority", "build_query_authority",
    "build_target_authority", "materialize_exact_query_subset", "materialize_external",
    "materialize_query", "materialize_target", "run_metric", "validate_external_authority",
    "validate_metric_authority", "validate_query_authority", "validate_target_authority",
]
