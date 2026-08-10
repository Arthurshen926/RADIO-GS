#!/usr/bin/env python3
"""Build and run the health-gated contrast-V2.1 exact LERF relevance chain.

The source/descriptor/health/preregistration gate is deliberately evaluated
before this module opens an exact query manifest or any text embedding cache.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.interfaces import factorized_native_contrast_v21_lerf_exact as formal
from radio_gs.losses import source_global_response_listwise_loss_v21 as relevance_loss
from radio_gs.losses.source_global_response_listwise_loss_v21 import (
    CANONICAL_NEGATIVE_MODEL,
    load_frozen_canonical_negative_bank,
)
from radio_gs.querying import unified_query
from radio_gs.querying import v21_absolute_relevance_adapter as relevance_adapter
from radio_gs.querying.v21_absolute_relevance_adapter import (
    calibrated_v21_absolute_relevance,
    load_v21_positive_text_bank,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


IMPLEMENTATION = Path(__file__).resolve()
AUTHORITY_STATUS = (
    "authorized_contrast_v21_exact_query_after_source_student_envelope_health_v4_pass"
)


def _new(value: object, *, label: str) -> Path:
    raw = str(value)
    path = Path(raw).expanduser().resolve()
    if raw != str(path):
        raise ValueError(f"{label} must be canonical absolute")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{label} already exists: {path}")
    return path


def _argument_record(path: object, digest: object, *, label: str) -> dict[str, str]:
    """Shape an expected record without opening the referenced file."""

    raw = str(path)
    canonical = str(Path(raw).expanduser().resolve())
    if raw != canonical:
        raise ValueError(f"{label} path must be canonical absolute")
    return formal.record(
        {"path": canonical, "sha256": str(digest)},
        label=label,
    )


def _load_frozen_all_query_cache(
    record: Mapping[str, str],
) -> tuple[list[str], torch.Tensor]:
    if dict(record) != formal.FROZEN_ALL_QUERY_CACHE:
        raise ValueError("contrast V2.1 all-query cache singleton differs")
    payload, _, _ = load_torch_mapping(
        record["path"],
        expected_sha256=record["sha256"],
        map_location="cpu",
        label="contrast V2.1 frozen all-query text cache",
    )
    required = {
        "queries",
        "prompt_templates",
        "text_encoder",
        "model_name",
        "embeddings",
    }
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
        raise ValueError("contrast V2.1 frozen all-query cache contract differs")
    norms = torch.linalg.vector_norm(embeddings, dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), rtol=0.0, atol=2e-4):
        raise ValueError("contrast V2.1 all-query embeddings are not unit L2")
    return list(queries), embeddings.detach().contiguous()


def _validate_exact_text_protocol(
    *,
    scene_id: str,
    manifest_record: Mapping[str, str],
    positive_record: Mapping[str, str],
) -> tuple[dict[str, Any], Any, Any]:
    """Open the four frozen query inputs. Call only after the prequery gate."""

    manifest_raw, manifest_digest, manifest_path = load_json_object(
        manifest_record["path"],
        expected_sha256=manifest_record["sha256"],
        label="contrast V2.1 exact scene query manifest",
    )
    if dict(manifest_record) != {
        "path": str(manifest_path),
        "sha256": manifest_digest,
    }:
        raise ValueError("contrast V2.1 exact manifest record differs")
    manifest = formal.validate_exact_query_manifest(manifest_raw, scene_id=scene_id)
    if (
        manifest["frozen_all_query_cache"] != formal.FROZEN_ALL_QUERY_CACHE
        or manifest["frozen_evaluator"] != formal.FROZEN_EVALUATOR
    ):
        raise ValueError("contrast V2.1 exact manifest protocol differs")
    validate_file_record(formal.FROZEN_EVALUATOR, label="frozen LERF evaluator")

    positive = load_v21_positive_text_bank(
        positive_record["path"],
        expected_file_sha256=positive_record["sha256"],
    )
    all_queries, all_embeddings = _load_frozen_all_query_cache(
        formal.FROZEN_ALL_QUERY_CACHE
    )
    positions = {query: index for index, query in enumerate(all_queries)}
    query_ids = list(manifest["query_ids"])
    if (
        list(positive.query_ids) != query_ids
        or any(query not in positions for query in query_ids)
    ):
        raise ValueError("contrast V2.1 positive cache query order differs")
    selected = all_embeddings[
        torch.tensor([positions[query] for query in query_ids], dtype=torch.long)
    ]
    if not torch.equal(positive.embeddings, selected):
        raise ValueError("contrast V2.1 positive cache is not the exact frozen subset")
    negative = load_frozen_canonical_negative_bank(
        formal.FROZEN_CANONICAL_NEGATIVE_BANK["path"],
        expected_file_sha256=formal.FROZEN_CANONICAL_NEGATIVE_BANK["sha256"],
    )
    return manifest, positive, negative


def _dependency_paths(gate: Mapping[str, Any]) -> dict[str, Path]:
    dispatch = gate["health_v4_dispatch"]
    return {
        "query_formal": Path(formal.__file__).resolve(),
        "target_descriptor_formal": Path(formal.target.__file__).resolve(),
        "health_v4_formal": Path(dispatch["formal_record"]["path"]),
        "health_v4_materializer": Path(dispatch["implementation_record"]["path"]),
        "absolute_relevance_adapter": Path(relevance_adapter.__file__).resolve(),
        "shared_exact_cosine_scorer": Path(unified_query.__file__).resolve(),
        "source_relevance_loss": Path(relevance_loss.__file__).resolve(),
    }


def _dependency_records(gate: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    return {
        name: file_record(path)
        for name, path in _dependency_paths(gate).items()
    }


def _validate_dependencies(
    value: object, *, gate: Mapping[str, Any]
) -> dict[str, dict[str, str]]:
    expected = _dependency_paths(gate)
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError("contrast V2.1 query dependencies differ")
    result: dict[str, dict[str, str]] = {}
    for name, expected_path in expected.items():
        verified = validate_file_record(
            value[name], label=f"contrast V2.1 query dependency {name}"
        )
        if verified != expected_path:
            raise ValueError(f"contrast V2.1 query dependency differs: {name}")
        result[name] = formal.record(value[name], label=f"query dependency {name}")
    return result


def build_authority(args: argparse.Namespace) -> dict[str, Any]:
    authority_output = _new(args.output_authority, label="query authority output")
    relevance_output = _new(args.query_relevance_output, label="query relevance output")
    source_record = _argument_record(
        args.source_result,
        args.expected_source_result_sha256,
        label="contrast V2.1 source result",
    )
    descriptor_record = _argument_record(
        args.target_descriptor,
        args.expected_target_descriptor_sha256,
        label="contrast V2.1 target descriptor",
    )
    health_record = _argument_record(
        args.health_v4_audit,
        args.expected_health_v4_audit_sha256,
        label="contrast V2.1 health-v4 audit",
    )

    # Hard ordering invariant: this is the first operation that opens any of
    # the supplied lineage files, and it does not receive query/text records.
    gate = formal.validate_prequery_gate(
        source_result_record=source_record,
        target_descriptor_record=descriptor_record,
        health_v4_audit_record=health_record,
    )

    manifest_record = _argument_record(
        args.exact_query_manifest,
        args.expected_exact_query_manifest_sha256,
        label="contrast V2.1 exact query manifest",
    )
    positive_record = _argument_record(
        args.positive_text_cache,
        args.expected_positive_text_cache_sha256,
        label="contrast V2.1 positive text cache",
    )
    _validate_exact_text_protocol(
        scene_id=gate["descriptor"]["scene_id"],
        manifest_record=manifest_record,
        positive_record=positive_record,
    )
    authority = {
        "schema": formal.QUERY_EXECUTION_SCHEMA,
        "schema_version": formal.SCHEMA_VERSION,
        "status": AUTHORITY_STATUS,
        "scene_id": gate["descriptor"]["scene_id"],
        "physical_space_id": gate["descriptor"]["physical_space_id"],
        "implementation": file_record(IMPLEMENTATION),
        "implementation_dependencies": _dependency_records(gate),
        "source_result": source_record,
        "target_descriptor": descriptor_record,
        "health_v4_audit": health_record,
        "health_v4_preregistration": dict(formal.HEALTH_V4_PREREGISTRATION),
        "query_preregistration": dict(formal.QUERY_PREREGISTRATION),
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
    return {
        "status": "contrast_v21_exact_query_authority_built",
        "authority": file_record(authority_output),
    }


def validate_authority(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_output: str | Path | None = None,
) -> dict[str, Any]:
    raw, digest, source_path = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="contrast V2.1 exact query execution authority",
    )
    required = {
        "schema",
        "schema_version",
        "status",
        "scene_id",
        "physical_space_id",
        "implementation",
        "implementation_dependencies",
        "source_result",
        "target_descriptor",
        "health_v4_audit",
        "health_v4_preregistration",
        "query_preregistration",
        "exact_query_manifest",
        "positive_text_cache",
        "all_query_text_cache",
        "canonical_negative_bank",
        "query_relevance_output",
        "query_execution_authorized",
        "metric_execution_authorized",
        "access_audit",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("contrast V2.1 exact query authority fields differ")
    authority = dict(raw)
    if (
        authority["schema"] != formal.QUERY_EXECUTION_SCHEMA
        or authority["schema_version"] != formal.SCHEMA_VERSION
        or authority["status"] != AUTHORITY_STATUS
        or authority["query_execution_authorized"] is not True
        or authority["metric_execution_authorized"] is not False
        or authority["access_audit"] != formal.query_access_audit()
    ):
        raise ValueError("contrast V2.1 exact query authority header differs")

    # Shape only. No query/text record is opened before this full hard gate.
    source_record = formal.record(authority["source_result"], label="source result")
    descriptor_record = formal.record(
        authority["target_descriptor"], label="target descriptor"
    )
    health_record = formal.record(authority["health_v4_audit"], label="health-v4 audit")
    gate = formal.validate_prequery_gate(
        source_result_record=source_record,
        target_descriptor_record=descriptor_record,
        health_v4_audit_record=health_record,
    )

    if validate_file_record(authority["implementation"], label="query implementation") != IMPLEMENTATION:
        raise ValueError("contrast V2.1 exact query implementation differs")
    dependencies = _validate_dependencies(
        authority["implementation_dependencies"], gate=gate
    )
    health_prereg = formal.record(
        authority["health_v4_preregistration"], label="health-v4 preregistration"
    )
    query_prereg = formal.record(
        authority["query_preregistration"], label="query preregistration"
    )
    if (
        health_prereg != formal.HEALTH_V4_PREREGISTRATION
        or query_prereg != formal.QUERY_PREREGISTRATION
    ):
        raise ValueError("contrast V2.1 exact query preregistration lineage differs")

    manifest_record = formal.record(
        authority["exact_query_manifest"], label="exact query manifest"
    )
    positive_record = formal.record(
        authority["positive_text_cache"], label="positive text cache"
    )
    all_query_record = formal.record(
        authority["all_query_text_cache"], label="all-query text cache"
    )
    negative_record = formal.record(
        authority["canonical_negative_bank"], label="canonical negative bank"
    )
    if (
        all_query_record != formal.FROZEN_ALL_QUERY_CACHE
        or negative_record != formal.FROZEN_CANONICAL_NEGATIVE_BANK
    ):
        raise ValueError("contrast V2.1 frozen query singleton differs")
    manifest, positive, negative = _validate_exact_text_protocol(
        scene_id=gate["descriptor"]["scene_id"],
        manifest_record=manifest_record,
        positive_record=positive_record,
    )
    if (
        authority["scene_id"] != gate["descriptor"]["scene_id"]
        or authority["physical_space_id"] != gate["descriptor"]["physical_space_id"]
    ):
        raise ValueError("contrast V2.1 exact query target identity differs")
    output = formal.canonical_output(
        authority["query_relevance_output"], label="query relevance output"
    )
    if expected_output is not None and output != str(
        Path(expected_output).expanduser().resolve()
    ):
        raise ValueError("contrast V2.1 exact query relevance output differs")
    authority.update(
        {
            "implementation": formal.record(
                authority["implementation"], label="query implementation"
            ),
            "implementation_dependencies": dependencies,
            "source_result": source_record,
            "target_descriptor": descriptor_record,
            "health_v4_audit": health_record,
            "health_v4_preregistration": health_prereg,
            "query_preregistration": query_prereg,
            "exact_query_manifest": manifest_record,
            "positive_text_cache": positive_record,
            "all_query_text_cache": all_query_record,
            "canonical_negative_bank": negative_record,
            "query_relevance_output": output,
            "verified_prequery_gate": gate,
            "verified_manifest": manifest,
            "verified_positive": positive,
            "verified_negative": negative,
            "verified_record": {"path": str(source_path), "sha256": digest},
        }
    )
    return authority


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = _new(args.output, label="query relevance output")
    execution = validate_authority(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
        expected_output=output,
    )
    descriptor = execution["verified_prequery_gate"]["descriptor_view"]
    relevance = calibrated_v21_absolute_relevance(
        descriptor["semantic_descriptor"],
        positive_bank=execution["verified_positive"],
        canonical_negative_bank=execution["verified_negative"],
    ).detach().to(device="cpu", dtype=torch.float32).contiguous()
    payload = {
        "schema": formal.QUERY_RELEVANCE_SCHEMA,
        "schema_version": formal.SCHEMA_VERSION,
        "contract": formal.query_contract(),
        "contract_sha256": formal.QUERY_CONTRACT_SHA256,
        "scene_id": descriptor["scene_id"],
        "physical_space_id": descriptor["physical_space_id"],
        "producer": file_record(IMPLEMENTATION),
        "query_execution_authority": dict(execution["verified_record"]),
        "input_authority": {
            "source_result": execution["source_result"],
            "target_descriptor": execution["target_descriptor"],
            "health_v4_audit": execution["health_v4_audit"],
            "health_v4_preregistration": execution["health_v4_preregistration"],
            "query_preregistration": execution["query_preregistration"],
            "exact_query_manifest": execution["exact_query_manifest"],
            "positive_text_cache": execution["positive_text_cache"],
            "all_query_text_cache": execution["all_query_text_cache"],
            "canonical_negative_bank": execution["canonical_negative_bank"],
        },
        "region_row_ids": list(descriptor["region_row_ids"]),
        "canonical_region_indices": descriptor[
            "canonical_region_indices"
        ].clone(),
        "region_fingerprints": list(descriptor["region_fingerprints"]),
        "query_ids": list(execution["verified_positive"].query_ids),
        "region_absolute_relevance": relevance,
        "access_audit": formal.query_access_audit(),
    }
    payload["channel_sha256"] = formal.query_channel_sha256(payload)
    payload = formal.validate_query_relevance(payload)
    write_torch_noclobber(output, payload)
    return {
        "status": "contrast_v21_exact_region_absolute_relevance_complete",
        "shape": list(relevance.shape),
        "output": file_record(output),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-authority")
    build.add_argument("--source-result", required=True)
    build.add_argument("--expected-source-result-sha256", required=True)
    build.add_argument("--target-descriptor", required=True)
    build.add_argument("--expected-target-descriptor-sha256", required=True)
    build.add_argument("--health-v4-audit", required=True)
    build.add_argument("--expected-health-v4-audit-sha256", required=True)
    build.add_argument("--exact-query-manifest", required=True)
    build.add_argument("--expected-exact-query-manifest-sha256", required=True)
    build.add_argument("--positive-text-cache", required=True)
    build.add_argument("--expected-positive-text-cache-sha256", required=True)
    build.add_argument("--query-relevance-output", required=True)
    build.add_argument("--output-authority", required=True)
    build.set_defaults(handler=build_authority)

    validate = subparsers.add_parser("validate-authority")
    validate.add_argument("--execution-authority", required=True)
    validate.add_argument("--expected-execution-authority-sha256", required=True)
    validate.add_argument("--expected-output")
    validate.set_defaults(
        handler=lambda args: {
            "status": "contrast_v21_exact_query_authority_valid",
            "authority": validate_authority(
                args.execution_authority,
                expected_sha256=args.expected_execution_authority_sha256,
                expected_output=args.expected_output,
            )["verified_record"],
        }
    )

    run = subparsers.add_parser("materialize")
    run.add_argument("--execution-authority", required=True)
    run.add_argument("--expected-execution-authority-sha256", required=True)
    run.add_argument("--output", required=True)
    run.set_defaults(handler=materialize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(args.handler(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "AUTHORITY_STATUS",
    "IMPLEMENTATION",
    "build_authority",
    "build_parser",
    "materialize",
    "validate_authority",
]
