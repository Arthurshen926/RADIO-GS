#!/usr/bin/env python3
"""Validate the immutable canonical-mpr-v3 ScanNet totality freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from radio_gs.scripts import eval_ours_scannet_vala_gaussian_protocol as evaluator
from radio_gs.scripts import (
    materialize_ours_scannet_gaussian_semantic_score_cache as materializer,
)


ROOT = Path(__file__).resolve().parents[2]
V1_FREEZE = (
    ROOT
    / "paper/artifacts/scannet_canonical_mpr_v3_gaussian_semantic_totality_freeze_20260802.json"
)
V2_FREEZE = (
    ROOT
    / "paper/artifacts/scannet_canonical_mpr_v3_gaussian_semantic_totality_freeze_20260802_v2.json"
)
DEFAULT_FREEZE = V2_FREEZE
V1_EXPECTED_FREEZE_SHA256 = (
    "8d11a80c8f1b535a53f740c25d2437194e2a4e08a3892fbc3e15b39c647ffb61"
)
V2_EXPECTED_FREEZE_SHA256 = (
    "9a11b6d6db11ed338c76450d239adcc8d41ed25aca484b6b9ab5f3fd8674f7b6"
)
# Backward-compatible name for callers that mean the current/default authority.
EXPECTED_FREEZE_SHA256 = V2_EXPECTED_FREEZE_SHA256
KNOWN_FREEZE_SHA256 = {
    V1_EXPECTED_FREEZE_SHA256: 1,
    V2_EXPECTED_FREEZE_SHA256: 2,
}
CONTRACT = "radio_gs.canonical_mpr_v3_gaussian_semantic_totality.v1"
ARTIFACT_TYPE = (
    "radio_gs_scannet_canonical_mpr_v3_gaussian_semantic_totality_freeze"
)
EXPECTED_SCENES = [
    "scene0000_00",
    "scene0062_00",
    "scene0070_00",
    "scene0097_00",
    "scene0140_00",
    "scene0347_00",
    "scene0400_00",
    "scene0590_00",
]


class TotalityFreezeError(ValueError):
    """Raised when the totality freeze or one bound source has drifted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TotalityFreezeError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TotalityFreezeError(f"{label} must be a mapping")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TotalityFreezeError(f"freeze JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        encoded = path.read_bytes()
    except OSError as error:
        raise TotalityFreezeError(f"cannot read totality freeze: {path}") from error
    observed_sha = hashlib.sha256(encoded).hexdigest()
    _require(
        observed_sha in KNOWN_FREEZE_SHA256,
        "totality freeze SHA256 is not a known immutable version; create a new version instead of mutating an existing freeze",
    )
    try:
        payload = json.loads(
            encoded.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TotalityFreezeError("totality freeze is not canonical UTF-8 JSON") from error
    return _mapping(payload, "totality freeze")


def _validate_file_record(value: Any, *, label: str) -> Path:
    record = _mapping(value, label)
    _require(
        set(record) == {"path", "bytes", "sha256"},
        f"{label} file-record schema differs",
    )
    path = Path(str(record["path"]))
    _require(path.is_absolute(), f"{label} path must be absolute")
    _require(path.is_file() and not path.is_symlink(), f"{label} is missing or a symlink")
    _require(path.stat().st_size == record["bytes"], f"{label} byte size drifted")
    _require(_sha256(path) == record["sha256"], f"{label} SHA256 drifted")
    return path


def _validate_semantics(payload: Mapping[str, Any]) -> None:
    schema_version = payload.get("schema_version")
    _require(schema_version in {1, 2}, "freeze schema differs")
    _require(payload.get("artifact_type") == ARTIFACT_TYPE, "artifact type differs")
    _require(payload.get("contract") == CONTRACT, "totality contract differs")
    _require(
        payload.get("status") == "frozen_before_formal_evaluation",
        "freeze status differs",
    )
    if schema_version == 1:
        _require(
            "artifact_version" not in payload and "supersession" not in payload,
            "v1 artifact version/supersession schema differs",
        )
    else:
        _require(payload.get("artifact_version") == "v2", "artifact version differs")
        supersession = _mapping(payload.get("supersession"), "supersession")
        _require(
            dict(supersession)
            == {
                "supersedes_artifact": str(V1_FREEZE),
                "supersedes_sha256": V1_EXPECTED_FREEZE_SHA256,
                "scope": "pre_formal_evaluation_authority_replacement_only",
                "predecessor_formal_result_materialized": False,
                "reason": (
                    "Bind activated geometry row authority to the evaluator CPU "
                    "domain before any formal result while preserving the semantic "
                    "totality v1 contract."
                ),
            },
            "v2 pre-formal-evaluation supersession differs",
        )
    benchmark = _mapping(payload.get("benchmark_binding"), "benchmark binding")
    _require(
        benchmark.get("canonical_task_id") == evaluator.EXTERNAL_PROTOCOL_FREEZE_TASK
        and benchmark.get("registry_row") == evaluator.EXTERNAL_PROTOCOL_REGISTRY_ROW
        and benchmark.get("external_protocol_freeze_id")
        == evaluator.EXTERNAL_PROTOCOL_FREEZE_ID,
        "external protocol binding differs",
    )
    cohort = _mapping(benchmark.get("cohort"), "benchmark cohort")
    _require(
        cohort.get("scenes") == EXPECTED_SCENES and cohort.get("splits") == [19, 15, 10],
        "paper8 cohort differs",
    )
    method = _mapping(payload.get("method_binding"), "method binding")
    expected_method = {
        "method_family": evaluator.CURRENT_METHOD_FAMILY,
        "mainline_name": evaluator.CANONICAL_MAINLINE_NAME,
        "method_freeze_name": evaluator.CANONICAL_METHOD_FREEZE_NAME,
        "materializer_contract": evaluator.CURRENT_MATERIALIZER_CONTRACT,
        "semantic_score_cache_contract": evaluator.PROTOCOL_CONTRACT,
        "prediction_domain": evaluator.PREDICTION_DOMAIN,
        "row_order": evaluator.ROW_ORDER,
        "semantic_readout": evaluator.SEMANTIC_READOUT,
    }
    _require(dict(method) == expected_method, "method binding differs")

    totality = _mapping(payload.get("semantic_totality"), "semantic totality")
    geometry = totality.get("geometry_row_authority")
    if schema_version == 1:
        _require(geometry is None, "v1 geometry authority schema differs")
    else:
        _require(
            dict(_mapping(geometry, "geometry row authority"))
            == {
                "source": "canonical_method_geometry_checkpoint",
                "activation_device": "cpu",
                "activation_dtype": "float32",
                "activated_tensors": ["xyz", "scale", "quaternion", "opacity"],
                "tensor_layout": "contiguous",
                "row_order": evaluator.ROW_ORDER,
                "identity": (
                    "bitwise_activated_geometry_row_authority_for_materializer_cache_and_evaluator"
                ),
                "materializer_metadata": {
                    "geometry_authority_activation_device": "cpu"
                },
                "semantic_field_and_readout_remain_on_requested_scoring_device": True,
            },
            "v2 CPU bitwise activated geometry row authority differs",
        )
        _require(
            materializer._geometry_authority_device(
                evaluator.CURRENT_METHOD_FAMILY, torch.device("cuda:0")
            )
            == torch.device("cpu"),
            "producer no longer activates canonical geometry on CPU",
        )
    observed = _mapping(totality.get("graph_observed"), "observed-row semantics")
    fallback = _mapping(
        totality.get("no_graph_evidence"), "no-evidence-row semantics"
    )
    query = _mapping(totality.get("query_bank"), "query-bank semantics")
    _require(totality.get("valid_is_total") is True, "valid domain is not total")
    _require(
        observed.get("physical_radii_m") == evaluator.CANONICAL_REGION_RADII_M
        and observed.get("per_scale_score")
        == "independent_l2_normalized_cosine_to_exact_split19_query"
        and observed.get("scale_aggregation")
        == "per_class_max_over_three_independent_cosines",
        "observed h128 three-scale cosine-max semantics differ",
    )
    _require(
        fallback.get("descriptor")
        == "same_row_canonical_field_primitive_radio_then_frozen_official_c_radio_v4_siglip2_g_summary_head"
        and fallback.get("score")
        == "independent_l2_normalized_cosine_to_exact_split19_query"
        and fallback.get("spatial_fill") is False
        and fallback.get("neighbor_transfer") is False,
        "no-evidence primitive fallback semantics differ",
    )
    _require(
        query
        == {
            "model": materializer.SIGLIP2_MODEL_NAME,
            "queries": "exact_scannet_nyu40_split19_in_frozen_class_order",
            "shape": [19, 1536],
            "unit_normalized": True,
            "prompt_ensemble": False,
        },
        "query-bank semantics differ",
    )
    _require(
        totality.get("spatial_transfer") == evaluator.SPATIAL_TRANSFER == "none"
        and totality.get("mesh_vertices_used") is False
        and totality.get("knn_used") is False
        and totality.get("query_set_invariant") is True
        and totality.get("query_set_calibration") is False
        and totality.get("logit_calibration") == "none"
        and totality.get("logit_smoothing") == "none",
        "calibration-free/no-spatial-transfer contract differs",
    )


def validate_freeze(path: str | Path = DEFAULT_FREEZE) -> dict[str, Any]:
    freeze_path = Path(path).absolute()
    payload = _load_json(freeze_path)
    freeze_sha256 = _sha256(freeze_path)
    schema_version = KNOWN_FREEZE_SHA256[freeze_sha256]
    _require(
        payload.get("schema_version") == schema_version,
        "freeze SHA256/version binding differs",
    )
    _validate_semantics(payload)
    sources = _mapping(payload.get("immutable_sources"), "immutable sources")
    _require(
        set(sources)
        == {
            "external_protocol_freeze",
            "canonical_method_freeze",
            "canonical_mainline",
            "surface_region_readout",
            "official_c_radio_checkpoint",
            "exact_split19_query_bank",
            "producer_source",
            "evaluator_source",
        },
        "immutable source roles differ",
    )
    paths = {
        role: _validate_file_record(record, label=role)
        for role, record in sources.items()
    }
    expected_hashes = {
        "external_protocol_freeze": evaluator.EXTERNAL_PROTOCOL_FREEZE_SHA256,
        "canonical_method_freeze": evaluator.CANONICAL_METHOD_FREEZE_SHA256,
        "canonical_mainline": evaluator.CANONICAL_MAINLINE_SHA256,
        "surface_region_readout": evaluator.CANONICAL_READOUT_SHA256,
        "official_c_radio_checkpoint": evaluator.OFFICIAL_RADIO_SHA256,
    }
    for role, expected in expected_hashes.items():
        _require(sources[role]["sha256"] == expected, f"{role} binding differs")

    external = yaml.safe_load(paths["external_protocol_freeze"].read_text("utf-8"))
    parent = yaml.safe_load(paths["canonical_method_freeze"].read_text("utf-8"))
    mainline = yaml.safe_load(paths["canonical_mainline"].read_text("utf-8"))
    _require(
        external.get("freeze_id") == evaluator.EXTERNAL_PROTOCOL_FREEZE_ID
        and evaluator.EXTERNAL_PROTOCOL_FREEZE_TASK in external.get("canonical_tasks", {}),
        "external protocol freeze payload differs",
    )
    _require(
        parent.get("name") == evaluator.CANONICAL_METHOD_FREEZE_NAME
        and parent.get("method", {}).get("mainline_manifest_sha256")
        == evaluator.CANONICAL_MAINLINE_SHA256,
        "canonical method freeze payload differs",
    )
    _require(
        mainline.get("name") == evaluator.CANONICAL_MAINLINE_NAME
        and mainline.get("status") == "promoted_mainline",
        "canonical mainline payload differs",
    )

    readout = torch.load(
        paths["surface_region_readout"], map_location="cpu", weights_only=False
    )
    architecture = _mapping(readout.get("architecture"), "readout architecture")
    provenance = _mapping(readout.get("provenance"), "readout provenance")
    _require(
        architecture
        == {
            "name": "surface_region_summary_readout_v1",
            "feature_dim": 1280,
            "geometry_dim": 12,
            "hidden_dim": 128,
            "digest": "2ea0107b914ed2e4498893e75475f5597f55b6a1a2d6bdf7bcc5aefab2545a88",
        },
        "frozen h128 readout architecture differs",
    )
    _require(
        provenance.get("frozen") is True
        and provenance.get("uses_benchmark_scenes") is False
        and provenance.get("uses_benchmark_test_vocabulary") is False
        and provenance.get("scene_disjoint") is True
        and provenance.get("official_summary_head") == "c-radio_v4 siglip2-g"
        and provenance.get("custom_text_projection") is False,
        "frozen h128 readout provenance differs",
    )
    query, query_sha, query_path = materializer.load_frozen_split19_text_bank(
        paths["exact_split19_query_bank"], device=torch.device("cpu")
    )
    _require(
        query_sha == sources["exact_split19_query_bank"]["sha256"]
        and query_path == paths["exact_split19_query_bank"]
        and tuple(query.shape) == (19, 1536),
        "exact split19 query-bank identity differs",
    )
    _require(
        sources["producer_source"]["sha256"]
        == _sha256(Path(materializer.__file__).resolve())
        and sources["evaluator_source"]["sha256"]
        == _sha256(Path(evaluator.__file__).resolve()),
        "producer/evaluator source binding differs",
    )
    return {
        "status": "validated",
        "contract": CONTRACT,
        "schema_version": schema_version,
        "artifact_version": payload.get("artifact_version", "v1"),
        "freeze": str(freeze_path),
        "freeze_sha256": freeze_sha256,
        "immutable_source_count": len(sources),
        "paper8_scene_count": len(EXPECTED_SCENES),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    args = parser.parse_args(argv)
    print(json.dumps(validate_freeze(args.freeze), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
