#!/usr/bin/env python3
"""Materialize the diagnostic contrast-exact to native-V3 readout bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from radio_gs.interfaces import (
    factorized_native_contrast_v21_exact_native_v3_bridge as bridge,
)
from radio_gs.interfaces import (
    surface_region_v21_native_v3_absolute_readout as readout_formal,
)
from radio_gs.interfaces.factorized_primitive_state import (
    load_factorized_primitive_state,
)
from radio_gs.scripts import (
    materialize_factorized_native_contrast_v21_lerf_exact_relevance as exact_script,
)
from radio_gs.scripts.materialize_region_comembership_features_native_v3 import (
    validate_feature_authority,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_torch_mapping,
    write_torch_noclobber,
)


IMPLEMENTATION = Path(__file__).resolve()


def _new(value: object) -> Path:
    raw = str(value)
    path = Path(raw).expanduser().resolve()
    if raw != str(path):
        raise ValueError("diagnostic native-V3 bridge output must be canonical absolute")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"diagnostic native-V3 bridge output exists: {path}")
    return path


def _readout_payload(
    *,
    relevance,
    readout,
    selected_rule: dict[str, Any],
    input_authority: dict[str, dict[str, str]],
) -> dict[str, Any]:
    payload = {
        "schema": readout_formal.READOUT_SCHEMA,
        "schema_version": readout_formal.READOUT_SCHEMA_VERSION,
        "contract": readout_formal.readout_contract(),
        "contract_sha256": readout_formal.READOUT_CONTRACT_SHA256,
        "scene_id": relevance.scene_id,
        "physical_space_id": relevance.physical_space_id,
        "producer": file_record(IMPLEMENTATION),
        "input_authority": input_authority,
        "region_fingerprints_sha256": relevance.region_fingerprints_sha256,
        "query_axis_count": int(readout.absolute_relevance.shape[1]),
        "selected_rule": selected_rule,
        "applied_rule": {
            "path_method": readout_formal.APPLIED_PATH_METHOD,
            "maximum_regions": readout_formal.MAXIMUM_REGIONS,
            "relation_threshold": readout_formal.RELATION_THRESHOLD,
            "absolute_boundary": readout_formal.ABSOLUTE_BOUNDARY,
        },
        "canonical_region_indices": relevance.canonical_region_indices,
        "absolute_relevance": readout.absolute_relevance,
        "final_relevance": readout.final_relevance,
        "seed_region_indices": readout.seed_region_indices,
        "query_gate": readout.query_gate,
        "relation_selected_region_masks": readout.relation_selected_region_masks,
        "relation_path_support": readout.relation_path_support,
        "primitive_valid": readout.primitive_valid,
        "primitive_membership": readout.primitive_membership,
        "union_selected_region_indices": readout.union_selected_region_indices,
        "union_selected_region_scores": readout.union_selected_region_scores,
        "union_selected_marginal_core_rows": (
            readout.union_selected_marginal_core_rows
        ),
        "audit": {
            "opaque_query_axes": int(readout.absolute_relevance.shape[1]),
            "query_gate_passed": int(readout.query_gate.sum()),
            "query_gate_failed_exact_unary": int((~readout.query_gate).sum()),
            "maximum_relation_support_regions": int(
                readout.relation_selected_region_masks.sum(dim=0).max()
            ),
            "maximum_union_regions": max(
                len(rows) for rows in readout.union_selected_region_indices
            ),
            "unary_decreases": int(
                (readout.final_relevance < readout.absolute_relevance).sum()
            ),
            "seed_changes": int(
                (
                    readout.final_relevance[
                        readout.seed_region_indices,
                        range(readout.absolute_relevance.shape[1]),
                    ]
                    != readout.absolute_relevance[
                        readout.seed_region_indices,
                        range(readout.absolute_relevance.shape[1]),
                    ]
                ).sum()
            ),
            "fallback_pair_count": readout.fallback_pair_count,
            "fallback_pairs_above_relation_threshold": (
                readout.fallback_pairs_above_relation_threshold
            ),
            "invalid_primitive_memberships_removed": (
                readout.invalid_primitive_memberships_removed
            ),
            "query_identifiers_consumed": False,
            "query_strings_consumed": False,
            "target_metric_computed": False,
        },
        "channel_sha256": {},
        "access_audit": readout_formal.access_audit(),
    }
    payload["channel_sha256"] = readout_formal.readout_channel_sha256(payload)
    return readout_formal.validate_readout_authority(payload)


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = _new(args.output)
    relevance_raw, relevance_sha, relevance_path = load_torch_mapping(
        args.exact_relevance,
        expected_sha256=args.expected_exact_relevance_sha256,
        map_location="cpu",
        label="contrast V2.1 exact relevance bridge input",
    )
    relevance_record = {"path": str(relevance_path), "sha256": relevance_sha}
    dispatched = bridge.dispatch_relevance_schema(relevance_raw)
    execution = exact_script.validate_authority(
        dispatched.payload["query_execution_authority"]["path"],
        expected_sha256=dispatched.payload["query_execution_authority"]["sha256"],
        expected_output=relevance_record["path"],
    )
    validated = bridge.validate_contrast_exact_lineage(
        dispatched=dispatched,
        relevance_record=relevance_record,
        query_execution=execution,
    )
    relevance = bridge.query_opaque_view(validated)

    # Native-V3 files are not opened until the complete exact-query lineage,
    # including health-V4 PASS, has returned successfully.
    feature_raw, feature_sha, feature_path = load_torch_mapping(
        args.native_v3_feature_authority,
        expected_sha256=args.expected_native_v3_feature_authority_sha256,
        map_location="cpu",
        label="contrast exact bridge native V3 feature authority",
    )
    inference_raw, inference_sha, inference_path = load_torch_mapping(
        args.native_v3_inference_authority,
        expected_sha256=args.expected_native_v3_inference_authority_sha256,
        map_location="cpu",
        label="contrast exact bridge native V3 inference authority",
    )
    feature = validate_feature_authority(feature_raw)
    state_record = feature["input_authority"]["factorized_state"]
    state = load_factorized_primitive_state(
        state_record["path"], expected_sha256=state_record["sha256"]
    )
    readout = readout_formal.apply_native_v3_absolute_readout(
        relevance=relevance,
        feature_authority=feature_raw,
        inference_authority=inference_raw,
        primitive_valid=state.valid,
    )
    payload = _readout_payload(
        relevance=relevance,
        readout=readout,
        selected_rule=dict(inference_raw["selected_rule"]),
        input_authority={
            "absolute_relevance": relevance_record,
            "native_v3_feature": {
                "path": str(feature_path),
                "sha256": feature_sha,
            },
            "native_v3_inference": {
                "path": str(inference_path),
                "sha256": inference_sha,
            },
            "factorized_primitive_state": dict(state_record),
        },
    )
    written = write_torch_noclobber(output, payload)
    return {
        "status": "contrast_v21_exact_native_v3_diagnostic_readout_complete",
        "candidate_status": "diagnostic_only_not_final_candidate",
        "input_schema_dispatch": dispatched.dispatch_name,
        "scene_id": payload["scene_id"],
        "opaque_query_axes": payload["query_axis_count"],
        "audit": payload["audit"],
        "bridge_contract": bridge.bridge_contract(),
        "bridge_contract_sha256": bridge.BRIDGE_CONTRACT_SHA256,
        "bridge_access_audit": bridge.bridge_access_audit(),
        "output": file_record(written),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-relevance", required=True)
    parser.add_argument("--expected-exact-relevance-sha256", required=True)
    parser.add_argument("--native-v3-feature-authority", required=True)
    parser.add_argument("--expected-native-v3-feature-authority-sha256", required=True)
    parser.add_argument("--native-v3-inference-authority", required=True)
    parser.add_argument("--expected-native-v3-inference-authority-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    print(json.dumps(materialize(build_parser().parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = ["IMPLEMENTATION", "build_parser", "materialize"]
