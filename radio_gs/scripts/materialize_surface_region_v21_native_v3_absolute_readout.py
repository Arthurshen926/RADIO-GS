#!/usr/bin/env python3
"""Materialize query-opaque V2.1 absolute relevance with native-V3 completion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from radio_gs.interfaces.factorized_primitive_state import (
    load_factorized_primitive_state,
)
from radio_gs.interfaces import surface_region_v21_native_v3_absolute_readout as formal
from radio_gs.scripts.materialize_region_comembership_features_native_v3 import (
    validate_feature_authority,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_torch_mapping,
    write_torch_noclobber,
)


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"V2.1 native-V3 readout already exists: {output}")
    relevance_raw, relevance_sha, relevance_path = load_torch_mapping(
        args.absolute_relevance_authority,
        expected_sha256=args.expected_absolute_relevance_authority_sha256,
        map_location="cpu",
        label="V2.1 opaque absolute relevance authority",
    )
    feature_raw, feature_sha, feature_path = load_torch_mapping(
        args.native_v3_feature_authority,
        expected_sha256=args.expected_native_v3_feature_authority_sha256,
        map_location="cpu",
        label="native V3 feature authority",
    )
    inference_raw, inference_sha, inference_path = load_torch_mapping(
        args.native_v3_inference_authority,
        expected_sha256=args.expected_native_v3_inference_authority_sha256,
        map_location="cpu",
        label="native V3 inference authority",
    )
    relevance = formal.query_opaque_absolute_relevance_view(relevance_raw)
    # Validate the feature before following its primitive-state record.  The
    # adapter itself repeats the complete feature+inference canonical check.
    feature = validate_feature_authority(feature_raw)
    state_record = feature["input_authority"]["factorized_state"]
    state = load_factorized_primitive_state(
        state_record["path"], expected_sha256=state_record["sha256"]
    )
    readout = formal.apply_native_v3_absolute_readout(
        relevance=relevance,
        feature_authority=feature_raw,
        inference_authority=inference_raw,
        primitive_valid=state.valid,
    )
    selected_rule = dict(inference_raw["selected_rule"])
    payload = {
        "schema": formal.READOUT_SCHEMA,
        "schema_version": formal.READOUT_SCHEMA_VERSION,
        "contract": formal.readout_contract(),
        "contract_sha256": formal.READOUT_CONTRACT_SHA256,
        "scene_id": relevance.scene_id,
        "physical_space_id": relevance.physical_space_id,
        "producer": file_record(Path(__file__).resolve()),
        "input_authority": {
            "absolute_relevance": {
                "path": str(relevance_path),
                "sha256": relevance_sha,
            },
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
        "region_fingerprints_sha256": relevance.region_fingerprints_sha256,
        "query_axis_count": int(readout.absolute_relevance.shape[1]),
        "selected_rule": selected_rule,
        "applied_rule": {
            "path_method": formal.APPLIED_PATH_METHOD,
            "maximum_regions": formal.MAXIMUM_REGIONS,
            "relation_threshold": formal.RELATION_THRESHOLD,
            "absolute_boundary": formal.ABSOLUTE_BOUNDARY,
        },
        "canonical_region_indices": relevance.canonical_region_indices,
        "absolute_relevance": readout.absolute_relevance,
        "final_relevance": readout.final_relevance,
        "seed_region_indices": readout.seed_region_indices,
        "query_gate": readout.query_gate,
        "relation_selected_region_masks": (
            readout.relation_selected_region_masks
        ),
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
        "access_audit": formal.access_audit(),
    }
    payload["channel_sha256"] = formal.readout_channel_sha256(payload)
    formal.validate_readout_authority(payload)
    written = write_torch_noclobber(output, payload)
    return {
        "status": "v21_native_v3_query_opaque_absolute_readout_complete",
        "scene_id": payload["scene_id"],
        "opaque_query_axes": payload["query_axis_count"],
        "selected_rule": selected_rule,
        "applied_rule": payload["applied_rule"],
        "audit": payload["audit"],
        "output": file_record(written),
        "access_audit": formal.access_audit(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--absolute-relevance-authority", required=True)
    parser.add_argument(
        "--expected-absolute-relevance-authority-sha256", required=True
    )
    parser.add_argument("--native-v3-feature-authority", required=True)
    parser.add_argument(
        "--expected-native-v3-feature-authority-sha256", required=True
    )
    parser.add_argument("--native-v3-inference-authority", required=True)
    parser.add_argument(
        "--expected-native-v3-inference-authority-sha256", required=True
    )
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    print(
        json.dumps(
            materialize(build_parser().parse_args()), indent=2, allow_nan=False
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "materialize"]
