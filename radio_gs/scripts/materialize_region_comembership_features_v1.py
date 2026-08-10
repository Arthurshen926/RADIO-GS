#!/usr/bin/env python3
"""Materialize query-independent RegionCoMembershipV1 pair features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces.factorized_primitive_state import (
    load_factorized_primitive_state,
)
from radio_gs.interfaces.surface_region_typed_context_adaptive import (
    validate_adaptive_typed_context_authority,
)
from radio_gs.models.region_comembership_v1 import PAIR_FEATURE_NAMES
from radio_gs.scripts import materialize_full_scalar_clean_training_shard as shard
from radio_gs.scripts.build_source_region_comembership_v1 import (
    TRAIN_SCENES,
    VALIDATION_SCENES,
    build_query_independent_pair_features,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
    write_torch_noclobber,
)


SCHEMA = "radio_gs.region_comembership_feature_authority.v1"
TARGET_EXECUTION_SCHEMA = (
    "radio_gs.region_comembership_target_feature_execution_authority.v1"
)
TARGET_GATE = Path(
    "paper/artifacts/source_only_region_comembership_v1_target_execution_gate_20260807.json"
)


def _target_execution_authorized(
    path: str | Path, *, expected_sha256: str, scene_id: str
) -> dict[str, Any]:
    value, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="RegionCoMembership target feature execution authority",
    )
    required = {
        "schema",
        "schema_version",
        "status",
        "scene_id",
        "target_gate",
        "four_plus_two_result",
        "target_feature_materialization_authorized",
        "target_checkpoint_inference_authorized",
        "target_metric_authorized",
    }
    if (
        set(value) != required
        or value.get("schema") != TARGET_EXECUTION_SCHEMA
        or value.get("schema_version") != 1
        or value.get("status") != "authorized_after_topology_4plus2_promotion"
        or value.get("scene_id") != scene_id
        or value.get("target_feature_materialization_authorized") is not True
        or value.get("target_checkpoint_inference_authorized") is not True
        or value.get("target_metric_authorized") is not False
    ):
        raise ValueError("target feature execution authority differs")
    target_gate = validate_file_record(value["target_gate"], label="target gate")
    expected_gate = Path(__file__).resolve().parents[2] / TARGET_GATE
    if target_gate != expected_gate:
        raise ValueError("target feature authority binds another target gate")
    result_record = value["four_plus_two_result"]
    result_path = validate_file_record(result_record, label="4+2 pilot result")
    result, _, _ = load_json_object(
        result_path,
        expected_sha256=result_record["sha256"],
        label="4+2 pilot result",
    )
    promotion = result.get("promotion_gate")
    if (
        result.get("schema") != "radio_gs.region_comembership_v1_pilot_result.v1"
        or result.get("status") != "source_only_4train_2validation_pilot_complete"
        or result.get("target_execution_performed") is not False
        or result.get("automatic_epoch_zero_fallback") is not False
        or int(result.get("selected_epoch", -1)) <= 0
        or not isinstance(promotion, Mapping)
        or promotion.get("passed") is not True
        or promotion.get("selected_epoch_positive") is not True
        or promotion.get("selected_topology_score_strictly_exceeds_epoch_zero")
        is not True
        or float(promotion.get("selected_topology_score", float("-inf")))
        <= float(promotion.get("epoch_zero_topology_score", float("inf")))
    ):
        raise ValueError("4+2 pilot did not pass the source promotion gate")
    validate_file_record(result["checkpoint"], label="promoted 4+2 checkpoint")
    return {"path": str(source), "sha256": digest}


def _source_access(domain: str) -> dict[str, bool]:
    return {
        "source_instance_labels_opened": False,
        "benchmark_images_opened": False,
        "benchmark_labels_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_queries_opened": False,
        "target_feature_authorities_opened": domain == "target",
        "target_metrics_computed": False,
    }


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"feature-only authority already exists: {output}")
    scene_id = str(args.scene_id)
    domain = str(args.domain)
    execution = None
    if domain == "source_parity":
        if scene_id not in TRAIN_SCENES | VALIDATION_SCENES:
            raise ValueError("source parity scene is outside the fixed cohort")
        if args.execution_authority or args.expected_execution_authority_sha256:
            raise ValueError("source parity must not supply target execution authority")
    elif domain == "target":
        if not args.execution_authority or not args.expected_execution_authority_sha256:
            raise ValueError(
                "target materialization requires frozen execution authority"
            )
        execution = _target_execution_authorized(
            args.execution_authority,
            expected_sha256=args.expected_execution_authority_sha256,
            scene_id=scene_id,
        )
    else:
        raise ValueError("feature-only authority domain differs")

    accepted_raw, accepted_sha, accepted_path = load_torch_mapping(
        args.accepted_v2,
        expected_sha256=args.expected_accepted_v2_sha256,
        map_location="cpu",
        label="feature-only AcceptedV2 authority",
    )
    accepted = shard.validate_accepted_region_authority(accepted_raw)
    context_raw, context_sha, context_path = load_torch_mapping(
        args.typed_context,
        expected_sha256=args.expected_typed_context_sha256,
        map_location="cpu",
        label="feature-only typed-context authority",
    )
    context = validate_adaptive_typed_context_authority(context_raw)
    graph, graph_sha, graph_path = load_torch_mapping(
        args.support_graph,
        expected_sha256=args.expected_support_graph_sha256,
        map_location="cpu",
        label="feature-only support graph",
    )
    state = load_factorized_primitive_state(
        args.factorized_state,
        expected_sha256=args.expected_factorized_state_sha256,
    )
    if (
        accepted["scene_id"] != scene_id
        or context["scene_id"] != scene_id
        or context["region_row_ids"]
        != [
            f"{scene_id}:accepted-v2-canonical-v1:{value}"
            for value in accepted["region_fingerprints"]
        ]
        or not torch.equal(
            accepted["canonical_region_indices"],
            context["canonical_region_indices"],
        )
        or not torch.equal(
            torch.as_tensor(graph["global_rows"]).long(), state.global_rows
        )
        or not torch.equal(
            torch.as_tensor(graph["xyz"]).float(), state.xyz[state.global_rows]
        )
        or accepted["input_authority"]["support_graph_authority"][
            "support_graph_file_sha256"
        ]
        != graph_sha
        or accepted["input_authority"]["geometry_authority"][
            "factorized_primitive_state_file_sha256"
        ]
        != state.sha256
    ):
        raise ValueError("feature-only scene/region/geometry authorities differ")
    pairs, features, audit = build_query_independent_pair_features(
        accepted=accepted,
        context=context,
        state=state,
        graph=graph,
    )
    identity = {
        "schema": SCHEMA,
        "schema_version": 1,
        "scene_id": scene_id,
        "domain": domain,
        "producer": file_record(Path(__file__).resolve()),
        "target_execution_authority": execution,
        "input_authority": {
            "accepted_v2": {"path": str(accepted_path), "sha256": accepted_sha},
            "typed_context": {"path": str(context_path), "sha256": context_sha},
            "support_graph": {"path": str(graph_path), "sha256": graph_sha},
            "factorized_state": {
                "path": str(state.source),
                "sha256": state.sha256,
            },
        },
        "candidate_policy": {
            "descriptor_neighbors": 16,
            "centroid_neighbors": 16,
            "anchor_support_edges": True,
        },
        "feature_names": list(PAIR_FEATURE_NAMES),
        "feature_names_sha256": canonical_json_sha256(list(PAIR_FEATURE_NAMES)),
        "source_access": _source_access(domain),
    }
    payload = {
        **identity,
        "content_authority_sha256": canonical_json_sha256(identity),
        "region_fingerprints": list(accepted["region_fingerprints"]),
        "canonical_region_indices": accepted["canonical_region_indices"],
        "region_rows": accepted["region_rows"],
        "token_mask": accepted["token_mask"],
        "pair_indices": pairs,
        "pair_features": features,
        "channel_sha256": {},
        "audit": {
            "canonical_regions": int(accepted["region_rows"].shape[0]),
            **audit,
        },
    }
    payload["channel_sha256"] = {
        name: tensor_sha256(payload[name])
        for name in (
            "canonical_region_indices",
            "region_rows",
            "token_mask",
            "pair_indices",
            "pair_features",
        )
    }
    written = write_torch_noclobber(output, payload)
    return {
        "status": "region_comembership_feature_authority_complete",
        "scene_id": scene_id,
        "domain": domain,
        "output": file_record(written),
        "audit": payload["audit"],
        "target_metric_computed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--domain", choices=("source_parity", "target"), required=True)
    for name in ("accepted-v2", "typed-context", "support-graph", "factorized-state"):
        parser.add_argument(f"--{name}", required=True)
        parser.add_argument(f"--expected-{name}-sha256", required=True)
    parser.add_argument("--execution-authority")
    parser.add_argument("--expected-execution-authority-sha256")
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    print(json.dumps(materialize(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
