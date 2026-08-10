#!/usr/bin/env python3
"""Build a frozen-evaluator LERF cache from calibrated V2.1 region relevance.

Unlike the legacy V2 adapter, this bridge never re-enters O0 primitive score
caches and never applies KNN or scene-wise min-max.  V2.1 absolute relevance
owns the semantic boundary, while greedy novelty selects complementary
query-independent region supports.  Promoted source-only co-membership remains
bound for audit but cannot collapse deployment to graph-isolated singletons.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any

import torch

from radio_gs.interfaces.factorized_primitive_state import (
    load_factorized_primitive_state,
)
from radio_gs.interfaces.surface_region_target_accepted_v2 import (
    validate_target_accepted_v2_authority,
)
from radio_gs.interfaces import surface_region_v21_query_relevance as query_formal
from radio_gs.interfaces.surface_region_v21_query_relevance import (
    validate_query_execution_authority,
    validate_query_relevance_authority,
)
from radio_gs.interfaces.surface_region_v21_source_gate import (
    validate_source_pilot_chain,
)
from radio_gs.querying import multi_region_union_readout as multi_union
from radio_gs.querying import (
    semantic_conditioned_region_comembership_readout as semantic_audit,
)
from radio_gs.scripts import (
    build_lerf_region_comembership_external_cache_v2 as v2_cache,
)
from radio_gs.scripts import (
    materialize_lerf_multiscale_query_score_cache as score_cache,
)
from radio_gs.scripts.infer_region_comembership_v2 import (
    validate_inference_authority,
)
from radio_gs.scripts.materialize_region_comembership_features_v2 import (
    validate_feature_authority,
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


SCHEMA = "radio_gs.lerf_region_comembership_external_scores.v21"
EXECUTION_AUTHORITY_SCHEMA = (
    "radio_gs.lerf_v21_calibrated_region_readout_execution_authority.v1"
)
EXECUTION_AUTHORITY_STATUS = (
    "authorized_after_v21_source_promotion_for_frozen_lerf_query_readout"
)
IMPLEMENTATION_DEPENDENCIES = {
    "greedy_novelty_union_readout": Path(multi_union.__file__).resolve(),
    "semantic_conditioned_graph_audit": Path(semantic_audit.__file__).resolve(),
    "v2_formal_binding": Path(v2_cache.__file__).resolve(),
    "renderer_geometry_parser": Path(score_cache.__file__).resolve(),
    "query_relevance_authority": Path(query_formal.__file__).resolve(),
}
PREREGISTRATION = (
    Path(__file__).resolve().parents[2]
    / "paper/artifacts/lerf_v21_absolute_relevance_greedy_novelty_union_preregistration_20260807.json"
)
V21_ABSOLUTE_RELEVANCE_BOUNDARY = 0.5
V21_MAXIMUM_REGIONS = 8
V21_CANDIDATE_CHUNK_ROWS = 4096
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def external_cache_access_audit() -> dict[str, bool]:
    return {
        "source_promotion_validated_before_target_or_query_files": True,
        "target_descriptor_opened": True,
        "benchmark_queries_opened": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "target_metrics_computed": False,
        "legacy_o0_query_scores_opened": False,
    }


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    path = str(value["path"])
    digest = str(value["sha256"])
    if not Path(path).is_absolute() or _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} file record differs")
    return {"path": path, "sha256": digest}


def _canonical_output(value: object, *, label: str) -> str:
    raw = str(value)
    resolved = str(Path(raw).expanduser().resolve())
    if raw != resolved:
        raise ValueError(f"{label} must be absolute and canonical")
    return resolved


def greedy_novelty_readout_from_v21(
    *,
    feature: Mapping[str, Any],
    inference: Mapping[str, Any],
    relevance: Mapping[str, Any],
    num_primitives: int,
):
    """Select complementary semantic-positive regions without a graph hard gate."""

    rule = inference.get("selected_rule")
    if (
        not isinstance(rule, Mapping)
        or set(rule) != {"method", "maximum_regions", "threshold"}
        or relevance.get("scene_id") != feature.get("scene_id")
        or relevance.get("region_fingerprints") != feature.get("region_fingerprints")
        or not torch.equal(
            torch.as_tensor(relevance.get("canonical_region_indices")),
            torch.as_tensor(feature.get("canonical_region_indices")),
        )
    ):
        raise ValueError("V2.1 relevance/co-membership canonical binding differs")
    return multi_union.greedy_novelty_union_readout(
        relevance["region_absolute_relevance"],
        region_rows=feature["region_rows"],
        core_mask=feature["token_mask"],
        num_primitives=int(num_primitives),
        config=multi_union.MultiRegionUnionConfig(
            score_threshold=V21_ABSOLUTE_RELEVANCE_BOUNDARY,
            maximum_regions=V21_MAXIMUM_REGIONS,
            candidate_chunk_rows=V21_CANDIDATE_CHUNK_ROWS,
        ),
    )


def mask_union_to_valid(
    membership: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, int]:
    values = torch.as_tensor(membership).detach().float().cpu().contiguous()
    mask = torch.as_tensor(valid).detach().bool().cpu().contiguous()
    if (
        values.ndim != 2
        or mask.shape != (values.shape[0],)
        or not bool(torch.isfinite(values).all())
        or bool(((values != 0.0) & (values != 1.0)).any())
    ):
        raise ValueError("V2.1 novelty union validity inputs differ")
    removed = int(values[~mask].count_nonzero())
    values[~mask] = 0.0
    return values.contiguous(), removed


def validate_external_execution_authority(
    path: str | Path, *, expected_sha256: str
) -> dict[str, Any]:
    raw, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="LERF V2.1 external-cache execution authority",
    )
    required = {
        "schema",
        "schema_version",
        "status",
        "scene_id",
        "physical_space_id",
        "source_pilot_result",
        "v21_checkpoint",
        "v21_normalization",
        "canonical_negative_bank",
        "target_descriptor",
        "positive_text_cache",
        "query_relevance_execution_authority",
        "query_relevance_authority",
        "comembership_feature_authority",
        "comembership_inference_authority",
        "renderer_geometry_checkpoint",
        "preregistration",
        "implementation",
        "implementation_dependencies",
        "output_cache",
        "output_report",
        "query_readout_authorized",
        "target_metric_authorized",
        "access_audit",
    }
    authority = dict(raw)
    if (
        set(authority) != required
        or authority.get("schema") != EXECUTION_AUTHORITY_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("status") != EXECUTION_AUTHORITY_STATUS
        or not isinstance(authority.get("scene_id"), str)
        or not authority["scene_id"]
        or not isinstance(authority.get("physical_space_id"), str)
        or not authority["physical_space_id"]
        or authority.get("query_readout_authorized") is not True
        or authority.get("target_metric_authorized") is not False
        or authority.get("access_audit") != external_cache_access_audit()
    ):
        raise ValueError("LERF V2.1 external execution header differs")
    records = {
        name: _record(authority[name], label=f"LERF V2.1 {name}")
        for name in (
            "source_pilot_result",
            "v21_checkpoint",
            "v21_normalization",
            "canonical_negative_bank",
            "target_descriptor",
            "positive_text_cache",
            "query_relevance_execution_authority",
            "query_relevance_authority",
            "comembership_feature_authority",
            "comembership_inference_authority",
            "renderer_geometry_checkpoint",
            "preregistration",
        )
    }

    # The only artifact opened before this call is this execution JSON.  All
    # target/query/renderer/code records remain unopened until source PASS.
    source_gate = validate_source_pilot_chain(
        records["source_pilot_result"]["path"],
        expected_sha256=records["source_pilot_result"]["sha256"],
        require_promotion=True,
    )
    if source_gate.get("source_promotion_authorized") is not True:
        raise ValueError("LERF V2.1 source promotion is not authorized")
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
        records["v21_checkpoint"] != source_gate["checkpoint"]
        or records["v21_normalization"] != source_gate["normalization_authority"]
        or records["canonical_negative_bank"] != promoted_negative
    ):
        raise ValueError("LERF V2.1 promoted model/calibration binding differs")

    implementation = validate_file_record(
        authority["implementation"], label="LERF V2.1 cache implementation"
    )
    if implementation != Path(__file__).resolve():
        raise ValueError("LERF V2.1 cache implementation differs")
    preregistration = validate_file_record(
        records["preregistration"], label="LERF V2.1 deployment preregistration"
    )
    if preregistration != PREREGISTRATION:
        raise ValueError("LERF V2.1 deployment preregistration differs")
    dependencies = authority.get("implementation_dependencies")
    if not isinstance(dependencies, Mapping) or set(dependencies) != set(
        IMPLEMENTATION_DEPENDENCIES
    ):
        raise ValueError("LERF V2.1 cache implementation dependencies differ")
    for name, expected in IMPLEMENTATION_DEPENDENCIES.items():
        verified = validate_file_record(
            dependencies[name], label=f"LERF V2.1 dependency {name}"
        )
        if verified != expected:
            raise ValueError(f"LERF V2.1 dependency differs: {name}")

    query_execution = validate_query_execution_authority(
        records["query_relevance_execution_authority"]["path"],
        expected_sha256=records["query_relevance_execution_authority"]["sha256"],
        expected_output=records["query_relevance_authority"]["path"],
    )
    if (
        query_execution["source_pilot_result"] != records["source_pilot_result"]
        or query_execution["target_descriptor"] != records["target_descriptor"]
        or query_execution["positive_text_cache"] != records["positive_text_cache"]
        or query_execution["canonical_negative_bank"]
        != records["canonical_negative_bank"]
        or query_execution["verified_source_gate"]["checkpoint"]
        != records["v21_checkpoint"]
        or query_execution["verified_source_gate"]["normalization_authority"]
        != records["v21_normalization"]
    ):
        raise ValueError("LERF V2.1 nested query execution binding differs")
    descriptor = query_execution["verified_descriptor"]

    relevance_path = validate_file_record(
        records["query_relevance_authority"],
        label="LERF V2.1 query relevance authority",
    )
    relevance_raw, relevance_sha, relevance_source = load_torch_mapping(
        relevance_path,
        expected_sha256=records["query_relevance_authority"]["sha256"],
        map_location="cpu",
        label="LERF V2.1 query relevance authority",
    )
    relevance = validate_query_relevance_authority(relevance_raw)
    if (
        relevance["query_execution_authority"]
        != records["query_relevance_execution_authority"]
        or relevance["input_authority"]
        != {
            "target_descriptor": records["target_descriptor"],
            "positive_text_cache": records["positive_text_cache"],
            "canonical_negative_bank": records["canonical_negative_bank"],
        }
        or relevance["scene_id"] != authority["scene_id"]
        or relevance["physical_space_id"] != authority["physical_space_id"]
    ):
        raise ValueError("LERF V2.1 relevance execution/input binding differs")

    feature_path = validate_file_record(
        records["comembership_feature_authority"],
        label="LERF V2.1 co-membership feature authority",
    )
    feature_raw, feature_sha, feature_source = load_torch_mapping(
        feature_path,
        expected_sha256=records["comembership_feature_authority"]["sha256"],
        map_location="cpu",
        label="LERF V2.1 co-membership feature authority",
    )
    feature = validate_feature_authority(feature_raw)
    inference_path = validate_file_record(
        records["comembership_inference_authority"],
        label="LERF V2.1 co-membership inference authority",
    )
    inference_raw, inference_sha, inference_source = load_torch_mapping(
        inference_path,
        expected_sha256=records["comembership_inference_authority"]["sha256"],
        map_location="cpu",
        label="LERF V2.1 co-membership inference authority",
    )
    inference = validate_inference_authority(inference_raw)
    selected_rule = v2_cache.validate_v2_authority_binding(
        feature=feature,
        inference=inference,
        feature_record={"path": str(feature_source), "sha256": feature_sha},
    )
    accepted_record = feature.get("input_authority", {}).get("accepted_v2")
    accepted_path = validate_file_record(
        accepted_record, label="LERF V2.1 target AcceptedV2 authority"
    )
    accepted_raw, _, _ = load_torch_mapping(
        accepted_path,
        expected_sha256=accepted_record["sha256"],
        map_location="cpu",
        label="LERF V2.1 target AcceptedV2 authority",
    )
    accepted = validate_target_accepted_v2_authority(accepted_raw)

    renderer_path = validate_file_record(
        records["renderer_geometry_checkpoint"],
        label="LERF V2.1 renderer geometry checkpoint",
    )
    renderer_raw, renderer_sha, renderer_source = (
        load_sha_bound_project_checkpoint_mapping(
            renderer_path,
            expected_sha256=records["renderer_geometry_checkpoint"]["sha256"],
            map_location="cpu",
            label="LERF V2.1 renderer geometry checkpoint",
        )
    )
    renderer_xyz = score_cache._renderer_checkpoint_xyz(renderer_raw)
    v2_cache.validate_renderer_geometry_binding(
        feature=feature,
        accepted=accepted,
        accepted_record=accepted_record,
        renderer_geometry_checkpoint_sha256=renderer_sha,
    )
    state_record = descriptor["input_authority"]["factorized_primitive_state"]
    state = load_factorized_primitive_state(
        state_record["path"], expected_sha256=state_record["sha256"]
    )
    if (
        accepted_record != descriptor["input_authority"]["target_accepted_v2"]
        or records["v21_checkpoint"] != descriptor["input_authority"]["v21_checkpoint"]
        or records["v21_normalization"]
        != descriptor["input_authority"]["v21_normalization"]
        or feature["scene_id"] != authority["scene_id"]
        or descriptor["scene_id"] != authority["scene_id"]
        or descriptor["physical_space_id"] != authority["physical_space_id"]
        or accepted["physical_space_id"] != authority["physical_space_id"]
        or accepted["physical_space_authority"]["geometry_checkpoint_sha256"]
        != renderer_sha
        or relevance["region_row_ids"] != descriptor["region_row_ids"]
        or relevance["region_fingerprints"] != descriptor["region_fingerprints"]
        or descriptor["region_fingerprints"] != feature["region_fingerprints"]
        or not torch.equal(
            relevance["canonical_region_indices"],
            descriptor["canonical_region_indices"],
        )
        or not torch.equal(
            descriptor["canonical_region_indices"],
            feature["canonical_region_indices"],
        )
        or state.xyz.shape != renderer_xyz.shape
        or not torch.equal(state.xyz.float(), renderer_xyz)
        or state.valid.shape != (renderer_xyz.shape[0],)
    ):
        raise ValueError("LERF V2.1 descriptor/graph/renderer binding differs")

    output = _canonical_output(authority["output_cache"], label="output_cache")
    report = _canonical_output(authority["output_report"], label="output_report")
    if output == report:
        raise ValueError("LERF V2.1 cache and report outputs must differ")
    authority.update(records)
    authority["renderer_geometry_checkpoint"] = {
        "path": str(renderer_source),
        "sha256": renderer_sha,
    }
    authority["output_cache"] = output
    authority["output_report"] = report
    authority["verified_record"] = {"path": str(source), "sha256": digest}
    authority["verified_source_gate"] = source_gate
    authority["verified_query_execution"] = query_execution
    authority["verified_descriptor"] = descriptor
    authority["verified_relevance"] = relevance
    authority["verified_relevance_record"] = {
        "path": str(relevance_source),
        "sha256": relevance_sha,
    }
    authority["verified_feature"] = feature
    authority["verified_feature_record"] = {
        "path": str(feature_source),
        "sha256": feature_sha,
    }
    authority["verified_inference"] = inference
    authority["verified_inference_record"] = {
        "path": str(inference_source),
        "sha256": inference_sha,
    }
    authority["verified_state"] = state
    authority["verified_renderer_xyz"] = renderer_xyz
    authority["selected_rule"] = selected_rule
    return authority


def run(args: argparse.Namespace) -> dict[str, Any]:
    execution = validate_external_execution_authority(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
    )
    output = Path(execution["output_cache"])
    report_path = Path(execution["output_report"])
    if (
        output.exists()
        or output.is_symlink()
        or report_path.exists()
        or report_path.is_symlink()
    ):
        raise FileExistsError("LERF V2.1 external cache outputs must be new")
    state = execution["verified_state"]
    readout = greedy_novelty_readout_from_v21(
        feature=execution["verified_feature"],
        inference=execution["verified_inference"],
        relevance=execution["verified_relevance"],
        num_primitives=int(state.xyz.shape[0]),
    )
    membership, removed_invalid = mask_union_to_valid(
        readout.primitive_membership, state.valid
    )
    graph_audit = semantic_audit.semantic_conditioned_bounded_region_union(
        region_absolute_relevance=execution["verified_relevance"][
            "region_absolute_relevance"
        ],
        pair_indices=execution["verified_feature"]["pair_indices"],
        pair_probabilities=execution["verified_inference"]["pair_probabilities"],
        probability_threshold=float(execution["selected_rule"]["threshold"]),
        method=str(execution["selected_rule"]["method"]),
        maximum_regions=int(execution["selected_rule"]["maximum_regions"]),
        region_rows=execution["verified_feature"]["region_rows"],
        token_mask=execution["verified_feature"]["token_mask"],
        num_primitives=int(state.xyz.shape[0]),
    )
    query_ids = list(execution["verified_relevance"]["query_ids"])
    cache = {
        "schema": SCHEMA,
        "query_scores": membership,
        "valid": state.valid.bool().cpu().contiguous(),
        "xyz": state.xyz.float().cpu().contiguous(),
        "metadata": {
            "query_names": query_ids,
            "score_semantics": "binary_v21_absolute_relevance_greedy_novelty_union_membership",
            "semantic_selection": "source_calibrated_v21_absolute_relevance_greedy_novelty",
            "semantic_boundary": V21_ABSOLUTE_RELEVANCE_BOUNDARY,
            "maximum_regions": V21_MAXIMUM_REGIONS,
            "positive_relevancy": "exact_source_training_canonical_negative_logit_scale_10",
            "postprocess_before_region_readout": "none",
            "co_membership_role": "auxiliary_audit_only_not_a_hard_edge",
            "co_membership_audit_rule": execution["selected_rule"],
            "preregistration": execution["preregistration"],
            "query_relevance_authority": execution["verified_relevance_record"],
            "feature_authority": execution["verified_feature_record"],
            "inference_authority": execution["verified_inference_record"],
            "renderer_geometry_checkpoint": execution["renderer_geometry_checkpoint"],
            "producer": file_record(Path(__file__).resolve()),
            "execution_authority": execution["verified_record"],
        },
        "selection": {
            "region_indices": readout.selected_region_indices,
            "region_scores": readout.selected_region_scores,
            "marginal_core_rows": readout.selected_marginal_core_rows,
            "canonical_region_indices": execution["verified_feature"][
                "canonical_region_indices"
            ],
            "invalid_memberships_removed": removed_invalid,
            "co_membership_audit_seed_region_indices": graph_audit.seed_region_indices,
            "co_membership_audit_selected_region_masks": graph_audit.selected_region_masks,
        },
    }
    written = write_torch_noclobber(output, cache)
    report = {
        "schema": SCHEMA,
        "status": "v21_calibrated_greedy_novelty_external_cache_complete",
        "cache": file_record(written),
        "query_ids": query_ids,
        "selection_rule": {
            "score_threshold": V21_ABSOLUTE_RELEVANCE_BOUNDARY,
            "maximum_regions": V21_MAXIMUM_REGIONS,
            "utility": "probability_times_novel_core_fraction",
            "stop": "zero_novelty_or_no_semantic_positive_candidate",
        },
        "selected_region_indices": [
            list(values) for values in readout.selected_region_indices
        ],
        "selected_region_counts": [
            len(values) for values in readout.selected_region_indices
        ],
        "selected_primitive_counts": membership.sum(dim=0).int().tolist(),
        "invalid_memberships_removed": removed_invalid,
        "co_membership_audit_selected_region_counts": graph_audit.selected_region_masks.sum(
            dim=0
        ).tolist(),
        "query_relevance_authority": execution["verified_relevance_record"],
        "execution_authority": execution["verified_record"],
        "access_audit": external_cache_access_audit(),
    }
    write_frozen_json(report_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-authority", required=True)
    parser.add_argument("--expected-execution-authority-sha256", required=True)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
