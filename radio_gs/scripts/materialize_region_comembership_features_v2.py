#!/usr/bin/env python3
"""Materialize query-independent RegionCoMembershipV2 pair features.

Source parity appends capability channels to a V1 source feature authority.
Target materialization is independent of the V1 promotion gate: it validates
the formal source-only V2 promotion before opening target inputs, then builds
the fixed V1 base channels directly from canonical authorities.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces.factorized_primitive_state import (
    load_factorized_primitive_state,
)
from radio_gs.interfaces.region_comembership_v2_formal import (
    TARGET_INPUT_NAMES,
    validate_target_execution_authority,
)
from radio_gs.interfaces.surface_region_target_accepted_v2 import (
    validate_target_accepted_v2_authority,
)
from radio_gs.interfaces.surface_region_target_adaptive_typed_context import (
    validate_target_adaptive_typed_context_authority,
)
from radio_gs.models.region_comembership_v2 import PAIR_FEATURE_NAMES
from radio_gs.scripts.audit_region_comembership_v2_scene1_to_scene2 import (
    capability_pair_features,
)
from radio_gs.scripts.build_source_region_comembership_v1 import (
    build_query_independent_pair_features,
)
from radio_gs.scripts.infer_region_comembership_v1 import (
    validate_feature_authority as validate_v1_feature_authority,
)
from radio_gs.scripts.materialize_region_capability_descriptors_v2 import (
    validate_region_capability_descriptor_authority,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_torch_mapping,
    validate_file_record,
    write_torch_noclobber,
)


SCHEMA = "radio_gs.region_comembership_feature_authority.v2"
SCHEMA_VERSION = 2
CHANNEL_NAMES = (
    "canonical_region_indices",
    "region_rows",
    "token_mask",
    "pair_indices",
    "pair_features",
)
SOURCE_INPUT_NAMES = ("v1_feature_authority", "capability_descriptor")


def _identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    names = (
        "schema",
        "schema_version",
        "scene_id",
        "domain",
        "producer",
        "target_execution_authority",
        "input_authority",
        "candidate_policy",
        "feature_names",
        "feature_names_sha256",
        "source_access",
        "region_fingerprints_sha256",
        "canonical_axis_sha256",
        "pair_axis_sha256",
        "tensor_authority_sha256",
    )
    return {name: payload[name] for name in names}


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


def _canonical_axis_sha256(payload: Mapping[str, Any]) -> str:
    channels = payload["channel_sha256"]
    return canonical_json_sha256(
        {
            "region_fingerprints_sha256": payload["region_fingerprints_sha256"],
            "canonical_region_indices_sha256": channels[
                "canonical_region_indices"
            ],
            "region_rows_sha256": channels["region_rows"],
            "token_mask_sha256": channels["token_mask"],
        }
    )


def _pair_axis_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {
            "canonical_axis_sha256": payload["canonical_axis_sha256"],
            "pair_indices_sha256": payload["channel_sha256"]["pair_indices"],
        }
    )


def validate_feature_authority(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("RegionCoMembership V2 feature authority must be a mapping")
    payload = dict(value)
    required = {
        "schema",
        "schema_version",
        "scene_id",
        "domain",
        "producer",
        "target_execution_authority",
        "input_authority",
        "candidate_policy",
        "feature_names",
        "feature_names_sha256",
        "source_access",
        "region_fingerprints_sha256",
        "canonical_axis_sha256",
        "pair_axis_sha256",
        "tensor_authority_sha256",
        "content_authority_sha256",
        "region_fingerprints",
        *CHANNEL_NAMES,
        "channel_sha256",
        "audit",
    }
    if (
        set(payload) != required
        or payload.get("schema") != SCHEMA
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("domain") not in {"source_parity", "target"}
        or payload.get("feature_names") != list(PAIR_FEATURE_NAMES)
        or payload.get("feature_names_sha256")
        != canonical_json_sha256(list(PAIR_FEATURE_NAMES))
        or payload.get("source_access") != _source_access(payload.get("domain"))
        or payload.get("content_authority_sha256")
        != canonical_json_sha256(_identity(payload))
        or (payload.get("domain") == "target")
        != (payload.get("target_execution_authority") is not None)
    ):
        raise ValueError("RegionCoMembership V2 feature identity differs")
    validate_file_record(payload["producer"], label="V2 feature producer")
    inputs = payload.get("input_authority")
    expected_inputs = (
        SOURCE_INPUT_NAMES
        if payload.get("domain") == "source_parity"
        else TARGET_INPUT_NAMES
    )
    if not isinstance(inputs, Mapping) or set(inputs) != set(expected_inputs):
        raise ValueError("RegionCoMembership V2 feature inputs differ")
    for name, record in inputs.items():
        validate_file_record(record, label=f"V2 feature {name}")
    if payload.get("domain") == "target":
        validate_file_record(
            payload["target_execution_authority"],
            label="V2 target feature execution authority",
        )

    canonical = torch.as_tensor(payload["canonical_region_indices"])
    rows = torch.as_tensor(payload["region_rows"])
    mask = torch.as_tensor(payload["token_mask"])
    pairs = torch.as_tensor(payload["pair_indices"])
    features = torch.as_tensor(payload["pair_features"])
    count = int(canonical.numel())
    if (
        count <= 0
        or canonical.dtype != torch.int64
        or canonical.ndim != 1
        or rows.dtype not in {torch.int32, torch.int64}
        or rows.ndim != 2
        or rows.shape[0] != count
        or mask.dtype != torch.bool
        or mask.shape != rows.shape
        or not bool(mask.any(dim=1).all())
        or bool((rows[mask] < 0).any())
        or bool((canonical < 0).any())
        or int(torch.unique(canonical).numel()) != count
        or pairs.dtype != torch.int64
        or pairs.ndim != 2
        or pairs.shape[0] != 2
        or bool((pairs < 0).any())
        or bool((pairs >= count).any())
        or bool((pairs[0] >= pairs[1]).any())
        or features.dtype != torch.float32
        or features.shape != (pairs.shape[1], len(PAIR_FEATURE_NAMES))
        or not bool(torch.isfinite(features).all())
        or len(payload["region_fingerprints"]) != count
        or len(set(payload["region_fingerprints"])) != count
        or not all(
            isinstance(value, str) and bool(value)
            for value in payload["region_fingerprints"]
        )
    ):
        raise ValueError("RegionCoMembership V2 feature tensors differ")
    pair_keys = pairs[0] * count + pairs[1]
    if pair_keys.numel() <= 0 or (
        pair_keys.numel() > 1 and not bool((pair_keys[1:] > pair_keys[:-1]).all())
    ):
        raise ValueError("RegionCoMembership V2 pairs are not sorted unique")
    if set(payload["channel_sha256"]) != set(CHANNEL_NAMES):
        raise ValueError("RegionCoMembership V2 feature channel mapping differs")
    for name in CHANNEL_NAMES:
        if payload["channel_sha256"].get(name) != tensor_sha256(payload[name]):
            raise ValueError(f"RegionCoMembership V2 feature changed: {name}")
    if (
        payload.get("region_fingerprints_sha256")
        != canonical_json_sha256(payload["region_fingerprints"])
        or payload.get("tensor_authority_sha256")
        != canonical_json_sha256(payload["channel_sha256"])
        or payload.get("canonical_axis_sha256") != _canonical_axis_sha256(payload)
        or payload.get("pair_axis_sha256") != _pair_axis_sha256(payload)
    ):
        raise ValueError("RegionCoMembership V2 canonical/pair SHA axis differs")
    return payload


def combine_feature_authorities(
    *,
    v1: Mapping[str, Any],
    descriptor: Mapping[str, Any],
) -> torch.Tensor:
    """Return the fixed 21-channel pair tensor after exact axis validation."""

    base = validate_v1_feature_authority(v1)
    capability = validate_region_capability_descriptor_authority(descriptor)
    if (
        capability["scene_id"] != base["scene_id"]
        or capability["input_authority"]["accepted_v2"]["sha256"]
        != base["input_authority"]["accepted_v2"]["sha256"]
        or capability["region_fingerprints"] != base["region_fingerprints"]
        or not torch.equal(
            capability["canonical_region_indices"],
            base["canonical_region_indices"],
        )
        or not torch.equal(capability["region_rows"], base["region_rows"])
        or not torch.equal(capability["token_mask"], base["token_mask"])
    ):
        raise ValueError("V1 feature and capability descriptor axes differ")
    appended = capability_pair_features(
        pair_indices=base["pair_indices"],
        appearance_direction=capability["appearance_direction"],
        boundary_direction=capability["boundary_direction"],
        appearance_concentration=capability["appearance_concentration"],
        boundary_concentration=capability["boundary_concentration"],
    )
    return torch.cat((base["pair_features"], appended), dim=1).float().contiguous()


def _append_capability_features(
    *, pair_indices: torch.Tensor, base_features: torch.Tensor, descriptor: Mapping[str, Any]
) -> torch.Tensor:
    appended = capability_pair_features(
        pair_indices=pair_indices,
        appearance_direction=descriptor["appearance_direction"],
        boundary_direction=descriptor["boundary_direction"],
        appearance_concentration=descriptor["appearance_concentration"],
        boundary_concentration=descriptor["boundary_concentration"],
    )
    return torch.cat((base_features, appended), dim=1).float().contiguous()


def _finalize_payload(
    *,
    identity: Mapping[str, Any],
    region_fingerprints: list[str],
    canonical_region_indices: torch.Tensor,
    region_rows: torch.Tensor,
    token_mask: torch.Tensor,
    pair_indices: torch.Tensor,
    pair_features: torch.Tensor,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        **dict(identity),
        "region_fingerprints": list(region_fingerprints),
        "canonical_region_indices": canonical_region_indices,
        "region_rows": region_rows,
        "token_mask": token_mask,
        "pair_indices": pair_indices,
        "pair_features": pair_features,
        "channel_sha256": {},
        "audit": dict(audit),
    }
    payload["channel_sha256"] = {
        name: tensor_sha256(payload[name]) for name in CHANNEL_NAMES
    }
    payload["region_fingerprints_sha256"] = canonical_json_sha256(
        payload["region_fingerprints"]
    )
    payload["canonical_axis_sha256"] = _canonical_axis_sha256(payload)
    payload["pair_axis_sha256"] = _pair_axis_sha256(payload)
    payload["tensor_authority_sha256"] = canonical_json_sha256(
        payload["channel_sha256"]
    )
    payload["content_authority_sha256"] = canonical_json_sha256(_identity(payload))
    validate_feature_authority(payload)
    return payload


def _load_target_base(
    *, scene_id: str, records: Mapping[str, Mapping[str, str]]
) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor, torch.Tensor, dict[str, Any]]:
    accepted_raw, accepted_sha, accepted_path = load_torch_mapping(
        records["accepted_v2"]["path"],
        expected_sha256=records["accepted_v2"]["sha256"],
        map_location="cpu",
        label="V2 target AcceptedV2 authority",
    )
    accepted = validate_target_accepted_v2_authority(accepted_raw)
    context_raw, context_sha, context_path = load_torch_mapping(
        records["typed_context"]["path"],
        expected_sha256=records["typed_context"]["sha256"],
        map_location="cpu",
        label="V2 target typed-context authority",
    )
    context = validate_target_adaptive_typed_context_authority(context_raw)
    graph, graph_sha, graph_path = load_torch_mapping(
        records["support_graph"]["path"],
        expected_sha256=records["support_graph"]["sha256"],
        map_location="cpu",
        label="V2 target support graph",
    )
    state = load_factorized_primitive_state(
        records["factorized_state"]["path"],
        expected_sha256=records["factorized_state"]["sha256"],
    )
    if (
        accepted["scene_id"] != scene_id
        or context["scene_id"] != scene_id
        or context["physical_space_id"] != accepted["physical_space_id"]
        or context["physical_space_authority"]
        != accepted["physical_space_authority"]
        or context["input_authority"][
            "accepted_v2_canonical_region_authority"
        ]
        != {"path": str(accepted_path), "sha256": accepted_sha}
        or context["region_row_ids"]
        != [
            f"{scene_id}:accepted-v2-canonical-v1:{value}"
            for value in accepted["region_fingerprints"]
        ]
        or not torch.equal(
            accepted["canonical_region_indices"], context["canonical_region_indices"]
        )
        or not torch.equal(torch.as_tensor(graph["global_rows"]).long(), state.global_rows)
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
        raise ValueError("V2 target scene/region/geometry authorities differ")
    pairs, features, audit = build_query_independent_pair_features(
        accepted=accepted, context=context, state=state, graph=graph
    )
    verified_records = {
        "accepted_v2": {"path": str(accepted_path), "sha256": accepted_sha},
        "typed_context": {"path": str(context_path), "sha256": context_sha},
        "support_graph": {"path": str(graph_path), "sha256": graph_sha},
        "factorized_state": {"path": str(state.source), "sha256": state.sha256},
    }
    return accepted, verified_records, pairs, features, audit


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"V2 feature authority exists: {output}")
    domain = str(args.domain)
    target_execution = None
    if domain == "source_parity":
        if (
            not args.v1_feature_authority
            or not args.expected_v1_feature_authority_sha256
            or not args.capability_descriptor
            or not args.expected_capability_descriptor_sha256
        ):
            raise ValueError("V2 source parity requires V1 feature and capability")
        if args.scene_id or args.execution_authority or args.expected_execution_authority_sha256:
            raise ValueError("V2 source parity must not supply target authority")
        v1_raw, v1_sha, v1_path = load_torch_mapping(
            args.v1_feature_authority,
            expected_sha256=args.expected_v1_feature_authority_sha256,
            map_location="cpu",
            label="V1 source feature authority",
        )
        v1 = validate_v1_feature_authority(v1_raw)
        if v1["domain"] != "source_parity" or v1["source_access"] != _source_access(
            "source_parity"
        ):
            raise ValueError("V2 source parity cannot consume a target V1 authority")
        descriptor_raw, descriptor_sha, descriptor_path = load_torch_mapping(
            args.capability_descriptor,
            expected_sha256=args.expected_capability_descriptor_sha256,
            map_location="cpu",
            label="V2 source capability descriptor",
        )
        descriptor = validate_region_capability_descriptor_authority(descriptor_raw)
        pair_features = combine_feature_authorities(v1=v1, descriptor=descriptor)
        scene_id = str(v1["scene_id"])
        input_authority = {
            "v1_feature_authority": {"path": str(v1_path), "sha256": v1_sha},
            "capability_descriptor": {
                "path": str(descriptor_path),
                "sha256": descriptor_sha,
            },
        }
        region_fingerprints = list(v1["region_fingerprints"])
        canonical = v1["canonical_region_indices"]
        rows = v1["region_rows"]
        mask = v1["token_mask"]
        pairs = v1["pair_indices"]
        audit = {
            "canonical_regions": int(canonical.numel()),
            "candidate_pairs": int(pairs.shape[1]),
            "pair_feature_dimension": len(PAIR_FEATURE_NAMES),
        }
    elif domain == "target":
        if (
            not args.scene_id
            or not args.execution_authority
            or not args.expected_execution_authority_sha256
        ):
            raise ValueError("V2 target materialization requires exact execution authority")
        if (
            args.v1_feature_authority
            or args.expected_v1_feature_authority_sha256
            or args.capability_descriptor
            or args.expected_capability_descriptor_sha256
        ):
            raise ValueError("V2 target materialization must use authority-bound inputs")
        scene_id = str(args.scene_id)
        execution = validate_target_execution_authority(
            args.execution_authority,
            expected_sha256=args.expected_execution_authority_sha256,
            scene_id=scene_id,
            expected_feature_output=output,
        )
        target_execution = execution["verified_record"]
        records = execution["target_feature_inputs"]
        accepted, input_authority, pairs, base_features, audit = _load_target_base(
            scene_id=scene_id, records=records
        )
        descriptor_raw, descriptor_sha, descriptor_path = load_torch_mapping(
            records["capability_descriptor"]["path"],
            expected_sha256=records["capability_descriptor"]["sha256"],
            map_location="cpu",
            label="V2 target capability descriptor",
        )
        descriptor = validate_region_capability_descriptor_authority(descriptor_raw)
        geometry = accepted["input_authority"]["geometry_authority"]
        if (
            descriptor["scene_id"] != scene_id
            or descriptor["input_authority"]["accepted_v2"]["sha256"]
            != input_authority["accepted_v2"]["sha256"]
            or descriptor["input_authority"]["factorized_field_checkpoint_sha256"]
            != geometry["factorized_field_checkpoint_file_sha256"]
            or descriptor["input_authority"]["primitive_row_authority_sha256"]
            != geometry["primitive_row_authority_sha256"]
            or descriptor["region_fingerprints"] != accepted["region_fingerprints"]
            or not torch.equal(
                descriptor["canonical_region_indices"],
                accepted["canonical_region_indices"],
            )
            or not torch.equal(descriptor["region_rows"], accepted["region_rows"])
            or not torch.equal(descriptor["token_mask"], accepted["token_mask"])
        ):
            raise ValueError("V2 target capability/canonical axes differ")
        input_authority["capability_descriptor"] = {
            "path": str(descriptor_path),
            "sha256": descriptor_sha,
        }
        if input_authority != records:
            raise ValueError("V2 target inputs differ from execution authority")
        pair_features = _append_capability_features(
            pair_indices=pairs, base_features=base_features, descriptor=descriptor
        )
        region_fingerprints = list(accepted["region_fingerprints"])
        canonical = accepted["canonical_region_indices"]
        rows = accepted["region_rows"]
        mask = accepted["token_mask"]
        audit = {
            "canonical_regions": int(canonical.numel()),
            **audit,
            "pair_feature_dimension": len(PAIR_FEATURE_NAMES),
        }
    else:
        raise ValueError("RegionCoMembership V2 feature domain differs")
    identity = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "scene_id": scene_id,
        "domain": domain,
        "producer": file_record(Path(__file__).resolve()),
        "target_execution_authority": target_execution,
        "input_authority": input_authority,
        "candidate_policy": {
            "descriptor_neighbors": 16,
            "centroid_neighbors": 16,
            "anchor_support_edges": True,
        },
        "feature_names": list(PAIR_FEATURE_NAMES),
        "feature_names_sha256": canonical_json_sha256(list(PAIR_FEATURE_NAMES)),
        "source_access": _source_access(domain),
    }
    payload = _finalize_payload(
        identity=identity,
        region_fingerprints=region_fingerprints,
        canonical_region_indices=canonical,
        region_rows=rows,
        token_mask=mask,
        pair_indices=pairs,
        pair_features=pair_features,
        audit=audit,
    )
    written = write_torch_noclobber(output, payload)
    return {
        "status": "region_comembership_v2_feature_authority_complete",
        "scene_id": scene_id,
        "domain": domain,
        "output": file_record(written),
        "audit": payload["audit"],
        "target_metric_computed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("source_parity", "target"), default="source_parity")
    parser.add_argument("--scene-id")
    parser.add_argument("--v1-feature-authority")
    parser.add_argument("--expected-v1-feature-authority-sha256")
    parser.add_argument("--capability-descriptor")
    parser.add_argument("--expected-capability-descriptor-sha256")
    parser.add_argument("--execution-authority")
    parser.add_argument("--expected-execution-authority-sha256")
    parser.add_argument("--output", required=True)
    print(json.dumps(materialize(parser.parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
