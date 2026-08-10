#!/usr/bin/env python3
"""Append factorized-native relations to a frozen target V2 pair authority."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces import region_comembership_native_v3_target as formal
from radio_gs.interfaces.factorized_native_region_relation import (
    FEATURE_NAMES as NATIVE_FEATURE_NAMES,
    FEATURE_NAMES_SHA256 as NATIVE_FEATURE_NAMES_SHA256,
    INTERFACE_CONTRACT_SHA256 as NATIVE_INTERFACE_CONTRACT_SHA256,
    FactorizedNativeRegionSummary,
    factorized_native_pair_features,
    factorized_native_region_summaries,
)
from radio_gs.interfaces.factorized_primitive_state import (
    FactorizedPrimitiveState,
    load_factorized_primitive_state,
)
from radio_gs.interfaces.surface_region_target_accepted_v2 import (
    validate_target_accepted_v2_authority,
)
from radio_gs.models.region_comembership_native_v3 import PAIR_FEATURE_NAMES
from radio_gs.models.region_comembership_v2 import (
    PAIR_FEATURE_NAMES as V2_PAIR_FEATURE_NAMES,
)
from radio_gs.scripts import materialize_region_comembership_features_v2 as parent
from radio_gs.scripts.materialize_source_region_comembership_native_v3 import (
    _concatenate_summaries,
    _factorized_state_index,
    _gather_indexed_factorized_native_region_inputs,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_torch_mapping,
    validate_file_record,
    write_torch_noclobber,
)


SCHEMA = formal.FEATURE_SCHEMA
SCHEMA_VERSION = formal.SCHEMA_VERSION
CHANNEL_NAMES = (
    "canonical_region_indices",
    "region_rows",
    "token_mask",
    "canonical_anchor_index",
    "exact_state_anchor_mask",
    "pair_indices",
    "native_pair_active_mask",
    "legacy_v2_fallback_pair_mask",
    "v2_pair_features",
    "native_pair_features",
    "pair_features",
)
IDENTITY_NAMES = (
    "schema",
    "schema_version",
    "scene_id",
    "domain",
    "producer",
    "target_execution_authority",
    "input_authority",
    "feature_names",
    "feature_names_sha256",
    "native_feature_names",
    "native_feature_names_sha256",
    "native_interface_contract_sha256",
    "fallback_contract",
    "source_access",
    "region_fingerprints_sha256",
    "canonical_axis_sha256",
    "pair_axis_sha256",
    "tensor_authority_sha256",
)


def _identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {name: payload[name] for name in IDENTITY_NAMES}


def _canonical_axis_sha256(payload: Mapping[str, Any]) -> str:
    channels = payload["channel_sha256"]
    return canonical_json_sha256(
        {
            "region_fingerprints_sha256": payload["region_fingerprints_sha256"],
            "canonical_region_indices_sha256": channels["canonical_region_indices"],
            "region_rows_sha256": channels["region_rows"],
            "token_mask_sha256": channels["token_mask"],
            "canonical_anchor_index_sha256": channels["canonical_anchor_index"],
        }
    )


def _pair_axis_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {
            "canonical_axis_sha256": payload["canonical_axis_sha256"],
            "pair_indices_sha256": payload["channel_sha256"]["pair_indices"],
        }
    )


def materialize_native_pair_block(
    *,
    state: FactorizedPrimitiveState,
    region_rows: torch.Tensor,
    token_mask: torch.Tensor,
    canonical_anchor_index: torch.Tensor,
    pair_indices: torch.Tensor,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Compute native channels only for pairs with two exact canonical anchors."""

    rows = torch.as_tensor(region_rows).detach().long().cpu().contiguous()
    mask = torch.as_tensor(token_mask).detach().bool().cpu().contiguous()
    anchor = torch.as_tensor(canonical_anchor_index).detach().long().cpu().contiguous()
    pairs = torch.as_tensor(pair_indices).detach().long().cpu().contiguous()
    regions = int(rows.shape[0]) if rows.ndim == 2 else -1
    batch = torch.arange(regions) if regions > 0 else torch.empty(0, dtype=torch.long)
    if (
        regions <= 1
        or mask.shape != rows.shape
        or anchor.shape != (regions,)
        or pairs.ndim != 2
        or pairs.shape[0] != 2
        or pairs.shape[1] <= 0
        or bool((anchor < 0).any())
        or bool((anchor >= rows.shape[1]).any())
        or not bool(mask[batch, anchor].all())
        or int(batch_size) <= 0
    ):
        raise ValueError("native V3 target relation axes differ")
    anchor_rows = rows[batch, anchor]
    if bool((anchor_rows < 0).any()) or bool((anchor_rows >= state.valid.numel()).any()):
        raise ValueError("native V3 target canonical anchor row differs")
    exact_anchor = state.valid[anchor_rows].detach().bool().cpu().contiguous()
    active_pair = (exact_anchor[pairs[0]] & exact_anchor[pairs[1]]).contiguous()
    fallback_pair = (~active_pair).contiguous()
    native = torch.zeros(pairs.shape[1], len(NATIVE_FEATURE_NAMES), dtype=torch.float32)
    if bool(active_pair.any()):
        eligible = torch.nonzero(exact_anchor, as_tuple=False).flatten()
        state_index = _factorized_state_index(state)
        summaries: list[FactorizedNativeRegionSummary] = []
        for start in range(0, int(eligible.numel()), int(batch_size)):
            selected = eligible[start : start + int(batch_size)]
            gathered = _gather_indexed_factorized_native_region_inputs(
                state,
                state_index,
                rows[selected],
                mask[selected],
                anchor[selected],
            )
            summaries.append(
                factorized_native_region_summaries(
                    unit_direction=gathered.unit_direction,
                    log_amplitude=gathered.log_amplitude,
                    state=gathered.state,
                    state_known_mask=gathered.state_known_mask,
                    token_mask=gathered.token_mask,
                    anchor_index=gathered.anchor_index,
                )
            )
        summary = _concatenate_summaries(summaries)
        region_to_eligible = torch.full((regions,), -1, dtype=torch.int64)
        region_to_eligible[eligible] = torch.arange(eligible.numel(), dtype=torch.int64)
        compact_pairs = region_to_eligible[pairs[:, active_pair]]
        if bool((compact_pairs < 0).any()):
            raise RuntimeError("native V3 target active-pair compaction differs")
        native[active_pair] = factorized_native_pair_features(summary, compact_pairs)
    if bool(native[fallback_pair].count_nonzero()):
        raise RuntimeError("native V3 fallback feature sentinel changed")
    return {
        "exact_state_anchor_mask": exact_anchor,
        "native_pair_active_mask": active_pair,
        "legacy_v2_fallback_pair_mask": fallback_pair,
        "native_pair_features": native.contiguous(),
    }


def validate_feature_authority(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("native V3 target feature authority must be a mapping")
    payload = dict(value)
    required = {
        *IDENTITY_NAMES,
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
        or payload.get("domain") != "target"
        or payload.get("feature_names") != list(PAIR_FEATURE_NAMES)
        or payload.get("feature_names_sha256")
        != canonical_json_sha256(list(PAIR_FEATURE_NAMES))
        or payload.get("native_feature_names") != list(NATIVE_FEATURE_NAMES)
        or payload.get("native_feature_names_sha256") != NATIVE_FEATURE_NAMES_SHA256
        or payload.get("native_interface_contract_sha256")
        != NATIVE_INTERFACE_CONTRACT_SHA256
        or payload.get("fallback_contract") != formal.fallback_contract()
        or payload.get("source_access") != formal.access_audit(target_opened=True)
        or payload.get("content_authority_sha256")
        != canonical_json_sha256(_identity(payload))
    ):
        raise ValueError("native V3 target feature identity differs")
    validate_file_record(payload["producer"], label="native V3 feature producer")
    execution_record = payload.get("target_execution_authority")
    validate_file_record(
        execution_record, label="native V3 feature target execution authority"
    )
    inputs = payload.get("input_authority")
    if not isinstance(inputs, Mapping) or set(inputs) != set(formal.TARGET_INPUT_NAMES):
        raise ValueError("native V3 target feature input set differs")
    for name, record in inputs.items():
        validate_file_record(record, label=f"native V3 target feature {name}")

    canonical = torch.as_tensor(payload["canonical_region_indices"])
    rows = torch.as_tensor(payload["region_rows"])
    mask = torch.as_tensor(payload["token_mask"])
    anchor = torch.as_tensor(payload["canonical_anchor_index"])
    exact = torch.as_tensor(payload["exact_state_anchor_mask"])
    pairs = torch.as_tensor(payload["pair_indices"])
    active = torch.as_tensor(payload["native_pair_active_mask"])
    fallback = torch.as_tensor(payload["legacy_v2_fallback_pair_mask"])
    v2_features = torch.as_tensor(payload["v2_pair_features"])
    native = torch.as_tensor(payload["native_pair_features"])
    features = torch.as_tensor(payload["pair_features"])
    count = int(canonical.numel()) if canonical.ndim == 1 else -1
    if (
        count <= 1
        or canonical.dtype != torch.int64
        or rows.dtype not in {torch.int32, torch.int64}
        or rows.ndim != 2
        or rows.shape[0] != count
        or mask.dtype != torch.bool
        or mask.shape != rows.shape
        or not bool(mask.any(dim=1).all())
        or anchor.dtype != torch.int64
        or anchor.shape != (count,)
        or bool((anchor < 0).any())
        or bool((anchor >= rows.shape[1]).any())
        or not bool(mask[torch.arange(count), anchor].all())
        or exact.dtype != torch.bool
        or exact.shape != (count,)
        or pairs.dtype != torch.int64
        or pairs.ndim != 2
        or pairs.shape[0] != 2
        or bool((pairs < 0).any())
        or bool((pairs >= count).any())
        or bool((pairs[0] >= pairs[1]).any())
        or active.dtype != torch.bool
        or active.shape != (pairs.shape[1],)
        or fallback.dtype != torch.bool
        or fallback.shape != active.shape
        or not torch.equal(fallback, ~active)
        or not torch.equal(active, exact[pairs[0]] & exact[pairs[1]])
        or v2_features.dtype != torch.float32
        or v2_features.shape != (pairs.shape[1], len(V2_PAIR_FEATURE_NAMES))
        or native.dtype != torch.float32
        or native.shape != (pairs.shape[1], len(NATIVE_FEATURE_NAMES))
        or features.dtype != torch.float32
        or features.shape != (pairs.shape[1], len(PAIR_FEATURE_NAMES))
        or not torch.equal(features[:, : len(V2_PAIR_FEATURE_NAMES)], v2_features)
        or not torch.equal(features[:, len(V2_PAIR_FEATURE_NAMES) :], native)
        or not bool(torch.isfinite(features).all())
        or bool(native[fallback].count_nonzero())
        or len(payload["region_fingerprints"]) != count
        or len(set(payload["region_fingerprints"])) != count
    ):
        raise ValueError("native V3 target feature tensors differ")
    pair_keys = pairs[0] * count + pairs[1]
    if pair_keys.numel() <= 0 or (
        pair_keys.numel() > 1 and not bool((pair_keys[1:] > pair_keys[:-1]).all())
    ):
        raise ValueError("native V3 target pair axis is not sorted unique")
    channels = payload.get("channel_sha256")
    if not isinstance(channels, Mapping) or set(channels) != set(CHANNEL_NAMES):
        raise ValueError("native V3 target feature channels differ")
    for name in CHANNEL_NAMES:
        if channels[name] != tensor_sha256(payload[name]):
            raise ValueError(f"native V3 target feature changed: {name}")
    if (
        payload.get("region_fingerprints_sha256")
        != canonical_json_sha256(payload["region_fingerprints"])
        or payload.get("canonical_axis_sha256") != _canonical_axis_sha256(payload)
        or payload.get("pair_axis_sha256") != _pair_axis_sha256(payload)
        or payload.get("tensor_authority_sha256")
        != canonical_json_sha256(payload["channel_sha256"])
    ):
        raise ValueError("native V3 target feature SHA axis differs")

    parent_raw, _, _ = load_torch_mapping(
        inputs["parent_v2_feature_authority"]["path"],
        expected_sha256=inputs["parent_v2_feature_authority"]["sha256"],
        map_location="cpu",
        label="native V3 bound parent V2 feature",
    )
    parent_value = parent.validate_feature_authority(parent_raw)
    accepted_raw, _, _ = load_torch_mapping(
        inputs["accepted_v2"]["path"],
        expected_sha256=inputs["accepted_v2"]["sha256"],
        map_location="cpu",
        label="native V3 bound AcceptedV2",
    )
    accepted = validate_target_accepted_v2_authority(accepted_raw)
    state = load_factorized_primitive_state(
        inputs["factorized_state"]["path"],
        expected_sha256=inputs["factorized_state"]["sha256"],
    )
    anchor_rows = rows.long()[torch.arange(count), anchor]
    if (
        payload["scene_id"] != parent_value["scene_id"]
        or payload["scene_id"] != accepted["scene_id"]
        or inputs["accepted_v2"] != parent_value["input_authority"]["accepted_v2"]
        or inputs["factorized_state"]
        != parent_value["input_authority"]["factorized_state"]
        or not torch.equal(canonical, parent_value["canonical_region_indices"])
        or not torch.equal(rows, parent_value["region_rows"])
        or not torch.equal(mask, parent_value["token_mask"])
        or not torch.equal(anchor, accepted["anchor_index"])
        or not torch.equal(pairs, parent_value["pair_indices"])
        or not torch.equal(v2_features, parent_value["pair_features"])
        or payload["region_fingerprints"] != parent_value["region_fingerprints"]
        or not torch.equal(exact, state.valid[anchor_rows].cpu())
    ):
        raise ValueError("native V3 target feature parent chain differs")
    return payload


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"native V3 target feature exists: {output}")
    execution = formal.validate_target_execution_authority(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
        scene_id=str(args.scene_id),
        expected_feature_output=output,
    )
    records = execution["verified_target_inputs"]
    parent_value = execution["verified_parent_feature"]
    accepted_raw, _, _ = load_torch_mapping(
        records["accepted_v2"]["path"],
        expected_sha256=records["accepted_v2"]["sha256"],
        map_location="cpu",
        label="native V3 target AcceptedV2",
    )
    accepted = validate_target_accepted_v2_authority(accepted_raw)
    state = load_factorized_primitive_state(
        records["factorized_state"]["path"],
        expected_sha256=records["factorized_state"]["sha256"],
    )
    if (
        accepted["scene_id"] != str(args.scene_id)
        or accepted["input_authority"]["geometry_authority"]
        ["factorized_primitive_state_file_sha256"]
        != state.sha256
        or state.metadata.get("query_independent") is not True
        or any(
            state.metadata.get(name) is not False
            for name in (
                "benchmark_images_opened",
                "benchmark_masks_opened",
                "text_queries_opened",
            )
        )
        or accepted["region_fingerprints"] != parent_value["region_fingerprints"]
        or not torch.equal(
            accepted["canonical_region_indices"],
            parent_value["canonical_region_indices"],
        )
        or not torch.equal(accepted["region_rows"], parent_value["region_rows"])
        or not torch.equal(accepted["token_mask"], parent_value["token_mask"])
    ):
        raise ValueError("native V3 target Accepted/state/parent axes differ")
    block = materialize_native_pair_block(
        state=state,
        region_rows=accepted["region_rows"],
        token_mask=accepted["token_mask"],
        canonical_anchor_index=accepted["anchor_index"],
        pair_indices=parent_value["pair_indices"],
        batch_size=int(args.batch_size),
    )
    v2_features = parent_value["pair_features"].float().cpu().contiguous()
    combined = torch.cat((v2_features, block["native_pair_features"]), dim=1)
    identity = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "scene_id": str(args.scene_id),
        "domain": "target",
        "producer": file_record(Path(__file__).resolve()),
        "target_execution_authority": dict(execution["verified_record"]),
        "input_authority": records,
        "feature_names": list(PAIR_FEATURE_NAMES),
        "feature_names_sha256": canonical_json_sha256(list(PAIR_FEATURE_NAMES)),
        "native_feature_names": list(NATIVE_FEATURE_NAMES),
        "native_feature_names_sha256": NATIVE_FEATURE_NAMES_SHA256,
        "native_interface_contract_sha256": NATIVE_INTERFACE_CONTRACT_SHA256,
        "fallback_contract": formal.fallback_contract(),
        "source_access": formal.access_audit(target_opened=True),
        "region_fingerprints_sha256": canonical_json_sha256(
            parent_value["region_fingerprints"]
        ),
        "canonical_axis_sha256": "",
        "pair_axis_sha256": "",
        "tensor_authority_sha256": "",
    }
    payload = {
        **identity,
        "content_authority_sha256": "",
        "region_fingerprints": list(parent_value["region_fingerprints"]),
        "canonical_region_indices": parent_value["canonical_region_indices"],
        "region_rows": parent_value["region_rows"],
        "token_mask": parent_value["token_mask"],
        "canonical_anchor_index": accepted["anchor_index"].long().cpu().contiguous(),
        "exact_state_anchor_mask": block["exact_state_anchor_mask"],
        "pair_indices": parent_value["pair_indices"],
        "native_pair_active_mask": block["native_pair_active_mask"],
        "legacy_v2_fallback_pair_mask": block["legacy_v2_fallback_pair_mask"],
        "v2_pair_features": v2_features,
        "native_pair_features": block["native_pair_features"],
        "pair_features": combined.float().contiguous(),
        "channel_sha256": {},
        "audit": {
            "canonical_regions": int(parent_value["canonical_region_indices"].numel()),
            "candidate_pairs": int(parent_value["pair_indices"].shape[1]),
            "exact_state_anchor_regions": int(block["exact_state_anchor_mask"].sum()),
            "legacy_anchor_fallback_regions": int((~block["exact_state_anchor_mask"]).sum()),
            "native_active_pairs": int(block["native_pair_active_mask"].sum()),
            "legacy_v2_fallback_pairs": int(block["legacy_v2_fallback_pair_mask"].sum()),
            "native_pair_feature_dimension": len(NATIVE_FEATURE_NAMES),
            "combined_pair_feature_dimension": len(PAIR_FEATURE_NAMES),
            "alternate_anchor_substitution": False,
            "query_readout_executed": False,
        },
    }
    payload["channel_sha256"] = {
        name: tensor_sha256(payload[name]) for name in CHANNEL_NAMES
    }
    payload["canonical_axis_sha256"] = _canonical_axis_sha256(payload)
    payload["pair_axis_sha256"] = _pair_axis_sha256(payload)
    payload["tensor_authority_sha256"] = canonical_json_sha256(
        payload["channel_sha256"]
    )
    payload["content_authority_sha256"] = canonical_json_sha256(_identity(payload))
    validate_feature_authority(payload)
    written = write_torch_noclobber(output, payload)
    return {
        "status": "native_v3_query_free_target_feature_complete",
        "scene_id": payload["scene_id"],
        "output": file_record(written),
        "audit": payload["audit"],
        "query_executed": False,
        "target_metric_computed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--execution-authority", required=True)
    parser.add_argument("--expected-execution-authority-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", required=True)
    print(json.dumps(materialize(parser.parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
