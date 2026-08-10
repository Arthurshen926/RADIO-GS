#!/usr/bin/env python3
"""Append factorized-native relations to one frozen source V2 pair axis."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces.factorized_native_gauge_state_readout import (
    FactorizedNativeRegionInputs,
    factorized_state_known_mask,
)
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
from radio_gs.models.region_comembership_native_v3 import PAIR_FEATURE_NAMES
from radio_gs.models.region_comembership_v2 import (
    PAIR_FEATURE_NAMES as V2_PAIR_FEATURE_NAMES,
)
from radio_gs.scripts import materialize_full_scalar_clean_training_shard as shard
from radio_gs.scripts.materialize_source_region_comembership_v2 import (
    densify_primitive_instance_mass,
    source_access as v2_source_access,
    validate_source_region_comembership_v2,
)
from radio_gs.scripts.train_source_region_comembership_v1 import (
    TRAIN_SCENES,
    VALIDATION_SCENES,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_torch_mapping,
    validate_file_record,
    write_torch_noclobber,
)


SCHEMA = "radio_gs.source_region_comembership_authority.native_v3"
SCHEMA_VERSION = 1
TENSOR_CHANNELS = (
    "canonical_region_indices",
    "region_rows",
    "token_mask",
    "dominant_instance_ids",
    "instance_purity",
    "instance_label_coverage",
    "instance_observed",
    "pair_indices",
    "v2_pair_features",
    "native_pair_features",
    "pair_features",
    "same_instance_targets",
    "pair_evidence_weights",
    "primitive_instance_flat_keys",
    "primitive_instance_mass",
)
PRIMITIVE_METRIC_CONTRACT = {
    "storage": "sorted_sparse_flat_primitive_times_instance_column",
    "instance_zero": "included_as_selected_contamination_never_target",
    "selected_union": "deduplicate_primitive_rows_before_mass_sum",
}


@dataclass(frozen=True)
class _FactorizedStateIndex:
    """One scene-local index reused by every bounded region batch.

    Indexing before converting the 1280-D direction to float32 prevents an
    accidental full-scene float32 copy for every batch.  This is a materializer
    implementation detail and does not expose a reconstructed RADIO vector.
    """

    global_to_compact: torch.Tensor
    scalar_state: torch.Tensor
    state_known: torch.Tensor


def _factorized_state_index(state: FactorizedPrimitiveState) -> _FactorizedStateIndex:
    global_to_compact = torch.full((state.valid.numel(),), -1, dtype=torch.int64)
    global_to_compact[state.global_rows] = torch.arange(
        state.global_rows.numel(), dtype=torch.int64
    )
    return _FactorizedStateIndex(
        global_to_compact=global_to_compact,
        scalar_state=state.scalar_encoding_input().detach().float().cpu().contiguous(),
        state_known=factorized_state_known_mask(state).contiguous(),
    )


def _gather_indexed_factorized_native_region_inputs(
    state: FactorizedPrimitiveState,
    index: _FactorizedStateIndex,
    region_rows: torch.Tensor,
    token_mask: torch.Tensor,
    anchor_index: torch.Tensor,
) -> FactorizedNativeRegionInputs:
    rows = torch.as_tensor(region_rows).detach().long().cpu()
    declared = torch.as_tensor(token_mask).detach().bool().cpu()
    anchor = torch.as_tensor(anchor_index).detach().long().cpu()
    batch = torch.arange(rows.shape[0]) if rows.ndim == 2 else torch.empty(0)
    if (
        rows.ndim != 2
        or declared.shape != rows.shape
        or anchor.shape != (rows.shape[0],)
        or not bool(declared.any(dim=1).all())
        or bool(rows[~declared].ne(-1).any())
        or bool((rows[declared] < 0).any())
        or bool((rows[declared] >= state.valid.numel()).any())
        or bool((anchor < 0).any())
        or bool((anchor >= rows.shape[1]).any())
        or not bool(declared[batch, anchor].all())
    ):
        raise ValueError("native V3 indexed region rows/mask differ")
    safe_rows = rows.clamp_min(0)
    exact_mask = declared & state.valid[safe_rows]
    if not bool(exact_mask[batch, anchor].all()):
        raise ValueError("native V3 indexed anchor lacks exact factorized state")
    compact = index.global_to_compact[safe_rows]
    if bool((compact[exact_mask] < 0).any()):
        raise RuntimeError("native V3 indexed compact-row map is incomplete")
    safe_compact = compact.clamp_min(0)
    # Index the stored direction first.  Calling ``state.semantic_direction.float()``
    # here would materialize the entire scene in float32 for every region batch.
    direction = state.semantic_direction[safe_compact].detach().float().cpu()
    scalars = index.scalar_state[safe_compact]
    known = index.state_known[safe_compact]
    direction = direction.masked_fill(~exact_mask[..., None], 0.0)
    scalars = scalars.masked_fill(~exact_mask[..., None], 0.0)
    known = known & exact_mask[..., None]
    return FactorizedNativeRegionInputs(
        unit_direction=direction.contiguous(),
        log_amplitude=scalars[..., 0].clone().contiguous(),
        state=scalars.contiguous(),
        state_known_mask=known.contiguous(),
        token_mask=exact_mask.contiguous(),
        anchor_index=anchor.contiguous(),
    )


def source_access() -> dict[str, bool]:
    return {
        **v2_source_access(),
        "factorized_native_state_opened": True,
        "raw_radio_vector_reconstructed": False,
        "target_pair_features_opened": False,
    }


def _identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    names = (
        "schema",
        "schema_version",
        "scene_id",
        "split",
        "producer",
        "input_authority",
        "pair_feature_names",
        "pair_feature_names_sha256",
        "native_feature_names",
        "native_feature_names_sha256",
        "native_interface_contract_sha256",
        "primitive_metric_contract",
        "source_access",
    )
    return {name: payload[name] for name in names}


def validate_source_region_comembership_native_v3(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("source native V3 authority must be a mapping")
    payload = dict(value)
    required = {
        *_identity_keys(),
        "content_authority_sha256",
        "region_fingerprints",
        *TENSOR_CHANNELS,
        "primitive_count",
        "instance_columns_including_zero",
        "channel_sha256",
        "audit",
    }
    if (
        set(payload) != required
        or payload.get("schema") != SCHEMA
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("pair_feature_names") != list(PAIR_FEATURE_NAMES)
        or payload.get("pair_feature_names_sha256")
        != canonical_json_sha256(list(PAIR_FEATURE_NAMES))
        or payload.get("native_feature_names") != list(NATIVE_FEATURE_NAMES)
        or payload.get("native_feature_names_sha256")
        != NATIVE_FEATURE_NAMES_SHA256
        or payload.get("native_interface_contract_sha256")
        != NATIVE_INTERFACE_CONTRACT_SHA256
        or payload.get("scene_id") not in set(TRAIN_SCENES) | set(VALIDATION_SCENES)
        or payload.get("split")
        != (
            "source_train"
            if payload.get("scene_id") in TRAIN_SCENES
            else "source_validation"
        )
        or payload.get("primitive_metric_contract") != PRIMITIVE_METRIC_CONTRACT
        or payload.get("source_access") != source_access()
        or payload.get("content_authority_sha256")
        != canonical_json_sha256(_identity(payload))
    ):
        raise ValueError("source native V3 authority header differs")
    validate_file_record(payload["producer"], label="native V3 producer")
    inputs = payload.get("input_authority")
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "parent_v2_source_authority",
        "accepted_v2",
        "factorized_state",
    }:
        raise ValueError("source native V3 input authority differs")
    for name, record in inputs.items():
        validate_file_record(record, label=f"native V3 {name}")
    canonical = torch.as_tensor(payload["canonical_region_indices"])
    rows = torch.as_tensor(payload["region_rows"])
    mask = torch.as_tensor(payload["token_mask"])
    pairs = torch.as_tensor(payload["pair_indices"])
    v2_features = torch.as_tensor(payload["v2_pair_features"])
    native = torch.as_tensor(payload["native_pair_features"])
    features = torch.as_tensor(payload["pair_features"])
    targets = torch.as_tensor(payload["same_instance_targets"])
    weights = torch.as_tensor(payload["pair_evidence_weights"])
    dominant = torch.as_tensor(payload["dominant_instance_ids"])
    purity = torch.as_tensor(payload["instance_purity"])
    coverage = torch.as_tensor(payload["instance_label_coverage"])
    observed = torch.as_tensor(payload["instance_observed"])
    count = int(canonical.numel()) if canonical.ndim == 1 else -1
    primitive_count = int(payload["primitive_count"])
    columns = int(payload["instance_columns_including_zero"])
    if (
        count <= 0
        or canonical.dtype != torch.int64
        or rows.dtype != torch.int64
        or rows.ndim != 2
        or rows.shape[0] != count
        or mask.dtype != torch.bool
        or mask.shape != rows.shape
        or not bool(mask.any(dim=1).all())
        or bool(rows[~mask].ne(-1).any())
        or bool((rows[mask] < 0).any())
        or bool((rows[mask] >= primitive_count).any())
        or pairs.dtype != torch.int64
        or pairs.ndim != 2
        or pairs.shape[0] != 2
        or bool((pairs < 0).any())
        or bool((pairs >= count).any())
        or bool((pairs[0] >= pairs[1]).any())
        or v2_features.dtype != torch.float32
        or v2_features.shape != (pairs.shape[1], len(V2_PAIR_FEATURE_NAMES))
        or native.dtype != torch.float32
        or native.shape != (pairs.shape[1], len(NATIVE_FEATURE_NAMES))
        or features.dtype != torch.float32
        or features.shape != (pairs.shape[1], len(PAIR_FEATURE_NAMES))
        or not torch.equal(features[:, : len(V2_PAIR_FEATURE_NAMES)], v2_features)
        or not torch.equal(features[:, len(V2_PAIR_FEATURE_NAMES) :], native)
        or not bool(torch.isfinite(features).all())
        or targets.dtype != torch.bool
        or targets.shape != (pairs.shape[1],)
        or weights.dtype != torch.float32
        or weights.shape != targets.shape
        or not bool(torch.isfinite(weights).all())
        or bool((weights < 0).any())
        or dominant.dtype != torch.int64
        or dominant.shape != (count,)
        or purity.dtype != torch.float32
        or purity.shape != (count,)
        or coverage.dtype != torch.float32
        or coverage.shape != (count,)
        or observed.dtype != torch.bool
        or observed.shape != (count,)
        or not bool(torch.isfinite(purity).all())
        or not bool(torch.isfinite(coverage).all())
        or bool((purity < 0).any())
        or bool((purity > 1.0001).any())
        or bool((coverage < 0).any())
        or bool((coverage > 1.0001).any())
        or len(payload["region_fingerprints"]) != count
        or len(set(payload["region_fingerprints"])) != count
    ):
        raise ValueError("source native V3 tensor axes differ")
    pair_keys = pairs[0] * count + pairs[1]
    if pair_keys.numel() <= 0 or (
        pair_keys.numel() > 1 and not bool((pair_keys[1:] > pair_keys[:-1]).all())
    ):
        raise ValueError("source native V3 pair keys are not sorted unique")
    densify_primitive_instance_mass(
        flat_keys=payload["primitive_instance_flat_keys"],
        mass=payload["primitive_instance_mass"],
        primitive_count=primitive_count,
        instance_columns_including_zero=columns,
    )
    channels = payload.get("channel_sha256")
    if not isinstance(channels, Mapping) or set(channels) != set(TENSOR_CHANNELS):
        raise ValueError("source native V3 channel authority differs")
    for name in TENSOR_CHANNELS:
        if channels[name] != tensor_sha256(payload[name]):
            raise ValueError(f"source native V3 channel changed: {name}")
    return payload


def _identity_keys() -> tuple[str, ...]:
    return (
        "schema",
        "schema_version",
        "scene_id",
        "split",
        "producer",
        "input_authority",
        "pair_feature_names",
        "pair_feature_names_sha256",
        "native_feature_names",
        "native_feature_names_sha256",
        "native_interface_contract_sha256",
        "primitive_metric_contract",
        "source_access",
    )


def _concatenate_summaries(
    values: list[FactorizedNativeRegionSummary],
) -> FactorizedNativeRegionSummary:
    if not values:
        raise ValueError("native V3 summary list is empty")
    fields = FactorizedNativeRegionSummary.__dataclass_fields__
    return FactorizedNativeRegionSummary(
        **{
            name: torch.cat([getattr(value, name) for value in values], dim=0)
            .float()
            .contiguous()
            for name in fields
        }
    )


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"source native V3 authority exists: {output}")
    parent_raw, parent_sha, parent_path = load_torch_mapping(
        args.v2_source_authority,
        expected_sha256=args.expected_v2_source_authority_sha256,
        map_location="cpu",
        label="native V3 parent source V2 authority",
    )
    parent = validate_source_region_comembership_v2(parent_raw)
    v1_record = parent["input_authority"]["v1_source_authority"]
    v1_raw, _, _ = load_torch_mapping(
        v1_record["path"],
        expected_sha256=v1_record["sha256"],
        map_location="cpu",
        label="native V3 source V1 lineage",
    )
    accepted_record = dict(v1_raw["input_authority"]["accepted_v2"])
    state_record = dict(v1_raw["input_authority"]["factorized_state"])
    accepted_raw, accepted_sha, accepted_path = load_torch_mapping(
        accepted_record["path"],
        expected_sha256=accepted_record["sha256"],
        map_location="cpu",
        label="native V3 AcceptedV2 authority",
    )
    accepted = shard.validate_accepted_region_authority(accepted_raw)
    state = load_factorized_primitive_state(
        state_record["path"], expected_sha256=state_record["sha256"]
    )
    if (
        accepted["scene_id"] != parent["scene_id"]
        or accepted["region_fingerprints"] != parent["region_fingerprints"]
        or not torch.equal(
            accepted["canonical_region_indices"], parent["canonical_region_indices"]
        )
        or not torch.equal(accepted["region_rows"], parent["region_rows"])
        or not torch.equal(accepted["token_mask"], parent["token_mask"])
        or accepted["input_authority"]["geometry_authority"][
            "factorized_primitive_state_file_sha256"
        ]
        != state.sha256
    ):
        raise ValueError("native V3 parent/AcceptedV2/state axes differ")
    batch_size = int(args.batch_size)
    if batch_size <= 1:
        raise ValueError("native V3 batch_size must exceed one")
    summaries: list[FactorizedNativeRegionSummary] = []
    state_index = _factorized_state_index(state)
    for start in range(0, parent["region_rows"].shape[0], batch_size):
        stop = min(start + batch_size, parent["region_rows"].shape[0])
        gathered = _gather_indexed_factorized_native_region_inputs(
            state,
            state_index,
            parent["region_rows"][start:stop],
            parent["token_mask"][start:stop],
            accepted["anchor_index"][start:stop],
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
    native = factorized_native_pair_features(summary, parent["pair_indices"])
    v2_features = parent["pair_features"].float().cpu().contiguous()
    combined = torch.cat((v2_features, native), dim=1).float().contiguous()
    identity = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "scene_id": parent["scene_id"],
        "split": parent["split"],
        "producer": file_record(Path(__file__).resolve()),
        "input_authority": {
            "parent_v2_source_authority": {
                "path": str(parent_path),
                "sha256": parent_sha,
            },
            "accepted_v2": {"path": str(accepted_path), "sha256": accepted_sha},
            "factorized_state": {
                "path": str(state.source),
                "sha256": state.sha256,
            },
        },
        "pair_feature_names": list(PAIR_FEATURE_NAMES),
        "pair_feature_names_sha256": canonical_json_sha256(
            list(PAIR_FEATURE_NAMES)
        ),
        "native_feature_names": list(NATIVE_FEATURE_NAMES),
        "native_feature_names_sha256": NATIVE_FEATURE_NAMES_SHA256,
        "native_interface_contract_sha256": NATIVE_INTERFACE_CONTRACT_SHA256,
        "primitive_metric_contract": dict(parent["primitive_metric_contract"]),
        "source_access": source_access(),
    }
    copied = {
        name: parent[name]
        for name in (
            "canonical_region_indices",
            "region_rows",
            "token_mask",
            "dominant_instance_ids",
            "instance_purity",
            "instance_label_coverage",
            "instance_observed",
            "pair_indices",
            "same_instance_targets",
            "pair_evidence_weights",
            "primitive_instance_flat_keys",
            "primitive_instance_mass",
        )
    }
    payload = {
        **identity,
        "content_authority_sha256": canonical_json_sha256(identity),
        "region_fingerprints": list(parent["region_fingerprints"]),
        **copied,
        "v2_pair_features": v2_features,
        "native_pair_features": native,
        "pair_features": combined,
        "primitive_count": int(parent["primitive_count"]),
        "instance_columns_including_zero": int(
            parent["instance_columns_including_zero"]
        ),
        "channel_sha256": {},
        "audit": {
            "regions": int(parent["canonical_region_indices"].numel()),
            "candidate_pairs": int(parent["pair_indices"].shape[1]),
            "v2_pair_feature_dimension": len(V2_PAIR_FEATURE_NAMES),
            "native_pair_feature_dimension": len(NATIVE_FEATURE_NAMES),
            "combined_pair_feature_dimension": len(PAIR_FEATURE_NAMES),
            "region_summary_batch_size": batch_size,
            "raw_radio_vector_reconstructed": False,
        },
    }
    payload["channel_sha256"] = {
        name: tensor_sha256(payload[name]) for name in TENSOR_CHANNELS
    }
    validate_source_region_comembership_native_v3(payload)
    written = write_torch_noclobber(output, payload)
    return {
        "status": "source_region_comembership_native_v3_complete",
        "scene_id": payload["scene_id"],
        "split": payload["split"],
        "output": file_record(written),
        "audit": payload["audit"],
        "target_metric_computed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-source-authority", required=True)
    parser.add_argument("--expected-v2-source-authority-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", required=True)
    print(json.dumps(materialize(parser.parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
