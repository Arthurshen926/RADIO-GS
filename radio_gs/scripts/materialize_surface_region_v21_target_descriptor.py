#!/usr/bin/env python3
"""Materialize query-free V2.1 descriptors under a promoted source gate."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.interfaces.factorized_primitive_state import (
    FactorizedPrimitiveState,
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
from radio_gs.interfaces.surface_region_v21_source_gate import (
    validate_checkpoint_payload,
    validate_normalization_authority,
)
from radio_gs.interfaces.surface_region_v21_target import (
    TARGET_DESCRIPTOR_CONTRACT_SHA256,
    TARGET_DESCRIPTOR_SCHEMA,
    target_descriptor_access_audit,
    target_descriptor_channel_sha256,
    target_descriptor_contract,
    validate_target_descriptor_authority,
    validate_target_execution_authority,
)
from radio_gs.models.surface_region_typed_context_residual import (
    SurfaceRegionAcceptedV2TypedContextResidualV1,
)
from radio_gs.scripts import (
    train_surface_region_typed_context_response_listwise_v21_pilot as pilot,
)
from radio_gs.scripts import materialize_full_scalar_clean_training_shard as shard
from radio_gs.scripts import train_surface_region_typed_context_residual as v1_trainer
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_torch_mapping,
    write_torch_noclobber,
)


def apply_v21_canonical_forward(
    *,
    accepted_v2_e0: torch.Tensor,
    pooled_context_radio_direction: torch.Tensor,
    raw_full_scalar_summary: torch.Tensor,
    typed_context_statistics: torch.Tensor,
    full_scalar_eligible: torch.Tensor,
    typed_context_valid: torch.Tensor,
    model: SurfaceRegionAcceptedV2TypedContextResidualV1,
    normalization: Mapping[str, Any],
    batch_size: int = 256,
) -> dict[str, torch.Tensor | bool]:
    """Apply the complete canonical V2.1 forward without query relevance."""

    base = torch.as_tensor(accepted_v2_e0).detach().float().cpu().contiguous()
    context = (
        torch.as_tensor(pooled_context_radio_direction)
        .detach()
        .float()
        .cpu()
        .contiguous()
    )
    full_scalar = (
        torch.as_tensor(raw_full_scalar_summary).detach().float().cpu().contiguous()
    )
    statistics = (
        torch.as_tensor(typed_context_statistics).detach().float().cpu().contiguous()
    )
    eligible = torch.as_tensor(full_scalar_eligible).detach().bool().cpu().contiguous()
    typed = torch.as_tensor(typed_context_valid).detach().bool().cpu().contiguous()
    regions = int(base.shape[0]) if base.ndim == 2 else -1
    if (
        regions <= 0
        or base.shape != (regions, 1536)
        or context.shape != (regions, 1280)
        or full_scalar.shape != (regions, 18)
        or statistics.shape != (regions, 12)
        or eligible.shape != (regions,)
        or typed.shape != (regions,)
        or not bool(torch.isfinite(base).all())
        or not bool(torch.isfinite(context).all())
        or not bool(torch.isfinite(full_scalar).all())
        or not bool(torch.isfinite(statistics).all())
        or int(batch_size) <= 0
    ):
        raise ValueError("V2.1 canonical forward inputs differ")
    base_norm = torch.linalg.vector_norm(base, dim=-1)
    if not torch.allclose(base_norm, torch.ones_like(base_norm), rtol=0.0, atol=2e-4):
        raise ValueError("V2.1 canonical base descriptor is not unit L2")
    if bool(context[~typed].count_nonzero()) or bool(
        statistics[~typed].count_nonzero()
    ):
        raise ValueError("V2.1 inactive typed-context carrier is not exact zero")
    scene = {
        "raw_full_scalar_summary": full_scalar,
        "typed_context_statistics": statistics,
        "eligible": eligible,
        "typed_context_valid": typed,
    }
    declared, effective_ood, active = pilot._pilot_routing(scene, normalization)
    normalized_ood = effective_ood & eligible
    output = torch.empty_like(base)
    model = model.cpu().eval().requires_grad_(False)
    with torch.inference_mode():
        for start in range(0, regions, int(batch_size)):
            stop = min(start + int(batch_size), regions)
            output[start:stop] = model(
                base[start:stop],
                context[start:stop],
                full_scalar[start:stop],
                statistics[start:stop],
                active_mask=declared[start:stop],
                ood_mask=effective_ood[start:stop],
            )
    fallback = ~active
    fallback_equal = torch.equal(output[fallback], base[fallback])
    changed = (output != base).any(dim=-1)
    if not fallback_equal or bool((changed & fallback).any()):
        raise RuntimeError("V2.1 canonical immutable fallback changed")
    return {
        "semantic_descriptor": output.contiguous(),
        "full_scalar_eligible_mask": eligible,
        "typed_context_valid_mask": typed,
        "normalization_ood_mask": normalized_ood.contiguous(),
        "effective_ood_mask": effective_ood.contiguous(),
        "active_update_mask": active.contiguous(),
        "immutable_fallback_mask": fallback.contiguous(),
        "descriptor_changed_mask": changed.contiguous(),
        "fallback_bitwise_equal": fallback_equal,
    }


def _load_inputs(execution: Mapping[str, Any]) -> dict[str, Any]:
    records = execution["target_inputs"]
    accepted_raw, _, _ = load_torch_mapping(
        records["target_accepted_v2"]["path"],
        expected_sha256=records["target_accepted_v2"]["sha256"],
        map_location="cpu",
        label="target AcceptedV2 authority",
    )
    accepted = validate_target_accepted_v2_authority(accepted_raw)
    adaptive_raw, _, _ = load_torch_mapping(
        records["target_adaptive_typed_context"]["path"],
        expected_sha256=records["target_adaptive_typed_context"]["sha256"],
        map_location="cpu",
        label="target adaptive typed-context authority",
    )
    adaptive = validate_target_adaptive_typed_context_authority(adaptive_raw)
    state = load_factorized_primitive_state(
        records["factorized_primitive_state"]["path"],
        expected_sha256=records["factorized_primitive_state"]["sha256"],
    )
    normalization_raw, _, _ = load_torch_mapping(
        records["v21_normalization"]["path"],
        expected_sha256=records["v21_normalization"]["sha256"],
        map_location="cpu",
        label="V2.1 target normalization",
    )
    normalization = validate_normalization_authority(normalization_raw)
    checkpoint_raw, _, _ = load_torch_mapping(
        records["v21_checkpoint"]["path"],
        expected_sha256=records["v21_checkpoint"]["sha256"],
        map_location="cpu",
        label="V2.1 target checkpoint",
    )
    checkpoint = validate_checkpoint_payload(
        checkpoint_raw, normalization=normalization
    )
    return {
        "records": records,
        "accepted": accepted,
        "adaptive": adaptive,
        "state": state,
        "normalization": normalization,
        "checkpoint": checkpoint,
    }


def _validate_alignment(inputs: Mapping[str, Any]) -> None:
    accepted = inputs["accepted"]
    adaptive = inputs["adaptive"]
    state: FactorizedPrimitiveState = inputs["state"]
    records = inputs["records"]
    regions = int(accepted["canonical_region_indices"].numel())
    anchor_global = accepted["region_rows"][
        torch.arange(regions), accepted["anchor_index"]
    ]
    expected_ids = [
        shard.stable_region_id(accepted["scene_id"], fingerprint)
        for fingerprint in accepted["region_fingerprints"]
    ]
    adaptive_inputs = adaptive["input_authority"]
    accepted_geometry = accepted["input_authority"]["geometry_authority"]
    if (
        adaptive["scene_id"] != accepted["scene_id"]
        or adaptive["physical_space_id"] != accepted["physical_space_id"]
        or adaptive["physical_space_authority"] != accepted["physical_space_authority"]
        or adaptive["region_row_ids"] != expected_ids
        or not torch.equal(
            adaptive["canonical_region_indices"],
            accepted["canonical_region_indices"],
        )
        or not torch.equal(adaptive["scale_indices"], accepted["scale_indices"])
        or not torch.equal(adaptive["anchor_global_rows"], anchor_global)
        or adaptive_inputs["accepted_v2_canonical_region_authority"]
        != records["target_accepted_v2"]
        or adaptive_inputs["accepted_region_channel_sha256"]
        != canonical_json_sha256(accepted["channel_sha256"])
        or adaptive_inputs["accepted_region_fingerprints_sha256"]
        != canonical_json_sha256(accepted["region_fingerprints"])
        or adaptive_inputs["factorized_primitive_state"]
        != records["factorized_primitive_state"]
        or accepted_geometry["factorized_primitive_state_file_sha256"]
        != records["factorized_primitive_state"]["sha256"]
        or state.metadata["geometry_fingerprint"] != accepted["geometry_fingerprint"]
        or state.valid.shape != accepted["accepted_base_valid"].shape
    ):
        raise ValueError("V2.1 target Accepted/adaptive/state alignment differs")


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuses to clobber V2.1 target descriptor: {output}")
    execution = validate_target_execution_authority(
        args.target_execution_authority,
        expected_sha256=args.expected_target_execution_authority_sha256,
        expected_output=output,
    )
    inputs = _load_inputs(execution)
    _validate_alignment(inputs)
    accepted = inputs["accepted"]
    adaptive = inputs["adaptive"]
    state = inputs["state"]
    summary = aggregate_surface_region_full_scalars(
        state,
        accepted["accepted_base_valid"],
        accepted["region_rows"],
        accepted["token_mask"],
        accepted["anchor_index"],
    )
    normalization = inputs["normalization"]
    checkpoint = inputs["checkpoint"]
    model = SurfaceRegionAcceptedV2TypedContextResidualV1(
        scalar_median=normalization["median"],
        scalar_robust_scale=normalization["robust_scale"],
        max_angle_radians=v1_trainer.MAX_ANGLE_RADIANS,
        max_alpha=v1_trainer.MAX_ALPHA,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    forward = apply_v21_canonical_forward(
        accepted_v2_e0=accepted["accepted_v2_e0"],
        pooled_context_radio_direction=adaptive["pooled_context_radio_direction"],
        raw_full_scalar_summary=summary.summary,
        typed_context_statistics=adaptive["typed_context_statistics"],
        full_scalar_eligible=summary.use_full_scalar_mask,
        typed_context_valid=adaptive["typed_context_valid"],
        model=model,
        normalization=normalization,
        batch_size=int(args.batch_size),
    )
    masks = {
        name: forward[name]
        for name in (
            "full_scalar_eligible_mask",
            "typed_context_valid_mask",
            "normalization_ood_mask",
            "effective_ood_mask",
            "active_update_mask",
            "immutable_fallback_mask",
            "descriptor_changed_mask",
        )
    }
    payload: dict[str, Any] = {
        "schema": TARGET_DESCRIPTOR_SCHEMA,
        "schema_version": 1,
        "contract": target_descriptor_contract(),
        "contract_sha256": TARGET_DESCRIPTOR_CONTRACT_SHA256,
        "scene_id": accepted["scene_id"],
        "physical_space_id": accepted["physical_space_id"],
        "physical_space_authority": dict(accepted["physical_space_authority"]),
        "producer": file_record(Path(__file__).resolve()),
        "target_execution_authority": dict(execution["verified_record"]),
        "input_authority": dict(inputs["records"]),
        "region_row_ids": list(adaptive["region_row_ids"]),
        "canonical_region_indices": accepted["canonical_region_indices"].clone(),
        "region_fingerprints": list(accepted["region_fingerprints"]),
        "semantic_descriptor": forward["semantic_descriptor"],
        **masks,
        "fallback_bitwise_equal": forward["fallback_bitwise_equal"],
        "routing_audit": {
            "regions": len(adaptive["region_row_ids"]),
            "full_scalar_eligible": int(masks["full_scalar_eligible_mask"].sum()),
            "typed_context_valid": int(masks["typed_context_valid_mask"].sum()),
            "normalization_ood": int(masks["normalization_ood_mask"].sum()),
            "effective_ood": int(masks["effective_ood_mask"].sum()),
            "active_update": int(masks["active_update_mask"].sum()),
            "immutable_fallback": int(masks["immutable_fallback_mask"].sum()),
            "descriptor_changed": int(masks["descriptor_changed_mask"].sum()),
        },
        "access_audit": target_descriptor_access_audit(),
    }
    payload["channel_sha256"] = target_descriptor_channel_sha256(payload)
    payload = validate_target_descriptor_authority(payload)
    write_torch_noclobber(output, payload)
    return {
        "status": "materialized_query_free_v21_target_descriptor",
        "scene_id": payload["scene_id"],
        "regions": len(payload["region_row_ids"]),
        "routing_audit": payload["routing_audit"],
        "output": file_record(output),
        "access_audit": target_descriptor_access_audit(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-execution-authority", required=True)
    parser.add_argument("--expected-target-execution-authority-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser


def main() -> None:
    print(json.dumps(materialize(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
