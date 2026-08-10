#!/usr/bin/env python3
"""Select the frozen source arm and materialize a query-free target descriptor."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from radio_gs.interfaces import factorized_native_gauge_state_readout as readout
from radio_gs.interfaces import factorized_native_target_descriptor as formal
from radio_gs.interfaces.factorized_primitive_state import (
    FACTORIZED_PRIMITIVE_STATE_SCHEMA,
    FACTORIZED_PRIMITIVE_STATE_SCHEMA_VERSION,
    FactorizedPrimitiveState,
    load_factorized_primitive_state,
)
from radio_gs.interfaces.surface_region_target_accepted_v2 import (
    validate_target_accepted_v2_authority,
)
from radio_gs.models.factorized_native_gauge_state_readout import (
    FACTORIZED_NATIVE_READOUT_ARMS,
    FactorizedNativeGaugeStateReadout,
)
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.scripts import materialize_full_scalar_clean_training_shard as shard
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_torch_mapping,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


def apply_factorized_native_canonical_forward(
    *,
    accepted_v2_e0: torch.Tensor,
    region_rows: torch.Tensor,
    token_mask: torch.Tensor,
    anchor_index: torch.Tensor,
    state: FactorizedPrimitiveState,
    model: FactorizedNativeGaugeStateReadout,
    head: torch.nn.Module,
    device: str | torch.device = "cpu",
    batch_size: int = 256,
) -> dict[str, torch.Tensor | bool]:
    """Apply the native readout only where the anchor has exact state."""

    base = torch.as_tensor(accepted_v2_e0).detach().float().cpu().contiguous()
    rows = torch.as_tensor(region_rows).detach().long().cpu().contiguous()
    mask = torch.as_tensor(token_mask).detach().bool().cpu().contiguous()
    anchor = torch.as_tensor(anchor_index).detach().long().cpu().contiguous()
    regions = int(base.shape[0]) if base.ndim == 2 else -1
    if (
        regions <= 0
        or base.shape != (regions, shard.trainer.DESCRIPTOR_DIM)
        or rows.ndim != 2
        or rows.shape[0] != regions
        or mask.shape != rows.shape
        or anchor.shape != (regions,)
        or bool((anchor < 0).any())
        or bool((anchor >= rows.shape[1]).any())
        or not bool(mask[torch.arange(regions), anchor].all())
        or int(batch_size) <= 0
    ):
        raise ValueError("factorized-native target forward layout differs")
    norms = torch.linalg.vector_norm(base, dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), rtol=0.0, atol=2e-4):
        raise ValueError("factorized-native target base is not unit L2")
    anchor_rows = rows[torch.arange(regions), anchor]
    if bool((anchor_rows < 0).any()) or bool((anchor_rows >= state.valid.numel()).any()):
        raise ValueError("factorized-native target anchor row differs")
    exact = state.valid[anchor_rows].detach().bool().cpu().contiguous()
    eligible = torch.where(exact)[0]
    output = base.clone()
    runtime = torch.device(device)
    model = model.to(runtime).eval().requires_grad_(False)
    head = head.to(runtime).eval().requires_grad_(False)
    with torch.inference_mode():
        for start in range(0, int(eligible.numel()), int(batch_size)):
            selected = eligible[start : start + int(batch_size)]
            inputs = readout.gather_factorized_native_region_inputs(
                state,
                rows[selected],
                mask[selected],
                anchor[selected],
            )
            summary = model(
                inputs.unit_direction.to(runtime),
                inputs.log_amplitude.to(runtime),
                inputs.state.to(runtime),
                inputs.state_known_mask.to(runtime),
                token_mask=inputs.token_mask.to(runtime),
                anchor_index=inputs.anchor_index.to(runtime),
            )
            projected = head(summary[:, None])[:, 0].float()
            if projected.shape != (selected.numel(), shard.trainer.DESCRIPTOR_DIM):
                raise ValueError("official summary head output layout differs")
            projected = F.normalize(projected, dim=-1)
            if not bool(torch.isfinite(projected).all()):
                raise ValueError("factorized-native projected descriptor is non-finite")
            output[selected] = projected.cpu()
    fallback = ~exact
    changed = (output != base).any(dim=-1)
    fallback_equal = torch.equal(output[fallback], base[fallback])
    if not fallback_equal or bool((changed & fallback).any()):
        raise RuntimeError("factorized-native immutable fallback changed")
    return {
        "semantic_descriptor": output.contiguous(),
        "exact_state_anchor_mask": exact,
        "active_update_mask": exact.clone(),
        "immutable_fallback_mask": fallback.contiguous(),
        "descriptor_changed_mask": changed.contiguous(),
        "fallback_bitwise_equal": fallback_equal,
    }


def _load_target_inputs(execution: Mapping[str, Any]) -> dict[str, Any]:
    records = execution["target_inputs"]
    accepted_raw, _, _ = load_torch_mapping(
        records["target_accepted_v2"]["path"],
        expected_sha256=records["target_accepted_v2"]["sha256"],
        map_location="cpu",
        label="factorized-native target AcceptedV2",
    )
    accepted = validate_target_accepted_v2_authority(accepted_raw)
    state = load_factorized_primitive_state(
        records["factorized_primitive_state"]["path"],
        expected_sha256=records["factorized_primitive_state"]["sha256"],
    )
    geometry = accepted["input_authority"]["geometry_authority"]
    if (
        geometry["factorized_primitive_state_file_sha256"] != state.sha256
        or state.schema != FACTORIZED_PRIMITIVE_STATE_SCHEMA
        or state.schema_version != FACTORIZED_PRIMITIVE_STATE_SCHEMA_VERSION
        or state.metadata.get("geometry_fingerprint")
        != accepted["geometry_fingerprint"]
        or state.valid.shape != accepted["accepted_base_valid"].shape
        or state.metadata.get("query_independent") is not True
        or any(
            state.metadata.get(key) is not False
            for key in (
                "benchmark_images_opened",
                "benchmark_masks_opened",
                "text_queries_opened",
            )
        )
    ):
        raise ValueError("factorized-native target Accepted/state alignment differs")
    return {"records": records, "accepted": accepted, "state": state}


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuses to clobber target descriptor: {output}")
    execution = formal.validate_target_execution_authority(
        args.target_execution_authority,
        expected_sha256=args.expected_target_execution_authority_sha256,
        expected_output=output,
    )
    target = _load_target_inputs(execution)
    source = execution["verified_source_gate"]
    accepted = target["accepted"]
    state = target["state"]
    model = readout.build_model(source["winner_arm"], source["winner_normalization"])
    model.load_state_dict(source["winner_checkpoint"]["model_state_dict"], strict=True)
    head = SigLIP2SummaryHead.from_radio_checkpoint(
        source["official_radio_checkpoint"]["path"],
        expected_sha256=source["official_radio_checkpoint"]["sha256"],
    )
    forward = apply_factorized_native_canonical_forward(
        accepted_v2_e0=accepted["accepted_v2_e0"],
        region_rows=accepted["region_rows"],
        token_mask=accepted["token_mask"],
        anchor_index=accepted["anchor_index"],
        state=state,
        model=model,
        head=head,
        device=args.device,
        batch_size=int(args.batch_size),
    )
    region_ids = [
        shard.stable_region_id(accepted["scene_id"], fingerprint)
        for fingerprint in accepted["region_fingerprints"]
    ]
    masks = {
        name: forward[name]
        for name in (
            "exact_state_anchor_mask",
            "active_update_mask",
            "immutable_fallback_mask",
            "descriptor_changed_mask",
        )
    }
    winner_result = source["winner_result"]
    payload: dict[str, Any] = {
        "schema": formal.TARGET_DESCRIPTOR_SCHEMA,
        "schema_version": 1,
        "contract": formal.target_descriptor_contract(),
        "contract_sha256": formal.TARGET_DESCRIPTOR_CONTRACT_SHA256,
        "scene_id": accepted["scene_id"],
        "physical_space_id": accepted["physical_space_id"],
        "physical_space_authority": dict(accepted["physical_space_authority"]),
        "producer": file_record(Path(__file__).resolve()),
        "target_execution_authority": dict(execution["verified_record"]),
        "input_authority": {
            "source_arm_results": dict(execution["source_arm_results"]),
            "winner_checkpoint": dict(winner_result["checkpoint"]),
            "winner_normalization": dict(winner_result["normalization"]),
            "official_radio_checkpoint": dict(source["official_radio_checkpoint"]),
            **dict(target["records"]),
        },
        "winner_arm": source["winner_arm"],
        "winner_selected_step": winner_result["selected_step"],
        "winner_source_ranking": dict(source["ranking"]),
        "region_row_ids": region_ids,
        "canonical_region_indices": accepted["canonical_region_indices"].clone(),
        "region_fingerprints": list(accepted["region_fingerprints"]),
        "semantic_descriptor": forward["semantic_descriptor"],
        **masks,
        "fallback_bitwise_equal": forward["fallback_bitwise_equal"],
        "routing_audit": {
            "regions": len(region_ids),
            "exact_state_anchor": int(masks["exact_state_anchor_mask"].sum()),
            "active_update": int(masks["active_update_mask"].sum()),
            "immutable_fallback": int(masks["immutable_fallback_mask"].sum()),
            "descriptor_changed": int(masks["descriptor_changed_mask"].sum()),
        },
        "access_audit": formal.target_descriptor_access_audit(),
    }
    payload["channel_sha256"] = formal.target_descriptor_channel_sha256(payload)
    payload = formal.validate_target_descriptor_authority(payload)
    write_torch_noclobber(output, payload)
    return {
        "status": "materialized_query_free_factorized_native_target_descriptor",
        "scene_id": payload["scene_id"],
        "winner_arm": payload["winner_arm"],
        "winner_selected_step": payload["winner_selected_step"],
        "regions": len(region_ids),
        "routing_audit": payload["routing_audit"],
        "output": file_record(output),
        "exact_query_descriptor_view_schema": (
            formal.EXACT_QUERY_DESCRIPTOR_VIEW_SCHEMA
        ),
        "access_audit": formal.target_descriptor_access_audit(),
    }


def _bound_record(path: str, digest: str, *, label: str) -> dict[str, str]:
    record = {"path": str(Path(path).expanduser().resolve()), "sha256": str(digest)}
    validate_file_record(record, label=label)
    return record


def build_authority(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.authority_output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuses to clobber target authority: {output}")
    source_records = {
        arm: _bound_record(
            getattr(args, f"{arm}_result"),
            getattr(args, f"{arm}_result_sha256"),
            label=f"factorized-native {arm} result",
        )
        for arm in FACTORIZED_NATIVE_READOUT_ARMS
    }
    # Fail closed on all source arms before binding/opening target artifacts.
    winner = formal.validate_source_arm_winner(source_records)
    target_inputs = {
        "target_accepted_v2": _bound_record(
            args.target_accepted_v2,
            args.target_accepted_v2_sha256,
            label="target AcceptedV2",
        ),
        "factorized_primitive_state": _bound_record(
            args.factorized_primitive_state,
            args.factorized_primitive_state_sha256,
            label="target factorized primitive state",
        ),
    }
    descriptor_output = str(Path(args.descriptor_output).expanduser().resolve())
    authority = {
        "schema": formal.TARGET_EXECUTION_SCHEMA,
        "schema_version": 1,
        "status": "authorized_after_three_arm_source_selection_for_query_free_target",
        "source_arm_results": source_records,
        "implementation": file_record(Path(__file__).resolve()),
        "implementation_dependencies": {
            name: file_record(path)
            for name, path in formal.TARGET_IMPLEMENTATION_DEPENDENCIES.items()
        },
        "target_inputs": target_inputs,
        "target_descriptor_output": descriptor_output,
        "materialization_authorized": True,
        "query_execution_authorized": False,
        "metric_execution_authorized": False,
        "access_audit": formal.target_descriptor_access_audit(),
    }
    write_frozen_json(output, authority)
    return {
        "status": "factorized_native_target_authority_built",
        "winner_arm": winner["winner_arm"],
        "winner_selected_step": winner["winner_result"]["selected_step"],
        "authority": file_record(output),
        "descriptor_output": descriptor_output,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-authority")
    for arm in FACTORIZED_NATIVE_READOUT_ARMS:
        option = arm.replace("_", "-")
        build.add_argument(f"--{option}-result", required=True)
        build.add_argument(f"--{option}-result-sha256", required=True)
    build.add_argument("--target-accepted-v2", required=True)
    build.add_argument("--target-accepted-v2-sha256", required=True)
    build.add_argument("--factorized-primitive-state", required=True)
    build.add_argument("--factorized-primitive-state-sha256", required=True)
    build.add_argument("--descriptor-output", required=True)
    build.add_argument("--authority-output", required=True)
    validate = commands.add_parser("validate-authority")
    validate.add_argument("--target-execution-authority", required=True)
    validate.add_argument("--expected-target-execution-authority-sha256", required=True)
    run = commands.add_parser("materialize")
    run.add_argument("--target-execution-authority", required=True)
    run.add_argument("--expected-target-execution-authority-sha256", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--device", default="cpu")
    run.add_argument("--batch-size", type=int, default=256)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "build-authority":
        result = build_authority(args)
    elif args.command == "validate-authority":
        authority = formal.validate_target_execution_authority(
            args.target_execution_authority,
            expected_sha256=args.expected_target_execution_authority_sha256,
        )
        result = {
            "status": "factorized_native_target_authority_validated",
            "winner_arm": authority["verified_source_gate"]["winner_arm"],
            "target_files_opened_after_source_gate": True,
        }
    else:
        result = materialize(args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "apply_factorized_native_canonical_forward",
    "build_authority",
    "build_parser",
    "materialize",
]
