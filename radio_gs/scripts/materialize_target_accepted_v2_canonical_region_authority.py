#!/usr/bin/env python3
"""Materialize query-free AcceptedV2 canonical regions for a target scene.

The mathematical preflight, sparse selection and descriptor computation are
the frozen source producer's pure cores.  Publication uses an independent
target schema with an explicit geometry-checkpoint physical-space authority;
it never invokes the source ScanNet identity policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from radio_gs.interfaces.surface_region_target_accepted_v2 import (
    TARGET_ACCEPTED_V2_CONTRACT_SHA256,
    TARGET_ACCEPTED_V2_SCHEMA,
    TARGET_ACCEPTED_V2_SCHEMA_VERSION,
    target_accepted_v2_contract,
    target_access_audit,
    target_physical_space_authority,
    validate_target_accepted_v2_authority,
)
from radio_gs.scripts import (
    materialize_accepted_v2_canonical_region_authority as source,
)
from radio_gs.scripts import materialize_full_scalar_clean_training_shard as shard
from radio_gs.utils.immutable_artifacts import file_record, write_torch_noclobber


def build_target_authority_payload(
    *,
    scene_id: str,
    dataset_id: str,
    geometry_checkpoint_sha256: str,
    physical_space_id: str,
    geometry_fingerprint: Mapping[str, Any],
    accepted_base_valid: torch.Tensor,
    canonical_region_indices: torch.Tensor,
    region_fingerprints: list[str],
    selection_audit: Mapping[str, Any],
    region_rows: torch.Tensor,
    token_mask: torch.Tensor,
    anchor_index: torch.Tensor,
    scale_indices: torch.Tensor,
    accepted_v2_e0: torch.Tensor,
    input_authority: Mapping[str, Any],
    producer: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    physical_authority = target_physical_space_authority(
        dataset_id=dataset_id,
        scene_id=scene_id,
        geometry_checkpoint_sha256=geometry_checkpoint_sha256,
    )
    if str(physical_space_id) != physical_authority["physical_space_id"]:
        raise ValueError("target AcceptedV2 physical-space ID differs from authority")
    payload: dict[str, Any] = {
        "schema": TARGET_ACCEPTED_V2_SCHEMA,
        "schema_version": TARGET_ACCEPTED_V2_SCHEMA_VERSION,
        "contract": target_accepted_v2_contract(),
        "contract_sha256": TARGET_ACCEPTED_V2_CONTRACT_SHA256,
        "scene_id": str(scene_id),
        "physical_space_id": str(physical_space_id),
        "physical_space_authority": physical_authority,
        "producer": dict(
            producer if producer is not None else file_record(Path(__file__).resolve())
        ),
        "accepted_v2_authority": shard.trainer._accepted_v2_authority(),
        "geometry_fingerprint": dict(geometry_fingerprint),
        "accepted_base_valid": torch.as_tensor(accepted_base_valid)
        .detach()
        .bool()
        .cpu()
        .contiguous(),
        "canonical_region_indices": torch.as_tensor(canonical_region_indices)
        .detach()
        .long()
        .cpu()
        .contiguous(),
        "region_fingerprints": list(region_fingerprints),
        "selection_audit": dict(selection_audit),
        "region_rows": torch.as_tensor(region_rows).detach().long().cpu().contiguous(),
        "token_mask": torch.as_tensor(token_mask).detach().bool().cpu().contiguous(),
        "anchor_index": torch.as_tensor(anchor_index)
        .detach()
        .long()
        .cpu()
        .contiguous(),
        "scale_indices": torch.as_tensor(scale_indices)
        .detach()
        .long()
        .cpu()
        .contiguous(),
        "accepted_v2_e0": torch.as_tensor(accepted_v2_e0)
        .detach()
        .float()
        .cpu()
        .contiguous(),
        "input_authority": dict(input_authority),
        "access_audit": target_access_audit(),
    }
    payload["channel_sha256"] = shard.accepted_region_channel_sha256(payload)
    return validate_target_accepted_v2_authority(payload)


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if not bool(getattr(args, "preflight_only", False)) and (
        output.exists() or output.is_symlink()
    ):
        raise FileExistsError(
            f"refuses to clobber target AcceptedV2 authority: {output}"
        )
    physical = target_physical_space_authority(
        dataset_id=args.dataset_id,
        scene_id=args.scene_id,
        geometry_checkpoint_sha256=args.geometry_checkpoint_sha256,
    )
    if str(args.physical_space_id) != physical["physical_space_id"]:
        raise ValueError("target AcceptedV2 CLI physical-space ID differs")
    runtime = source.preflight(args)
    selection = source._select_canonical_regions(
        runtime, batch_size=int(args.selection_batch_size)
    )
    summary = {
        "status": "ready",
        "schema": TARGET_ACCEPTED_V2_SCHEMA,
        "scene_id": str(args.scene_id),
        "physical_space_id": physical["physical_space_id"],
        "active_primitive_rows": int(runtime.state.global_rows.numel()),
        "region_scales": list(runtime.contract.radii_m),
        "selected_region_rows": int(selection["canonical_region_indices"].numel()),
        "selection_audit": dict(selection["selection_audit"]),
        "input_records": dict(runtime.input_records),
        "access_audit": target_access_audit(),
        "outputs_written": False,
    }
    if bool(getattr(args, "preflight_only", False)):
        return summary
    computed = source._compute_region_tensors(
        runtime,
        selection,
        batch_size=int(args.readout_batch_size),
        device=str(args.device),
    )
    payload = build_target_authority_payload(
        scene_id=str(args.scene_id),
        dataset_id=str(args.dataset_id),
        geometry_checkpoint_sha256=str(args.geometry_checkpoint_sha256),
        physical_space_id=str(args.physical_space_id),
        geometry_fingerprint=runtime.state.metadata["geometry_fingerprint"],
        accepted_base_valid=runtime.state.valid,
        input_authority=runtime.input_authority,
        **computed,
    )
    write_torch_noclobber(output, payload)
    return {
        **summary,
        "status": "materialized",
        "output": file_record(output),
        "outputs_written": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--geometry-checkpoint-sha256", required=True)
    parser.add_argument("--physical-space-id", required=True)
    parser.add_argument("--factorized-field-checkpoint", required=True)
    parser.add_argument("--expected-factorized-field-checkpoint-sha256", required=True)
    parser.add_argument("--factorized-primitive-state", required=True)
    parser.add_argument("--expected-factorized-primitive-state-sha256", required=True)
    parser.add_argument("--expected-factorized-radio-cache-sha256", required=True)
    parser.add_argument("--support-graph", required=True)
    parser.add_argument("--expected-support-graph-sha256", required=True)
    parser.add_argument("--exact-marginal-responsibility-authority", required=True)
    parser.add_argument(
        "--expected-exact-marginal-responsibility-authority-sha256",
        required=True,
    )
    parser.add_argument("--accepted-v2-checkpoint", required=True)
    parser.add_argument("--expected-accepted-v2-checkpoint-sha256", required=True)
    parser.add_argument("--official-radio-checkpoint", required=True)
    parser.add_argument("--expected-official-radio-checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selection-batch-size", type=int, default=8192)
    parser.add_argument("--readout-batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> None:
    print(json.dumps(materialize(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
