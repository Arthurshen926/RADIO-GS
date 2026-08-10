#!/usr/bin/env python3
"""Pool official canonical DINO/SAM capabilities onto AcceptedV2 regions."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces.capability_cache import load_canonical_capability_bank
from radio_gs.scripts import materialize_full_scalar_clean_training_shard as shard
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_torch_mapping,
    sha256_file,
    write_torch_noclobber,
)


SCHEMA = "radio_gs.region_capability_descriptor_authority.v2"
CHANNEL_NAMES = (
    "canonical_region_indices",
    "region_rows",
    "token_mask",
    "appearance_direction",
    "boundary_direction",
    "appearance_concentration",
    "boundary_concentration",
)


def source_access() -> dict[str, bool]:
    return {
        "source_instance_labels_opened": False,
        "benchmark_images_opened": False,
        "benchmark_labels_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_queries_opened": False,
        "target_metrics_computed": False,
    }


def pool_region_capability(
    *,
    compact_features: torch.Tensor,
    active_global_rows: torch.Tensor,
    region_rows: torch.Tensor,
    token_mask: torch.Tensor,
    batch_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unit pooled direction and mean-direction concentration per region."""

    features = torch.as_tensor(compact_features).detach().cpu()
    active = torch.as_tensor(active_global_rows).detach().long().cpu()
    rows = torch.as_tensor(region_rows).detach().long().cpu()
    mask = torch.as_tensor(token_mask).detach().bool().cpu()
    if (
        features.ndim != 2
        or active.ndim != 1
        or features.shape[0] != active.numel()
        or rows.ndim != 2
        or mask.shape != rows.shape
        or int(batch_size) <= 0
        or not bool((active[1:] > active[:-1]).all())
    ):
        raise ValueError("region capability pooling axes differ")
    full_count = int(active[-1]) + 1
    global_to_compact = torch.full((full_count,), -1, dtype=torch.long)
    global_to_compact[active] = torch.arange(active.numel())
    valid_rows = rows[mask]
    if (
        valid_rows.numel() == 0
        or bool((valid_rows < 0).any())
        or int(valid_rows.max()) >= full_count
        or bool((global_to_compact[valid_rows] < 0).any())
        or not bool(torch.isfinite(features).all())
    ):
        raise ValueError("region rows are outside the canonical capability authority")
    compact_rows = global_to_compact[rows.clamp_min(0)]
    directions: list[torch.Tensor] = []
    concentrations: list[torch.Tensor] = []
    for start in range(0, rows.shape[0], int(batch_size)):
        stop = min(start + int(batch_size), rows.shape[0])
        selected_mask = mask[start:stop]
        selected = features[compact_rows[start:stop].clamp_min(0)].float()
        selected = F.normalize(selected, dim=-1)
        mean = (selected * selected_mask[..., None]).sum(dim=1) / selected_mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        concentration = torch.linalg.vector_norm(mean, dim=-1)
        directions.append(F.normalize(mean, dim=-1))
        concentrations.append(concentration)
    direction = torch.cat(directions).to(torch.float16).contiguous()
    concentration = torch.cat(concentrations).float().contiguous()
    if (
        direction.shape != (rows.shape[0], features.shape[1])
        or concentration.shape != (rows.shape[0],)
        or not bool(torch.isfinite(direction).all())
        or not bool(torch.isfinite(concentration).all())
        or bool((concentration < 0).any())
        or bool((concentration > 1.0001).any())
    ):
        raise RuntimeError("region capability pooling failed")
    return direction, concentration.clamp(0.0, 1.0)


def validate_region_capability_descriptor_authority(
    value: object,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("region capability descriptor authority must be a mapping")
    payload = dict(value)
    required = {
        "schema",
        "schema_version",
        "scene_id",
        "producer",
        "input_authority",
        "pooling_contract",
        "source_access",
        "content_authority_sha256",
        "region_fingerprints",
        *CHANNEL_NAMES,
        "channel_sha256",
        "audit",
    }
    if (
        set(payload) != required
        or payload.get("schema") != SCHEMA
        or payload.get("schema_version") != 2
        or payload.get("source_access") != source_access()
        or payload.get("pooling_contract")
        != {
            "primitive_normalization": "explicit_l2",
            "aggregation": "uniform_mean_over_unpadded_region_tokens",
            "direction": "l2_normalized_mean",
            "concentration": "l2_norm_of_mean_primitive_unit_directions",
            "storage": "float16_direction_float32_concentration",
        }
    ):
        raise ValueError("region capability descriptor header differs")
    identity_names = (
        "schema",
        "schema_version",
        "scene_id",
        "producer",
        "input_authority",
        "pooling_contract",
        "source_access",
    )
    if payload.get("content_authority_sha256") != canonical_json_sha256(
        {name: payload[name] for name in identity_names}
    ):
        raise ValueError("region capability descriptor identity changed")
    rows = torch.as_tensor(payload["region_rows"])
    mask = torch.as_tensor(payload["token_mask"])
    canonical = torch.as_tensor(payload["canonical_region_indices"])
    appearance = torch.as_tensor(payload["appearance_direction"])
    boundary = torch.as_tensor(payload["boundary_direction"])
    appearance_concentration = torch.as_tensor(payload["appearance_concentration"])
    boundary_concentration = torch.as_tensor(payload["boundary_concentration"])
    count = int(rows.shape[0]) if rows.ndim == 2 else -1
    if (
        count <= 0
        or mask.dtype != torch.bool
        or mask.shape != rows.shape
        or canonical.shape != (count,)
        or appearance.dtype != torch.float16
        or boundary.dtype != torch.float16
        or appearance.ndim != 2
        or boundary.ndim != 2
        or appearance.shape[0] != count
        or boundary.shape[0] != count
        or appearance_concentration.dtype != torch.float32
        or boundary_concentration.dtype != torch.float32
        or appearance_concentration.shape != (count,)
        or boundary_concentration.shape != (count,)
        or len(payload["region_fingerprints"]) != count
    ):
        raise ValueError("region capability descriptor channels differ")
    if set(payload["channel_sha256"]) != set(CHANNEL_NAMES):
        raise ValueError("region capability descriptor channel mapping differs")
    for name in CHANNEL_NAMES:
        if payload["channel_sha256"].get(name) != tensor_sha256(payload[name]):
            raise ValueError(f"region capability descriptor channel changed: {name}")
    return payload


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"region capability descriptor exists: {output}")
    accepted_raw, accepted_sha, accepted_path = load_torch_mapping(
        args.accepted_v2,
        expected_sha256=args.expected_accepted_v2_sha256,
        map_location="cpu",
        label="region capability AcceptedV2",
    )
    accepted = shard.validate_accepted_region_authority(accepted_raw)
    capability_path = Path(args.capability_bank).expanduser().resolve()
    capability_sha = sha256_file(capability_path)
    if capability_sha != args.expected_capability_bank_sha256:
        raise ValueError("region capability bank SHA-256 differs")
    geometry = accepted["input_authority"]["geometry_authority"]
    bank = load_canonical_capability_bank(
        capability_path,
        expected_field_checkpoint_sha256=geometry[
            "factorized_field_checkpoint_file_sha256"
        ],
        require_signatures=True,
        require_row_authority=True,
        require_formal_projection_order=True,
    )
    if (
        canonical_json_sha256(bank.metadata["primitive_row_authority"])
        != geometry["primitive_row_authority_sha256"]
        or int(geometry["geometry_fingerprint"]["num_gaussians"])
        != bank.num_gaussians
        or geometry["geometry_fingerprint"]["xyz_sha256"]
        != bank.metadata["mpr_geometry_fingerprint"]["xyz_sha256"]
    ):
        raise ValueError("AcceptedV2 and capability primitive authority differ")
    rows = accepted["region_rows"].long().cpu()
    mask = accepted["token_mask"].bool().cpu()
    active_rows = bank.global_rows.long().cpu()
    feature_banks = bank.valid_feature_banks()
    appearance_direction, appearance_concentration = pool_region_capability(
        compact_features=feature_banks["appearance"],
        active_global_rows=active_rows,
        region_rows=rows,
        token_mask=mask,
        batch_size=args.batch_size,
    )
    boundary_direction, boundary_concentration = pool_region_capability(
        compact_features=feature_banks["boundary"],
        active_global_rows=active_rows,
        region_rows=rows,
        token_mask=mask,
        batch_size=args.batch_size,
    )
    identity = {
        "schema": SCHEMA,
        "schema_version": 2,
        "scene_id": accepted["scene_id"],
        "producer": file_record(Path(__file__).resolve()),
        "input_authority": {
            "accepted_v2": {"path": str(accepted_path), "sha256": accepted_sha},
            "capability_bank": {
                "path": str(capability_path),
                "sha256": capability_sha,
            },
            "factorized_field_checkpoint_sha256": geometry[
                "factorized_field_checkpoint_file_sha256"
            ],
            "primitive_row_authority_sha256": geometry[
                "primitive_row_authority_sha256"
            ],
        },
        "pooling_contract": {
            "primitive_normalization": "explicit_l2",
            "aggregation": "uniform_mean_over_unpadded_region_tokens",
            "direction": "l2_normalized_mean",
            "concentration": "l2_norm_of_mean_primitive_unit_directions",
            "storage": "float16_direction_float32_concentration",
        },
        "source_access": source_access(),
    }
    payload = {
        **identity,
        "content_authority_sha256": canonical_json_sha256(identity),
        "region_fingerprints": list(accepted["region_fingerprints"]),
        "canonical_region_indices": accepted["canonical_region_indices"],
        "region_rows": rows,
        "token_mask": mask,
        "appearance_direction": appearance_direction,
        "boundary_direction": boundary_direction,
        "appearance_concentration": appearance_concentration,
        "boundary_concentration": boundary_concentration,
        "channel_sha256": {},
        "audit": {
            "regions": int(rows.shape[0]),
            "appearance_dim": int(appearance_direction.shape[1]),
            "boundary_dim": int(boundary_direction.shape[1]),
            "appearance_concentration_mean": float(appearance_concentration.mean()),
            "boundary_concentration_mean": float(boundary_concentration.mean()),
        },
    }
    payload["channel_sha256"] = {
        name: tensor_sha256(payload[name]) for name in CHANNEL_NAMES
    }
    validate_region_capability_descriptor_authority(payload)
    written = write_torch_noclobber(output, payload)
    return {
        "status": "region_capability_descriptor_complete",
        "scene_id": accepted["scene_id"],
        "output": file_record(written),
        "audit": payload["audit"],
        "target_metric_computed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-v2", required=True)
    parser.add_argument("--expected-accepted-v2-sha256", required=True)
    parser.add_argument("--capability-bank", required=True)
    parser.add_argument("--expected-capability-bank-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", required=True)
    print(json.dumps(materialize(parser.parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
