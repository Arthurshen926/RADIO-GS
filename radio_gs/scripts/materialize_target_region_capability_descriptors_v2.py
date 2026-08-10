#!/usr/bin/env python3
"""Pool canonical DINO/SAM capabilities onto target AcceptedV2 regions.

This is a target-safe wrapper over the frozen V2 pooling mathematics and
output schema.  Its only semantic difference from the source producer is the
independent target AcceptedV2 validator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces.capability_cache import load_canonical_capability_bank
from radio_gs.interfaces.surface_region_target_accepted_v2 import (
    validate_target_accepted_v2_authority,
)
from radio_gs.scripts import materialize_region_capability_descriptors_v2 as source
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_torch_mapping,
    sha256_file,
    write_torch_noclobber,
)


def materialize(args: argparse.Namespace) -> dict:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"target region capability descriptor exists: {output}")
    accepted_raw, accepted_sha, accepted_path = load_torch_mapping(
        args.target_accepted_v2,
        expected_sha256=args.expected_target_accepted_v2_sha256,
        map_location="cpu",
        label="target region capability AcceptedV2",
    )
    accepted = validate_target_accepted_v2_authority(accepted_raw)
    capability_path = Path(args.capability_bank).expanduser().resolve()
    capability_sha = sha256_file(capability_path)
    if capability_sha != args.expected_capability_bank_sha256:
        raise ValueError("target region capability bank SHA-256 differs")
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
        or int(geometry["geometry_fingerprint"]["num_gaussians"]) != bank.num_gaussians
        or geometry["geometry_fingerprint"]["xyz_sha256"]
        != bank.metadata["mpr_geometry_fingerprint"]["xyz_sha256"]
    ):
        raise ValueError("target AcceptedV2 and capability primitive authority differ")
    rows = accepted["region_rows"].long().cpu()
    mask = accepted["token_mask"].bool().cpu()
    active_rows = bank.global_rows.long().cpu()
    feature_banks = bank.valid_feature_banks()
    appearance_direction, appearance_concentration = source.pool_region_capability(
        compact_features=feature_banks["appearance"],
        active_global_rows=active_rows,
        region_rows=rows,
        token_mask=mask,
        batch_size=args.batch_size,
    )
    boundary_direction, boundary_concentration = source.pool_region_capability(
        compact_features=feature_banks["boundary"],
        active_global_rows=active_rows,
        region_rows=rows,
        token_mask=mask,
        batch_size=args.batch_size,
    )
    identity = {
        "schema": source.SCHEMA,
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
        "source_access": source.source_access(),
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
        name: tensor_sha256(payload[name]) for name in source.CHANNEL_NAMES
    }
    source.validate_region_capability_descriptor_authority(payload)
    written = write_torch_noclobber(output, payload)
    return {
        "status": "target_region_capability_descriptor_complete",
        "scene_id": accepted["scene_id"],
        "physical_space_id": accepted["physical_space_id"],
        "output": file_record(written),
        "audit": payload["audit"],
        "target_metric_computed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-accepted-v2", required=True)
    parser.add_argument("--expected-target-accepted-v2-sha256", required=True)
    parser.add_argument("--capability-bank", required=True)
    parser.add_argument("--expected-capability-bank-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    print(
        json.dumps(materialize(build_parser().parse_args()), indent=2, allow_nan=False)
    )


if __name__ == "__main__":
    main()
