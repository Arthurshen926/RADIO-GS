#!/usr/bin/env python3
"""Build the immutable inventory for migrated Universal Field v1 assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from radio_gs.universal_field_v1 import (
    PRIMITIVE_READOUT_ID,
    RELIABILITY_NAMES,
    UNIVERSAL_FIELD_ID,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    write_frozen_json,
)


LERF_SCENES = ("figurines", "ramen", "teatime", "waldo_kitchen")
SCANNET_SCENES = (
    "scene0000_00",
    "scene0062_00",
    "scene0070_00",
    "scene0097_00",
    "scene0140_00",
    "scene0347_00",
    "scene0400_00",
    "scene0590_00",
)


def build_inventory(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    scene_records: dict[str, Any] = {}
    for scene in (*LERF_SCENES, *SCANNET_SCENES):
        filename = (
            "universal_field_v1_shared_reliability.pth"
            if scene == "figurines"
            else "universal_field_v1.pth"
        )
        field_path = root / scene / filename
        report_path = field_path.with_suffix(field_path.suffix + ".json")
        report, report_sha, report_path = load_json_object(
            report_path,
            label=f"{scene} Universal Field v1 migration report",
        )
        if (
            report.get("status") != "complete"
            or report.get("universal_field") != file_record(field_path)
            or report.get("reliability_dim") != len(RELIABILITY_NAMES)
            or report.get("reliability_serialized_once") is not True
            or report.get("coefficient_decode_bitwise_equal") is not True
            or report.get("radio_decode_bitwise_equal") is not True
            or report.get("serialized_radio_decode_bitwise_equal") is not True
            or report.get("field_training_rerun") is not False
        ):
            raise ValueError(f"{scene} migration report fails the inventory gate")
        migration = report.get("migration", {})
        if (
            migration.get("universal_field_id") != UNIVERSAL_FIELD_ID
            or migration.get("baseline_readout_id") != PRIMITIVE_READOUT_ID
            or migration.get("reliability_scalar_names") != list(RELIABILITY_NAMES)
            or migration.get("registration_weight_mode")
            != "exact_front_to_back_marginal_responsibility"
            or migration.get("fusion_reliability") is not False
            or migration.get("decode_state_changed") is not False
        ):
            raise ValueError(f"{scene} migration contract differs")
        scene_records[scene] = {
            "status": "complete",
            "num_gaussians": report["num_gaussians"],
            "source_field": report["source_field"],
            "factorized_cache": report["factorized_cache"],
            "universal_field": report["universal_field"],
            "migration_report": {"path": str(report_path), "sha256": report_sha},
            "storage_delta_bytes": report["storage_delta_bytes"],
            "storage_overhead_bytes": report["storage_overhead_bytes"],
            "reliability_serialized_once": True,
            "coefficient_decode_bitwise_equal": True,
            "radio_decode_bitwise_equal": True,
            "serialized_radio_decode_bitwise_equal": True,
        }
    return {
        "schema_version": 1,
        "artifact_type": "radio_gs_universal_field_v1_asset_inventory",
        "status": "complete_12_of_12_existing_d512_l512_fields",
        "universal_field_id": UNIVERSAL_FIELD_ID,
        "baseline_readout_id": PRIMITIVE_READOUT_ID,
        "authority": file_record(
            "paper/artifacts/universal_field_v1_authority_20260816.json"
        ),
        "registration_weight_mode": "exact_front_to_back_marginal_responsibility",
        "reliability": {
            "dim": len(RELIABILITY_NAMES),
            "scalar_names": list(RELIABILITY_NAMES),
            "usage": "query_posterior_calibration_only",
            "fused_into_decoder": False,
        },
        "accounting": {
            "expected": 12,
            "complete": len(scene_records),
            "lerf_shared": len(LERF_SCENES),
            "scannet_ovs": len(SCANNET_SCENES),
            "field_training_reruns": 0,
            "decode_changed": 0,
        },
        "scenes": scene_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260816/universal_field_v1",
    )
    parser.add_argument(
        "--output",
        default="paper/artifacts/universal_field_v1_asset_inventory_20260816.json",
    )
    args = parser.parse_args()
    inventory = build_inventory(Path(args.root))
    write_frozen_json(args.output, inventory)
    print(json.dumps(inventory, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
