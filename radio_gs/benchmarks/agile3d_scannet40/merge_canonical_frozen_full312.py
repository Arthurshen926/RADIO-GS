#!/usr/bin/env python3
"""Fail-closed merger for the frozen Ours AGILE3D ScanNet40 full312 row."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from radio_gs.field.observation_lifting_contract import (
    CANONICAL_FULL_OBSERVATION_CONTRACT_NAMES,
)

from .frozen_full312_contract import (
    FROZEN_FULL312_OBJECT_COUNT,
    FROZEN_FULL312_SCENE_COUNT,
    FROZEN_FULL312_SCHEMA,
    bind_frozen_method_contract,
    require_sha256,
    source_contract_bindings_sha256,
)
from .protocol import (
    FROZEN_FULL312_IOU_CLICK_COUNTS,
    FROZEN_FULL312_MAX_CLICKS,
    aggregate_frozen_full312_metrics,
    load_official_object_list,
)


_BENCHMARK_NAME = "AGILE3D ScanNet40 single-object"
_CLICK_POLICY = "center_of_largest_FP_or_FN_error_by_inradius"
_COORDINATE_CONTRACT = (
    "released_shifted_5cm_callback_plus_label_free_scene_origin_to_scannet_world"
)


def _stable_support_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in dict(record).items()
        if str(key) != "geometry_cache_reused"
    }


def _require_frozen_protocol(protocol: Mapping[str, Any], *, source: Path) -> None:
    expected = {
        "result_status": "formal",
        "formal_comparable": True,
        "diagnostic_no_support_gate": False,
        "observation_contract": "scannet_full_observation_v1",
        "support_gate_required": True,
        "max_clicks": FROZEN_FULL312_MAX_CLICKS,
        "voxel_size_m": 0.05,
        "evaluation_voxel_size_m": 0.05,
        "click_policy": _CLICK_POLICY,
        "clicked_labels_forced": True,
        "test_set_calibration": False,
        "official_coordinate_contract": _COORDINATE_CONTRACT,
        "observation_lift": "none",
        "official_point_readout": "continuous_opacity_weighted_gaussian",
    }
    mismatches = {
        key: {"expected": value, "actual": protocol.get(key)}
        for key, value in expected.items()
        if protocol.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{source} violates frozen full312 protocol: {mismatches}")
    if float(protocol.get("minimum_support_fraction", 0.0)) != 0.95:
        raise ValueError(f"{source} must use the frozen 0.95 support gate")
    if (
        str(protocol.get("canonical_mpr_contract", ""))
        not in CANONICAL_FULL_OBSERVATION_CONTRACT_NAMES
        or not bool(protocol.get("canonical_mpr_coverage_ranked", False))
    ):
        raise ValueError(
            f"{source} lacks a coverage-ranked full-observation MPR contract"
        )


def _require_frozen_support(
    record: Mapping[str, Any], *, source: Path
) -> None:
    scene_id = str(record.get("scene_id", ""))
    if (
        str(record.get("declared_source_contract", ""))
        != "scannet_full_observation_v1"
        or str(record.get("field_source_contract_version", ""))
        != "scannet_full_observation_v1"
    ):
        raise ValueError(f"{source} has a non-frozen source contract for {scene_id}")
    if (
        str(record.get("mpr_observation_contract", ""))
        not in CANONICAL_FULL_OBSERVATION_CONTRACT_NAMES
        or not bool(record.get("mpr_full_observation_coverage_order_applied", False))
    ):
        raise ValueError(f"{source} has non-ranked MPR evidence for {scene_id}")
    fraction = float(record.get("continuous_support_fraction", float("nan")))
    if (
        not math.isfinite(fraction)
        or fraction < 0.95
        or not bool(record.get("support_gate_passed", False))
    ):
        raise ValueError(f"{source} has failed support for {scene_id}: {fraction}")


def merge(
    benchmark_root: str | Path,
    inputs: Sequence[str | Path],
    output: str | Path,
    *,
    expected_method_contract_sha256: str = "",
    expected_source_contract_bindings_sha256: str = "",
) -> dict[str, Any]:
    """Merge exact immutable shards into the only accepted full312 shape."""

    if not inputs:
        raise ValueError("at least one frozen full312 shard is required")
    root = Path(benchmark_root)
    official = load_official_object_list(root)
    official_keys = [(item.scene_id, int(item.object_id)) for item in official]
    if len(official) != FROZEN_FULL312_OBJECT_COUNT:
        raise ValueError(
            "official AGILE3D object list does not match frozen full312: "
            f"expected={FROZEN_FULL312_OBJECT_COUNT}, actual={len(official)}"
        )
    if len(set(official_keys)) != len(official_keys):
        raise ValueError("official AGILE3D object list contains duplicate keys")
    official_by_key = dict(zip(official_keys, official))
    official_scenes = list(dict.fromkeys(item.scene_id for item in official))
    if len(official_scenes) != FROZEN_FULL312_SCENE_COUNT:
        raise ValueError(
            "official AGILE3D scene set does not match frozen full312: "
            f"expected={FROZEN_FULL312_SCENE_COUNT}, actual={len(official_scenes)}"
        )
    official_scene_set = set(official_scenes)

    rows: dict[tuple[str, int], dict[str, Any]] = {}
    support: dict[str, dict[str, Any]] = {}
    method_contract: dict[str, Any] | None = None
    method_contract_sha256 = ""
    source_paths: list[str] = []
    shard_bindings: list[dict[str, Any]] = []

    for raw_path in inputs:
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_paths.append(str(path.resolve()))
        if payload.get("benchmark") != _BENCHMARK_NAME:
            raise ValueError(f"{path} is not an AGILE3D single-object report")
        protocol = payload.get("protocol")
        if not isinstance(protocol, dict):
            raise ValueError(f"{path} lacks a complete protocol record")
        _require_frozen_protocol(protocol, source=path)

        declared_contract = payload.get("method_contract")
        if not isinstance(declared_contract, dict):
            raise ValueError(f"{path} lacks method_contract")
        rebuilt_contract, rebuilt_sha256 = bind_frozen_method_contract(protocol)
        declared_sha256 = require_sha256(
            payload.get("method_contract_sha256", ""),
            label=f"{path}.method_contract_sha256",
        )
        if declared_contract != rebuilt_contract or declared_sha256 != rebuilt_sha256:
            raise ValueError(f"{path} has a stale or forged method contract binding")
        if method_contract is None:
            method_contract = declared_contract
            method_contract_sha256 = declared_sha256
        elif (
            method_contract != declared_contract
            or method_contract_sha256 != declared_sha256
        ):
            raise ValueError(f"{path} uses a different method contract hash")

        scene_records = payload.get("scene_support", [])
        if not isinstance(scene_records, list) or not scene_records:
            raise ValueError(f"{path} lacks scene_support")
        rebuilt_bindings, rebuilt_bindings_sha256 = (
            source_contract_bindings_sha256(scene_records)
        )
        declared_bindings = payload.get("source_contract_bindings")
        declared_bindings_sha256 = require_sha256(
            payload.get("source_contract_bindings_sha256", ""),
            label=f"{path}.source_contract_bindings_sha256",
        )
        if (
            declared_bindings != rebuilt_bindings
            or declared_bindings_sha256 != rebuilt_bindings_sha256
        ):
            raise ValueError(f"{path} has a stale or forged source contract binding")
        shard_bindings.append(
            {
                "source": str(path.resolve()),
                "source_contract_bindings_sha256": declared_bindings_sha256,
            }
        )

        for raw_record in scene_records:
            record = dict(raw_record)
            scene_id = str(record.get("scene_id", ""))
            if scene_id not in official_scene_set:
                raise ValueError(f"{path} reports an unknown scene: {scene_id}")
            _require_frozen_support(record, source=path)
            existing = support.get(scene_id)
            if existing is not None:
                # Object shards repeat the same pre-label support audit.  A
                # repeated scene is accepted only when every source identity
                # and support value is identical.
                if _stable_support_record(existing) != _stable_support_record(record):
                    raise ValueError(
                        f"{path} has a conflicting duplicate source contract: {scene_id}"
                    )
                continue
            support[scene_id] = record

        payload_rows = payload.get("rows", [])
        if not isinstance(payload_rows, list) or not payload_rows:
            raise ValueError(f"{path} lacks result rows")
        if int(protocol.get("objects", -1)) != len(payload_rows):
            raise ValueError(f"{path} protocol object count disagrees with its rows")
        if int(protocol.get("scenes", -1)) != len(scene_records):
            raise ValueError(f"{path} protocol scene count disagrees with scene_support")
        shard_scene_ids = {str(record["scene_id"]) for record in scene_records}
        for raw_row in payload_rows:
            row = dict(raw_row)
            key = (str(row.get("scene_id", "")), int(row.get("object_id", -1)))
            if key[0] not in shard_scene_ids:
                raise ValueError(
                    f"{path} has a row without a same-shard source binding: {key}"
                )
            expected = official_by_key.get(key)
            if expected is None:
                raise ValueError(f"{path} reports an unknown AGILE3D object: {key}")
            if str(row.get("semantic_class", "")) != expected.semantic_class:
                raise ValueError(f"{path} semantic class disagrees for {key}")
            if key in rows:
                raise ValueError(f"duplicate frozen full312 object across shards: {key}")
            # This validates exact steps, bounds, and finiteness before a row
            # can enter the complete cohort.
            aggregate_frozen_full312_metrics([row.get("trajectory", {})])
            rows[key] = row

    assert method_contract is not None
    if expected_method_contract_sha256:
        expected_method_sha = require_sha256(
            expected_method_contract_sha256,
            label="expected_method_contract_sha256",
        )
        if method_contract_sha256 != expected_method_sha:
            raise ValueError("merged method contract does not match its frozen authority")

    missing_scenes = [scene for scene in official_scenes if scene not in support]
    missing_objects = [
        item.key
        for item, key in zip(official, official_keys)
        if key not in rows
    ]
    if missing_scenes or missing_objects:
        raise ValueError(
            "frozen full312 merge is incomplete: "
            f"scenes={len(missing_scenes)} {missing_scenes[:5]}, "
            f"objects={len(missing_objects)} {missing_objects[:5]}"
        )
    if len(support) != FROZEN_FULL312_SCENE_COUNT or len(rows) != FROZEN_FULL312_OBJECT_COUNT:
        raise AssertionError("frozen cohort cardinality changed after validation")

    ordered_support = [support[scene] for scene in official_scenes]
    ordered_rows = [rows[key] for key in official_keys]
    source_bindings, source_bindings_sha256 = source_contract_bindings_sha256(
        ordered_support
    )
    if expected_source_contract_bindings_sha256:
        expected_source_sha = require_sha256(
            expected_source_contract_bindings_sha256,
            label="expected_source_contract_bindings_sha256",
        )
        if source_bindings_sha256 != expected_source_sha:
            raise ValueError("merged source contracts do not match their frozen authority")

    fractions = [float(record["continuous_support_fraction"]) for record in ordered_support]
    report: dict[str, Any] = {
        "schema": FROZEN_FULL312_SCHEMA,
        "benchmark": _BENCHMARK_NAME,
        "result_status": "formal_frozen",
        "protocol": {
            "official_preprocessed_data": str(root.resolve()),
            "scenes": FROZEN_FULL312_SCENE_COUNT,
            "objects": FROZEN_FULL312_OBJECT_COUNT,
            "max_clicks": FROZEN_FULL312_MAX_CLICKS,
            "click_counts": list(FROZEN_FULL312_IOU_CLICK_COUNTS),
            "aggregation": "query_micro_over_released_objects",
            "metric_domain": "point_level_object_iou",
        },
        "method_contract": method_contract,
        "method_contract_sha256": method_contract_sha256,
        "source_contract_bindings": source_bindings,
        "source_contract_bindings_sha256": source_bindings_sha256,
        "scene_support": ordered_support,
        "support_summary": {
            "minimum_continuous_support_fraction": min(fractions),
            "mean_continuous_support_fraction": sum(fractions) / len(fractions),
            "scenes_passing_095": sum(value >= 0.95 for value in fractions),
            "scenes_total": len(fractions),
        },
        "metrics": aggregate_frozen_full312_metrics(
            [row["trajectory"] for row in ordered_rows]
        ),
        "rows": ordered_rows,
        "merge": {
            "source_shards": source_paths,
            "source_shard_bindings": shard_bindings,
            "prediction_recomputed": False,
            "clicks_recomputed": False,
        },
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-method-contract-sha256", default="")
    parser.add_argument("--expected-source-contract-bindings-sha256", default="")
    args = parser.parse_args()
    print(
        json.dumps(
            merge(
                args.benchmark_root,
                args.inputs,
                args.output,
                expected_method_contract_sha256=(
                    args.expected_method_contract_sha256
                ),
                expected_source_contract_bindings_sha256=(
                    args.expected_source_contract_bindings_sha256
                ),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
