#!/usr/bin/env python3
"""Derive the UQIS v0.2 Core/relational text profile from frozen v0.1 targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from radio_gs.benchmarks.scannet_uqis.construction import (
    load_reference_rows,
    select_profiled_expression,
)
from radio_gs.benchmarks.scannet_uqis.official_constructor import (
    TEXT_PROFILE_UQIS_V2,
    UQIS_V2_CONSTRUCTION_CANDIDATE,
    _reference_index,
)
from radio_gs.benchmarks.scannet_uqis.protocol import canonical_json_sha256, sha256_file


def build_candidate(target_records_path: Path, nr3d_path: Path) -> dict:
    source = json.loads(target_records_path.read_text(encoding="utf-8"))
    targets = source.get("targets")
    if source.get("benchmark_version") != "scannet-uqis-9-v0.1" or not isinstance(targets, list):
        raise ValueError("source must be frozen UQIS v0.1 target records")
    references = _reference_index(load_reference_rows(nr3d_path))
    derived = []
    for raw in targets:
        record = dict(raw)
        scene_id = str(record["scene_id"])
        instance_id = int(record["instance_id"])
        expression = select_profiled_expression(
            references.get((scene_id, instance_id - 1), ()),
            scene_id=scene_id,
            official_instance_id=instance_id,
        )
        record.update(
            {
                "expression": expression["expression"],
                "expression_annotation_id": expression["annotation_id"],
                "expression_source": expression["source"],
                "expression_view_independent": True,
                "expression_view_dependence_rule": expression["view_dependence_rule"],
                "evaluation_tier": expression["evaluation_tier"],
                "relational_language_required": expression[
                    "relational_language_required"
                ],
                "spatial_language_evidence": expression[
                    "spatial_language_evidence"
                ],
            }
        )
        derived.append(record)
    core = sum(row["evaluation_tier"] == "unified_core" for row in derived)
    relational = len(derived) - core
    body = {
        "schema_version": "scannet_uqis_v2_text_profile_candidate_v1",
        "benchmark_version": UQIS_V2_CONSTRUCTION_CANDIDATE,
        "status": "construction_candidate_complete",
        "formal_benchmark_eligible": False,
        "text_profile": TEXT_PROFILE_UQIS_V2,
        "source_v1_target_records": {
            "path": str(target_records_path.resolve()),
            "sha256": sha256_file(target_records_path),
        },
        "nr3d": {"path": str(nr3d_path.resolve()), "sha256": sha256_file(nr3d_path)},
        "selection_policy": (
            "prefer_correct_class_mentioning_view_independent_nr3d_expression_"
            "with_uses_spatial_lang_false_else_relational_challenge"
        ),
        "target_count": len(derived),
        "unified_core_target_count": core,
        "relational_text_target_count": relational,
        "targets": derived,
    }
    return {**body, "candidate_sha256": canonical_json_sha256(body)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-records", type=Path, required=True)
    parser.add_argument("--nr3d", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = build_candidate(args.target_records.resolve(), args.nr3d.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "target_count": payload["target_count"],
                "unified_core_target_count": payload["unified_core_target_count"],
                "relational_text_target_count": payload["relational_text_target_count"],
                "candidate_sha256": payload["candidate_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
