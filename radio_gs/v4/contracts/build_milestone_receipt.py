"""Seal a v4 milestone against source files and evidence reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .geometry_receipt import sha256_file


def run(args: argparse.Namespace) -> dict:
    source_root = Path(args.source_root).resolve(strict=True)
    evidence_paths = [Path(value).resolve(strict=True) for value in args.evidence]
    source_files = sorted(path for path in source_root.rglob("*.py") if path.is_file())
    if not source_files or not evidence_paths:
        raise ValueError("milestone receipt requires source files and evidence reports")
    evidence = []
    for path in evidence_paths:
        payload = json.loads(path.read_text())
        evidence.append({
            "path": str(path),
            "sha256": sha256_file(path),
            "schema": payload.get("schema"),
        })
    geometry = next(
        (json.loads(path.read_text()) for path in evidence_paths if "geometry_gate_complete" in path.name),
        None,
    )
    object_gate = next(
        (json.loads(path.read_text()) for path in evidence_paths if "object_codebook_oracle_gate" in path.name),
        None,
    )
    if geometry is None or geometry.get("milestone_1_complete") is not True:
        raise ValueError("complete geometry gate evidence is missing")
    if object_gate is None or object_gate.get("passes_oracle_object_gate") is not True:
        raise ValueError("passing oracle object-codebook evidence is missing")
    report = {
        "schema": "radio_gs.surface_object_memory_v4.milestone_receipt.v1",
        "milestone": "geometry_and_oracle_codebook",
        "git_head": args.git_head,
        "source_snapshot": [
            {
                "path": str(path.relative_to(source_root)),
                "sha256": sha256_file(path),
            }
            for path in source_files
        ],
        "evidence": evidence,
        "decisions": {
            "geometry_milestone_complete": True,
            "oracle_object_codebook_passed": True,
            "learned_soft_codebook_authorized": bool(
                object_gate.get("learned_soft_codebook_authorized", False)
            ),
            "query_encoder_authorized": False,
            "compression_authorized": False,
            "carrier_parameters_frozen": True,
            "historical_v3_method_imported": False,
        },
        "carrier_contract": {
            "voxel_size_metres": 0.04,
            "maximum_splat_radius_at_reference_raster": 1,
            "surface_band_voxels": 1.5,
            "maximum_contributors_per_pixel": 8,
            "resolution_invariance_audit_pending": True,
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--git-head", required=True)
    parser.add_argument("--evidence", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report["decisions"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
