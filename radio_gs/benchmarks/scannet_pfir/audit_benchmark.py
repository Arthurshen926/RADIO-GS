#!/usr/bin/env python3
"""Audit frozen PFIR manifests for leakage, hashes and release readiness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from radio_gs.benchmarks.scannet_pfir.build_benchmark import readiness
from radio_gs.benchmarks.scannet_pfir.protocol import audit_manifest, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", required=True)
    args = parser.parse_args()
    root = Path(args.benchmark_dir)
    public = json.loads((root / "manifest.public.json").read_text(encoding="utf-8"))
    method = json.loads((root / "manifest.method.json").read_text(encoding="utf-8"))
    internal = json.loads((root / "manifest.internal.json").read_text(encoding="utf-8"))
    release = json.loads((root / "release.json").read_text(encoding="utf-8"))
    manifest_hashes = {
        name: sha256_file(root / name)
        for name in release.get("manifest_sha256", {})
    }
    report = {
        "manifest_audit": audit_manifest(public),
        "readiness": readiness(
            internal.get("queries", []), internal.get("scene_reports", [])
        ),
        "release_hashes_match": manifest_hashes == release.get("manifest_sha256", {}),
        "computed_manifest_sha256": manifest_hashes,
        "method_crop_hashes_match": all(
            Path(record["crop_rgb_path"]).is_file()
            and sha256_file(record["crop_rgb_path"]) == record["crop_rgb_sha256"]
            and set(record.get("available_method_inputs", ()))
            == {"scene_id", "crop_rgb"}
            for record in method.get("queries", [])
        ),
        "nyu40_labels_valid": all(
            0 < int(record["nyu40_class_id"]) <= 40
            and all(
                0 < int(value) <= 40
                for value in record["candidate_instance_class_ids"].values()
            )
            for record in internal.get("queries", [])
        ),
    }
    report["valid"] = bool(
        report["manifest_audit"]["valid"]
        and report["release_hashes_match"]
        and report["method_crop_hashes_match"]
        and report["nyu40_labels_valid"]
    )
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
