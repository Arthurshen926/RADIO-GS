#!/usr/bin/env python3
"""Stage opaque one-query v0.2 Core text workspaces from private construction authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from radio_gs.benchmarks.scannet_uqis.protocol import (
    BENCHMARK_VERSION,
    BENCHMARK_VERSION_V2_CANDIDATE,
    sha256_file,
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def stage(
    *,
    source_release: Path,
    text_profile_path: Path,
    output_root: Path,
    evaluation_tier: str = "unified_core",
) -> dict:
    if output_root.exists():
        raise FileExistsError(output_root)
    source_query_path = source_release / "query_manifest.text.json"
    evaluator_path = source_release / "target_manifest.evaluator.json"
    release_path = source_release / "release.json"
    source_query = json.loads(source_query_path.read_text(encoding="utf-8"))
    evaluator = json.loads(evaluator_path.read_text(encoding="utf-8"))
    profile = json.loads(text_profile_path.read_text(encoding="utf-8"))
    if (
        source_query.get("benchmark_version") != BENCHMARK_VERSION
        or evaluator.get("benchmark_version") != BENCHMARK_VERSION
        or profile.get("benchmark_version") != BENCHMARK_VERSION_V2_CANDIDATE
        or profile.get("formal_benchmark_eligible") is not False
    ):
        raise ValueError("v0.1 authority or v0.2 profile identity changed")

    evaluator_by_instance = {
        (str(row["scene_id"]), int(row["instance_id"])): row
        for row in evaluator["targets"]
    }
    expression_by_query: dict[str, str] = {}
    selected_ids: set[str] = set()
    for target in profile["targets"]:
        authority = evaluator_by_instance[(str(target["scene_id"]), int(target["instance_id"]))]
        query_id = str(authority["queries"]["text"])
        expression_by_query[query_id] = str(target["expression"])
        if target["evaluation_tier"] == evaluation_tier:
            selected_ids.add(query_id)
    expected_count_key = (
        "unified_core_target_count"
        if evaluation_tier == "unified_core"
        else "relational_text_target_count"
    )
    if len(expression_by_query) != int(profile["target_count"]) or len(selected_ids) != int(
        profile[expected_count_key]
    ):
        raise ValueError("v0.2 target/query inventory changed")

    candidate_query = json.loads(json.dumps(source_query))
    candidate_query["benchmark_version"] = BENCHMARK_VERSION_V2_CANDIDATE
    for row in candidate_query["queries"]:
        row["expression"] = expression_by_query[str(row["query_id"])]
    output_root.mkdir(parents=True)
    candidate_query_path = output_root / "query_manifest.text.json"
    _write(candidate_query_path, candidate_query)
    candidate_release = {
        "schema_version": "scannet_uqis_v2_candidate_method_release_v1",
        "benchmark_version": BENCHMARK_VERSION_V2_CANDIDATE,
        "status": (
            "core_text_method_inputs_staged"
            if evaluation_tier == "unified_core"
            else "relational_text_method_inputs_staged"
        ),
        "formal_benchmark_eligible": False,
        "source_v1_release": {
            "path": str(release_path.resolve()),
            "sha256": sha256_file(release_path),
        },
        "private_text_profile_sha256": sha256_file(text_profile_path),
        "method_text_manifest_sha256": sha256_file(candidate_query_path),
        "target_count": len(expression_by_query),
        (
            "core_text_query_count"
            if evaluation_tier == "unified_core"
            else "relational_text_query_count"
        ): len(selected_ids),
        "selected_evaluation_tier": evaluation_tier,
        "evaluator_pairing_published": False,
    }
    candidate_release_path = output_root / "candidate_release.json"
    _write(candidate_release_path, candidate_release)

    domains = {str(row["scene_id"]): row for row in candidate_query["scene_domains"]}
    queries = {str(row["query_id"]): row for row in candidate_query["queries"]}
    common = {
        key: value
        for key, value in candidate_query.items()
        if key not in {"scene_domains", "queries"}
    }
    inventory = []
    for query_id in sorted(selected_ids):
        query = dict(queries[query_id])
        scene_id = str(query["scene_id"])
        workspace = output_root / "workspaces" / "text" / query_id
        assets = workspace / "assets"
        assets.mkdir(parents=True)
        source_mesh = Path(str(domains[scene_id]["mesh_xyz_path"])).resolve()
        mesh = assets / "mesh_xyz.npy"
        shutil.copyfile(source_mesh, mesh)
        domain = {
            **dict(domains[scene_id]),
            "mesh_xyz_path": str(mesh),
            "mesh_xyz_sha256": sha256_file(mesh),
        }
        method_manifest = {**common, "scene_domains": [domain], "queries": [query]}
        workspace_manifest = workspace / "query_manifest.json"
        _write(workspace_manifest, method_manifest)
        receipt = {
            "schema_version": "scannet_uqis_v2_candidate_query_workspace_v1",
            "status": "staged",
            "formal_benchmark_eligible": False,
            "benchmark_version": BENCHMARK_VERSION_V2_CANDIDATE,
            "release_manifest_sha256": sha256_file(candidate_release_path),
            "source_query_manifest_sha256": sha256_file(candidate_query_path),
            "workspace_query_manifest_sha256": sha256_file(workspace_manifest),
            "query_id": query_id,
            "scene_id": scene_id,
            "modality": "text",
            "query_count": 1,
            "independent_workspace_required": True,
            "mount_policy": "workspace_only_read_only",
            "evaluator_private_files_staged": False,
        }
        receipt_path = workspace / "workspace_receipt.json"
        _write(receipt_path, receipt)
        for path in workspace.rglob("*"):
            path.chmod(0o555 if path.is_dir() else 0o444)
        workspace.chmod(0o555)
        inventory.append(
            {
                "query_id": query_id,
                "scene_id": scene_id,
                "workspace_manifest_sha256": sha256_file(workspace_manifest),
                "workspace_receipt_sha256": sha256_file(receipt_path),
            }
        )
    result = {
        **candidate_release,
        "status": (
            "core_text_workspaces_complete"
            if evaluation_tier == "unified_core"
            else "relational_text_workspaces_complete"
        ),
        "candidate_release_sha256": sha256_file(candidate_release_path),
        "workspaces": inventory,
    }
    _write(output_root / "workspace_inventory.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-release", type=Path, required=True)
    parser.add_argument("--text-profile", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--evaluation-tier",
        choices=("unified_core", "relational_text_challenge"),
        default="unified_core",
    )
    args = parser.parse_args()
    result = stage(
        source_release=args.source_release.resolve(),
        text_profile_path=args.text_profile.resolve(),
        output_root=args.output_root.resolve(),
        evaluation_tier=args.evaluation_tier,
    )
    count_key = (
        "core_text_query_count"
        if args.evaluation_tier == "unified_core"
        else "relational_text_query_count"
    )
    print(json.dumps({key: result[key] for key in ("status", count_key, "candidate_release_sha256")}, indent=2))


if __name__ == "__main__":
    main()
