#!/usr/bin/env python3
"""Validate the canonical external-method evaluation protocol freeze."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


EXPECTED_TASKS = {
    "concept_lerf2d_occamlgs",
    "concept_lerf3d_vala",
    "concept_scannet_ovs_vala_paper8",
    "spatial_agile3d_easy3d",
    "spatial_nvos_ludvig",
    "spatial_spin9_ludvig",
    "correspondence_pfpr_ludvig_adapter",
}
PAPER8_SCENES = [
    "scene0000_00",
    "scene0062_00",
    "scene0070_00",
    "scene0097_00",
    "scene0140_00",
    "scene0347_00",
    "scene0400_00",
    "scene0590_00",
]
LERF_SCENES = ["figurines", "ramen", "teatime", "waldo_kitchen"]
SPIN9_SCENES = [
    "fern",
    "fortress",
    "horns",
    "leaves",
    "lego",
    "orchids",
    "pinecone",
    "room",
    "truck",
]
NVOS_TASKS = [
    "fern",
    "flower",
    "fortress",
    "horns_center",
    "horns_left",
    "leaves",
    "orchids",
    "trex",
]
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class FreezeError(ValueError):
    """Raised when the freeze could select an ambiguous or unsafe protocol."""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FreezeError(f"{path} must be a mapping")
    return value


def _repo_path(root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_mapping(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FreezeError(f"{label} does not exist: {path}")
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)


def _require_equal(actual: Any, expected: Any, path: str) -> None:
    if actual != expected:
        raise FreezeError(f"{path} must equal {expected!r}, got {actual!r}")


def validate_freeze(
    payload: Mapping[str, Any],
    *,
    root: Path,
    verify_hashes: bool = True,
) -> None:
    """Fail closed on task selection, protocol invariants, and artifact identity."""

    _require_equal(payload.get("schema_version"), 1, "schema_version")
    governance = _mapping(payload.get("governance"), "governance")
    _require_equal(
        governance.get("historical_results_may_replace_canonical_rows"),
        False,
        "governance.historical_results_may_replace_canonical_rows",
    )
    _require_equal(
        governance.get("oracle_may_select_reported_configuration_or_metric"),
        False,
        "governance.oracle_may_select_reported_configuration_or_metric",
    )
    _require_equal(
        governance.get("cleanup_requires_user_approval"),
        True,
        "governance.cleanup_requires_user_approval",
    )
    _require_equal(
        governance.get("execution_safety_is_evaluation_semantics"),
        False,
        "governance.execution_safety_is_evaluation_semantics",
    )

    registry_path = _repo_path(root, str(governance.get("primary_registry", "")))
    registry = _load_mapping(registry_path, "primary registry")
    registry_rows = _mapping(registry.get("evaluations"), "primary registry rows")
    promptable_path = _repo_path(
        root, str(governance.get("promptable_registry", ""))
    )
    promptable = _load_mapping(promptable_path, "promptable registry")
    promptable_rows = _mapping(
        promptable.get("protocols"), "promptable registry protocols"
    )

    tasks = _mapping(payload.get("canonical_tasks"), "canonical_tasks")
    if set(tasks) != EXPECTED_TASKS:
        raise FreezeError(
            "canonical_tasks mismatch; "
            f"missing={sorted(EXPECTED_TASKS - set(tasks))}, "
            f"extra={sorted(set(tasks) - EXPECTED_TASKS)}"
        )
    selected_rows: set[str] = set()
    for task_id, raw_task in tasks.items():
        task = _mapping(raw_task, task_id)
        registry_row = task.get("registry_row")
        if not isinstance(registry_row, str) or registry_row not in registry_rows:
            raise FreezeError(f"{task_id}.registry_row is not present in primary registry")
        if registry_row in selected_rows:
            raise FreezeError(f"registry row selected more than once: {registry_row}")
        selected_rows.add(registry_row)
        promptable_row = task.get("promptable_registry_row")
        if promptable_row is not None and promptable_row not in promptable_rows:
            raise FreezeError(
                f"{task_id}.promptable_registry_row is not present in promptable registry"
            )
        for index, raw_entrypoint in enumerate(task.get("entrypoints", [])):
            if not isinstance(raw_entrypoint, str):
                raise FreezeError(f"{task_id}.entrypoints[{index}] must be a path")
            if not _repo_path(root, raw_entrypoint).is_file():
                raise FreezeError(
                    f"{task_id}.entrypoints[{index}] does not exist: {raw_entrypoint}"
                )
        artifacts = task.get("authoritative_artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise FreezeError(f"{task_id}.authoritative_artifacts must be non-empty")
        for index, raw_artifact in enumerate(artifacts):
            artifact = _mapping(
                raw_artifact, f"{task_id}.authoritative_artifacts[{index}]"
            )
            raw_path = artifact.get("path")
            expected_sha = artifact.get("sha256")
            if not isinstance(raw_path, str) or not raw_path:
                raise FreezeError(f"{task_id} artifact path must be non-empty")
            if not isinstance(expected_sha, str) or not SHA256_RE.fullmatch(expected_sha):
                raise FreezeError(f"{task_id} artifact has invalid sha256")
            _require_equal(
                artifact.get("retention"),
                "must_keep",
                f"{task_id}.authoritative_artifacts[{index}].retention",
            )
            path = _repo_path(root, raw_path)
            if not path.is_file():
                raise FreezeError(f"authoritative artifact does not exist: {raw_path}")
            if verify_hashes:
                actual_sha = _sha256(path)
                if actual_sha != expected_sha:
                    raise FreezeError(
                        f"authoritative artifact hash mismatch: {raw_path}; "
                        f"expected {expected_sha}, got {actual_sha}"
                    )

    occam = _mapping(tasks["concept_lerf2d_occamlgs"], "occam")
    _require_equal(occam["cohort"]["scenes"], LERF_SCENES, "OccamLGS scenes")
    _require_equal(
        occam["frozen_protocol"]["segmentation_threshold"],
        0.5,
        "OccamLGS threshold",
    )

    vala_lerf = _mapping(tasks["concept_lerf3d_vala"], "VALA LERF-3D")
    _require_equal(vala_lerf["cohort"]["scenes"], LERF_SCENES, "VALA LERF scenes")
    _require_equal(
        vala_lerf["cohort"]["extensionless_test_stems_required"],
        True,
        "VALA extensionless test split",
    )
    _require_equal(
        vala_lerf["frozen_protocol"]["mask_threshold"], 0.6, "VALA LERF threshold"
    )

    scannet = _mapping(tasks["concept_scannet_ovs_vala_paper8"], "ScanNet")
    _require_equal(scannet["cohort"]["scenes"], PAPER8_SCENES, "ScanNet paper8 scenes")
    _require_equal(scannet["cohort"]["splits"], [19, 15, 10], "ScanNet splits")
    _require_equal(
        scannet["cohort"]["scene0645_00_role"],
        "post-paper code9 sensitivity only",
        "ScanNet scene0645 role",
    )
    pseudo_gt = str(scannet["frozen_protocol"].get("pseudo_ground_truth", ""))
    if "Mahalanobis" not in pseudo_gt:
        raise FreezeError("ScanNet pseudo-GT must remain anisotropic Mahalanobis")

    easy3d = _mapping(tasks["spatial_agile3d_easy3d"], "Easy3D")
    _require_equal(easy3d["cohort"], {"scenes": 312, "objects": 10357, "failures": 0}, "Easy3D cohort")
    _require_equal(
        easy3d["frozen_protocol"]["interaction_contract"],
        "agile3d_release",
        "Easy3D interaction contract",
    )

    nvos = _mapping(tasks["spatial_nvos_ludvig"], "NVOS")
    _require_equal(nvos["cohort"]["tasks"], NVOS_TASKS, "NVOS tasks")
    _require_equal(nvos["cohort"]["seeds"], [0, 1, 2], "NVOS seeds")
    _require_equal(nvos["cohort"]["completed_runs"], 24, "NVOS completed runs")
    _require_equal(
        nvos["frozen_protocol"]["oracle_values_aggregated"], False, "NVOS oracle"
    )

    spin = _mapping(tasks["spatial_spin9_ludvig"], "SPIn-NeRF")
    _require_equal(spin["cohort"]["scenes"], SPIN9_SCENES, "SPIn-NeRF scenes")
    _require_equal(spin["cohort"]["missing_scenes"], ["fork"], "SPIn-NeRF missing scene")
    _require_equal(
        spin["cohort"]["full_10scene_table_eligible"],
        False,
        "SPIn-NeRF full-table eligibility",
    )

    pfpr = _mapping(tasks["correspondence_pfpr_ludvig_adapter"], "PFPR")
    _require_equal(
        pfpr["frozen_protocol"]["paper_comparison"], "forbidden", "PFPR paper comparison"
    )
    _require_equal(
        pfpr["frozen_protocol"]["oracle_completion_deferred"], True, "PFPR oracle deferral"
    )
    _require_equal(
        pfpr["frozen_protocol"]["formal_full_benchmark_deferred"],
        True,
        "PFPR formal deferral",
    )


def load_and_validate(
    path: Path, *, root: Path | None = None, verify_hashes: bool = True
) -> Mapping[str, Any]:
    payload = _load_mapping(path, str(path))
    validate_freeze(
        payload,
        root=root or path.resolve().parents[2],
        verify_hashes=verify_hashes,
    )
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "freeze",
        nargs="?",
        type=Path,
        default=Path("paper/artifacts/evaluation_protocol_freeze_20260801.yaml"),
    )
    parser.add_argument(
        "--skip-hashes",
        action="store_true",
        help="Validate structure and paths without recomputing artifact hashes.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = load_and_validate(args.freeze, verify_hashes=not args.skip_hashes)
    print(
        f"validated {len(payload['canonical_tasks'])} frozen protocols: {args.freeze}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
