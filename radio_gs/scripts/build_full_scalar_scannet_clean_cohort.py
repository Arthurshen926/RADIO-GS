#!/usr/bin/env python3
"""Seal a physical-space-disjoint ScanNet clean 24/8 cohort authority.

This builder reads only JSON protocol/materialization metadata and tar member
headers.  It never extracts or decodes a ``.sens`` payload and never opens a
benchmark image, mask, query, label, or metric.  Selection is fixed before any
source feature is built:

1. union all frozen ScanNet benchmark scene IDs and collapse repeated scans to
   canonical ``scene####`` physical spaces;
2. enumerate the source archive in its asserted lexicographic member order;
3. retain the lexicographically first scan of each non-benchmark physical
   space and take the first 32 physical spaces;
4. assign every fourth selected physical space to validation, leaving 24
   train and 8 validation spaces.

The output cohort authority is immediately usable by the full-scalar trainer.
Training shards remain a later materialization stage and must bind the exact
authority and exclusion-manifest file SHA-256 values produced here.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import tarfile
from typing import Any

from radio_gs.scripts.train_surface_region_full_scalar_residual import (
    BENCHMARK_EXCLUSION_MANIFEST_CONTRACT_SHA256,
    BENCHMARK_EXCLUSION_MANIFEST_SCHEMA,
    COHORT_AUTHORITY_CONTRACT_SHA256,
    COHORT_AUTHORITY_SCHEMA,
    COHORT_AUTHORITY_SCHEMA_VERSION,
    SOURCE_MANIFEST_SCHEMA_VERSION,
    TRAIN_SCENE_COUNT,
    VALIDATION_SCENE_COUNT,
    _cohort_authority_access,
    _manifest_content_sha256,
    _source_manifest_access,
    benchmark_exclusion_manifest_contract,
    canonical_physical_space_id,
    cohort_authority_content_sha256,
    cohort_authority_contract,
    validate_benchmark_exclusion_manifest,
    validate_cohort_authority_payload,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_json_object,
    sha256_file,
    write_frozen_json,
)


BENCHMARK_REGISTRY_SCHEMA = (
    "radio_gs.full_scalar_frozen_benchmark_scene_registry.v1"
)
INVENTORY_SCHEMA = "radio_gs.full_scalar_scannet_clean_cohort_inventory.v1"
SELECTION_RULE = (
    "lexicographic_first_scan_per_nonbenchmark_physical_space_then_first32_"
    "validation_every_fourth_v1"
)
REQUIRED_SCENES = TRAIN_SCENE_COUNT + VALIDATION_SCENE_COUNT


def _content_sha256(value: Mapping[str, Any]) -> str:
    content = dict(value)
    content.pop("authority_sha256", None)
    return canonical_json_sha256(content)


def _load_json(
    path: str | Path,
    expected_sha256: str,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    value, observed, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label=label,
    )
    return value, {"path": str(source), "sha256": observed}


def _materialization_scene_ids(value: Mapping[str, Any], *, label: str) -> list[str]:
    if (
        value.get("valid") is not True
        or value.get("uses_instances_or_semantic_labels") is not False
        or value.get("uses_private_anchor") is not False
        or value.get("uses_private_depth_pixel") is not False
        or not isinstance(value.get("scenes"), list)
    ):
        raise ValueError(f"{label} is not a query/label-free materialization report")
    scenes = [
        str(record.get("scene_id", ""))
        for record in value["scenes"]
        if isinstance(record, Mapping)
    ]
    if (
        not scenes
        or len(scenes) != len(value["scenes"])
        or any(not scene for scene in scenes)
        or len(set(scenes)) != len(scenes)
    ):
        raise ValueError(f"{label} scene IDs differ")
    for scene in scenes:
        canonical_physical_space_id(scene)
    return sorted(scenes)


def _prompt_manifest_scene_ids(value: Mapping[str, Any], *, label: str) -> list[str]:
    records = value.get("scenes")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{label} has no scene records")
    scenes = [
        str(record.get("base_scene_id") or record.get("scene_id") or "")
        for record in records
        if isinstance(record, Mapping)
    ]
    if len(scenes) != len(records) or any(not scene for scene in scenes):
        raise ValueError(f"{label} scene IDs differ")
    # Prompt protocols may contain multiple tasks for the same physical scene
    # (for example the two NVOS horns prompts).  The frozen benchmark registry
    # is a scene exclusion registry, so collapse those task-level repetitions
    # without consulting prompt text, masks, labels, or metrics.
    return sorted(set(scenes))


def _source_record(
    *,
    dataset_namespace: str,
    kind: str,
    file_record: Mapping[str, str] | None,
    scene_ids: Sequence[str],
) -> dict[str, Any]:
    return {
        "dataset_namespace": dataset_namespace,
        "kind": kind,
        "file": dict(file_record) if file_record is not None else None,
        "scene_ids": sorted(str(scene) for scene in scene_ids),
        "scene_ids_sha256": canonical_json_sha256(
            sorted(str(scene) for scene in scene_ids)
        ),
    }


def _benchmark_registry(
    *,
    agile_sources: Sequence[tuple[Mapping[str, Any], Mapping[str, str]]],
    pfpr_source: tuple[Mapping[str, Any], Mapping[str, str]],
    nvos_source: tuple[Mapping[str, Any], Mapping[str, str]],
    spin_source: tuple[Mapping[str, Any], Mapping[str, str]],
    additional_scannet_scene_ids: Sequence[str],
    lerf_scene_ids: Sequence[str],
) -> dict[str, Any]:
    source_records: list[dict[str, Any]] = []
    scannet_scenes: set[str] = set()
    for index, (value, record) in enumerate(agile_sources):
        scenes = _materialization_scene_ids(
            value, label=f"AGILE materialization shard {index}"
        )
        scannet_scenes.update(scenes)
        source_records.append(
            _source_record(
                dataset_namespace="scannet",
                kind="agile3d_query_free_materialization_report",
                file_record=record,
                scene_ids=scenes,
            )
        )
    pfpr_value, pfpr_record = pfpr_source
    pfpr_scenes = _materialization_scene_ids(
        pfpr_value, label="PFPR materialization report"
    )
    scannet_scenes.update(pfpr_scenes)
    source_records.append(
        _source_record(
            dataset_namespace="scannet",
            kind="pfpr_query_free_materialization_report",
            file_record=pfpr_record,
            scene_ids=pfpr_scenes,
        )
    )
    additional = sorted(set(str(scene) for scene in additional_scannet_scene_ids))
    if len(additional) != len(additional_scannet_scene_ids):
        raise ValueError("additional ScanNet benchmark scenes must be unique")
    for scene in additional:
        canonical_physical_space_id(scene)
    scannet_scenes.update(additional)
    source_records.append(
        _source_record(
            dataset_namespace="scannet",
            kind="frozen_scannet_og_scene_registry",
            file_record=None,
            scene_ids=additional,
        )
    )
    nvos_value, nvos_record = nvos_source
    nvos_scenes = _prompt_manifest_scene_ids(nvos_value, label="NVOS manifest")
    source_records.append(
        _source_record(
            dataset_namespace="nvos",
            kind="frozen_prompt_manifest",
            file_record=nvos_record,
            scene_ids=nvos_scenes,
        )
    )
    spin_value, spin_record = spin_source
    spin_scenes = _prompt_manifest_scene_ids(spin_value, label="SPIn manifest")
    source_records.append(
        _source_record(
            dataset_namespace="spin_nerf",
            kind="frozen_prompt_manifest",
            file_record=spin_record,
            scene_ids=spin_scenes,
        )
    )
    lerf = sorted(set(str(scene) for scene in lerf_scene_ids))
    if not lerf or len(lerf) != len(lerf_scene_ids):
        raise ValueError("LERF benchmark scenes must be nonempty and unique")
    source_records.append(
        _source_record(
            dataset_namespace="lerf",
            kind="frozen_protocol_scene_registry",
            file_record=None,
            scene_ids=lerf,
        )
    )
    scannet = sorted(scannet_scenes)
    physical = sorted({canonical_physical_space_id(scene) for scene in scannet})
    registry = {
        "schema": BENCHMARK_REGISTRY_SCHEMA,
        "schema_version": 1,
        "source_records": sorted(
            source_records,
            key=lambda item: (item["dataset_namespace"], item["kind"]),
        ),
        "dataset_scene_ids": {
            "lerf": lerf,
            "nvos": nvos_scenes,
            "scannet": scannet,
            "spin_nerf": spin_scenes,
        },
        "scannet_physical_space_ids": physical,
        "scannet_physical_space_ids_sha256": canonical_json_sha256(physical),
        "source_access": _source_manifest_access(),
    }
    registry["authority_sha256"] = _content_sha256(registry)
    return registry


def _selected_sens_headers(
    archive: Path,
    *,
    excluded_physical_spaces: set[str],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_physical: set[str] = set()
    last_scene = ""
    with tarfile.open(archive, mode="r:") as handle:
        for member in handle:
            name = str(member.name)
            pieces = name.split("/")
            if (
                len(pieces) != 3
                or pieces[0] != "scans"
                or pieces[2] != f"{pieces[1]}.sens"
                or not member.isfile()
            ):
                continue
            scene = pieces[1]
            physical = canonical_physical_space_id(scene)
            if scene < last_scene:
                raise ValueError("ScanNet archive .sens members are not lexicographic")
            last_scene = scene
            if physical in excluded_physical_spaces or physical in seen_physical:
                continue
            seen_physical.add(physical)
            selected.append(
                {
                    "scene_id": scene,
                    "physical_space_id": physical,
                    "archive_member": name,
                    "sens_size_bytes": int(member.size),
                }
            )
            if len(selected) == REQUIRED_SCENES:
                break
    if len(selected) != REQUIRED_SCENES:
        raise RuntimeError(
            f"archive provides only {len(selected)} clean physical spaces; "
            f"{REQUIRED_SCENES} are required"
        )
    return selected


def build(args: argparse.Namespace) -> dict[str, Any]:
    archive = Path(args.scan_archive_part).expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"ScanNet archive part is missing: {archive}")
    agile = [
        _load_json(path, digest, label=f"AGILE report {index}")
        for index, (path, digest) in enumerate(
            zip(args.agile_report, args.expected_agile_report_sha256)
        )
    ]
    if len(agile) != 2 or len(args.agile_report) != len(
        args.expected_agile_report_sha256
    ):
        raise ValueError("exactly two aligned AGILE report/SHA pairs are required")
    pfpr = _load_json(
        args.pfpr_report,
        args.expected_pfpr_report_sha256,
        label="PFPR report",
    )
    nvos = _load_json(
        args.nvos_manifest,
        args.expected_nvos_manifest_sha256,
        label="NVOS manifest",
    )
    spin = _load_json(
        args.spin_manifest,
        args.expected_spin_manifest_sha256,
        label="SPIn manifest",
    )
    registry = _benchmark_registry(
        agile_sources=agile,
        pfpr_source=pfpr,
        nvos_source=nvos,
        spin_source=spin,
        additional_scannet_scene_ids=args.additional_scannet_benchmark_scene_id,
        lerf_scene_ids=args.lerf_scene_id,
    )
    scannet_scenes = registry["dataset_scene_ids"]["scannet"]
    physical_spaces = registry["scannet_physical_space_ids"]
    # Finish every read-only precondition before creating any immutable
    # output.  In particular, a truncated archive or an undersized clean pool
    # must not leave a registry/exclusion pair that prevents a corrected
    # no-clobber rerun.
    selected = _selected_sens_headers(
        archive,
        excluded_physical_spaces=set(physical_spaces),
    )
    registry_path = write_frozen_json(args.benchmark_registry_output, registry)
    registry_file_sha = sha256_file(registry_path)

    exclusion = {
        "schema": BENCHMARK_EXCLUSION_MANIFEST_SCHEMA,
        "schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
        "contract": benchmark_exclusion_manifest_contract(),
        "contract_sha256": BENCHMARK_EXCLUSION_MANIFEST_CONTRACT_SHA256,
        "source_identifier": BENCHMARK_REGISTRY_SCHEMA,
        "source_artifact_sha256": registry_file_sha,
        "scene_ids": scannet_scenes,
        "scene_ids_sha256": canonical_json_sha256(scannet_scenes),
        "physical_space_ids": physical_spaces,
        "physical_space_ids_sha256": canonical_json_sha256(physical_spaces),
        "source_access": _source_manifest_access(),
    }
    exclusion["authority_sha256"] = _manifest_content_sha256(exclusion)
    validate_benchmark_exclusion_manifest(exclusion)
    exclusion_path = write_frozen_json(args.exclusion_manifest_output, exclusion)
    exclusion_file_sha = sha256_file(exclusion_path)

    validation = [
        record for index, record in enumerate(selected) if index % 4 == 3
    ]
    train = [
        record for index, record in enumerate(selected) if index % 4 != 3
    ]
    train_scenes = sorted(record["scene_id"] for record in train)
    validation_scenes = sorted(record["scene_id"] for record in validation)
    cohort = {
        "schema": COHORT_AUTHORITY_SCHEMA,
        "schema_version": COHORT_AUTHORITY_SCHEMA_VERSION,
        "contract": cohort_authority_contract(),
        "contract_sha256": COHORT_AUTHORITY_CONTRACT_SHA256,
        "source_train_scene_ids": train_scenes,
        "source_validation_scene_ids": validation_scenes,
        "source_train_physical_space_ids": sorted(
            record["physical_space_id"] for record in train
        ),
        "source_validation_physical_space_ids": sorted(
            record["physical_space_id"] for record in validation
        ),
        "benchmark_exclusion": {
            "manifest_authority_sha256": exclusion["authority_sha256"],
            "manifest_file_sha256": exclusion_file_sha,
        },
        "source_access": _cohort_authority_access(),
    }
    cohort["authority_sha256"] = cohort_authority_content_sha256(cohort)
    validate_cohort_authority_payload(cohort)
    cohort_path = write_frozen_json(args.cohort_authority_output, cohort)
    cohort_file_sha = sha256_file(cohort_path)

    split_by_scene = {
        **{scene: "source_train" for scene in train_scenes},
        **{scene: "source_validation" for scene in validation_scenes},
    }
    inventory_records = [
        {**record, "split": split_by_scene[record["scene_id"]]}
        for record in selected
    ]
    inventory = {
        "schema": INVENTORY_SCHEMA,
        "schema_version": 1,
        "selection_rule": SELECTION_RULE,
        "archive": {
            "path": str(archive),
            "size_bytes": archive.stat().st_size,
            "container_sha256": None,
            "container_sha256_policy": (
                "deferred_inventory_only_selected_sens_payloads_must_be_"
                "sha256_bound_after_extraction_before_source_use"
            ),
            "payload_content_opened": False,
            "member_headers_opened": True,
        },
        "benchmark_registry": {
            "path": str(registry_path),
            "sha256": registry_file_sha,
            "authority_sha256": registry["authority_sha256"],
        },
        "benchmark_exclusion_manifest": {
            "path": str(exclusion_path),
            "sha256": exclusion_file_sha,
            "authority_sha256": exclusion["authority_sha256"],
            "scene_count": len(scannet_scenes),
            "physical_space_count": len(physical_spaces),
        },
        "cohort_authority": {
            "path": str(cohort_path),
            "sha256": cohort_file_sha,
            "authority_sha256": cohort["authority_sha256"],
        },
        "selected_records": inventory_records,
        "source_train_scene_count": len(train),
        "source_validation_scene_count": len(validation),
        "selected_sens_total_bytes": sum(
            int(record["sens_size_bytes"]) for record in selected
        ),
        "materialization_status": {
            "sens_extracted": False,
            "query_free_rgb_pose_prepared": False,
            "geometry_checkpoint_built": False,
            "official_teacher_features_built": False,
            "exact_responsibility_built": False,
            "factorized_primitive_state_built": False,
            "accepted_v2_region_cache_built": False,
            "training_shards_built": False,
        },
        "source_access": _source_manifest_access(),
    }
    inventory["authority_sha256"] = _content_sha256(inventory)
    inventory_path = write_frozen_json(args.inventory_output, inventory)
    return {
        "benchmark_registry": {
            "path": str(registry_path),
            "sha256": registry_file_sha,
        },
        "benchmark_exclusion_manifest": {
            "path": str(exclusion_path),
            "sha256": exclusion_file_sha,
        },
        "cohort_authority": {
            "path": str(cohort_path),
            "sha256": cohort_file_sha,
        },
        "inventory": {
            "path": str(inventory_path),
            "sha256": sha256_file(inventory_path),
            "selected_sens_total_bytes": inventory[
                "selected_sens_total_bytes"
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-archive-part", required=True)
    parser.add_argument("--agile-report", action="append", required=True)
    parser.add_argument(
        "--expected-agile-report-sha256", action="append", required=True
    )
    parser.add_argument("--pfpr-report", required=True)
    parser.add_argument("--expected-pfpr-report-sha256", required=True)
    parser.add_argument("--nvos-manifest", required=True)
    parser.add_argument("--expected-nvos-manifest-sha256", required=True)
    parser.add_argument("--spin-manifest", required=True)
    parser.add_argument("--expected-spin-manifest-sha256", required=True)
    parser.add_argument(
        "--additional-scannet-benchmark-scene-id",
        action="append",
        default=[],
    )
    parser.add_argument("--lerf-scene-id", action="append", default=[])
    parser.add_argument("--benchmark-registry-output", required=True)
    parser.add_argument("--exclusion-manifest-output", required=True)
    parser.add_argument("--cohort-authority-output", required=True)
    parser.add_argument("--inventory-output", required=True)
    print(json.dumps(build(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
