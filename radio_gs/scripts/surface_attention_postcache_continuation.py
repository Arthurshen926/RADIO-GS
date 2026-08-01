#!/usr/bin/env python3
"""Formal post-cache continuation for the Surface c1024 attention screen.

The parent run completed all ten cache-producing stages but its pairing
authority treated an intentionally absent legacy ``scene_intermediate`` key as
an empty mapping.  This continuation never writes the parent tree.  It binds
the parent manifest, complete attempt inventory, and every control/treatment
cache+sidecar pair before training the six frozen readouts in a new run root.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping

import torch

from radio_gs.scripts import surface_attention_pooling_screen as base
from radio_gs.scripts.surface_region_run_guard import (
    audit_attempt_inventory,
    build_runtime_closure,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    load_torch_mapping,
    sha256_file,
    validate_file_record,
    write_frozen_json,
)


SCREEN_NAME = "surface-c1024-attention-postcache-continuation-v1"
PARENT_SCREEN_NAME = base.SCREEN_NAME
EXPECTED_PARENT_CACHE_STAGES = (
    "cache_control_c256_geometric_train_2",
    "cache_control_c256_geometric_train_3",
    "cache_control_c256_geometric_validation_0",
    "cache_control_c256_geometric_validation_1",
    "cache_context_c1024_geometric_train_0",
    "cache_context_c1024_geometric_train_1",
    "cache_context_c1024_geometric_train_2",
    "cache_context_c1024_geometric_train_3",
    "cache_context_c1024_geometric_validation_0",
    "cache_context_c1024_geometric_validation_1",
)
EXPECTED_CHILD_READOUT_STAGES = tuple(
    f"readout_{variant}_seed{seed}"
    for variant in base.VARIANTS
    for seed in base.SEEDS
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_disjoint_roots(output_root: Path, parent_root: Path) -> None:
    output_root = output_root.resolve()
    parent_root = parent_root.resolve()
    _require(
        output_root != parent_root
        and parent_root not in output_root.parents
        and output_root not in parent_root.parents,
        "continuation OUTPUT_ROOT must be disjoint from the immutable parent",
    )


def _validate_child_inventory(inventory: Mapping) -> None:
    attempts = inventory.get("attempts", [])
    _require(
        len(attempts) == len(EXPECTED_CHILD_READOUT_STAGES)
        and {row["stage"] for row in attempts}
        == set(EXPECTED_CHILD_READOUT_STAGES)
        and all(row["attempt_index"] == 1 for row in attempts)
        and all(row["result"] == "completed" for row in attempts),
        "continuation inventory is not the exact six completed readouts",
    )


def _conditional_sidecar(
    cache_path: Path,
    metadata: Mapping,
    *,
    label: str,
) -> dict[str, str]:
    """Validate exactly the sidecar shape emitted by the frozen builder.

    Absence and an empty mapping are deliberately distinct.  The builder adds
    ``scene_intermediate`` to both metadata and sidecar only when the fastpath
    is enabled, so legacy train0/1 must omit the key on both sides.
    """

    sidecar_path = cache_path.with_suffix(cache_path.suffix + ".json")
    sidecar, sidecar_sha, sidecar_source = load_json_object(
        sidecar_path,
        label=f"{label} sidecar",
    )
    expected = {
        "output": str(cache_path.resolve()),
        "regions": len(metadata["region_records"]),
        "scenes": len(metadata["scene_names"]),
        "failed_scenes": {},
        "split_role": metadata["split_role"],
        "split_file_sha256": metadata["split_file_sha256"],
        "teacher_target_source": metadata["teacher_target_source"],
        "teacher_replay_cache": metadata["teacher_replay_cache"],
        "teacher_replay_authority": metadata.get(
            "teacher_replay_authority", {}
        ),
    }
    if "scene_intermediate" in metadata:
        expected["scene_intermediate"] = metadata["scene_intermediate"]
    _require(sidecar == expected, f"{label} sidecar differs from cache metadata")
    return {"path": str(sidecar_source), "sha256": sidecar_sha}


def _parent_source_closure(parent: Mapping) -> None:
    root = Path(parent["source_snapshot_root"]).resolve(strict=True)
    recorded = parent["runtime_closure"]["repository_sources"]
    _require(
        parent["source_snapshot_tree_sha256"] == recorded["digest"],
        "parent source closure digest differs",
    )
    for relative, digest in recorded["files"].items():
        _require(
            sha256_file(root / relative) == digest,
            f"parent source changed: {relative}",
        )
    validate_file_record(parent["active_runner"], label="parent active runner")
    validate_file_record(parent["active_authority"], label="parent active authority")
    checkpoint = parent["runtime_closure"]["radio_checkpoint"]
    _require(
        sha256_file(parent["radio_checkpoint"]) == checkpoint["sha256"]
        == parent["radio_checkpoint_sha256"],
        "parent RADIO checkpoint changed",
    )


def _control_path(parent: Mapping, role: str, shard: int) -> Path:
    rows = [
        row
        for row in parent["control_sources"]
        if row["role"] == role and int(row["shard"]) == shard
    ]
    _require(len(rows) == 1, "parent control source mapping is not unique")
    row = rows[0]
    return Path(row["cache"]["path"] if "cache" in row else row["cache_path"])


def _validate_parent_inventory(
    parent_manifest_path: Path,
    parent: Mapping,
) -> dict:
    contract = parent["attempt_receipt_contract"]
    inventory = audit_attempt_inventory(
        manifest_path=parent_manifest_path,
        attempt_root=contract["root"],
        log_root=contract["log_root"],
    )
    attempts = inventory["attempts"]
    observed_stages = {row["stage"] for row in attempts}
    _require(
        len(attempts) == len(EXPECTED_PARENT_CACHE_STAGES)
        and observed_stages == set(EXPECTED_PARENT_CACHE_STAGES)
        and all(row["attempt_index"] == 1 for row in attempts)
        and all(row["result"] == "completed" for row in attempts),
        "parent attempt inventory is not the exact ten completed cache stages",
    )
    return inventory


def _validate_cache_bundle(
    parent_manifest_path: Path,
    parent: Mapping,
) -> list[dict]:
    root = parent_manifest_path.parent
    rows: list[dict] = []
    protocol_hashes: set[str] = set()
    teacher_contract_hashes: set[str] = set()
    for role, count, split_path in (
        ("train", base.TRAIN_SHARDS, Path(parent["train_split"])),
        ("validation", base.VALIDATION_SHARDS, Path(parent["validation_split"])),
    ):
        for shard in range(count):
            control_path = _control_path(parent, role, shard)
            expected_sha = None
            expected_builder = parent["implementation_sources"][
                "radio_gs/scripts/build_scannet_surface_region_cache.py"
            ]
            if role == "train" and shard < 2:
                expected_sha = base.LEGACY_CONTROL_SHA256[shard]
                expected_builder = base.LEGACY_BUILDER_SHA256
            control, control_sha, control_source = base._validate_control_payload(
                control_path,
                split_file=split_path,
                split_role=role,
                shard=shard,
                shard_count=count,
                checkpoint_sha256=parent["radio_checkpoint_sha256"],
                expected_sha256=expected_sha,
                expected_builder_sha256=expected_builder,
            )
            if role == "train" and shard < 2:
                external = parent["external_control_authority"]["controls"][shard]
                control_sidecar_source = validate_file_record(
                    external["sidecar"],
                    label=f"parent legacy control train shard {shard} sidecar",
                )
                control_sidecar = {
                    "path": str(control_sidecar_source),
                    "sha256": external["sidecar"]["sha256"],
                }
            else:
                control_sidecar = _conditional_sidecar(
                    control_source,
                    control["metadata"],
                    label=f"parent control {role} shard {shard}",
                )

            treatment_path = (
                root
                / "caches"
                / base.CACHE_NAME
                / f"{role}_shard{shard}.pt"
            )
            treatment, treatment_sha, treatment_source = load_torch_mapping(
                treatment_path,
                map_location="cpu",
                label=f"parent c1024 {role} shard {shard}",
            )
            metadata = treatment.get("metadata", {})
            control_meta = control["metadata"]
            scenes = base._split_scenes(split_path, shard, count)
            row_count = len(scenes) * base.EXPECTED_CACHE_CONTRACT[
                "regions_per_scene"
            ]
            expected_replay_authority: dict[str, str] = {}
            if role == "train" and shard < 2:
                authority_path = Path(
                    parent["legacy_teacher_replay_authorities"][shard]["path"]
                )
                authority_payload, authority_sha, authority_source = load_json_object(
                    authority_path,
                    label=f"parent legacy replay authority train shard {shard}",
                )
                _require(
                    authority_payload
                    == base._legacy_replay_authority_payload(
                        parent_manifest_path, parent, shard
                    ),
                    "parent legacy replay authority payload differs",
                )
                expected_replay_authority = {
                    "path": str(authority_source),
                    "sha256": authority_sha,
                }
            _require(
                metadata.get("schema_version") == 3
                and metadata.get("split_role") == role
                and metadata.get("split_file_sha256") == sha256_file(split_path)
                and metadata.get("scene_names") == scenes
                and metadata.get("scene_region_counts")
                == {
                    scene: base.EXPECTED_CACHE_CONTRACT["regions_per_scene"]
                    for scene in scenes
                }
                and metadata.get("failed_scenes") == {}
                and metadata.get("complete_scene_regions") is True
                and metadata.get("teacher_target_source") == "exact_cache_replay"
                and metadata.get("teacher_regions_saturated") == 0
                and metadata.get("teacher_target_protocol_sha256")
                == control_meta.get("teacher_target_protocol_sha256")
                and metadata.get("teacher_region_contract_sha256")
                == control_meta.get("teacher_region_contract_sha256")
                and metadata.get("radio_checkpoint_sha256")
                == parent["radio_checkpoint_sha256"],
                "parent c1024 cache provenance differs",
            )
            _require(
                metadata.get("builder_script_sha256")
                == parent["implementation_sources"][
                    "radio_gs/scripts/build_scannet_surface_region_cache.py"
                ]
                and metadata.get("teacher_replay_cache")
                == {"path": str(control_source), "sha256": control_sha}
                and metadata.get("teacher_replay_authority", {})
                == expected_replay_authority,
                "parent c1024 implementation/replay binding differs",
            )
            treatment_sidecar = _conditional_sidecar(
                treatment_source,
                metadata,
                label=f"parent c1024 {role} shard {shard}",
            )
            uses_legacy = role == "train" and shard < 2
            control_intermediate = control_meta.get("scene_intermediate", {})
            treatment_intermediate = metadata.get("scene_intermediate", {})
            if uses_legacy:
                _require(
                    "scene_intermediate" not in metadata,
                    "legacy c1024 must omit scene_intermediate on both sides",
                )
            else:
                _require(
                    control_intermediate.get("mode") == "fresh_publish"
                    and treatment_intermediate.get("mode") == "exact_replay"
                    and treatment_intermediate.get("manifest")
                    == control_intermediate.get("manifest")
                    and treatment_intermediate.get("scene_records")
                    == control_intermediate.get("scene_records"),
                    "parent c1024 did not replay its control intermediate",
                )
            base._validate_region_contract(
                metadata, candidate_limit=1024, context_ratio=1.20
            )
            base._validate_tensor_payload(
                treatment, row_count, label="parent c1024 cache"
            )
            _require(
                [base._teacher_identity(row) for row in metadata["region_records"]]
                == [
                    base._teacher_identity(row)
                    for row in control_meta["region_records"]
                ],
                "parent fixed teacher identities differ",
            )
            for key in (
                "official_summary_tokens",
                "official_crop_summaries",
                "teacher_mask",
            ):
                _require(
                    torch.equal(treatment[key], control[key]),
                    f"parent {key} replay differs",
                )
            protocol_hashes.add(str(metadata["teacher_target_protocol_sha256"]))
            teacher_contract_hashes.add(
                str(metadata["teacher_region_contract_sha256"])
            )
            rows.append(
                {
                    "role": role,
                    "shard": shard,
                    "control": {"path": str(control_source), "sha256": control_sha},
                    "control_sidecar": control_sidecar,
                    "c1024": {
                        "path": str(treatment_source),
                        "sha256": treatment_sha,
                    },
                    "c1024_sidecar": treatment_sidecar,
                    "regions": row_count,
                    "teacher_target_protocol_sha256": metadata[
                        "teacher_target_protocol_sha256"
                    ],
                }
            )
    _require(len(protocol_hashes) == 1, "parent c1024 target protocols differ")
    _require(
        len(teacher_contract_hashes) == 1,
        "parent c1024 teacher contracts differ",
    )
    return rows


def validate_parent(parent_manifest_path: Path) -> dict:
    parent, _, parent_source = load_json_object(
        parent_manifest_path,
        label="Surface attention parent manifest",
    )
    _require(
        parent.get("schema_version") == 1
        and parent.get("screen") == PARENT_SCREEN_NAME,
        "wrong Surface attention parent manifest",
    )
    _parent_source_closure(parent)
    inventory = _validate_parent_inventory(parent_source, parent)
    rows = _validate_cache_bundle(parent_source, parent)
    return {
        "parent_manifest": file_record(parent_source),
        "attempt_inventory_digest": inventory["digest"],
        "attempt_receipts": [
            row["receipt"]
            for row in sorted(inventory["attempts"], key=lambda item: item["stage"])
        ],
        "cache_rows": rows,
    }


def create_manifest(args: argparse.Namespace) -> dict:
    repo_root = Path(args.repo_root).resolve()
    output_root = Path(args.output_root).resolve()
    parent_manifest = Path(args.parent_manifest)
    parent, _, parent_source = load_json_object(
        parent_manifest, label="post-cache parent manifest"
    )
    parent_root = parent_source.parent.resolve()
    _require_disjoint_roots(output_root, parent_root)
    validation = validate_parent(parent_source)
    closure = build_runtime_closure(
        repo_root=repo_root,
        radio_repo=parent["radio_repo"],
        radio_checkpoint=parent["radio_checkpoint"],
        checkpoint_sha256=parent["radio_checkpoint_sha256"],
    )
    thermal = dict(parent["thermal_safety_contract"])
    _require(
        thermal.get("peer_gpu") is None
        and thermal.get("physical_gpu") == 1
        and thermal.get("gpu_uuid") == args.gpu_uuid,
        "continuation GPU1 identity differs from parent",
    )
    payload = {
        "schema_version": 1,
        "screen": SCREEN_NAME,
        "source_snapshot_root": closure["runtime_fingerprint"][
            "repository_import_root"
        ],
        "source_snapshot_import_root": closure["runtime_fingerprint"][
            "repository_import_root"
        ],
        "source_snapshot_tree_sha256": closure["repository_sources"]["digest"],
        "radio_repo": parent["radio_repo"],
        "radio_checkpoint": parent["radio_checkpoint"],
        "radio_checkpoint_sha256": parent["radio_checkpoint_sha256"],
        "runner_sha256": closure["repository_sources"]["files"][
            "radio_gs/scripts/run_surface_region_context_recovery_screen.sh"
        ],
        "active_runner": file_record(args.runner),
        "active_authority": file_record(Path(__file__).resolve()),
        "implementation_sources": {
            "base_attention_authority": file_record(
                repo_root / "radio_gs/scripts/surface_attention_pooling_screen.py"
            ),
            "run_guard": file_record(
                repo_root / "radio_gs/scripts/surface_region_run_guard.py"
            ),
            "thermal_guard": file_record(
                repo_root / "radio_gs/scripts/run_with_gpu_thermal_guard.sh"
            ),
            "readout_trainer": file_record(
                repo_root / "radio_gs/scripts/train_surface_region_summary_readout.py"
            ),
        },
        "runtime_closure": closure,
        "continuation_contract": {
            "mode": "post_cache_only_no_parent_mutation_v1",
            "reason": "legacy_no_intermediate_sidecar_optional_key_validator_bug",
            "parent_run_manifest": validation["parent_manifest"],
            "parent_attempt_inventory_digest": validation[
                "attempt_inventory_digest"
            ],
            "parent_attempt_receipts": validation["attempt_receipts"],
            "parent_cache_stage_count": len(EXPECTED_PARENT_CACHE_STAGES),
            "cache_writes_forbidden": True,
        },
        "cache_bundle": validation["cache_rows"],
        "readout_contract": parent["readout_contract"],
        "selection_contract": parent["selection_contract"],
        "thermal_safety_contract": thermal,
        "attempt_receipt_contract": {
            "artifact_type": "surface-region-stage-attempt-v1",
            "schema_version": 1,
            "root": str(output_root / "stage_attempts"),
            "log_root": str(output_root / "logs"),
            "telemetry_path": str(Path(args.telemetry).resolve()),
            "immutable_no_clobber": True,
            "owner_audit_required": True,
            "owner_audit_location": "beside_receipt",
        },
    }
    output = Path(args.manifest)
    if output.is_file():
        previous, _, _ = load_json_object(output, label="continuation manifest")
        _require(previous == payload, "OUTPUT_ROOT belongs to another continuation")
    else:
        existing = [path for path in output_root.rglob("*") if path.is_file()]
        _require(not existing, "continuation OUTPUT_ROOT is not empty")
        write_frozen_json(output, payload)
    return payload


def verify_manifest(path: Path) -> dict:
    manifest, _, source = load_json_object(path, label="continuation manifest")
    _require(
        manifest.get("schema_version") == 1
        and manifest.get("screen") == SCREEN_NAME,
        "wrong post-cache continuation manifest",
    )
    closure = build_runtime_closure(
        repo_root=Path(__file__).resolve().parents[2],
        radio_repo=manifest["radio_repo"],
        radio_checkpoint=manifest["radio_checkpoint"],
        checkpoint_sha256=manifest["radio_checkpoint_sha256"],
    )
    _require(closure == manifest["runtime_closure"], "continuation closure changed")
    validate_file_record(manifest["active_runner"], label="continuation runner")
    validate_file_record(manifest["active_authority"], label="continuation authority")
    for label, record in manifest["implementation_sources"].items():
        validate_file_record(record, label=f"continuation source {label}")
    contract = manifest["continuation_contract"]
    parent_path = validate_file_record(
        contract["parent_run_manifest"], label="continuation parent manifest"
    )
    validation = validate_parent(parent_path)
    _require(
        contract
        == {
            "mode": "post_cache_only_no_parent_mutation_v1",
            "reason": "legacy_no_intermediate_sidecar_optional_key_validator_bug",
            "parent_run_manifest": validation["parent_manifest"],
            "parent_attempt_inventory_digest": validation[
                "attempt_inventory_digest"
            ],
            "parent_attempt_receipts": validation["attempt_receipts"],
            "parent_cache_stage_count": len(EXPECTED_PARENT_CACHE_STAGES),
            "cache_writes_forbidden": True,
        }
        and manifest["cache_bundle"] == validation["cache_rows"],
        "continuation parent/cache binding changed",
    )
    attempt = manifest["attempt_receipt_contract"]
    _require(
        Path(attempt["root"]).resolve() == source.parent / "stage_attempts"
        and Path(attempt["log_root"]).resolve() == source.parent / "logs"
        and attempt["owner_audit_required"] is True
        and attempt["owner_audit_location"] == "beside_receipt",
        "continuation attempt contract differs",
    )
    return manifest


def verify_pairing(manifest_path: Path, output: Path) -> dict:
    manifest = verify_manifest(manifest_path)
    report = {
        "schema_version": 1,
        "artifact_type": "surface_c1024_exact_teacher_pairing",
        "status": "single_c1024_cache_exact_teacher_replay_verified",
        "run_manifest": file_record(manifest_path),
        "parent_run_manifest": manifest["continuation_contract"][
            "parent_run_manifest"
        ],
        "rows": manifest["cache_bundle"],
        "legacy_sidecar_rule": (
            "scene_intermediate_key_absent_iff_frozen_builder_metadata_omits_key"
        ),
        "benchmark_queries_opened": False,
        "benchmark_masks_opened": False,
    }
    write_frozen_json(output, report)
    return report


def finalize(manifest_path: Path, pairing_path: Path, output: Path) -> dict:
    manifest = verify_manifest(manifest_path)
    pairing, _, _ = load_json_object(pairing_path, label="continuation pairing")
    _require(
        pairing.get("run_manifest") == file_record(manifest_path)
        and pairing.get("status")
        == "single_c1024_cache_exact_teacher_replay_verified"
        and pairing.get("rows") == manifest["cache_bundle"],
        "continuation pairing report differs",
    )
    expected_cache_records = {
        (row["role"], int(row["shard"])): row["c1024"]
        for row in pairing["rows"]
    }
    root = manifest_path.parent
    variants: dict[str, dict] = {}
    for variant in base.VARIANTS:
        seed_rows = []
        for seed in base.SEEDS:
            checkpoint_path = root / "readouts" / f"{variant}_seed{seed}.pt"
            checkpoint, checkpoint_sha, checkpoint_source = load_torch_mapping(
                checkpoint_path,
                map_location="cpu",
                label=f"continuation readout {variant} seed {seed}",
            )
            report, _, _ = load_json_object(
                checkpoint_path.with_suffix(checkpoint_path.suffix + ".json"),
                label="continuation readout report",
            )
            architecture = checkpoint.get("architecture", {})
            config = checkpoint.get("training_config", {})
            provenance = checkpoint.get("provenance", {})
            expected_train = [
                expected_cache_records[("train", shard)]["path"]
                for shard in range(base.TRAIN_SHARDS)
            ]
            expected_validation = [
                expected_cache_records[("validation", shard)]["path"]
                for shard in range(base.VALIDATION_SHARDS)
            ]
            _require(
                checkpoint.get("schema_version") == 3
                and architecture.get("context_pooling_mode", base.JOINT_CONTEXT_POOLING)
                == variant
                and config.get("context_pooling_mode") == variant
                and int(config.get("seed", -1)) == seed
                and provenance.get("train", {}).get("cache_paths")
                == expected_train
                and provenance.get("validation", {}).get("cache_paths")
                == expected_validation
                and report.get("checkpoint_sha256") == checkpoint_sha
                and report.get("architecture") == architecture,
                "continuation readout binding differs",
            )
            validation = report.get("validation", {})
            _require(
                all(
                    isinstance(validation.get(key), (int, float))
                    and not isinstance(validation.get(key), bool)
                    and math.isfinite(float(validation[key]))
                    for key in base.ALL_VALIDATION_COMPONENTS
                ),
                "continuation readout metrics differ",
            )
            seed_rows.append(
                {
                    "seed": seed,
                    "checkpoint": {
                        "path": str(checkpoint_source),
                        "sha256": checkpoint_sha,
                    },
                    "best_epoch": int(report["best_epoch"]),
                    "best_selection_score": float(report["best_selection_score"]),
                    "validation": {
                        key: float(validation[key])
                        for key in base.ALL_VALIDATION_COMPONENTS
                    },
                }
            )
        variants[variant] = {
            "seeds": seed_rows,
            "mean_selection_score": sum(
                row["best_selection_score"] for row in seed_rows
            )
            / len(seed_rows),
            "mean_validation": {
                key: sum(row["validation"][key] for row in seed_rows)
                / len(seed_rows)
                for key in base.ALL_VALIDATION_COMPONENTS
            },
        }
    decision = base.promotion_decision(variants, manifest["selection_contract"])
    variants[base.SEPARATE_CONTEXT_POOLING].update(decision)
    attempt_contract = manifest["attempt_receipt_contract"]
    child_inventory = audit_attempt_inventory(
        manifest_path=manifest_path,
        attempt_root=attempt_contract["root"],
        log_root=attempt_contract["log_root"],
    )
    child_attempts = child_inventory["attempts"]
    _validate_child_inventory(child_inventory)
    passed = bool(decision["eligible_for_query_free_promotion"])
    report = {
        "schema_version": 1,
        "artifact_type": "surface_c1024_attention_pooling_postcache_continuation",
        "selection_status": (
            "separate_attention_promoted_benchmark_gate_still_closed"
            if passed
            else "joint_attention_retained"
        ),
        "selected_variant": (
            base.SEPARATE_CONTEXT_POOLING if passed else base.JOINT_CONTEXT_POOLING
        ),
        "promotion_gate_passed": passed,
        "run_manifest": file_record(manifest_path),
        "parent_run_manifest": manifest["continuation_contract"][
            "parent_run_manifest"
        ],
        "cache_pairing_report": file_record(pairing_path),
        "child_attempt_inventory_digest": child_inventory["digest"],
        "child_attempt_receipts": [
            row["receipt"] for row in sorted(
                child_attempts, key=lambda row: row["stage"]
            )
        ],
        "variants": variants,
        "benchmark_queries_opened": False,
        "benchmark_masks_opened": False,
        "next_gate": "freeze winning readout before text-response benchmark",
    }
    write_frozen_json(output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    parent = sub.add_parser("validate-parent")
    parent.add_argument("--parent-manifest", required=True, type=Path)
    create = sub.add_parser("create-manifest")
    for name in (
        "repo-root",
        "output-root",
        "runner",
        "manifest",
        "parent-manifest",
        "telemetry",
        "gpu-uuid",
    ):
        create.add_argument(f"--{name}", required=True)
    verify = sub.add_parser("verify-manifest")
    verify.add_argument("--manifest", required=True, type=Path)
    pairing = sub.add_parser("verify-pairing")
    pairing.add_argument("--manifest", required=True, type=Path)
    pairing.add_argument("--output", required=True, type=Path)
    finish = sub.add_parser("finalize")
    finish.add_argument("--manifest", required=True, type=Path)
    finish.add_argument("--pairing", required=True, type=Path)
    finish.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "validate-parent":
        result = validate_parent(args.parent_manifest)
    elif args.command == "create-manifest":
        result = create_manifest(args)
    elif args.command == "verify-manifest":
        result = verify_manifest(args.manifest)
    elif args.command == "verify-pairing":
        result = verify_pairing(args.manifest, args.output)
    else:
        result = finalize(args.manifest, args.pairing, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
