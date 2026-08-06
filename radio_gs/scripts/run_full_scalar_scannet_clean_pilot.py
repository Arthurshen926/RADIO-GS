#!/usr/bin/env python3
"""Run one sealed clean-ScanNet full-scalar construction pilot.

The launcher is deliberately narrower than a benchmark queue.  It accepts one
scene already frozen by the clean 24/8 cohort inventory, proves canonical
``scene####`` physical-space separation, extracts exactly one declared
``.sens`` archive member, and then runs only query/label-free construction
stages.  Every completed stage writes a no-clobber receipt that binds the
three externally supplied protocol files, the previous receipt, all direct
file inputs/outputs, the exact command, and source-access flags.

An interrupted run is resumable at completed stage boundaries.  If a stage
left unreceipted outputs, the launcher fails closed instead of adopting or
overwriting them; use a new pilot root for a fresh attempt.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile
from typing import Any, Callable

from radio_gs.scripts.build_full_scalar_scannet_clean_cohort import (
    INVENTORY_SCHEMA,
    _content_sha256,
)
from radio_gs.scripts.train_surface_region_full_scalar_residual import (
    canonical_physical_space_id,
    validate_benchmark_exclusion_manifest,
    validate_cohort_authority_payload,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_json_object,
    sha256_file,
    write_frozen_json,
)


STAGE_RECEIPT_SCHEMA = "radio_gs.full_scalar_scannet_clean_pilot_stage.v1"
FINAL_RECEIPT_SCHEMA = "radio_gs.full_scalar_scannet_clean_pilot_receipt.v1"
STAGES = (
    "sens_extraction",
    "query_free_materialization",
    "geometry",
    "radio_features",
    "render_contract",
    "factorized_radio",
    "exact_raw_reference",
    "exact_dino_v3",
    "exact_sam3",
    "capability_cohort",
    "factorized_field",
    "factorized_state",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
RADIO_CHECKPOINT = Path(
    "/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar"
)
RADIO_CHECKPOINT_SHA256 = (
    "bace44df72e750bc8555ea6979cc19d1a87e12ade89582edfe090513d5d6aab9"
)
IMMUTABLE_AUTHORITY_KEYS = (
    "cohort_authority",
    "exclusion_manifest",
    "inventory",
    "launcher_implementation",
)


def _require_sha256(value: object, *, label: str) -> str:
    digest = str(value)
    if _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _file_record(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "size_bytes": source.stat().st_size,
    }


def _validate_file_record(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise ValueError(f"{label} file record differs")
    path = Path(str(value["path"])).expanduser().resolve()
    if (
        path.stat().st_size != int(value["size_bytes"])
        or sha256_file(path) != _require_sha256(value["sha256"], label=label)
    ):
        raise ValueError(f"{label} file differs")
    return dict(value)


def _source_access() -> dict[str, bool]:
    return {
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_queries_opened": False,
        "benchmark_labels_opened": False,
        "target_heldout_opened": False,
        "text_queries_opened": False,
        "target_metrics_computed": False,
    }


def _load_json(path: str | Path, expected_sha256: str, *, label: str):
    return load_json_object(
        path,
        expected_sha256=_require_sha256(expected_sha256, label=label),
        label=label,
    )


def validate_pilot_authorities(args: argparse.Namespace) -> dict[str, Any]:
    cohort, cohort_sha, cohort_path = _load_json(
        args.cohort_authority,
        args.expected_cohort_authority_sha256,
        label="clean cohort authority",
    )
    exclusion, exclusion_sha, exclusion_path = _load_json(
        args.exclusion_manifest,
        args.expected_exclusion_manifest_sha256,
        label="benchmark exclusion manifest",
    )
    inventory, inventory_sha, inventory_path = _load_json(
        args.inventory,
        args.expected_inventory_sha256,
        label="clean cohort inventory",
    )
    validate_cohort_authority_payload(cohort)
    validate_benchmark_exclusion_manifest(exclusion)
    if (
        inventory.get("schema") != INVENTORY_SCHEMA
        or inventory.get("schema_version") != 1
        or inventory.get("authority_sha256") != _content_sha256(inventory)
        or inventory.get("source_access", {}).get("benchmark_labels_opened")
        is not False
        or any(inventory.get("source_access", {}).values())
    ):
        raise ValueError("clean cohort inventory contract differs")
    inventory_cohort = inventory.get("cohort_authority")
    inventory_exclusion = inventory.get("benchmark_exclusion_manifest")
    if (
        not isinstance(inventory_cohort, Mapping)
        or inventory_cohort.get("sha256") != cohort_sha
        or inventory_cohort.get("authority_sha256")
        != cohort.get("authority_sha256")
        or not isinstance(inventory_exclusion, Mapping)
        or inventory_exclusion.get("sha256") != exclusion_sha
        or inventory_exclusion.get("authority_sha256")
        != exclusion.get("authority_sha256")
        or cohort.get("benchmark_exclusion", {}).get("manifest_file_sha256")
        != exclusion_sha
        or cohort.get("benchmark_exclusion", {}).get(
            "manifest_authority_sha256"
        )
        != exclusion.get("authority_sha256")
    ):
        raise ValueError("clean cohort authority chain differs")

    scene = str(args.scene_id)
    physical = canonical_physical_space_id(scene)
    train = list(cohort["source_train_scene_ids"])
    validation = list(cohort["source_validation_scene_ids"])
    if (scene in train) == (scene in validation):
        raise ValueError("pilot scene is not a unique clean cohort member")
    split = "source_train" if scene in train else "source_validation"
    expected_physical = (
        cohort["source_train_physical_space_ids"]
        if split == "source_train"
        else cohort["source_validation_physical_space_ids"]
    )
    if physical not in expected_physical:
        raise ValueError("pilot physical space differs from cohort authority")
    if physical in exclusion["physical_space_ids"]:
        raise ValueError("pilot physical space occurs in benchmark exclusion")
    records = [
        record
        for record in inventory.get("selected_records", [])
        if isinstance(record, Mapping) and record.get("scene_id") == scene
    ]
    if len(records) != 1:
        raise ValueError("pilot scene has no unique inventory record")
    record = dict(records[0])
    expected_member = f"scans/{scene}/{scene}.sens"
    if (
        record.get("physical_space_id") != physical
        or record.get("split") != split
        or record.get("archive_member") != expected_member
        or int(record.get("sens_size_bytes", -1)) <= 0
    ):
        raise ValueError("pilot inventory record differs")
    archive = Path(args.scan_archive_part).expanduser().resolve()
    archive_record = inventory.get("archive")
    if (
        not archive.is_file()
        or not isinstance(archive_record, Mapping)
        or Path(str(archive_record.get("path", ""))).expanduser().resolve()
        != archive
        or int(archive_record.get("size_bytes", -1)) != archive.stat().st_size
        or archive_record.get("payload_content_opened") is not False
        or archive_record.get("member_headers_opened") is not True
    ):
        raise ValueError("pilot archive authority differs")
    return {
        "scene_id": scene,
        "physical_space_id": physical,
        "split": split,
        "archive": {
            "path": str(archive),
            "size_bytes": archive.stat().st_size,
            "member": expected_member,
            "member_size_bytes": int(record["sens_size_bytes"]),
        },
        "cohort_authority": {
            "path": str(cohort_path),
            "sha256": cohort_sha,
            "authority_sha256": cohort["authority_sha256"],
        },
        "exclusion_manifest": {
            "path": str(exclusion_path),
            "sha256": exclusion_sha,
            "authority_sha256": exclusion["authority_sha256"],
        },
        "inventory": {
            "path": str(inventory_path),
            "sha256": inventory_sha,
            "authority_sha256": inventory["authority_sha256"],
        },
        "launcher_implementation": _file_record(Path(__file__).resolve()),
    }


def _receipt_content_sha256(value: Mapping[str, Any]) -> str:
    content = dict(value)
    content.pop("authority_sha256", None)
    return canonical_json_sha256(content)


def _validate_stage_receipt(
    value: object,
    *,
    stage: str,
    common: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{stage} stage receipt must be a mapping")
    receipt = dict(value)
    if (
        receipt.get("schema") != STAGE_RECEIPT_SCHEMA
        or receipt.get("schema_version") != 1
        or receipt.get("stage") != stage
        or receipt.get("scene_id") != common["scene_id"]
        or receipt.get("physical_space_id") != common["physical_space_id"]
        or receipt.get("split") != common["split"]
        or receipt.get("immutable_authorities")
        != {key: common[key] for key in IMMUTABLE_AUTHORITY_KEYS}
        or receipt.get("source_access") != _source_access()
        or any(receipt.get("source_access", {}).values())
        or receipt.get("authority_sha256") != _receipt_content_sha256(receipt)
    ):
        raise ValueError(f"{stage} stage receipt contract differs")
    inputs = receipt.get("inputs")
    outputs = receipt.get("outputs")
    if not isinstance(inputs, Mapping) or not isinstance(outputs, Mapping):
        raise ValueError(f"{stage} receipt inputs/outputs differ")
    for label, record in [*inputs.items(), *outputs.items()]:
        if isinstance(record, Mapping) and set(record) == {
            "path",
            "sha256",
            "size_bytes",
        }:
            _validate_file_record(record, label=f"{stage} {label}")
    return receipt


def _write_stage_receipt(
    path: Path,
    *,
    stage: str,
    common: Mapping[str, Any],
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Any],
    command: Sequence[str] | None,
) -> dict[str, Any]:
    receipt = {
        "schema": STAGE_RECEIPT_SCHEMA,
        "schema_version": 1,
        "stage": stage,
        "scene_id": common["scene_id"],
        "physical_space_id": common["physical_space_id"],
        "split": common["split"],
        "immutable_authorities": {
            key: common[key] for key in IMMUTABLE_AUTHORITY_KEYS
        },
        "inputs": dict(inputs),
        "outputs": dict(outputs),
        "command": list(command) if command is not None else None,
        "command_sha256": (
            canonical_json_sha256(list(command)) if command is not None else None
        ),
        "source_access": _source_access(),
    }
    receipt["authority_sha256"] = _receipt_content_sha256(receipt)
    write_frozen_json(path, receipt)
    return receipt


def extract_inventory_sens(
    *,
    common: Mapping[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(
            f"unreceipted .sens output already exists: {output_path}"
        )
    archive = common["archive"]
    expected_name = str(archive["member"])
    expected_size = int(archive["member_size_bytes"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tarfile.open(str(archive["path"]), mode="r:") as handle:
            matches = []
            for member in handle:
                if member.name == expected_name:
                    matches.append(member)
                    break
            if len(matches) != 1:
                raise ValueError("sealed .sens archive member is missing")
            member = matches[0]
            if not member.isfile() or int(member.size) != expected_size:
                raise ValueError("sealed .sens archive header differs")
            source = handle.extractfile(member)
            if source is None:
                raise ValueError("sealed .sens archive payload is unreadable")
            digest = hashlib.sha256()
            written = 0
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                dir=output_path.parent,
                delete=False,
            ) as target:
                temporary_name = target.name
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    target.write(block)
                    digest.update(block)
                    written += len(block)
                target.flush()
                os.fsync(target.fileno())
        if written != expected_size:
            raise ValueError("extracted .sens payload size differs")
        os.link(temporary_name, output_path)
        os.unlink(temporary_name)
        temporary_name = None
        record = _file_record(output_path)
        if record["sha256"] != digest.hexdigest():
            raise ValueError("extracted .sens payload changed after publication")
        return record
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _run(command: Sequence[str], *, log_path: Path, env: Mapping[str, str] | None = None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        raise FileExistsError(f"unreceipted stage log already exists: {log_path}")
    with log_path.open("xb") as log:
        subprocess.run(
            list(command),
            cwd=Path(__file__).resolve().parents[2],
            env=(dict(os.environ) | dict(env or {})),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _thermal_command(args: argparse.Namespace, command: Sequence[str]) -> list[str]:
    return [
        "bash",
        "radio_gs/scripts/run_with_gpu_thermal_guard.sh",
        "--",
        "env",
        f"CUDA_VISIBLE_DEVICES={args.gpu}",
        *command,
    ]


def _thermal_env(args: argparse.Namespace, telemetry: Path) -> dict[str, str]:
    return {
        "GPU": str(args.gpu),
        "GPU_TELEMETRY_LOG": str(telemetry),
        "GPU_MAX_TEMP_C": "88",
        "GPU_START_MAX_TEMP_C": "82",
        "GPU_MAX_POWER_LIMIT_W": "300.5",
        "GPU_POLL_SECONDS": "30",
        "GPU_MAX_CONSECUTIVE_OVERHEAT_POLLS": "2",
        "GPU_MAX_CONSECUTIVE_TELEMETRY_FAILURES": "3",
    }


def _radio_feature_command(
    *,
    scene: str,
    image_dir: Path,
    feature_dir: Path,
) -> list[str]:
    """Build the only feature-extraction command accepted by the validator.

    ``_load_feature_bundle`` requires the extractor's strict atomic-resume
    contract.  Keeping this command in one helper prevents the producer from
    silently emitting a non-resumable manifest that its own stage validator
    must reject.
    """

    return [
        "bash",
        "radio_gs/scripts/run_repo_python.sh",
        "radio_gs/scripts/extract_radio_features.py",
        "--scene",
        str(scene),
        "--image_dir",
        str(image_dir),
        "--output_dir",
        str(feature_dir),
        "--radio_repo",
        "/root/RADIO",
        "--radio_version",
        "c-radio_v4-h",
        "--batch_size",
        "1",
        "--resolution_scale",
        "0.5",
        "--skip_pca_stats",
        "--resume-partial",
    ]


def _require_absent(paths: Sequence[Path], *, stage: str) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            f"{stage} has unreceipted outputs and will not overwrite them: "
            + ", ".join(existing)
        )


def _command_stage(
    *,
    args: argparse.Namespace,
    stage: str,
    command: Sequence[str],
    log_path: Path,
    expected_paths: Sequence[Path],
    output_builder: Callable[[], Mapping[str, Any]],
    gpu: bool,
) -> tuple[dict[str, Any], list[str]]:
    _require_absent([*expected_paths, log_path], stage=stage)
    executed = _thermal_command(args, command) if gpu else list(command)
    _run(
        executed,
        log_path=log_path,
        env=(
            _thermal_env(args, Path(args.pilot_root) / "gpu1_telemetry.csv")
            if gpu
            else None
        ),
    )
    missing = [str(path) for path in expected_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"{stage} did not produce outputs: {missing}")
    outputs = dict(output_builder())
    outputs["stage_log"] = _file_record(log_path)
    return outputs, executed


def _mpr_command(
    *,
    args: argparse.Namespace,
    config: Path,
    checkpoint: Path,
    feature_dir: Path,
    feature_bundle_sha256: str,
    geometry_sha256: str,
    output: Path,
    feature_space: str,
    observation_contract: str,
    responsibility: Path | None = None,
    responsibility_sha256: str = "",
    save_responsibility: Path | None = None,
    normalize_each_view: bool,
) -> list[str]:
    command = [
        "bash",
        "radio_gs/scripts/run_repo_python.sh",
        "radio_gs/scripts/build_gaussian_multiview_teacher_cache.py",
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--device",
        "cuda:0",
        "--observation-contract",
        observation_contract,
        "--max-views",
        "120",
        "--feature-space",
        feature_space,
        "--expected-feature-scene",
        args.scene_id,
        "--expected-feature-image-dir",
        str(Path(args.pilot_root) / "field240" / args.scene_id / "color"),
        "--expected-geometry-checkpoint-sha256",
        geometry_sha256,
        "--expected-feature-output-bundle-sha256",
        feature_bundle_sha256,
        "--aggregation-mode",
        "raster_marginal_responsibility",
        "--registration-weight-mode",
        "alpha_depth",
        "--raster-view-fusion",
        "contribution_mean",
        "--no-robust-mpr",
    ]
    if normalize_each_view:
        command.append("--normalize-each-view")
    if responsibility is not None:
        command.extend(
            [
                "--responsibility-cache",
                str(responsibility),
                "--expected-responsibility-cache-sha256",
                responsibility_sha256,
            ]
        )
    if save_responsibility is not None:
        command.extend(["--save-responsibility-cache", str(save_responsibility)])
    if feature_space in {"dino_v3", "sam3"}:
        command.extend(
            [
                "--radio-checkpoint",
                str(RADIO_CHECKPOINT),
                "--capability-map-source",
                "project_raw",
            ]
        )
    return command


def _load_feature_bundle(feature_dir: Path) -> tuple[dict[str, Any], str]:
    from radio_gs.scripts.extract_radio_features import (
        _validate_final_output_bundle,
    )

    validation = _validate_final_output_bundle(
        feature_dir,
        verify_source_images=False,
    )
    digest = _require_sha256(
        validation["output_bundle_sha256"], label="feature output bundle"
    )
    return dict(validation), digest


def _common_inputs(
    common: Mapping[str, Any], previous_receipt: Path | None
) -> dict[str, Any]:
    inputs: dict[str, Any] = {
        "archive_authority": dict(common["archive"]),
    }
    if previous_receipt is not None:
        inputs["previous_stage_receipt"] = _file_record(previous_receipt)
    return inputs


def run(args: argparse.Namespace) -> dict[str, Any]:
    common = validate_pilot_authorities(args)
    root = Path(args.pilot_root).expanduser().resolve()
    receipt_root = root / "receipts"
    log_root = root / "logs"
    receipt_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    scene = common["scene_id"]
    source_sens = root / "source" / "scans" / scene / f"{scene}.sens"
    field_root = root / "field240"
    scene_root = field_root / scene
    materialization_report = field_root / "materialization_report.json"
    field_contract = scene_root / "field_source_contract.json"
    run_root = root / "run"
    geometry_root = run_root / "geometry"
    geometry_dir = geometry_root / scene
    geometry_final = geometry_dir / "final.pth"
    geometry_ply = (
        geometry_dir / "point_cloud" / "iteration_15000" / "point_cloud.ply"
    )
    feature_dir = run_root / "radio_features" / scene
    feature_manifest = feature_dir / "frame_manifest.json"
    contract_root = run_root / "render_contracts"
    config = contract_root / f"{scene}.yaml"
    render_checkpoint = contract_root / f"{scene}.geometry_renderer.pth"
    exact_root = run_root / "exact_state" / scene
    factorized_radio = exact_root / "factorized_raw_radio.pt"
    responsibility = exact_root / "exact_marginal_responsibility_authority.json"
    exact_raw = exact_root / "raw_radio_matched_exact_marginal.pt"
    exact_dino = exact_root / "dino_v3_matched_exact_marginal.pt"
    exact_sam = exact_root / "sam3_matched_exact_marginal.pt"
    capability_authority = exact_root / "factorized_capability_cohort.json"
    field_checkpoint = exact_root / "factorized_field_d512_l512_exact_marginal.pth"
    factorized_state = exact_root / "factorized_primitive_state_v2.pt"
    exact_root.mkdir(parents=True, exist_ok=True)
    contract_root.mkdir(parents=True, exist_ok=True)

    stop_index = STAGES.index(args.stop_after)
    previous_receipt: Path | None = None
    receipts: list[Path] = []
    stage_outputs: dict[str, Mapping[str, Any]] = {}

    for stage_index, stage in enumerate(STAGES):
        if stage_index > stop_index:
            break
        receipt_path = receipt_root / f"{stage_index:02d}_{stage}.json"
        if receipt_path.is_file():
            value, _digest, _source = load_json_object(
                receipt_path, label=f"{stage} stage receipt"
            )
            receipt = _validate_stage_receipt(value, stage=stage, common=common)
            receipts.append(receipt_path)
            previous_receipt = receipt_path
            stage_outputs[stage] = dict(receipt["outputs"])
            continue

        inputs = _common_inputs(common, previous_receipt)
        outputs: dict[str, Any]
        command: list[str] | None = None
        if stage == "sens_extraction":
            outputs = {"sens_payload": extract_inventory_sens(
                common=common, output_path=source_sens
            )}
            outputs["archive_member"] = {
                "path": common["archive"]["path"],
                "member": common["archive"]["member"],
                "member_size_bytes": common["archive"]["member_size_bytes"],
                "payload_sha256": outputs["sens_payload"]["sha256"],
            }
        elif stage == "query_free_materialization":
            inputs["sens_payload"] = _file_record(source_sens)
            command = [
                "bash",
                "radio_gs/scripts/run_repo_python.sh",
                "-m",
                "radio_gs.benchmarks.scannet_pfpr.prepare_field_contract",
                "--full-scannet-observations",
                "--sens-root",
                str(source_sens.parents[1]),
                "--output-root",
                str(field_root),
                "--scenes",
                scene,
                "--max-frames",
                "240",
                "--candidate-stride",
                "1",
                "--frame-selection-policy",
                "depth_voxel_coverage",
                "--pose-orientation-weight",
                "0.25",
                "--coverage-voxel-size-m",
                "0.05",
                "--coverage-depth-stride",
                "8",
            ]

            def materialization_outputs():
                report, _sha, _path = load_json_object(
                    materialization_report, label="pilot materialization report"
                )
                rows = report.get("scenes")
                if (
                    report.get("valid") is not True
                    or report.get("uses_instances_or_semantic_labels") is not False
                    or report.get("uses_private_anchor") is not False
                    or report.get("uses_private_depth_pixel") is not False
                    or not isinstance(rows, list)
                    or len(rows) != 1
                    or rows[0].get("scene_id") != scene
                    or rows[0].get("source_sens_sha256")
                    != inputs["sens_payload"]["sha256"]
                    or int(rows[0].get("field_frame_count", -1)) != 240
                ):
                    raise ValueError("pilot materialization report differs")
                return {
                    "materialization_report": _file_record(materialization_report),
                    "field_source_contract": _file_record(field_contract),
                    "field_frame_manifest_sha256": _require_sha256(
                        rows[0]["field_frame_manifest_sha256"],
                        label="field frame manifest",
                    ),
                }

            outputs, command = _command_stage(
                args=args,
                stage=stage,
                command=command,
                log_path=log_root / f"{stage}.log",
                expected_paths=[materialization_report, field_contract],
                output_builder=materialization_outputs,
                gpu=False,
            )
        elif stage == "geometry":
            inputs["field_source_contract"] = _file_record(field_contract)
            command = [
                "bash",
                "radio_gs/scripts/run_repo_python.sh",
                "radio_gs/scripts/train_scannet_gs.py",
                "--scene_root",
                str(scene_root),
                "--scene",
                scene,
                "--output_dir",
                str(geometry_root),
                "--iters",
                "15000",
                "--frame_stride",
                "1",
                "--init_frames",
                "50",
                "--init_stride",
                "8",
                "--max_points",
                "200000",
                "--init-selection-policy",
                "coverage_prefix",
                "--field-source-contract",
                str(field_contract),
            ]
            outputs, command = _command_stage(
                args=args,
                stage=stage,
                command=command,
                log_path=log_root / f"{stage}.log",
                expected_paths=[geometry_final, geometry_ply],
                output_builder=lambda: {
                    "geometry_checkpoint": _file_record(geometry_final),
                    "geometry_ply": _file_record(geometry_ply),
                },
                gpu=True,
            )
        elif stage == "radio_features":
            inputs["field_source_contract"] = _file_record(field_contract)
            command = _radio_feature_command(
                scene=scene,
                image_dir=scene_root / "color",
                feature_dir=feature_dir,
            )

            def feature_outputs():
                validation, bundle_sha = _load_feature_bundle(feature_dir)
                return {
                    "feature_manifest": _file_record(feature_manifest),
                    "feature_output_bundle_sha256": bundle_sha,
                    "num_frames": int(validation["num_frames"]),
                }

            outputs, command = _command_stage(
                args=args,
                stage=stage,
                command=command,
                log_path=log_root / f"{stage}.log",
                expected_paths=[feature_manifest],
                output_builder=feature_outputs,
                gpu=True,
            )
        elif stage == "render_contract":
            inputs["geometry_ply"] = _file_record(geometry_ply)
            inputs["feature_manifest"] = _file_record(feature_manifest)
            command = [
                "bash",
                "radio_gs/scripts/run_repo_python.sh",
                "radio_gs/scripts/build_geometry_render_contract.py",
                "--ply-path",
                str(geometry_ply),
                "--scene-root",
                str(scene_root),
                "--feature-dir",
                str(feature_dir),
                "--output-config",
                str(config),
                "--output-checkpoint",
                str(render_checkpoint),
                "--observation-contract",
                "scannet_full_observation_v1",
            ]
            outputs, command = _command_stage(
                args=args,
                stage=stage,
                command=command,
                log_path=log_root / f"{stage}.log",
                expected_paths=[config, render_checkpoint],
                output_builder=lambda: {
                    "render_config": _file_record(config),
                    "render_checkpoint": _file_record(render_checkpoint),
                },
                gpu=False,
            )
        elif stage in {
            "factorized_radio",
            "exact_raw_reference",
            "exact_dino_v3",
            "exact_sam3",
        }:
            feature_sha = _require_sha256(
                stage_outputs["radio_features"][
                    "feature_output_bundle_sha256"
                ],
                label="feature output bundle",
            )
            geometry_sha = sha256_file(render_checkpoint)
            inputs.update(
                {
                    "render_config": _file_record(config),
                    "render_checkpoint": _file_record(render_checkpoint),
                    "feature_manifest": _file_record(feature_manifest),
                    "feature_output_bundle_sha256": feature_sha,
                }
            )
            if stage == "factorized_radio":
                output = factorized_radio
                feature_space = "radio"
                contract = "canonical-factorized-radio-v1"
                loaded_responsibility = None
                save_responsibility = responsibility
                normalize = False
            else:
                output = {
                    "exact_raw_reference": exact_raw,
                    "exact_dino_v3": exact_dino,
                    "exact_sam3": exact_sam,
                }[stage]
                feature_space = {
                    "exact_raw_reference": "radio",
                    "exact_dino_v3": "dino_v3",
                    "exact_sam3": "sam3",
                }[stage]
                contract = "canonical-exact-marginal-mpr-v1"
                loaded_responsibility = responsibility
                save_responsibility = None
                normalize = True
                inputs["responsibility_authority"] = _file_record(responsibility)
            command = _mpr_command(
                args=args,
                config=config,
                checkpoint=render_checkpoint,
                feature_dir=feature_dir,
                feature_bundle_sha256=feature_sha,
                geometry_sha256=geometry_sha,
                output=output,
                feature_space=feature_space,
                observation_contract=contract,
                responsibility=loaded_responsibility,
                responsibility_sha256=(
                    sha256_file(responsibility)
                    if loaded_responsibility is not None
                    else ""
                ),
                save_responsibility=save_responsibility,
                normalize_each_view=normalize,
            )
            expected = [output, output.with_suffix(output.suffix + ".json")]
            if save_responsibility is not None:
                expected.append(responsibility)

            def mpr_outputs():
                result = {
                    "mpr_cache": _file_record(output),
                    "mpr_report": _file_record(
                        output.with_suffix(output.suffix + ".json")
                    ),
                }
                if save_responsibility is not None:
                    result["responsibility_authority"] = _file_record(
                        responsibility
                    )
                return result

            outputs, command = _command_stage(
                args=args,
                stage=stage,
                command=command,
                log_path=log_root / f"{stage}.log",
                expected_paths=expected,
                output_builder=mpr_outputs,
                gpu=True,
            )
        elif stage == "capability_cohort":
            inputs.update(
                {
                    "factorized_radio": _file_record(factorized_radio),
                    "exact_raw": _file_record(exact_raw),
                    "exact_dino_v3": _file_record(exact_dino),
                    "exact_sam3": _file_record(exact_sam),
                }
            )
            feature_sha = _require_sha256(
                stage_outputs["radio_features"][
                    "feature_output_bundle_sha256"
                ],
                label="feature output bundle",
            )
            _require_absent([capability_authority], stage=stage)
            authority = {
                "schema_version": 1,
                "artifact_type": "factorized_capability_cohort_authority",
                "experiment": (
                    "canonical-factorized-radio-v1-formal-capability-cohort"
                ),
                "scene": scene,
                "physical_space_id": common["physical_space_id"],
                "feature_output_bundle_sha256": feature_sha,
                "frozen_cache_authorities": {
                    "radio": {
                        "path": str(exact_raw),
                        "sha256": sha256_file(exact_raw),
                    },
                    "dino_v3": {
                        "path": str(exact_dino),
                        "sha256": sha256_file(exact_dino),
                    },
                    "sam3": {
                        "path": str(exact_sam),
                        "sha256": sha256_file(exact_sam),
                    },
                },
                "target_access": {
                    "benchmark_images_opened": False,
                    "benchmark_masks_opened": False,
                    "text_queries_opened": False,
                    "target_metrics_used_for_selection": False,
                },
            }
            write_frozen_json(capability_authority, authority)
            outputs = {"capability_cohort": _file_record(capability_authority)}
        elif stage == "factorized_field":
            if sha256_file(RADIO_CHECKPOINT) != RADIO_CHECKPOINT_SHA256:
                raise ValueError("official RADIO checkpoint SHA-256 differs")
            feature_sha = _require_sha256(
                stage_outputs["radio_features"][
                    "feature_output_bundle_sha256"
                ],
                label="feature output bundle",
            )
            inputs.update(
                {
                    "official_radio_checkpoint": _file_record(RADIO_CHECKPOINT),
                    "factorized_radio": _file_record(factorized_radio),
                    "exact_raw": _file_record(exact_raw),
                    "exact_dino_v3": _file_record(exact_dino),
                    "exact_sam3": _file_record(exact_sam),
                    "capability_cohort": _file_record(capability_authority),
                }
            )
            command = [
                "bash",
                "radio_gs/scripts/run_repo_python.sh",
                "radio_gs/scripts/train_canonical_radio_field.py",
                "--mpr-cache",
                str(factorized_radio),
                "--expected-mpr-cache-sha256",
                sha256_file(factorized_radio),
                "--observation-contract",
                "canonical-factorized-radio-v1",
                "--radio-checkpoint",
                str(RADIO_CHECKPOINT),
                "--expected-radio-checkpoint-sha256",
                RADIO_CHECKPOINT_SHA256,
                "--expected-feature-output-bundle-sha256",
                feature_sha,
                "--output",
                str(field_checkpoint),
                "--device",
                "cuda:0",
                "--coefficient-dim",
                "512",
                "--local-dim",
                "512",
                "--no-fusion-reliability",
                "--primitive-fusion",
                "--basis-fit-device",
                "cuda:0",
                "--official-capability-loss",
                "--capability-target-contract",
                "matched_exact_marginal",
                "--dino-mpr-cache",
                str(exact_dino),
                "--expected-dino-v3-mpr-cache-sha256",
                sha256_file(exact_dino),
                "--sam3-mpr-cache",
                str(exact_sam),
                "--expected-sam3-mpr-cache-sha256",
                sha256_file(exact_sam),
                "--factorized-capability-reference-mpr-cache",
                str(exact_raw),
                "--expected-factorized-capability-reference-mpr-cache-sha256",
                sha256_file(exact_raw),
                "--factorized-capability-cohort-authority",
                str(capability_authority),
                "--expected-factorized-capability-cohort-authority-sha256",
                sha256_file(capability_authority),
                "--epochs",
                "20",
                "--min-epochs",
                "20",
                "--batch-size",
                "4096",
                "--eval-batch-size",
                "16384",
                "--learning-rate",
                "0.002",
                "--weight-decay",
                "0.00001",
                "--validation-fraction",
                "0.05",
                "--target-cosine",
                "0.985",
                "--seed",
                "0",
            ]
            outputs, command = _command_stage(
                args=args,
                stage=stage,
                command=command,
                log_path=log_root / f"{stage}.log",
                expected_paths=[
                    field_checkpoint,
                    field_checkpoint.with_suffix(field_checkpoint.suffix + ".json"),
                ],
                output_builder=lambda: {
                    "field_checkpoint": _file_record(field_checkpoint),
                    "field_report": _file_record(
                        field_checkpoint.with_suffix(
                            field_checkpoint.suffix + ".json"
                        )
                    ),
                },
                gpu=True,
            )
        elif stage == "factorized_state":
            inputs.update(
                {
                    "field_checkpoint": _file_record(field_checkpoint),
                    "factorized_radio": _file_record(factorized_radio),
                }
            )
            command = [
                "bash",
                "radio_gs/scripts/run_repo_python.sh",
                "radio_gs/scripts/build_factorized_primitive_state.py",
                "--field-checkpoint",
                str(field_checkpoint),
                "--expected-field-checkpoint-sha256",
                sha256_file(field_checkpoint),
                "--factorized-radio-cache",
                str(factorized_radio),
                "--expected-factorized-radio-cache-sha256",
                sha256_file(factorized_radio),
                "--output",
                str(factorized_state),
            ]
            outputs, command = _command_stage(
                args=args,
                stage=stage,
                command=command,
                log_path=log_root / f"{stage}.log",
                expected_paths=[
                    factorized_state,
                    factorized_state.with_suffix(factorized_state.suffix + ".json"),
                ],
                output_builder=lambda: {
                    "factorized_state": _file_record(factorized_state),
                    "factorized_state_report": _file_record(
                        factorized_state.with_suffix(
                            factorized_state.suffix + ".json"
                        )
                    ),
                },
                gpu=False,
            )
        else:
            raise AssertionError(stage)

        receipt = _write_stage_receipt(
            receipt_path,
            stage=stage,
            common=common,
            inputs=inputs,
            outputs=outputs,
            command=command,
        )
        _validate_stage_receipt(receipt, stage=stage, common=common)
        receipts.append(receipt_path)
        previous_receipt = receipt_path
        stage_outputs[stage] = outputs

    result = {
        "scene_id": scene,
        "physical_space_id": common["physical_space_id"],
        "split": common["split"],
        "completed_stage": STAGES[len(receipts) - 1],
        "stage_receipts": [_file_record(path) for path in receipts],
        "final_receipt": None,
    }
    if stop_index == len(STAGES) - 1:
        final_path = root / "pilot_receipt.json"
        final = {
            "schema": FINAL_RECEIPT_SCHEMA,
            "schema_version": 1,
            "scene_id": scene,
            "physical_space_id": common["physical_space_id"],
            "split": common["split"],
            "immutable_authorities": {
                key: common[key] for key in IMMUTABLE_AUTHORITY_KEYS
            },
            "archive": common["archive"],
            "stage_receipts": result["stage_receipts"],
            "factorized_state": _file_record(factorized_state),
            "source_access": _source_access(),
        }
        final["authority_sha256"] = _receipt_content_sha256(final)
        if final_path.exists():
            existing, _sha, _source = load_json_object(
                final_path, label="full scalar pilot receipt"
            )
            if existing != final:
                raise ValueError("existing final pilot receipt differs")
        else:
            write_frozen_json(final_path, final)
        result["final_receipt"] = _file_record(final_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--scan-archive-part", required=True)
    parser.add_argument("--cohort-authority", required=True)
    parser.add_argument("--expected-cohort-authority-sha256", required=True)
    parser.add_argument("--exclusion-manifest", required=True)
    parser.add_argument("--expected-exclusion-manifest-sha256", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--expected-inventory-sha256", required=True)
    parser.add_argument("--pilot-root", required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--stop-after", choices=STAGES, default=STAGES[-1])
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
