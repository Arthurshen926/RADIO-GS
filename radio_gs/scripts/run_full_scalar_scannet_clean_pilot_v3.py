#!/usr/bin/env python3
"""Run any one sealed clean-cohort ScanNet scene from stage 00.

This is a new receipt namespace, not a continuation of the v1/v2 pilot.  It
reuses the already exercised v1 stage implementations while binding every
stage and final receipt to this launcher's own byte identity.  All exact MPR
commands additionally carry the frozen ``--alpha-threshold 0`` contract.

Only a contiguous, fully validated receipt prefix may be resumed.  A missing
receipt followed by a later receipt, a changed predecessor, an unexpected
stage output layout, or an unreceipted output fails closed.  Existing v1/v2
receipts are never adopted by this launcher.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any

from radio_gs.scripts import run_full_scalar_scannet_clean_pilot as v1
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_json_object,
)


STAGE_RECEIPT_SCHEMA = "radio_gs.full_scalar_scannet_clean_pilot_stage.v3"
FINAL_RECEIPT_SCHEMA = "radio_gs.full_scalar_scannet_clean_pilot_receipt.v3"
STAGES = v1.STAGES

_V1_VALIDATE_AUTHORITIES = v1.validate_pilot_authorities
_V1_VALIDATE_STAGE_RECEIPT = v1._validate_stage_receipt
_V1_MPR_COMMAND = v1._mpr_command
_V1_WRITE_STAGE_RECEIPT = v1._write_stage_receipt

_STAGE_OUTPUT_KEYS: dict[str, frozenset[str]] = {
    "sens_extraction": frozenset({"sens_payload", "archive_member"}),
    "query_free_materialization": frozenset(
        {
            "materialization_report",
            "field_source_contract",
            "field_frame_manifest_sha256",
            "stage_log",
        }
    ),
    "geometry": frozenset(
        {"geometry_checkpoint", "geometry_ply", "stage_log"}
    ),
    "radio_features": frozenset(
        {
            "feature_manifest",
            "feature_output_bundle_sha256",
            "num_frames",
            "stage_log",
        }
    ),
    "render_contract": frozenset(
        {"render_config", "render_checkpoint", "stage_log"}
    ),
    "factorized_radio": frozenset(
        {
            "mpr_cache",
            "mpr_report",
            "responsibility_authority",
            "stage_log",
        }
    ),
    "exact_raw_reference": frozenset(
        {"mpr_cache", "mpr_report", "stage_log"}
    ),
    "exact_dino_v3": frozenset({"mpr_cache", "mpr_report", "stage_log"}),
    "exact_sam3": frozenset({"mpr_cache", "mpr_report", "stage_log"}),
    "capability_cohort": frozenset({"capability_cohort"}),
    "factorized_field": frozenset(
        {"field_checkpoint", "field_report", "stage_log"}
    ),
    "factorized_state": frozenset(
        {"factorized_state", "factorized_state_report", "stage_log"}
    ),
}
_COMMANDLESS_STAGES = frozenset({"sens_extraction", "capability_cohort"})
_GPU_STAGES = frozenset(
    {
        "geometry",
        "radio_features",
        "factorized_radio",
        "exact_raw_reference",
        "exact_dino_v3",
        "exact_sam3",
        "factorized_field",
    }
)
_GPU_THERMAL_GUARD_INPUT = "gpu_thermal_guard_authority"
_DELEGATED_STAGE_RUNNER_INPUT = "delegated_stage_runner_implementation"
_THERMAL_GUARD_PATH = Path(__file__).with_name(
    "run_with_gpu_thermal_guard.sh"
).resolve()
_LEGACY_STAGE00_LAUNCHER_IMPLEMENTATION = {
    "path": str(Path(__file__).resolve()),
    "sha256": "3933f590c295b741be4bac07613f609a112de589556d516a60f480e14a7be4db",
    "size_bytes": 11654,
}
_LEGACY_PRE_STRICT_RESUME_LAUNCHER_IMPLEMENTATION = {
    "path": str(Path(__file__).resolve()),
    "sha256": "6037f98ad440bdca3343e60545a4d03a07de84c4cf1dfb6aac3150cddd41e7ad",
    "size_bytes": 17978,
}
_LEGACY_PRE_STRICT_RESUME_STAGE_RUNNER_IMPLEMENTATION = {
    "path": str(Path(v1.__file__).resolve()),
    "sha256": "1ff4a3bc6fcb251f0076170d16e68449f0e46a95793fb7b4c09804d50f0b4e12",
    "size_bytes": 43526,
}
_LEGACY_PRE_STRICT_RESUME_STAGES = frozenset(
    {"query_free_materialization", "geometry"}
)
_FIXED_THERMAL_ENV = {
    "GPU_MAX_TEMP_C": "88",
    "GPU_START_MAX_TEMP_C": "82",
    # The host power cap is nominally 300 W.  nvidia-smi may report its
    # decimal representation slightly above that value, so the guard accepts
    # at most 300.5 W as a readback tolerance, not as a higher requested cap.
    "GPU_MAX_POWER_LIMIT_W": "300.5",
    "GPU_POLL_SECONDS": "30",
    "GPU_MAX_CONSECUTIVE_OVERHEAT_POLLS": "6",
    "GPU_MAX_CONSECUTIVE_TELEMETRY_FAILURES": "3",
    # Declare every optional guard control explicitly so inherited shell
    # variables cannot silently change the production policy.
    "GPU_OWNER_AUDIT_LOG": "",
    "GPU_OWNER_PID_NAMESPACE_MODE": "exclusive-singleton-after-clear-v1",
    "GPU_SOFT_PAUSE_TEMP_C": "0",
    "GPU_SOFT_RESUME_TEMP_C": "0",
    "GPU_PEER_INDEX": "",
    "GPU_PEER_PAUSE_TEMP_C": "0",
    "GPU_PEER_RESUME_TEMP_C": "0",
    "GPU_PEER_QUIET_SECONDS": "0",
    "GPU_PEER_MAX_POWER_W": "0",
    "GPU_PEER_MAX_MEMORY_MIB": "0",
    "GPU_PEER_MAX_UTIL_PCT": "100",
    "GPU_PEER_ACTIVITY_ACTION": "pause",
}


def _telemetry_path(args: argparse.Namespace) -> Path:
    return (
        Path(args.pilot_root).expanduser().resolve()
        / f"gpu{int(args.gpu)}_telemetry.csv"
    )


def _gpu_thermal_guard_authority(args: argparse.Namespace) -> dict[str, Any]:
    poll_seconds = int(_FIXED_THERMAL_ENV["GPU_POLL_SECONDS"])
    overheat_polls = int(
        _FIXED_THERMAL_ENV["GPU_MAX_CONSECUTIVE_OVERHEAT_POLLS"]
    )
    return {
        "schema": "radio_gs.gpu_thermal_guard_authority.v1",
        "guard_implementation": v1._file_record(_THERMAL_GUARD_PATH),
        "physical_gpu_index": int(args.gpu),
        "telemetry_log": str(_telemetry_path(args)),
        "production_policy": {
            "nominal_board_power_limit_w": 300,
            "nvidia_smi_reported_limit_ceiling_w": "300.5",
            "reported_limit_ceiling_semantics": (
                "readback_tolerance_only_not_a_nominal_limit_above_300w"
            ),
            "maximum_temperature_c": 88,
            "poll_seconds": poll_seconds,
            "consecutive_overheat_polls": overheat_polls,
            "nominal_overheat_window_seconds": (
                poll_seconds * overheat_polls
            ),
        },
        "guard_environment": dict(_FIXED_THERMAL_ENV),
    }


def validate_pilot_authorities(args: argparse.Namespace) -> dict[str, Any]:
    """Validate the frozen cohort chain and bind this v3 implementation."""

    common = _V1_VALIDATE_AUTHORITIES(args)
    common["launcher_implementation"] = v1._file_record(
        Path(__file__).resolve()
    )
    common[_DELEGATED_STAGE_RUNNER_INPUT] = v1._file_record(
        Path(v1.__file__).resolve()
    )
    common[_GPU_THERMAL_GUARD_INPUT] = _gpu_thermal_guard_authority(args)
    setattr(
        args,
        "_v3_expected_gpu_thermal_guard_authority",
        dict(common[_GPU_THERMAL_GUARD_INPUT]),
    )
    return common


def _mpr_command(**kwargs: Any) -> list[str]:
    """Return the verified MPR command with exact alpha support enabled."""

    command = _V1_MPR_COMMAND(**kwargs)
    if "--alpha-threshold" in command:
        raise ValueError("base MPR command unexpectedly declares alpha threshold")
    insertion = command.index("--aggregation-mode")
    command[insertion:insertion] = ["--alpha-threshold", "0"]
    return command


def _thermal_env(
    args: argparse.Namespace, telemetry: Path
) -> dict[str, str]:
    """Return the complete, non-inheritable v3 production guard policy."""

    del telemetry
    expected = getattr(args, "_v3_expected_gpu_thermal_guard_authority", None)
    current = _gpu_thermal_guard_authority(args)
    if expected is not None and current != expected:
        raise ValueError("GPU thermal guard changed before stage launch")
    return {
        "GPU": str(int(args.gpu)),
        "GPU_TELEMETRY_LOG": str(_telemetry_path(args)),
        **_FIXED_THERMAL_ENV,
    }


def _write_stage_receipt(
    path: Path,
    *,
    stage: str,
    common: Mapping[str, Any],
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Any],
    command: Sequence[str] | None,
) -> dict[str, Any]:
    """Bind only GPU stages to the exact guard bytes and fixed policy."""

    bound_inputs = dict(inputs)
    if (
        _GPU_THERMAL_GUARD_INPUT in bound_inputs
        or _DELEGATED_STAGE_RUNNER_INPUT in bound_inputs
    ):
        raise ValueError("caller must not supply v3 execution authorities")
    stage_runner = common[_DELEGATED_STAGE_RUNNER_INPUT]
    if stage_runner != v1._file_record(Path(v1.__file__).resolve()):
        raise ValueError("delegated stage runner changed during stage execution")
    bound_inputs[_DELEGATED_STAGE_RUNNER_INPUT] = dict(stage_runner)
    if stage in _GPU_STAGES:
        authority = common[_GPU_THERMAL_GUARD_INPUT]
        if authority["guard_implementation"] != v1._file_record(
            _THERMAL_GUARD_PATH
        ):
            raise ValueError("GPU thermal guard changed during stage execution")
        bound_inputs[_GPU_THERMAL_GUARD_INPUT] = dict(
            authority
        )
    return _V1_WRITE_STAGE_RECEIPT(
        path,
        stage=stage,
        common=common,
        inputs=bound_inputs,
        outputs=outputs,
        command=command,
    )


@contextmanager
def _v3_runtime() -> Iterator[None]:
    """Temporarily route the verified stage runner through v3 authorities."""

    original = {
        "validate_pilot_authorities": v1.validate_pilot_authorities,
        "_validate_stage_receipt": v1._validate_stage_receipt,
        "_mpr_command": v1._mpr_command,
        "_thermal_env": v1._thermal_env,
        "_write_stage_receipt": v1._write_stage_receipt,
        "STAGE_RECEIPT_SCHEMA": v1.STAGE_RECEIPT_SCHEMA,
        "FINAL_RECEIPT_SCHEMA": v1.FINAL_RECEIPT_SCHEMA,
    }
    v1.validate_pilot_authorities = validate_pilot_authorities
    v1._validate_stage_receipt = _validate_stage_receipt
    v1._mpr_command = _mpr_command
    v1._thermal_env = _thermal_env
    v1._write_stage_receipt = _write_stage_receipt
    v1.STAGE_RECEIPT_SCHEMA = STAGE_RECEIPT_SCHEMA
    v1.FINAL_RECEIPT_SCHEMA = FINAL_RECEIPT_SCHEMA
    try:
        yield
    finally:
        for name, value in original.items():
            setattr(v1, name, value)


def _validate_command_binding(receipt: Mapping[str, Any], *, stage: str) -> None:
    command = receipt.get("command")
    digest = receipt.get("command_sha256")
    if stage in _COMMANDLESS_STAGES:
        if command is not None or digest is not None:
            raise ValueError(f"{stage} command binding differs")
        return
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(value, str) or not value for value in command)
        or digest != canonical_json_sha256(command)
    ):
        raise ValueError(f"{stage} command binding differs")
    if stage in {
        "factorized_radio",
        "exact_raw_reference",
        "exact_dino_v3",
        "exact_sam3",
    }:
        positions = [
            index
            for index, value in enumerate(command)
            if value == "--alpha-threshold"
        ]
        if len(positions) != 1 or command[positions[0] + 1] != "0":
            raise ValueError(f"{stage} exact MPR alpha contract differs")


def _validate_stage_receipt(
    value: object,
    *,
    stage: str,
    common: Mapping[str, Any],
) -> dict[str, Any]:
    validation_common = common
    legacy_unbound_runner = False
    if isinstance(value, Mapping) and stage == "sens_extraction":
        immutable = value.get("immutable_authorities")
        launcher = (
            immutable.get("launcher_implementation")
            if isinstance(immutable, Mapping)
            else None
        )
        if launcher == _LEGACY_STAGE00_LAUNCHER_IMPLEMENTATION:
            # The only pre-guard v3 receipt permitted to cross this launcher
            # revision is the exact, commandless CPU stage 00 implementation.
            # No GPU receipt receives this compatibility path.
            validation_common = dict(common)
            validation_common["launcher_implementation"] = dict(
                _LEGACY_STAGE00_LAUNCHER_IMPLEMENTATION
            )
            legacy_unbound_runner = True
    if isinstance(value, Mapping) and stage in _LEGACY_PRE_STRICT_RESUME_STAGES:
        immutable = value.get("immutable_authorities")
        launcher = (
            immutable.get("launcher_implementation")
            if isinstance(immutable, Mapping)
            else None
        )
        if launcher == _LEGACY_PRE_STRICT_RESUME_LAUNCHER_IMPLEMENTATION:
            # Narrow compatibility for the already materialized scene0002
            # query-free and geometry prefix.  Those stages predate both the
            # strict-resume feature command and explicit delegated-runner
            # binding; no feature or later receipt can cross this boundary.
            validation_common = dict(common)
            validation_common["launcher_implementation"] = dict(
                _LEGACY_PRE_STRICT_RESUME_LAUNCHER_IMPLEMENTATION
            )
            legacy_unbound_runner = True
    receipt = _V1_VALIDATE_STAGE_RECEIPT(
        value, stage=stage, common=validation_common
    )
    outputs = receipt["outputs"]
    if set(outputs) != _STAGE_OUTPUT_KEYS[stage]:
        raise ValueError(f"{stage} stage output contract differs")
    _validate_command_binding(receipt, stage=stage)
    inputs = receipt["inputs"]
    delegated_runner = inputs.get(_DELEGATED_STAGE_RUNNER_INPUT)
    if legacy_unbound_runner:
        if delegated_runner is not None:
            raise ValueError(f"{stage} legacy delegated stage runner differs")
    elif delegated_runner != common[_DELEGATED_STAGE_RUNNER_INPUT]:
        raise ValueError(f"{stage} delegated stage runner differs")
    guard_authority = inputs.get(_GPU_THERMAL_GUARD_INPUT)
    if stage in _GPU_STAGES:
        if guard_authority != common[_GPU_THERMAL_GUARD_INPUT]:
            raise ValueError(f"{stage} GPU thermal guard authority differs")
        command = receipt["command"]
        expected_prefix = [
            "bash",
            "radio_gs/scripts/run_with_gpu_thermal_guard.sh",
            "--",
            "env",
            (
                "CUDA_VISIBLE_DEVICES="
                f"{common[_GPU_THERMAL_GUARD_INPUT]['physical_gpu_index']}"
            ),
        ]
        if command[: len(expected_prefix)] != expected_prefix:
            raise ValueError(f"{stage} GPU thermal guard command differs")
    elif guard_authority is not None:
        raise ValueError(f"{stage} unexpectedly binds a GPU thermal guard")
    return receipt


def _receipt_paths(root: Path) -> list[Path]:
    receipt_root = root / "receipts"
    return [
        receipt_root / f"{index:02d}_{stage}.json"
        for index, stage in enumerate(STAGES)
    ]


def _validate_final_receipt(
    path: Path,
    *,
    common: Mapping[str, Any],
    stage_paths: Sequence[Path],
) -> dict[str, Any]:
    value, _digest, _source = load_json_object(
        path, label="full scalar v3 pilot receipt"
    )
    if not isinstance(value, Mapping):
        raise ValueError("v3 final receipt must be a mapping")
    receipt = dict(value)
    expected_keys = {
        "schema",
        "schema_version",
        "scene_id",
        "physical_space_id",
        "split",
        "immutable_authorities",
        "archive",
        "stage_receipts",
        "factorized_state",
        "source_access",
        "authority_sha256",
    }
    immutable = {
        key: common[key] for key in v1.IMMUTABLE_AUTHORITY_KEYS
    }
    expected_stage_records = [v1._file_record(item) for item in stage_paths]
    if (
        set(receipt) != expected_keys
        or receipt.get("schema") != FINAL_RECEIPT_SCHEMA
        or receipt.get("schema_version") != 1
        or receipt.get("scene_id") != common["scene_id"]
        or receipt.get("physical_space_id") != common["physical_space_id"]
        or receipt.get("split") != common["split"]
        or receipt.get("immutable_authorities") != immutable
        or receipt.get("archive") != common["archive"]
        or receipt.get("stage_receipts") != expected_stage_records
        or receipt.get("source_access") != v1._source_access()
        or any(receipt.get("source_access", {}).values())
        or receipt.get("authority_sha256")
        != v1._receipt_content_sha256(receipt)
    ):
        raise ValueError("v3 final receipt contract differs")
    v1._validate_file_record(
        receipt.get("factorized_state"), label="v3 final factorized state"
    )
    return receipt


def _validate_resume_boundary(
    root: str | Path,
    *,
    common: Mapping[str, Any],
) -> list[Path]:
    """Validate the only stage prefix that v3 is permitted to resume."""

    resolved = Path(root).expanduser().resolve()
    paths = _receipt_paths(resolved)
    exists = [path.is_file() for path in paths]
    first_missing = next(
        (index for index, present in enumerate(exists) if not present),
        len(paths),
    )
    if any(exists[first_missing + 1 :]):
        raise ValueError("v3 stage receipts are not one contiguous prefix")

    previous: Path | None = None
    prefix: list[Path] = []
    for index, path in enumerate(paths[:first_missing]):
        stage = STAGES[index]
        value, _digest, _source = load_json_object(
            path, label=f"v3 {stage} stage receipt"
        )
        receipt = _validate_stage_receipt(value, stage=stage, common=common)
        declared = receipt["inputs"].get("previous_stage_receipt")
        if previous is None:
            if declared is not None:
                raise ValueError("first v3 stage declares a predecessor")
        elif declared != v1._file_record(previous):
            raise ValueError("v3 stage receipt predecessor chain differs")
        previous = path
        prefix.append(path)

    final_path = resolved / "pilot_receipt.json"
    if final_path.exists():
        if len(prefix) != len(STAGES):
            raise ValueError("v3 final receipt exists before every stage")
        _validate_final_receipt(
            final_path, common=common, stage_paths=prefix
        )
    return prefix


def validate_resume_boundary(
    root: str | Path,
    *,
    common: Mapping[str, Any],
) -> list[Path]:
    """Public receipt audit independent of the caller's runtime context."""

    with _v3_runtime():
        return _validate_resume_boundary(root, common=common)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run or resume one clean-cohort scene at an immutable stage boundary."""

    if str(args.stop_after) not in STAGES:
        raise ValueError("v3 stop-after stage differs")
    if int(args.gpu) < 0:
        raise ValueError("v3 GPU index must be non-negative")

    with _v3_runtime():
        common = validate_pilot_authorities(args)
        validate_resume_boundary(args.pilot_root, common=common)
        result = v1.run(args)
        prefix = validate_resume_boundary(args.pilot_root, common=common)

    stop_index = STAGES.index(str(args.stop_after))
    if (
        len(prefix) < stop_index + 1
        or result.get("completed_stage") != STAGES[stop_index]
        or result.get("scene_id") != common["scene_id"]
    ):
        raise RuntimeError("v3 stage runner returned an incomplete boundary")
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
