#!/usr/bin/env python3
"""Freeze and independently validate one successful guarded GPU stage."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    stable_descriptor_load,
    validate_file_record,
    write_frozen_json,
)


SCHEMA_VERSION = 1
COMMAND_ARTIFACT_TYPE = "guarded_gpu_stage_command"
RECEIPT_ARTIFACT_TYPE = "guarded_gpu_stage_receipt"
TELEMETRY_COLUMNS = (
    "timestamp",
    "gpu",
    "bus_id",
    "temp_c",
    "power_w",
    "power_limit_w",
    "util_pct",
    "memory_mib",
    "pstate",
    "event",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _gpu_identity(index: int, uuid: str, bus_id: str) -> dict[str, Any]:
    value = {
        "physical_index": int(index),
        "uuid": str(uuid),
        "pci_bus_id": str(bus_id),
    }
    _require(
        value["physical_index"] == 1
        and value["uuid"].startswith("GPU-")
        and len(value["uuid"]) > 8
        and bool(value["pci_bus_id"]),
        "guard receipt GPU identity is invalid",
    )
    return value


def _bus_suffix(value: str) -> str:
    parts = str(value).strip().lower().split(":")
    return ":".join(parts[-2:])


def _finite_number(value: str, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid telemetry {label}") from exc
    _require(math.isfinite(number), f"non-finite telemetry {label}")
    return number


def summarize_telemetry(
    path: str | Path,
    *,
    gpu_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], str, Path]:
    def load(handle) -> list[dict[str, str]]:
        try:
            text = handle.read().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("GPU telemetry is not UTF-8") from exc
        reader = csv.DictReader(io.StringIO(text))
        _require(tuple(reader.fieldnames or ()) == TELEMETRY_COLUMNS, "GPU telemetry header differs")
        return [dict(row) for row in reader]

    rows, digest, source = stable_descriptor_load(
        path,
        load,
        label="GPU guard telemetry",
    )
    _require(rows, "GPU guard telemetry has no runtime samples")
    temperatures: list[float] = []
    powers: list[float] = []
    power_limits: list[float] = []
    utilizations: list[float] = []
    memories: list[float] = []
    events: Counter[str] = Counter()
    for index, row in enumerate(rows):
        _require(set(row) == set(TELEMETRY_COLUMNS), f"telemetry row {index} differs")
        _require(
            row["gpu"] == str(gpu_identity["physical_index"])
            and _bus_suffix(row["bus_id"])
            == _bus_suffix(str(gpu_identity["pci_bus_id"])),
            f"telemetry row {index} belongs to another GPU",
        )
        event = str(row["event"])
        _require(bool(event), f"telemetry row {index} lacks an event")
        _require(
            not any(token in event for token in ("abort", "failed", "unresponsive")),
            f"telemetry row {index} records a failed guard event",
        )
        temperatures.append(_finite_number(row["temp_c"], "temperature"))
        powers.append(_finite_number(row["power_w"], "power"))
        power_limits.append(_finite_number(row["power_limit_w"], "power limit"))
        utilizations.append(_finite_number(row["util_pct"], "utilization"))
        memories.append(_finite_number(row["memory_mib"], "memory"))
        _require(
            -50.0 <= temperatures[-1] <= 150.0,
            f"telemetry row {index} temperature is out of bounds",
        )
        _require(
            powers[-1] >= 0.0 and power_limits[-1] > 0.0,
            f"telemetry row {index} power is out of bounds",
        )
        _require(
            0.0 <= utilizations[-1] <= 100.0 and memories[-1] >= 0.0,
            f"telemetry row {index} utilization/memory is out of bounds",
        )
        events[event] += 1
    return (
        {
            "sample_count": len(rows),
            "first_timestamp": rows[0]["timestamp"],
            "last_timestamp": rows[-1]["timestamp"],
            "maximum_temperature_c": max(temperatures),
            "maximum_power_w": max(powers),
            "maximum_reported_power_limit_w": max(power_limits),
            "maximum_utilization_percent": max(utilizations),
            "maximum_memory_mib": max(memories),
            "event_counts": dict(sorted(events.items())),
        },
        digest,
        source,
    )


def prepare_command(
    *,
    output: Path,
    run_manifest: Path,
    seed: int,
    scene: str,
    gpu_identity: Mapping[str, Any],
    command: Sequence[str],
    prepared_epoch: int | None = None,
) -> dict[str, Any]:
    argv = [str(value) for value in command]
    if argv and argv[0] == "--":
        argv = argv[1:]
    _require(argv and all(argv), "guarded GPU command is empty")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": COMMAND_ARTIFACT_TYPE,
        "run_manifest": file_record(run_manifest),
        "seed": int(seed),
        "scene": str(scene),
        "gpu_identity": dict(gpu_identity),
        "argv": argv,
        "argv_sha256": canonical_json_sha256(argv),
    }
    if prepared_epoch is not None:
        _require(
            isinstance(prepared_epoch, int)
            and not isinstance(prepared_epoch, bool)
            and prepared_epoch > 0,
            "guarded GPU command prepared epoch is invalid",
        )
        payload["prepared_epoch"] = prepared_epoch
    write_frozen_json(output, payload)
    return {"command_record": file_record(output), "argv_sha256": payload["argv_sha256"]}


def _validate_command_payload(payload: object) -> dict[str, Any]:
    required_fields = {
        "schema_version",
        "artifact_type",
        "run_manifest",
        "seed",
        "scene",
        "gpu_identity",
        "argv",
        "argv_sha256",
    }
    _require(
        isinstance(payload, Mapping)
        and frozenset(payload)
        in {frozenset(required_fields), frozenset(required_fields | {"prepared_epoch"})}
        and payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("artifact_type") == COMMAND_ARTIFACT_TYPE,
        "guarded GPU command record differs",
    )
    if "prepared_epoch" in payload:
        prepared_epoch = payload.get("prepared_epoch")
        _require(
            isinstance(prepared_epoch, int)
            and not isinstance(prepared_epoch, bool)
            and prepared_epoch > 0,
            "guarded GPU command prepared epoch is invalid",
        )
    argv = payload.get("argv")
    _require(
        isinstance(argv, list)
        and argv
        and all(isinstance(value, str) and value for value in argv)
        and payload.get("argv_sha256") == canonical_json_sha256(argv),
        "guarded GPU argv binding differs",
    )
    validate_file_record(payload["run_manifest"], label="guard command run manifest")
    identity = payload.get("gpu_identity")
    _require(
        isinstance(identity, Mapping)
        and set(identity) == {"physical_index", "uuid", "pci_bus_id"},
        "guarded GPU command identity differs",
    )
    _gpu_identity(
        int(identity.get("physical_index", -1)),
        str(identity.get("uuid", "")),
        str(identity.get("pci_bus_id", "")),
    )
    return dict(payload)


def finalize_receipt(
    *,
    output: Path,
    command_record: Path,
    telemetry: Path,
    guard: Path,
    stage_output: Path,
    exit_status: int,
) -> dict[str, Any]:
    _require(int(exit_status) == 0, "guarded GPU command did not exit zero")
    command, _, _ = load_json_object(command_record, label="guard command record")
    command = _validate_command_payload(command)
    identity = command["gpu_identity"]
    summary, telemetry_sha256, telemetry_path = summarize_telemetry(
        telemetry,
        gpu_identity=identity,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": RECEIPT_ARTIFACT_TYPE,
        "status": "guarded_command_exit_zero_with_verified_telemetry",
        "seed": command["seed"],
        "scene": command["scene"],
        "gpu_identity": identity,
        "exit_status": 0,
        "guard": file_record(guard),
        "command": file_record(command_record),
        "telemetry": {"path": str(telemetry_path), "sha256": telemetry_sha256},
        "telemetry_summary": summary,
        "stage_output": file_record(stage_output),
    }
    write_frozen_json(output, payload)
    return {"receipt": file_record(output), "status": payload["status"]}


def validate_receipt(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    payload, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="GPU guard receipt",
    )
    _require(
        set(payload)
        == {
            "schema_version",
            "artifact_type",
            "status",
            "seed",
            "scene",
            "gpu_identity",
            "exit_status",
            "guard",
            "command",
            "telemetry",
            "telemetry_summary",
            "stage_output",
        }
        and payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("artifact_type") == RECEIPT_ARTIFACT_TYPE
        and payload.get("status")
        == "guarded_command_exit_zero_with_verified_telemetry"
        and payload.get("exit_status") == 0,
        "GPU guard receipt schema/status differs",
    )
    identity = payload.get("gpu_identity")
    _require(isinstance(identity, Mapping), "GPU guard receipt lacks identity")
    _gpu_identity(
        int(identity.get("physical_index", -1)),
        str(identity.get("uuid", "")),
        str(identity.get("pci_bus_id", "")),
    )
    validate_file_record(payload["guard"], label="GPU guard implementation")
    command_path = validate_file_record(payload["command"], label="GPU guard command")
    command, _, _ = load_json_object(command_path, label="GPU guard command")
    command = _validate_command_payload(command)
    _require(
        command["seed"] == payload["seed"]
        and command["scene"] == payload["scene"]
        and command["gpu_identity"] == identity,
        "GPU guard receipt differs from its command",
    )
    telemetry_path = validate_file_record(payload["telemetry"], label="GPU telemetry")
    summary, telemetry_sha256, _ = summarize_telemetry(
        telemetry_path,
        gpu_identity=identity,
    )
    _require(
        telemetry_sha256 == payload["telemetry"]["sha256"]
        and summary == payload["telemetry_summary"],
        "GPU telemetry summary differs",
    )
    validate_file_record(payload["stage_output"], label="guarded stage output")
    return {"receipt": {"path": str(source), "sha256": digest}, "payload": payload, "command": command}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    prepare = subparsers.add_parser("prepare-command")
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--run-manifest", required=True, type=Path)
    prepare.add_argument("--seed", required=True, type=int)
    prepare.add_argument("--scene", required=True)
    prepare.add_argument("--gpu", required=True, type=int)
    prepare.add_argument("--gpu-uuid", required=True)
    prepare.add_argument("--gpu-bus-id", required=True)
    prepare.add_argument("--prepared-epoch", type=int)
    prepare.add_argument("command", nargs=argparse.REMAINDER)
    final = subparsers.add_parser("finalize")
    final.add_argument("--output", required=True, type=Path)
    final.add_argument("--command-record", required=True, type=Path)
    final.add_argument("--telemetry", required=True, type=Path)
    final.add_argument("--guard", required=True, type=Path)
    final.add_argument("--stage-output", required=True, type=Path)
    final.add_argument("--exit-status", required=True, type=int)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--receipt", required=True, type=Path)
    validate.add_argument("--expected-sha256", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.subcommand == "prepare-command":
        result = prepare_command(
            output=args.output,
            run_manifest=args.run_manifest,
            seed=args.seed,
            scene=args.scene,
            gpu_identity=_gpu_identity(
                args.gpu,
                args.gpu_uuid,
                args.gpu_bus_id,
            ),
            command=args.command,
            prepared_epoch=args.prepared_epoch,
        )
    elif args.subcommand == "finalize":
        result = finalize_receipt(
            output=args.output,
            command_record=args.command_record,
            telemetry=args.telemetry,
            guard=args.guard,
            stage_output=args.stage_output,
            exit_status=args.exit_status,
        )
    else:
        result = validate_receipt(
            args.receipt,
            expected_sha256=args.expected_sha256 or None,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
