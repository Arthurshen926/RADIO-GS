"""Content-addressed receipts for the five-stage candidate lifecycle.

This module is the public evidence seam between the Candidate Authority
Bundle and later lifecycle work.  It records exact stage inputs and outputs,
the execution identities that produced them, and a predecessor chain.  It is
deliberately an evidence binder: process observation and independent runtime
verification belong to the later Runtime Compliance Proof ticket.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from radio_gs.candidate_authority import (
    CandidateAuthorityBundle,
    validate_candidate_authority,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_json_object,
    stable_descriptor_load,
    write_frozen_json,
)


STAGE_RECEIPT_SCHEMA = "radio_gs.stage_receipt.v1"
STAGE_ORDER = (
    "mapping_training",
    "deployment_sealing",
    "warm_cache_compilation",
    "query_prediction_sealing",
    "evaluation",
)
PREDICTION_STAGE = "query_prediction_sealing"
EVALUATION_STAGE = "evaluation"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PRIVATE_ARTIFACT_NAMES = {
    "ground_truth",
    "labels",
    "metric",
    "metrics",
    "private_target",
    "private_targets",
    "target",
    "targets",
}


class StageReceiptError(ValueError):
    """Raised when a receipt or artifact cannot be proven consistent."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise StageReceiptError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _canonical_digest(value: object, *, label: str) -> str:
    try:
        return canonical_json_sha256(value)
    except (TypeError, ValueError) as error:
        raise StageReceiptError(f"{label} is not finite canonical JSON") from error


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise StageReceiptError(
            f"{label} fields differ; missing={missing}, extra={extra}"
        )


def _nonempty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StageReceiptError(f"{label} must be a non-empty string")
    return value


def _artifact_marker(record: Mapping[str, Any]) -> str | None:
    marker = record.get("producer_stage")
    if marker is not None and (
        not isinstance(marker, str) or marker not in STAGE_ORDER
    ):
        raise StageReceiptError("artifact producer stage differs")
    return marker


def _without_marker(record: Mapping[str, Any]) -> dict[str, Any]:
    value = _plain(record)
    value.pop("producer_stage", None)
    return value


def _file_identity(path: str | Path) -> dict[str, Any]:
    def read_size(handle: Any) -> int:
        return int(os.fstat(handle.fileno()).st_size)

    try:
        size, digest, source = stable_descriptor_load(
            path,
            read_size,
            label="opaque lifecycle file",
        )
    except (OSError, TypeError, ValueError) as error:
        raise StageReceiptError(f"cannot bind opaque file {path}: {error}") from error
    return {
        "schema": "radio_gs.opaque_file.v1",
        "path": str(source),
        "sha256": digest,
        "size_bytes": size,
    }


def opaque_file(path: str | Path) -> dict[str, Any]:
    """Bind one opaque regular file by its raw bytes and length."""

    return _file_identity(path)


def canonical_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    """Bind a manifest by its finite canonical JSON representation."""

    if not isinstance(value, Mapping):
        raise StageReceiptError("manifest must be a mapping")
    payload = _plain(value)
    digest = _canonical_digest(payload, label="manifest")
    return {
        "schema": "radio_gs.canonical_manifest.v1",
        "sha256": digest,
        "value": payload,
    }


def _directory_root(entries: Sequence[Mapping[str, Any]]) -> str:
    leaves: list[bytes] = []
    for entry in entries:
        leaf = _canonical_digest(
            {
                "path": entry["path"],
                "sha256": entry["sha256"],
                "size_bytes": entry["size_bytes"],
            },
            label="directory leaf",
        )
        leaves.append(bytes.fromhex(leaf))
    if not leaves:
        return hashlib.sha256(b"radio_gs.directory_merkle.v1\0empty").hexdigest()
    while len(leaves) > 1:
        next_level: list[bytes] = []
        for index in range(0, len(leaves), 2):
            right = leaves[index + 1] if index + 1 < len(leaves) else leaves[index]
            next_level.append(
                hashlib.sha256(
                    b"radio_gs.directory_merkle.v1\0node" + leaves[index] + right
                ).digest()
            )
        leaves = next_level
    return hashlib.sha256(b"radio_gs.directory_merkle.v1\0root" + leaves[0]).hexdigest()


def _directory_entries(root: Path) -> list[dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise StageReceiptError(f"directory is not a real directory: {root}")
    entries: list[dict[str, Any]] = []
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directories:
            if (current_path / name).is_symlink():
                raise StageReceiptError(f"directory contains symlink: {current_path / name}")
        for name in files:
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise StageReceiptError(f"directory contains non-regular file: {path}")
            record = _file_identity(path)
            entries.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": record["sha256"],
                    "size_bytes": record["size_bytes"],
                }
            )
    return sorted(entries, key=lambda entry: entry["path"])


def directory_merkle(path: str | Path) -> dict[str, Any]:
    """Bind a directory through sorted file leaves and a deterministic Merkle root."""

    try:
        root = Path(path).expanduser().absolute()
    except (OSError, TypeError, ValueError) as error:
        raise StageReceiptError(f"cannot resolve directory {path}: {error}") from error
    entries = _directory_entries(root)
    return {
        "schema": "radio_gs.directory_merkle.v1",
        "path": str(root),
        "entries": entries,
        "merkle_root_sha256": _directory_root(entries),
    }


def _logical_tensor_member(value: Any, *, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        member = _plain(value)
        required = {"dtype", "shape", "sha256"}
        if not required.issubset(member):
            raise StageReceiptError(
                f"tensor member {name} must bind dtype, shape, and logical SHA-256"
            )
        _require_sha256(member["sha256"], label=f"tensor member {name}")
        if not isinstance(member["shape"], list) or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim < 0
            for dim in member["shape"]
        ):
            raise StageReceiptError(f"tensor member {name} shape differs")
        return member

    try:
        import numpy as np
        import torch

        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            raw = tensor.view(torch.uint8).numpy().tobytes()
            return {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "strides": list(tensor.stride()),
                "numel": int(tensor.numel()),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            return {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "strides": [int(stride // array.itemsize) for stride in array.strides],
                "numel": int(array.size),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
    except (ImportError, TypeError, ValueError, RuntimeError) as error:
        raise StageReceiptError(f"tensor member {name} cannot be bound: {error}") from error
    raise StageReceiptError(f"tensor member {name} must be a tensor or logical identity")


def tensor_container(
    path: str | Path,
    logical_members: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a tensor container plus sorted logical member identities."""

    if not isinstance(logical_members, Mapping) or not logical_members:
        raise StageReceiptError("tensor container requires logical members")
    members = [
        {"name": name, **_logical_tensor_member(logical_members[name], name=name)}
        for name in sorted(logical_members)
    ]
    return {
        "schema": "radio_gs.tensor_container.v1",
        "container": _file_identity(path),
        "members": members,
    }


def prediction_inventory(
    path: str | Path,
    prediction_ids: Sequence[str],
) -> dict[str, Any]:
    """Bind a complete prediction directory and its exact relative inventory."""

    directory = directory_merkle(path)
    ids = list(prediction_ids)
    if not ids or len(ids) != len(set(ids)) or any(
        not isinstance(value, str) or not value for value in ids
    ):
        raise StageReceiptError("prediction inventory identifiers are incomplete")
    ids = sorted(ids)
    entry_paths = [entry["path"] for entry in directory["entries"]]
    if ids != entry_paths:
        raise StageReceiptError(
            "prediction inventory identifiers do not cover the complete directory"
        )
    return {
        "schema": "radio_gs.prediction_inventory.v1",
        "directory": directory,
        "prediction_ids": ids,
        "prediction_count": len(ids),
        "merkle_root_sha256": directory["merkle_root_sha256"],
        "complete": True,
    }


def _validate_manifest(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StageReceiptError(f"{label} must be a manifest")
    _exact_keys(value, {"schema", "sha256", "value"}, label=label)
    if value["schema"] != "radio_gs.canonical_manifest.v1":
        raise StageReceiptError(f"{label} schema differs")
    expected = _canonical_digest(value["value"], label=label)
    if value["sha256"] != expected:
        raise StageReceiptError(f"{label} digest differs")
    return _plain(value)


def _validate_opaque_file(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StageReceiptError(f"{label} must be an opaque file")
    _exact_keys(
        _without_marker(value),
        {"schema", "path", "sha256", "size_bytes"},
        label=label,
    )
    if value["schema"] != "radio_gs.opaque_file.v1":
        raise StageReceiptError(f"{label} schema differs")
    _require_sha256(value["sha256"], label=label)
    if not isinstance(value["size_bytes"], int) or value["size_bytes"] < 0:
        raise StageReceiptError(f"{label} size differs")
    _artifact_marker(value)
    observed = _file_identity(str(value["path"]))
    if observed != _without_marker(value):
        raise StageReceiptError(f"{label} digest or size differs")
    return _plain(value)


def _validate_directory(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StageReceiptError(f"{label} must be a directory Merkle identity")
    _exact_keys(
        _without_marker(value),
        {"schema", "path", "entries", "merkle_root_sha256"},
        label=label,
    )
    if value["schema"] != "radio_gs.directory_merkle.v1":
        raise StageReceiptError(f"{label} schema differs")
    _artifact_marker(value)
    observed = directory_merkle(str(value["path"]))
    if observed != _without_marker(value):
        raise StageReceiptError(f"{label} Merkle identity differs")
    return _plain(value)


def _validate_tensor(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StageReceiptError(f"{label} must be a tensor container")
    _exact_keys(_without_marker(value), {"schema", "container", "members"}, label=label)
    if value["schema"] != "radio_gs.tensor_container.v1":
        raise StageReceiptError(f"{label} schema differs")
    _artifact_marker(value)
    _validate_opaque_file(value["container"], label=f"{label}.container")
    members = value["members"]
    if not isinstance(members, list) or not members:
        raise StageReceiptError(f"{label} logical members are incomplete")
    names: list[str] = []
    for member in members:
        if not isinstance(member, Mapping) or "name" not in member:
            raise StageReceiptError(f"{label} logical member differs")
        name = _nonempty_string(member["name"], label=f"{label}.member.name")
        names.append(name)
        _logical_tensor_member(_without_marker(member), name=name)
    if names != sorted(names) or len(names) != len(set(names)):
        raise StageReceiptError(f"{label} logical members are not canonically ordered")
    return _plain(value)


def _validate_prediction_inventory(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StageReceiptError(f"{label} must be a prediction inventory")
    _exact_keys(
        _without_marker(value),
        {
            "schema",
            "directory",
            "prediction_ids",
            "prediction_count",
            "merkle_root_sha256",
            "complete",
        },
        label=label,
    )
    if value["schema"] != "radio_gs.prediction_inventory.v1":
        raise StageReceiptError(f"{label} schema differs")
    _artifact_marker(value)
    directory = _validate_directory(value["directory"], label=f"{label}.directory")
    ids = value["prediction_ids"]
    if not isinstance(ids, list) or ids != sorted(ids) or not ids:
        raise StageReceiptError(f"{label} identifiers are incomplete")
    if ids != [entry["path"] for entry in directory["entries"]]:
        raise StageReceiptError(f"{label} coverage differs")
    if value["prediction_count"] != len(ids) or value["complete"] is not True:
        raise StageReceiptError(f"{label} is not complete")
    _require_sha256(value["merkle_root_sha256"], label=f"{label}.merkle_root_sha256")
    if value["merkle_root_sha256"] != directory["merkle_root_sha256"]:
        raise StageReceiptError(f"{label} Merkle root differs")
    return _plain(value)


def _validate_artifact(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StageReceiptError(f"{label} must be an artifact identity")
    schema = value.get("schema")
    if schema == "radio_gs.opaque_file.v1":
        return _validate_opaque_file(value, label=label)
    if schema == "radio_gs.canonical_manifest.v1":
        _artifact_marker(value)
        return _validate_manifest(_without_marker(value), label=label)
    if schema == "radio_gs.directory_merkle.v1":
        return _validate_directory(value, label=label)
    if schema == "radio_gs.tensor_container.v1":
        return _validate_tensor(value, label=label)
    if schema == "radio_gs.prediction_inventory.v1":
        return _validate_prediction_inventory(value, label=label)
    raise StageReceiptError(f"{label} artifact schema differs")


def _tag_output(value: Mapping[str, Any], stage: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StageReceiptError("stage output must be an artifact identity")
    record = _plain(value)
    if "producer_stage" in record and record["producer_stage"] != stage:
        raise StageReceiptError("mixed stage outputs are not allowed")
    record["producer_stage"] = stage
    return record


def _validate_artifact_map(
    value: object,
    *,
    label: str,
    stage: str,
    predecessor_stage: str | None,
    outputs: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise StageReceiptError(f"{label} must be a non-empty artifact map")
    result: dict[str, Any] = {}
    for name in sorted(value):
        if not isinstance(name, str) or not name:
            raise StageReceiptError(f"{label} contains an invalid artifact name")
        if (
            not outputs
            and stage != EVALUATION_STAGE
            and name.lower() in _PRIVATE_ARTIFACT_NAMES
        ):
            raise StageReceiptError("private evidence was opened before evaluation")
        record = value[name]
        if not isinstance(record, Mapping):
            raise StageReceiptError(f"{label}.{name} is not an artifact identity")
        marker = _artifact_marker(record)
        if outputs:
            if marker != stage:
                raise StageReceiptError(
                    f"{label}.{name} must be produced by {stage}"
                )
        elif marker is not None and marker != predecessor_stage:
            raise StageReceiptError(
                f"{label}.{name} has an unexpected producer stage"
            )
        elif (
            not outputs
            and stage != EVALUATION_STAGE
            and predecessor_stage is not None
            and marker is None
        ):
            raise StageReceiptError(
                f"{label}.{name} must be sealed by the predecessor stage"
            )
        result[name] = _validate_artifact(record, label=f"{label}.{name}")
        if marker is not None:
            result[name]["producer_stage"] = marker
    if not outputs and predecessor_stage is None and any(
        "producer_stage" in record for record in result.values()
    ):
        raise StageReceiptError(f"{label} cannot reference a predecessor")
    return result


def _validate_private_evidence(value: object, *, stage: str) -> dict[str, bool]:
    if not isinstance(value, Mapping):
        raise StageReceiptError("private evidence access must be a mapping")
    _exact_keys(value, {"targets_opened", "metrics_computed"}, label="private evidence")
    if any(not isinstance(item, bool) for item in value.values()):
        raise StageReceiptError("private evidence flags must be boolean")
    flags = {key: bool(item) for key, item in value.items()}
    if stage != EVALUATION_STAGE and any(flags.values()):
        raise StageReceiptError("private evidence was opened before evaluation")
    return flags


def _validate_code_identity(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StageReceiptError("code identity must be a mapping")
    _exact_keys(
        value,
        {"repository", "commit", "code_tree", "dirty_patch_sha256"},
        label="code identity",
    )
    _nonempty_string(value["repository"], label="code identity.repository")
    _nonempty_string(value["commit"], label="code identity.commit")
    _require_sha256(value["dirty_patch_sha256"], label="dirty patch")
    _validate_artifact(value["code_tree"], label="code identity.code_tree")
    return _plain(value)


def _validate_execution(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StageReceiptError("execution identity must be a mapping")
    _exact_keys(
        value,
        {
            "code_identity",
            "configuration",
            "command",
            "command_sha256",
            "dependency_container_identity",
            "seeds",
            "determinism",
            "environment",
            "runtime_trace",
            "terminal_status",
        },
        label="execution identity",
    )
    _validate_code_identity(value["code_identity"])
    for key in (
        "configuration",
        "dependency_container_identity",
        "seeds",
        "determinism",
        "environment",
        "runtime_trace",
    ):
        _validate_manifest(value[key], label=f"execution identity.{key}")
    command = value["command"]
    if not isinstance(command, list) or not command or any(
        not isinstance(item, str) or not item for item in command
    ):
        raise StageReceiptError("execution command differs")
    _require_sha256(value["command_sha256"], label="command")
    if value["command_sha256"] != _canonical_digest(command, label="command"):
        raise StageReceiptError("execution command digest differs")
    if value["terminal_status"] != "succeeded":
        raise StageReceiptError("only successful stages can be sealed")
    return _plain(value)


def _predecessor_reference(receipt: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if receipt is None:
        return None
    return {
        "stage": receipt["stage"],
        "stage_index": receipt["stage_index"],
        "receipt_id": receipt["receipt_id"],
    }


def _validate_receipt_mapping(
    value: object,
    *,
    candidate_authority: Mapping[str, Any] | CandidateAuthorityBundle | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StageReceiptError("stage receipt must be a mapping")
    receipt = _plain(value)
    _exact_keys(
        receipt,
        {
            "schema",
            "schema_version",
            "receipt_id",
            "candidate_id",
            "stage",
            "stage_index",
            "stage_order",
            "stage_contract",
            "predecessor",
            "inputs",
            "outputs",
            "execution",
            "private_evidence",
            "ordering_evidence",
            "prediction_inventory",
        },
        label="stage receipt",
    )
    if receipt["schema"] != STAGE_RECEIPT_SCHEMA or receipt["schema_version"] != 1:
        raise StageReceiptError("stage receipt schema differs")
    candidate_id = _require_sha256(receipt["candidate_id"], label="candidate_id")
    if candidate_authority is not None:
        candidate = validate_candidate_authority(candidate_authority)
        if candidate_id != candidate["candidate_id"]:
            raise StageReceiptError("stage receipt candidate identity differs")
        expected_order = tuple(candidate["execution_matrix"]["required_stage_order"])
    else:
        expected_order = STAGE_ORDER
    if expected_order != STAGE_ORDER:
        raise StageReceiptError("stage order is not the five-stage lifecycle")
    if not isinstance(receipt["stage"], str) or receipt["stage"] not in STAGE_ORDER:
        raise StageReceiptError("stage identity differs")
    stage = receipt["stage"]
    index = STAGE_ORDER.index(stage)
    if receipt["stage_index"] != index or tuple(receipt["stage_order"]) != STAGE_ORDER:
        raise StageReceiptError("stage order or index differs")
    _validate_manifest(receipt["stage_contract"], label="stage contract")
    predecessor = receipt["predecessor"]
    if index == 0:
        if predecessor is not None:
            raise StageReceiptError("mapping stage cannot have a predecessor")
    else:
        if not isinstance(predecessor, Mapping):
            raise StageReceiptError("stage predecessor is missing")
        _exact_keys(
            predecessor,
            {"stage", "stage_index", "receipt_id"},
            label="stage predecessor",
        )
        if (
            predecessor["stage"] != STAGE_ORDER[index - 1]
            or predecessor["stage_index"] != index - 1
        ):
            raise StageReceiptError("stage predecessor order differs")
        _require_sha256(predecessor["receipt_id"], label="stage predecessor")
    predecessor_stage = STAGE_ORDER[index - 1] if index else None
    inputs = _validate_artifact_map(
        receipt["inputs"],
        label="stage inputs",
        stage=stage,
        predecessor_stage=predecessor_stage,
        outputs=False,
    )
    outputs = _validate_artifact_map(
        receipt["outputs"],
        label="stage outputs",
        stage=stage,
        predecessor_stage=predecessor_stage,
        outputs=True,
    )
    execution = _validate_execution(receipt["execution"])
    private_evidence = _validate_private_evidence(
        receipt["private_evidence"], stage=stage
    )
    ordering = receipt["ordering_evidence"]
    if not isinstance(ordering, Mapping):
        raise StageReceiptError("ordering evidence differs")
    _exact_keys(
        ordering,
        {"stage_order", "stage_index", "predecessor_stage"},
        label="ordering evidence",
    )
    if (
        tuple(ordering["stage_order"]) != STAGE_ORDER
        or ordering["stage_index"] != index
        or ordering["predecessor_stage"] != predecessor_stage
    ):
        raise StageReceiptError("ordering evidence differs")

    inventory = receipt["prediction_inventory"]
    if stage == PREDICTION_STAGE:
        if "prediction_inventory" not in outputs:
            raise StageReceiptError("prediction inventory is missing")
        inventory = _validate_prediction_inventory(
            outputs["prediction_inventory"], label="prediction inventory"
        )
        if inventory != _plain(receipt["prediction_inventory"]):
            raise StageReceiptError("prediction inventory binding differs")
    elif inventory is not None:
        raise StageReceiptError("only query sealing may bind a prediction inventory")
    if stage == EVALUATION_STAGE and predecessor["stage"] != PREDICTION_STAGE:
        raise StageReceiptError("evaluation must follow prediction sealing")

    body = dict(receipt)
    body.pop("receipt_id", None)
    expected_id = _canonical_digest(body, label="stage receipt")
    if receipt["receipt_id"] != expected_id:
        raise StageReceiptError("stage receipt content identity differs")
    return receipt


@dataclass(frozen=True)
class StageReceipt(Mapping[str, Any]):
    """Recursively immutable in-memory representation of one stage receipt."""

    _payload: Mapping[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self._payload[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._payload)

    def __len__(self) -> int:
        return len(self._payload)

    def as_dict(self) -> dict[str, Any]:
        return _plain(self._payload)


def _make_receipt(
    *,
    candidate_authority: Mapping[str, Any] | CandidateAuthorityBundle,
    stage: str,
    stage_contract: Mapping[str, Any],
    inputs: Mapping[str, Mapping[str, Any]],
    outputs: Mapping[str, Mapping[str, Any]],
    code_identity: Mapping[str, Any],
    configuration: Mapping[str, Any],
    command: Sequence[str],
    dependency_container: Mapping[str, Any],
    seeds: Mapping[str, Any],
    determinism: Mapping[str, Any],
    environment: Mapping[str, Any],
    runtime_trace: Mapping[str, Any],
    private_evidence: Mapping[str, bool],
    predecessor: Mapping[str, Any] | StageReceipt | None,
) -> StageReceipt:
    candidate = validate_candidate_authority(candidate_authority)
    if stage not in STAGE_ORDER:
        raise StageReceiptError("stage identity differs")
    index = STAGE_ORDER.index(stage)
    predecessor_mapping = None if predecessor is None else _plain(predecessor)
    if predecessor_mapping is not None and "receipt_id" not in predecessor_mapping:
        raise StageReceiptError("stage predecessor is missing")
    predecessor_stage = STAGE_ORDER[index - 1] if index else None
    if index == 0 and predecessor_mapping is not None:
        raise StageReceiptError("mapping stage cannot have a predecessor")
    if index > 0 and (
        predecessor_mapping is None
        or predecessor_mapping.get("stage") != predecessor_stage
        or predecessor_mapping.get("stage_index") != index - 1
    ):
        raise StageReceiptError("stage predecessor order differs")
    if not isinstance(stage_contract, Mapping):
        raise StageReceiptError("stage contract must be a mapping")
    if not isinstance(private_evidence, Mapping):
        raise StageReceiptError("private evidence access must be a mapping")

    output_records = {
        name: _tag_output(record, stage) for name, record in outputs.items()
    }
    if stage == PREDICTION_STAGE:
        if "prediction_inventory" not in output_records:
            raise StageReceiptError("prediction inventory is missing")
        inventory = output_records["prediction_inventory"]
    else:
        inventory = None
    if stage == EVALUATION_STAGE:
        if not isinstance(predecessor_mapping, Mapping) or predecessor_mapping.get(
            "prediction_inventory"
        ) is None:
            raise StageReceiptError("evaluation predecessor lacks prediction inventory")
        if (
            not isinstance(inputs, Mapping)
            or _plain(inputs.get("prediction_inventory"))
            != predecessor_mapping["prediction_inventory"]
        ):
            raise StageReceiptError(
                "evaluation prediction inventory differs from predecessor"
            )
    body = {
        "schema": STAGE_RECEIPT_SCHEMA,
        "schema_version": 1,
        "candidate_id": candidate["candidate_id"],
        "stage": stage,
        "stage_index": index,
        "stage_order": list(STAGE_ORDER),
        "stage_contract": canonical_manifest(stage_contract),
        "predecessor": (
            None
            if predecessor_mapping is None
            else {
                "stage": predecessor_mapping["stage"],
                "stage_index": predecessor_mapping["stage_index"],
                "receipt_id": predecessor_mapping["receipt_id"],
            }
        ),
        "inputs": _plain(inputs),
        "outputs": output_records,
        "execution": {
            "code_identity": _plain(code_identity),
            "configuration": canonical_manifest(configuration),
            "command": list(command),
            "command_sha256": _canonical_digest(list(command), label="command"),
            "dependency_container_identity": canonical_manifest(dependency_container),
            "seeds": canonical_manifest(seeds),
            "determinism": canonical_manifest(determinism),
            "environment": canonical_manifest(environment),
            "runtime_trace": canonical_manifest(runtime_trace),
            "terminal_status": "succeeded",
        },
        "private_evidence": _plain(private_evidence),
        "ordering_evidence": {
            "stage_order": list(STAGE_ORDER),
            "stage_index": index,
            "predecessor_stage": predecessor_stage,
        },
        "prediction_inventory": inventory,
    }
    body["receipt_id"] = _canonical_digest(
        {key: value for key, value in body.items() if key != "receipt_id"},
        label="stage receipt",
    )
    return StageReceipt(_freeze(_validate_receipt_mapping(body, candidate_authority=candidate)))


class StageReceiptChain:
    """Seal the ordered five-stage chain for one Candidate Authority identity."""

    def __init__(
        self,
        candidate_authority: Mapping[str, Any] | CandidateAuthorityBundle,
    ) -> None:
        self._candidate = validate_candidate_authority(candidate_authority)
        if tuple(self._candidate["execution_matrix"]["required_stage_order"]) != STAGE_ORDER:
            raise StageReceiptError("candidate authority stage order differs")
        self._receipts: list[StageReceipt] = []

    @property
    def receipts(self) -> tuple[StageReceipt, ...]:
        return tuple(self._receipts)

    def seal_stage(self, **kwargs: Any) -> StageReceipt:
        expected = (
            STAGE_ORDER[len(self._receipts)]
            if len(self._receipts) < len(STAGE_ORDER)
            else None
        )
        if kwargs.get("stage") != expected:
            raise StageReceiptError(
                f"stage order differs; expected predecessor stage before {expected}"
            )
        receipt = _make_receipt(
            candidate_authority=self._candidate,
            predecessor=self._receipts[-1] if self._receipts else None,
            **kwargs,
        )
        self._receipts.append(receipt)
        return receipt


def validate_stage_receipt(
    receipt: Mapping[str, Any] | StageReceipt,
    candidate_authority: Mapping[str, Any] | CandidateAuthorityBundle | None = None,
) -> StageReceipt:
    """Recompute one receipt identity and all currently addressable artifacts."""

    return StageReceipt(
        _freeze(_validate_receipt_mapping(receipt, candidate_authority=candidate_authority))
    )


def write_stage_receipt(
    path: str | Path,
    receipt: Mapping[str, Any] | StageReceipt,
) -> Path:
    """Publish one receipt without replacing a different receipt."""

    validated = validate_stage_receipt(receipt)
    return write_frozen_json(path, validated.as_dict())


def load_stage_receipt(path: str | Path) -> StageReceipt:
    """Load and validate one immutable receipt and its referenced artifacts."""

    try:
        payload, _digest, _source = load_json_object(
            path,
            label="stage receipt",
        )
    except (OSError, TypeError, ValueError) as error:
        raise StageReceiptError(f"cannot load stage receipt: {error}") from error
    return validate_stage_receipt(payload)


def validate_receipt_chain(
    paths: Sequence[str | Path],
    candidate_authority: Mapping[str, Any] | CandidateAuthorityBundle,
) -> list[dict[str, Any]]:
    """Validate every persisted receipt, predecessor link, and stage order."""

    if len(paths) != len(STAGE_ORDER):
        raise StageReceiptError("receipt chain is incomplete")
    candidate = validate_candidate_authority(candidate_authority)
    validated: list[dict[str, Any]] = []
    previous: StageReceipt | None = None
    for index, path in enumerate(paths):
        receipt = validate_stage_receipt(load_stage_receipt(path), candidate)
        if receipt["stage"] != STAGE_ORDER[index]:
            raise StageReceiptError("receipt chain stage order differs")
        expected_predecessor = _predecessor_reference(previous)
        if receipt["predecessor"] != expected_predecessor:
            raise StageReceiptError("receipt predecessor chain differs")
        if receipt["stage"] == EVALUATION_STAGE:
            if previous is None or receipt["inputs"].get("prediction_inventory") != previous[
                "prediction_inventory"
            ]:
                raise StageReceiptError(
                    "evaluation prediction inventory differs from predecessor"
                )
        validated.append(receipt.as_dict())
        previous = receipt
    return validated


__all__ = [
    "EVALUATION_STAGE",
    "PREDICTION_STAGE",
    "STAGE_ORDER",
    "STAGE_RECEIPT_SCHEMA",
    "StageReceipt",
    "StageReceiptChain",
    "StageReceiptError",
    "canonical_manifest",
    "directory_merkle",
    "load_stage_receipt",
    "opaque_file",
    "prediction_inventory",
    "tensor_container",
    "validate_receipt_chain",
    "validate_stage_receipt",
    "write_stage_receipt",
]
