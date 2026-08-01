"""Durable, fail-closed per-scene resume artifacts for SurfaceRegion caches.

The final cache remains the only promotable ``.pt`` artifact.  Scene partials
live under an explicit resume directory, use a non-``.pt`` suffix, and become
trusted only after a separate immutable JSON terminal binds their SHA-256 to
the frozen full-run contract.  Tensor deserialization is always
``weights_only=True`` through one no-follow descriptor.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
import os
from pathlib import Path
import random
import re
from typing import Any

import torch

from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    fsync_directory,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


RESUME_CONTRACT_ARTIFACT_TYPE = "surface-region-scene-resume-contract-v1"
SCENE_PARTIAL_ARTIFACT_TYPE = "surface-region-scene-partial-v1"
SCENE_TERMINAL_ARTIFACT_TYPE = "surface-region-scene-partial-terminal-v1"
RESUME_SCHEMA_VERSION = 1
SCENE_PARTIAL_SUFFIX = ".surface-scene.partial"
SCENE_TERMINAL_SUFFIX = ".surface-scene.complete.json"

SCENE_TENSOR_KEYS = (
    "radio_features",
    "geometry",
    "token_mask",
    "reliability",
    "official_summary_tokens",
    "official_crop_summaries",
    "teacher_mask",
    "anchor_index",
)
_SCENE_NAME = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")


class SceneResumeStateError(RuntimeError):
    """A resume directory cannot be trusted without explicit quarantine."""


def _resume_failure(resume_dir: Path, reason: str) -> SceneResumeStateError:
    quarantine = resume_dir.with_name(f"{resume_dir.name}.quarantine-required")
    return SceneResumeStateError(
        f"stale/corrupt SurfaceRegion scene resume state: {reason}; "
        f"move the whole resume directory to {quarantine} before rebuilding"
    )


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _canonical_output_path(path: str | Path) -> Path:
    raw = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    raw.parent.mkdir(parents=True, exist_ok=True)
    return raw.parent.resolve(strict=True) / raw.name


def _prepare_resume_directory(path: str | Path) -> Path:
    raw = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    if _lexists(raw) and raw.is_symlink():
        raise ValueError(f"refuse symlinked scene resume directory: {raw}")
    raw.mkdir(parents=True, exist_ok=True)
    info = os.lstat(raw)
    if not os.path.isdir(raw) or os.path.islink(raw):
        raise ValueError(f"scene resume path is not a real directory: {raw}")
    if not info.st_nlink:
        raise ValueError(f"scene resume directory has invalid link state: {raw}")
    return raw.resolve(strict=True)


def _require_exact_keys(
    value: object,
    expected: set[str] | frozenset[str],
    *,
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    actual = set(value)
    if actual != set(expected):
        raise ValueError(
            f"{label} fields differ: missing={sorted(set(expected) - actual)}, "
            f"unexpected={sorted(actual - set(expected))}"
        )
    return value


def _scene_stem(scene_index: int, scene_name: str) -> str:
    if not isinstance(scene_index, int) or isinstance(scene_index, bool):
        raise TypeError("scene index must be an integer")
    if scene_index < 0:
        raise ValueError("scene index must be non-negative")
    if not isinstance(scene_name, str) or _SCENE_NAME.fullmatch(scene_name) is None:
        raise ValueError(f"invalid ScanNet scene name: {scene_name!r}")
    return f"scene-{scene_index:04d}-{scene_name}"


def scene_resume_paths(
    resume_dir: str | Path,
    *,
    scene_index: int,
    scene_name: str,
) -> tuple[Path, Path]:
    root = Path(resume_dir)
    stem = _scene_stem(scene_index, scene_name)
    return (
        root / f"{stem}{SCENE_PARTIAL_SUFFIX}",
        root / f"{stem}{SCENE_TERMINAL_SUFFIX}",
    )


def _is_abandoned_staging_file(path: Path, final_names: set[str]) -> bool:
    name = path.name
    if not (name.startswith(".") and name.endswith(".tmp")):
        return False
    return any(name.startswith(f".{final_name}.") for final_name in final_names)


def validate_resume_inventory(
    resume_dir: str | Path,
    selected_scenes: Sequence[str],
) -> None:
    """Reject unknown or half-published state; ignore only private staging files."""

    root = Path(resume_dir)
    allowed = {"contract.json"}
    for index, scene in enumerate(selected_scenes):
        partial, terminal = scene_resume_paths(
            root,
            scene_index=index,
            scene_name=scene,
        )
        allowed.update((partial.name, terminal.name))
    unexpected: list[str] = []
    for path in root.iterdir():
        if path.name in allowed:
            continue
        if _is_abandoned_staging_file(path, allowed):
            info = os.lstat(path)
            if not os.path.isfile(path) or os.path.islink(path) or info.st_nlink < 1:
                unexpected.append(path.name)
            continue
        unexpected.append(path.name)
    if unexpected:
        raise _resume_failure(
            root,
            f"unexpected inventory entries: {sorted(unexpected)}",
        )
    for index, scene in enumerate(selected_scenes):
        partial, terminal = scene_resume_paths(
            root,
            scene_index=index,
            scene_name=scene,
        )
        partial_present = _lexists(partial)
        terminal_present = _lexists(terminal)
        if partial_present and not terminal_present:
            # The data file is not authoritative until its external terminal
            # exists.  Preserve an interrupted publication as private staging
            # evidence, then recompute that one scene; earlier terminals stay
            # usable.  Symlinks/non-regular files fail in ``file_record``.
            record = file_record(partial)
            abandoned = root / (
                f".{partial.name}.abandoned-{record['sha256']}.tmp"
            )
            if _lexists(abandoned):
                if file_record(abandoned)["sha256"] != record["sha256"]:
                    raise _resume_failure(
                        root,
                        f"conflicting abandoned scene {index}:{scene}",
                    )
            else:
                os.link(partial, abandoned, follow_symlinks=False)
                fsync_directory(root)
            partial.unlink()
            fsync_directory(root)
            continue
        if terminal_present and not partial_present:
            raise _resume_failure(root, f"half-published scene {index}:{scene}")


def open_or_create_resume_contract(
    resume_dir: str | Path,
    contract: Mapping[str, Any],
) -> tuple[Path, dict[str, str], str]:
    """Freeze or exactly reopen the complete CLI/input resume contract."""

    root = _prepare_resume_directory(resume_dir)
    payload = dict(contract)
    _require_exact_keys(
        payload,
        {
            "artifact_type",
            "schema_version",
            "builder",
            "cli",
            "inputs",
            "selected_scenes",
            "row_contract",
            "resume_protocol",
        },
        label="scene resume contract",
    )
    if payload["artifact_type"] != RESUME_CONTRACT_ARTIFACT_TYPE:
        raise ValueError("scene resume contract artifact type differs")
    if payload["schema_version"] != RESUME_SCHEMA_VERSION:
        raise ValueError("scene resume contract schema version differs")
    scenes = payload["selected_scenes"]
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("scene resume contract has no selected scenes")
    if any(
        not isinstance(scene, str) or _SCENE_NAME.fullmatch(scene) is None
        for scene in scenes
    ) or len(scenes) != len(set(scenes)):
        raise ValueError("scene resume contract selected scenes differ")

    contract_path = root / "contract.json"
    if _lexists(contract_path):
        try:
            observed, _, _ = load_json_object(
                contract_path,
                label="SurfaceRegion scene resume contract",
            )
        except (OSError, ValueError) as error:
            raise _resume_failure(root, "contract cannot be reopened") from error
        if observed != payload:
            raise _resume_failure(root, "complete CLI/input contract drifted")
    else:
        unknown = [
            path.name
            for path in root.iterdir()
            if not _is_abandoned_staging_file(path, {"contract.json"})
        ]
        if unknown:
            raise _resume_failure(
                root,
                f"artifacts exist without a contract: {sorted(unknown)}",
            )
        write_frozen_json(contract_path, payload)

    observed, _, _ = load_json_object(
        contract_path,
        label="SurfaceRegion scene resume contract",
    )
    if observed != payload:
        raise _resume_failure(root, "contract differs after publication")
    contract_record = file_record(contract_path)
    contract_payload_sha256 = canonical_json_sha256(payload)
    validate_resume_inventory(root, scenes)
    return root, contract_record, contract_payload_sha256


def encode_rng_state(state: object) -> dict[str, object]:
    """Encode ``random.Random`` state using only JSON-safe basic values."""

    if not isinstance(state, tuple) or len(state) != 3:
        raise ValueError("Python RNG state structure differs")
    version, internal, gaussian = state
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or not isinstance(internal, tuple)
        or not internal
        or any(not isinstance(value, int) or isinstance(value, bool) for value in internal)
        or gaussian is not None
        and (not isinstance(gaussian, float) or not math.isfinite(gaussian))
    ):
        raise ValueError("Python RNG state values differ")
    encoded = {
        "version": version,
        "internal": list(internal),
        "gaussian": gaussian,
    }
    # Round-trip through Random.setstate to reject malformed but JSON-shaped data.
    decode_rng_state(encoded)
    return encoded


def decode_rng_state(value: object) -> tuple[int, tuple[int, ...], float | None]:
    state = _require_exact_keys(
        value,
        {"version", "internal", "gaussian"},
        label="Python RNG state",
    )
    version = state["version"]
    internal = state["internal"]
    gaussian = state["gaussian"]
    if not isinstance(version, int) or isinstance(version, bool):
        raise ValueError("Python RNG state version differs")
    if not isinstance(internal, list) or not internal or any(
        not isinstance(item, int) or isinstance(item, bool) for item in internal
    ):
        raise ValueError("Python RNG internal state differs")
    if gaussian is not None and (
        not isinstance(gaussian, float) or not math.isfinite(gaussian)
    ):
        raise ValueError("Python RNG gaussian state differs")
    result = (version, tuple(internal), gaussian)
    checker = random.Random()
    try:
        checker.setstate(result)
    except (TypeError, ValueError) as error:
        raise ValueError("Python RNG state is invalid") from error
    return result


def tensor_sha256(value: torch.Tensor) -> str:
    if not torch.is_tensor(value):
        raise TypeError("tensor digest expects a tensor")
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(
        canonical_json_sha256(
            {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            }
        ).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _require_tensor(
    value: object,
    *,
    key: str,
    dtype: torch.dtype,
    shape: tuple[int | None, ...],
) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"scene partial {key} must be a tensor")
    tensor = value
    if tensor.device.type != "cpu" or tensor.dtype != dtype:
        raise ValueError(f"scene partial {key} device/dtype differs")
    if tensor.ndim != len(shape) or any(
        expected is not None and tensor.shape[index] != expected
        for index, expected in enumerate(shape)
    ):
        raise ValueError(
            f"scene partial {key} shape differs: {tuple(tensor.shape)}"
        )
    return tensor


def validate_scene_rows(
    scene_rows: Mapping[str, Any],
    *,
    scene_name: str,
    expected_rows: int,
    maximum_tokens: int,
    teacher_views: int,
) -> dict[str, Any]:
    rows = dict(
        _require_exact_keys(
            scene_rows,
            set(SCENE_TENSOR_KEYS) | {"records"},
            label="scene partial rows",
        )
    )
    if not isinstance(expected_rows, int) or expected_rows <= 0:
        raise ValueError("expected scene row count must be positive")
    records = rows["records"]
    if not isinstance(records, list) or len(records) != expected_rows:
        raise ValueError("scene partial records have a wrong row count")

    features = _require_tensor(
        rows["radio_features"],
        key="radio_features",
        dtype=torch.float16,
        shape=(expected_rows, maximum_tokens, 1280),
    )
    geometry = _require_tensor(
        rows["geometry"],
        key="geometry",
        dtype=torch.float16,
        shape=(expected_rows, maximum_tokens, 14),
    )
    token_mask = _require_tensor(
        rows["token_mask"],
        key="token_mask",
        dtype=torch.bool,
        shape=(expected_rows, maximum_tokens),
    )
    reliability = _require_tensor(
        rows["reliability"],
        key="reliability",
        dtype=torch.float16,
        shape=(expected_rows, maximum_tokens, 1),
    )
    summary = _require_tensor(
        rows["official_summary_tokens"],
        key="official_summary_tokens",
        dtype=torch.float16,
        shape=(expected_rows, teacher_views, 1280),
    )
    descriptors = _require_tensor(
        rows["official_crop_summaries"],
        key="official_crop_summaries",
        dtype=torch.float16,
        shape=(expected_rows, teacher_views, None),
    )
    teacher_mask = _require_tensor(
        rows["teacher_mask"],
        key="teacher_mask",
        dtype=torch.bool,
        shape=(expected_rows, teacher_views),
    )
    anchor = _require_tensor(
        rows["anchor_index"],
        key="anchor_index",
        dtype=torch.int64,
        shape=(expected_rows,),
    )
    if descriptors.shape[2] <= 0:
        raise ValueError("scene partial descriptor dimension must be positive")
    for key, tensor in (
        ("radio_features", features),
        ("geometry", geometry),
        ("reliability", reliability),
        ("official_summary_tokens", summary),
        ("official_crop_summaries", descriptors),
    ):
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"scene partial {key} contains non-finite values")
    token_counts = token_mask.sum(dim=1)
    teacher_counts = teacher_mask.sum(dim=1)
    if bool((token_counts <= 0).any()) or bool((teacher_counts < 2).any()):
        raise ValueError("scene partial masks contain incomplete rows")
    expected_teacher_mask = (
        torch.arange(teacher_views)[None, :] < teacher_counts[:, None]
    )
    if not torch.equal(teacher_mask, expected_teacher_mask):
        raise ValueError("scene partial teacher mask is not left aligned")
    if bool((anchor < 0).any()) or bool((anchor >= token_counts).any()):
        raise ValueError("scene partial anchor index lies outside valid tokens")
    if (
        bool(features[~token_mask].count_nonzero())
        or bool(geometry[~token_mask].count_nonzero())
        or bool(reliability[~token_mask].count_nonzero())
        or bool(summary[~teacher_mask].count_nonzero())
        or bool(descriptors[~teacher_mask].count_nonzero())
    ):
        raise ValueError("scene partial padding must be exactly zero")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError("scene partial record must be a dictionary")
        if (
            record.get("scene") != scene_name
            or int(record.get("tokens", -1)) != int(token_counts[index])
            or int(record.get("anchor_local_index", -1)) != int(anchor[index])
            or len(record.get("teacher_views", [])) != int(teacher_counts[index])
        ):
            raise ValueError("scene partial record differs from its tensor row")
    return rows


def scene_rows_digest(scene_rows: Mapping[str, Any]) -> tuple[dict[str, str], str]:
    tensor_digests = {
        key: tensor_sha256(torch.as_tensor(scene_rows[key]))
        for key in SCENE_TENSOR_KEYS
    }
    components = {
        "tensors": tensor_digests,
        "records_sha256": canonical_json_sha256(scene_rows["records"]),
    }
    return tensor_digests, canonical_json_sha256(components)


def _validate_partial_payload(
    payload: Mapping[str, Any],
    *,
    scene_index: int,
    scene_name: str,
    expected_rows: int,
    maximum_tokens: int,
    teacher_views: int,
    contract_record: Mapping[str, str],
    contract_payload_sha256: str,
) -> dict[str, Any]:
    value = _require_exact_keys(
        payload,
        {
            "artifact_type",
            "schema_version",
            "scene_index",
            "scene_name",
            "resume_contract",
            "resume_contract_payload_sha256",
            "rng_state_before",
            "rng_state_after",
            "rows",
            "tensor_sha256",
            "row_bundle_sha256",
        },
        label="scene partial payload",
    )
    if (
        value["artifact_type"] != SCENE_PARTIAL_ARTIFACT_TYPE
        or value["schema_version"] != RESUME_SCHEMA_VERSION
        or value["scene_index"] != scene_index
        or value["scene_name"] != scene_name
        or value["resume_contract"] != dict(contract_record)
        or value["resume_contract_payload_sha256"]
        != contract_payload_sha256
    ):
        raise ValueError("scene partial identity/contract differs")
    decode_rng_state(value["rng_state_before"])
    decode_rng_state(value["rng_state_after"])
    rows = validate_scene_rows(
        value["rows"],
        scene_name=scene_name,
        expected_rows=expected_rows,
        maximum_tokens=maximum_tokens,
        teacher_views=teacher_views,
    )
    tensor_digests, bundle_digest = scene_rows_digest(rows)
    if value["tensor_sha256"] != tensor_digests:
        raise ValueError("scene partial tensor digests differ")
    if value["row_bundle_sha256"] != bundle_digest:
        raise ValueError("scene partial row bundle digest differs")
    return dict(value)


def commit_scene_partial(
    resume_dir: str | Path,
    *,
    scene_index: int,
    scene_name: str,
    scene_rows: Mapping[str, Any],
    rng_state_before: object,
    rng_state_after: object,
    expected_rows: int,
    maximum_tokens: int,
    teacher_views: int,
    contract_record: Mapping[str, str],
    contract_payload_sha256: str,
) -> dict[str, Any]:
    """No-clobber publish a scene data file, then its SHA authority terminal."""

    root = Path(resume_dir)
    partial_path, terminal_path = scene_resume_paths(
        root,
        scene_index=scene_index,
        scene_name=scene_name,
    )
    if _lexists(partial_path) or _lexists(terminal_path):
        raise FileExistsError(
            f"scene resume artifact already exists: {scene_index}:{scene_name}"
        )
    rows = validate_scene_rows(
        scene_rows,
        scene_name=scene_name,
        expected_rows=expected_rows,
        maximum_tokens=maximum_tokens,
        teacher_views=teacher_views,
    )
    tensor_digests, bundle_digest = scene_rows_digest(rows)
    payload = {
        "artifact_type": SCENE_PARTIAL_ARTIFACT_TYPE,
        "schema_version": RESUME_SCHEMA_VERSION,
        "scene_index": scene_index,
        "scene_name": scene_name,
        "resume_contract": dict(contract_record),
        "resume_contract_payload_sha256": contract_payload_sha256,
        "rng_state_before": encode_rng_state(rng_state_before),
        "rng_state_after": encode_rng_state(rng_state_after),
        "rows": rows,
        "tensor_sha256": tensor_digests,
        "row_bundle_sha256": bundle_digest,
    }
    write_torch_noclobber(partial_path, payload)
    partial_record = file_record(partial_path)
    terminal = {
        "artifact_type": SCENE_TERMINAL_ARTIFACT_TYPE,
        "schema_version": RESUME_SCHEMA_VERSION,
        "scene_index": scene_index,
        "scene_name": scene_name,
        "rows": expected_rows,
        "resume_contract": dict(contract_record),
        "resume_contract_payload_sha256": contract_payload_sha256,
        "partial": partial_record,
        "row_bundle_sha256": bundle_digest,
    }
    write_frozen_json(terminal_path, terminal)
    return terminal


def load_scene_partial(
    resume_dir: str | Path,
    *,
    scene_index: int,
    scene_name: str,
    expected_rows: int,
    maximum_tokens: int,
    teacher_views: int,
    contract_record: Mapping[str, str],
    contract_payload_sha256: str,
) -> dict[str, Any] | None:
    """Strictly reopen one completed scene or return ``None`` if absent."""

    root = Path(resume_dir)
    partial_path, terminal_path = scene_resume_paths(
        root,
        scene_index=scene_index,
        scene_name=scene_name,
    )
    partial_present = _lexists(partial_path)
    terminal_present = _lexists(terminal_path)
    if not partial_present and not terminal_present:
        return None
    if partial_present != terminal_present:
        raise _resume_failure(root, f"half-published scene {scene_index}:{scene_name}")
    try:
        terminal, _, _ = load_json_object(
            terminal_path,
            label="SurfaceRegion scene partial terminal",
        )
        expected_terminal = _require_exact_keys(
            terminal,
            {
                "artifact_type",
                "schema_version",
                "scene_index",
                "scene_name",
                "rows",
                "resume_contract",
                "resume_contract_payload_sha256",
                "partial",
                "row_bundle_sha256",
            },
            label="scene partial terminal",
        )
        if (
            expected_terminal["artifact_type"] != SCENE_TERMINAL_ARTIFACT_TYPE
            or expected_terminal["schema_version"] != RESUME_SCHEMA_VERSION
            or expected_terminal["scene_index"] != scene_index
            or expected_terminal["scene_name"] != scene_name
            or expected_terminal["rows"] != expected_rows
            or expected_terminal["resume_contract"] != dict(contract_record)
            or expected_terminal["resume_contract_payload_sha256"]
            != contract_payload_sha256
        ):
            raise ValueError("scene partial terminal identity/contract differs")
        if not isinstance(expected_terminal["partial"], Mapping):
            raise ValueError("scene partial terminal lacks a file record")
        expected_partial_path = _canonical_output_path(partial_path)
        if expected_terminal["partial"].get("path") != str(expected_partial_path):
            raise ValueError("scene partial terminal path differs")
        validate_file_record(
            expected_terminal["resume_contract"],
            label="scene partial resume contract",
        )
        source = validate_file_record(
            expected_terminal["partial"],
            label="SurfaceRegion scene partial",
        )
        payload, _, _ = load_torch_mapping(
            source,
            expected_sha256=str(expected_terminal["partial"]["sha256"]),
            map_location="cpu",
            label="SurfaceRegion scene partial",
        )
        value = _validate_partial_payload(
            payload,
            scene_index=scene_index,
            scene_name=scene_name,
            expected_rows=expected_rows,
            maximum_tokens=maximum_tokens,
            teacher_views=teacher_views,
            contract_record=contract_record,
            contract_payload_sha256=contract_payload_sha256,
        )
        if value["row_bundle_sha256"] != expected_terminal["row_bundle_sha256"]:
            raise ValueError("scene partial terminal row digest differs")
        return value
    except Exception as error:
        raise _resume_failure(
            root,
            f"scene {scene_index}:{scene_name} cannot be trusted",
        ) from error


def append_scene_rows(
    scene_rows: Mapping[str, Any],
    *,
    records: list[dict[str, Any]],
    tensor_rows: Mapping[str, list[torch.Tensor]],
) -> None:
    """Append one validated scene in row order without changing tensor bytes."""

    records.extend(scene_rows["records"])
    for key in SCENE_TENSOR_KEYS:
        destination = tensor_rows[key]
        value = torch.as_tensor(scene_rows[key])
        destination.extend(value.unbind(0))
