#!/usr/bin/env python3
"""Content-address legacy LERF official crop-summary maps without inference.

This is a sibling formalization utility.  It never edits the historical tensor
directory and does not call feature extraction, a renderer, or a benchmark
evaluator.  The exact source-frame inventory is derived from a hash-bound
responsibility authority plus preregistered source-heldout frame IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping

import torch


SCHEMA = "radio_gs.lerf_official_crop_summary_reseal.v1"
SCHEMA_VERSION = 1
PREREGISTRATION_SCHEMA = (
    "radio_gs.lerf_official_crop_summary_source_heldout_preregistration.v1"
)
RAMEN_SOURCE_HELDOUT_FRAME_IDS = (2, 45, 87, 130)
_FRAME_PATTERN = re.compile(r"rgb_(\d+)\.pt")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _trusted_json(
    path: str | Path,
    expected_sha256: str,
    *,
    label: str,
) -> tuple[dict[str, object], dict[str, str]]:
    source = Path(path).expanduser().resolve()
    expected = str(expected_sha256)
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"{label} requires a lowercase trusted SHA-256")
    raw = source.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise ValueError(f"{label} SHA-256 differs: expected {expected}, got {actual}")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object")
    return value, {"path": str(source), "sha256": actual}


def _scene_contract(
    preregistration: Mapping[str, object], scene: str
) -> dict[str, object]:
    if preregistration.get("schema") != PREREGISTRATION_SCHEMA:
        raise ValueError("official crop-summary preregistration schema differs")
    if int(preregistration.get("schema_version", -1)) != 1:
        raise ValueError("official crop-summary preregistration version differs")
    if preregistration.get("status") != "sealed_before_source_gate_execution":
        raise ValueError("official crop-summary preregistration is not sealed")
    if preregistration.get("target_data_or_metrics_opened_at_seal") is not False:
        raise ValueError("preregistration is not target blind")
    if preregistration.get("target_metric_execution_authorized") is not False:
        raise ValueError("preregistration unexpectedly authorizes target metrics")
    scenes = preregistration.get("scenes")
    if not isinstance(scenes, dict) or scene not in scenes:
        raise ValueError(f"scene is absent from preregistration: {scene}")
    contract = scenes[scene]
    if not isinstance(contract, dict):
        raise ValueError(f"invalid preregistered scene contract: {scene}")
    return dict(contract)


def _integer_list(value: object, *, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{label} must contain integers")
    result = [int(item) for item in value]
    if result != sorted(set(result)):
        raise ValueError(f"{label} must be sorted and unique")
    return result


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must be an exact path/SHA-256 record")
    path = str(value["path"])
    digest = str(value["sha256"])
    if not path or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"{label} is invalid")
    return {"path": path, "sha256": digest}


def validate_preregistered_implementation(
    preregistration: Mapping[str, object], role: str
) -> dict[str, str]:
    implementations = preregistration.get("implementation")
    if not isinstance(implementations, dict):
        raise ValueError("preregistered implementation records are missing")
    record = _record(
        implementations.get(role), label=f"preregistered {role} implementation"
    )
    path = Path(record["path"]).expanduser().resolve()
    if file_sha256(path) != record["sha256"]:
        raise ValueError(f"preregistered {role} implementation SHA-256 differs")
    return {"path": str(path), "sha256": record["sha256"]}


def _selected_frame_ids(contract: Mapping[str, object]) -> tuple[list[int], dict[str, str]]:
    authority_record = _record(
        contract.get("selected_view_authority"), label="selected-view authority"
    )
    authority, verified = _trusted_json(
        authority_record["path"],
        authority_record["sha256"],
        label="selected-view authority",
    )
    frame_ids = _integer_list(authority.get("frame_indices"), label="authority frame IDs")
    metadata = authority.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("selected-view authority metadata is missing")
    metadata_ids = _integer_list(
        metadata.get("selected_frame_indices"),
        label="authority metadata selected-frame IDs",
    )
    if frame_ids != metadata_ids:
        raise ValueError("selected-view authority frame declarations differ")
    if len(frame_ids) != int(contract.get("selected_frame_count", -1)):
        raise ValueError("selected-view authority frame count differs")
    return frame_ids, verified


def validate_scene_contract(
    preregistration: Mapping[str, object], scene: str
) -> tuple[dict[str, object], list[int], list[int], list[int], dict[str, str]]:
    contract = _scene_contract(preregistration, scene)
    selected, authority_record = _selected_frame_ids(contract)
    heldout = _integer_list(
        contract.get("source_heldout_frame_ids"), label="source-heldout frame IDs"
    )
    forbidden = _integer_list(
        contract.get("forbidden_target_frame_ids"), label="forbidden target frame IDs"
    )
    if scene == "ramen" and tuple(heldout) != RAMEN_SOURCE_HELDOUT_FRAME_IDS:
        raise ValueError(
            "Ramen source-heldout gate requires exactly explicit frames 2,45,87,130"
        )
    if set(selected).intersection(heldout):
        raise ValueError("source-heldout frames overlap selected source views")
    if set(selected).intersection(forbidden) or set(heldout).intersection(forbidden):
        raise ValueError("source frames overlap forbidden target frames")
    expected = sorted(selected + heldout)
    if len(expected) != int(contract.get("raw_frame_count", -1)):
        raise ValueError("preregistered raw frame count differs")
    return contract, selected, heldout, forbidden, authority_record


def load_preregistration(
    path: str | Path, expected_sha256: str
) -> tuple[dict[str, object], dict[str, str]]:
    return _trusted_json(
        path,
        expected_sha256,
        label="official crop-summary source-heldout preregistration",
    )


def _inventory(source_dir: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in source_dir.glob("rgb_*.pt"):
        match = _FRAME_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected crop-summary tensor filename: {path.name}")
        frame_id = int(match.group(1))
        if frame_id in result:
            raise ValueError(f"duplicated crop-summary frame ID: {frame_id}")
        result[frame_id] = path.resolve()
    return result


def build_reseal_plan(
    preregistration: Mapping[str, object],
    preregistration_record: Mapping[str, str],
    *,
    scene: str,
) -> dict[str, object]:
    resealer_record = validate_preregistered_implementation(
        preregistration, "resealer"
    )
    contract, selected, heldout, forbidden, authority_record = validate_scene_contract(
        preregistration, scene
    )
    source_dir = Path(str(contract.get("raw_tensor_dir", ""))).expanduser().resolve()
    if not source_dir.is_dir():
        raise ValueError(f"crop-summary tensor directory is missing: {source_dir}")
    inventory = _inventory(source_dir)
    expected = sorted(selected + heldout)
    if sorted(inventory) != expected:
        missing = sorted(set(expected) - set(inventory))
        unexpected = sorted(set(inventory) - set(expected))
        raise ValueError(
            f"crop-summary frame inventory differs; missing={missing}, unexpected={unexpected}"
        )
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "mode": "metadata_only_reseal_plan_no_tensor_bytes_opened",
        "scene": scene,
        "preregistration": dict(preregistration_record),
        "resealer_implementation": resealer_record,
        "selected_view_authority": authority_record,
        "source_tensor_dir": str(source_dir),
        "selected_frame_ids": selected,
        "source_heldout_frame_ids": heldout,
        "forbidden_target_frame_ids": forbidden,
        "expected_frame_ids": expected,
        "num_frames": len(expected),
        "tensor_contract": dict(contract.get("tensor_contract", {})),
        "target_data_or_metrics_opened": False,
        "tensor_content_hashes_computed": False,
    }


def _load_tensor(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("crop-summary reseal requires weights_only=True support") from exc
    if not torch.is_tensor(value):
        raise ValueError(f"crop-summary artifact is not a tensor: {path}")
    return value


def _tensor_record(
    path: Path,
    *,
    source_dir: Path,
    frame_id: int,
    expected_shape: list[int],
    expected_dtype: str,
    norm_tolerance: float,
) -> dict[str, object]:
    before = path.stat()
    digest = file_sha256(path)
    value = _load_tensor(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"crop-summary tensor changed while sealing: {path}")
    shape = [int(dimension) for dimension in value.shape]
    dtype = str(value.dtype).removeprefix("torch.")
    if shape != expected_shape:
        raise ValueError(f"crop-summary tensor shape differs for frame {frame_id}: {shape}")
    if dtype != expected_dtype:
        raise ValueError(f"crop-summary tensor dtype differs for frame {frame_id}: {dtype}")
    values = value.float()
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f"crop-summary tensor is non-finite for frame {frame_id}")
    norms = torch.linalg.vector_norm(values, dim=0)
    nonzero = norms > 0
    maximum_norm_deviation = (
        float((norms[nonzero] - 1.0).abs().max()) if bool(nonzero.any()) else 0.0
    )
    if maximum_norm_deviation > norm_tolerance:
        raise ValueError(
            f"crop-summary descriptor normalization differs for frame {frame_id}: "
            f"{maximum_norm_deviation} > {norm_tolerance}"
        )
    return {
        "frame_id": int(frame_id),
        "relative_path": path.relative_to(source_dir).as_posix(),
        "sha256": digest,
        "file_size_bytes": int(after.st_size),
        "shape": shape,
        "dtype": dtype,
        "zero_descriptor_count": int((~nonzero).sum()),
        "maximum_nonzero_norm_deviation": maximum_norm_deviation,
    }


def seal_bundle(
    preregistration_path: str | Path,
    expected_preregistration_sha256: str,
    *,
    scene: str,
    output: str | Path,
) -> dict[str, object]:
    preregistration, preregistration_record = load_preregistration(
        preregistration_path, expected_preregistration_sha256
    )
    plan = build_reseal_plan(
        preregistration, preregistration_record, scene=scene
    )
    contract = _scene_contract(preregistration, scene)
    tensor_contract = contract.get("tensor_contract")
    if not isinstance(tensor_contract, dict):
        raise ValueError("crop-summary tensor contract is missing")
    shape = tensor_contract.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or any(isinstance(item, bool) or not isinstance(item, int) for item in shape)
    ):
        raise ValueError("crop-summary tensor shape contract is invalid")
    expected_shape = [int(item) for item in shape]
    expected_dtype = str(tensor_contract.get("dtype", ""))
    norm_tolerance = float(tensor_contract.get("maximum_norm_deviation", -1.0))
    if expected_dtype != "float16" or norm_tolerance < 0:
        raise ValueError("crop-summary dtype/normalization contract is invalid")

    source_dir = Path(str(plan["source_tensor_dir"]))
    inventory = _inventory(source_dir)
    records = [
        _tensor_record(
            inventory[frame_id],
            source_dir=source_dir,
            frame_id=frame_id,
            expected_shape=expected_shape,
            expected_dtype=expected_dtype,
            norm_tolerance=norm_tolerance,
        )
        for frame_id in plan["expected_frame_ids"]
    ]
    bundle_contract = {
        "scene": scene,
        "source_tensor_dir": str(source_dir),
        "frames": records,
    }
    payload = {
        **plan,
        "mode": "content_addressed_immutable_reseal",
        "frame_records": records,
        "frame_record_bundle_sha256": canonical_json_sha256(bundle_contract),
        "tensor_content_hashes_computed": True,
        "source_directory_modified": False,
    }
    output_path = Path(output).expanduser().resolve()
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise FileExistsError(f"refusing to replace different reseal: {output_path}")
        return {**payload, "idempotent_existing_seal": True}
    _atomic_json(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--expected-preregistration-sha256", required=True)
    parser.add_argument("--scene", required=True, choices=("ramen", "teatime"))
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate filenames/authorities only; do not open or hash tensor bytes.",
    )
    args = parser.parse_args()
    preregistration, preregistration_record = load_preregistration(
        args.preregistration, args.expected_preregistration_sha256
    )
    if args.plan_only:
        payload = build_reseal_plan(
            preregistration, preregistration_record, scene=args.scene
        )
    else:
        if not str(args.output).strip():
            parser.error("--output is required unless --plan-only is used")
        payload = seal_bundle(
            args.preregistration,
            args.expected_preregistration_sha256,
            scene=args.scene,
            output=args.output,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
