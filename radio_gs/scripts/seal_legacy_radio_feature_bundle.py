#!/usr/bin/env python3
"""Content-address an existing legacy RADIO tensor directory without inference.

The sealer is intentionally narrower than feature extraction.  It does not
claim missing runtime/checkpoint provenance and never recomputes a tensor.  It
only wraps an externally hash-bound legacy manifest after reopening every
declared source image and feature tensor with the current fail-closed loaders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

from radio_gs.scripts.extract_radio_features import (
    INCOMPLETE_RUNTIME_RESEAL_CONTRACT,
    LEGACY_RESEAL_CONTRACT,
    LEGACY_SOURCE_MANIFEST_FILENAME,
    OUTPUT_BUNDLE_SCHEMA_VERSION,
    _canonical_json_sha256,
    _expected_tensor_relative_paths,
    _feature_signature,
    _load_validated_tensor,
    _merge_feature_signature,
    _sha256_file,
    _validate_final_output_bundle,
)
from radio_gs.utils.immutable_artifacts import stable_descriptor_load


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, payload: object) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    _atomic_bytes(path, encoded)


def _load_source_manifest_bytes(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, object], bytes, str]:
    def load(handle) -> bytes:
        return handle.read()

    raw, digest, _source = stable_descriptor_load(
        path,
        load,
        expected_sha256=str(expected_sha256),
        label="legacy RADIO feature manifest",
    )
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("legacy RADIO feature manifest is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("legacy RADIO feature manifest must contain an object")
    return value, raw, digest


def _tensor_record(root: Path, relative_path: str) -> tuple[dict[str, object], torch.Tensor]:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe legacy tensor path: {relative_path}")

    def load(handle):
        try:
            return torch.load(handle, map_location="cpu", weights_only=True)
        except TypeError as exc:
            raise RuntimeError(
                "legacy tensor sealing requires weights_only=True support"
            ) from exc

    value, digest, _source = stable_descriptor_load(
        root / relative,
        load,
        label="legacy RADIO feature tensor",
    )
    if not torch.is_tensor(value):
        raise ValueError(f"legacy feature artifact is not a tensor: {relative_path}")
    record: dict[str, object] = {
        "relative_path": relative.as_posix(),
        "sha256": digest,
        "dtype": str(value.dtype).removeprefix("torch."),
        "shape": [int(dimension) for dimension in value.shape],
        "num_bytes": int(value.nelement() * value.element_size()),
    }
    reopened = _load_validated_tensor(root / relative, record)
    if not torch.equal(value, reopened):
        raise ValueError(f"legacy tensor changed while sealing: {relative_path}")
    return record, value


def seal_legacy_bundle(
    feature_dir: str | Path,
    *,
    expected_legacy_manifest_sha256: str,
    receipt_path: str | Path | None = None,
    source_image_dir_override: str | Path | None = None,
) -> dict[str, object]:
    root = Path(feature_dir).expanduser().resolve()
    manifest_path = root / "frame_manifest.json"
    expected = str(expected_legacy_manifest_sha256)
    if len(expected) != 64 or any(value not in "0123456789abcdef" for value in expected):
        raise ValueError("a lowercase trusted legacy manifest SHA-256 is required")

    current = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(current, dict) and isinstance(current.get("execution"), dict):
        execution = current["execution"]
        formalization_contract = execution.get("formalization_contract")
        expected_source_sha = (
            execution.get("legacy_source_manifest_sha256")
            if formalization_contract == LEGACY_RESEAL_CONTRACT
            else execution.get("incomplete_runtime_source_manifest_sha256")
        )
        if formalization_contract in {
            LEGACY_RESEAL_CONTRACT,
            INCOMPLETE_RUNTIME_RESEAL_CONTRACT,
        } and expected_source_sha == expected:
            validation = _validate_final_output_bundle(root, current)
            return {
                **validation,
                "feature_dir": str(root),
                "idempotent_existing_seal": True,
            }
        is_incomplete_runtime = (
            not str(execution.get("resume_contract", ""))
            and not str(execution.get("resume_contract_sha256", ""))
            and not str(execution.get("resume_contract_file_sha256", ""))
            and current.get("output_bundle") is None
            and not str(current.get("output_bundle_sha256", ""))
        )
        if not is_incomplete_runtime:
            raise ValueError(
                "feature manifest is already formalized under another contract"
            )
    else:
        is_incomplete_runtime = False

    legacy, legacy_bytes, legacy_sha256 = _load_source_manifest_bytes(
        manifest_path,
        expected_sha256=expected,
    )
    if is_incomplete_runtime:
        source_execution = legacy.get("execution")
        if not isinstance(source_execution, dict):
            raise ValueError("incomplete-runtime execution provenance is missing")
        if (
            str(source_execution.get("resume_contract", ""))
            or str(source_execution.get("resume_contract_sha256", ""))
            or str(source_execution.get("resume_contract_file_sha256", ""))
            or legacy.get("output_bundle") is not None
            or str(legacy.get("output_bundle_sha256", ""))
        ):
            raise ValueError(
                "input is not an unbundled completed runtime extraction"
            )
        formalization_contract = INCOMPLETE_RUNTIME_RESEAL_CONTRACT
        source_manifest_name = f"frame_manifest.original.{legacy_sha256}.json"
        source_manifest_sha_key = "incomplete_runtime_source_manifest_sha256"
        source_manifest_name_key = "incomplete_runtime_source_manifest"
    else:
        if any(
            key in legacy
            for key in ("execution", "output_bundle", "output_bundle_sha256")
        ):
            raise ValueError("input manifest is not an unsealed legacy manifest")
        source_execution = None
        formalization_contract = LEGACY_RESEAL_CONTRACT
        source_manifest_name = LEGACY_SOURCE_MANIFEST_FILENAME
        source_manifest_sha_key = "legacy_source_manifest_sha256"
        source_manifest_name_key = "legacy_source_manifest"
    frames = legacy.get("frames")
    radio = legacy.get("radio")
    features = legacy.get("features")
    if not isinstance(frames, list) or not frames:
        raise ValueError("legacy manifest has no frames")
    if int(legacy.get("num_frames", -1)) != len(frames):
        raise ValueError("legacy manifest frame count differs")
    if not isinstance(radio, dict) or not isinstance(features, dict):
        raise ValueError("legacy RADIO/features declaration is incomplete")
    requested_adaptors = radio.get("requested_adaptors", [])
    if not isinstance(requested_adaptors, list) or any(
        not isinstance(value, str) for value in requested_adaptors
    ):
        raise ValueError("legacy requested_adaptors declaration is invalid")
    adaptor_names = [str(value) for value in requested_adaptors]

    declared_image_dir = Path(
        str(legacy.get("image_dir", ""))
    ).expanduser().resolve()
    image_dir = (
        Path(source_image_dir_override).expanduser().resolve()
        if source_image_dir_override is not None
        else declared_image_dir
    )
    source_image_resolution = (
        "declared_path"
        if image_dir == declared_image_dir
        else "explicit_override_all_frame_sha256_v1"
    )
    snapshots: list[dict[str, object]] = []
    signature: dict[str, object] | None = None
    logical_bytes = 0
    expected_tensor_paths: set[str] = set()
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("legacy frame declaration is invalid")
        source = image_dir / str(frame.get("source_file", ""))
        if not source.is_file():
            raise ValueError(f"legacy source image is missing: {source}")
        if _sha256_file(source) != str(frame.get("source_sha256", "")):
            raise ValueError(f"legacy source image SHA256 differs: {source}")
        stem = str(frame.get("saved_stem", ""))
        paths = _expected_tensor_relative_paths(stem, adaptor_names)
        records: list[dict[str, object]] = []
        values: dict[str, torch.Tensor] = {}
        for relative_path in paths:
            if relative_path in expected_tensor_paths:
                raise ValueError(f"legacy tensor path is repeated: {relative_path}")
            expected_tensor_paths.add(relative_path)
            record, value = _tensor_record(root, relative_path)
            records.append(record)
            values[relative_path] = value
            logical_bytes += int(record["num_bytes"])
        adaptors = {
            name: values[f"{name}/{stem}.pt"]
            for name in adaptor_names
        }
        frame_signature = _feature_signature(
            values[f"backbone/{stem}.pt"],
            values[f"summary/{stem}.pt"],
            adaptors,
            adaptor_names,
            require_all_adaptors=True,
        )
        signature = _merge_feature_signature(signature, frame_signature)
        snapshots.append(
            {
                "frame": frame,
                "feature_signature": frame_signature,
                "tensors": records,
            }
        )
    if signature != features:
        raise ValueError("legacy tensor signatures differ from the source manifest")

    feature_subdirs = {
        str(record.get("subdir", ""))
        for record in [
            features.get("backbone", {}),
            features.get("summary", {}),
            *list(features.get("adaptors", [])),
        ]
        if isinstance(record, dict) and str(record.get("subdir", ""))
    }
    observed_tensor_paths: set[str] = set()
    for subdir in feature_subdirs:
        directory = root / subdir
        if not directory.is_dir():
            raise ValueError(f"legacy feature directory is missing: {directory}")
        observed_tensor_paths.update(
            path.relative_to(root).as_posix()
            for path in directory.iterdir()
            if path.is_file() and path.suffix == ".pt"
        )
    if observed_tensor_paths != expected_tensor_paths:
        missing = sorted(expected_tensor_paths - observed_tensor_paths)
        extra = sorted(observed_tensor_paths - expected_tensor_paths)
        raise ValueError(
            f"legacy tensor disk set differs: missing={missing[:3]} extra={extra[:3]}"
        )

    source_copy = root / source_manifest_name
    if source_copy.exists():
        if _sha256_file(source_copy) != legacy_sha256:
            raise ValueError("existing legacy source-manifest copy differs")
    else:
        _atomic_bytes(source_copy, legacy_bytes)
    if is_incomplete_runtime:
        source_copy.chmod(source_copy.stat().st_mode & ~0o222)
    sealer_sha256 = _sha256_file(Path(__file__).resolve())
    execution: dict[str, object] = {
        "formalization_contract": formalization_contract,
        source_manifest_name_key: source_manifest_name,
        source_manifest_sha_key: legacy_sha256,
        "sealer": str(Path(__file__).resolve()),
        "sealer_sha256": sealer_sha256,
        "tensor_load_contract": "same_fd_sha256_weights_only_dtype_shape_finite_v1",
        "source_image_validation": "path_and_sha256_from_legacy_manifest_v1",
        "declared_source_image_dir": str(declared_image_dir),
        "resolved_source_image_dir": str(image_dir),
        "source_image_dir_resolution": source_image_resolution,
        "reseal_inputs": "legacy_manifest_declared_source_images_and_feature_tensors_only",
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
        "original_extraction_runtime_provenance": (
            "preserved_verbatim_from_incomplete_runtime_manifest"
            if is_incomplete_runtime
            else "legacy_unavailable_not_invented"
        ),
    }
    if source_execution is not None:
        execution["original_extraction_execution"] = source_execution
    bundle: dict[str, object] = {
        "schema_version": OUTPUT_BUNDLE_SCHEMA_VERSION,
        "contract": "radio-feature-output-bundle-v1",
        "source_contract": formalization_contract,
        source_manifest_sha_key: legacy_sha256,
        "frames": snapshots,
    }
    bundle_sha256 = _canonical_json_sha256(bundle)
    sealed = {
        **legacy,
        "execution": execution,
        "output_bundle": bundle,
        "output_bundle_sha256": bundle_sha256,
    }
    _atomic_json(manifest_path, sealed)
    validation = _validate_final_output_bundle(
        root,
        expected_output_bundle_sha256=bundle_sha256,
    )
    receipt: dict[str, object] = {
        "schema_version": (
            "radio_feature_incomplete_runtime_reseal_receipt_v1"
            if is_incomplete_runtime
            else "radio_feature_legacy_reseal_receipt_v1"
        ),
        "feature_dir": str(root),
        "legacy_source_manifest": str(source_copy),
        "legacy_source_manifest_sha256": legacy_sha256,
        "sealed_manifest": str(manifest_path),
        "sealed_manifest_sha256": validation["manifest_sha256"],
        "output_bundle_sha256": bundle_sha256,
        "sealer_sha256": sealer_sha256,
        "num_frames": len(frames),
        "num_tensors": len(expected_tensor_paths),
        "logical_tensor_bytes": logical_bytes,
        "feature_signature": signature,
        "declared_source_image_dir": str(declared_image_dir),
        "resolved_source_image_dir": str(image_dir),
        "source_image_dir_resolution": source_image_resolution,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
        "tensor_values_modified": False,
        "formalization_contract": formalization_contract,
        "original_extraction_runtime_provenance": (
            "preserved_verbatim_from_incomplete_runtime_manifest"
            if is_incomplete_runtime
            else "legacy_unavailable_not_invented"
        ),
    }
    destination = (
        Path(receipt_path).expanduser().resolve()
        if receipt_path is not None
        else root / "legacy_reseal_receipt.json"
    )
    _atomic_json(destination, receipt)
    return {**receipt, "receipt": str(destination), "receipt_sha256": _sha256_file(destination)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--expected-legacy-manifest-sha256", required=True)
    parser.add_argument("--receipt", default="")
    parser.add_argument("--source-image-dir-override", default="")
    args = parser.parse_args()
    result = seal_legacy_bundle(
        args.feature_dir,
        expected_legacy_manifest_sha256=args.expected_legacy_manifest_sha256,
        receipt_path=args.receipt or None,
        source_image_dir_override=args.source_image_dir_override or None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
