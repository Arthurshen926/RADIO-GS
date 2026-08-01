#!/usr/bin/env python3
"""Build fit/dev/audit target-blind SigLIP2 embeddings in one CPU process.

The persisted artifacts remain the unchanged per-split v1 schema.  This
orchestrator validates the fixed snapshot once, retains one text tower, and
fails closed unless the output directory is empty or already contains one
complete byte-valid bundle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.scripts import (
    build_target_blind_siglip2_embedding_artifact as split_builder,
)


ARTIFACT_NAME = "target_blind_siglip2_{split}_embeddings.pt"
SIDECAR_NAME = "target_blind_siglip2_{split}_embeddings.manifest.json"

_PAYLOAD_KEYS = {
    "schema_version",
    "artifact_type",
    "algorithm_version",
    "benchmark_vocabulary_opened",
    "uses_benchmark_vocabulary_for_construction",
    "split",
    "split_synset_tab_query_lf_sha256",
    "prompt_templates",
    "text_canonicalization",
    "records",
    "queries",
    "synsets",
    "ordered_records_sha256",
    "vocabulary_path",
    "vocabulary_sha256",
    "vocabulary_manifest_path",
    "vocabulary_manifest_sha256",
    "embeddings",
    "embedding_semantic_sha256",
    "embedding_tensor_sha256",
    "text_encoder",
}
_SIDECAR_KEYS = {
    "schema_version",
    "artifact_type",
    "algorithm_version",
    "benchmark_vocabulary_opened",
    "uses_benchmark_vocabulary_for_construction",
    "split",
    "split_synset_tab_query_lf_sha256",
    "prompt_templates",
    "text_canonicalization",
    "records",
    "queries",
    "synsets",
    "ordered_records_sha256",
    "vocabulary",
    "text_encoder",
    "embedding",
    "artifact",
    "builder",
}


def bundle_output_paths(output_root: Path) -> dict[str, tuple[Path, Path]]:
    root = Path(output_root).resolve()
    return {
        split: (
            root / ARTIFACT_NAME.format(split=split),
            root / SIDECAR_NAME.format(split=split),
        )
        for split in split_builder.SPLITS
    }


def _classify_output_root(
    output_root: Path,
    paths: Mapping[str, tuple[Path, Path]],
) -> str:
    if not output_root.exists():
        return "empty"
    if not output_root.is_dir():
        raise FileExistsError("output_root exists but is not a directory")
    entries = {entry.name: entry for entry in output_root.iterdir()}
    if not entries:
        return "empty"
    expected = {path.name for split_paths in paths.values() for path in split_paths}
    if set(entries) != expected:
        missing = sorted(expected - set(entries))
        unexpected = sorted(set(entries) - expected)
        raise FileExistsError(
            "output_root contains a partial or mixed bundle; refusing overwrite: "
            f"missing={missing}, unexpected={unexpected}"
        )
    invalid = sorted(
        name
        for name, path in entries.items()
        if path.is_symlink() or not path.is_file()
    )
    if invalid:
        raise FileExistsError(
            "bundle outputs must be regular non-symlink files: " + ", ".join(invalid)
        )
    return "complete"


def _read_canonical_sidecar(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    if raw != split_builder._canonical_json_bytes(value) + b"\n":
        raise ValueError(f"{path} is not the canonical builder-emitted JSON bytes")
    return value


def _load_payload(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a mapping")
    return value


def _validate_existing_split(
    *,
    split: str,
    artifact: Path,
    sidecar_path: Path,
    vocabulary_contract: Mapping[str, Any],
    encoder_contract: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _load_payload(artifact)
    sidecar = _read_canonical_sidecar(sidecar_path)
    if (
        set(payload) != _PAYLOAD_KEYS
        or payload.get("schema_version") != split_builder.SCHEMA_VERSION
        or payload.get("artifact_type") != split_builder.ARTIFACT_TYPE
        or payload.get("algorithm_version") != split_builder.ALGORITHM_VERSION
        or set(sidecar) != _SIDECAR_KEYS
        or sidecar.get("schema_version") != split_builder.SCHEMA_VERSION
        or sidecar.get("artifact_type") != split_builder.MANIFEST_ARTIFACT_TYPE
        or sidecar.get("algorithm_version") != split_builder.ALGORITHM_VERSION
    ):
        raise ValueError(f"existing {split} artifact has the wrong split-v1 schema")
    for record in (payload, sidecar):
        if (
            record.get("benchmark_vocabulary_opened") is not False
            or record.get("uses_benchmark_vocabulary_for_construction") is not False
            or record.get("split") != split
            or record.get("prompt_templates") != ["{query}"]
            or record.get("text_canonicalization")
            != split_builder.TEXT_CANONICALIZATION
        ):
            raise ValueError(f"existing {split} artifact violates the frozen policy")

    records = list(vocabulary_contract["records"])
    queries = [record["query"] for record in records]
    synsets = [record["synset"] for record in records]
    common = {
        "split": split,
        "split_synset_tab_query_lf_sha256": vocabulary_contract["split_sha256"],
        "prompt_templates": ["{query}"],
        "text_canonicalization": split_builder.TEXT_CANONICALIZATION,
        "records": records,
        "queries": queries,
        "synsets": synsets,
        "ordered_records_sha256": vocabulary_contract["ordered_records_sha256"],
    }
    if any(payload.get(key) != value for key, value in common.items()):
        raise ValueError(f"existing {split} payload differs from its vocabulary")
    if any(sidecar.get(key) != value for key, value in common.items()):
        raise ValueError(f"existing {split} sidecar differs from its vocabulary")

    expected_vocabulary = {
        "path": str(vocabulary_contract["vocabulary_path"]),
        "sha256": vocabulary_contract["vocabulary_sha256"],
        "manifest_path": str(vocabulary_contract["vocabulary_manifest_path"]),
        "manifest_sha256": vocabulary_contract["vocabulary_manifest_sha256"],
    }
    if (
        payload.get("vocabulary_path") != expected_vocabulary["path"]
        or payload.get("vocabulary_sha256") != expected_vocabulary["sha256"]
        or payload.get("vocabulary_manifest_path")
        != expected_vocabulary["manifest_path"]
        or payload.get("vocabulary_manifest_sha256")
        != expected_vocabulary["manifest_sha256"]
        or sidecar.get("vocabulary") != expected_vocabulary
    ):
        raise ValueError(f"existing {split} vocabulary provenance is stale")
    expected_encoder = dict(encoder_contract)
    if (
        payload.get("text_encoder") != expected_encoder
        or sidecar.get("text_encoder") != expected_encoder
    ):
        raise ValueError(f"existing {split} snapshot provenance is stale")

    embeddings = split_builder._validate_embedding_batch(
        torch.as_tensor(payload.get("embeddings")),
        rows=len(records),
    )
    semantic_sha = split_builder.embedding_semantic_sha256(embeddings)
    typed_sha = tensor_sha256(embeddings)
    if (
        payload.get("embedding_semantic_sha256") != semantic_sha
        or payload.get("embedding_tensor_sha256") != typed_sha
    ):
        raise ValueError(f"existing {split} embedding tensor hash is invalid")
    expected_embedding = {
        "shape": [len(records), split_builder.OUTPUT_DIMENSION],
        "dtype": "float32",
        "byte_order": "little_endian",
        "normalization": "l2",
        "semantic_sha256": semantic_sha,
        "tensor_sha256": typed_sha,
    }
    if sidecar.get("embedding") != expected_embedding:
        raise ValueError(f"existing {split} embedding sidecar is invalid")

    artifact_record = sidecar.get("artifact")
    if not isinstance(artifact_record, Mapping) or set(artifact_record) != {
        "path",
        "sha256",
    }:
        raise ValueError(f"existing {split} artifact binding is invalid")
    if Path(str(artifact_record["path"])).resolve() != artifact or artifact_record[
        "sha256"
    ] != split_builder._sha256_file(artifact):
        raise ValueError(f"existing {split} artifact bytes are invalid")

    builder_path = Path(split_builder.__file__).resolve()
    expected_builder = {
        "path": str(builder_path),
        "sha256": split_builder._sha256_file(builder_path),
    }
    if sidecar.get("builder") != expected_builder:
        raise ValueError(f"existing {split} builder provenance is stale")
    return sidecar


def _report(
    *,
    status: str,
    output_root: Path,
    paths: Mapping[str, tuple[Path, Path]],
    sidecars: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "status": status,
        "output_root": str(output_root),
        "artifacts": {
            split: {
                "artifact": str(paths[split][0]),
                "artifact_sha256": sidecars[split]["artifact"]["sha256"],
                "sidecar": str(paths[split][1]),
                "sidecar_sha256": split_builder._sha256_file(paths[split][1]),
            }
            for split in split_builder.SPLITS
        },
    }


def build_embedding_bundle(
    *,
    vocabulary: Path,
    vocabulary_manifest: Path,
    snapshot: Path,
    output_root: Path,
    model_id: str = split_builder.MODEL_ID,
    revision: str = split_builder.MODEL_REVISION,
    batch_size: int = 32,
    batch_encoder: split_builder.BatchEncoder | None = None,
    _test_vocabulary_contract: Mapping[str, Any] | None = None,
    _test_snapshot_files_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build or byte-validate all three split-v1 artifacts exactly once."""

    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    vocabulary = Path(vocabulary).resolve()
    vocabulary_manifest = Path(vocabulary_manifest).resolve()
    snapshot = Path(snapshot).resolve()
    output_root = Path(output_root).resolve()
    if output_root == snapshot or snapshot in output_root.parents:
        raise ValueError("output_root must not be inside the frozen snapshot")
    paths = bundle_output_paths(output_root)
    state = _classify_output_root(output_root, paths)

    vocabulary_contracts = {
        split: split_builder._validate_vocabulary(
            vocabulary,
            vocabulary_manifest,
            split,
            frozen_contract=(
                split_builder.FROZEN_VOCABULARY_CONTRACT
                if _test_vocabulary_contract is None
                else _test_vocabulary_contract
            ),
        )
        for split in split_builder.SPLITS
    }
    # This is intentionally the sole snapshot validation/hash call for both
    # fresh builds and byte-valid idempotent skips.
    encoder_contract = split_builder._validate_snapshot(
        snapshot,
        model_id=str(model_id),
        revision=str(revision),
        expected_files_sha256=(
            split_builder.FROZEN_SNAPSHOT_FILES_SHA256
            if _test_snapshot_files_sha256 is None
            else _test_snapshot_files_sha256
        ),
    )

    if state == "complete":
        try:
            sidecars = {
                split: _validate_existing_split(
                    split=split,
                    artifact=paths[split][0],
                    sidecar_path=paths[split][1],
                    vocabulary_contract=vocabulary_contracts[split],
                    encoder_contract=encoder_contract,
                )
                for split in split_builder.SPLITS
            }
        except Exception as error:
            raise ValueError(
                "output_root contains a complete-looking but invalid/mixed bundle; "
                "refusing overwrite"
            ) from error
        return _report(
            status="byte_valid_skip",
            output_root=output_root,
            paths=paths,
            sidecars=sidecars,
        )

    output_root.mkdir(parents=True, exist_ok=True)
    encoder = batch_encoder
    if encoder is None:
        encoder = split_builder._LocalSiglip2BatchEncoder(snapshot)
    sidecars: dict[str, Mapping[str, Any]] = {}
    for split in split_builder.SPLITS:
        artifact, sidecar_path = paths[split]
        sidecars[split] = split_builder._build_embedding_artifact_from_contracts(
            vocabulary_contract=vocabulary_contracts[split],
            encoder_contract=encoder_contract,
            snapshot=snapshot,
            output=artifact,
            sidecar_output=sidecar_path,
            batch_size=batch_size,
            encoder=encoder,
        )
    return _report(
        status="built",
        output_root=output_root,
        paths=paths,
        sidecars=sidecars,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--vocabulary-manifest", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-id", default=split_builder.MODEL_ID)
    parser.add_argument("--revision", default=split_builder.MODEL_REVISION)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    report = build_embedding_bundle(
        vocabulary=args.vocabulary,
        vocabulary_manifest=args.vocabulary_manifest,
        snapshot=args.snapshot,
        output_root=args.output_root,
        model_id=args.model_id,
        revision=args.revision,
        batch_size=args.batch_size,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
