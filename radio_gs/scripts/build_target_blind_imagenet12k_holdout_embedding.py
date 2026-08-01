#!/usr/bin/env python3
"""Encode one frozen ImageNet12K-minus-ImageNet1K holdout split on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.scripts import build_target_blind_siglip2_embedding_artifact as encoder_v1
from radio_gs.scripts.build_target_blind_imagenet12k_holdout_bank import (
    ALGORITHM_VERSION as VOCABULARY_ALGORITHM_VERSION,
    ARTIFACT_TYPE as VOCABULARY_ARTIFACT_TYPE,
    MANIFEST_ARTIFACT_TYPE as VOCABULARY_MANIFEST_ARTIFACT_TYPE,
    normalize_alias,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_bytes,
    load_json_object,
    sha256_file,
    write_bytes_noclobber,
    write_torch_noclobber,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "target_blind_text_embedding_cache"
MANIFEST_ARTIFACT_TYPE = "target_blind_text_embedding_cache_manifest"
ALGORITHM_VERSION = "siglip2-target-blind-imagenet12k-minus1k-holdout-v1"
SPLITS = ("dev", "audit")
BatchEncoder = Callable[[Sequence[str], Path], torch.Tensor]
FROZEN_HOLDOUT_CONTRACT: Mapping[str, Any] = {
    "manifest_sha256": "7c02b9dd4b1b861dbdef0f2bb91e7944a338cda19a2ddecf09280b2c3d39e549",
    "splits": {
        "dev": {
            "vocabulary_sha256": "09564b880d9261431fd3d6036be9227c1df67198087e6c00c7ff7671a9ea8d6e",
            "records": 101,
            "record_sha256": "c4a111be6e171d9934767cccc2fc2aae4fd1e46f21839e8dd5d69bbcdb243b3b",
        },
        "audit": {
            "vocabulary_sha256": "57edab4c74c0ed5e5a025bfe8df204bdaad8ddcb8b99c408bc98c8ddc263823d",
            "records": 90,
            "record_sha256": "008f35ac963f0af7e378abe18841e72eb16c06c7955ec0872e363b83937c7d94",
        },
    },
}


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _record_bytes(records: Sequence[Mapping[str, str]]) -> bytes:
    return "".join(
        f"{record['synset']}\t{record['query']}\n" for record in records
    ).encode("utf-8")


def _validate_vocabulary(
    *,
    vocabulary: Path,
    vocabulary_manifest: Path,
    split: str,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    payload, payload_sha, _ = load_json_object(vocabulary, label="holdout vocabulary")
    manifest, manifest_sha, _ = load_json_object(
        vocabulary_manifest, label="holdout vocabulary manifest"
    )
    split_contract = contract.get("splits", {}).get(split)
    if not isinstance(split_contract, Mapping):
        raise ValueError(f"holdout contract does not bind split {split}")
    if manifest_sha != contract.get("manifest_sha256"):
        raise ValueError("holdout manifest does not match frozen SHA256")
    if payload_sha != split_contract.get("vocabulary_sha256"):
        raise ValueError("holdout vocabulary does not match frozen SHA256")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("artifact_type") != VOCABULARY_ARTIFACT_TYPE
        or payload.get("algorithm_version") != VOCABULARY_ALGORITHM_VERSION
        or payload.get("split") != split
        or payload.get("prompt_templates") != ["{query}"]
        or payload.get("benchmark_vocabulary_opened") is not False
        or payload.get("uses_benchmark_vocabulary_for_construction") is not False
    ):
        raise ValueError("invalid frozen holdout vocabulary schema")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("artifact_type") != VOCABULARY_MANIFEST_ARTIFACT_TYPE
        or manifest.get("algorithm_version") != VOCABULARY_ALGORITHM_VERSION
        or manifest.get("benchmark_vocabulary_opened") is not False
        or manifest.get("uses_benchmark_vocabulary_for_construction") is not False
    ):
        raise ValueError("invalid frozen holdout manifest schema")
    manifest_artifact = manifest.get("artifacts", {}).get(split)
    if (
        not isinstance(manifest_artifact, Mapping)
        or manifest_artifact.get("sha256") != payload_sha
        or manifest.get("synset_tab_query_lf_sha256", {}).get(split)
        != split_contract.get("record_sha256")
    ):
        raise ValueError("holdout vocabulary is not bound by its manifest")

    raw_records = payload.get("records")
    if (
        not isinstance(raw_records, list)
        or len(raw_records) != int(split_contract.get("records", -1))
    ):
        raise ValueError("holdout vocabulary record count differs")
    records: list[dict[str, str]] = []
    for raw in raw_records:
        if not isinstance(raw, Mapping) or set(raw) != {"synset", "query", "split"}:
            raise ValueError("holdout vocabulary record schema differs")
        synset, query = str(raw["synset"]), str(raw["query"])
        if raw["split"] != split or normalize_alias(query) != query:
            raise ValueError("holdout vocabulary record is not canonical")
        records.append({"synset": synset, "query": query, "split": split})
    if len({row["synset"] for row in records}) != len(records):
        raise ValueError("holdout vocabulary synsets are not unique")
    if len({row["query"] for row in records}) != len(records):
        raise ValueError("holdout vocabulary queries are not unique")
    observed_record_sha = hashlib.sha256(_record_bytes(records)).hexdigest()
    if observed_record_sha != split_contract.get("record_sha256"):
        raise ValueError("holdout vocabulary record SHA256 differs")
    return {
        "records": records,
        "split_sha256": observed_record_sha,
        "ordered_records_sha256": _canonical_json_sha256(records),
        "vocabulary_path": vocabulary.resolve(),
        "vocabulary_sha256": payload_sha,
        "vocabulary_manifest_path": vocabulary_manifest.resolve(),
        "vocabulary_manifest_sha256": manifest_sha,
    }


def _validate_embedding_batch(value: torch.Tensor, rows: int) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if (
        tensor.device.type != "cpu"
        or tensor.dtype != torch.float32
        or tensor.shape != (rows, encoder_v1.OUTPUT_DIMENSION)
    ):
        raise ValueError("SigLIP2 holdout embeddings have invalid device/dtype/shape")
    tensor = tensor.detach().contiguous()
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("SigLIP2 holdout embeddings contain non-finite values")
    norms = torch.linalg.vector_norm(tensor, dim=-1)
    if not bool(torch.allclose(norms, torch.ones_like(norms), atol=5e-5, rtol=5e-5)):
        raise ValueError("SigLIP2 holdout embeddings are not L2-normalized")
    return tensor


def build_embedding(
    *,
    vocabulary: Path,
    vocabulary_manifest: Path,
    split: str,
    snapshot: Path,
    output: Path,
    sidecar_output: Path,
    batch_size: int = 32,
    batch_encoder: BatchEncoder | None = None,
    _test_holdout_contract: Mapping[str, Any] | None = None,
    _test_snapshot_files_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    vocabulary = Path(vocabulary).resolve(strict=True)
    vocabulary_manifest = Path(vocabulary_manifest).resolve(strict=True)
    snapshot = Path(snapshot).resolve(strict=True)
    output = Path(output).resolve()
    sidecar_output = Path(sidecar_output).resolve()
    if output == sidecar_output or output in {vocabulary, vocabulary_manifest}:
        raise ValueError("embedding output paths overlap inputs")
    contract = _validate_vocabulary(
        vocabulary=vocabulary,
        vocabulary_manifest=vocabulary_manifest,
        split=split,
        contract=(
            FROZEN_HOLDOUT_CONTRACT
            if _test_holdout_contract is None
            else _test_holdout_contract
        ),
    )
    encoder_contract = encoder_v1._validate_snapshot(
        snapshot,
        model_id=encoder_v1.MODEL_ID,
        revision=encoder_v1.MODEL_REVISION,
        expected_files_sha256=(
            encoder_v1.FROZEN_SNAPSHOT_FILES_SHA256
            if _test_snapshot_files_sha256 is None
            else _test_snapshot_files_sha256
        ),
    )
    encoder = batch_encoder or encoder_v1._LocalSiglip2BatchEncoder(snapshot)
    records = contract["records"]
    queries = [row["query"] for row in records]
    batches = []
    for start in range(0, len(queries), batch_size):
        values = encoder(queries[start : start + batch_size], snapshot)
        batches.append(_validate_embedding_batch(values, len(queries[start : start + batch_size])))
    embeddings = torch.cat(batches).to(torch.float32).contiguous()
    semantic_sha = encoder_v1.embedding_semantic_sha256(embeddings)
    typed_sha = tensor_sha256(embeddings)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "algorithm_version": ALGORITHM_VERSION,
        "benchmark_vocabulary_opened": False,
        "uses_benchmark_vocabulary_for_construction": False,
        "split": split,
        "split_synset_tab_query_lf_sha256": contract["split_sha256"],
        "prompt_templates": ["{query}"],
        "text_canonicalization": encoder_v1.TEXT_CANONICALIZATION,
        "records": records,
        "queries": queries,
        "synsets": [row["synset"] for row in records],
        "ordered_records_sha256": contract["ordered_records_sha256"],
        "vocabulary_path": str(vocabulary),
        "vocabulary_sha256": contract["vocabulary_sha256"],
        "vocabulary_manifest_path": str(vocabulary_manifest),
        "vocabulary_manifest_sha256": contract["vocabulary_manifest_sha256"],
        "embeddings": embeddings,
        "embedding_semantic_sha256": semantic_sha,
        "embedding_tensor_sha256": typed_sha,
        "text_encoder": encoder_contract,
    }
    written = write_torch_noclobber(output, payload)
    artifact_sha = sha256_file(written)
    builder_path = Path(__file__).resolve(strict=True)
    sidecar = {
        key: payload[key]
        for key in (
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
        )
    }
    sidecar.update(
        {
            "vocabulary": {
                "path": str(vocabulary),
                "sha256": contract["vocabulary_sha256"],
                "manifest_path": str(vocabulary_manifest),
                "manifest_sha256": contract["vocabulary_manifest_sha256"],
            },
            "text_encoder": encoder_contract,
            "embedding": {
                "shape": list(embeddings.shape),
                "dtype": "float32",
                "byte_order": "little_endian",
                "normalization": "l2",
                "semantic_sha256": semantic_sha,
                "tensor_sha256": typed_sha,
            },
            "artifact": {"path": str(written), "sha256": artifact_sha},
            "builder": {"path": str(builder_path), "sha256": sha256_file(builder_path)},
        }
    )
    write_bytes_noclobber(sidecar_output, canonical_json_bytes(sidecar) + b"\n")
    return sidecar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--vocabulary-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sidecar-output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    sidecar = build_embedding(
        vocabulary=args.vocabulary,
        vocabulary_manifest=args.vocabulary_manifest,
        split=args.split,
        snapshot=args.snapshot,
        output=args.output,
        sidecar_output=args.sidecar_output,
        batch_size=args.batch_size,
    )
    print(
        json.dumps(
            {
                "split": sidecar["split"],
                "records": len(sidecar["records"]),
                "artifact": sidecar["artifact"],
                "embedding": sidecar["embedding"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
