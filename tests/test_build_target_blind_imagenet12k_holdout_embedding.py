from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from radio_gs.scripts import build_target_blind_siglip2_embedding_artifact as encoder_v1
from radio_gs.scripts.build_target_blind_imagenet12k_holdout_bank import (
    ALGORITHM_VERSION as VOCABULARY_ALGORITHM_VERSION,
    ARTIFACT_TYPE as VOCABULARY_ARTIFACT_TYPE,
    MANIFEST_ARTIFACT_TYPE as VOCABULARY_MANIFEST_ARTIFACT_TYPE,
)
from radio_gs.scripts.build_target_blind_imagenet12k_holdout_embedding import (
    ALGORITHM_VERSION,
    build_embedding,
)
from radio_gs.utils.immutable_artifacts import canonical_json_bytes


def _fake_snapshot(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    snapshot = tmp_path / encoder_v1.MODEL_REVISION
    snapshot.mkdir()
    files = {
        "config.json": json.dumps(
            {
                "model_type": "siglip",
                "text_config": {
                    "hidden_size": 8,
                    "projection_size": encoder_v1.OUTPUT_DIMENSION,
                },
            }
        ).encode(),
        "model.safetensors.index.json": json.dumps(
            {
                "weight_map": {
                    "text_model.head.weight": "model-00001-of-00002.safetensors",
                    "text_model.head.bias": "model-00001-of-00002.safetensors",
                    "text_model.embeddings.token_embedding.weight": (
                        "model-00002-of-00002.safetensors"
                    ),
                }
            }
        ).encode(),
        "tokenizer.json": b"{}",
        "tokenizer.model": b"tokenizer",
        "tokenizer_config.json": b"{}",
        "special_tokens_map.json": b"{}",
        "preprocessor_config.json": b"{}",
        "model-00001-of-00002.safetensors": b"one",
        "model-00002-of-00002.safetensors": b"two",
    }
    for name, value in files.items():
        (snapshot / name).write_bytes(value)
    return snapshot, {
        name: hashlib.sha256(value).hexdigest() for name, value in files.items()
    }


def test_build_dev_embedding_from_bound_holdout(tmp_path: Path) -> None:
    records = [
        {"synset": "n00000001", "query": "alpha object", "split": "dev"},
        {"synset": "n00000002", "query": "beta object", "split": "dev"},
    ]
    record_bytes = "".join(
        f"{row['synset']}\t{row['query']}\n" for row in records
    ).encode()
    record_sha = hashlib.sha256(record_bytes).hexdigest()
    vocabulary = tmp_path / "dev.json"
    vocabulary.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": 1,
                "artifact_type": VOCABULARY_ARTIFACT_TYPE,
                "algorithm_version": VOCABULARY_ALGORITHM_VERSION,
                "split": "dev",
                "prompt_templates": ["{query}"],
                "benchmark_vocabulary_opened": False,
                "uses_benchmark_vocabulary_for_construction": False,
                "records": records,
            }
        )
        + b"\n"
    )
    vocabulary_sha = hashlib.sha256(vocabulary.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": 1,
                "artifact_type": VOCABULARY_MANIFEST_ARTIFACT_TYPE,
                "algorithm_version": VOCABULARY_ALGORITHM_VERSION,
                "benchmark_vocabulary_opened": False,
                "uses_benchmark_vocabulary_for_construction": False,
                "synset_tab_query_lf_sha256": {"dev": record_sha},
                "artifacts": {"dev": {"path": str(vocabulary), "sha256": vocabulary_sha}},
            }
        )
        + b"\n"
    )
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    snapshot, snapshot_hashes = _fake_snapshot(tmp_path)

    def encoder(queries, _snapshot):
        values = torch.arange(
            len(queries) * encoder_v1.OUTPUT_DIMENSION, dtype=torch.float32
        ).reshape(len(queries), encoder_v1.OUTPUT_DIMENSION)
        return F.normalize(values + 1.0, dim=-1)

    output = tmp_path / "embeddings.pt"
    sidecar = tmp_path / "embeddings.json"
    result = build_embedding(
        vocabulary=vocabulary,
        vocabulary_manifest=manifest,
        split="dev",
        snapshot=snapshot,
        output=output,
        sidecar_output=sidecar,
        batch_encoder=encoder,
        _test_holdout_contract={
            "manifest_sha256": manifest_sha,
            "splits": {
                "dev": {
                    "vocabulary_sha256": vocabulary_sha,
                    "records": 2,
                    "record_sha256": record_sha,
                }
            },
        },
        _test_snapshot_files_sha256=snapshot_hashes,
    )
    payload = torch.load(output, map_location="cpu")
    assert payload["algorithm_version"] == ALGORITHM_VERSION
    assert payload["split"] == "dev"
    assert payload["embeddings"].shape == (2, encoder_v1.OUTPUT_DIMENSION)
    assert result["artifact_type"] == "target_blind_text_embedding_cache_manifest"
    assert json.loads(sidecar.read_text()) == result
    with pytest.raises(FileExistsError, match="already exists"):
        build_embedding(
            vocabulary=vocabulary,
            vocabulary_manifest=manifest,
            split="dev",
            snapshot=snapshot,
            output=output,
            sidecar_output=tmp_path / "second.json",
            batch_encoder=encoder,
            _test_holdout_contract={
                "manifest_sha256": manifest_sha,
                "splits": {
                    "dev": {
                        "vocabulary_sha256": vocabulary_sha,
                        "records": 2,
                        "record_sha256": record_sha,
                    }
                },
            },
            _test_snapshot_files_sha256=snapshot_hashes,
        )
