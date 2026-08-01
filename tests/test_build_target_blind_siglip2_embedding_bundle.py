from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import pytest
import torch

from radio_gs.scripts import (
    build_target_blind_siglip2_embedding_artifact as split_builder,
)
from radio_gs.scripts.build_target_blind_siglip2_embedding_bundle import (
    _PAYLOAD_KEYS,
    ARTIFACT_NAME,
    SIDECAR_NAME,
    build_embedding_bundle,
    bundle_output_paths,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split_sha256(records: list[dict[str, str]], split: str) -> str:
    lines = "".join(
        f"{record['synset']}\t{record['query']}\n"
        for record in records
        if record["split"] == split
    )
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


_TEST_SOURCE_SHA256 = {"mock_imagenet_source.txt": "b" * 64}


def _test_contracts(vocabulary: Path, snapshot: Path) -> dict:
    records = json.loads(vocabulary.read_text(encoding="utf-8"))["records"]
    split_counts = {
        split: sum(record["split"] == split for record in records)
        for split in split_builder.SPLITS
    }
    index = json.loads(
        (snapshot / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    snapshot_names = {
        "config.json",
        "model.safetensors.index.json",
        *split_builder.TOKENIZER_FILES,
        *split_builder.SNAPSHOT_AUXILIARY_FILES,
        *(str(value) for value in index["weight_map"].values()),
    }
    return {
        "_test_vocabulary_contract": {
            "canonical_vocabulary_sha256": _sha256(vocabulary),
            "counts": {
                "source_synsets": len(records),
                "deduplicated_queries": len(records),
                **split_counts,
            },
            "source_sha256": _TEST_SOURCE_SHA256,
            "split_sha256": {
                split: _split_sha256(records, split)
                for split in split_builder.SPLITS
            },
        },
        "_test_snapshot_files_sha256": {
            name: _sha256(snapshot / name) for name in snapshot_names
        },
    }


def _write_inputs(root: Path) -> tuple[Path, Path, Path]:
    records = [
        {"synset": "n00000001", "query": "alpha object", "split": "fit"},
        {"synset": "n00000002", "query": "beta object", "split": "dev"},
        {"synset": "n00000003", "query": "gamma object", "split": "audit"},
    ]
    vocabulary_payload = {
        "schema_version": 1,
        "artifact_type": split_builder.VOCABULARY_ARTIFACT_TYPE,
        "algorithm_version": split_builder.VOCABULARY_ALGORITHM_VERSION,
        "prompt_templates": ["{query}"],
        "benchmark_vocabulary_opened": False,
        "records": records,
    }
    vocabulary = root / "vocabulary.json"
    vocabulary.write_text(
        json.dumps(
            vocabulary_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    vocabulary_manifest = root / "vocabulary.manifest.json"
    vocabulary_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": (split_builder.VOCABULARY_MANIFEST_ARTIFACT_TYPE),
                "algorithm_version": split_builder.VOCABULARY_ALGORITHM_VERSION,
                "benchmark_vocabulary_opened": False,
                "counts": {
                    "source_synsets": len(records),
                    "deduplicated_queries": len(records),
                    **{
                        split: sum(record["split"] == split for record in records)
                        for split in split_builder.SPLITS
                    },
                },
                "sources": {
                    name: {"path": f"/mock/{name}", "sha256": digest}
                    for name, digest in _TEST_SOURCE_SHA256.items()
                },
                "canonical_json": {
                    "path": str(vocabulary),
                    "sha256": _sha256(vocabulary),
                },
                "split_synset_tab_query_lf_sha256": {
                    split: _split_sha256(records, split)
                    for split in split_builder.SPLITS
                },
            }
        ),
        encoding="utf-8",
    )

    snapshot = root / "snapshots" / split_builder.MODEL_REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps(
            {
                "model_type": "siglip",
                "text_config": {
                    "hidden_size": 8,
                    "projection_size": split_builder.OUTPUT_DIMENSION,
                },
            }
        ),
        encoding="utf-8",
    )
    for name in split_builder.TOKENIZER_FILES:
        (snapshot / name).write_bytes(name.encode("utf-8"))
    for name in split_builder.SNAPSHOT_AUXILIARY_FILES:
        (snapshot / name).write_bytes(name.encode("utf-8"))
    shard = "model-00001-of-00001.safetensors"
    (snapshot / shard).write_bytes(b"mock-shard")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "text_model.head.weight": shard,
                    "text_model.head.bias": shard,
                    "text_model.embeddings.token_embedding.weight": shard,
                }
            }
        ),
        encoding="utf-8",
    )
    return vocabulary, vocabulary_manifest, snapshot


def _unit_embeddings(rows: int) -> torch.Tensor:
    value = torch.zeros(rows, split_builder.OUTPUT_DIMENSION, dtype=torch.float32)
    value[:, 0] = 1.0
    return value


def test_bundle_loads_one_encoder_and_hashes_snapshot_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vocabulary, vocabulary_manifest, snapshot = _write_inputs(tmp_path)
    output_root = tmp_path / "bundle"
    validation_calls = 0
    encoder_initializations = 0
    encoded_queries: list[list[str]] = []
    original_validate = split_builder._validate_snapshot

    def validate_once(*args: object, **kwargs: object) -> dict:
        nonlocal validation_calls
        validation_calls += 1
        return original_validate(*args, **kwargs)

    class MockEncoder:
        def __init__(self, path: Path) -> None:
            nonlocal encoder_initializations
            encoder_initializations += 1
            assert path == snapshot.resolve()

        def __call__(self, queries: Sequence[str], path: Path) -> torch.Tensor:
            assert path == snapshot.resolve()
            encoded_queries.append(list(queries))
            return _unit_embeddings(len(queries))

    monkeypatch.setattr(split_builder, "_validate_snapshot", validate_once)
    monkeypatch.setattr(split_builder, "_LocalSiglip2BatchEncoder", MockEncoder)
    report = build_embedding_bundle(
        vocabulary=vocabulary,
        vocabulary_manifest=vocabulary_manifest,
        snapshot=snapshot,
        output_root=output_root,
        batch_size=2,
        **_test_contracts(vocabulary, snapshot),
    )

    assert report["status"] == "built"
    assert validation_calls == 1
    assert encoder_initializations == 1
    assert encoded_queries == [
        ["alpha object"],
        ["beta object"],
        ["gamma object"],
    ]
    paths = bundle_output_paths(output_root)
    assert {path.name for pair in paths.values() for path in pair} == {
        ARTIFACT_NAME.format(split=split) for split in split_builder.SPLITS
    } | {SIDECAR_NAME.format(split=split) for split in split_builder.SPLITS}
    builder_path = Path(split_builder.__file__).resolve()
    for split, (artifact, sidecar_path) in paths.items():
        payload = torch.load(artifact, map_location="cpu")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert set(payload) == _PAYLOAD_KEYS
        assert payload["split"] == split
        assert sidecar["split"] == split
        assert sidecar["builder"] == {
            "path": str(builder_path),
            "sha256": _sha256(builder_path),
        }
        assert sidecar["artifact"] == {
            "path": str(artifact),
            "sha256": _sha256(artifact),
        }
    assert not list(output_root.glob("*.tmp"))
    assert not torch.cuda.is_initialized()


def test_complete_byte_valid_bundle_is_idempotently_skipped(tmp_path: Path) -> None:
    vocabulary, vocabulary_manifest, snapshot = _write_inputs(tmp_path)
    output_root = tmp_path / "bundle"
    calls = 0

    def encoder(queries: Sequence[str], path: Path) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return _unit_embeddings(len(queries))

    first = build_embedding_bundle(
        vocabulary=vocabulary,
        vocabulary_manifest=vocabulary_manifest,
        snapshot=snapshot,
        output_root=output_root,
        batch_encoder=encoder,
        **_test_contracts(vocabulary, snapshot),
    )
    before = {path.name: path.read_bytes() for path in output_root.iterdir()}

    def forbidden_encoder(queries: Sequence[str], path: Path) -> torch.Tensor:
        raise AssertionError("a byte-valid bundle must not reload or call the encoder")

    second = build_embedding_bundle(
        vocabulary=vocabulary,
        vocabulary_manifest=vocabulary_manifest,
        snapshot=snapshot,
        output_root=output_root,
        batch_encoder=forbidden_encoder,
        **_test_contracts(vocabulary, snapshot),
    )
    after = {path.name: path.read_bytes() for path in output_root.iterdir()}

    assert first["status"] == "built"
    assert second["status"] == "byte_valid_skip"
    assert calls == len(split_builder.SPLITS)
    assert after == before


@pytest.mark.parametrize("existing_name", ["unexpected.txt", "partial"])
def test_bundle_rejects_partial_or_mixed_existing_outputs_before_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_name: str,
) -> None:
    vocabulary, vocabulary_manifest, snapshot = _write_inputs(tmp_path)
    output_root = tmp_path / "bundle"
    output_root.mkdir()
    if existing_name == "partial":
        path = bundle_output_paths(output_root)["fit"][0]
    else:
        path = output_root / existing_name
    path.write_bytes(b"do-not-overwrite")

    def forbidden_validation(*args: object, **kwargs: object) -> dict:
        raise AssertionError("partial output rejection should happen before hashing")

    monkeypatch.setattr(split_builder, "_validate_snapshot", forbidden_validation)
    with pytest.raises(FileExistsError, match="partial or mixed"):
        build_embedding_bundle(
            vocabulary=vocabulary,
            vocabulary_manifest=vocabulary_manifest,
            snapshot=snapshot,
            output_root=output_root,
            batch_encoder=lambda queries, path: _unit_embeddings(len(queries)),
            **_test_contracts(vocabulary, snapshot),
        )
    assert path.read_bytes() == b"do-not-overwrite"


def test_bundle_rejects_corrupt_complete_bundle_without_overwrite(
    tmp_path: Path,
) -> None:
    vocabulary, vocabulary_manifest, snapshot = _write_inputs(tmp_path)
    output_root = tmp_path / "bundle"
    build_embedding_bundle(
        vocabulary=vocabulary,
        vocabulary_manifest=vocabulary_manifest,
        snapshot=snapshot,
        output_root=output_root,
        batch_encoder=lambda queries, path: _unit_embeddings(len(queries)),
        **_test_contracts(vocabulary, snapshot),
    )
    artifact = bundle_output_paths(output_root)["dev"][0]
    corrupt = artifact.read_bytes() + b"corrupt"
    artifact.write_bytes(corrupt)

    with pytest.raises(ValueError, match="invalid/mixed bundle"):
        build_embedding_bundle(
            vocabulary=vocabulary,
            vocabulary_manifest=vocabulary_manifest,
            snapshot=snapshot,
            output_root=output_root,
            batch_encoder=lambda queries, path: pytest.fail(
                "corrupt existing output must never be regenerated"
            ),
            **_test_contracts(vocabulary, snapshot),
        )
    assert artifact.read_bytes() == corrupt
