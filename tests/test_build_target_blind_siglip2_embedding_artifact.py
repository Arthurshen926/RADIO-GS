from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Sequence

import pytest
import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.scripts.build_target_blind_siglip2_embedding_artifact import (
    ALGORITHM_VERSION,
    ARTIFACT_TYPE,
    MODEL_ID,
    MODEL_REVISION,
    OUTPUT_DIMENSION,
    _load_local_siglip2_text_tower,
    _validate_snapshot,
    build_embedding_artifact,
    embedding_semantic_sha256,
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split_sha256(records: list[dict[str, str]], split: str) -> str:
    lines = "".join(
        f"{record['synset']}\t{record['query']}\n"
        for record in records
        if record["split"] == split
    )
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


_TEST_SOURCE_SHA256 = {"mock_imagenet_source.txt": "a" * 64}


def _test_vocabulary_contract(
    vocabulary: Path,
    records: list[dict[str, str]],
) -> dict:
    split_counts = {
        split: sum(record["split"] == split for record in records)
        for split in ("fit", "dev", "audit")
    }
    return {
        "canonical_vocabulary_sha256": _file_sha256(vocabulary),
        "counts": {
            "source_synsets": len(records),
            "deduplicated_queries": len(records),
            **split_counts,
        },
        "source_sha256": _TEST_SOURCE_SHA256,
        "split_sha256": {
            split: _split_sha256(records, split)
            for split in ("fit", "dev", "audit")
        },
    }


def _test_snapshot_files_sha256(snapshot: Path) -> dict[str, str]:
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    names = {
        "config.json",
        "model.safetensors.index.json",
        *(
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ),
        *(str(value) for value in index["weight_map"].values()),
        "preprocessor_config.json",
    }
    return {name: _file_sha256(snapshot / name) for name in names}


def _write_vocabulary(
    root: Path,
    *,
    vocabulary_benchmark_opened: bool = False,
    manifest_benchmark_opened: bool = False,
) -> tuple[Path, Path, list[dict[str, str]]]:
    records = [
        {"synset": "n00000001", "query": "alpha object", "split": "fit"},
        {"synset": "n00000002", "query": "beta object", "split": "dev"},
        {"synset": "n00000003", "query": "gamma object", "split": "dev"},
        {"synset": "n00000004", "query": "delta object", "split": "dev"},
        {"synset": "n00000005", "query": "epsilon object", "split": "audit"},
    ]
    vocabulary = {
        "schema_version": 1,
        "artifact_type": "target_blind_imagenet1k_primary_text_bank",
        "algorithm_version": "imagenet1k-primary-v1",
        "prompt_templates": ["{query}"],
        "benchmark_vocabulary_opened": vocabulary_benchmark_opened,
        "records": records,
    }
    vocabulary_path = root / "vocabulary.json"
    vocabulary_path.write_text(
        json.dumps(
            vocabulary,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "artifact_type": (
            "target_blind_imagenet1k_primary_text_bank_manifest"
        ),
        "algorithm_version": "imagenet1k-primary-v1",
        "benchmark_vocabulary_opened": manifest_benchmark_opened,
        "counts": {
            "source_synsets": len(records),
            "deduplicated_queries": len(records),
            **{
                split: sum(record["split"] == split for record in records)
                for split in ("fit", "dev", "audit")
            },
        },
        "sources": {
            name: {"path": f"/mock/{name}", "sha256": digest}
            for name, digest in _TEST_SOURCE_SHA256.items()
        },
        "canonical_json": {
            "path": str(vocabulary_path),
            "sha256": _file_sha256(vocabulary_path),
        },
        "split_synset_tab_query_lf_sha256": {
            split: _split_sha256(records, split)
            for split in ("fit", "dev", "audit")
        },
    }
    manifest_path = root / "vocabulary.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return vocabulary_path, manifest_path, records


def _write_snapshot(root: Path, *, projection_size: int = OUTPUT_DIMENSION) -> Path:
    snapshot = root / "snapshots" / MODEL_REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps(
            {
                "model_type": "siglip",
                "text_config": {
                    "hidden_size": 8,
                    "projection_size": projection_size,
                },
            }
        ),
        encoding="utf-8",
    )
    for name, content in {
        "tokenizer.json": b"tokenizer-json",
        "tokenizer.model": b"tokenizer-model",
        "tokenizer_config.json": b"{}",
        "special_tokens_map.json": b"{}",
        "preprocessor_config.json": b"{}",
    }.items():
        (snapshot / name).write_bytes(content)
    shard_name = "model-00001-of-00001.safetensors"
    (snapshot / shard_name).write_bytes(b"mock-local-weight-shard")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "text_model.head.weight": shard_name,
                    "text_model.head.bias": shard_name,
                    "text_model.embeddings.token_embedding.weight": shard_name,
                }
            }
        ),
        encoding="utf-8",
    )
    return snapshot


def _replace_snapshot_files_with_hf_blob_links(
    snapshot: Path,
) -> tuple[dict[str, str], dict[str, Path]]:
    expected_files_sha256 = _test_snapshot_files_sha256(snapshot)
    blobs = snapshot.parent.parent / "blobs"
    blobs.mkdir()
    targets: dict[str, Path] = {}
    for name, digest in expected_files_sha256.items():
        source = snapshot / name
        content = source.read_bytes()
        target = blobs / digest
        if target.exists():
            assert target.read_bytes() == content
        else:
            target.write_bytes(content)
        source.unlink()
        source.symlink_to(Path("..") / ".." / "blobs" / digest)
        targets[name] = target
    return expected_files_sha256, targets


def _unit_embeddings(rows: int) -> torch.Tensor:
    values = torch.zeros(rows, OUTPUT_DIMENSION, dtype=torch.float32)
    values[:, 0] = 1.0
    return values


def _write_tiny_text_runtime_snapshot(
    root: Path,
) -> tuple[Path, torch.nn.Module, object, dict[str, torch.Tensor]]:
    safetensors_torch = pytest.importorskip("safetensors.torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.SiglipConfig(
        text_config={
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "max_position_embeddings": 8,
            "projection_size": OUTPUT_DIMENSION,
        },
        vision_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 4,
            "patch_size": 2,
        },
    )
    runtime_config = transformers.SiglipConfig.from_dict(config.to_dict())
    reference = transformers.SiglipModel(config)
    reference.text_model.head = torch.nn.Linear(8, OUTPUT_DIMENSION)
    reference.eval()
    text_state = {
        key: value.detach().cpu().contiguous()
        for key, value in reference.state_dict().items()
        if key.startswith("text_model.")
    }
    snapshot = root / "tiny-runtime-snapshot"
    snapshot.mkdir()
    runtime_config.to_json_file(snapshot / "config.json")
    text_shard = "text-only.safetensors"
    safetensors_torch.save_file(text_state, snapshot / text_shard)
    weight_map = {key: text_shard for key in text_state}
    # This deliberately missing file makes the test fail if the loader walks
    # a visual shard instead of selecting only indexed text parameters.
    weight_map["vision_model.deliberately_unread.weight"] = (
        "vision-must-not-be-opened.safetensors"
    )
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )
    return snapshot, reference, runtime_config, text_state


def _build(
    tmp_path: Path,
    encoder: Callable[[Sequence[str], Path], torch.Tensor],
    **overrides: object,
) -> tuple[dict, Path, Path, Path, Path]:
    vocabulary, vocabulary_manifest, records = _write_vocabulary(tmp_path)
    snapshot = _write_snapshot(tmp_path)
    output = tmp_path / "dev_embeddings.pt"
    sidecar = tmp_path / "dev_embeddings.manifest.json"
    arguments = {
        "vocabulary": vocabulary,
        "vocabulary_manifest": vocabulary_manifest,
        "split": "dev",
        "snapshot": snapshot,
        "output": output,
        "sidecar_output": sidecar,
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "batch_size": 2,
        "batch_encoder": encoder,
        "_test_vocabulary_contract": _test_vocabulary_contract(
            vocabulary, records
        ),
        "_test_snapshot_files_sha256": _test_snapshot_files_sha256(snapshot),
        **overrides,
    }
    result = build_embedding_artifact(**arguments)
    return result, output, sidecar, vocabulary, vocabulary_manifest


def test_builder_writes_frozen_per_split_artifact_in_batches(tmp_path: Path) -> None:
    calls: list[tuple[list[str], Path]] = []

    def encoder(queries: Sequence[str], snapshot: Path) -> torch.Tensor:
        calls.append((list(queries), snapshot))
        return _unit_embeddings(len(queries))

    sidecar, output, sidecar_path, vocabulary, manifest = _build(
        tmp_path,
        encoder,
    )
    payload = torch.load(output, map_location="cpu")
    persisted_sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

    assert [queries for queries, _ in calls] == [
        ["beta object", "gamma object"],
        ["delta object"],
    ]
    assert all(path.name == MODEL_REVISION for _, path in calls)
    assert persisted_sidecar == sidecar
    assert payload["schema_version"] == 1
    assert payload["artifact_type"] == ARTIFACT_TYPE
    assert payload["algorithm_version"] == ALGORITHM_VERSION
    assert payload["split"] == "dev"
    assert payload["queries"] == [
        "beta object",
        "gamma object",
        "delta object",
    ]
    assert payload["synsets"] == ["n00000002", "n00000003", "n00000004"]
    assert payload["embeddings"].shape == (3, OUTPUT_DIMENSION)
    assert payload["embeddings"].dtype == torch.float32
    torch.testing.assert_close(
        torch.linalg.vector_norm(payload["embeddings"], dim=-1),
        torch.ones(3),
    )
    assert payload["embedding_semantic_sha256"] == embedding_semantic_sha256(
        payload["embeddings"]
    )
    assert payload["embedding_tensor_sha256"] == tensor_sha256(
        payload["embeddings"]
    )
    assert payload["vocabulary_sha256"] == _file_sha256(vocabulary)
    assert payload["vocabulary_manifest_sha256"] == _file_sha256(manifest)
    assert payload["benchmark_vocabulary_opened"] is False
    assert payload["uses_benchmark_vocabulary_for_construction"] is False
    assert payload["text_encoder"]["model_id"] == MODEL_ID
    assert payload["text_encoder"]["revision"] == MODEL_REVISION
    assert payload["text_encoder"]["device"] == "cpu"
    assert payload["text_encoder"]["config_sha256"] == _file_sha256(
        Path(payload["text_encoder"]["snapshot_path"]) / "config.json"
    )
    assert sidecar["artifact"]["sha256"] == _file_sha256(output)
    assert sidecar["embedding"]["shape"] == [3, OUTPUT_DIMENSION]
    assert sidecar["embedding"]["byte_order"] == "little_endian"
    assert sidecar["builder"]["sha256"] == _file_sha256(
        Path(sidecar["builder"]["path"])
    )
    assert not torch.cuda.is_initialized()


def test_text_only_loader_matches_official_get_text_features_without_vision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformers = pytest.importorskip("transformers")
    snapshot, reference, config, text_state = _write_tiny_text_runtime_snapshot(
        tmp_path
    )
    config_calls: list[tuple[str, dict[str, object]]] = []

    class LocalOnlyAutoConfig:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> object:
            config_calls.append((path, kwargs))
            return config

    class ForbiddenAutoModel:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> object:
            raise AssertionError("the full vision+text AutoModel must not load")

    monkeypatch.setattr(transformers, "AutoConfig", LocalOnlyAutoConfig)
    monkeypatch.setattr(transformers, "AutoModel", ForbiddenAutoModel)
    loaded, loaded_config = _load_local_siglip2_text_tower(snapshot)

    assert loaded_config is config
    assert config_calls == [
        (
            str(snapshot.resolve()),
            {"local_files_only": True, "trust_remote_code": False},
        )
    ]
    assert not hasattr(loaded, "vision_model")
    assert sum(parameter.numel() for parameter in loaded.parameters()) == sum(
        value.numel() for value in text_state.values()
    )
    assert all(parameter.device.type == "cpu" for parameter in loaded.parameters())
    assert all(buffer.device.type == "cpu" for buffer in loaded.buffers())
    assert tuple(loaded.text_model.head.weight.shape) == (OUTPUT_DIMENSION, 8)

    input_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [8, 7, 6, 5, 4, 3, 2, 1],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        expected = reference.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        actual = loaded(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[1]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_builder_rejects_vocabulary_canonical_sha_mismatch(tmp_path: Path) -> None:
    vocabulary, manifest, records = _write_vocabulary(tmp_path)
    snapshot = _write_snapshot(tmp_path)
    vocabulary_contract = _test_vocabulary_contract(vocabulary, records)
    snapshot_contract = _test_snapshot_files_sha256(snapshot)
    vocabulary.write_text(vocabulary.read_text() + " ", encoding="utf-8")
    called = False

    def encoder(queries: Sequence[str], path: Path) -> torch.Tensor:
        nonlocal called
        called = True
        return _unit_embeddings(len(queries))

    with pytest.raises(ValueError, match="canonical SHA256"):
        build_embedding_artifact(
            vocabulary=vocabulary,
            vocabulary_manifest=manifest,
            split="dev",
            snapshot=snapshot,
            output=tmp_path / "bad.pt",
            sidecar_output=tmp_path / "bad.json",
            batch_encoder=encoder,
            _test_vocabulary_contract=vocabulary_contract,
            _test_snapshot_files_sha256=snapshot_contract,
        )
    assert called is False


def test_builder_rejects_vocabulary_split_sha_mismatch(tmp_path: Path) -> None:
    vocabulary, manifest, records = _write_vocabulary(tmp_path)
    snapshot = _write_snapshot(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["split_synset_tab_query_lf_sha256"]["dev"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="dev split SHA256 mismatch"):
        build_embedding_artifact(
            vocabulary=vocabulary,
            vocabulary_manifest=manifest,
            split="dev",
            snapshot=snapshot,
            output=tmp_path / "bad.pt",
            sidecar_output=tmp_path / "bad.json",
            batch_encoder=lambda queries, path: _unit_embeddings(len(queries)),
            _test_vocabulary_contract=_test_vocabulary_contract(vocabulary, records),
            _test_snapshot_files_sha256=_test_snapshot_files_sha256(snapshot),
        )


@pytest.mark.parametrize("which", ["vocabulary", "manifest"])
def test_builder_rejects_any_benchmark_vocabulary_access(
    tmp_path: Path,
    which: str,
) -> None:
    vocabulary, manifest, records = _write_vocabulary(
        tmp_path,
        vocabulary_benchmark_opened=which == "vocabulary",
        manifest_benchmark_opened=which == "manifest",
    )
    snapshot = _write_snapshot(tmp_path)

    with pytest.raises(ValueError, match="target-blind"):
        build_embedding_artifact(
            vocabulary=vocabulary,
            vocabulary_manifest=manifest,
            split="dev",
            snapshot=snapshot,
            output=tmp_path / "bad.pt",
            sidecar_output=tmp_path / "bad.json",
            batch_encoder=lambda queries, path: _unit_embeddings(len(queries)),
            _test_vocabulary_contract=_test_vocabulary_contract(vocabulary, records),
            _test_snapshot_files_sha256=_test_snapshot_files_sha256(snapshot),
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"model_id": "wrong/model"}, "model_id must be exactly"),
        ({"revision": "wrong-revision"}, "revision must be exactly"),
    ],
)
def test_builder_rejects_wrong_model_binding(
    tmp_path: Path,
    override: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _build(
            tmp_path,
            lambda queries, path: _unit_embeddings(len(queries)),
            **override,
        )


def test_builder_rejects_snapshot_with_wrong_text_dimension(tmp_path: Path) -> None:
    vocabulary, manifest, records = _write_vocabulary(tmp_path)
    snapshot = _write_snapshot(tmp_path, projection_size=1152)

    with pytest.raises(ValueError, match="1536-D"):
        build_embedding_artifact(
            vocabulary=vocabulary,
            vocabulary_manifest=manifest,
            split="dev",
            snapshot=snapshot,
            output=tmp_path / "bad.pt",
            sidecar_output=tmp_path / "bad.json",
            batch_encoder=lambda queries, path: _unit_embeddings(len(queries)),
            _test_vocabulary_contract=_test_vocabulary_contract(vocabulary, records),
            _test_snapshot_files_sha256=_test_snapshot_files_sha256(snapshot),
        )


def test_snapshot_validator_accepts_manifest_bound_hf_blob_links(
    tmp_path: Path,
) -> None:
    snapshot = _write_snapshot(
        tmp_path / "models--google--siglip2-giant-opt-patch16-384"
    )
    expected_files_sha256, targets = _replace_snapshot_files_with_hf_blob_links(
        snapshot
    )

    contract = _validate_snapshot(
        snapshot,
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        expected_files_sha256=expected_files_sha256,
    )

    assert set(targets) == set(expected_files_sha256)
    assert all((snapshot / name).is_symlink() for name in targets)
    assert all(target.is_file() for target in targets.values())
    assert contract["snapshot_files_sha256"] == hashlib.sha256(
        json.dumps(
            expected_files_sha256,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def test_snapshot_validator_rejects_hf_link_outside_same_model_blobs(
    tmp_path: Path,
) -> None:
    snapshot = _write_snapshot(
        tmp_path / "models--google--siglip2-giant-opt-patch16-384"
    )
    expected_files_sha256, targets = _replace_snapshot_files_with_hf_blob_links(
        snapshot
    )
    link = snapshot / "config.json"
    outside = tmp_path / "outside" / "config.json"
    outside.parent.mkdir()
    outside.write_bytes(targets["config.json"].read_bytes())
    assert _file_sha256(outside) == expected_files_sha256["config.json"]
    link.unlink()
    link.symlink_to(Path(os.path.relpath(outside, start=link.parent)))

    with pytest.raises(ValueError, match="not a Hugging Face blob"):
        _validate_snapshot(
            snapshot,
            model_id=MODEL_ID,
            revision=MODEL_REVISION,
            expected_files_sha256=expected_files_sha256,
        )


def test_snapshot_validator_rejects_hf_link_to_non_regular_blob(
    tmp_path: Path,
) -> None:
    snapshot = _write_snapshot(
        tmp_path / "models--google--siglip2-giant-opt-patch16-384"
    )
    expected_files_sha256, _ = _replace_snapshot_files_with_hf_blob_links(snapshot)
    link = snapshot / "config.json"
    directory_target = snapshot.parent.parent / "blobs" / ("f" * 64)
    directory_target.mkdir()
    link.unlink()
    link.symlink_to(Path("..") / ".." / "blobs" / directory_target.name)

    with pytest.raises(ValueError, match="blob is not regular"):
        _validate_snapshot(
            snapshot,
            model_id=MODEL_ID,
            revision=MODEL_REVISION,
            expected_files_sha256=expected_files_sha256,
        )


def test_snapshot_validator_rejects_hf_blob_sha_drift(tmp_path: Path) -> None:
    snapshot = _write_snapshot(
        tmp_path / "models--google--siglip2-giant-opt-patch16-384"
    )
    expected_files_sha256, targets = _replace_snapshot_files_with_hf_blob_links(
        snapshot
    )
    shard_name = "model-00001-of-00001.safetensors"
    targets[shard_name].write_bytes(b"drifted-weight-shard")
    assert _file_sha256(targets[shard_name]) != expected_files_sha256[shard_name]

    with pytest.raises(ValueError, match="digests|SHA-256"):
        _validate_snapshot(
            snapshot,
            model_id=MODEL_ID,
            revision=MODEL_REVISION,
            expected_files_sha256=expected_files_sha256,
        )


def test_production_builder_rejects_self_signed_noncanonical_vocabulary(
    tmp_path: Path,
) -> None:
    vocabulary, manifest, _ = _write_vocabulary(tmp_path)
    snapshot = _write_snapshot(tmp_path)

    with pytest.raises(ValueError, match="frozen target-blind canonical SHA256"):
        build_embedding_artifact(
            vocabulary=vocabulary,
            vocabulary_manifest=manifest,
            split="dev",
            snapshot=snapshot,
            output=tmp_path / "bad.pt",
            sidecar_output=tmp_path / "bad.json",
            batch_encoder=lambda queries, path: _unit_embeddings(len(queries)),
        )


def test_production_builder_rejects_same_revision_mock_snapshot(
    tmp_path: Path,
) -> None:
    vocabulary, manifest, records = _write_vocabulary(tmp_path)
    snapshot = _write_snapshot(tmp_path)

    with pytest.raises(ValueError, match="frozen official revision digests"):
        build_embedding_artifact(
            vocabulary=vocabulary,
            vocabulary_manifest=manifest,
            split="dev",
            snapshot=snapshot,
            output=tmp_path / "bad.pt",
            sidecar_output=tmp_path / "bad.json",
            batch_encoder=lambda queries, path: _unit_embeddings(len(queries)),
            _test_vocabulary_contract=_test_vocabulary_contract(vocabulary, records),
        )


def _wrong_shape(rows: int) -> torch.Tensor:
    return torch.zeros(rows, OUTPUT_DIMENSION - 1, dtype=torch.float32)


def _wrong_dtype(rows: int) -> torch.Tensor:
    return _unit_embeddings(rows).double()


def _wrong_norm(rows: int) -> torch.Tensor:
    return _unit_embeddings(rows) * 2.0


def _nonfinite(rows: int) -> torch.Tensor:
    values = _unit_embeddings(rows)
    values[0, 0] = float("nan")
    return values


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (_wrong_shape, "shape"),
        (_wrong_dtype, "float32"),
        (_wrong_norm, "L2-normalized"),
        (_nonfinite, "NaN or infinity"),
    ],
)
def test_builder_rejects_invalid_encoder_output(
    tmp_path: Path,
    factory: Callable[[int], torch.Tensor],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _build(
            tmp_path,
            lambda queries, path: factory(len(queries)),
        )
    assert not (tmp_path / "dev_embeddings.pt").exists()
