#!/usr/bin/env python3
"""Build one frozen target-blind SigLIP2 text-embedding split on CPU.

The production path is local-files-only and binds the exact model revision.
Tests may inject a batch encoder, but vocabulary and snapshot provenance are
still validated before that encoder is called.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.scripts.eval_lerf_grounding import (
    _canonicalize_siglip2_text,
    _resolve_siglip2_text_max_length,
)
from radio_gs.utils.immutable_artifacts import (
    load_json_object,
    sha256_file,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "target_blind_text_embedding_cache"
MANIFEST_ARTIFACT_TYPE = "target_blind_text_embedding_cache_manifest"
ALGORITHM_VERSION = "siglip2-target-blind-split-v1"
VOCABULARY_ARTIFACT_TYPE = "target_blind_imagenet1k_primary_text_bank"
VOCABULARY_MANIFEST_ARTIFACT_TYPE = (
    "target_blind_imagenet1k_primary_text_bank_manifest"
)
VOCABULARY_ALGORITHM_VERSION = "imagenet1k-primary-v1"
MODEL_ID = "google/siglip2-giant-opt-patch16-384"
MODEL_REVISION = "a713301b217d38485fb2204c808367d10bc3cc40"
TEXT_CANONICALIZATION = "official_c_radio_siglip2_g"
OUTPUT_DIMENSION = 1536
SPLITS = ("fit", "dev", "audit")
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
SNAPSHOT_AUXILIARY_FILES = ("preprocessor_config.json",)
BatchEncoder = Callable[[Sequence[str], Path], torch.Tensor]

# These values bind the production cache to the exact target-blind bank built
# from the two pinned timm ImageNet-1K metadata files.  A manifest is not an
# authority by itself: without these independent values, an edited vocabulary
# and an edited manifest could simply sign each other.
FROZEN_VOCABULARY_CONTRACT: Mapping[str, Any] = {
    "canonical_vocabulary_sha256": (
        "2644c8454c12b0d6ca16fc453ee63e5289112172b82b61136e003ddf65a090ab"
    ),
    "counts": {
        "source_synsets": 1000,
        "deduplicated_queries": 997,
        "fit": 806,
        "dev": 101,
        "audit": 90,
    },
    "source_sha256": {
        "imagenet_synsets.txt": (
            "70002b0ff5de60a3a17a82dbfcff291931f96225ddf941ad2e182fc39e183d15"
        ),
        "imagenet_synset_to_lemma.txt": (
            "1b8babda187421a4bde0c9c5a197c36f6bdda962f7ca11ffb2813806cbb2178f"
        ),
    },
    "split_sha256": {
        "fit": "fe1e6ca6ec1656fefb66681d14a933158a336452ee06fbe22c4a62eaac994530",
        "dev": "26b814d872a6455961097d1ab1390d951d69e69c2a88d4c58e9bd003217f6544",
        "audit": "3b78a2e81e2750dd7314d6431ac44ddea05dd505948d775e9d1e33e87ae0bc7b",
    },
}

# Exact SHA-256 digests for revision a713301b....  The two safetensor values
# are the Hugging Face LFS object IDs (content SHA-256); the remaining values
# were independently hashed from the local snapshot.  Checking the mapping,
# rather than only recording it in a sidecar, prevents a same-named directory
# containing another model from being accepted as the official checkpoint.
FROZEN_SNAPSHOT_FILES_SHA256: Mapping[str, str] = {
    "config.json": "04d69dffcd86c212b5433f791ac685a54a951c00da5ed556e968f10aa101f663",
    "model.safetensors.index.json": (
        "3766cd1194767496943ae5deb6add8c994b41158c69e6c13d4037cbaf8a4e118"
    ),
    "tokenizer.json": "cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322",
    "tokenizer.model": "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2",
    "tokenizer_config.json": (
        "14afe629fe4959b9e0d51e1852b8d9f7ad074f90a1a7125a4fcdd17f06e78fc8"
    ),
    "special_tokens_map.json": (
        "baec30ea10906f16adb8c18af7a34023002c1746542612b8b41c9f09e1351351"
    ),
    "preprocessor_config.json": (
        "fb2817d3523ca3b666c859f15320c7138416bc38ffc515e2963f78c868c51c90"
    ),
    "model-00001-of-00002.safetensors": (
        "9f6da5ad2e3178b0c1197b740c9ffefee0bc733ee8759095a5ea8c4c845a3f4e"
    ),
    "model-00002-of-00002.safetensors": (
        "06e64949dd7cafc4d278723533e2591145ae8e5616d2c0c5d8458de25ceede34"
    ),
}
FROZEN_SNAPSHOT_FILES_SHA256_DIGEST = (
    "974060e07c95d89ab19af8a0388bedee99c949af9df63a90038baf11d823ff77"
)


def _sha256_file(path: Path) -> str:
    return sha256_file(path)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def embedding_semantic_sha256(value: torch.Tensor) -> str:
    """Hash contiguous C-order little-endian float32 embedding bytes."""

    tensor = torch.as_tensor(value).detach().cpu().to(torch.float32).contiguous()
    array = tensor.numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    value, _, _ = load_json_object(path, label=label)
    return value


def _split_line_sha256(records: Sequence[Mapping[str, str]], split: str) -> str:
    lines = "".join(
        f"{record['synset']}\t{record['query']}\n"
        for record in records
        if record["split"] == split
    )
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def _ordered_records_sha256(records: Sequence[Mapping[str, str]]) -> str:
    return _canonical_json_sha256(
        [
            {
                "synset": record["synset"],
                "query": record["query"],
                "split": record["split"],
            }
            for record in records
        ]
    )


def _validate_vocabulary(
    vocabulary_path: Path,
    vocabulary_manifest_path: Path,
    split: str,
    *,
    frozen_contract: Mapping[str, Any] = FROZEN_VOCABULARY_CONTRACT,
) -> dict[str, Any]:
    vocabulary_path = vocabulary_path.resolve()
    vocabulary_manifest_path = vocabulary_manifest_path.resolve()
    if not vocabulary_path.is_file() or not vocabulary_manifest_path.is_file():
        raise FileNotFoundError("vocabulary JSON and manifest must both exist")
    vocabulary_sha256 = _sha256_file(vocabulary_path)
    vocabulary_manifest_sha256 = _sha256_file(vocabulary_manifest_path)
    vocabulary = _read_json_object(vocabulary_path, "vocabulary")
    manifest = _read_json_object(vocabulary_manifest_path, "vocabulary manifest")
    if (
        vocabulary.get("schema_version") != 1
        or vocabulary.get("artifact_type") != VOCABULARY_ARTIFACT_TYPE
        or vocabulary.get("algorithm_version") != VOCABULARY_ALGORITHM_VERSION
        or vocabulary.get("prompt_templates") != ["{query}"]
        or vocabulary.get("benchmark_vocabulary_opened") is not False
    ):
        raise ValueError("vocabulary is not the target-blind primary bank v1")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_type")
        != VOCABULARY_MANIFEST_ARTIFACT_TYPE
        or manifest.get("algorithm_version") != VOCABULARY_ALGORITHM_VERSION
        or manifest.get("benchmark_vocabulary_opened") is not False
    ):
        raise ValueError("vocabulary manifest is not target-blind primary v1")
    canonical_record = manifest.get("canonical_json")
    if (
        not isinstance(canonical_record, Mapping)
        or canonical_record.get("sha256") != vocabulary_sha256
    ):
        raise ValueError("vocabulary canonical SHA256 does not match its manifest")
    if vocabulary_sha256 != frozen_contract.get("canonical_vocabulary_sha256"):
        raise ValueError(
            "vocabulary is self-consistent but does not match the frozen "
            "target-blind canonical SHA256"
        )

    raw_records = vocabulary.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("vocabulary records must be a non-empty list")
    records: list[dict[str, str]] = []
    for raw in raw_records:
        if not isinstance(raw, Mapping):
            raise ValueError("every vocabulary record must be an object")
        record = {
            "synset": str(raw.get("synset", "")),
            "query": str(raw.get("query", "")),
            "split": str(raw.get("split", "")),
        }
        if (
            not record["synset"]
            or not record["query"]
            or record["split"] not in SPLITS
        ):
            raise ValueError("vocabulary contains an invalid record")
        canonical_query = _canonicalize_siglip2_text(record["query"])
        if canonical_query != record["query"]:
            raise ValueError(
                "vocabulary query is not canonical under the official "
                "C-RADIO SigLIP2 text policy"
            )
        records.append(record)
    if len({record["synset"] for record in records}) != len(records):
        raise ValueError("vocabulary synsets must be unique")
    if len({record["query"] for record in records}) != len(records):
        raise ValueError("vocabulary queries must be first-wins deduplicated")

    declared_split_hashes = manifest.get(
        "split_synset_tab_query_lf_sha256"
    )
    if not isinstance(declared_split_hashes, Mapping):
        raise ValueError("vocabulary manifest lacks split SHA256 records")
    computed_split_hashes = {
        name: _split_line_sha256(records, name) for name in SPLITS
    }
    for name, digest in computed_split_hashes.items():
        if declared_split_hashes.get(name) != digest:
            raise ValueError(f"vocabulary {name} split SHA256 mismatch")
    expected_split_hashes = frozen_contract.get("split_sha256")
    if not isinstance(expected_split_hashes, Mapping) or dict(
        computed_split_hashes
    ) != dict(expected_split_hashes):
        raise ValueError("vocabulary split hashes differ from the frozen contract")
    split_counts = {
        name: sum(record["split"] == name for record in records) for name in SPLITS
    }
    actual_counts = {
        "source_synsets": len(records),
        "deduplicated_queries": len(records),
        **split_counts,
    }
    declared_counts = manifest.get("counts")
    expected_counts = frozen_contract.get("counts")
    if (
        not isinstance(declared_counts, Mapping)
        or not isinstance(expected_counts, Mapping)
        or dict(declared_counts) != dict(expected_counts)
        or actual_counts["deduplicated_queries"]
        != int(expected_counts.get("deduplicated_queries", -1))
        or split_counts
        != {name: int(expected_counts.get(name, -1)) for name in SPLITS}
    ):
        raise ValueError("vocabulary counts differ from the frozen contract")
    declared_sources = manifest.get("sources")
    expected_sources = frozen_contract.get("source_sha256")
    if not isinstance(declared_sources, Mapping) or not isinstance(
        expected_sources, Mapping
    ):
        raise ValueError("vocabulary manifest lacks frozen source provenance")
    source_hashes = {
        str(name): str(record.get("sha256", ""))
        for name, record in declared_sources.items()
        if isinstance(record, Mapping)
    }
    if source_hashes != dict(expected_sources):
        raise ValueError("vocabulary source hashes differ from the frozen contract")
    selected = [record for record in records if record["split"] == split]
    if not selected:
        raise ValueError(f"vocabulary split {split} is empty")
    return {
        "vocabulary_path": vocabulary_path,
        "vocabulary_sha256": vocabulary_sha256,
        "vocabulary_manifest_path": vocabulary_manifest_path,
        "vocabulary_manifest_sha256": vocabulary_manifest_sha256,
        "records": selected,
        "split_sha256": computed_split_hashes[split],
        "ordered_records_sha256": _ordered_records_sha256(selected),
    }


def _safe_snapshot_file(snapshot: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
        raise ValueError(f"snapshot index contains unsafe filename {name!r}")
    path = snapshot / relative
    try:
        before = os.lstat(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"local SigLIP2 snapshot lacks {name}") from exc
    if stat.S_ISREG(before.st_mode):
        return path
    if not stat.S_ISLNK(before.st_mode):
        raise ValueError(f"local SigLIP2 snapshot file is not regular: {name}")

    # Hugging Face snapshots are content-addressed views whose files point to
    # ../../blobs/<git-or-lfs-object-id>.  Accept only that exact one-hop
    # layout, then return the regular blob itself so immutable artifact reads
    # retain O_NOFOLLOW and descriptor rehash guarantees.  Arbitrary snapshot
    # symlinks and symlink chains remain forbidden.
    expected_model_root = "models--" + MODEL_ID.replace("/", "--")
    if (
        snapshot.parent.name != "snapshots"
        or snapshot.parent.parent.name != expected_model_root
    ):
        raise ValueError("local SigLIP2 snapshot symlink is outside the fixed model root")
    target = Path(os.readlink(path))
    parts = target.parts
    if (
        target.is_absolute()
        or len(parts) != 4
        or parts[:3] != ("..", "..", "blobs")
        or len(parts[3]) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in parts[3])
    ):
        raise ValueError(
            f"local SigLIP2 snapshot symlink is not a Hugging Face blob: {name}"
        )
    after = os.lstat(path)
    if (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError(f"local SigLIP2 snapshot symlink changed: {name}")
    blob_root = snapshot.parent.parent / "blobs"
    if blob_root.is_symlink() or not blob_root.is_dir():
        raise ValueError("local Hugging Face blob store is not a real directory")
    blob = blob_root / parts[3]
    try:
        blob_info = os.stat(blob, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"local SigLIP2 snapshot blob is missing: {name}"
        ) from exc
    if not stat.S_ISREG(blob_info.st_mode):
        raise ValueError(f"local SigLIP2 snapshot blob is not regular: {name}")
    return blob


def _validate_snapshot(
    snapshot: Path,
    *,
    model_id: str,
    revision: str,
    expected_files_sha256: Mapping[str, str] = FROZEN_SNAPSHOT_FILES_SHA256,
) -> dict[str, Any]:
    if model_id != MODEL_ID:
        raise ValueError(f"model_id must be exactly {MODEL_ID}")
    if revision != MODEL_REVISION:
        raise ValueError(f"revision must be exactly {MODEL_REVISION}")
    snapshot = snapshot.resolve()
    if not snapshot.is_dir():
        raise FileNotFoundError(f"local SigLIP2 snapshot does not exist: {snapshot}")
    if snapshot.name != revision:
        raise ValueError("local snapshot directory name must equal the fixed revision")

    config_path = _safe_snapshot_file(snapshot, "config.json")
    index_path = _safe_snapshot_file(snapshot, "model.safetensors.index.json")
    tokenizer_paths = {
        name: _safe_snapshot_file(snapshot, name) for name in TOKENIZER_FILES
    }
    auxiliary_paths = {
        name: _safe_snapshot_file(snapshot, name)
        for name in SNAPSHOT_AUXILIARY_FILES
    }
    config = _read_json_object(config_path, "SigLIP2 config")
    text_config = config.get("text_config")
    if (
        config.get("model_type") != "siglip"
        or not isinstance(text_config, Mapping)
        or int(text_config.get("projection_size", 0)) != OUTPUT_DIMENSION
        or int(text_config.get("hidden_size", 0)) <= 0
    ):
        raise ValueError("local snapshot is not the fixed 1536-D SigLIP2 model")
    index = _read_json_object(index_path, "SigLIP2 safetensors index")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ValueError("SigLIP2 safetensors index lacks a weight_map")
    for required_key in (
        "text_model.head.weight",
        "text_model.head.bias",
        "text_model.embeddings.token_embedding.weight",
    ):
        if not str(weight_map.get(required_key, "")):
            raise ValueError(f"SigLIP2 index lacks {required_key}")
    shard_names = sorted({str(value) for value in weight_map.values()})
    shard_paths = {
        name: _safe_snapshot_file(snapshot, name) for name in shard_names
    }

    config_sha256 = _sha256_file(config_path)
    model_index_sha256 = _sha256_file(index_path)
    tokenizer_files_sha256 = {
        name: _sha256_file(path) for name, path in tokenizer_paths.items()
    }
    auxiliary_files_sha256 = {
        name: _sha256_file(path) for name, path in auxiliary_paths.items()
    }
    weight_shards_sha256 = {
        name: _sha256_file(path) for name, path in shard_paths.items()
    }
    snapshot_files_sha256 = {
        "config.json": config_sha256,
        "model.safetensors.index.json": model_index_sha256,
        **tokenizer_files_sha256,
        **auxiliary_files_sha256,
        **weight_shards_sha256,
    }
    if snapshot_files_sha256 != dict(expected_files_sha256):
        raise ValueError(
            "local SigLIP2 snapshot files do not match the frozen official "
            "revision digests"
        )
    snapshot_digest = _canonical_json_sha256(snapshot_files_sha256)
    if expected_files_sha256 is FROZEN_SNAPSHOT_FILES_SHA256 and (
        snapshot_digest != FROZEN_SNAPSHOT_FILES_SHA256_DIGEST
    ):
        raise RuntimeError("internal frozen SigLIP2 snapshot digest is inconsistent")
    return {
        "model_id": model_id,
        "revision": revision,
        "snapshot_path": str(snapshot),
        "snapshot_files_sha256": snapshot_digest,
        "config_sha256": config_sha256,
        "tokenizer_sha256": _canonical_json_sha256(
            tokenizer_files_sha256
        ),
        "tokenizer_files_sha256": tokenizer_files_sha256,
        "auxiliary_files_sha256": auxiliary_files_sha256,
        "model_index_sha256": model_index_sha256,
        "weight_shards_sha256": weight_shards_sha256,
        "output_dimension": OUTPUT_DIMENSION,
        "dtype": "float32",
        "normalization": "l2",
        "device": "cpu",
    }


def _replace_parameter(
    model: nn.Module,
    name: str,
    value: torch.Tensor,
) -> None:
    """Bind one checkpoint tensor without first allocating a CPU model copy."""

    module_name, _, parameter_name = name.rpartition(".")
    module = model.get_submodule(module_name) if module_name else model
    parameter = module._parameters.get(parameter_name)
    if parameter is None:
        raise RuntimeError(f"SigLIP2 text model lacks parameter {name}")
    if tuple(parameter.shape) != tuple(value.shape):
        raise RuntimeError(
            f"SigLIP2 parameter {name} has shape {tuple(value.shape)}, "
            f"expected {tuple(parameter.shape)}"
        )
    if value.device.type != "cpu":
        raise RuntimeError(f"SigLIP2 parameter {name} was not loaded on CPU")
    module._parameters[parameter_name] = nn.Parameter(
        value,
        requires_grad=parameter.requires_grad,
    )


def _materialize_text_tower_buffers(
    model: nn.Module,
    *,
    max_position_embeddings: int,
) -> None:
    """Materialize the one non-persistent buffer created on the meta device."""

    meta_buffers = {
        name: value
        for name, value in model.named_buffers()
        if value.device.type == "meta"
    }
    expected_name = "text_model.embeddings.position_ids"
    unexpected = sorted(set(meta_buffers) - {expected_name})
    if unexpected:
        raise RuntimeError(
            "SigLIP2 text model has unsupported meta buffers: " + ", ".join(unexpected)
        )
    if expected_name in meta_buffers:
        embeddings = model.get_submodule("text_model.embeddings")
        embeddings.position_ids = torch.arange(
            max_position_embeddings,
            dtype=meta_buffers[expected_name].dtype,
            device=torch.device("cpu"),
        ).expand((1, -1))


def _load_local_siglip2_text_tower(
    snapshot: Path,
) -> tuple[nn.Module, object]:
    """Load only indexed ``text_model.*`` weights from the local snapshot.

    ``SiglipTextModel.from_pretrained`` still walks every shard of a combined
    SigLIP checkpoint, while ``AutoModel`` additionally instantiates the vision
    tower.  Building the official text wrapper on ``meta`` and binding only its
    indexed tensors avoids both costs.  The projection head is resized before
    loading because transformers 4.46--4.49 otherwise constructs 1152->1152
    despite the fixed checkpoint's configured 1152->1536 projection.
    """

    try:
        from safetensors import safe_open
        from transformers import AutoConfig, SiglipTextModel
    except ImportError as error:
        raise RuntimeError(
            "transformers and safetensors are required for local SigLIP2 restore"
        ) from error
    snapshot = snapshot.resolve()
    config = AutoConfig.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
    )
    text_config = getattr(config, "text_config", None)
    projection_size = int(getattr(text_config, "projection_size", 0) or 0)
    hidden_size = int(getattr(text_config, "hidden_size", 0) or 0)
    max_position_embeddings = int(
        getattr(text_config, "max_position_embeddings", 0) or 0
    )
    if (
        text_config is None
        or projection_size != OUTPUT_DIMENSION
        or hidden_size <= 0
        or max_position_embeddings <= 0
    ):
        raise RuntimeError("local SigLIP2 runtime config has the wrong text shape")

    with torch.device("meta"):
        # SiglipModel.__init__ uses this same factory, including transformers'
        # official attention-backend selection for the text sub-config.
        model = SiglipTextModel._from_config(text_config)
        model.text_model.head = nn.Linear(
            hidden_size,
            projection_size,
            bias=True,
        )

    index = _read_json_object(
        _safe_snapshot_file(snapshot, "model.safetensors.index.json"),
        "SigLIP2 safetensors index",
    )
    weight_map = index["weight_map"]
    if not isinstance(weight_map, Mapping):
        raise RuntimeError("SigLIP2 safetensors index lacks a weight_map")
    expected = set(dict(model.named_parameters()))
    indexed_text = {
        str(key) for key in weight_map if str(key).startswith("text_model.")
    }
    missing = sorted(expected - indexed_text)
    unexpected = sorted(indexed_text - expected)
    if missing or unexpected:
        detail = []
        if missing:
            detail.append("missing=" + ",".join(missing))
        if unexpected:
            detail.append("unexpected=" + ",".join(unexpected))
        raise RuntimeError(
            "SigLIP2 text checkpoint does not exactly match the official "
            "text tower: " + "; ".join(detail)
        )

    keys_by_shard: dict[str, list[str]] = {}
    for key in sorted(expected):
        shard_name = str(weight_map[key])
        keys_by_shard.setdefault(shard_name, []).append(key)
    for shard_name, keys in sorted(keys_by_shard.items()):
        shard = _safe_snapshot_file(snapshot, shard_name)
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for key in keys:
                if key not in available:
                    raise RuntimeError(f"SigLIP2 shard does not contain {key}")
                _replace_parameter(model, key, handle.get_tensor(key))

    _materialize_text_tower_buffers(
        model,
        max_position_embeddings=max_position_embeddings,
    )
    non_cpu = [
        name
        for name, value in (*model.named_parameters(), *model.named_buffers())
        if value.device.type != "cpu"
    ]
    if non_cpu:
        raise RuntimeError(
            "SigLIP2 text tower was not fully materialized on CPU: "
            + ", ".join(non_cpu)
        )
    return model.eval(), config


class _LocalSiglip2BatchEncoder:
    """Local-files-only official text tower retained across encoder batches."""

    def __init__(self, snapshot: Path) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "transformers is required to encode the frozen SigLIP2 bank"
            ) from error
        snapshot = snapshot.resolve()
        self.model, config = _load_local_siglip2_text_tower(snapshot)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot),
            local_files_only=True,
            trust_remote_code=False,
        )
        self.max_length = _resolve_siglip2_text_max_length(config)
        if any(parameter.device.type != "cpu" for parameter in self.model.parameters()):
            raise RuntimeError("SigLIP2 text encoder must remain entirely on CPU")

    @torch.inference_mode()
    def __call__(self, queries: Sequence[str], snapshot: Path) -> torch.Tensor:
        del snapshot
        inputs = self.tokenizer(
            list(queries),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tensor_inputs = {
            str(name): value.to(torch.device("cpu"))
            for name, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }
        # This is the exact body of SiglipModel.get_text_features after the
        # full model dispatches to its text transformer.  SiglipTextModel in
        # transformers 4.46 exposes the pooled value through forward only;
        # 4.49 additionally offers a convenience get_text_features method.
        embeddings = self.model(**tensor_inputs)[1]
        return F.normalize(embeddings.float(), dim=-1).cpu().contiguous()


def _validate_embedding_batch(
    value: torch.Tensor,
    *,
    rows: int,
) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.device.type != "cpu":
        raise ValueError("SigLIP2 embedding batches must remain on CPU")
    if tensor.dtype != torch.float32 or tensor.shape != (rows, OUTPUT_DIMENSION):
        raise ValueError(
            "SigLIP2 embedding batch must be float32 with shape "
            f"[{rows},{OUTPUT_DIMENSION}]"
        )
    tensor = tensor.detach().contiguous()
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("SigLIP2 embedding batch contains NaN or infinity")
    norms = torch.linalg.vector_norm(tensor, dim=-1)
    if not bool(torch.allclose(norms, torch.ones_like(norms), atol=5e-5, rtol=5e-5)):
        raise ValueError("SigLIP2 embedding rows must already be L2-normalized")
    return tensor


def _build_embedding_artifact_from_contracts(
    *,
    vocabulary_contract: Mapping[str, Any],
    encoder_contract: Mapping[str, Any],
    snapshot: Path,
    output: Path,
    sidecar_output: Path,
    batch_size: int,
    encoder: BatchEncoder,
) -> dict[str, Any]:
    """Encode and atomically write one split after shared input validation."""

    records = list(vocabulary_contract["records"])
    queries = [record["query"] for record in records]
    synsets = [record["synset"] for record in records]
    batches: list[torch.Tensor] = []
    for start in range(0, len(queries), batch_size):
        batch_queries = queries[start : start + batch_size]
        encoded = encoder(batch_queries, snapshot)
        batches.append(
            _validate_embedding_batch(encoded, rows=len(batch_queries))
        )
    embeddings = torch.cat(batches, dim=0).to(torch.float32).contiguous()
    if embeddings.shape != (len(records), OUTPUT_DIMENSION):
        raise RuntimeError("concatenated embedding artifact has an invalid shape")
    semantic_sha256 = embedding_semantic_sha256(embeddings)
    typed_tensor_sha256 = tensor_sha256(embeddings)
    encoder_contract = dict(encoder_contract)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "algorithm_version": ALGORITHM_VERSION,
        "benchmark_vocabulary_opened": False,
        "uses_benchmark_vocabulary_for_construction": False,
        "split": records[0]["split"],
        "split_synset_tab_query_lf_sha256": vocabulary_contract[
            "split_sha256"
        ],
        "prompt_templates": ["{query}"],
        "text_canonicalization": TEXT_CANONICALIZATION,
        "records": records,
        "queries": queries,
        "synsets": synsets,
        "ordered_records_sha256": vocabulary_contract[
            "ordered_records_sha256"
        ],
        "vocabulary_path": str(vocabulary_contract["vocabulary_path"]),
        "vocabulary_sha256": vocabulary_contract["vocabulary_sha256"],
        "vocabulary_manifest_path": str(
            vocabulary_contract["vocabulary_manifest_path"]
        ),
        "vocabulary_manifest_sha256": vocabulary_contract[
            "vocabulary_manifest_sha256"
        ],
        "embeddings": embeddings,
        "embedding_semantic_sha256": semantic_sha256,
        "embedding_tensor_sha256": typed_tensor_sha256,
        "text_encoder": encoder_contract,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary_output)
    temporary_output.replace(output)
    artifact_sha256 = _sha256_file(output)
    builder_path = Path(__file__).resolve()
    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": MANIFEST_ARTIFACT_TYPE,
        "algorithm_version": ALGORITHM_VERSION,
        "benchmark_vocabulary_opened": False,
        "uses_benchmark_vocabulary_for_construction": False,
        "split": records[0]["split"],
        "split_synset_tab_query_lf_sha256": vocabulary_contract[
            "split_sha256"
        ],
        "prompt_templates": ["{query}"],
        "text_canonicalization": TEXT_CANONICALIZATION,
        "records": records,
        "queries": queries,
        "synsets": synsets,
        "ordered_records_sha256": vocabulary_contract[
            "ordered_records_sha256"
        ],
        "vocabulary": {
            "path": str(vocabulary_contract["vocabulary_path"]),
            "sha256": vocabulary_contract["vocabulary_sha256"],
            "manifest_path": str(
                vocabulary_contract["vocabulary_manifest_path"]
            ),
            "manifest_sha256": vocabulary_contract[
                "vocabulary_manifest_sha256"
            ],
        },
        "text_encoder": encoder_contract,
        "embedding": {
            "shape": [len(records), OUTPUT_DIMENSION],
            "dtype": "float32",
            "byte_order": "little_endian",
            "normalization": "l2",
            "semantic_sha256": semantic_sha256,
            "tensor_sha256": typed_tensor_sha256,
        },
        "artifact": {
            "path": str(output),
            "sha256": artifact_sha256,
        },
        "builder": {
            "path": str(builder_path),
            "sha256": _sha256_file(builder_path),
        },
    }
    sidecar_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_sidecar = sidecar_output.with_suffix(sidecar_output.suffix + ".tmp")
    temporary_sidecar.write_bytes(_canonical_json_bytes(sidecar) + b"\n")
    temporary_sidecar.replace(sidecar_output)
    return sidecar


def build_embedding_artifact(
    *,
    vocabulary: Path,
    vocabulary_manifest: Path,
    split: str,
    snapshot: Path,
    output: Path,
    sidecar_output: Path,
    model_id: str = MODEL_ID,
    revision: str = MODEL_REVISION,
    batch_size: int = 32,
    batch_encoder: BatchEncoder | None = None,
    _test_vocabulary_contract: Mapping[str, Any] | None = None,
    _test_snapshot_files_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    split = str(split)
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    vocabulary = Path(vocabulary).resolve()
    vocabulary_manifest = Path(vocabulary_manifest).resolve()
    snapshot = Path(snapshot).resolve()
    output = Path(output).resolve()
    sidecar_output = Path(sidecar_output).resolve()
    if output == sidecar_output:
        raise ValueError("output and sidecar_output must be different files")
    if output in {vocabulary, vocabulary_manifest} or sidecar_output in {
        vocabulary,
        vocabulary_manifest,
    }:
        raise ValueError("embedding outputs must not overwrite vocabulary inputs")

    vocabulary_contract = _validate_vocabulary(
        vocabulary,
        vocabulary_manifest,
        split,
        frozen_contract=(
            FROZEN_VOCABULARY_CONTRACT
            if _test_vocabulary_contract is None
            else _test_vocabulary_contract
        ),
    )
    encoder_contract = _validate_snapshot(
        snapshot,
        model_id=str(model_id),
        revision=str(revision),
        expected_files_sha256=(
            FROZEN_SNAPSHOT_FILES_SHA256
            if _test_snapshot_files_sha256 is None
            else _test_snapshot_files_sha256
        ),
    )
    encoder = batch_encoder
    if encoder is None:
        encoder = _LocalSiglip2BatchEncoder(snapshot)
    return _build_embedding_artifact_from_contracts(
        vocabulary_contract=vocabulary_contract,
        encoder_contract=encoder_contract,
        snapshot=snapshot,
        output=output,
        sidecar_output=sidecar_output,
        batch_size=batch_size,
        encoder=encoder,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--vocabulary-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sidecar-output", type=Path, required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    sidecar = build_embedding_artifact(
        vocabulary=args.vocabulary,
        vocabulary_manifest=args.vocabulary_manifest,
        split=args.split,
        snapshot=args.snapshot,
        output=args.output,
        sidecar_output=args.sidecar_output,
        model_id=args.model_id,
        revision=args.revision,
        batch_size=args.batch_size,
    )
    print(json.dumps(sidecar, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
