#!/usr/bin/env python3
"""Evaluate and gate target-blind generic text-response fidelity on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import string
from pathlib import Path
from typing import Mapping

import torch

from radio_gs.evaluation.text_response_fidelity import (
    IMAGENET12K_HOLDOUT_BANK_FAMILY,
    IMAGENET1K_PRIMARY_BANK_FAMILY,
    REPORT_ARTIFACT_TYPE,
    REPORT_SCHEMA_VERSION,
    aggregate_paired_seed_gate,
    canonical_json_sha256,
    evaluate_response_fidelity,
    row_identity_sha256,
    selection_contract_for_bank_family,
    tensor_sha256,
)
from radio_gs.scripts import (
    build_target_blind_siglip2_embedding_artifact as frozen_text_builder,
)
from radio_gs.scripts import (
    build_target_blind_imagenet12k_holdout_embedding as holdout_text_builder,
)
from radio_gs.scripts import (
    build_target_blind_imagenet12k_holdout_bank as holdout_vocabulary_builder,
)
from radio_gs.utils.immutable_artifacts import (
    load_json_object,
    load_torch_mapping,
    sha256_file,
)


DESCRIPTOR_SCHEMA_VERSION = 1
DESCRIPTOR_ARTIFACT_TYPE = "surface_text_response_descriptor_pair"
TEXT_BANK_SCHEMA_VERSION = 1
TEXT_BANK_ARTIFACT_TYPE = "target_blind_text_embedding_cache"
TEXT_BANK_MANIFEST_ARTIFACT_TYPE = "target_blind_text_embedding_cache_manifest"
TEXT_BANK_ALGORITHM_VERSION = "siglip2-target-blind-split-v1"
HOLDOUT_TEXT_BANK_ALGORITHM_VERSION = holdout_text_builder.ALGORITHM_VERSION
TEXT_BANK_CANONICALIZATION = "official_c_radio_siglip2_g"
TEXT_BANK_MODEL_ID = "google/siglip2-giant-opt-patch16-384"
TEXT_BANK_MODEL_REVISION = "a713301b217d38485fb2204c808367d10bc3cc40"
TEXT_BANK_OUTPUT_DIMENSION = 1536
TEXT_BANK_TOKENIZER_FILES = frozenset(
    {
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
    }
)
ALLOWED_HELDOUT_SPLITS = frozenset({"dev", "audit"})
FROZEN_VOCABULARY_CONTRACT = frozen_text_builder.FROZEN_VOCABULARY_CONTRACT
FROZEN_SNAPSHOT_FILES_SHA256 = frozen_text_builder.FROZEN_SNAPSHOT_FILES_SHA256
FROZEN_HOLDOUT_VOCABULARY_CONTRACT = (
    holdout_text_builder.FROZEN_HOLDOUT_CONTRACT
)

# The formal held-out banks predate the builder's fail-closed HuggingFace blob
# resolver.  They remain admissible only as this exact immutable pair; this is
# deliberately not a generic old-builder compatibility path.
FORMAL_HISTORICAL_TEXT_BANKS = {
    "dev": {
        "artifact_path": "/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260731/target_blind_siglip2_text_bank_v1/target_blind_siglip2_dev_embeddings.pt",
        "artifact_sha256": "37c8d1f160b3ad69b5d6372c40dcc6207bca5fb9ef0143e139965e95e7beceb4",
        "manifest_path": "/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260731/target_blind_siglip2_text_bank_v1/target_blind_siglip2_dev_embeddings.manifest.json",
        "manifest_sha256": "50335f0f7f1a0f47388b600844bbfeba9b6a8a8290f3f74a88d3814b13b671d3",
    },
    "audit": {
        "artifact_path": "/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260731/target_blind_siglip2_text_bank_v1/target_blind_siglip2_audit_embeddings.pt",
        "artifact_sha256": "46dd338340a310e2b59997d1b6ea4882590c76f8aca389d4aa0abc2b3c5c2721",
        "manifest_path": "/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260731/target_blind_siglip2_text_bank_v1/target_blind_siglip2_audit_embeddings.manifest.json",
        "manifest_sha256": "1f19257393b45d713fb80be707e962a9680b84b793328a7364ba78bdd57b46b4",
    },
}
FORMAL_HISTORICAL_BUILDER = {
    "path": "/root/RADIO-GS/radio_gs/scripts/build_target_blind_siglip2_embedding_artifact.py",
    "sha256": "e8e815b5f15796c21205788769a48d1bef95e21b9eac4c2777cf6754b424d136",
}
FORMAL_IMAGENET12K_HOLDOUT_TEXT_BANKS = {
    "dev": {
        "artifact_path": "/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260801/target_blind_imagenet12k_minus1k_siglip2_holdout_dev_v2/target_blind_siglip2_dev_embeddings.pt",
        "artifact_sha256": "4496ac3cbdb95472c69a3cf315f202b35d274964d59be8e1f89419afad252481",
        "manifest_path": "/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260801/target_blind_imagenet12k_minus1k_siglip2_holdout_dev_v2/target_blind_siglip2_dev_embeddings.manifest.json",
        "manifest_sha256": "e7afd43c6700b4893196ef2ed6811db2a50368a5d7b225fb813f3a6eec50587d",
    },
    "audit": {
        "artifact_path": "/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260801/target_blind_imagenet12k_minus1k_siglip2_holdout_audit_v1/target_blind_siglip2_audit_embeddings.pt",
        "artifact_sha256": "b681d20e4d5096a11ea9971d9049195bcb4ffd43a78d6138e350f3e0727baf74",
        "manifest_path": "/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260801/target_blind_imagenet12k_minus1k_siglip2_holdout_audit_v1/target_blind_siglip2_audit_embeddings.manifest.json",
        "manifest_sha256": "090683cc42e087f91b993aa83ed06d50194436428401b67c18d3e15c833a99d6",
    },
}
FORMAL_IMAGENET12K_HOLDOUT_BUILDER = {
    "path": "/root/RADIO-GS/radio_gs/scripts/build_target_blind_imagenet12k_holdout_embedding.py",
    "sha256": "bedc7e1ed736c61a205890b44403f6dc60fcf2322930e194481b927309a37ffb",
}


def classify_formal_text_bank_pair(
    artifact_path: Path,
    manifest_path: Path,
    query_split: str,
    *,
    _hash_cache: dict[Path, str] | None = None,
) -> str:
    """Classify only an exact registered formal artifact/sidecar pair."""

    if query_split not in ALLOWED_HELDOUT_SPLITS:
        raise ValueError("formal text-bank split must be dev or audit")
    artifact_path = Path(artifact_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    registries = (
        (IMAGENET1K_PRIMARY_BANK_FAMILY, FORMAL_HISTORICAL_TEXT_BANKS),
        (IMAGENET12K_HOLDOUT_BANK_FAMILY, FORMAL_IMAGENET12K_HOLDOUT_TEXT_BANKS),
    )
    for family, registry in registries:
        expected = registry[query_split]
        if (
            artifact_path == Path(expected["artifact_path"]).resolve()
            and manifest_path == Path(expected["manifest_path"]).resolve()
            and _sha256_file_cached(artifact_path, _hash_cache)
            == expected["artifact_sha256"]
            and _sha256_file_cached(manifest_path, _hash_cache)
            == expected["manifest_sha256"]
        ):
            return family
    raise ValueError(
        f"unregistered or changed formal {query_split} text-bank pair"
    )

_DESCRIPTOR_PAYLOAD_KEYS = {
    "schema_version",
    "artifact_type",
    "method_id",
    "seed",
    "split_role",
    "student_descriptors",
    "teacher_descriptors",
    "scene_ids",
    "region_ids",
    "student_descriptors_sha256",
    "teacher_descriptors_sha256",
    "descriptor_rows_sha256",
    "descriptor_space",
    "provenance",
}
_DESCRIPTOR_PROVENANCE_KEYS = {
    "uses_benchmark_scenes",
    "uses_benchmark_test_vocabulary",
    "annotations_opened",
    "labels_opened",
    "instances_opened",
    "masks_opened",
    "text_opened",
    "device",
    "readout_checkpoint",
    "readout_checkpoint_sha256",
    "readout_report",
    "readout_report_sha256",
    "readout_binding_authority",
    "radio_checkpoint",
    "radio_checkpoint_sha256",
    "region_contract_sha256",
    "validation_split_sha256",
    "validation_scenes",
    "teacher_region",
    "validation_caches",
}
_CACHE_BINDING_KEYS = {
    "path",
    "sha256",
    "rows",
    "split_file_sha256",
    "region_contract_sha256",
    "radio_checkpoint_sha256",
    "teacher_target_protocol_sha256",
}
_QUERY_FREE_FLAGS = (
    "uses_benchmark_scenes",
    "uses_benchmark_test_vocabulary",
    "annotations_opened",
    "labels_opened",
    "instances_opened",
    "masks_opened",
    "text_opened",
)


def _sha256_file(path: Path) -> str:
    return sha256_file(path)


def _sha256_file_cached(
    path: Path,
    cache: dict[Path, str] | None,
) -> str:
    path = Path(path).resolve()
    if cache is None:
        return _sha256_file(path)
    digest = cache.get(path)
    if digest is None:
        digest = _sha256_file(path)
        cache[path] = digest
    return digest


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _read_json(path: Path) -> Mapping:
    value, _, _ = load_json_object(path, label="text-fidelity JSON artifact")
    return value


def _resolve_bound_path(raw: object, *, relative_to: Path, name: str) -> Path:
    value = Path(str(raw))
    path = value if value.is_absolute() else relative_to / value
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"bound {name} does not exist: {path}")
    return path


def _verify_hash(value: object, expected: object, name: str) -> None:
    if not _is_sha256(expected) or str(value) != str(expected):
        raise ValueError(f"{name} SHA256 mismatch")


def _torch_load(path: Path) -> Mapping:
    value, _, _ = load_torch_mapping(
        path,
        map_location="cpu",
        label="text-fidelity torch artifact",
    )
    return value


def _legacy_region_id(record: Mapping) -> str:
    identity = {
        "scene": record.get("scene"),
        "seed": record.get("seed"),
        "physical_radius_m": record.get("physical_radius_m"),
        "teacher_views": record.get("teacher_views"),
        "teacher_target_sha256": record.get("teacher_target_sha256"),
        "teacher_support_sha256": record.get("teacher_support_sha256"),
    }
    if not str(identity["scene"] or "") or identity["seed"] is None:
        raise ValueError("legacy cache region lacks a stable scene/seed identity")
    return "legacy-" + canonical_json_sha256(identity)


def _verify_bound_file(
    raw_path: object,
    expected_sha256: object,
    *,
    relative_to: Path,
    name: str,
    hash_cache: dict[Path, str] | None,
) -> Path:
    path = _resolve_bound_path(raw_path, relative_to=relative_to, name=name)
    _verify_hash(
        _sha256_file_cached(path, hash_cache),
        expected_sha256,
        name,
    )
    return path


def _validate_binding_authority(
    value: object,
    *,
    relative_to: Path,
    hash_cache: dict[Path, str] | None,
) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError("descriptor readout_binding_authority must be a mapping")
    authority = dict(value)
    authority_type = authority.get("type")
    if authority_type == "embedded_distill_run_manifest":
        expected_keys = {"type", "path", "sha256", "candidate"}
        if set(authority) != expected_keys or not str(authority.get("candidate", "")):
            raise ValueError("distill authority fields differ from the frozen schema")
        _verify_bound_file(
            authority["path"],
            authority["sha256"],
            relative_to=relative_to,
            name="distill run manifest",
            hash_cache=hash_cache,
        )
    elif authority_type in {
        "query_free_promotion_bundle",
        "attention_postcache_screen",
    }:
        expected_keys = {
            "type",
            "path",
            "sha256",
            "completion",
            "completion_sha256",
            "candidate",
        }
        if set(authority) != expected_keys or not str(authority.get("candidate", "")):
            raise ValueError("Surface authority fields differ from the frozen schema")
        label = (
            "attention-postcache screen"
            if authority_type == "attention_postcache_screen"
            else "query-free promotion bundle"
        )
        _verify_bound_file(
            authority["path"],
            authority["sha256"],
            relative_to=relative_to,
            name=label,
            hash_cache=hash_cache,
        )
        _verify_bound_file(
            authority["completion"],
            authority["completion_sha256"],
            relative_to=relative_to,
            name=f"{label} completion",
            hash_cache=hash_cache,
        )
    else:
        raise ValueError("descriptor has an unsupported readout binding authority")
    return authority


def _validate_descriptor_caches(
    value: object,
    *,
    descriptor_parent: Path,
    expected_scene_ids: list[str],
    expected_region_ids: list[str],
    expected_scenes: list[str],
    expected_split_sha256: str,
    expected_region_contract_sha256: str,
    expected_radio_sha256: str,
    expected_teacher_region: object,
    hash_cache: dict[Path, str] | None,
) -> list[dict]:
    if not isinstance(value, list) or not value:
        raise ValueError("descriptor provenance requires validation cache bindings")
    bindings: list[dict] = []
    rebound_scene_ids: list[str] = []
    rebound_region_ids: list[str] = []
    resolved_paths: list[Path] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != _CACHE_BINDING_KEYS:
            raise ValueError("validation cache binding fields differ from the frozen schema")
        binding = dict(raw)
        rows = binding.get("rows")
        if not isinstance(rows, int) or rows <= 0:
            raise ValueError("validation cache rows must be a positive integer")
        cache_path = _verify_bound_file(
            binding["path"],
            binding["sha256"],
            relative_to=descriptor_parent,
            name=f"validation cache {index}",
            hash_cache=hash_cache,
        )
        resolved_paths.append(cache_path)
        cache = _torch_load(cache_path)
        metadata = cache.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("validation cache lacks metadata")
        if metadata.get("schema_version") != 3 or metadata.get("split_role") != "validation":
            raise ValueError("descriptor binds a non-validation schema-v3 cache")
        for flag in _QUERY_FREE_FLAGS:
            if metadata.get(flag) is not False:
                raise ValueError(f"validation cache must certify {flag}=false")
        records = metadata.get("region_records")
        if not isinstance(records, list) or len(records) != rows:
            raise ValueError("validation cache row binding differs from region_records")
        if (
            binding.get("split_file_sha256") != expected_split_sha256
            or metadata.get("split_file_sha256") != expected_split_sha256
            or binding.get("region_contract_sha256")
            != expected_region_contract_sha256
            or metadata.get("region_contract_sha256")
            != expected_region_contract_sha256
            or binding.get("radio_checkpoint_sha256") != expected_radio_sha256
            or metadata.get("radio_checkpoint_sha256") != expected_radio_sha256
            or binding.get("teacher_target_protocol_sha256")
            != metadata.get("teacher_target_protocol_sha256")
        ):
            raise ValueError("validation cache contract hashes differ from descriptor provenance")
        local_scene_ids: list[str] = []
        local_region_ids: list[str] = []
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError("validation cache contains a non-object region record")
            scene = str(record.get("scene", ""))
            region = str(record.get("region_id", "")) or _legacy_region_id(record)
            if not scene:
                raise ValueError("validation cache contains an empty scene identity")
            local_scene_ids.append(scene)
            local_region_ids.append(region)
        local_scenes = sorted(set(local_scene_ids))
        local_counts = {
            scene: sum(value == scene for value in local_scene_ids) for scene in local_scenes
        }
        if (
            metadata.get("scene_names") != local_scenes
            or metadata.get("scene_region_counts") != local_counts
        ):
            raise ValueError("validation cache scene metadata is inconsistent")
        teacher_region = {
            "semantics": metadata.get("teacher_region_semantics"),
            "contract": metadata.get("teacher_region_contract"),
            "contract_sha256": metadata.get("teacher_region_contract_sha256"),
            "target_source": metadata.get("teacher_target_source"),
            "target_protocol_sha256": metadata.get("teacher_target_protocol_sha256"),
        }
        if teacher_region != expected_teacher_region:
            raise ValueError("validation cache teacher-region contract differs")
        rebound_scene_ids.extend(local_scene_ids)
        rebound_region_ids.extend(local_region_ids)
        bindings.append(binding)
    if resolved_paths != sorted(resolved_paths):
        raise ValueError("validation cache bindings must use canonical sorted path order")
    if rebound_scene_ids != expected_scene_ids or rebound_region_ids != expected_region_ids:
        raise ValueError("descriptor rows do not exactly replay validation cache identities")
    if sorted(set(rebound_scene_ids)) != expected_scenes:
        raise ValueError("descriptor validation scene set differs from its caches")
    return bindings


def load_descriptor_pair(
    path: Path,
    *,
    _hash_cache: dict[Path, str] | None = None,
) -> dict:
    """Load a strict, already-materialized student/teacher descriptor pair."""

    path = Path(path).resolve()
    payload = _torch_load(path)
    if (
        set(payload) != _DESCRIPTOR_PAYLOAD_KEYS
        or
        payload.get("schema_version") != DESCRIPTOR_SCHEMA_VERSION
        or payload.get("artifact_type") != DESCRIPTOR_ARTIFACT_TYPE
    ):
        raise ValueError("invalid surface text-response descriptor schema")
    method_id = str(payload.get("method_id", ""))
    seed = payload.get("seed")
    if not method_id:
        raise ValueError("descriptor artifact requires method_id")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("descriptor artifact requires a non-negative integer seed")
    if payload.get("split_role") != "validation":
        raise ValueError("promotion descriptors must use the query-free validation split")
    provenance = payload.get("provenance", {})
    if not isinstance(provenance, Mapping) or set(provenance) != _DESCRIPTOR_PROVENANCE_KEYS:
        raise ValueError("descriptor provenance must be a mapping")
    for key in _QUERY_FREE_FLAGS:
        if provenance.get(key) is not False:
            raise ValueError(f"descriptor provenance must explicitly certify {key}=false")
    if provenance.get("device") != "cpu":
        raise ValueError("descriptor provenance must bind device=cpu")

    student = torch.as_tensor(payload.get("student_descriptors"))
    teacher = torch.as_tensor(payload.get("teacher_descriptors"))
    if student.device.type != "cpu" or teacher.device.type != "cpu":
        raise ValueError("descriptor tensors must remain on CPU")
    if (
        student.ndim != 2
        or teacher.ndim != 2
        or student.shape != teacher.shape
        or student.shape[0] == 0
        or student.shape[1] == 0
        or not student.is_floating_point()
        or not teacher.is_floating_point()
    ):
        raise ValueError("student/teacher descriptors must be aligned floating [N,D] tensors")
    if not bool(torch.isfinite(student).all()) or not bool(torch.isfinite(teacher).all()):
        raise ValueError("descriptor tensors must be finite")
    _verify_hash(
        tensor_sha256(student),
        payload.get("student_descriptors_sha256"),
        "student descriptor tensor",
    )
    _verify_hash(
        tensor_sha256(teacher),
        payload.get("teacher_descriptors_sha256"),
        "teacher descriptor tensor",
    )
    scene_ids = payload.get("scene_ids")
    region_ids = payload.get("region_ids")
    if not isinstance(scene_ids, list) or not isinstance(region_ids, list):
        raise ValueError("descriptor scene_ids/region_ids must be lists")
    if len(scene_ids) != student.shape[0] or len(region_ids) != student.shape[0]:
        raise ValueError("descriptor row identities are misaligned")
    rows_hash = row_identity_sha256(scene_ids, region_ids)
    _verify_hash(rows_hash, payload.get("descriptor_rows_sha256"), "descriptor rows")
    descriptor_space = payload.get("descriptor_space", {})
    if (
        not isinstance(descriptor_space, Mapping)
        or set(descriptor_space)
        != {"name", "dimension", "normalization", "official_summary_head"}
        or descriptor_space.get("name") != "official_siglip2_g_summary"
        or descriptor_space.get("normalization") != "l2"
        or descriptor_space.get("dimension") != student.shape[1]
        or descriptor_space.get("official_summary_head")
        != "c-radio_v4 _heads.siglip2-g"
    ):
        raise ValueError("descriptor space must be normalized official SigLIP2-g summary space")
    for name, tensor in (("student", student), ("teacher", teacher)):
        norms = torch.linalg.vector_norm(tensor.float(), dim=-1)
        if not bool(
            torch.allclose(norms, torch.ones_like(norms), atol=5e-5, rtol=5e-5)
        ):
            raise ValueError(f"{name} descriptor rows must be L2-normalized")

    descriptor_parent = path.parent
    readout_checkpoint = _verify_bound_file(
        provenance.get("readout_checkpoint"),
        provenance.get("readout_checkpoint_sha256"),
        relative_to=descriptor_parent,
        name="readout checkpoint",
        hash_cache=_hash_cache,
    )
    readout_report = _verify_bound_file(
        provenance.get("readout_report"),
        provenance.get("readout_report_sha256"),
        relative_to=descriptor_parent,
        name="readout report",
        hash_cache=_hash_cache,
    )
    radio_checkpoint = _verify_bound_file(
        provenance.get("radio_checkpoint"),
        provenance.get("radio_checkpoint_sha256"),
        relative_to=descriptor_parent,
        name="RADIO checkpoint",
        hash_cache=_hash_cache,
    )
    authority = _validate_binding_authority(
        provenance.get("readout_binding_authority"),
        relative_to=descriptor_parent,
        hash_cache=_hash_cache,
    )
    validation_scenes = provenance.get("validation_scenes")
    if (
        not isinstance(validation_scenes, list)
        or len(validation_scenes) < 2
        or validation_scenes != sorted(set(str(value) for value in validation_scenes))
    ):
        raise ValueError("descriptor must bind at least two unique sorted validation scenes")
    for name in (
        "region_contract_sha256",
        "validation_split_sha256",
        "radio_checkpoint_sha256",
    ):
        if not _is_sha256(provenance.get(name)):
            raise ValueError(f"descriptor provenance has an invalid {name}")
    cache_bindings = _validate_descriptor_caches(
        provenance.get("validation_caches"),
        descriptor_parent=descriptor_parent,
        expected_scene_ids=[str(value) for value in scene_ids],
        expected_region_ids=[str(value) for value in region_ids],
        expected_scenes=[str(value) for value in validation_scenes],
        expected_split_sha256=str(provenance["validation_split_sha256"]),
        expected_region_contract_sha256=str(provenance["region_contract_sha256"]),
        expected_radio_sha256=str(provenance["radio_checkpoint_sha256"]),
        expected_teacher_region=provenance.get("teacher_region"),
        hash_cache=_hash_cache,
    )
    return {
        "path": path,
        "file_sha256": _sha256_file_cached(path, _hash_cache),
        "payload": payload,
        "method_id": method_id,
        "seed": int(seed),
        "student": student.float(),
        "teacher": teacher.float(),
        "scene_ids": [str(value) for value in scene_ids],
        "region_ids": [str(value) for value in region_ids],
        "rows_sha256": rows_hash,
        "teacher_sha256": tensor_sha256(teacher),
        "provenance_bindings": {
            "readout_checkpoint": str(readout_checkpoint),
            "readout_report": str(readout_report),
            "radio_checkpoint": str(radio_checkpoint),
            "readout_binding_authority": authority,
            "validation_caches": cache_bindings,
            "validation_scenes": [str(value) for value in validation_scenes],
        },
    }


def _records_sha256(records: list[Mapping]) -> str:
    return canonical_json_sha256(
        [
            {
                "synset": str(record.get("synset", "")),
                "query": str(record.get("query", "")),
                "split": str(record.get("split", "")),
            }
            for record in records
        ]
    )


def _split_sha256(records: list[Mapping], split: str) -> str:
    lines = "".join(
        f"{record['synset']}\t{record['query']}\n"
        for record in records
        if record["split"] == split
    )
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def _embedding_semantic_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().to(torch.float32).contiguous()
    array = tensor.numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _safe_snapshot_file(snapshot: Path, name: object) -> Path:
    relative = Path(str(name))
    if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
        raise ValueError(f"text encoder snapshot contains unsafe filename {name!r}")
    # Reuse the builder's fixed-model, one-hop HuggingFace blob resolver so
    # immutable reads still open the real regular file with O_NOFOLLOW.
    return frozen_text_builder._safe_snapshot_file(snapshot, str(relative))


def _validate_text_encoder(
    value: object,
    *,
    expected_files_sha256: Mapping[str, str] = FROZEN_SNAPSHOT_FILES_SHA256,
    hash_cache: dict[Path, str] | None = None,
) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError("text_encoder provenance must be a mapping")
    expected_keys = {
        "model_id",
        "revision",
        "snapshot_path",
        "snapshot_files_sha256",
        "config_sha256",
        "tokenizer_sha256",
        "tokenizer_files_sha256",
        "auxiliary_files_sha256",
        "model_index_sha256",
        "weight_shards_sha256",
        "output_dimension",
        "dtype",
        "normalization",
        "device",
    }
    if set(value) != expected_keys:
        raise ValueError("text_encoder fields differ from the frozen split-v1 schema")
    if (
        value.get("model_id") != TEXT_BANK_MODEL_ID
        or value.get("revision") != TEXT_BANK_MODEL_REVISION
        or value.get("output_dimension") != TEXT_BANK_OUTPUT_DIMENSION
        or value.get("dtype") != "float32"
        or value.get("normalization") != "l2"
        or value.get("device") != "cpu"
    ):
        raise ValueError("text_encoder does not match the frozen CPU SigLIP2 contract")
    snapshot = Path(str(value.get("snapshot_path", ""))).resolve()
    if not snapshot.is_dir() or snapshot.name != TEXT_BANK_MODEL_REVISION:
        raise ValueError("text_encoder snapshot path does not bind the fixed revision")
    tokenizer_files = value.get("tokenizer_files_sha256")
    auxiliary_files = value.get("auxiliary_files_sha256")
    weight_shards = value.get("weight_shards_sha256")
    if (
        not isinstance(tokenizer_files, Mapping)
        or set(tokenizer_files) != TEXT_BANK_TOKENIZER_FILES
        or not isinstance(auxiliary_files, Mapping)
        or set(auxiliary_files) != set(frozen_text_builder.SNAPSHOT_AUXILIARY_FILES)
        or not isinstance(weight_shards, Mapping)
        or not weight_shards
    ):
        raise ValueError("text_encoder file indexes differ from the frozen schema")
    config_path = _safe_snapshot_file(snapshot, "config.json")
    config = _read_json(config_path)
    text_config = config.get("text_config")
    if (
        config.get("model_type") != "siglip"
        or not isinstance(text_config, Mapping)
        or text_config.get("projection_size") != TEXT_BANK_OUTPUT_DIMENSION
        or not isinstance(text_config.get("hidden_size"), int)
        or int(text_config["hidden_size"]) <= 0
    ):
        raise ValueError("text_encoder config is not the frozen 1536-D SigLIP2 model")
    index_path = _safe_snapshot_file(snapshot, "model.safetensors.index.json")
    index = _read_json(index_path)
    weight_map = index.get("weight_map")
    required_weight_keys = (
        "text_model.head.weight",
        "text_model.head.bias",
        "text_model.embeddings.token_embedding.weight",
    )
    if (
        not isinstance(weight_map, Mapping)
        or not weight_map
        or any(not str(weight_map.get(key, "")) for key in required_weight_keys)
    ):
        raise ValueError("text_encoder model index lacks required text weights")
    indexed_shards = {str(name) for name in weight_map.values()}
    if set(weight_shards) != indexed_shards:
        raise ValueError("text_encoder shard provenance does not cover its model index")
    declared_files = {
        "config.json": value.get("config_sha256"),
        "model.safetensors.index.json": value.get("model_index_sha256"),
        **{str(name): digest for name, digest in tokenizer_files.items()},
        **{str(name): digest for name, digest in auxiliary_files.items()},
        **{str(name): digest for name, digest in weight_shards.items()},
    }
    if declared_files != dict(expected_files_sha256):
        raise ValueError(
            "text encoder files do not match the frozen official snapshot digests"
        )
    for name, expected in declared_files.items():
        path = _safe_snapshot_file(snapshot, name)
        _verify_hash(
            _sha256_file_cached(path, hash_cache),
            expected,
            f"text encoder file {name}",
        )
    _verify_hash(
        canonical_json_sha256(dict(tokenizer_files)),
        value.get("tokenizer_sha256"),
        "text encoder tokenizer index",
    )
    _verify_hash(
        canonical_json_sha256(declared_files),
        value.get("snapshot_files_sha256"),
        "text encoder snapshot index",
    )
    return dict(value)


def _canonical_vocabulary_records(raw_records: object) -> list[dict[str, str]]:
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("target-blind vocabulary records are empty")
    records = []
    for raw in raw_records:
        if not isinstance(raw, Mapping):
            raise ValueError("target-blind vocabulary records must be objects")
        record = {
            "synset": str(raw.get("synset", "")),
            "query": str(raw.get("query", "")),
            "split": str(raw.get("split", "")),
        }
        if (
            not record["synset"]
            or not record["query"]
            or record["split"] not in {"fit", "dev", "audit"}
        ):
            raise ValueError("target-blind vocabulary contains an invalid record")
        canonical_query = record["query"].replace("_", " ")
        canonical_query = canonical_query.translate(
            str.maketrans("", "", string.punctuation)
        ).lower()
        canonical_query = " ".join(canonical_query.split()).strip()
        if canonical_query != record["query"]:
            raise ValueError("target-blind vocabulary query violates text canonicalization")
        records.append(record)
    if len({record["synset"] for record in records}) != len(records):
        raise ValueError("target-blind vocabulary synsets must be unique")
    if len({record["query"] for record in records}) != len(records):
        raise ValueError("target-blind vocabulary queries must be deduplicated")
    return records


def _validate_embedding_builder_provenance(
    *,
    builder_path: Path,
    builder_sha256: object,
    artifact_path: Path,
    artifact_sha256: str,
    manifest_path: Path,
    manifest_sha256: str,
    query_split: str,
    algorithm_version: str,
    hash_cache: dict[Path, str] | None,
    test_contract_injected: bool = False,
) -> None:
    current_builder = Path(__file__).resolve().with_name(
        "build_target_blind_siglip2_embedding_artifact.py"
    )
    declared_sha = str(builder_sha256)
    if (
        builder_path == current_builder
        and declared_sha == _sha256_file_cached(current_builder, hash_cache)
    ):
        return
    historical = FORMAL_HISTORICAL_TEXT_BANKS.get(query_split)
    if (
        isinstance(historical, Mapping)
        and str(builder_path) == FORMAL_HISTORICAL_BUILDER["path"]
        and declared_sha == FORMAL_HISTORICAL_BUILDER["sha256"]
        and str(artifact_path) == historical.get("artifact_path")
        and artifact_sha256 == historical.get("artifact_sha256")
        and str(manifest_path) == historical.get("manifest_path")
        and manifest_sha256 == historical.get("manifest_sha256")
    ):
        return
    formal_holdout = FORMAL_IMAGENET12K_HOLDOUT_TEXT_BANKS.get(query_split)
    if (
        algorithm_version == HOLDOUT_TEXT_BANK_ALGORITHM_VERSION
        and isinstance(formal_holdout, Mapping)
        and str(builder_path) == FORMAL_IMAGENET12K_HOLDOUT_BUILDER["path"]
        and declared_sha == FORMAL_IMAGENET12K_HOLDOUT_BUILDER["sha256"]
        and str(artifact_path) == formal_holdout.get("artifact_path")
        and artifact_sha256 == formal_holdout.get("artifact_sha256")
        and str(manifest_path) == formal_holdout.get("manifest_path")
        and manifest_sha256 == formal_holdout.get("manifest_sha256")
    ):
        return
    current_holdout_builder = Path(__file__).resolve().with_name(
        "build_target_blind_imagenet12k_holdout_embedding.py"
    )
    if (
        test_contract_injected
        and algorithm_version == HOLDOUT_TEXT_BANK_ALGORITHM_VERSION
        and builder_path == current_holdout_builder
        and declared_sha
        == _sha256_file_cached(current_holdout_builder, hash_cache)
    ):
        return
    raise ValueError("embedding sidecar names an unexpected or changed builder")


def load_text_embedding_bank(
    path: Path,
    sidecar_path: Path,
    query_split: str,
    *,
    _hash_cache: dict[Path, str] | None = None,
    _test_vocabulary_contract: Mapping[str, object] | None = None,
    _test_snapshot_files_sha256: Mapping[str, str] | None = None,
) -> dict:
    """Load one frozen held-out embedding split and verify its sidecar."""

    if query_split not in ALLOWED_HELDOUT_SPLITS:
        raise ValueError(
            f"query_split must be a held-out split in {sorted(ALLOWED_HELDOUT_SPLITS)}"
        )
    path = Path(path).resolve()
    sidecar_path = Path(sidecar_path).resolve()
    payload = _torch_load(path)
    sidecar = _read_json(sidecar_path)
    expected_payload_keys = {
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
    expected_sidecar_keys = {
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
    algorithm_version = payload.get("algorithm_version")
    is_imagenet12k_holdout = (
        algorithm_version == HOLDOUT_TEXT_BANK_ALGORITHM_VERSION
    )
    if (
        set(payload) != expected_payload_keys
        or payload.get("schema_version") != TEXT_BANK_SCHEMA_VERSION
        or payload.get("artifact_type") != TEXT_BANK_ARTIFACT_TYPE
        or algorithm_version
        not in {TEXT_BANK_ALGORITHM_VERSION, HOLDOUT_TEXT_BANK_ALGORITHM_VERSION}
        or set(sidecar) != expected_sidecar_keys
        or sidecar.get("schema_version") != TEXT_BANK_SCHEMA_VERSION
        or sidecar.get("artifact_type") != TEXT_BANK_MANIFEST_ARTIFACT_TYPE
        or sidecar.get("algorithm_version") != algorithm_version
    ):
        raise ValueError("invalid target-blind text embedding cache schema")
    if (
        payload.get("benchmark_vocabulary_opened") is not False
        or sidecar.get("benchmark_vocabulary_opened") is not False
    ):
        raise ValueError("text embedding artifacts must certify benchmark_vocabulary_opened=false")
    if (
        payload.get("uses_benchmark_vocabulary_for_construction") is not False
        or sidecar.get("uses_benchmark_vocabulary_for_construction") is not False
    ):
        raise ValueError(
            "text embedding artifacts must certify "
            "uses_benchmark_vocabulary_for_construction=false"
        )
    if (
        payload.get("split") != query_split
        or sidecar.get("split") != query_split
    ):
        raise ValueError("requested query_split differs from the frozen embedding split")
    if (
        payload.get("prompt_templates") != ["{query}"]
        or sidecar.get("prompt_templates") != ["{query}"]
        or payload.get("text_canonicalization") != TEXT_BANK_CANONICALIZATION
        or sidecar.get("text_canonicalization") != TEXT_BANK_CANONICALIZATION
    ):
        raise ValueError("text embedding artifacts violate the frozen text policy")

    root = path.parent
    vocabulary_path = _resolve_bound_path(
        payload.get("vocabulary_path"), relative_to=root, name="vocabulary"
    )
    vocabulary_manifest_path = _resolve_bound_path(
        payload.get("vocabulary_manifest_path"),
        relative_to=root,
        name="vocabulary manifest",
    )
    vocabulary_sha = _sha256_file_cached(vocabulary_path, _hash_cache)
    vocabulary_manifest_sha = _sha256_file_cached(
        vocabulary_manifest_path, _hash_cache
    )
    _verify_hash(vocabulary_sha, payload.get("vocabulary_sha256"), "vocabulary")
    _verify_hash(
        vocabulary_manifest_sha,
        payload.get("vocabulary_manifest_sha256"),
        "vocabulary manifest",
    )
    vocabulary = _read_json(vocabulary_path)
    vocabulary_manifest = _read_json(vocabulary_manifest_path)
    frozen_vocabulary = (
        (
            FROZEN_HOLDOUT_VOCABULARY_CONTRACT
            if is_imagenet12k_holdout
            else FROZEN_VOCABULARY_CONTRACT
        )
        if _test_vocabulary_contract is None
        else _test_vocabulary_contract
    )
    if is_imagenet12k_holdout:
        if (
            vocabulary.get("schema_version") != 1
            or vocabulary.get("artifact_type")
            != holdout_vocabulary_builder.ARTIFACT_TYPE
            or vocabulary.get("algorithm_version")
            != holdout_vocabulary_builder.ALGORITHM_VERSION
            or vocabulary.get("split") != query_split
            or vocabulary.get("benchmark_vocabulary_opened") is not False
            or vocabulary.get("uses_benchmark_vocabulary_for_construction")
            is not False
            or vocabulary.get("prompt_templates") != ["{query}"]
        ):
            raise ValueError("bound vocabulary is not the frozen ImageNet12K holdout")
        if (
            vocabulary_manifest.get("schema_version") != 1
            or vocabulary_manifest.get("artifact_type")
            != holdout_vocabulary_builder.MANIFEST_ARTIFACT_TYPE
            or vocabulary_manifest.get("algorithm_version")
            != holdout_vocabulary_builder.ALGORITHM_VERSION
            or vocabulary_manifest.get("benchmark_vocabulary_opened") is not False
            or vocabulary_manifest.get(
                "uses_benchmark_vocabulary_for_construction"
            )
            is not False
        ):
            raise ValueError("bound ImageNet12K holdout manifest schema differs")
        split_contract = frozen_vocabulary.get("splits", {}).get(query_split)
        if not isinstance(split_contract, Mapping):
            raise ValueError("frozen ImageNet12K holdout lacks the requested split")
        _verify_hash(
            vocabulary_manifest_sha,
            frozen_vocabulary.get("manifest_sha256"),
            "ImageNet12K holdout manifest",
        )
        _verify_hash(
            vocabulary_sha,
            split_contract.get("vocabulary_sha256"),
            "ImageNet12K holdout vocabulary",
        )
        manifest_artifact = vocabulary_manifest.get("artifacts", {}).get(query_split)
        if (
            not isinstance(manifest_artifact, Mapping)
            or manifest_artifact.get("sha256") != vocabulary_sha
        ):
            raise ValueError("ImageNet12K holdout manifest binds another vocabulary")
        canonical_records = _canonical_vocabulary_records(vocabulary.get("records"))
        selected_records = canonical_records
        if (
            len(selected_records) != int(split_contract.get("records", -1))
            or any(record["split"] != query_split for record in selected_records)
        ):
            raise ValueError("ImageNet12K holdout split record count differs")
        split_sha = _split_sha256(selected_records, query_split)
        _verify_hash(
            split_sha,
            split_contract.get("record_sha256"),
            "ImageNet12K holdout split",
        )
        _verify_hash(
            split_sha,
            vocabulary_manifest.get("synset_tab_query_lf_sha256", {}).get(
                query_split
            ),
            "ImageNet12K holdout manifest split",
        )
        if _test_vocabulary_contract is None:
            if (
                vocabulary_manifest.get("counts")
                != dict(holdout_vocabulary_builder.EXPECTED_COUNTS)
                or vocabulary_manifest.get("synset_tab_query_lf_sha256")
                != dict(holdout_vocabulary_builder.EXPECTED_RECORD_SHA256)
            ):
                raise ValueError("production ImageNet12K holdout contract differs")
            sources = vocabulary_manifest.get("sources")
            if not isinstance(sources, Mapping) or set(sources) != set(
                holdout_vocabulary_builder.EXPECTED_SOURCE_SHA256
            ):
                raise ValueError("ImageNet12K holdout source index differs")
            for name, expected_sha in (
                holdout_vocabulary_builder.EXPECTED_SOURCE_SHA256.items()
            ):
                record = sources[name]
                if not isinstance(record, Mapping):
                    raise ValueError("ImageNet12K holdout source record differs")
                source_path = _resolve_bound_path(
                    record.get("path"),
                    relative_to=vocabulary_manifest_path.parent,
                    name=f"ImageNet12K source {name}",
                )
                _verify_hash(
                    _sha256_file_cached(source_path, _hash_cache),
                    expected_sha,
                    f"ImageNet12K source {name}",
                )
    else:
        if (
            vocabulary.get("schema_version") != 1
            or vocabulary.get("artifact_type")
            != "target_blind_imagenet1k_primary_text_bank"
            or vocabulary.get("algorithm_version") != "imagenet1k-primary-v1"
            or vocabulary.get("benchmark_vocabulary_opened") is not False
            or vocabulary.get("prompt_templates") != ["{query}"]
        ):
            raise ValueError("bound vocabulary is not the target-blind primary bank v1")
        if (
            vocabulary_manifest.get("schema_version") != 1
            or vocabulary_manifest.get("artifact_type")
            != "target_blind_imagenet1k_primary_text_bank_manifest"
            or vocabulary_manifest.get("algorithm_version") != "imagenet1k-primary-v1"
            or vocabulary_manifest.get("benchmark_vocabulary_opened") is not False
            or vocabulary_manifest.get("canonical_json", {}).get("sha256")
            != vocabulary_sha
        ):
            raise ValueError(
                "bound vocabulary manifest does not certify the canonical bank"
            )
        canonical_records = _canonical_vocabulary_records(vocabulary.get("records"))
        if vocabulary_sha != frozen_vocabulary.get("canonical_vocabulary_sha256"):
            raise ValueError(
                "bound vocabulary is self-consistent but not the frozen target-blind bank"
            )
        selected_records = [
            record for record in canonical_records if record["split"] == query_split
        ]
        if not selected_records:
            raise ValueError(f"target-blind vocabulary split {query_split} is empty")
        declared_split_hashes = vocabulary_manifest.get(
            "split_synset_tab_query_lf_sha256", {}
        )
        if not isinstance(declared_split_hashes, Mapping) or set(
            declared_split_hashes
        ) != {"fit", "dev", "audit"}:
            raise ValueError("vocabulary manifest has an incomplete split hash index")
        computed_split_hashes = {
            split: _split_sha256(canonical_records, split)
            for split in ("fit", "dev", "audit")
        }
        for split, digest in computed_split_hashes.items():
            _verify_hash(
                digest,
                declared_split_hashes.get(split),
                f"{split} vocabulary split",
            )
        expected_split_hashes = frozen_vocabulary.get("split_sha256")
        if not isinstance(expected_split_hashes, Mapping) or computed_split_hashes != dict(
            expected_split_hashes
        ):
            raise ValueError(
                "bound vocabulary split hashes differ from the frozen contract"
            )
        split_counts = {
            split: sum(record["split"] == split for record in canonical_records)
            for split in ("fit", "dev", "audit")
        }
        expected_counts = frozen_vocabulary.get("counts")
        if (
            not isinstance(expected_counts, Mapping)
            or vocabulary_manifest.get("counts") != expected_counts
            or len(canonical_records)
            != int(expected_counts.get("deduplicated_queries", -1))
            or split_counts
            != {
                split: int(expected_counts.get(split, -1))
                for split in ("fit", "dev", "audit")
            }
        ):
            raise ValueError("bound vocabulary counts differ from the frozen contract")
        declared_sources = vocabulary_manifest.get("sources")
        expected_sources = frozen_vocabulary.get("source_sha256")
        if not isinstance(declared_sources, Mapping) or not isinstance(
            expected_sources, Mapping
        ):
            raise ValueError("bound vocabulary lacks frozen source provenance")
        source_hashes = {
            str(name): str(record.get("sha256", ""))
            for name, record in declared_sources.items()
            if isinstance(record, Mapping)
        }
        if source_hashes != dict(expected_sources):
            raise ValueError(
                "bound vocabulary source hashes differ from the frozen contract"
            )
        split_sha = computed_split_hashes[query_split]
    records_sha = _records_sha256(selected_records)
    _verify_hash(records_sha, payload.get("ordered_records_sha256"), "ordered records")
    _verify_hash(
        split_sha,
        payload.get("split_synset_tab_query_lf_sha256"),
        "embedding split",
    )
    queries = [record["query"] for record in selected_records]
    synsets = [record["synset"] for record in selected_records]
    if (
        payload.get("records") != selected_records
        or payload.get("queries") != queries
        or payload.get("synsets") != synsets
    ):
        raise ValueError("embedding rows do not exactly match the canonical selected split")

    embeddings = torch.as_tensor(payload.get("embeddings"))
    if (
        embeddings.device.type != "cpu"
        or embeddings.ndim != 2
        or embeddings.dtype != torch.float32
        or embeddings.shape != (len(selected_records), TEXT_BANK_OUTPUT_DIMENSION)
        or not bool(torch.isfinite(embeddings).all())
    ):
        raise ValueError(
            "text embeddings must be finite CPU float32 [selected_queries,1536] tensors"
        )
    norms = torch.linalg.vector_norm(embeddings, dim=-1)
    if not bool(torch.allclose(norms, torch.ones_like(norms), atol=5e-5, rtol=5e-5)):
        raise ValueError("text embedding rows must be L2-normalized")
    semantic_sha = _embedding_semantic_sha256(embeddings)
    _verify_hash(
        semantic_sha,
        payload.get("embedding_semantic_sha256"),
        "embedding semantic tensor",
    )
    _verify_hash(
        tensor_sha256(embeddings),
        payload.get("embedding_tensor_sha256"),
        "embedding tensor",
    )
    text_encoder = _validate_text_encoder(
        payload.get("text_encoder"),
        expected_files_sha256=(
            FROZEN_SNAPSHOT_FILES_SHA256
            if _test_snapshot_files_sha256 is None
            else _test_snapshot_files_sha256
        ),
        hash_cache=_hash_cache,
    )

    common_fields = (
        "split",
        "split_synset_tab_query_lf_sha256",
        "prompt_templates",
        "text_canonicalization",
        "records",
        "queries",
        "synsets",
        "ordered_records_sha256",
    )
    if any(sidecar.get(key) != payload.get(key) for key in common_fields):
        raise ValueError("embedding sidecar differs from its tensor payload")
    if sidecar.get("text_encoder") != text_encoder:
        raise ValueError("embedding sidecar text_encoder differs from its payload")
    sidecar_vocabulary = sidecar.get("vocabulary")
    if not isinstance(sidecar_vocabulary, Mapping) or dict(sidecar_vocabulary) != {
        "path": str(vocabulary_path),
        "sha256": vocabulary_sha,
        "manifest_path": str(vocabulary_manifest_path),
        "manifest_sha256": vocabulary_manifest_sha,
    }:
        raise ValueError("embedding sidecar vocabulary binding differs from its payload")
    sidecar_embedding = sidecar.get("embedding")
    expected_embedding = {
        "shape": [len(selected_records), TEXT_BANK_OUTPUT_DIMENSION],
        "dtype": "float32",
        "byte_order": "little_endian",
        "normalization": "l2",
        "semantic_sha256": semantic_sha,
        "tensor_sha256": tensor_sha256(embeddings),
    }
    if not isinstance(sidecar_embedding, Mapping) or dict(sidecar_embedding) != expected_embedding:
        raise ValueError("embedding sidecar tensor binding differs from its payload")
    sidecar_artifact = sidecar.get("artifact")
    if (
        not isinstance(sidecar_artifact, Mapping)
        or set(sidecar_artifact) != {"path", "sha256"}
    ):
        raise ValueError("embedding sidecar lacks its artifact binding")
    bound_artifact = _resolve_bound_path(
        sidecar_artifact.get("path"),
        relative_to=sidecar_path.parent,
        name="embedding artifact",
    )
    if bound_artifact != path:
        raise ValueError("embedding sidecar points to another tensor artifact")
    file_sha = _sha256_file_cached(path, _hash_cache)
    _verify_hash(file_sha, sidecar_artifact.get("sha256"), "embedding artifact")
    sidecar_builder = sidecar.get("builder")
    if (
        not isinstance(sidecar_builder, Mapping)
        or set(sidecar_builder) != {"path", "sha256"}
    ):
        raise ValueError("embedding sidecar lacks builder provenance")
    builder_path = _resolve_bound_path(
        sidecar_builder.get("path"),
        relative_to=sidecar_path.parent,
        name="embedding builder",
    )
    _validate_embedding_builder_provenance(
        builder_path=builder_path,
        builder_sha256=sidecar_builder.get("sha256"),
        artifact_path=path,
        artifact_sha256=file_sha,
        manifest_path=sidecar_path,
        manifest_sha256=_sha256_file_cached(sidecar_path, _hash_cache),
        query_split=query_split,
        algorithm_version=str(algorithm_version),
        hash_cache=_hash_cache,
        test_contract_injected=_test_vocabulary_contract is not None,
    )

    return {
        "path": path,
        "file_sha256": file_sha,
        "manifest_path": sidecar_path,
        "manifest_sha256": _sha256_file_cached(sidecar_path, _hash_cache),
        "payload": payload,
        "embeddings": embeddings.contiguous(),
        "query_ids": synsets,
        "selected_records": selected_records,
        "selected_records_sha256": split_sha,
        "ordered_records_sha256": records_sha,
        "query_split": query_split,
        "algorithm_version": str(algorithm_version),
        "bank_family": (
            "imagenet12k_minus_imagenet1k_holdout_v1"
            if is_imagenet12k_holdout
            else "imagenet1k_primary_v1"
        ),
        "vocabulary_sha256": vocabulary_sha,
        "embedding_tensor_sha256": tensor_sha256(embeddings),
        "embedding_semantic_sha256": semantic_sha,
        "text_encoder": text_encoder,
    }


def evaluate_artifacts(
    descriptor_path: Path,
    text_bank_path: Path,
    text_bank_manifest_path: Path,
    *,
    query_split: str,
    _hash_cache: dict[Path, str] | None = None,
    _descriptor_cache: dict[Path, dict] | None = None,
    _bank_cache: dict[tuple[Path, Path, str], dict] | None = None,
    _test_vocabulary_contract: Mapping[str, object] | None = None,
    _test_snapshot_files_sha256: Mapping[str, str] | None = None,
) -> dict:
    descriptor_path = Path(descriptor_path).resolve()
    descriptor = (
        _descriptor_cache.get(descriptor_path)
        if _descriptor_cache is not None
        else None
    )
    if descriptor is None:
        descriptor = load_descriptor_pair(
            descriptor_path,
            _hash_cache=_hash_cache,
        )
        if _descriptor_cache is not None:
            _descriptor_cache[descriptor_path] = descriptor
    bank_key = (
        Path(text_bank_path).resolve(),
        Path(text_bank_manifest_path).resolve(),
        str(query_split),
    )
    bank = _bank_cache.get(bank_key) if _bank_cache is not None else None
    if bank is None:
        bank = load_text_embedding_bank(
            bank_key[0],
            bank_key[1],
            query_split,
            _hash_cache=_hash_cache,
            _test_vocabulary_contract=_test_vocabulary_contract,
            _test_snapshot_files_sha256=_test_snapshot_files_sha256,
        )
        if _bank_cache is not None:
            _bank_cache[bank_key] = bank
    if descriptor["student"].shape[1] != bank["embeddings"].shape[1]:
        raise ValueError(
            "descriptor/text dimension mismatch: "
            f"{descriptor['student'].shape[1]} vs {bank['embeddings'].shape[1]}"
        )
    metrics = evaluate_response_fidelity(
        descriptor["student"],
        descriptor["teacher"],
        bank["embeddings"],
        scene_ids=descriptor["scene_ids"],
        region_ids=descriptor["region_ids"],
        query_ids=bank["query_ids"],
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_type": REPORT_ARTIFACT_TYPE,
        "method_id": descriptor["method_id"],
        "seed": descriptor["seed"],
        "split_role": "query_free_validation",
        "query_split": query_split,
        "selection_contract": selection_contract_for_bank_family(
            bank["bank_family"]
        ),
        "descriptor_artifact": {
            "path": str(descriptor["path"]),
            "sha256": descriptor["file_sha256"],
        },
        "descriptor_rows_sha256": descriptor["rows_sha256"],
        "teacher_descriptors_sha256": descriptor["teacher_sha256"],
        "query_bank": {
            "path": str(bank["path"]),
            "sha256": bank["file_sha256"],
            "manifest_path": str(bank["manifest_path"]),
            "manifest_sha256": bank["manifest_sha256"],
            "vocabulary_sha256": bank["vocabulary_sha256"],
            "query_split": query_split,
            "selected_queries": len(bank["selected_records"]),
            "selected_records_sha256": bank["selected_records_sha256"],
            "ordered_records_sha256": bank["ordered_records_sha256"],
            "embedding_tensor_sha256": bank["embedding_tensor_sha256"],
            "embedding_semantic_sha256": bank["embedding_semantic_sha256"],
            "text_encoder": bank["text_encoder"],
        },
        "metrics": metrics,
    }


def evaluate_many_artifacts(
    descriptor_paths: list[Path],
    text_bank_path: Path,
    text_bank_manifest_path: Path,
    *,
    query_split: str,
) -> list[dict]:
    if not descriptor_paths:
        raise ValueError("evaluate-many requires at least one descriptor")
    hash_cache: dict[Path, str] = {}
    descriptor_cache: dict[Path, dict] = {}
    bank_cache: dict[tuple[Path, Path, str], dict] = {}
    return [
        evaluate_artifacts(
            path,
            text_bank_path,
            text_bank_manifest_path,
            query_split=query_split,
            _hash_cache=hash_cache,
            _descriptor_cache=descriptor_cache,
            _bank_cache=bank_cache,
        )
        for path in descriptor_paths
    ]


def _write_json(path: Path, value: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_report(path: Path) -> Mapping:
    return _read_json(Path(path).resolve())


def _parse_seeds(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in str(raw).split(",") if value.strip())
    if not values:
        raise ValueError("required seeds cannot be empty")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--descriptors", required=True, type=Path)
    evaluate_parser.add_argument("--text-bank", required=True, type=Path)
    evaluate_parser.add_argument("--text-bank-manifest", required=True, type=Path)
    evaluate_parser.add_argument(
        "--query-split", choices=sorted(ALLOWED_HELDOUT_SPLITS), default="dev"
    )
    evaluate_parser.add_argument("--output", required=True, type=Path)

    evaluate_many_parser = subparsers.add_parser("evaluate-many")
    evaluate_many_parser.add_argument(
        "--descriptors", action="append", required=True, type=Path
    )
    evaluate_many_parser.add_argument("--text-bank", required=True, type=Path)
    evaluate_many_parser.add_argument(
        "--text-bank-manifest", required=True, type=Path
    )
    evaluate_many_parser.add_argument(
        "--query-split", choices=sorted(ALLOWED_HELDOUT_SPLITS), default="dev"
    )
    evaluate_many_parser.add_argument(
        "--output", action="append", required=True, type=Path
    )

    gate_parser = subparsers.add_parser("gate")
    gate_parser.add_argument(
        "--control-report", action="append", required=True, type=Path
    )
    gate_parser.add_argument(
        "--candidate-report", action="append", required=True, type=Path
    )
    gate_parser.add_argument(
        "--phase",
        choices=("dev", "audit"),
        required=True,
        help="dev is selection-only; audit is confirmation-only with no retuning",
    )
    gate_parser.add_argument("--required-seeds", default="0,1,2")
    gate_parser.add_argument("--minimum-improved-seeds", type=int, default=2)
    gate_parser.add_argument("--bootstrap-samples", type=int, default=2000)
    gate_parser.add_argument("--bootstrap-seed", type=int, default=20260731)
    gate_parser.add_argument(
        "--quality-noninferiority-tolerance", type=float, default=0.0
    )
    gate_parser.add_argument("--output", required=True, type=Path)

    args = parser.parse_args()
    if args.command == "evaluate":
        result = evaluate_artifacts(
            args.descriptors,
            args.text_bank,
            args.text_bank_manifest,
            query_split=args.query_split,
        )
    elif args.command == "evaluate-many":
        if len(args.descriptors) != len(args.output):
            raise ValueError(
                "evaluate-many requires one --output per --descriptors argument"
            )
        results = evaluate_many_artifacts(
            args.descriptors,
            args.text_bank,
            args.text_bank_manifest,
            query_split=args.query_split,
        )
        for output, report in zip(args.output, results):
            _write_json(output, report)
        print(
            json.dumps(
                {
                    "status": "complete_shared_text_bank_validation",
                    "query_split": args.query_split,
                    "reports": [str(Path(path).resolve()) for path in args.output],
                },
                indent=2,
            )
        )
        return
    else:
        result = aggregate_paired_seed_gate(
            [_load_report(path) for path in args.control_report],
            [_load_report(path) for path in args.candidate_report],
            required_seeds=_parse_seeds(args.required_seeds),
            minimum_improved_seeds=args.minimum_improved_seeds,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            quality_noninferiority_tolerance=args.quality_noninferiority_tolerance,
            phase=args.phase,
        )
    _write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
