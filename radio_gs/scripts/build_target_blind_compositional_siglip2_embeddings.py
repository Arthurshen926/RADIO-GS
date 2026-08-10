#!/usr/bin/env python3
"""Materialize frozen fit-only compositional SigLIP2 banks on one GPU.

The source bank is target blind and already split by ImageNet synset.  This
builder keeps semantic strata in separate artifacts so their row counts cannot
silently become loss coefficients.  It loads only the official SigLIP2 text
tower from the pinned local snapshot and never opens benchmark data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.scripts import (
    build_target_blind_siglip2_embedding_artifact as frozen_encoder,
)
from radio_gs.scripts.eval_lerf_grounding import (
    _resolve_siglip2_text_max_length,
)
from radio_gs.utils.immutable_artifacts import sha256_file


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "target_blind_compositional_text_embedding_cache"
MANIFEST_ARTIFACT_TYPE = f"{ARTIFACT_TYPE}_manifest"
ALGORITHM_VERSION = "imagenet1k-compositional-v2-siglip2-fit-v1"
SOURCE_ALGORITHM_VERSION = "imagenet1k-compositional-v2"
SOURCE_ARTIFACT_TYPE = "target_blind_imagenet1k_compositional_text_bank"
SOURCE_SHA256 = "b53693a2821c29a5cc18b3ab69a9e7d9189b2c0746343b702747234ce5704b7a"
SOURCE_MANIFEST_SHA256 = (
    "e031a6bc38242af990ecf488c96c59f667f167126d67fecae66a0b23aeb1cd96"
)
COMPONENTS = (
    "synonym_relation",
    "lexical_sibling_relation",
    "counterfactual_attributes",
    "high_precision_part_of",
)
ATTRIBUTE_STRUCTURES = {
    "color_plus_noun",
    "material_plus_noun",
    "shape_plus_noun",
    "adjective_plus_noun",
}


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _semantic_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().to(torch.float32).contiguous().numpy()
    return hashlib.sha256(array.astype("<f4", copy=False).tobytes()).hexdigest()


def _read_source(source: Path, source_manifest: Path) -> dict[str, Any]:
    source = source.resolve()
    source_manifest = source_manifest.resolve()
    if sha256_file(source) != SOURCE_SHA256:
        raise ValueError("compositional source bank SHA-256 differs")
    if sha256_file(source_manifest) != SOURCE_MANIFEST_SHA256:
        raise ValueError("compositional source manifest SHA-256 differs")
    payload = json.loads(source.read_text(encoding="utf-8"))
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    if (
        payload.get("artifact_type") != SOURCE_ARTIFACT_TYPE
        or payload.get("algorithm_version") != SOURCE_ALGORITHM_VERSION
        or payload.get("benchmark_vocabulary_opened") is not False
        or payload.get("uses_benchmark_vocabulary_for_construction") is not False
        or payload.get("prompt_templates") != ["{query}"]
        or manifest.get("canonical_json", {}).get("sha256") != SOURCE_SHA256
    ):
        raise ValueError("compositional source contract differs")
    return payload


def component_queries(source: Mapping[str, Any]) -> dict[str, list[str]]:
    """Return deterministic fit-only query lists for the four optional strata."""

    query_records = source.get("query_records")
    relation_records = source.get("relation_records")
    if not isinstance(query_records, list) or not isinstance(relation_records, list):
        raise ValueError("compositional source records are missing")
    fit_queries = [row for row in query_records if row.get("split") == "fit"]
    fit_relations = [row for row in relation_records if row.get("split") == "fit"]
    result = {
        "synonym_relation": sorted(
            {
                str(row["right_query"])
                for row in fit_relations
                if row.get("relation") == "synonym"
            }
        ),
        "lexical_sibling_relation": sorted(
            {
                str(row[key])
                for row in fit_relations
                if row.get("relation") == "lexical_head_sibling_contrast"
                for key in ("left_query", "right_query")
            }
        ),
        "counterfactual_attributes": sorted(
            {
                str(row["query"])
                for row in fit_queries
                if row.get("structure") in ATTRIBUTE_STRUCTURES
            }
        ),
        "high_precision_part_of": sorted(
            {
                str(row["query"])
                for row in fit_queries
                if row.get("structure") == "part_of_object"
            }
        ),
    }
    if tuple(result) != COMPONENTS or any(not rows for rows in result.values()):
        raise ValueError("one or more frozen compositional components are empty")
    if any(len(rows) != len(set(rows)) for rows in result.values()):
        raise ValueError("compositional component queries must be unique")
    return result


class _GpuTextEncoder:
    def __init__(self, snapshot: Path, device: torch.device) -> None:
        try:
            from transformers import AutoConfig, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "transformers is required for SigLIP2 text encoding"
            ) from error
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(
                "this production builder requires an available CUDA device"
            )
        model, _ = frozen_encoder._load_local_siglip2_text_tower(snapshot)
        self.model = model.float().to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot), local_files_only=True, trust_remote_code=False
        )
        config = AutoConfig.from_pretrained(
            str(snapshot), local_files_only=True, trust_remote_code=False
        )
        self.max_length = _resolve_siglip2_text_max_length(config)
        self.device = device

    @torch.inference_mode()
    def __call__(self, queries: Sequence[str], batch_size: int) -> torch.Tensor:
        output: list[torch.Tensor] = []
        for start in range(0, len(queries), batch_size):
            batch = list(queries[start : start + batch_size])
            tokens = self.tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {
                str(key): value.to(self.device, non_blocking=True)
                for key, value in tokens.items()
                if torch.is_tensor(value)
            }
            embeddings = self.model(**inputs)[1]
            output.append(F.normalize(embeddings.float(), dim=-1).cpu())
        torch.cuda.synchronize(self.device)
        return torch.cat(output, dim=0).contiguous()


def _atomic_torch_save(payload: Mapping[str, Any], output: Path) -> None:
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(output)


def build_banks(
    *,
    source: Path,
    source_manifest: Path,
    snapshot: Path,
    output_root: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    source = source.resolve()
    source_manifest = source_manifest.resolve()
    snapshot = snapshot.resolve()
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("output root must be absent or empty")
    source_payload = _read_source(source, source_manifest)
    queries_by_component = component_queries(source_payload)
    encoder_contract = frozen_encoder._validate_snapshot(
        snapshot,
        model_id=frozen_encoder.MODEL_ID,
        revision=frozen_encoder.MODEL_REVISION,
        expected_files_sha256=frozen_encoder.FROZEN_SNAPSHOT_FILES_SHA256,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    encoder = _GpuTextEncoder(snapshot, device)
    builder = Path(__file__).resolve()
    report_components: dict[str, Any] = {}
    for component_id, queries in queries_by_component.items():
        embeddings = encoder(queries, batch_size)
        if embeddings.shape != (len(queries), frozen_encoder.OUTPUT_DIMENSION):
            raise RuntimeError("SigLIP2 compositional embedding shape differs")
        embedding_hash = tensor_sha256(embeddings)
        text_encoder = {
            **encoder_contract,
            "device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "compute_dtype": "float32",
        }
        artifact = output_root / f"{component_id}_fit_embeddings.pt"
        manifest_path = output_root / f"{component_id}_fit_embeddings.manifest.json"
        payload = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "algorithm_version": ALGORITHM_VERSION,
            "benchmark_vocabulary_opened": False,
            "uses_benchmark_vocabulary_for_construction": False,
            "split": "fit",
            "component_id": component_id,
            "prompt_templates": ["{query}"],
            "queries": queries,
            "text_encoder": text_encoder,
            "embeddings": embeddings,
            "embedding_tensor_sha256": embedding_hash,
        }
        _atomic_torch_save(payload, artifact)
        artifact_hash = sha256_file(artifact)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": MANIFEST_ARTIFACT_TYPE,
            "algorithm_version": ALGORITHM_VERSION,
            "benchmark_vocabulary_opened": False,
            "uses_benchmark_vocabulary_for_construction": False,
            "split": "fit",
            "component_id": component_id,
            "query_rows": len(queries),
            "query_list_sha256": hashlib.sha256(
                _canonical_json_bytes(queries)
            ).hexdigest(),
            "source": {
                "path": str(source),
                "sha256": SOURCE_SHA256,
                "manifest_path": str(source_manifest),
                "manifest_sha256": SOURCE_MANIFEST_SHA256,
            },
            "text_encoder": text_encoder,
            "embedding": {
                "shape": list(embeddings.shape),
                "dtype": "float32",
                "normalization": "l2",
                "semantic_sha256": _semantic_sha256(embeddings),
                "tensor_sha256": embedding_hash,
            },
            "artifact": {"path": str(artifact), "sha256": artifact_hash},
            "builder": {"path": str(builder), "sha256": sha256_file(builder)},
        }
        manifest_path.write_bytes(_canonical_json_bytes(manifest))
        report_components[component_id] = {
            "artifact": str(artifact),
            "artifact_sha256": artifact_hash,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "shape": list(embeddings.shape),
            "embedding_tensor_sha256": embedding_hash,
        }
    return {
        "status": "built",
        "output_root": str(output_root),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "components": report_components,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = build_banks(
        source=args.source,
        source_manifest=args.source_manifest,
        snapshot=args.snapshot,
        output_root=args.output_root,
        device=torch.device(args.device),
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
