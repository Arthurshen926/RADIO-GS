#!/usr/bin/env python3
"""Build SHA-bound fit-only synonym and sibling relation pair indices."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_json_object,
    load_torch_mapping,
    sha256_file,
    write_torch_noclobber,
)


SCHEMA = "radio_gs.target_blind_typed_text_relation_authority.v1"
SOURCE_SHA256 = "b53693a2821c29a5cc18b3ab69a9e7d9189b2c0746343b702747234ce5704b7a"
MODEL_ID = "google/siglip2-giant-opt-patch16-384"
EXPECTED_COUNTS = {"synonym": 657, "lexical_head_sibling_contrast": 167}


def _load_bank(
    path: Path, expected_sha256: str, *, component_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, digest, source = load_torch_mapping(
        path,
        expected_sha256=expected_sha256,
        map_location="cpu",
        label=f"typed relation {component_id} embedding bank",
    )
    queries = payload.get("queries")
    embeddings = payload.get("embeddings")
    encoder = payload.get("text_encoder")
    if (
        payload.get("split") != "fit"
        or payload.get("benchmark_vocabulary_opened") is not False
        or payload.get("uses_benchmark_vocabulary_for_construction") is not False
        or payload.get("prompt_templates") != ["{query}"]
        or not isinstance(queries, list)
        or not queries
        or len(queries) != len(set(queries))
        or not torch.is_tensor(embeddings)
        or embeddings.dtype != torch.float32
        or embeddings.device.type != "cpu"
        or embeddings.shape != (len(queries), 1536)
        or not isinstance(encoder, Mapping)
        or encoder.get("model_id") != MODEL_ID
        or payload.get("embedding_tensor_sha256") != tensor_sha256(embeddings)
    ):
        raise ValueError(f"typed relation {component_id} bank differs")
    declared = payload.get("component_id")
    if component_id == "primary":
        if declared is not None:
            raise ValueError("primary relation bank unexpectedly declares a component")
    elif declared != component_id:
        raise ValueError(f"typed relation {component_id} identity differs")
    return payload, {
        "path": str(source),
        "sha256": digest,
        "embedding_tensor_sha256": tensor_sha256(embeddings),
        "query_rows": len(queries),
    }


def _unique_index(queries: list[str], *, label: str) -> dict[str, int]:
    if any(not isinstance(value, str) or not value.strip() for value in queries):
        raise ValueError(f"{label} queries contain an invalid string")
    result = {value: index for index, value in enumerate(queries)}
    if len(result) != len(queries):
        raise ValueError(f"{label} queries are not unique")
    return result


def build_authority(
    *,
    source_bank: Path,
    component_paths: Mapping[str, Path],
    component_sha256: Mapping[str, str],
    output: Path,
) -> Path:
    source, source_digest, source_path = load_json_object(
        source_bank,
        expected_sha256=SOURCE_SHA256,
        label="target-blind compositional relation source",
    )
    if (
        source.get("artifact_type") != "target_blind_imagenet1k_compositional_text_bank"
        or source.get("algorithm_version") != "imagenet1k-compositional-v2"
        or source.get("benchmark_vocabulary_opened") is not False
        or source.get("uses_benchmark_vocabulary_for_construction") is not False
    ):
        raise ValueError("typed relation source contract differs")
    expected_components = {
        "primary",
        "synonym_relation",
        "lexical_sibling_relation",
        "counterfactual_attributes",
        "high_precision_part_of",
    }
    if (
        set(component_paths) != expected_components
        or set(component_sha256) != expected_components
    ):
        raise ValueError("typed relation embedding component set differs")
    banks: dict[str, dict[str, Any]] = {}
    records: dict[str, dict[str, Any]] = {}
    for component in sorted(expected_components):
        banks[component], records[component] = _load_bank(
            component_paths[component],
            component_sha256[component],
            component_id=component,
        )
    primary = _unique_index(banks["primary"]["queries"], label="primary")
    synonym = _unique_index(
        banks["synonym_relation"]["queries"], label="synonym relation"
    )

    relation_rows = source.get("relation_records")
    if not isinstance(relation_rows, list):
        raise ValueError("typed relation source lacks relation_records")
    fit = [row for row in relation_rows if row.get("split") == "fit"]
    counts = Counter(str(row.get("relation")) for row in fit)
    if dict(counts) != EXPECTED_COUNTS:
        raise ValueError("typed relation fit counts differ")

    synonym_ids: list[str] = []
    synonym_left: list[int] = []
    synonym_right: list[int] = []
    sibling_ids: list[str] = []
    sibling_left: list[int] = []
    sibling_right: list[int] = []
    for row in fit:
        relation = str(row.get("relation"))
        record_id = str(row.get("record_id", ""))
        left = str(row.get("left_query", ""))
        right = str(row.get("right_query", ""))
        if not record_id or left not in primary:
            raise ValueError("typed relation left query is absent from primary")
        if relation == "synonym":
            if right not in synonym:
                raise ValueError("typed synonym alias is absent from its component")
            synonym_ids.append(record_id)
            synonym_left.append(primary[left])
            synonym_right.append(synonym[right])
        elif relation == "lexical_head_sibling_contrast":
            if right not in primary or left == right:
                raise ValueError("typed sibling query pair differs")
            sibling_ids.append(record_id)
            sibling_left.append(primary[left])
            sibling_right.append(primary[right])
        else:
            raise ValueError(
                "typed relation source contains an unsupported fit relation"
            )
    if (
        len(synonym_ids) != EXPECTED_COUNTS["synonym"]
        or len(sibling_ids) != EXPECTED_COUNTS["lexical_head_sibling_contrast"]
        or len(set(synonym_ids)) != len(synonym_ids)
        or len(set(sibling_ids)) != len(sibling_ids)
    ):
        raise ValueError("typed relation record identity differs")

    identity = {
        "schema": SCHEMA,
        "schema_version": 1,
        "split": "fit",
        "source": {"path": str(source_path), "sha256": source_digest},
        "components": records,
        "counts": {
            "synonym_pairs": len(synonym_ids),
            "sibling_pairs": len(sibling_ids),
        },
        "index_semantics": {
            "synonym_left": "primary",
            "synonym_right": "synonym_relation",
            "sibling_left": "primary",
            "sibling_right": "primary",
        },
        "source_access": {
            "benchmark_vocabulary_opened": False,
            "benchmark_images_opened": False,
            "benchmark_labels_opened": False,
            "benchmark_masks_opened": False,
            "target_metrics_computed": False,
        },
    }
    payload = {
        **identity,
        "content_authority_sha256": canonical_json_sha256(identity),
        "synonym_record_ids": synonym_ids,
        "synonym_left_primary_indices": torch.tensor(synonym_left, dtype=torch.int64),
        "synonym_right_component_indices": torch.tensor(
            synonym_right, dtype=torch.int64
        ),
        "sibling_record_ids": sibling_ids,
        "sibling_left_primary_indices": torch.tensor(sibling_left, dtype=torch.int64),
        "sibling_right_primary_indices": torch.tensor(sibling_right, dtype=torch.int64),
    }
    return write_torch_noclobber(output, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bank", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--primary-sha256", required=True)
    for component in (
        "synonym-relation",
        "lexical-sibling-relation",
        "counterfactual-attributes",
        "high-precision-part-of",
    ):
        parser.add_argument(f"--{component}", type=Path, required=True)
        parser.add_argument(f"--{component}-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "primary": args.primary,
        "synonym_relation": args.synonym_relation,
        "lexical_sibling_relation": args.lexical_sibling_relation,
        "counterfactual_attributes": args.counterfactual_attributes,
        "high_precision_part_of": args.high_precision_part_of,
    }
    digests = {
        "primary": args.primary_sha256,
        "synonym_relation": args.synonym_relation_sha256,
        "lexical_sibling_relation": args.lexical_sibling_relation_sha256,
        "counterfactual_attributes": args.counterfactual_attributes_sha256,
        "high_precision_part_of": args.high_precision_part_of_sha256,
    }
    print(
        build_authority(
            source_bank=args.source_bank,
            component_paths=paths,
            component_sha256=digests,
            output=args.output,
        )
    )


if __name__ == "__main__":
    main()
