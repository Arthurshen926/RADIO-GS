#!/usr/bin/env python3
"""Build a target-blind generic text vocabulary from timm ImageNet-1K data.

This builder deliberately has no benchmark-vocabulary input.  It reads only
the two pinned timm ImageNet metadata files supplied on the command line and
does not import timm, torch, or any GPU runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ALGORITHM_VERSION = "imagenet1k-primary-v1"
ARTIFACT_TYPE = "target_blind_imagenet1k_primary_text_bank"
EXPECTED_SOURCE_SHA256 = {
    "imagenet_synsets.txt": (
        "70002b0ff5de60a3a17a82dbfcff291931f96225ddf941ad2e182fc39e183d15"
    ),
    "imagenet_synset_to_lemma.txt": (
        "1b8babda187421a4bde0c9c5a197c36f6bdda962f7ca11ffb2813806cbb2178f"
    ),
}
EXPECTED_COUNTS = {
    "source_synsets": 1000,
    "deduplicated_queries": 997,
    "fit": 806,
    "dev": 101,
    "audit": 90,
}
_SYNSET_PATTERN = re.compile(r"n[0-9]{8}\Z")
_ASCII_PUNCTUATION_TRANSLATION = str.maketrans("", "", string.punctuation)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def normalize_primary_alias(value: str) -> str:
    """Apply the frozen primary-alias canonicalization contract."""

    return " ".join(
        value.replace("_", " ")
        .translate(_ASCII_PUNCTUATION_TRANSLATION)
        .casefold()
        .split()
    )


def _read_synsets(path: Path) -> list[str]:
    synsets: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            synset = raw_line.rstrip("\r\n")
            if not _SYNSET_PATTERN.fullmatch(synset):
                raise ValueError(
                    f"{path}:{line_number}: invalid ImageNet synset {synset!r}"
                )
            if synset in seen:
                raise ValueError(
                    f"{path}:{line_number}: duplicate ImageNet synset {synset}"
                )
            seen.add(synset)
            synsets.append(synset)
    return synsets


def _read_synset_to_lemma(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            fields = line.split("\t", maxsplit=1)
            if len(fields) != 2:
                raise ValueError(
                    f"{path}:{line_number}: expected synset<TAB>lemma"
                )
            synset, aliases = fields
            if not _SYNSET_PATTERN.fullmatch(synset) or not aliases.strip():
                raise ValueError(
                    f"{path}:{line_number}: invalid synset-to-lemma row"
                )
            if synset in mapping:
                raise ValueError(
                    f"{path}:{line_number}: duplicate lemma for {synset}"
                )
            mapping[synset] = aliases
    return mapping


def split_for_synset(synset: str) -> str:
    digest = hashlib.sha256(
        ALGORITHM_VERSION.encode("ascii") + b"\0" + synset.encode("ascii")
    ).digest()
    bucket = int.from_bytes(digest[:8], byteorder="big", signed=False) % 10
    if bucket <= 7:
        return "fit"
    if bucket == 8:
        return "dev"
    return "audit"


def _build_records(
    synsets: Iterable[str],
    synset_to_lemma: dict[str, str],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    records: list[dict[str, str]] = []
    duplicates: list[dict[str, str]] = []
    first_synset_by_query: dict[str, str] = {}
    for synset in synsets:
        aliases = synset_to_lemma.get(synset)
        if aliases is None:
            raise ValueError(f"missing lemma row for ImageNet-1K synset {synset}")
        primary_alias = aliases.split(",", maxsplit=1)[0]
        query = normalize_primary_alias(primary_alias)
        if not query:
            raise ValueError(f"primary alias for {synset} is empty after normalization")
        first_synset = first_synset_by_query.get(query)
        if first_synset is not None:
            duplicates.append(
                {
                    "dropped_synset": synset,
                    "first_synset": first_synset,
                    "query": query,
                }
            )
            continue
        first_synset_by_query[query] = synset
        records.append(
            {
                "synset": synset,
                "query": query,
                "split": split_for_synset(synset),
            }
        )
    return records, duplicates


def _split_line_stream(records: list[dict[str, str]], split: str) -> bytes:
    return "".join(
        f"{record['synset']}\t{record['query']}\n"
        for record in records
        if record["split"] == split
    ).encode("utf-8")


def _validate_official_sources(
    synsets_path: Path,
    synset_to_lemma_path: Path,
) -> dict[str, dict[str, Any]]:
    paths = {
        "imagenet_synsets.txt": synsets_path,
        "imagenet_synset_to_lemma.txt": synset_to_lemma_path,
    }
    provenance: dict[str, dict[str, Any]] = {}
    for source_name, path in paths.items():
        if path.name != source_name:
            raise ValueError(
                f"expected timm source filename {source_name}, got {path.name}"
            )
        digest = _sha256_file(path)
        expected = EXPECTED_SOURCE_SHA256[source_name]
        if digest != expected:
            raise ValueError(
                f"{path}: SHA-256 mismatch for pinned timm source "
                f"(expected {expected}, got {digest})"
            )
        provenance[source_name] = {
            "path": str(path.resolve()),
            "sha256": digest,
        }
    return provenance


def build_bank(
    *,
    imagenet_synsets: Path,
    imagenet_synset_to_lemma: Path,
    output: Path,
    manifest_output: Path,
) -> dict[str, Any]:
    imagenet_synsets = imagenet_synsets.resolve()
    imagenet_synset_to_lemma = imagenet_synset_to_lemma.resolve()
    output = output.resolve()
    manifest_output = manifest_output.resolve()
    sources = {imagenet_synsets, imagenet_synset_to_lemma}
    if output == manifest_output:
        raise ValueError("output and manifest_output must be different files")
    if output in sources or manifest_output in sources:
        raise ValueError("outputs must not overwrite either timm source file")

    source_provenance = _validate_official_sources(
        imagenet_synsets,
        imagenet_synset_to_lemma,
    )
    synsets = _read_synsets(imagenet_synsets)
    synset_to_lemma = _read_synset_to_lemma(imagenet_synset_to_lemma)
    records, duplicates = _build_records(synsets, synset_to_lemma)
    split_counts = Counter(record["split"] for record in records)
    actual_counts = {
        "source_synsets": len(synsets),
        "deduplicated_queries": len(records),
        "fit": split_counts["fit"],
        "dev": split_counts["dev"],
        "audit": split_counts["audit"],
    }
    if actual_counts != EXPECTED_COUNTS:
        raise ValueError(
            "pinned ImageNet text-bank count contract failed: "
            f"expected {EXPECTED_COUNTS}, got {actual_counts}"
        )

    payload = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "algorithm_version": ALGORITHM_VERSION,
        "prompt_templates": ["{query}"],
        "benchmark_vocabulary_opened": False,
        "records": records,
    }
    output_bytes = _canonical_json_bytes(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(output_bytes)

    builder_path = Path(__file__).resolve()
    split_hashes = {
        split: _sha256_bytes(_split_line_stream(records, split))
        for split in ("fit", "dev", "audit")
    }
    manifest = {
        "schema_version": 1,
        "artifact_type": f"{ARTIFACT_TYPE}_manifest",
        "algorithm_version": ALGORITHM_VERSION,
        "benchmark_vocabulary_opened": False,
        "builder": {
            "path": str(builder_path),
            "sha256": _sha256_file(builder_path),
        },
        "sources": source_provenance,
        "normalization": {
            "alias_selection": "first_comma_separated_alias",
            "steps_in_order": [
                "underscore_to_space",
                "delete_ascii_punctuation",
                "unicode_casefold",
                "whitespace_collapse",
            ],
            "deduplication": "normalized_query_first_wins_in_imagenet_synset_order",
        },
        "split_assignment": {
            "hash_input": "imagenet1k-primary-v1\\0<synset>",
            "hash": "sha256",
            "bucket": "uint64_big_endian(digest[0:8]) mod 10",
            "fit_buckets": list(range(8)),
            "dev_buckets": [8],
            "audit_buckets": [9],
        },
        "counts": actual_counts,
        "duplicates_first_wins": duplicates,
        "split_synset_tab_query_lf_sha256": split_hashes,
        "canonical_json": {
            "path": str(output),
            "sha256": _sha256_bytes(output_bytes),
        },
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_bytes(_canonical_json_bytes(manifest))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--imagenet-synsets",
        type=Path,
        required=True,
        help="Pinned timm data/_info/imagenet_synsets.txt",
    )
    parser.add_argument(
        "--imagenet-synset-to-lemma",
        type=Path,
        required=True,
        help="Pinned timm data/_info/imagenet_synset_to_lemma.txt",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_bank(
        imagenet_synsets=args.imagenet_synsets,
        imagenet_synset_to_lemma=args.imagenet_synset_to_lemma,
        output=args.output,
        manifest_output=args.manifest_output,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
