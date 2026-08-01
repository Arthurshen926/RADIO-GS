#!/usr/bin/env python3
"""Freeze a target-blind ImageNet12K-minus-ImageNet1K holdout bank.

The builder reads only pinned ``timm`` metadata.  It never accepts a benchmark
vocabulary, semantic allow-list, or deny-list.  Dev and audit are emitted as
separate immutable files so the audit vocabulary need not be read when dev is
evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import string
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from radio_gs.utils.immutable_artifacts import (
    fsync_directory,
    write_bytes_noclobber,
)


SCHEMA_VERSION = 1
ALGORITHM_VERSION = "imagenet12k-minus-imagenet1k-target-blind-holdout-v1"
ARTIFACT_TYPE = "target_blind_imagenet12k_minus_imagenet1k_holdout_text_bank"
MANIFEST_ARTIFACT_TYPE = f"{ARTIFACT_TYPE}_manifest"
SOURCE_NAMES = (
    "imagenet12k_synsets.txt",
    "imagenet_synset_to_lemma.txt",
    "imagenet_synsets.txt",
)
EXPECTED_SOURCE_SHA256: Mapping[str, str] = {
    "imagenet12k_synsets.txt": (
        "f6483e79f18a9b670d43d915ddba40bbcaf82cfc4e92a07bd44057b9d44935f3"
    ),
    "imagenet_synset_to_lemma.txt": (
        "1b8babda187421a4bde0c9c5a197c36f6bdda962f7ca11ffb2813806cbb2178f"
    ),
    "imagenet_synsets.txt": (
        "70002b0ff5de60a3a17a82dbfcff291931f96225ddf941ad2e182fc39e183d15"
    ),
}
EXPECTED_COUNTS: Mapping[str, int] = {
    "imagenet12k_synsets": 11821,
    "imagenet1k_synsets": 1000,
    "direct_imagenet1k_excluded": 993,
    "imagenet1k_alias_conflict_excluded": 318,
    "normalized_primary_duplicate_excluded": 702,
    "candidates": 9808,
    "dev": 101,
    "audit": 90,
}
EXPECTED_RECORD_SHA256: Mapping[str, str] = {
    "candidates": "ca3ed853145424c3e5e4c0fe20c66ef7a3ee7be8d4bbe93c6f4d6dce984b8387",
    "dev": "c4a111be6e171d9934767cccc2fc2aae4fd1e46f21839e8dd5d69bbcdb243b3b",
    "audit": "008f35ac963f0af7e378abe18841e72eb16c06c7955ec0872e363b83937c7d94",
}
QUALITY_NONINFERIORITY_TOLERANCE = 0.005
DEV_SIZE = 101
AUDIT_SIZE = 90

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


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def normalize_alias(value: str) -> str:
    return " ".join(
        value.replace("_", " ")
        .translate(_ASCII_PUNCTUATION_TRANSLATION)
        .casefold()
        .split()
    )


def _read_synsets(path: Path) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            synset = raw_line.rstrip("\r\n")
            if not _SYNSET_PATTERN.fullmatch(synset):
                raise ValueError(f"{path}:{line_number}: invalid synset {synset!r}")
            if synset in seen:
                raise ValueError(f"{path}:{line_number}: duplicate synset {synset}")
            seen.add(synset)
            values.append(synset)
    return values


def _read_aliases(path: Path) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            fields = raw_line.rstrip("\r\n").split("\t", maxsplit=1)
            if len(fields) != 2 or not _SYNSET_PATTERN.fullmatch(fields[0]):
                raise ValueError(f"{path}:{line_number}: invalid lemma row")
            synset, raw_aliases = fields
            if synset in result:
                raise ValueError(f"{path}:{line_number}: duplicate lemma {synset}")
            aliases = tuple(
                normalized
                for raw in raw_aliases.split(",")
                if (normalized := normalize_alias(raw))
            )
            if not aliases:
                raise ValueError(f"{path}:{line_number}: no non-empty alias")
            result[synset] = aliases
    return result


def _records_bytes(records: Iterable[Mapping[str, str]]) -> bytes:
    return "".join(
        f"{record['synset']}\t{record['query']}\n" for record in records
    ).encode("utf-8")


def _build_candidates(
    imagenet12k: Sequence[str],
    imagenet1k: Sequence[str],
    aliases: Mapping[str, Sequence[str]],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    imagenet1k_set = set(imagenet1k)
    missing = sorted(
        synset for synset in (*imagenet12k, *imagenet1k) if synset not in aliases
    )
    if missing:
        raise ValueError(f"missing aliases for {len(missing)} required synsets")
    imagenet1k_aliases = {
        alias for synset in imagenet1k for alias in aliases[synset]
    }
    seen_primary: set[str] = set()
    candidates: list[dict[str, str]] = []
    excluded = {
        "direct_imagenet1k_excluded": 0,
        "imagenet1k_alias_conflict_excluded": 0,
        "normalized_primary_duplicate_excluded": 0,
    }
    for synset in imagenet12k:
        if synset in imagenet1k_set:
            excluded["direct_imagenet1k_excluded"] += 1
            continue
        local_aliases = aliases[synset]
        if any(alias in imagenet1k_aliases for alias in local_aliases):
            excluded["imagenet1k_alias_conflict_excluded"] += 1
            continue
        primary = str(local_aliases[0])
        if primary in seen_primary:
            excluded["normalized_primary_duplicate_excluded"] += 1
            continue
        seen_primary.add(primary)
        candidates.append({"synset": synset, "query": primary})
    return candidates, excluded


def _selection_key(role: str, record: Mapping[str, str]) -> tuple[bytes, str]:
    synset = str(record["synset"])
    digest = hashlib.sha256(
        f"{ALGORITHM_VERSION}\0{role}\0{synset}".encode("ascii")
    ).digest()
    return digest, synset


def _select_holdouts(
    candidates: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    dev = [
        {"synset": str(row["synset"]), "query": str(row["query"]), "split": "dev"}
        for row in sorted(candidates, key=lambda row: _selection_key("new-dev", row))[
            :DEV_SIZE
        ]
    ]
    dev_synsets = {row["synset"] for row in dev}
    remainder = [row for row in candidates if str(row["synset"]) not in dev_synsets]
    audit = [
        {
            "synset": str(row["synset"]),
            "query": str(row["query"]),
            "split": "audit",
        }
        for row in sorted(
            remainder, key=lambda row: _selection_key("new-audit", row)
        )[:AUDIT_SIZE]
    ]
    if dev_synsets & {row["synset"] for row in audit}:
        raise RuntimeError("dev and audit selection overlap")
    return dev, audit


def _validate_source_dir(source_dir: Path) -> dict[str, Path]:
    paths = {name: source_dir / name for name in SOURCE_NAMES}
    for name, path in paths.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"pinned source must be a regular non-symlink file: {path}")
        observed = _sha256_file(path)
        if observed != EXPECTED_SOURCE_SHA256[name]:
            raise ValueError(
                f"pinned source SHA mismatch for {name}: "
                f"expected {EXPECTED_SOURCE_SHA256[name]}, got {observed}"
            )
    return paths


def _bank_payload(split: str, records: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "algorithm_version": ALGORITHM_VERSION,
        "split": split,
        "prompt_templates": ["{query}"],
        "benchmark_vocabulary_opened": False,
        "uses_benchmark_vocabulary_for_construction": False,
        "records": list(records),
    }


def build_holdout_bank(*, source_dir: Path, output_root: Path) -> dict[str, Any]:
    source_dir = Path(source_dir).resolve(strict=True)
    output_root = Path(output_root).resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"immutable output root already exists: {output_root}")
    source_paths = _validate_source_dir(source_dir)
    imagenet12k = _read_synsets(source_paths["imagenet12k_synsets.txt"])
    imagenet1k = _read_synsets(source_paths["imagenet_synsets.txt"])
    aliases = _read_aliases(source_paths["imagenet_synset_to_lemma.txt"])
    candidates, excluded = _build_candidates(imagenet12k, imagenet1k, aliases)
    dev, audit = _select_holdouts(candidates)

    counts = {
        "imagenet12k_synsets": len(imagenet12k),
        "imagenet1k_synsets": len(imagenet1k),
        **excluded,
        "candidates": len(candidates),
        "dev": len(dev),
        "audit": len(audit),
    }
    if counts != dict(EXPECTED_COUNTS):
        raise ValueError(f"holdout count contract differs: {counts}")
    record_hashes = {
        "candidates": _sha256_bytes(_records_bytes(candidates)),
        "dev": _sha256_bytes(_records_bytes(dev)),
        "audit": _sha256_bytes(_records_bytes(audit)),
    }
    if record_hashes != dict(EXPECTED_RECORD_SHA256):
        raise ValueError(f"holdout record hash contract differs: {record_hashes}")

    # Exclusive mkdir is the race-safe point of no return.  A later failure
    # leaves a partial root that blocks retries instead of silently replacing it.
    output_root.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(output_root, mode=0o700)
    fsync_directory(output_root.parent)
    bundle_root = output_root / "source_bundle"
    os.mkdir(bundle_root, mode=0o700)
    fsync_directory(output_root)
    source_records: dict[str, dict[str, str]] = {}
    for name in SOURCE_NAMES:
        copied = write_bytes_noclobber(bundle_root / name, source_paths[name].read_bytes())
        source_records[name] = {
            "path": str(copied),
            "sha256": _sha256_file(copied),
        }

    builder_path = Path(__file__).resolve(strict=True)
    builder_copy = write_bytes_noclobber(bundle_root / builder_path.name, builder_path.read_bytes())
    dev_path = output_root / "target_blind_holdout_dev.json"
    audit_path = output_root / "target_blind_holdout_audit.json"
    dev_bytes = _canonical_json_bytes(_bank_payload("dev", dev))
    audit_bytes = _canonical_json_bytes(_bank_payload("audit", audit))
    write_bytes_noclobber(dev_path, dev_bytes)
    write_bytes_noclobber(audit_path, audit_bytes)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": MANIFEST_ARTIFACT_TYPE,
        "algorithm_version": ALGORITHM_VERSION,
        "benchmark_vocabulary_opened": False,
        "uses_benchmark_vocabulary_for_construction": False,
        "sources": source_records,
        "builder": {
            "path": str(builder_copy),
            "sha256": _sha256_file(builder_copy),
        },
        "normalization": {
            "alias_delimiter": "comma",
            "steps_in_order": [
                "underscore_to_space",
                "delete_ascii_punctuation",
                "unicode_casefold",
                "whitespace_collapse",
            ],
            "imagenet1k_exclusion": "all_normalized_aliases",
            "query": "normalized_primary_alias",
            "deduplication": "normalized_primary_first_wins_in_imagenet12k_order",
        },
        "selection": {
            "contract": ALGORITHM_VERSION,
            "dev_hash_input": f"{ALGORITHM_VERSION}\\0new-dev\\0<synset>",
            "audit_hash_input": f"{ALGORITHM_VERSION}\\0new-audit\\0<synset>",
            "ordering": "sha256_digest_ascending_then_synset",
            "audit_pool": "candidates_minus_dev",
            "semantic_or_benchmark_filtering": False,
        },
        "preregistered_gate": {
            "dev_role": "selection_only",
            "audit_role": "one_shot_confirmation_only_no_retuning",
            "required_seeds": [0, 1, 2],
            "minimum_error_improved_seeds": 2,
            "error_scene_ci_rule": "strictly_positive",
            "quality_scene_ci_rule": "lower_bound_gte_minus_tolerance",
            "quality_noninferiority_tolerance": QUALITY_NONINFERIORITY_TOLERANCE,
            "tolerance_rationale": (
                "absolute 0.5 percentage-point allowance on bounded response-profile, "
                "ranking, and top-decile quality metrics"
            ),
            "audit_requires_dev_promote": True,
        },
        "counts": counts,
        "synset_tab_query_lf_sha256": record_hashes,
        "artifacts": {
            "dev": {"path": str(dev_path), "sha256": _sha256_bytes(dev_bytes)},
            "audit": {"path": str(audit_path), "sha256": _sha256_bytes(audit_bytes)},
        },
    }
    manifest_path = output_root / "manifest.json"
    manifest_bytes = _canonical_json_bytes(manifest)
    write_bytes_noclobber(manifest_path, manifest_bytes)
    fsync_directory(bundle_root)
    fsync_directory(output_root)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_holdout_bank(
        source_dir=args.source_dir,
        output_root=args.output_root,
    )
    # Never print selected records.  These aggregate hashes are sufficient to
    # verify that the frozen authority matches the preregistered contract.
    print(
        json.dumps(
            {
                "artifact_type": manifest["artifact_type"],
                "algorithm_version": manifest["algorithm_version"],
                "counts": manifest["counts"],
                "synset_tab_query_lf_sha256": manifest[
                    "synset_tab_query_lf_sha256"
                ],
                "manifest_path": str(Path(args.output_root).resolve() / "manifest.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
