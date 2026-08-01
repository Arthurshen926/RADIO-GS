from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from radio_gs.scripts.build_target_blind_imagenet12k_holdout_bank import (
    ALGORITHM_VERSION,
    EXPECTED_COUNTS,
    EXPECTED_RECORD_SHA256,
    EXPECTED_SOURCE_SHA256,
    QUALITY_NONINFERIORITY_TOLERANCE,
    SOURCE_NAMES,
    _build_candidates,
    _records_bytes,
    _select_holdouts,
    build_holdout_bank,
    normalize_alias,
)


def _official_timm_info_dir() -> Path:
    candidates = sorted(
        (Path(sys.prefix) / "lib").glob("python*/site-packages/timm/data/_info")
    )
    for candidate in candidates:
        if all((candidate / name).is_file() for name in SOURCE_NAMES):
            return candidate
    pytest.skip("pinned timm ImageNet metadata is not installed")


def test_normalization_all_alias_exclusion_and_primary_first_wins() -> None:
    assert normalize_alias(" Jack-o'-Lantern__DOG!! ") == "jackolantern dog"
    aliases = {
        "n00000001": ("one", "shared alias"),
        "n00000002": ("two",),
        "n00000003": ("three", "shared alias"),
        "n00000004": ("two", "distinct"),
        "n00000005": ("five",),
    }
    candidates, excluded = _build_candidates(
        ["n00000001", "n00000003", "n00000002", "n00000004", "n00000005"],
        ["n00000001"],
        aliases,
    )
    assert candidates == [
        {"synset": "n00000002", "query": "two"},
        {"synset": "n00000005", "query": "five"},
    ]
    assert excluded == {
        "direct_imagenet1k_excluded": 1,
        "imagenet1k_alias_conflict_excluded": 1,
        "normalized_primary_duplicate_excluded": 1,
    }


def test_selection_is_disjoint_and_deterministic() -> None:
    candidates = [
        {"synset": f"n{index:08d}", "query": f"query {index}"}
        for index in range(300)
    ]
    dev, audit = _select_holdouts(candidates)
    assert len(dev) == 101
    assert len(audit) == 90
    assert not ({row["synset"] for row in dev} & {row["synset"] for row in audit})
    assert (dev, audit) == _select_holdouts(candidates)
    first = min(
        candidates,
        key=lambda row: (
            hashlib.sha256(
                f"{ALGORITHM_VERSION}\0new-dev\0{row['synset']}".encode("ascii")
            ).digest(),
            row["synset"],
        ),
    )
    assert dev[0]["synset"] == first["synset"]


def test_official_holdout_freeze_matches_independent_aggregate_contract(
    tmp_path: Path,
) -> None:
    source_dir = _official_timm_info_dir()
    output_root = tmp_path / "holdout"
    manifest = build_holdout_bank(source_dir=source_dir, output_root=output_root)
    persisted = json.loads((output_root / "manifest.json").read_text())

    assert persisted == manifest
    assert manifest["counts"] == dict(EXPECTED_COUNTS)
    assert manifest["synset_tab_query_lf_sha256"] == dict(EXPECTED_RECORD_SHA256)
    assert (
        manifest["preregistered_gate"]["quality_noninferiority_tolerance"]
        == QUALITY_NONINFERIORITY_TOLERANCE
    )
    assert manifest["preregistered_gate"]["audit_requires_dev_promote"] is True
    for name, expected_sha in EXPECTED_SOURCE_SHA256.items():
        copied = output_root / "source_bundle" / name
        assert copied.read_bytes() == (source_dir / name).read_bytes()
        assert hashlib.sha256(copied.read_bytes()).hexdigest() == expected_sha
    for split in ("dev", "audit"):
        payload = json.loads(
            (output_root / f"target_blind_holdout_{split}.json").read_text()
        )
        assert payload["split"] == split
        assert payload["benchmark_vocabulary_opened"] is False
        assert payload["uses_benchmark_vocabulary_for_construction"] is False
        assert hashlib.sha256(_records_bytes(payload["records"])).hexdigest() == (
            EXPECTED_RECORD_SHA256[split]
        )
    with pytest.raises(FileExistsError, match="already exists"):
        build_holdout_bank(source_dir=source_dir, output_root=output_root)
