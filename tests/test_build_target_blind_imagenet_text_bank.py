from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from radio_gs.scripts.build_target_blind_imagenet_text_bank import (
    ALGORITHM_VERSION,
    EXPECTED_COUNTS,
    EXPECTED_SOURCE_SHA256,
    _build_records,
    _split_line_stream,
    build_bank,
    normalize_primary_alias,
    split_for_synset,
)


def _official_timm_info_dir() -> Path:
    candidates = sorted(
        (Path(sys.prefix) / "lib").glob(
            "python*/site-packages/timm/data/_info"
        )
    )
    for candidate in candidates:
        if all((candidate / name).is_file() for name in EXPECTED_SOURCE_SHA256):
            return candidate
    pytest.skip("pinned timm ImageNet metadata is not installed")


def test_primary_alias_normalization_and_first_wins() -> None:
    assert normalize_primary_alias("  Jack-o'-Lantern__DOG!!  ") == "jackolantern dog"
    synsets = ["n00000001", "n00000002", "n00000003"]
    records, duplicates = _build_records(
        synsets,
        {
            "n00000001": "Blue_Crane, alternate",
            "n00000002": "blue crane, second alternate",
            "n00000003": "Cardigan!!!, sweater",
        },
    )

    assert [(row["synset"], row["query"]) for row in records] == [
        ("n00000001", "blue crane"),
        ("n00000003", "cardigan"),
    ]
    assert duplicates == [
        {
            "dropped_synset": "n00000002",
            "first_synset": "n00000001",
            "query": "blue crane",
        }
    ]


def test_split_contract_uses_first_eight_digest_bytes_big_endian() -> None:
    synset = "n01440764"
    digest = hashlib.sha256(
        ALGORITHM_VERSION.encode("ascii") + b"\0" + synset.encode("ascii")
    ).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10
    expected = "fit" if bucket <= 7 else "dev" if bucket == 8 else "audit"

    assert split_for_synset(synset) == expected


def test_official_bank_is_exact_target_blind_and_reproducible(tmp_path: Path) -> None:
    info_dir = _official_timm_info_dir()
    synsets_path = info_dir / "imagenet_synsets.txt"
    lemma_path = info_dir / "imagenet_synset_to_lemma.txt"
    output = tmp_path / "bank.json"
    manifest_output = tmp_path / "bank.manifest.json"

    manifest = build_bank(
        imagenet_synsets=synsets_path,
        imagenet_synset_to_lemma=lemma_path,
        output=output,
        manifest_output=manifest_output,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    persisted_manifest = json.loads(manifest_output.read_text(encoding="utf-8"))

    assert payload["benchmark_vocabulary_opened"] is False
    assert manifest["benchmark_vocabulary_opened"] is False
    assert persisted_manifest == manifest
    assert manifest["counts"] == EXPECTED_COUNTS
    assert len(payload["records"]) == 997
    assert manifest["duplicates_first_wins"] == [
        {
            "dropped_synset": "n02963159",
            "first_synset": "n02113186",
            "query": "cardigan",
        },
        {
            "dropped_synset": "n03126707",
            "first_synset": "n02012849",
            "query": "crane",
        },
        {
            "dropped_synset": "n03710721",
            "first_synset": "n03710637",
            "query": "maillot",
        },
    ]
    assert {
        name: value["sha256"] for name, value in manifest["sources"].items()
    } == EXPECTED_SOURCE_SHA256
    assert manifest["canonical_json"]["sha256"] == hashlib.sha256(
        output.read_bytes()
    ).hexdigest()
    assert manifest["builder"]["sha256"] == hashlib.sha256(
        Path(__file__)
        .parents[1]
        .joinpath("radio_gs/scripts/build_target_blind_imagenet_text_bank.py")
        .read_bytes()
    ).hexdigest()
    for split in ("fit", "dev", "audit"):
        assert manifest["split_synset_tab_query_lf_sha256"][split] == (
            hashlib.sha256(_split_line_stream(payload["records"], split)).hexdigest()
        )

    second_output = tmp_path / "bank.second.json"
    second_manifest_output = tmp_path / "bank.second.manifest.json"
    second_manifest = build_bank(
        imagenet_synsets=synsets_path,
        imagenet_synset_to_lemma=lemma_path,
        output=second_output,
        manifest_output=second_manifest_output,
    )
    assert second_output.read_bytes() == output.read_bytes()
    assert (
        second_manifest["split_synset_tab_query_lf_sha256"]
        == manifest["split_synset_tab_query_lf_sha256"]
    )


def test_builder_rejects_unpinned_source_before_writing(tmp_path: Path) -> None:
    synsets_path = tmp_path / "imagenet_synsets.txt"
    lemma_path = tmp_path / "imagenet_synset_to_lemma.txt"
    synsets_path.write_text("n00000001\n", encoding="utf-8")
    lemma_path.write_text("n00000001\tobject\n", encoding="utf-8")
    output = tmp_path / "bank.json"

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_bank(
            imagenet_synsets=synsets_path,
            imagenet_synset_to_lemma=lemma_path,
            output=output,
            manifest_output=tmp_path / "bank.manifest.json",
        )

    assert not output.exists()
