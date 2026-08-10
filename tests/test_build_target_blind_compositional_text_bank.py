from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from radio_gs.scripts.build_target_blind_compositional_text_bank import (
    ATTRIBUTE_LEXICON,
    EXPECTED_SOURCE_SHA256,
    PARTS_BY_LEXICAL_HEAD,
    build_bank,
    build_records,
)


def _official_lemma_path() -> Path:
    candidates = sorted(
        (Path(sys.prefix) / "lib").glob(
            "python*/site-packages/timm/data/_info/imagenet_synset_to_lemma.txt"
        )
    )
    for candidate in candidates:
        if hashlib.sha256(candidate.read_bytes()).hexdigest() == (
            EXPECTED_SOURCE_SHA256["imagenet_synset_to_lemma.txt"]
        ):
            return candidate
    pytest.skip("pinned timm ImageNet/WordNet lemma source is unavailable")


def test_structure_generation_is_deterministic_and_split_local() -> None:
    primary = [
        {"synset": "n00000001", "query": "alpha chair", "split": "fit"},
        {"synset": "n00000002", "query": "beta chair", "split": "fit"},
        {"synset": "n00000003", "query": "gamma device", "split": "dev"},
    ]
    aliases = {
        "n00000001": ("alpha chair", "first container"),
        "n00000002": ("beta chair", "second container"),
        "n00000003": ("gamma device", "third appliance"),
    }
    first = build_records(primary, aliases)
    second = build_records(primary, aliases)
    assert first == second
    queries, relations, counts = first
    assert {row["structure"] for row in queries}.issuperset(
        {
            "object_noun",
            "color_plus_noun",
            "material_plus_noun",
            "shape_plus_noun",
            "adjective_plus_noun",
            "part_of_object",
            "synonym_alias_1",
        }
    )
    assert counts["lexical_sibling_relations"] == 2
    for relation in relations:
        if relation["relation"] == "lexical_head_sibling_contrast":
            assert relation["split"] == "fit"
            assert relation["shared_lexical_head"] == "chair"
    part_queries = [
        row["query"] for row in queries if row["structure"] == "part_of_object"
    ]
    assert len(part_queries) == 2
    assert all(query.endswith(" chair") for query in part_queries)
    assert len({row["query"] for row in queries}) == len(queries)


def test_official_bank_is_reproducible_and_target_blind(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    primary = (
        repository / "paper/artifacts/target_blind_imagenet1k_primary_text_bank_v1.json"
    )
    lemma = _official_lemma_path()
    output = tmp_path / "compositional.json"
    manifest_output = tmp_path / "compositional.manifest.json"
    manifest = build_bank(
        primary_bank=primary,
        imagenet_synset_to_lemma=lemma,
        output=output,
        manifest_output=manifest_output,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["benchmark_vocabulary_opened"] is False
    assert payload["uses_benchmark_vocabulary_for_construction"] is False
    assert payload["target_metrics_computed"] is False
    assert payload["attribute_lexicon"] == {
        key: list(value) for key, value in ATTRIBUTE_LEXICON.items()
    }
    assert payload["parts_by_lexical_head"] == {
        key: list(value) for key, value in PARTS_BY_LEXICAL_HEAD.items()
    }
    assert (
        payload["recommended_pilot_strata"]["counterfactual_attributes_collective"]
        == 0.30
    )
    assert manifest["counts"]["object_noun"] == 997
    assert manifest["counts"]["query_records"] > 5700
    assert manifest["counts"]["synonym_relations"] > 500
    assert manifest["counts"]["lexical_sibling_relations"] > 50
    assert manifest["counts"]["part_of_object"] < 200
    assert (
        manifest["counts"]["random_part_records_removed_vs_unrestricted_cross_product"]
        > 800
    )
    for row in payload["query_records"]:
        if row["structure"] == "part_of_object":
            assert row["query"].split()[-1] in PARTS_BY_LEXICAL_HEAD
    assert (
        manifest["canonical_json"]["sha256"]
        == hashlib.sha256(output.read_bytes()).hexdigest()
    )
    assert json.loads(manifest_output.read_text(encoding="utf-8")) == manifest

    with pytest.raises(FileExistsError, match="new paths"):
        build_bank(
            primary_bank=primary,
            imagenet_synset_to_lemma=lemma,
            output=output,
            manifest_output=manifest_output,
        )
