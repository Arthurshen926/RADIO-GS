#!/usr/bin/env python3
"""Build a query-structure bank without opening benchmark vocabulary.

The object and synonym vocabulary comes only from the pinned timm
ImageNet-1K/WordNet metadata.  A small generic English attribute lexicon is
frozen in this source file.  Compositions inherit the source synset split, so
no benchmark text, image, label, mask, or metric is an input to this builder.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import string
from typing import Any, Iterable, Mapping, Sequence


ALGORITHM_VERSION = "imagenet1k-compositional-v2"
ARTIFACT_TYPE = "target_blind_imagenet1k_compositional_text_bank"
EXPECTED_SOURCE_SHA256 = {
    "imagenet_synset_to_lemma.txt": (
        "1b8babda187421a4bde0c9c5a197c36f6bdda962f7ca11ffb2813806cbb2178f"
    ),
    "primary_bank": (
        "2644c8454c12b0d6ca16fc453ee63e5289112172b82b61136e003ddf65a090ab"
    ),
}
_SYNSET = re.compile(r"n[0-9]{8}\Z")
_TRANSLATION = str.maketrans("", "", string.punctuation)

# These are generic visual-language primitives, not a harvested dataset
# vocabulary.  Their exact bytes are audited through the builder SHA-256.
ATTRIBUTE_LEXICON: Mapping[str, tuple[str, ...]] = {
    "color": (
        "black",
        "blue",
        "brown",
        "gray",
        "green",
        "orange",
        "pink",
        "purple",
        "red",
        "white",
        "yellow",
    ),
    "material": (
        "ceramic",
        "fabric",
        "glass",
        "leather",
        "metal",
        "paper",
        "plastic",
        "rubber",
        "stone",
        "wooden",
    ),
    "shape": (
        "circular",
        "curved",
        "cylindrical",
        "flat",
        "oval",
        "rectangular",
        "round",
        "square",
        "triangular",
    ),
    "adjective": (
        "large",
        "small",
        "tall",
        "short",
        "thick",
        "thin",
        "wide",
        "narrow",
        "old",
        "new",
    ),
}
# Exact lexical-head rules are deliberately conservative.  Unlike the
# counterfactual attribute probes above, a part-of phrase asserts a relation
# and is emitted only where the head itself makes that relation high precision.
PARTS_BY_LEXICAL_HEAD: Mapping[str, tuple[str, ...]] = {
    "airplane": ("wing", "tail"),
    "bicycle": ("handlebar", "wheel"),
    "bird": ("beak", "wing"),
    "boat": ("deck", "hull"),
    "book": ("cover", "page"),
    "bottle": ("cap", "neck"),
    "bus": ("door", "wheel"),
    "camera": ("body", "lens"),
    "car": ("door", "wheel"),
    "chair": ("back", "leg", "seat"),
    "clock": ("face", "hand"),
    "cup": ("handle", "rim"),
    "door": ("frame", "handle"),
    "fish": ("fin", "tail"),
    "guitar": ("body", "neck"),
    "hat": ("brim", "crown"),
    "lamp": ("base", "shade"),
    "motorcycle": ("handlebar", "wheel"),
    "mug": ("handle", "rim"),
    "pan": ("handle",),
    "pot": ("handle", "lid"),
    "ship": ("deck", "hull"),
    "shoe": ("heel", "sole"),
    "table": ("leg", "top"),
    "telephone": ("keypad", "receiver"),
    "truck": ("door", "wheel"),
    "violin": ("body", "neck"),
    "window": ("frame", "pane"),
}
LEXICON_PROVENANCE = {
    "kind": "fixed_generic_english_visual_relation_seed",
    "selection": "small task-agnostic visual attributes and meronyms",
    "benchmark_vocabulary_opened": False,
    "uses_benchmark_vocabulary_for_construction": False,
    "per_scene_or_per_benchmark_selection": False,
}
COMPOSITION_SEMANTICS = {
    "color_plus_noun": "counterfactual_attribute_probe_not_assumed_true",
    "material_plus_noun": "counterfactual_attribute_probe_not_assumed_true",
    "shape_plus_noun": "counterfactual_attribute_probe_not_assumed_true",
    "adjective_plus_noun": "counterfactual_attribute_probe_not_assumed_true",
    "part_of_object": "asserted_only_by_exact_high_precision_lexical_head_rule",
}
RECOMMENDED_PILOT_STRATA = {
    "object_noun": 0.25,
    "synonym_relation": 0.20,
    "lexical_sibling_relation": 0.20,
    "counterfactual_attributes_collective": 0.30,
    "high_precision_part_of": 0.05,
    "counterfactual_attribute_internal_rule": (
        "uniform_across_color_material_shape_adjective"
    ),
    "purpose": (
        "prevent numerous low-response counterfactual compositions from "
        "dominating absolute-response distillation"
    ),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def normalize_alias(value: str) -> str:
    return " ".join(value.replace("_", " ").translate(_TRANSLATION).casefold().split())


def _read_aliases(path: Path) -> dict[str, tuple[str, ...]]:
    aliases: dict[str, tuple[str, ...]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            fields = raw.rstrip("\r\n").split("\t", maxsplit=1)
            if len(fields) != 2 or _SYNSET.fullmatch(fields[0]) is None:
                raise ValueError(f"{path}:{line_number}: invalid WordNet lemma row")
            normalized = tuple(
                item
                for item in (normalize_alias(value) for value in fields[1].split(","))
                if item
            )
            if not normalized or fields[0] in aliases:
                raise ValueError(f"{path}:{line_number}: duplicate or empty lemma row")
            aliases[fields[0]] = normalized
    return aliases


def _choose(synset: str, kind: str, values: Sequence[str]) -> str:
    digest = hashlib.sha256(
        f"{ALGORITHM_VERSION}\0{kind}\0{synset}".encode("ascii")
    ).digest()
    return values[int.from_bytes(digest[:8], "big") % len(values)]


def _query_record(
    *, synset: str, split: str, structure: str, query: str
) -> dict[str, str]:
    return {
        "record_id": f"{synset}:{structure}",
        "source_synset": synset,
        "split": split,
        "structure": structure,
        "query": query,
    }


def build_records(
    primary_records: Iterable[Mapping[str, str]],
    aliases_by_synset: Mapping[str, Sequence[str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, int]]:
    """Build deterministic split-local query and semantic-relation records."""

    primary = [dict(row) for row in primary_records]
    if not primary:
        raise ValueError("primary bank is empty")
    reserved_primary_queries = [
        normalize_alias(str(row.get("query", ""))) for row in primary
    ]
    if any(not query for query in reserved_primary_queries) or len(
        set(reserved_primary_queries)
    ) != len(reserved_primary_queries):
        raise ValueError("primary bank query is not globally unique")
    reserved_primary = set(reserved_primary_queries)
    seen_queries: set[str] = set()
    query_records: list[dict[str, str]] = []
    synonym_relations: list[dict[str, str]] = []
    primary_by_split: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in primary:
        synset = str(row.get("synset", ""))
        query = normalize_alias(str(row.get("query", "")))
        split = str(row.get("split", ""))
        aliases = tuple(aliases_by_synset.get(synset, ()))
        if (
            _SYNSET.fullmatch(synset) is None
            or split not in {"fit", "dev", "audit"}
            or not query
            or not aliases
            or normalize_alias(aliases[0]) != query
        ):
            raise ValueError("primary bank and WordNet lemma identity differ")
        if query in seen_queries:
            raise ValueError("primary bank query is not globally unique")
        seen_queries.add(query)
        query_records.append(
            _query_record(
                synset=synset, split=split, structure="object_noun", query=query
            )
        )
        primary_by_split[split].append((synset, query))

        for kind in ("color", "material", "shape", "adjective"):
            modifier = _choose(synset, kind, ATTRIBUTE_LEXICON[kind])
            composed = f"{modifier} {query}"
            if composed not in seen_queries and composed not in reserved_primary:
                seen_queries.add(composed)
                query_records.append(
                    _query_record(
                        synset=synset,
                        split=split,
                        structure=f"{kind}_plus_noun",
                        query=composed,
                    )
                )
        head = query.split()[-1]
        parts = PARTS_BY_LEXICAL_HEAD.get(head)
        if parts is not None:
            part = _choose(synset, "part_of_object", parts)
            composed = f"the {part} of the {query}"
            if composed not in seen_queries and composed not in reserved_primary:
                seen_queries.add(composed)
                query_records.append(
                    _query_record(
                        synset=synset,
                        split=split,
                        structure="part_of_object",
                        query=composed,
                    )
                )

        alias_rank = 0
        for raw_alias in aliases[1:]:
            alias = normalize_alias(raw_alias)
            if (
                not alias
                or alias in seen_queries
                or alias in reserved_primary
                or len(alias.split()) > 4
                or not alias.isascii()
            ):
                continue
            alias_rank += 1
            if alias_rank > 3:
                break
            seen_queries.add(alias)
            query_records.append(
                _query_record(
                    synset=synset,
                    split=split,
                    structure=f"synonym_alias_{alias_rank}",
                    query=alias,
                )
            )
            synonym_relations.append(
                {
                    "record_id": f"{synset}:synonym:{alias_rank}",
                    "split": split,
                    "relation": "synonym",
                    "left_query": query,
                    "right_query": alias,
                    "left_synset": synset,
                    "right_synset": synset,
                }
            )

    sibling_relations: list[dict[str, str]] = []
    for split, records in sorted(primary_by_split.items()):
        by_head: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for synset, query in records:
            tokens = query.split()
            if len(tokens) >= 2:
                by_head[tokens[-1]].append((synset, query))
        for head, group in sorted(by_head.items()):
            ordered = sorted(set(group))
            if len(ordered) < 2:
                continue
            for left, right in zip(ordered, ordered[1:] + ordered[:1]):
                sibling_relations.append(
                    {
                        "record_id": f"{left[0]}:lexical_sibling:{right[0]}",
                        "split": split,
                        "relation": "lexical_head_sibling_contrast",
                        "shared_lexical_head": head,
                        "left_query": left[1],
                        "right_query": right[1],
                        "left_synset": left[0],
                        "right_synset": right[0],
                    }
                )

    query_records.sort(
        key=lambda row: (row["split"], row["source_synset"], row["record_id"])
    )
    relation_records = sorted(
        synonym_relations + sibling_relations,
        key=lambda row: (row["split"], row["record_id"]),
    )
    counts = Counter(row["structure"] for row in query_records)
    counts.update(
        {
            "query_records": len(query_records),
            "relation_records": len(relation_records),
            "synonym_relations": len(synonym_relations),
            "lexical_sibling_relations": len(sibling_relations),
            "random_part_records_removed_vs_unrestricted_cross_product": (
                len(primary) - counts["part_of_object"]
            ),
        }
    )
    return query_records, relation_records, dict(sorted(counts.items()))


def build_bank(
    *,
    primary_bank: Path,
    imagenet_synset_to_lemma: Path,
    output: Path,
    manifest_output: Path,
) -> dict[str, Any]:
    primary_bank = primary_bank.expanduser().resolve()
    imagenet_synset_to_lemma = imagenet_synset_to_lemma.expanduser().resolve()
    output = output.expanduser().resolve()
    manifest_output = manifest_output.expanduser().resolve()
    if output == manifest_output or output.exists() or manifest_output.exists():
        raise FileExistsError("output and manifest must be distinct new paths")
    observed = {
        "primary_bank": _sha256_file(primary_bank),
        "imagenet_synset_to_lemma.txt": _sha256_file(imagenet_synset_to_lemma),
    }
    if observed != EXPECTED_SOURCE_SHA256:
        raise ValueError("pinned target-blind text source SHA-256 differs")
    primary = json.loads(primary_bank.read_text(encoding="utf-8"))
    if (
        primary.get("benchmark_vocabulary_opened") is not False
        or primary.get("artifact_type") != "target_blind_imagenet1k_primary_text_bank"
    ):
        raise ValueError("primary bank is not sealed target-blind")
    aliases = _read_aliases(imagenet_synset_to_lemma)
    queries, relations, counts = build_records(primary["records"], aliases)
    payload = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "algorithm_version": ALGORITHM_VERSION,
        "benchmark_vocabulary_opened": False,
        "uses_benchmark_vocabulary_for_construction": False,
        "benchmark_images_opened": False,
        "benchmark_labels_opened": False,
        "benchmark_masks_opened": False,
        "target_metrics_computed": False,
        "prompt_templates": ["{query}"],
        "lexicon_provenance": LEXICON_PROVENANCE,
        "attribute_lexicon": {
            key: list(value) for key, value in ATTRIBUTE_LEXICON.items()
        },
        "parts_by_lexical_head": {
            key: list(value) for key, value in PARTS_BY_LEXICAL_HEAD.items()
        },
        "composition_semantics": COMPOSITION_SEMANTICS,
        "recommended_pilot_strata": RECOMMENDED_PILOT_STRATA,
        "query_records": queries,
        "relation_records": relations,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_json_bytes(payload))
    builder = Path(__file__).resolve()
    manifest = {
        "schema_version": 1,
        "artifact_type": f"{ARTIFACT_TYPE}_manifest",
        "algorithm_version": ALGORITHM_VERSION,
        "benchmark_vocabulary_opened": False,
        "uses_benchmark_vocabulary_for_construction": False,
        "sources": {
            "primary_bank": {
                "path": str(primary_bank),
                "sha256": observed["primary_bank"],
            },
            "imagenet_synset_to_lemma.txt": {
                "path": str(imagenet_synset_to_lemma),
                "sha256": observed["imagenet_synset_to_lemma.txt"],
            },
        },
        "builder": {"path": str(builder), "sha256": _sha256_file(builder)},
        "canonical_json": {"path": str(output), "sha256": _sha256_file(output)},
        "counts": counts,
        "split_query_counts": dict(
            sorted(Counter(row["split"] for row in queries).items())
        ),
        "split_relation_counts": dict(
            sorted(Counter(row["split"] for row in relations).items())
        ),
    }
    manifest_output.write_bytes(_canonical_json_bytes(manifest))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-bank", type=Path, required=True)
    parser.add_argument("--imagenet-synset-to-lemma", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = build_bank(
        primary_bank=args.primary_bank,
        imagenet_synset_to_lemma=args.imagenet_synset_to_lemma,
        output=args.output,
        manifest_output=args.manifest_output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
