"""Build fresh source-only ternary language authority without historical fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from radio_gs.utils.immutable_artifacts import write_torch_noclobber
from radio_gs.v3.training.instance_upper_bound import proposal_supports, sha256_file


def _embedding_rows(payload: dict, names: list[str]) -> torch.Tensor:
    cached = [str(value).casefold() for value in payload["queries"]]
    lookup = {name: index for index, name in enumerate(cached)}
    if any(name.casefold() not in lookup for name in names):
        raise ValueError("native language text cache misses query")
    return torch.as_tensor(payload["embeddings"])[
        [lookup[name.casefold()] for name in names]
    ].float()


def _query_names(path: Path) -> list[str]:
    if path.suffix.casefold() == ".json":
        payload = json.loads(path.read_text())
        values = payload.get("query_ids")
        if not isinstance(values, list):
            raise ValueError("native language query manifest misses query_ids")
        return [str(value).strip() for value in values if str(value).strip()]
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _relevancy(
    features: torch.Tensor, text: torch.Tensor, negatives: torch.Tensor
) -> torch.Tensor:
    visual = F.normalize(torch.as_tensor(features).float(), dim=-1, eps=1e-8)
    positive = visual @ F.normalize(text.float(), dim=-1, eps=1e-8).T
    null = (visual @ F.normalize(negatives.float(), dim=-1, eps=1e-8).T).max(1).values
    return ((positive - null[:, None]) * 10.0).sigmoid()


def _support_graph(
    membership: dict, descriptors: torch.Tensor, contexts: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    count = int(membership["num_proposals"])
    supports = proposal_supports(
        membership["row_indices"], membership["proposal_indices"],
        membership["weights"], count,
    )
    support_sets: list[set[int]] = []
    for rows, weights in supports:
        if not rows.numel():
            support_sets.append(set())
            continue
        maximum = weights.max().clamp_min(1e-8)
        support_sets.append(set(rows[weights / maximum >= 0.5].tolist()))
    views = torch.as_tensor(membership["proposal_view_indices"]).long()
    observed = torch.as_tensor(membership["view_observed"]).bool()
    visibility = torch.zeros(count, observed.shape[0])
    for index, support in enumerate(support_sets):
        if support:
            rows = torch.tensor(sorted(support), dtype=torch.long)
            visibility[index] = observed[:, rows].float().mean(1)
    descriptor = F.normalize(torch.as_tensor(descriptors).float(), dim=-1, eps=1e-8)
    context = F.normalize(torch.as_tensor(contexts).float(), dim=-1, eps=1e-8)
    records: list[tuple[int, int, float, bool]] = []
    best: dict[tuple[int, int], tuple[float, int]] = {}
    for left in range(count):
        for right in range(left + 1, count):
            lv, rv = int(views[left]), int(views[right])
            if lv == rv:
                continue
            a, b = support_sets[left], support_sets[right]
            intersection = len(a.intersection(b)) if a and b else 0
            union = len(a) + len(b) - intersection
            jaccard = intersection / max(union, 1)
            overlap = intersection / max(min(len(a), len(b)), 1)
            descriptor_cosine = float(descriptor[left] @ descriptor[right])
            context_cosine = float(context[left] @ context[right])
            strong = (
                (jaccard >= 0.2 or overlap >= 0.7)
                and descriptor_cosine >= 0.8
                and context_cosine >= 0.7
            )
            strength = (
                0.5 * jaccard + 0.3 * overlap
                + 0.1 * descriptor_cosine + 0.1 * context_cosine
            )
            records.append((left, right, strength, strong))
            if strong:
                if strength > best.get((left, rv), (-1.0, -1))[0]:
                    best[(left, rv)] = (strength, right)
                if strength > best.get((right, lv), (-1.0, -1))[0]:
                    best[(right, lv)] = (strength, left)
    lefts, rights, labels, strengths = [], [], [], []
    for left, right, strength, strong in records:
        lv, rv = int(views[left]), int(views[right])
        same = (
            strong
            and best[(left, rv)][1] == right
            and best[(right, lv)][1] == left
        )
        intersection = len(support_sets[left].intersection(support_sets[right]))
        if same:
            label = 1
        elif (
                intersection == 0
                and float(visibility[left, rv]) >= 0.5
                and float(visibility[right, lv]) >= 0.5
            ):
            label = 0
        else:
            label = -1
        lefts.append(left); rights.append(right); labels.append(label); strengths.append(strength)
    return (
        torch.tensor(lefts, dtype=torch.long),
        torch.tensor(rights, dtype=torch.long),
        torch.tensor(labels, dtype=torch.int8),
        torch.tensor(strengths, dtype=torch.float32),
    )


def _same_components(
    count: int,
    left: torch.Tensor,
    right: torch.Tensor,
    label: torch.Tensor,
    *,
    views: torch.Tensor | None = None,
    strength: torch.Tensor | None = None,
) -> torch.Tensor:
    parent = list(range(count))
    members = [{index} for index in range(count)]
    view_sets = (
        [{int(value)} for value in torch.as_tensor(views).tolist()]
        if views is not None else [set() for _ in range(count)]
    )
    conflicts = torch.zeros((count, count), dtype=torch.bool)
    conflicts[left[label == 0], right[label == 0]] = True
    conflicts[right[label == 0], left[label == 0]] = True

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if view_sets[ra].intersection(view_sets[rb]):
            return
        a_rows = torch.tensor(sorted(members[ra]), dtype=torch.long)
        b_rows = torch.tensor(sorted(members[rb]), dtype=torch.long)
        if bool(conflicts[a_rows[:, None], b_rows[None, :]].any()):
            return
        if len(members[ra]) < len(members[rb]):
            ra, rb = rb, ra
        parent[rb] = ra
        members[ra].update(members[rb])
        view_sets[ra].update(view_sets[rb])

    same_rows = torch.where(label == 1)[0]
    if strength is not None and same_rows.numel():
        same_rows = same_rows[torch.as_tensor(strength)[same_rows].argsort(descending=True)]
    for a, b in zip(left[same_rows].tolist(), right[same_rows].tolist()):
        union(a, b)
    roots = [find(index) for index in range(count)]
    lookup = {value: index for index, value in enumerate(sorted(set(roots)))}
    return torch.tensor([lookup[value] for value in roots], dtype=torch.long)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--membership", required=True)
    parser.add_argument("--proposal-teacher", required=True)
    parser.add_argument("--text-embeddings", required=True)
    parser.add_argument("--canonical-negatives", required=True)
    parser.add_argument(
        "--query-names", required=True,
        help="frozen JSON query manifest or newline-separated query list",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        name: Path(value).resolve(strict=True)
        for name, value in (
            ("membership", args.membership), ("proposal_teacher", args.proposal_teacher),
            ("text_embeddings", args.text_embeddings),
            ("canonical_negatives", args.canonical_negatives),
            ("query_names", args.query_names),
        )
    }
    membership = torch.load(paths["membership"], map_location="cpu")
    teacher = torch.load(paths["proposal_teacher"], map_location="cpu")
    text_payload = torch.load(paths["text_embeddings"], map_location="cpu")
    negative_payload = torch.load(paths["canonical_negatives"], map_location="cpu")
    names = _query_names(paths["query_names"])
    if not names or len(names) != len({name.casefold() for name in names}):
        raise ValueError("native language query axis differs")
    descriptors = torch.as_tensor(teacher["descriptors"]).float()
    contexts = torch.as_tensor(teacher["context_descriptors"]).float()
    text = _embedding_rows(text_payload, names)
    negatives = torch.as_tensor(negative_payload["embeddings"]).float()
    descriptor_relevancy = _relevancy(descriptors, text, negatives)
    context_relevancy = _relevancy(contexts, text, negatives)
    views = torch.as_tensor(membership["proposal_view_indices"]).long()
    train = (views % 4 == 1) | (views % 4 == 2)
    left, right, edge_label, edge_strength = _support_graph(
        membership, descriptors, contexts
    )
    components = _same_components(
        int(membership["num_proposals"]), left, right, edge_label,
        views=views, strength=edge_strength,
    )
    best_query = descriptor_relevancy.argmax(1)
    top_queries = descriptor_relevancy.topk(
        min(3, len(names)), dim=1
    ).indices
    top_three = torch.zeros_like(descriptor_relevancy, dtype=torch.bool).scatter_(
        1, top_queries, True
    )
    high_evidence = (
        (descriptor_relevancy >= 0.5)
        & (context_relevancy >= 0.5)
        & F.one_hot(best_query, num_classes=len(names)).bool()
    )
    agreement = (
        ((descriptor_relevancy + context_relevancy) * 0.5 >= 0.5)
        & top_three
        & train[:, None]
    )
    evidence_tier = torch.zeros_like(agreement, dtype=torch.int8)
    evidence_tier[agreement] = 1
    evidence_tier[high_evidence & train[:, None]] = 2
    query_state = torch.full(agreement.shape, -1, dtype=torch.int8)
    seed_count = torch.zeros(len(names), dtype=torch.long)
    component_count = torch.zeros(len(names), dtype=torch.long)
    for query in range(len(names)):
        candidate = torch.where(agreement[:, query])[0]
        if torch.unique(views[candidate]).numel() < 2:
            continue
        accepted_components = torch.unique(components[candidate]).tolist()
        accepted = torch.isin(components, torch.tensor(accepted_components))
        query_state[accepted, query] = 1
        seed_count[query] = int(candidate.numel())
        component_count[query] = len(accepted_components)
        positive_rows = torch.where(accepted)[0]
        different = torch.zeros(len(views), dtype=torch.bool)
        for a, b in zip(left[edge_label == 0].tolist(), right[edge_label == 0].tolist()):
            if bool(accepted[a]): different[b] = True
            if bool(accepted[b]): different[a] = True
        query_state[different & ~accepted, query] = 0
    payload = {
        "schema": "radio_gs.sugm_v3.native_language_authority.v3",
        "scene": membership["scene"],
        "query_names": names,
        "proposal_view_indices": views,
        "query_state": query_state,
        "descriptor_relevancy": descriptor_relevancy,
        "context_relevancy": context_relevancy,
        "seed_evidence_tier": evidence_tier,
        "seed_count": seed_count,
        "positive_component_count": component_count,
        "edge_left": left,
        "edge_right": right,
        "edge_relation": edge_label,
        "edge_strength": edge_strength,
        "metadata": {
            "source_only": True,
            "train_view_residues": [1, 2],
            "dev_and_audit_text_scores_used_for_label_selection": False,
            "historical_field_opened": False,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_metrics_opened": False,
            "positive_seed_rule": "mean_descriptor_context_above_canonical_null_and_descriptor_top3_with_query_evidence_in_two_train_views",
            "evidence_tiers": {"2": "both_modal_views_above_null_and_top1", "1": "mean_above_null_and_top3", "0": "weak_or_unproven"},
            "correspondence_rule": "mutual_best_strong_geometry_descriptor_context_match",
            "component_rule": "descending_strength_union_rejecting_repeated_view_or_explicit_different_conflict",
            "evaluation_label_rule": "conflict_aware_same_component_from_train_seed_only",
            "negative_rule": "explicit_support_different_edge_from_positive_component",
            "unknown_rule": "all_unproven_query_proposal_pairs",
            "inputs": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in paths.items()
            },
        },
    }
    output = Path(args.output).resolve()
    write_torch_noclobber(output, payload)
    print({
        "output": str(output), "sha256": sha256_file(output),
        "queries": len(names), "queries_with_positive_authority": int((seed_count > 0).sum()),
        "positive_pairs": int((query_state == 1).sum()),
        "negative_pairs": int((query_state == 0).sum()),
        "unknown_pairs": int((query_state == -1).sum()),
    })


if __name__ == "__main__":
    main()
