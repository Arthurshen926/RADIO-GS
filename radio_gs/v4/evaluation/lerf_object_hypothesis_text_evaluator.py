"""Cold-load text evaluation for the two-level LERF object memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radio_gs.data.lerf_dataset import _read_cameras_binary
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.v4.contracts.build_lerf_object_hypotheses import (
    validate_hypothesis_memory,
)
from radio_gs.v4.evaluation.lerf_common import (
    GENERIC_NEGATIVES,
    load_labels as _load_labels,
    load_text_cache as _load_text_cache,
    prototype_max_token_posterior,
)
from radio_gs.v4.evaluation.lerf_object_ceiling import _build_carrier, _load_state
from radio_gs.v4.evaluation.lerf_fragment_text_evaluator import (
    compose_ranked_fragment_union,
)
from radio_gs.v4.evaluation.lerf_text_cold_evaluator import (
    _evaluate_masks,
    compose_object_queries,
    prototype_cosine_set_posterior,
)
from radio_gs.v4.geometry.fuse_lerf_moge3 import _read_images


REPORT_SCHEMA = "radio_gs.surface_object_memory_v4.lerf_object_hypothesis_text_evaluation.v1"


def oracle_hypothesis_retrieval(
    token_probability: torch.Tensor,
    categories: list[str],
    oracle_categories: dict[str, Any],
) -> dict[str, Any]:
    """Measure text ranking against each category's oracle top-3 hypotheses."""

    probability = torch.as_tensor(token_probability, dtype=torch.float32).cpu()
    if probability.shape[0] != len(categories):
        raise ValueError("text probabilities and category inventory do not align")
    hits = {1: [], 3: [], 8: []}
    reciprocal_ranks = []
    per_category = {}
    for query_id, category in enumerate(categories):
        selected = {
            int(value)
            for value in oracle_categories[category]["stable_top3_token_ids"]
        }
        ranking = torch.argsort(
            probability[query_id], descending=True, stable=True
        ).tolist()
        best_rank = min(
            (rank + 1 for rank, token_id in enumerate(ranking) if token_id in selected),
            default=None,
        )
        if best_rank is not None:
            reciprocal_ranks.append(1.0 / best_rank)
        for count in hits:
            hits[count].append(any(token_id in selected for token_id in ranking[:count]))
        per_category[category] = {
            "oracle_hypothesis_count": len(selected),
            "best_oracle_hypothesis_rank": best_rank,
            "top1_hypothesis_id": ranking[0],
        }
    return {
        "recall_at_1": float(np.mean(hits[1])),
        "recall_at_3": float(np.mean(hits[3])),
        "recall_at_8": float(np.mean(hits[8])),
        "mean_reciprocal_rank": (
            float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
        ),
        "per_category": per_category,
        "oracle_definition": "greedy stable top-3 raw object-hypothesis extent",
    }


def _flatten_object_prototypes(
    hypothesis: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        "object_prototype_descriptors" not in hypothesis
        or "object_prototype_valid" not in hypothesis
    ):
        raise ValueError("object hypothesis memory lacks aligned region prototypes")
    descriptors = torch.as_tensor(
        hypothesis.get("object_prototype_descriptors"), dtype=torch.float32
    ).cpu()
    valid = torch.as_tensor(hypothesis.get("object_prototype_valid"), dtype=torch.bool).cpu()
    if descriptors.ndim != 3 or valid.shape != descriptors.shape[:2]:
        raise ValueError("object hypothesis memory lacks aligned region prototypes")
    token_ids = torch.arange(descriptors.shape[0])[:, None].expand_as(valid)
    return descriptors[valid], token_ids[valid]


def prototype_fragment_consensus_scores(
    hypothesis: dict[str, Any],
    queries: torch.Tensor,
    *,
    consensus_fragments: int = 2,
) -> torch.Tensor:
    """Pool region prototypes through cross-fragment agreement per hypothesis.

    Masked and context crops of one source fragment first compete by max.  The
    strongest distinct source views then vote by mean when view receipts are
    available. Legacy memories retain distinct-fragment replay. This prevents two
    crop variants of one accidental match from masquerading as multiview
    consensus.  Single-fragment hypotheses remain supported rather than being
    dropped by a hard gate.
    """

    descriptors = torch.as_tensor(
        hypothesis.get("object_prototype_descriptors"), dtype=torch.float32
    )
    valid = torch.as_tensor(hypothesis.get("object_prototype_valid"), dtype=torch.bool)
    fragment_ids = torch.as_tensor(
        hypothesis.get("object_prototype_fragment_ids"), dtype=torch.long
    )
    query = torch.nn.functional.normalize(
        torch.as_tensor(queries, dtype=torch.float32, device=descriptors.device),
        dim=-1,
    )
    if descriptors.ndim != 3 or valid.shape != descriptors.shape[:2]:
        raise ValueError("object hypothesis memory lacks aligned region prototypes")
    if fragment_ids.shape != valid.shape or consensus_fragments <= 0:
        raise ValueError("prototype fragment receipts or consensus capacity are invalid")
    vote_ids = fragment_ids
    if "object_prototype_view_ids" in hypothesis:
        vote_ids = torch.as_tensor(hypothesis["object_prototype_view_ids"], dtype=torch.long, device=descriptors.device)
        if vote_ids.shape != valid.shape or bool((vote_ids[valid] < 0).any()):
            raise ValueError("prototype source-view receipts are invalid")
    descriptors = torch.nn.functional.normalize(descriptors, dim=-1)
    output = descriptors.new_full((query.shape[0], descriptors.shape[0]), -torch.inf)
    for token_id in range(descriptors.shape[0]):
        chosen = valid[token_id]
        ids = vote_ids[token_id, chosen]
        values = query @ descriptors[token_id, chosen].T
        per_fragment = []
        for fragment_id in torch.unique(ids, sorted=True).tolist():
            per_fragment.append(values[:, ids == fragment_id].max(-1).values)
        if per_fragment:
            votes = torch.stack(per_fragment, dim=-1)
            count = min(consensus_fragments, votes.shape[1])
            output[:, token_id] = votes.topk(count, dim=-1).values.mean(-1)
    if not bool(torch.isfinite(output).all()):
        raise ValueError("every object hypothesis needs a finite fragment consensus")
    return output


def prototype_fragment_consensus_posterior(
    hypothesis: dict[str, Any],
    queries: torch.Tensor,
    negatives: torch.Tensor,
    *,
    temperature: float,
    null_similarity: float,
    consensus_fragments: int = 2,
) -> torch.Tensor:
    """Return generic-negative probability after distinct-fragment consensus."""

    positive = prototype_fragment_consensus_scores(
        hypothesis, queries, consensus_fragments=consensus_fragments
    )
    negative = prototype_fragment_consensus_scores(
        hypothesis, negatives, consensus_fragments=consensus_fragments
    ).max(0).values
    negative = torch.maximum(
        negative, torch.full_like(negative, float(null_similarity))
    )
    if temperature <= 0:
        raise ValueError("fragment-consensus temperature must be positive")
    return torch.sigmoid((positive - negative[None]) / float(temperature))


def cosine_consensus_set_posterior(
    similarity: torch.Tensor,
    *,
    temperature: float,
    set_mass: float,
    null_similarity: float,
) -> torch.Tensor:
    """Normalize pre-pooled object similarities over one query candidate set."""

    value = torch.as_tensor(similarity, dtype=torch.float32)
    if value.ndim != 2 or temperature <= 0 or set_mass <= 0:
        raise ValueError("consensus similarities or normalization controls are invalid")
    null = value.new_full((value.shape[0], 1), float(null_similarity))
    posterior = torch.softmax(torch.cat((value, null), -1) / temperature, -1)[:, :-1]
    return (posterior * float(set_mass)).clamp_max(1.0)


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.cpu_threads <= 0:
        raise ValueError("cpu thread count must be positive")
    torch.set_num_threads(int(args.cpu_threads))
    state_path = Path(args.scene_state)
    state = _load_state(state_path, expected_sha256=args.expected_scene_state_sha256)
    hypothesis_path = Path(args.hypothesis_memory).resolve(strict=True)
    if sha256_file(hypothesis_path) != args.expected_hypothesis_memory_sha256:
        raise ValueError("object-hypothesis memory SHA256 differs")
    hypothesis = validate_hypothesis_memory(
        torch.load(hypothesis_path, map_location="cpu", weights_only=False)
    )
    if hypothesis["scene_state_sha256"] != state["scene_state_sha256"]:
        raise ValueError("object-hypothesis memory and scene state are not bound")
    if hypothesis["scene_label"] != args.scene_label or state["scene_label"] != args.scene_label:
        raise ValueError("scene labels differ across text-evaluation inputs")
    semantic_valid = bool(hypothesis.get("appearance_audit", {}).get("valid_for_text_query"))
    if not semantic_valid and not args.allow_invalid_appearance_diagnostic:
        raise ValueError("object prototypes do not satisfy the region-level text contract")
    membership_state = getattr(args, "membership_state", "observed")
    if membership_state == "completed":
        if "completed_membership" not in hypothesis:
            raise ValueError("completed evaluation requires completed hypothesis membership")
        query_membership = hypothesis["completed_membership"]
    elif membership_state == "observed":
        query_membership = hypothesis["observed_membership"]
    else:
        raise ValueError("object-hypothesis membership state is unsupported")

    height, width = (int(value) for value in args.raster_shape)
    carrier = _build_carrier(state)
    scene_root = Path(args.scene_root).resolve(strict=True)
    sparse = scene_root / "sparse" / "0"
    views = _read_images(
        sparse / "images.bin", _read_cameras_binary(sparse / "cameras.bin")
    )
    annotations, categories, source_height, source_width = _load_labels(
        Path(args.label_root).resolve(strict=True), args.scene_label
    )
    device = torch.device(args.device)
    query_text = _load_text_cache(Path(args.text_query_cache), categories, device)
    negative_text = _load_text_cache(
        Path(args.negative_text_query_cache), list(GENERIC_NEGATIVES), device
    )
    prototypes, prototype_token_ids = _flatten_object_prototypes(hypothesis)
    token_count = int(hypothesis["observed_membership"].shape[1])
    relative_probability, _ = prototype_max_token_posterior(
        prototypes.to(device),
        prototype_token_ids.to(device),
        query_text,
        negative_text,
        num_tokens=token_count,
        temperature=float(args.query_temperature),
        null_similarity=float(args.null_similarity),
    )
    set_probability = prototype_cosine_set_posterior(
        prototypes.to(device),
        prototype_token_ids.to(device),
        query_text,
        num_tokens=token_count,
        temperature=float(args.query_temperature),
        set_mass=float(args.query_set_mass),
        null_similarity=float(args.null_similarity),
    )
    consensus_similarity = prototype_fragment_consensus_scores(
        hypothesis, query_text, consensus_fragments=int(args.consensus_fragments)
    )
    consensus_probability = prototype_fragment_consensus_posterior(
        hypothesis,
        query_text,
        negative_text,
        temperature=float(args.query_temperature),
        null_similarity=float(args.null_similarity),
        consensus_fragments=int(args.consensus_fragments),
    )
    consensus_set_probability = cosine_consensus_set_posterior(
        consensus_similarity,
        temperature=float(args.query_temperature),
        set_mass=float(args.query_set_mass),
        null_similarity=float(args.null_similarity),
    )
    probability_variants = {
        "generic_negative_independent_sigmoid": relative_probability.cpu(),
        "cosine_query_set_mass": set_probability.cpu(),
        "generic_negative_fragment_consensus": consensus_probability.cpu(),
        "cosine_fragment_consensus_set_mass": consensus_set_probability.cpu(),
    }
    evaluations = {}
    for score_name, token_probability in probability_variants.items():
        evaluations[score_name] = {}
        for name, capacity in (("all_tokens", None), ("top3_diagnostic", 3)):
            element_probability, composition = compose_object_queries(
                query_membership,
                token_probability,
                maximum_query_tokens=capacity,
            )
            evaluations[score_name][name] = {
                "composition": composition,
                "metrics": _evaluate_masks(
                    carrier=carrier,
                    element_probability=element_probability,
                    views=views,
                    annotations=annotations,
                    categories=categories,
                    source_height=source_height,
                    source_width=source_width,
                    height=height,
                    width=width,
                    pixel_threshold=float(args.pixel_threshold),
                ),
            }
        for capacity in (1, 3, 8):
            element_probability, composition = compose_ranked_fragment_union(
                query_membership,
                token_probability,
                maximum_query_tokens=capacity,
            )
            composition["selection_mode"] = (
                "multi_instance_ranked_object_set_diagnostic"
            )
            evaluations[score_name][f"rank_union_top{capacity}_diagnostic"] = {
                "composition": composition,
                "metrics": _evaluate_masks(
                    carrier=carrier,
                    element_probability=element_probability,
                    views=views,
                    annotations=annotations,
                    categories=categories,
                    source_height=source_height,
                    source_width=source_width,
                    height=height,
                    width=width,
                    pixel_threshold=float(args.pixel_threshold),
                ),
            }

    retrieval = None
    if args.hypothesis_ceiling:
        ceiling_path = Path(args.hypothesis_ceiling).resolve(strict=True)
        ceiling = json.loads(ceiling_path.read_text())
        if (
            ceiling.get("hypothesis_memory_sha256")
            != args.expected_hypothesis_memory_sha256
            or ceiling.get("scene_state_sha256") != state["scene_state_sha256"]
        ):
            raise ValueError("hypothesis ceiling is not bound to the evaluated memory")
        oracle_categories = ceiling["variants"]["raw_independent_token"]["categories"]
        retrieval = {
            name: oracle_hypothesis_retrieval(
                probability, categories, oracle_categories
            )
            for name, probability in probability_variants.items()
        }
        retrieval.update({
            "hypothesis_ceiling": str(ceiling_path),
            "hypothesis_ceiling_sha256": sha256_file(ceiling_path),
            "diagnostic_uses_benchmark_extent_oracle": True,
        })
    report = {
        "schema": REPORT_SCHEMA,
        "development_only": True,
        "promotion_eligible": False,
        "semantic_contract_valid": semantic_valid,
        "scene_label": args.scene_label,
        "scene_state": str(state_path.resolve()),
        "scene_state_sha256": state["scene_state_sha256"],
        "hypothesis_memory": str(hypothesis_path),
        "hypothesis_memory_sha256": args.expected_hypothesis_memory_sha256,
        "cold_loaded_in_independent_process": True,
        "appearance_audit": hypothesis["appearance_audit"],
        "association_audit": hypothesis["association_audit"],
        "membership_state": membership_state,
        "completion_audit": hypothesis.get("completion_audit"),
        "query_contract": {
            "selection_mode": "multi_instance",
            "temperature": float(args.query_temperature),
            "null_similarity": float(args.null_similarity),
            "probability_calibrations": [
                "fixed_generic_negative_sigmoid_development",
                "cosine_query_set_mass",
            ],
            "query_set_mass": float(args.query_set_mass),
            "benchmark_threshold_tuned": False,
            "object_prototypes_retained": int(
                torch.as_tensor(hypothesis["object_prototype_valid"]).sum(-1).max()
            ),
            "distinct_fragment_consensus": int(args.consensus_fragments),
        },
        "text_query_cache": str(Path(args.text_query_cache).resolve(strict=True)),
        "text_query_cache_sha256": sha256_file(Path(args.text_query_cache)),
        "negative_text_query_cache": str(
            Path(args.negative_text_query_cache).resolve(strict=True)
        ),
        "negative_text_query_cache_sha256": sha256_file(
            Path(args.negative_text_query_cache)
        ),
        "oracle_hypothesis_retrieval_diagnostic": retrieval,
        "raster_shape": [height, width],
        "pixel_threshold": float(args.pixel_threshold),
        "evaluations": evaluations,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-state", required=True)
    parser.add_argument("--expected-scene-state-sha256", required=True)
    parser.add_argument("--hypothesis-memory", required=True)
    parser.add_argument("--expected-hypothesis-memory-sha256", required=True)
    parser.add_argument("--hypothesis-ceiling")
    parser.add_argument(
        "--membership-state", choices=("observed", "completed"), default="observed"
    )
    parser.add_argument("--scene-root", required=True)
    parser.add_argument("--scene-label", required=True)
    parser.add_argument("--label-root", required=True)
    parser.add_argument("--raster-shape", nargs=2, type=int, required=True, metavar=("H", "W"))
    parser.add_argument("--text-query-cache", required=True)
    parser.add_argument("--negative-text-query-cache", required=True)
    parser.add_argument("--query-temperature", type=float, default=0.03)
    parser.add_argument("--query-set-mass", type=float, default=3.0)
    parser.add_argument("--consensus-fragments", type=int, default=2)
    parser.add_argument("--null-similarity", type=float, default=0.0)
    parser.add_argument("--pixel-threshold", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--allow-invalid-appearance-diagnostic", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        score: {key: value["metrics"] for key, value in modes.items()}
        for score, modes in report["evaluations"].items()
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
