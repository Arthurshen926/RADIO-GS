"""Text-to-fragment isolation for valid region-level LERF keys."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radio_gs.data.lerf_dataset import _read_cameras_binary
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.v4.evaluation.lerf_common import (
    GENERIC_NEGATIVES,
    load_labels as _load_labels,
    load_text_cache as _load_text_cache,
    prototype_max_token_posterior,
)
from radio_gs.v4.evaluation.lerf_fragment_ceiling import _load_fragment_memory
from radio_gs.v4.evaluation.lerf_object_ceiling import _build_carrier, _load_state
from radio_gs.v4.evaluation.lerf_text_cold_evaluator import (
    _evaluate_masks,
    _load_fragment_prototypes,
    compose_object_queries,
    prototype_cosine_set_posterior,
)
from radio_gs.v4.geometry.fuse_lerf_moge3 import _read_images
from radio_gs.v4.query import QueryPacket


REPORT_SCHEMA = "radio_gs.surface_object_memory_v4.lerf_fragment_text_evaluation.v1"


def compose_ranked_fragment_union(
    membership: torch.Tensor,
    token_score: torch.Tensor,
    *,
    maximum_query_tokens: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compose a query-selected fragment set without top-2 codebook loss.

    This is intentionally a diagnostic readout rather than the deployable v4
    object posterior.  It uses no labels, but it fixes every retained fragment
    probability to one so that the experiment isolates text ranking plus raw
    proposal extent from probability calibration and object association.
    Overlapping soft memberships are combined by a probabilistic union.
    """

    surface = torch.as_tensor(membership, dtype=torch.float32).cpu()
    score = torch.as_tensor(token_score, dtype=torch.float32).cpu()
    if surface.ndim != 2 or score.ndim != 2:
        raise ValueError("fragment membership and query score must be matrices")
    if surface.shape[1] != score.shape[1]:
        raise ValueError("fragment membership and query score axes differ")
    if not 0 < maximum_query_tokens <= surface.shape[1]:
        raise ValueError("ranked fragment capacity is outside the fragment axis")
    if not bool(torch.isfinite(surface).all()) or bool(
        ((surface < 0) | (surface > 1)).any()
    ):
        raise ValueError("fragment membership must be finite and lie in [0,1]")
    if not bool(torch.isfinite(score).all()):
        raise ValueError("fragment query scores must be finite")

    selected = torch.topk(
        score,
        k=maximum_query_tokens,
        dim=1,
        largest=True,
        sorted=True,
    ).indices
    posterior = []
    for query_index in range(score.shape[0]):
        chosen = surface[:, selected[query_index]]
        posterior.append(1.0 - torch.prod(1.0 - chosen, dim=1))
    element_probability = torch.stack(posterior, dim=-1)
    return element_probability, {
        "selection_mode": "multi_instance_ranked_fragment_set_diagnostic",
        "assignment_compression": "none_raw_fragment_membership",
        "composition": "probabilistic_union",
        "query_token_capacity": int(maximum_query_tokens),
        "retained_fragment_probability": 1.0,
        "calibration_isolated": True,
        "promotion_eligible": False,
        "element_probability_maximum": float(element_probability.max()),
    }


def compose_local_region_queries(
    membership: torch.Tensor,
    token_probability: torch.Tensor,
    *,
    fragment_chunk_size: int = 64,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Read valid regional evidence as a local semantic surface field.

    A local-semantic query is not an object-token query, so the object-codebook
    top-2 mixture contract does not apply.  Each element keeps the strongest
    compatible source-region response.  Chunking avoids materializing the
    full ``E x F x Q`` tensor, and max avoids double-counting nested SAM masks
    as conditionally independent observations.
    """

    query_packet = QueryPacket("local_semantic")
    surface = torch.as_tensor(membership, dtype=torch.float32).cpu()
    probability = torch.as_tensor(token_probability, dtype=torch.float32).cpu()
    if surface.ndim != 2 or probability.ndim != 2:
        raise ValueError("local region membership and probability must be matrices")
    if surface.shape[1] != probability.shape[1]:
        raise ValueError("local region membership and probability axes differ")
    if fragment_chunk_size <= 0:
        raise ValueError("local region fragment chunk size must be positive")
    if not bool(torch.isfinite(surface).all()) or not bool(torch.isfinite(probability).all()):
        raise ValueError("local region evidence must be finite")
    if bool(((surface < 0) | (surface > 1)).any()) or bool(
        ((probability < 0) | (probability > 1)).any()
    ):
        raise ValueError("local region evidence must lie in [0,1]")

    posterior = torch.zeros((surface.shape[0], probability.shape[0]))
    for start in range(0, surface.shape[1], fragment_chunk_size):
        stop = min(start + fragment_chunk_size, surface.shape[1])
        response = surface[:, start:stop, None] * probability.T[None, start:stop]
        posterior = torch.maximum(posterior, response.max(dim=1).values)
    return posterior, {
        "selection_mode": query_packet.selection_mode.value,
        "object_codebook_used": False,
        "assignment_compression": "none_raw_fragment_membership",
        "composition": "maximum_valid_source_region_response",
        "nested_region_independence_assumed": False,
        "fragment_chunk_size": int(fragment_chunk_size),
        "element_probability_maximum": float(posterior.max()),
    }


def oracle_fragment_retrieval(
    token_probability: torch.Tensor,
    categories: list[str],
    oracle_categories: dict[str, Any],
    *,
    capacity: int,
) -> dict[str, Any]:
    probability = torch.as_tensor(token_probability, dtype=torch.float32).cpu()
    if probability.shape[0] != len(categories):
        raise ValueError("text probabilities and category inventory do not align")
    per_category = {}
    reciprocal_ranks = []
    hits = {1: [], 3: [], 8: []}
    key = f"stable_top{capacity}_token_ids"
    for query_id, category in enumerate(categories):
        selected = {int(value) for value in oracle_categories[category][key]}
        ranking = torch.argsort(probability[query_id], descending=True, stable=True).tolist()
        best_rank = min(
            (rank + 1 for rank, fragment_id in enumerate(ranking) if fragment_id in selected),
            default=None,
        )
        if best_rank is not None:
            reciprocal_ranks.append(1.0 / best_rank)
        for count in hits:
            hits[count].append(any(fragment_id in selected for fragment_id in ranking[:count]))
        per_category[category] = {
            "oracle_fragment_count": len(selected),
            "best_oracle_fragment_rank": best_rank,
            "top1_fragment_id": ranking[0],
            "top1_probability": float(probability[query_id, ranking[0]]),
        }
    return {
        "recall_at_1": float(np.mean(hits[1])),
        "recall_at_3": float(np.mean(hits[3])),
        "recall_at_8": float(np.mean(hits[8])),
        "mean_reciprocal_rank": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "per_category": per_category,
        "oracle_definition": f"greedy stable top-{capacity} extent fragments",
    }


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.cpu_threads <= 0:
        raise ValueError("cpu thread count must be positive")
    torch.set_num_threads(int(args.cpu_threads))
    state_path = Path(args.scene_state)
    state = _load_state(state_path, expected_sha256=args.expected_scene_state_sha256)
    fragment_path = Path(args.fragment_memory)
    memory = _load_fragment_memory(
        fragment_path,
        expected_sha256=args.expected_fragment_memory_sha256,
        scene_state_sha256=state["scene_state_sha256"],
    )
    if state["scene_label"] != args.scene_label or memory["scene_label"] != args.scene_label:
        raise ValueError("scene labels differ across fragment text inputs")
    flat, _legacy_token_ids, semantic_audit = _load_fragment_prototypes(
        Path(args.fragment_language_manifest), state=state
    )
    fragment_count = int(memory["fragment_positive"].shape[0])
    if flat.shape[0] != 2 * fragment_count:
        raise ValueError("fragment language keys differ from fragment surface memory")
    prototype_token_ids = torch.arange(fragment_count).repeat(2)

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
    relative_probability, _ = prototype_max_token_posterior(
        flat.to(device),
        prototype_token_ids.to(device),
        query_text,
        negative_text,
        num_tokens=fragment_count,
        temperature=float(args.query_temperature),
        null_similarity=float(args.null_similarity),
    )
    set_probability = prototype_cosine_set_posterior(
        flat.to(device),
        prototype_token_ids.to(device),
        query_text,
        num_tokens=fragment_count,
        temperature=float(args.query_temperature),
        set_mass=float(args.query_set_mass),
        null_similarity=float(args.null_similarity),
    )
    probability_variants = {
        "generic_negative_independent_sigmoid": relative_probability.cpu(),
        "cosine_query_set_mass": set_probability.cpu(),
    }
    evaluations = {}
    for score_name, token_probability in probability_variants.items():
        evaluations[score_name] = {}
        local_probability, local_composition = compose_local_region_queries(
            memory["fragment_positive"].T.float(), token_probability
        )
        evaluations[score_name]["local_surface_region_max"] = {
            "composition": local_composition,
            "metrics": _evaluate_masks(
                carrier=carrier,
                element_probability=local_probability,
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
        for name, capacity in (("all_fragments", None), ("top3_diagnostic", 3)):
            element_probability, composition = compose_object_queries(
                memory["fragment_positive"].T.float(),
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
                memory["fragment_positive"].T.float(),
                token_probability,
                maximum_query_tokens=capacity,
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
    if args.fragment_ceiling:
        ceiling_path = Path(args.fragment_ceiling).resolve(strict=True)
        ceiling = json.loads(ceiling_path.read_text())
        if (
            ceiling.get("fragment_memory_sha256") != memory["memory_sha256"]
            or ceiling.get("scene_state_sha256") != state["scene_state_sha256"]
        ):
            raise ValueError("fragment ceiling is not bound to the evaluated memory")
        capacity = int(ceiling["maximum_fragments"])
        retrieval = {
            name: oracle_fragment_retrieval(
                probability, categories, ceiling["ceiling"]["categories"],
                capacity=capacity,
            )
            for name, probability in probability_variants.items()
        }
        retrieval.update({
            "fragment_ceiling": str(ceiling_path),
            "fragment_ceiling_sha256": sha256_file(ceiling_path),
            "diagnostic_uses_benchmark_extent_oracle": True,
        })

    report = {
        "schema": REPORT_SCHEMA,
        "development_only": True,
        "promotion_eligible": False,
        "semantic_contract_valid": bool(semantic_audit["semantic_contract_valid"]),
        "scene_label": args.scene_label,
        "scene_state": str(state_path.resolve()),
        "scene_state_sha256": state["scene_state_sha256"],
        "fragment_memory": str(fragment_path.resolve()),
        "fragment_memory_sha256": memory["memory_sha256"],
        "cold_loaded_in_independent_process": True,
        "semantic_memory": semantic_audit,
        "query_contract": {
            "selection_mode": "multi_instance",
            "temperature": float(args.query_temperature),
            "null_similarity": float(args.null_similarity),
            "probability_calibrations": [
                "fixed_generic_negative_sigmoid_development",
                "cosine_query_set_mass",
            ],
            "rank_union_diagnostics": {
                "capacities": [1, 3, 8],
                "uses_benchmark_labels_for_selection": False,
                "isolates": "text_ranking_plus_raw_fragment_extent",
                "deployment_readout": False,
            },
            "query_set_mass": float(args.query_set_mass),
            "benchmark_threshold_tuned": False,
        },
        "text_query_cache": str(Path(args.text_query_cache).resolve(strict=True)),
        "text_query_cache_sha256": sha256_file(Path(args.text_query_cache)),
        "negative_text_query_cache": str(
            Path(args.negative_text_query_cache).resolve(strict=True)
        ),
        "negative_text_query_cache_sha256": sha256_file(
            Path(args.negative_text_query_cache)
        ),
        "raster_shape": [height, width],
        "pixel_threshold": float(args.pixel_threshold),
        "evaluations": evaluations,
        "oracle_retrieval_diagnostic": retrieval,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-state", required=True)
    parser.add_argument("--expected-scene-state-sha256", required=True)
    parser.add_argument("--fragment-memory", required=True)
    parser.add_argument("--expected-fragment-memory-sha256", required=True)
    parser.add_argument("--fragment-language-manifest", required=True)
    parser.add_argument("--fragment-ceiling")
    parser.add_argument("--scene-root", required=True)
    parser.add_argument("--scene-label", required=True)
    parser.add_argument("--label-root", required=True)
    parser.add_argument("--raster-shape", nargs=2, type=int, required=True, metavar=("H", "W"))
    parser.add_argument("--text-query-cache", required=True)
    parser.add_argument("--negative-text-query-cache", required=True)
    parser.add_argument("--query-temperature", type=float, default=0.03)
    parser.add_argument("--query-set-mass", type=float, default=3.0)
    parser.add_argument("--null-similarity", type=float, default=0.0)
    parser.add_argument("--pixel-threshold", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "evaluations": {
            score: {key: value["metrics"] for key, value in modes.items()}
            for score, modes in report["evaluations"].items()
        },
        "oracle_retrieval_diagnostic": report["oracle_retrieval_diagnostic"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
