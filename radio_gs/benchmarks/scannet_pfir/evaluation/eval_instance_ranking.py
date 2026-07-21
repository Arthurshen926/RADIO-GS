"""Track A: image-to-3-D instance ranking."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np


def top_mean(values: np.ndarray, fraction: float = 0.20) -> float:
    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return float("-inf")
    count = max(1, int(np.ceil(scores.size * float(fraction))))
    partition = np.partition(scores, scores.size - count)
    return float(partition[-count:].mean())


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _macro(rows: Sequence[dict[str, Any]], key: str, group_key: str) -> float | None:
    groups: dict[Any, list[float]] = defaultdict(list)
    for row in rows:
        groups[row[group_key]].append(float(row[key]))
    return _mean([float(np.mean(values)) for values in groups.values()])


def evaluate_instance_ranking(
    query_records: Sequence[Mapping[str, Any]],
    scores_by_query: Mapping[str, np.ndarray],
    mesh_instance_ids_by_scene: Mapping[str, np.ndarray],
    *,
    top_fraction: float = 0.20,
) -> dict[str, Any]:
    """Rank all declared candidate instances using GT only inside evaluator."""

    rows: list[dict[str, Any]] = []
    for source in query_records:
        query_id, scene_id = str(source["query_id"]), str(source["scene_id"])
        if query_id not in scores_by_query:
            raise KeyError(f"missing mesh scores for {query_id}")
        mesh_instances = np.asarray(mesh_instance_ids_by_scene[scene_id]).reshape(-1)
        scores = np.asarray(scores_by_query[query_id]).reshape(-1)
        if scores.shape != mesh_instances.shape:
            raise ValueError(f"{query_id}: score/mesh row mismatch")
        candidate_ids = list(map(int, source["candidate_instance_ids_3d"]))
        target = int(source["instance_id_3d"])
        if target not in candidate_ids:
            raise ValueError(f"{query_id}: target absent from candidate set")
        instance_scores = {
            instance_id: top_mean(
                scores[mesh_instances == instance_id], fraction=top_fraction
            )
            for instance_id in candidate_ids
        }
        ordered = sorted(candidate_ids, key=lambda value: (-instance_scores[value], value))
        rank = ordered.index(target) + 1
        class_id = int(source["nyu40_class_id"])
        same_category = int(source.get("same_category_distractor_count", 0)) > 0
        rows.append(
            {
                "query_id": query_id,
                "scene_id": scene_id,
                "nyu40_class_id": class_id,
                "rank": rank,
                "recall_at_1": float(rank <= 1),
                "recall_at_5": float(rank <= 5),
                "reciprocal_rank": 1.0 / rank,
                "same_category": same_category,
                "target_score": instance_scores[target],
                "ordered_instance_ids": ordered,
            }
        )
    same = [row["recall_at_1"] for row in rows if row["same_category"]]
    return {
        "track": "A_image_to_3d_instance_ranking",
        "top_mean_fraction": float(top_fraction),
        "query_count": len(rows),
        "recall_at_1": _mean([row["recall_at_1"] for row in rows]),
        "recall_at_5": _mean([row["recall_at_5"] for row in rows]),
        "mrr": _mean([row["reciprocal_rank"] for row in rows]),
        "same_category_recall_at_1": _mean(same),
        "same_category_query_count": len(same),
        "category_macro_recall_at_1": _macro(
            rows, "recall_at_1", "nyu40_class_id"
        ),
        "scene_macro_recall_at_1": _macro(rows, "recall_at_1", "scene_id"),
        "per_query": rows,
    }

