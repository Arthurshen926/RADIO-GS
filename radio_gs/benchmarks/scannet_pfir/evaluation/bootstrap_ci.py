"""Scene-clustered bootstrap confidence intervals for PFIR metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Mapping, Sequence

import numpy as np


def scene_clustered_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    metric: Callable[[Sequence[Mapping[str, Any]]], float],
    *,
    samples: int = 2000,
    seed: int = 20260718,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["scene_id"])].append(row)
    scenes = sorted(groups)
    if not scenes:
        raise ValueError("cannot bootstrap an empty result")
    rng = np.random.default_rng(int(seed))
    estimates = []
    for _ in range(int(samples)):
        sampled = rng.choice(scenes, size=len(scenes), replace=True)
        replicate = [row for scene in sampled for row in groups[str(scene)]]
        estimates.append(float(metric(replicate)))
    alpha = (1.0 - float(confidence)) * 0.5
    return {
        "estimate": float(metric(rows)),
        "lower": float(np.quantile(estimates, alpha)),
        "upper": float(np.quantile(estimates, 1.0 - alpha)),
        "confidence": float(confidence),
        "bootstrap_samples": int(samples),
        "bootstrap_unit": "scene",
        "seed": int(seed),
    }
