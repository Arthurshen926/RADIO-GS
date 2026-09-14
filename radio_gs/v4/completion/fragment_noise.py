"""Deterministic fragment-set observation noise for completion training.

The clean ScanNet/PFIR object identity remains target-only supervision.  This
module touches only already-observed positive support and converts it into a
small set of spatially coherent partial fragments.  Dropped positives become
unknown for that token; factual negatives are never flipped to positives.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import torch

from .oracle import PartialObjectMembership, build_token_context


SCHEMA = "radio_gs.surface_object_memory_v4.fragment_set_observation_noise.v1"


def _digest(scene_id: str, token_id: int, seed: int) -> bytes:
    return hashlib.sha256(
        "\0".join((str(scene_id), str(int(token_id)), str(int(seed)))).encode()
    ).digest()


def fragment_set_observation_noise(
    partial: PartialObjectMembership,
    centres: torch.Tensor,
    *,
    scene_id: str,
    seed: int,
    minimum_keep_fraction: float = 0.35,
    maximum_keep_fraction: float = 0.75,
    maximum_fragments: int = 3,
) -> tuple[PartialObjectMembership, dict[str, Any]]:
    """Return coherent fragment subsets of immutable observed positives."""

    xyz = torch.as_tensor(centres, dtype=torch.float32).cpu()
    if xyz.shape != (partial.positive.shape[0], 3) or not bool(torch.isfinite(xyz).all()):
        raise ValueError("fragment-noise centres must be finite and align with elements")
    if not 0 < minimum_keep_fraction <= maximum_keep_fraction <= 1:
        raise ValueError("fragment-noise keep fractions must lie in (0,1]")
    if maximum_fragments <= 0:
        raise ValueError("maximum fragment count must be positive")

    retained = torch.zeros_like(partial.positive)
    token_receipts = []
    for token_id in range(partial.positive.shape[1]):
        candidates = torch.where(partial.positive[:, token_id])[0]
        if not candidates.numel():
            raise ValueError("fragment noise requires one observed seed per token")
        digest = _digest(scene_id, token_id, seed)
        unit = int.from_bytes(digest[:8], "little") / float(2**64 - 1)
        keep_fraction = minimum_keep_fraction + unit * (
            maximum_keep_fraction - minimum_keep_fraction
        )
        keep_count = max(1, min(int(candidates.numel()), math.ceil(candidates.numel() * keep_fraction)))
        fragment_count = 1 + digest[8] % min(maximum_fragments, keep_count)

        # Hash-selected observed anchors create spatially coherent parts.  A
        # point is ranked by distance to its nearest anchor, so the union of
        # the retained prefixes simulates one-to-three partial mask fragments.
        anchor_order = sorted(
            candidates.tolist(),
            key=lambda element_id: hashlib.sha256(
                digest + int(element_id).to_bytes(8, "little", signed=False)
            ).digest(),
        )
        anchors = torch.tensor(anchor_order[:fragment_count], dtype=torch.long)
        distance = torch.cdist(xyz[candidates], xyz[anchors]).min(-1).values
        # Element id is an explicit deterministic tie-breaker.
        ranked = sorted(
            range(candidates.numel()),
            key=lambda index: (float(distance[index]), int(candidates[index])),
        )
        chosen = candidates[torch.tensor(ranked[:keep_count], dtype=torch.long)]
        retained[chosen, token_id] = True
        token_receipts.append({
            "token_id": token_id,
            "observed_positive_count_before": int(candidates.numel()),
            "observed_positive_count_after": int(chosen.numel()),
            "realized_keep_fraction": float(chosen.numel() / candidates.numel()),
            "fragment_anchor_count": int(fragment_count),
            "anchor_element_ids": anchors.tolist(),
        })

    if bool((retained & ~partial.positive).any()):
        raise RuntimeError("fragment noise invented positive membership")
    dropped = partial.positive & ~retained
    noisy = PartialObjectMembership(
        positive=retained,
        negative=partial.negative,
        unknown=partial.unknown | dropped,
        eligible_elements=partial.eligible_elements,
    )
    receipt = {
        "schema": SCHEMA,
        "scene_id": str(scene_id),
        "seed": int(seed),
        "minimum_keep_fraction": float(minimum_keep_fraction),
        "maximum_keep_fraction": float(maximum_keep_fraction),
        "maximum_fragments": int(maximum_fragments),
        "input_positive_count": int(partial.positive.sum()),
        "retained_positive_count": int(retained.sum()),
        "dropped_positive_count": int(dropped.sum()),
        "negative_changed": False,
        "positive_invented": False,
        "complete_target_membership_read": False,
        "tokens": token_receipts,
    }
    return noisy, receipt


def apply_fragment_noise_to_runtime(
    runtime: dict[str, Any],
    *,
    seed: int,
    minimum_keep_fraction: float = 0.35,
    maximum_keep_fraction: float = 0.75,
    maximum_fragments: int = 3,
) -> dict[str, Any]:
    """Replace a clean runtime's observation side and rebuild its context."""

    result = dict(runtime)
    noisy, receipt = fragment_set_observation_noise(
        runtime["partial"],
        runtime["centres"],
        scene_id=str(runtime["payload"]["scene_id"]),
        seed=seed,
        minimum_keep_fraction=minimum_keep_fraction,
        maximum_keep_fraction=maximum_keep_fraction,
        maximum_fragments=maximum_fragments,
    )
    result["partial"] = noisy
    result["context"] = build_token_context(
        result["centres"],
        result["local_features"],
        noisy,
        result["carrier"].neighbors().edge_index,
        minimum_scale=result["minimum_scale"],
    )
    source_visible = result.get("source_visible")
    if source_visible is not None:
        source_visible = torch.as_tensor(source_visible, dtype=torch.bool).cpu()
        unknown = noisy.unknown.any(-1) & noisy.eligible_elements
        result["unknown_strata"] = {
            "visible_but_unmasked": unknown & source_visible,
            "never_visible": unknown & ~source_visible,
        }
    result["fragment_noise_receipt"] = receipt
    return result


__all__ = [
    "SCHEMA",
    "apply_fragment_noise_to_runtime",
    "fragment_set_observation_noise",
]
