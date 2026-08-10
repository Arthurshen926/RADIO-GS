"""Source-only metrics for deduplicated canonical-region primitive unions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def _validated_inputs(
    *,
    region_rows: torch.Tensor,
    token_mask: torch.Tensor,
    primitive_instance_mass: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = torch.as_tensor(region_rows).detach().long().cpu()
    mask = torch.as_tensor(token_mask).detach().bool().cpu()
    mass = torch.as_tensor(primitive_instance_mass).detach().float().cpu()
    if (
        rows.ndim != 2
        or mask.shape != rows.shape
        or mass.ndim != 2
        or mass.shape[0] <= 0
        or mass.shape[1] < 2
        or rows.shape[0] <= 0
        or not bool(mask.any(dim=1).all())
        or not bool(torch.isfinite(mass).all())
        or bool((mass < 0).any())
        or bool((rows[mask] < 0).any())
        or bool((rows[mask] >= mass.shape[0]).any())
    ):
        raise ValueError("primitive-instance union authority differs")
    return rows.contiguous(), mask.contiguous(), mass.contiguous()


def region_seed_instance_evidence(
    *,
    region_rows: torch.Tensor,
    token_mask: torch.Tensor,
    primitive_instance_mass: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Derive the fixed positive target and its exact mass for every region."""

    rows, mask, mass = _validated_inputs(
        region_rows=region_rows,
        token_mask=token_mask,
        primitive_instance_mass=primitive_instance_mass,
    )
    region_mass = torch.zeros(rows.shape[0], mass.shape[1], dtype=torch.float64)
    for region in range(rows.shape[0]):
        primitive_rows = torch.unique(rows[region, mask[region]], sorted=True)
        region_mass[region] = mass[primitive_rows].double().sum(dim=0)
    target_mass, target_zero = region_mass[:, 1:].max(dim=1)
    eligible = target_mass > 0
    target = torch.where(
        eligible,
        target_zero + 1,
        -torch.ones_like(target_zero),
    )
    return {
        "dominant_instance_ids": target.long().contiguous(),
        "dominant_instance_mass": target_mass.float().contiguous(),
        "eligible": eligible.bool().contiguous(),
        "region_instance_mass": region_mass.float().contiguous(),
    }


def primitive_instance_union_metrics(
    *,
    region_rows: torch.Tensor,
    token_mask: torch.Tensor,
    primitive_instance_mass: torch.Tensor,
    selections_by_seed: Mapping[int, Sequence[int]],
    maximum_regions: int | None = None,
) -> dict[str, Any]:
    """Score one bounded selection for every eligible source region seed.

    A primitive row shared by multiple selected canonical regions contributes
    exactly once.  Instance zero never defines a target but is included in
    selected mass, so unlabeled/background support is treated as contamination.
    """

    rows, mask, mass = _validated_inputs(
        region_rows=region_rows,
        token_mask=token_mask,
        primitive_instance_mass=primitive_instance_mass,
    )
    seed_evidence = region_seed_instance_evidence(
        region_rows=rows,
        token_mask=mask,
        primitive_instance_mass=mass,
    )
    dominant = seed_evidence["dominant_instance_ids"]
    dominant_mass = seed_evidence["dominant_instance_mass"]
    eligible = seed_evidence["eligible"]
    expected_seeds = set(
        torch.nonzero(eligible, as_tuple=False).flatten().tolist()
    )
    if set(selections_by_seed) != expected_seeds:
        raise ValueError("primitive-instance selections must cover eligible seeds")
    limit = None if maximum_regions is None else int(maximum_regions)
    if limit is not None and limit <= 0:
        raise ValueError("maximum_regions must be positive")

    target_total = mass[:, 1:].double().sum(dim=0)
    scene_total = float(mass.double().sum())
    if scene_total <= 0:
        raise ValueError("primitive-instance scene has no evidence")
    instances = sorted(set(int(dominant[seed]) for seed in expected_seeds))
    sums = {
        instance: {
            "seed_weight": 0.0,
            "iou": 0.0,
            "f1": 0.0,
            "contamination": 0.0,
            "giant_excess": 0.0,
            "selected_unique_primitives": 0.0,
            "selected_regions": 0.0,
        }
        for instance in instances
    }
    for seed in sorted(expected_seeds):
        selected = [int(value) for value in selections_by_seed[seed]]
        if (
            not selected
            or selected[0] != seed
            or len(set(selected)) != len(selected)
            or (limit is not None and len(selected) > limit)
            or min(selected) < 0
            or max(selected) >= rows.shape[0]
        ):
            raise ValueError("primitive-instance bounded seed selection differs")
        selected_tensor = torch.tensor(selected, dtype=torch.long)
        primitive_rows = torch.unique(
            rows[selected_tensor][mask[selected_tensor]], sorted=True
        )
        selected_by_instance = mass[primitive_rows].double().sum(dim=0)
        selected_total = float(selected_by_instance.sum())
        instance = int(dominant[seed])
        correct = float(selected_by_instance[instance])
        target = float(target_total[instance - 1])
        union = target + selected_total - correct
        values = {
            "iou": correct / max(union, 1e-12),
            "f1": 2.0 * correct / max(target + selected_total, 1e-12),
            "contamination": (selected_total - correct)
            / max(selected_total, 1e-12),
            "giant_excess": max(
                0.0, selected_total / scene_total - target / scene_total
            ),
            "selected_unique_primitives": float(primitive_rows.numel()),
            "selected_regions": float(len(selected)),
        }
        weight = float(dominant_mass[seed])
        sums[instance]["seed_weight"] += weight
        for name, value in values.items():
            sums[instance][name] += weight * value

    by_instance: dict[str, dict[str, float]] = {}
    metric_names = (
        "iou",
        "f1",
        "contamination",
        "giant_excess",
        "selected_unique_primitives",
        "selected_regions",
    )
    for instance in instances:
        values = sums[instance]
        denominator = max(values["seed_weight"], 1e-12)
        by_instance[str(instance)] = {
            name: values[name] / denominator for name in metric_names
        }
    macro = {
        name: sum(values[name] for values in by_instance.values())
        / len(by_instance)
        for name in metric_names
    }
    macro["topology_score"] = (
        macro["iou"] - macro["contamination"] - macro["giant_excess"]
    )
    return {
        "instance_macro": macro,
        "per_instance": by_instance,
        "eligible_seeds": len(expected_seeds),
        "eligible_instances": len(instances),
        "primitive_rows": int(mass.shape[0]),
        "primitive_instance_columns_including_zero": int(mass.shape[1]),
        "selection_mass_semantics": "deduplicated_primitive_rows_background_is_contamination",
    }


__all__ = [
    "primitive_instance_union_metrics",
    "region_seed_instance_evidence",
]
