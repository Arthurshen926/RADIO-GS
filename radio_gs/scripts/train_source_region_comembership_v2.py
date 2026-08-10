#!/usr/bin/env python3
"""Run the frozen source-only RegionCoMembershipV2 four-plus-two protocol."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.models.region_comembership_v2 import (
    PAIR_FEATURE_NAMES,
    RegionCoMembershipV2,
)
from radio_gs.querying.bounded_region_comembership_readout import (
    bounded_regions_for_seed,
    bridge_free_component_ids,
    thresholded_adjacency,
)
from radio_gs.scripts.materialize_source_region_comembership_v2 import (
    SCHEMA as SOURCE_AUTHORITY_SCHEMA,
    densify_primitive_instance_mass,
    source_access as builder_source_access,
    validate_source_region_comembership_v2,
)
from radio_gs.scripts.train_source_region_comembership_v1 import (
    LEARNING_RATE,
    SEED,
    THRESHOLDS,
    TRAIN_SCENES,
    VALIDATION_SCENES,
    WEIGHT_DECAY,
    balanced_scene_loss,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


EXECUTION_AUTHORITY_SCHEMA = (
    "radio_gs.source_region_comembership_v2_execution_authority.v1"
)
CHECKPOINT_SCHEMA = "radio_gs.region_comembership_v2_checkpoint.v1"
PREREGISTRATION = Path(
    "paper/artifacts/source_only_region_comembership_v2_preregistration_20260807.json"
)
EFFICIENCY_ADDENDUM = Path(
    "paper/artifacts/source_only_region_comembership_v2_formal_selection_efficiency_addendum_20260807.json"
)
METHODS = ("maximum_product", "dual_path_widest", "multipoint_consistency")
MAXIMUM_REGIONS = (1, 2, 4, 8)
EPOCHS = 100
SNAPSHOT_EPOCHS = (0, 25, 50, 75, 100)
PROXY_SEEDS_PER_SCENE = 256
GLOBAL_PROXY_SUPPLEMENT = 8


@dataclass(frozen=True)
class SceneAuthorityV2:
    scene_id: str
    split: str
    record: dict[str, str]
    region_count: int
    pair_indices: torch.Tensor
    pair_features: torch.Tensor
    targets: torch.Tensor
    evidence_weights: torch.Tensor
    region_rows: torch.Tensor
    token_mask: torch.Tensor
    primitive_instance_mass: torch.Tensor
    core_rows: tuple[torch.Tensor, ...]
    region_instance_mass: torch.Tensor
    dominant_instance_ids: torch.Tensor
    dominant_instance_mass: torch.Tensor
    eligible_seeds: torch.Tensor
    target_total: torch.Tensor
    scene_total: float


def source_access() -> dict[str, bool]:
    return {
        **builder_source_access(),
        "source_validation_used_for_selection": True,
        "target_pair_features_opened": False,
    }


def _metric_contract() -> dict[str, Any]:
    return {
        "checkpoint_epochs": list(SNAPSHOT_EPOCHS),
        "proxy_seeds_per_scene": PROXY_SEEDS_PER_SCENE,
        "methods": list(METHODS),
        "maximum_regions": list(MAXIMUM_REGIONS),
        "thresholds": list(THRESHOLDS),
        "global_proxy_supplement": GLOBAL_PROXY_SUPPLEMENT,
        "final_selection": "deduplicated_exact_primitive_instance_topology",
        "target_execution_before_promotion": False,
    }


def training_contract() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    return {
        "schema_version": 1,
        "model": {
            "class": "RegionCoMembershipV2",
            "pair_features": list(PAIR_FEATURE_NAMES),
            "architecture": "21_64_32_1_GELU",
            "epoch_zero_probability": 0.5,
            "query_input": False,
            "scene_embedding": False,
        },
        "cohort": {
            "source_train": list(TRAIN_SCENES),
            "source_validation": list(VALIDATION_SCENES),
        },
        "normalization": "source_train_positive_evidence_median_and_1.4826_MAD",
        "objective": "equal_scene_equal_class_evidence_weighted_BCE",
        "optimizer": {
            "name": "AdamW",
            "seed": SEED,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "early_stopping": False,
        },
        "selection": _metric_contract(),
        "preregistration": file_record(root / PREREGISTRATION),
        "efficiency_addendum": file_record(root / EFFICIENCY_ADDENDUM),
        "source_access": source_access(),
    }


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must be an exact file record")
    digest = str(value["sha256"])
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"{label} SHA-256 differs")
    return {"path": str(value["path"]), "sha256": digest}


def validate_execution_authority(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("V2 execution authority must be a mapping")
    authority = dict(value)
    required = {
        "schema",
        "schema_version",
        "status",
        "implementation",
        "preregistration",
        "efficiency_addendum",
        "source_train",
        "source_validation",
        "training_authorized",
        "target_execution_authorized",
        "source_access",
    }
    if (
        set(authority) != required
        or authority.get("schema") != EXECUTION_AUTHORITY_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("status")
        != "authorized_after_complete_v2_4train_2validation_preflight"
        or authority.get("training_authorized") is not True
        or authority.get("target_execution_authorized") is not False
        or authority.get("source_access") != source_access()
    ):
        raise ValueError("V2 execution authority header differs")
    for name in ("implementation", "preregistration", "efficiency_addendum"):
        authority[name] = _record(authority[name], label=name)
    for split, expected in (
        ("source_train", TRAIN_SCENES),
        ("source_validation", VALIDATION_SCENES),
    ):
        rows = authority[split]
        if not isinstance(rows, list) or len(rows) != len(expected):
            raise ValueError(f"V2 {split} count differs")
        normalized = []
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {"scene_id", "authority"}:
                raise ValueError(f"V2 {split} record differs")
            normalized.append(
                {
                    "scene_id": str(row["scene_id"]),
                    "authority": _record(row["authority"], label=f"{split} scene"),
                }
            )
        if tuple(row["scene_id"] for row in normalized) != tuple(expected):
            raise ValueError(f"V2 {split} cohort differs")
        authority[split] = normalized
    return authority


def _scene_core_evidence(
    region_rows: torch.Tensor,
    token_mask: torch.Tensor,
    primitive_mass: torch.Tensor,
) -> tuple[
    tuple[torch.Tensor, ...],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    rows = torch.as_tensor(region_rows).long().cpu()
    mask = torch.as_tensor(token_mask).bool().cpu()
    mass = torch.as_tensor(primitive_mass).float().cpu()
    core_rows = tuple(
        torch.unique(rows[index, mask[index]], sorted=True)
        for index in range(rows.shape[0])
    )
    region_mass = torch.stack(
        [mass[selected].double().sum(dim=0) for selected in core_rows]
    ).float()
    dominant_mass, dominant_zero = region_mass[:, 1:].max(dim=1)
    eligible = dominant_mass > 0
    dominant = torch.where(
        eligible, dominant_zero + 1, -torch.ones_like(dominant_zero)
    )
    return core_rows, region_mass, dominant.long(), dominant_mass, eligible


def load_scene_authority_v2(
    record: Mapping[str, str], *, expected_scene_id: str, expected_split: str
) -> SceneAuthorityV2:
    source = validate_file_record(record, label="source V2 scene authority")
    payload, digest, verified = load_torch_mapping(
        source,
        expected_sha256=record["sha256"],
        map_location="cpu",
        label="source V2 scene authority",
    )
    value = validate_source_region_comembership_v2(payload)
    if (
        value["schema"] != SOURCE_AUTHORITY_SCHEMA
        or value["scene_id"] != expected_scene_id
        or value["split"] != expected_split
    ):
        raise ValueError("source V2 scene identity differs")
    dense = densify_primitive_instance_mass(
        flat_keys=value["primitive_instance_flat_keys"],
        mass=value["primitive_instance_mass"],
        primitive_count=value["primitive_count"],
        instance_columns_including_zero=value[
            "instance_columns_including_zero"
        ],
    )
    core, region_mass, dominant, dominant_mass, eligible = _scene_core_evidence(
        value["region_rows"], value["token_mask"], dense
    )
    return SceneAuthorityV2(
        scene_id=expected_scene_id,
        split=expected_split,
        record={"path": str(verified), "sha256": digest},
        region_count=int(value["canonical_region_indices"].numel()),
        pair_indices=value["pair_indices"].long().cpu().contiguous(),
        pair_features=value["pair_features"].float().cpu().contiguous(),
        targets=value["same_instance_targets"].bool().cpu().contiguous(),
        evidence_weights=value["pair_evidence_weights"].float().cpu().contiguous(),
        region_rows=value["region_rows"].long().cpu().contiguous(),
        token_mask=value["token_mask"].bool().cpu().contiguous(),
        primitive_instance_mass=dense,
        core_rows=core,
        region_instance_mass=region_mass,
        dominant_instance_ids=dominant,
        dominant_instance_mass=dominant_mass.float(),
        eligible_seeds=eligible.bool(),
        target_total=dense[:, 1:].double().sum(dim=0),
        scene_total=float(dense.double().sum()),
    )


def fixed_proxy_seed_indices(scene: SceneAuthorityV2) -> torch.Tensor:
    eligible = torch.nonzero(scene.eligible_seeds, as_tuple=False).flatten().tolist()
    by_instance: dict[int, int] = {}
    for seed in eligible:
        by_instance.setdefault(int(scene.dominant_instance_ids[seed]), seed)
    selected = sorted(by_instance.values())
    if len(selected) > PROXY_SEEDS_PER_SCENE:
        raise ValueError("V2 proxy seed limit cannot cover every instance")
    selected_set = set(selected)
    remaining = [seed for seed in eligible if seed not in selected_set]
    slots = min(PROXY_SEEDS_PER_SCENE - len(selected), len(remaining))
    if slots > 0:
        positions = torch.linspace(0, len(remaining) - 1, steps=slots).round().long()
        selected.extend(remaining[int(position)] for position in positions)
    return torch.tensor(sorted(set(selected)), dtype=torch.int64)


def _selection_map(
    *,
    scene: SceneAuthorityV2,
    probability: torch.Tensor,
    method: str,
    threshold: float,
    seeds: Sequence[int],
) -> dict[int, tuple[int, ...]]:
    adjacency = thresholded_adjacency(
        region_count=scene.region_count,
        pair_indices=scene.pair_indices,
        pair_probabilities=probability,
        threshold=threshold,
    )
    components = (
        bridge_free_component_ids(adjacency)
        if method == "dual_path_widest"
        else None
    )
    return {
        int(seed): bounded_regions_for_seed(
            method=method,
            seed_region_index=int(seed),
            adjacency=adjacency,
            maximum_regions=max(MAXIMUM_REGIONS),
            bridge_free_components=components,
        )
        for seed in seeds
    }


def _mass_metrics(
    *,
    scene: SceneAuthorityV2,
    selections: Mapping[int, Sequence[int]],
    maximum_regions: int,
    exact_primitive_union: bool,
) -> dict[str, float]:
    target_total = (
        scene.target_total
        if exact_primitive_union
        else scene.region_instance_mass[:, 1:].double().sum(dim=0)
    )
    scene_total = (
        scene.scene_total
        if exact_primitive_union
        else float(scene.region_instance_mass.double().sum())
    )
    sums: dict[int, dict[str, float]] = {}
    for seed in sorted(selections):
        instance = int(scene.dominant_instance_ids[seed])
        if instance <= 0 or not bool(scene.eligible_seeds[seed]):
            raise ValueError("V2 metric seed is not eligible")
        selected = tuple(int(v) for v in selections[seed][: int(maximum_regions)])
        if not selected or selected[0] != seed or len(set(selected)) != len(selected):
            raise ValueError("V2 metric selection differs")
        if exact_primitive_union:
            primitives = torch.unique(
                torch.cat([scene.core_rows[index] for index in selected]), sorted=True
            )
            selected_mass = scene.primitive_instance_mass[primitives].double().sum(
                dim=0
            )
            selected_units = float(primitives.numel())
        else:
            selected_mass = scene.region_instance_mass[
                torch.tensor(selected, dtype=torch.long)
            ].double().sum(dim=0)
            selected_units = float(len(selected))
        total = float(selected_mass.sum())
        correct = float(selected_mass[instance])
        target = float(target_total[instance - 1])
        values = {
            "iou": correct / max(target + total - correct, 1e-12),
            "f1": 2 * correct / max(target + total, 1e-12),
            "contamination": (total - correct) / max(total, 1e-12),
            "giant_excess": max(
                0.0, total / scene_total - target / scene_total
            ),
            "selected_units": selected_units,
            "selected_regions": float(len(selected)),
        }
        row = sums.setdefault(
            instance,
            {"seed_weight": 0.0, **{name: 0.0 for name in values}},
        )
        weight = float(scene.dominant_instance_mass[seed])
        row["seed_weight"] += weight
        for name, value in values.items():
            row[name] += weight * value
    if not sums:
        raise ValueError("V2 metric has no seeds")
    names = (
        "iou",
        "f1",
        "contamination",
        "giant_excess",
        "selected_units",
        "selected_regions",
    )
    per_instance = {
        instance: {
            name: values[name] / max(values["seed_weight"], 1e-12)
            for name in names
        }
        for instance, values in sums.items()
    }
    macro = {
        name: sum(row[name] for row in per_instance.values()) / len(per_instance)
        for name in names
    }
    macro["topology_score"] = (
        macro["iou"] - macro["contamination"] - macro["giant_excess"]
    )
    return macro


def _scene_macro(per_scene: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    names = next(iter(per_scene.values())).keys()
    return {
        name: sum(float(row[name]) for row in per_scene.values()) / len(per_scene)
        for name in names
    }


def _rule_key(row: Mapping[str, Any], *, include_epoch: bool) -> tuple[float, ...]:
    metric = row["scene_macro"]
    result = (
        float(metric["topology_score"]),
        float(metric["iou"]),
        float(metric["f1"]),
        -float(metric["contamination"]),
        -float(metric["giant_excess"]),
        -float(row["maximum_regions"]),
        -float(row["threshold"]),
    )
    if include_epoch:
        result += (-float(row["epoch"]),)
    result += (-float(METHODS.index(str(row["method"]))),)
    return result


def _proxy_grid(
    scenes: Sequence[SceneAuthorityV2],
    probabilities: Mapping[str, torch.Tensor],
) -> list[dict[str, Any]]:
    seed_rows = {
        scene.scene_id: fixed_proxy_seed_indices(scene).tolist() for scene in scenes
    }
    singleton_per_scene = {
        scene.scene_id: _mass_metrics(
            scene=scene,
            selections={seed: (seed,) for seed in seed_rows[scene.scene_id]},
            maximum_regions=1,
            exact_primitive_union=False,
        )
        for scene in scenes
    }
    grid = [
        {
            "method": METHODS[0],
            "maximum_regions": 1,
            "threshold": max(THRESHOLDS),
            "scene_macro": _scene_macro(singleton_per_scene),
            "per_scene": singleton_per_scene,
        }
    ]
    for threshold in THRESHOLDS:
        for method in METHODS:
            maps = {
                scene.scene_id: _selection_map(
                    scene=scene,
                    probability=probabilities[scene.scene_id],
                    method=method,
                    threshold=threshold,
                    seeds=seed_rows[scene.scene_id],
                )
                for scene in scenes
            }
            for maximum in MAXIMUM_REGIONS[1:]:
                per_scene = {
                    scene.scene_id: _mass_metrics(
                        scene=scene,
                        selections=maps[scene.scene_id],
                        maximum_regions=maximum,
                        exact_primitive_union=False,
                    )
                    for scene in scenes
                }
                grid.append(
                    {
                        "method": method,
                        "maximum_regions": maximum,
                        "threshold": float(threshold),
                        "scene_macro": _scene_macro(per_scene),
                        "per_scene": per_scene,
                    }
                )
    return grid


def deterministic_proxy_shortlist(grid: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    singleton = [row for row in grid if int(row["maximum_regions"]) == 1]
    if len(singleton) != 1:
        raise ValueError("V2 proxy grid singleton differs")
    selected: list[Mapping[str, Any]] = [singleton[0]]
    nonsingleton = [row for row in grid if int(row["maximum_regions"]) > 1]
    for method in METHODS:
        for maximum in MAXIMUM_REGIONS[1:]:
            candidates = [
                row
                for row in nonsingleton
                if row["method"] == method
                and int(row["maximum_regions"]) == maximum
            ]
            selected.append(max(candidates, key=lambda row: _rule_key(row, include_epoch=False)))
    selected.extend(
        sorted(
            nonsingleton,
            key=lambda row: _rule_key(row, include_epoch=False),
            reverse=True,
        )[:GLOBAL_PROXY_SUPPLEMENT]
    )
    unique: dict[tuple[str, int, float], dict[str, Any]] = {}
    for row in selected:
        key = (
            str(row["method"]),
            int(row["maximum_regions"]),
            float(row["threshold"]),
        )
        unique.setdefault(key, dict(row))
    return list(unique.values())


def _exact_candidate(
    *,
    epoch: int,
    rule: Mapping[str, Any],
    scenes: Sequence[SceneAuthorityV2],
    probabilities: Mapping[str, torch.Tensor],
    selection_cache: dict[
        tuple[str, str, float], dict[int, tuple[int, ...]]
    ],
) -> dict[str, Any]:
    maximum = int(rule["maximum_regions"])
    per_scene = {}
    for scene in scenes:
        seeds = torch.nonzero(scene.eligible_seeds, as_tuple=False).flatten().tolist()
        if maximum == 1:
            selections = {seed: (seed,) for seed in seeds}
        else:
            cache_key = (
                scene.scene_id,
                str(rule["method"]),
                float(rule["threshold"]),
            )
            if cache_key not in selection_cache:
                selection_cache[cache_key] = _selection_map(
                    scene=scene,
                    probability=probabilities[scene.scene_id],
                    method=str(rule["method"]),
                    threshold=float(rule["threshold"]),
                    seeds=seeds,
                )
            selections = selection_cache[cache_key]
        per_scene[scene.scene_id] = _mass_metrics(
            scene=scene,
            selections=selections,
            maximum_regions=maximum,
            exact_primitive_union=True,
        )
    return {
        "epoch": int(epoch),
        "method": str(rule["method"]),
        "maximum_regions": maximum,
        "threshold": float(rule["threshold"]),
        "scene_macro": _scene_macro(per_scene),
        "per_scene": per_scene,
    }


def fit_normalization(scenes: Sequence[SceneAuthorityV2]) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.cat(
        [scene.pair_features[scene.evidence_weights > 0] for scene in scenes]
    )
    median = values.median(dim=0).values
    mad = (values - median).abs().median(dim=0).values
    scale = torch.where(mad > 0, mad * 1.4826, torch.ones_like(mad))
    return median.contiguous(), scale.contiguous()


def _state_copy(model: RegionCoMembershipV2) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _state_sha(state: Mapping[str, torch.Tensor]) -> str:
    return canonical_json_sha256(
        {name: tensor_sha256(value) for name, value in sorted(state.items())}
    )


def _parse_authority_records(values: Sequence[str]) -> dict[str, dict[str, str]]:
    result = {}
    for value in values:
        pieces = str(value).split("::")
        if len(pieces) != 3 or pieces[0] in result:
            raise ValueError("--scene-authority must be scene_id::path::sha256")
        result[pieces[0]] = {"path": pieces[1], "sha256": pieces[2]}
    expected = set(TRAIN_SCENES) | set(VALIDATION_SCENES)
    if set(result) != expected:
        raise ValueError("V2 preflight scene cohort differs")
    return result


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"V2 execution authority exists: {output}")
    records = _parse_authority_records(args.scene_authority)
    for scene in (*TRAIN_SCENES, *VALIDATION_SCENES):
        load_scene_authority_v2(
            records[scene],
            expected_scene_id=scene,
            expected_split=(
                "source_train" if scene in TRAIN_SCENES else "source_validation"
            ),
        )
    root = Path(__file__).resolve().parents[2]
    authority = {
        "schema": EXECUTION_AUTHORITY_SCHEMA,
        "schema_version": 1,
        "status": "authorized_after_complete_v2_4train_2validation_preflight",
        "implementation": file_record(Path(__file__).resolve()),
        "preregistration": file_record(root / PREREGISTRATION),
        "efficiency_addendum": file_record(root / EFFICIENCY_ADDENDUM),
        "source_train": [
            {"scene_id": scene, "authority": records[scene]}
            for scene in TRAIN_SCENES
        ],
        "source_validation": [
            {"scene_id": scene, "authority": records[scene]}
            for scene in VALIDATION_SCENES
        ],
        "training_authorized": True,
        "target_execution_authorized": False,
        "source_access": source_access(),
    }
    validate_execution_authority(authority)
    write_frozen_json(output, authority)
    return {"status": "V2_execution_authority_complete", "output": file_record(output)}


def _load_execution(args: argparse.Namespace) -> tuple[
    dict[str, Any], tuple[SceneAuthorityV2, ...], tuple[SceneAuthorityV2, ...]
]:
    raw, digest, source = load_json_object(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
        label="V2 execution authority",
    )
    authority = validate_execution_authority(raw)
    implementation = validate_file_record(
        authority["implementation"], label="V2 trainer implementation"
    )
    if implementation != Path(__file__).resolve():
        raise ValueError("V2 execution authority binds another trainer")
    root = Path(__file__).resolve().parents[2]
    for name, expected in (
        ("preregistration", root / PREREGISTRATION),
        ("efficiency_addendum", root / EFFICIENCY_ADDENDUM),
    ):
        verified_protocol = validate_file_record(
            authority[name], label=f"V2 {name}"
        )
        if verified_protocol != expected.resolve():
            raise ValueError(f"V2 execution authority binds another {name}")

    def load(split: str, expected: Sequence[str]) -> tuple[SceneAuthorityV2, ...]:
        return tuple(
            load_scene_authority_v2(
                row["authority"], expected_scene_id=scene, expected_split=split
            )
            for row, scene in zip(authority[split], expected)
        )

    authority["verified_path"] = str(source)
    authority["verified_sha256"] = digest
    return (
        authority,
        load("source_train", TRAIN_SCENES),
        load("source_validation", VALIDATION_SCENES),
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    report_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or report_path.exists():
        raise FileExistsError("V2 checkpoint/report output must be new")
    execution, train_scenes, validation_scenes = _load_execution(args)
    median, scale = fit_normalization(train_scenes)
    device = torch.device(str(args.device))
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    model = RegionCoMembershipV2(median, scale).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    device_rows = {
        scene.scene_id: (
            scene.pair_features.to(device),
            scene.targets.to(device),
            scene.evidence_weights.to(device),
        )
        for scene in train_scenes
    }
    snapshots = {0: _state_copy(model)}
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for scene in train_scenes:
            features, targets, weights = device_rows[scene.scene_id]
            loss = balanced_scene_loss(model(features), targets, weights)
            (loss / len(train_scenes)).backward()
            losses.append(float(loss.detach().cpu()))
        optimizer.step()
        row = {
            "epoch": epoch,
            "train_scene_macro_balanced_bce": sum(losses) / len(losses),
        }
        history.append(row)
        if epoch in SNAPSHOT_EPOCHS:
            snapshots[epoch] = _state_copy(model)
            print(json.dumps(row, sort_keys=True), flush=True)
    if tuple(sorted(snapshots)) != SNAPSHOT_EPOCHS:
        raise RuntimeError("V2 retained checkpoint schedule differs")
    model.cpu()
    del device_rows
    del optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    exact_candidates = []
    proxy_audit = {}
    for epoch in SNAPSHOT_EPOCHS:
        model.load_state_dict(snapshots[epoch], strict=True)
        model.eval()
        with torch.no_grad():
            probabilities = {
                scene.scene_id: model.probability(scene.pair_features).cpu()
                for scene in validation_scenes
            }
        proxy = _proxy_grid(validation_scenes, probabilities)
        shortlist = deterministic_proxy_shortlist(proxy)
        proxy_audit[str(epoch)] = {
            "grid_rows": len(proxy),
            "shortlist": [
                {
                    "method": row["method"],
                    "maximum_regions": row["maximum_regions"],
                    "threshold": row["threshold"],
                    "scene_macro": row["scene_macro"],
                }
                for row in shortlist
            ],
        }
        exact_selection_cache: dict[
            tuple[str, str, float], dict[int, tuple[int, ...]]
        ] = {}
        for rule in shortlist:
            candidate = _exact_candidate(
                epoch=epoch,
                rule=rule,
                scenes=validation_scenes,
                probabilities=probabilities,
                selection_cache=exact_selection_cache,
            )
            exact_candidates.append(candidate)
            print(
                json.dumps(
                    {"phase": "exact_validation_candidate", **candidate},
                    sort_keys=True,
                ),
                flush=True,
            )
    selected = max(
        exact_candidates, key=lambda row: _rule_key(row, include_epoch=True)
    )
    singleton = next(
        row
        for row in exact_candidates
        if row["epoch"] == 0 and row["maximum_regions"] == 1
    )
    selected_metric = selected["scene_macro"]
    baseline_metric = singleton["scene_macro"]
    promotion = {
        "selected_epoch_positive": int(selected["epoch"]) > 0,
        "topology_strictly_exceeds_singleton": selected_metric["topology_score"]
        > baseline_metric["topology_score"],
        "iou_strictly_exceeds_singleton": selected_metric["iou"]
        > baseline_metric["iou"],
        "f1_strictly_exceeds_singleton": selected_metric["f1"]
        > baseline_metric["f1"],
        "singleton": baseline_metric,
        "selected": selected_metric,
    }
    promotion["passed"] = all(
        promotion[name]
        for name in (
            "selected_epoch_positive",
            "topology_strictly_exceeds_singleton",
            "iou_strictly_exceeds_singleton",
            "f1_strictly_exceeds_singleton",
        )
    )
    selected_state = snapshots[int(selected["epoch"])]
    contract = training_contract()
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": 1,
        "training_contract": contract,
        "training_contract_sha256": canonical_json_sha256(contract),
        "execution_authority": {
            "path": execution["verified_path"],
            "sha256": execution["verified_sha256"],
        },
        "feature_names": list(PAIR_FEATURE_NAMES),
        "normalization": {"median": median, "robust_scale": scale},
        "model_state_dict": selected_state,
        "model_state_dict_sha256": _state_sha(selected_state),
        "selected_epoch": int(selected["epoch"]),
        "selected_rule": {
            name: selected[name]
            for name in ("method", "maximum_regions", "threshold")
        },
        "selected_validation": selected,
        "singleton_validation": singleton,
        "promotion_gate": promotion,
        "source_access": source_access(),
        "target_execution_performed": False,
    }
    checkpoint_path = write_torch_noclobber(output, checkpoint)
    report = {
        "schema": "radio_gs.region_comembership_v2_pilot_result.v1",
        "schema_version": 1,
        "status": "source_only_v2_4train_2validation_complete",
        "checkpoint": file_record(checkpoint_path),
        "selected_validation": selected,
        "singleton_validation": singleton,
        "promotion_gate": promotion,
        "exact_candidate_count": len(exact_candidates),
        "exact_candidates": exact_candidates,
        "proxy_audit": proxy_audit,
        "history": history,
        "source_access": source_access(),
        "target_execution_performed": False,
    }
    write_frozen_json(report_path, report)
    return report


def synthetic_dry_run() -> dict[str, Any]:
    torch.manual_seed(SEED)
    model = RegionCoMembershipV2(torch.zeros(21), torch.ones(21))
    features = torch.randn(8, 21)
    target = torch.tensor([True, False] * 4)
    weight = torch.ones(8)
    logits = model(features)
    loss = balanced_scene_loss(logits, target, weight)
    loss.backward()
    return {
        "epoch_zero_probability": float(torch.sigmoid(logits).mean()),
        "final_layer_gradient_nonzero": bool(
            model.network[-1].weight.grad.abs().sum() > 0
        ),
        "snapshot_epochs": list(SNAPSHOT_EPOCHS),
        "benchmark_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("synthetic-dry-run")
    prepare = commands.add_parser("preflight")
    prepare.add_argument("--scene-authority", action="append", required=True)
    prepare.add_argument("--output", required=True)
    run = commands.add_parser("train")
    run.add_argument("--execution-authority", required=True)
    run.add_argument("--expected-execution-authority-sha256", required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "synthetic-dry-run":
        result = synthetic_dry_run()
    elif args.command == "preflight":
        result = preflight(args)
    else:
        result = train(args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
