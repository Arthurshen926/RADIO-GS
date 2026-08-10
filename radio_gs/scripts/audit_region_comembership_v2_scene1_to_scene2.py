#!/usr/bin/env python3
"""Fit V2 on source scene0001 and audit bounded transfer on scene0002."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from radio_gs.evaluation.primitive_instance_union import (
    primitive_instance_union_metrics,
    region_seed_instance_evidence,
)
from radio_gs.models.region_comembership_v2 import (
    CAPABILITY_PAIR_FEATURE_NAMES,
    PAIR_FEATURE_NAMES,
    RegionCoMembershipV2,
)
from radio_gs.querying.bounded_region_comembership_readout import (
    bounded_regions_for_seed,
    bridge_free_component_ids,
    thresholded_adjacency,
)
from radio_gs.scripts.audit_bounded_region_comembership_readout_source_v1 import (
    _bounded_seed_metrics,
)
from radio_gs.scripts.build_source_region_comembership_v1 import (
    _load_exact_instance_mass,
)
from radio_gs.scripts.materialize_region_capability_descriptors_v2 import (
    validate_region_capability_descriptor_authority,
)
from radio_gs.scripts.train_source_region_comembership_v1 import (
    LEARNING_RATE,
    SEED,
    THRESHOLDS,
    WEIGHT_DECAY,
    SceneAuthority,
    balanced_scene_loss,
    load_scene_authority,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_torch_mapping,
    sha256_file,
    write_frozen_json,
)


PREREGISTRATION = Path(
    "paper/artifacts/source_only_region_comembership_v2_scene1_to_scene2_audit_preregistration_20260807.json"
)
METHODS = ("maximum_product", "dual_path_widest", "multipoint_consistency")
MAXIMUM_REGIONS = (1, 2, 4, 8)
EPOCHS = 100


def capability_pair_features(
    *,
    pair_indices: torch.Tensor,
    appearance_direction: torch.Tensor,
    boundary_direction: torch.Tensor,
    appearance_concentration: torch.Tensor,
    boundary_concentration: torch.Tensor,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Materialize the six preregistered symmetric capability pair channels."""

    pairs = torch.as_tensor(pair_indices).detach().long().cpu()
    appearance = torch.as_tensor(appearance_direction).detach().float().cpu()
    boundary = torch.as_tensor(boundary_direction).detach().float().cpu()
    appearance_c = (
        torch.as_tensor(appearance_concentration).detach().float().cpu()
    )
    boundary_c = torch.as_tensor(boundary_concentration).detach().float().cpu()
    count = int(appearance.shape[0]) if appearance.ndim == 2 else -1
    if (
        count <= 0
        or boundary.ndim != 2
        or boundary.shape[0] != count
        or appearance_c.shape != (count,)
        or boundary_c.shape != (count,)
        or pairs.ndim != 2
        or pairs.shape[0] != 2
        or bool((pairs < 0).any())
        or bool((pairs >= count).any())
        or bool((pairs[0] >= pairs[1]).any())
        or not bool(torch.isfinite(appearance).all())
        or not bool(torch.isfinite(boundary).all())
        or not bool(torch.isfinite(appearance_c).all())
        or not bool(torch.isfinite(boundary_c).all())
        or bool((appearance_c < 0).any())
        or bool((appearance_c > 1.0001).any())
        or bool((boundary_c < 0).any())
        or bool((boundary_c > 1.0001).any())
        or int(chunk_size) <= 0
    ):
        raise ValueError("V2 capability pair inputs differ")
    appearance = F.normalize(appearance, dim=1)
    boundary = F.normalize(boundary, dim=1)
    pieces: list[torch.Tensor] = []
    for start in range(0, pairs.shape[1], int(chunk_size)):
        selected = pairs[:, start : start + int(chunk_size)]
        left, right = selected[0], selected[1]
        appearance_cosine = (appearance[left] * appearance[right]).sum(dim=1)
        boundary_cosine = (boundary[left] * boundary[right]).sum(dim=1)
        pieces.append(
            torch.stack(
                (
                    appearance_cosine.clamp(-1.0, 1.0),
                    boundary_cosine.clamp(-1.0, 1.0),
                    torch.minimum(appearance_c[left], appearance_c[right]),
                    torch.minimum(boundary_c[left], boundary_c[right]),
                    (appearance_c[left] - appearance_c[right]).abs(),
                    (boundary_c[left] - boundary_c[right]).abs(),
                ),
                dim=1,
            )
        )
    result = torch.cat(pieces).float().contiguous()
    if (
        result.shape != (pairs.shape[1], len(CAPABILITY_PAIR_FEATURE_NAMES))
        or not bool(torch.isfinite(result).all())
    ):
        raise RuntimeError("V2 capability pair materialization failed")
    return result


def _load_capability_scene(
    *,
    base: SceneAuthority,
    descriptor_record: Mapping[str, str],
) -> tuple[SceneAuthority, dict[str, Any]]:
    payload, digest, source = load_torch_mapping(
        descriptor_record["path"],
        expected_sha256=descriptor_record["sha256"],
        map_location="cpu",
        label="V2 region capability descriptor authority",
    )
    descriptor = validate_region_capability_descriptor_authority(payload)
    if (
        descriptor["scene_id"] != base.scene_id
        or descriptor["input_authority"]["accepted_v2"]["sha256"]
        != torch.load(base.record["path"], map_location="cpu", weights_only=False)[
            "input_authority"
        ]["accepted_v2"]["sha256"]
        or descriptor["canonical_region_indices"].numel() != base.region_count
        or list(descriptor["region_fingerprints"])
        != list(
            torch.load(base.record["path"], map_location="cpu", weights_only=False)[
                "region_fingerprints"
            ]
        )
    ):
        raise ValueError("V2 capability and V1 region authority differ")
    appended = capability_pair_features(
        pair_indices=base.pair_indices,
        appearance_direction=descriptor["appearance_direction"],
        boundary_direction=descriptor["boundary_direction"],
        appearance_concentration=descriptor["appearance_concentration"],
        boundary_concentration=descriptor["boundary_concentration"],
    )
    features = torch.cat((base.pair_features, appended), dim=1).contiguous()
    if features.shape[1] != len(PAIR_FEATURE_NAMES):
        raise RuntimeError("V2 pair feature dimension differs")
    return (
        SceneAuthority(
            scene_id=base.scene_id,
            split=base.split,
            record=base.record,
            pair_features=features,
            pair_indices=base.pair_indices,
            targets=base.targets,
            evidence_weights=base.evidence_weights,
            region_count=base.region_count,
            dominant_instance_ids=base.dominant_instance_ids,
            instance_purity=base.instance_purity,
            instance_label_coverage=base.instance_label_coverage,
            instance_observed=base.instance_observed,
        ),
        {
            "record": {"path": str(source), "sha256": digest},
            "region_rows": descriptor["region_rows"].long().cpu().contiguous(),
            "token_mask": descriptor["token_mask"].bool().cpu().contiguous(),
        },
    )


def _fit(
    fit_scene: SceneAuthority, heldout_scene: SceneAuthority
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    active = fit_scene.evidence_weights > 0
    fit = fit_scene.pair_features[active]
    median = fit.median(dim=0).values
    mad = (fit - median).abs().median(dim=0).values
    scale = torch.where(mad > 0, mad * 1.4826, torch.ones_like(mad))
    torch.manual_seed(SEED)
    model = RegionCoMembershipV2(median, scale)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    history = []
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(fit_scene.pair_features)
        loss = balanced_scene_loss(
            logits, fit_scene.targets, fit_scene.evidence_weights
        )
        loss.backward()
        optimizer.step()
        if epoch in {1, 25, 50, 75, 100}:
            history.append({"epoch": epoch, "fit_balanced_bce": float(loss)})
    with torch.no_grad():
        probabilities = {
            scene.scene_id: model.probability(scene.pair_features).cpu()
            for scene in (fit_scene, heldout_scene)
        }
    return probabilities, {
        "median": median.tolist(),
        "robust_scale": scale.tolist(),
        "history": history,
        "heldout_contribution": False,
    }


def _region_curve(
    scene: SceneAuthority, probability: torch.Tensor
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    singleton = None
    for threshold in THRESHOLDS:
        for method_index, method in enumerate(METHODS):
            bounded = _bounded_seed_metrics(
                scene=scene,
                pair_probability=probability,
                threshold=threshold,
                method=method,
            )
            if singleton is None:
                singleton = _bounded_seed_metrics(
                    scene=scene,
                    pair_probability=torch.zeros_like(probability),
                    threshold=1.0,
                    method="maximum_product",
                )[2]
            for maximum in MAXIMUM_REGIONS:
                metric = singleton if maximum == 1 else bounded[maximum]
                rows.append(
                    {
                        "method": method,
                        "method_index": method_index,
                        "maximum_regions": maximum,
                        "threshold": float(threshold),
                        "metrics": metric,
                    }
                )
    return rows


def _selection_key(row: Mapping[str, Any]) -> tuple[float, ...]:
    metric = row["metrics"]
    return (
        float(metric["topology_score"]),
        float(metric["iou"]),
        float(metric["f1"]),
        -float(metric["contamination"]),
        -float(metric["giant_excess"]),
        -float(row["maximum_regions"]),
        -float(row["threshold"]),
        -float(row["method_index"]),
    )


def _region_metric_for_rule(
    scene: SceneAuthority, probability: torch.Tensor, rule: Mapping[str, Any]
) -> dict[str, float]:
    maximum = int(rule["maximum_regions"])
    if maximum == 1:
        return _bounded_seed_metrics(
            scene=scene,
            pair_probability=torch.zeros_like(probability),
            threshold=1.0,
            method="maximum_product",
        )[2]
    return _bounded_seed_metrics(
        scene=scene,
        pair_probability=probability,
        threshold=float(rule["threshold"]),
        method=str(rule["method"]),
    )[maximum]


def _selections_for_rule(
    *,
    scene: SceneAuthority,
    probability: torch.Tensor,
    rule: Mapping[str, Any],
    eligible: torch.Tensor,
) -> dict[int, tuple[int, ...]]:
    maximum = int(rule["maximum_regions"])
    seeds = torch.nonzero(eligible, as_tuple=False).flatten().tolist()
    if maximum == 1:
        return {seed: (seed,) for seed in seeds}
    adjacency = thresholded_adjacency(
        region_count=scene.region_count,
        pair_indices=scene.pair_indices,
        pair_probabilities=probability,
        threshold=float(rule["threshold"]),
    )
    method = str(rule["method"])
    components = (
        bridge_free_component_ids(adjacency)
        if method == "dual_path_widest"
        else None
    )
    return {
        seed: bounded_regions_for_seed(
            method=method,
            seed_region_index=seed,
            adjacency=adjacency,
            maximum_regions=maximum,
            bridge_free_components=components,
        )
        for seed in seeds
    }


def _primitive_metrics(
    *,
    scene: SceneAuthority,
    descriptor: Mapping[str, Any],
    probability: torch.Tensor,
    rule: Mapping[str, Any],
) -> dict[str, Any]:
    authority = torch.load(scene.record["path"], map_location="cpu", weights_only=False)
    inputs = authority["input_authority"]
    dense_mass, audit, exact_record = _load_exact_instance_mass(
        manifest_path=Path(inputs["exact_marginal"]["path"]),
        manifest_sha256=inputs["exact_marginal"]["sha256"],
        instance_zip=Path(inputs["instance_zip"]["path"]),
        instance_zip_sha256=inputs["instance_zip"]["sha256"],
    )
    seed = region_seed_instance_evidence(
        region_rows=descriptor["region_rows"],
        token_mask=descriptor["token_mask"],
        primitive_instance_mass=dense_mass,
    )
    selected = _selections_for_rule(
        scene=scene,
        probability=probability,
        rule=rule,
        eligible=seed["eligible"],
    )
    metric = primitive_instance_union_metrics(
        region_rows=descriptor["region_rows"],
        token_mask=descriptor["token_mask"],
        primitive_instance_mass=dense_mass,
        selections_by_seed=selected,
        maximum_regions=int(rule["maximum_regions"]),
    )
    singleton = primitive_instance_union_metrics(
        region_rows=descriptor["region_rows"],
        token_mask=descriptor["token_mask"],
        primitive_instance_mass=dense_mass,
        selections_by_seed={value: (value,) for value in selected},
        maximum_regions=1,
    )
    return {
        "selected_rule": metric,
        "singleton": singleton,
        "exact_marginal": exact_record,
        "instance_evidence_audit": audit,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"V2 scene1-to-scene2 audit output exists: {output}")
    preregistration = Path(__file__).resolve().parents[2] / PREREGISTRATION
    if sha256_file(preregistration) != args.expected_preregistration_sha256:
        raise ValueError("V2 scene1-to-scene2 preregistration SHA-256 differs")
    base1 = load_scene_authority(
        {"path": args.scene0001_authority, "sha256": args.scene0001_authority_sha256},
        expected_scene_id="scene0001_00",
        expected_split="source_train",
    )
    base2 = load_scene_authority(
        {"path": args.scene0002_authority, "sha256": args.scene0002_authority_sha256},
        expected_scene_id="scene0002_00",
        expected_split="source_train",
    )
    scene1, descriptor1 = _load_capability_scene(
        base=base1,
        descriptor_record={
            "path": args.scene0001_descriptor,
            "sha256": args.scene0001_descriptor_sha256,
        },
    )
    scene2, descriptor2 = _load_capability_scene(
        base=base2,
        descriptor_record={
            "path": args.scene0002_descriptor,
            "sha256": args.scene0002_descriptor_sha256,
        },
    )
    probability, fit = _fit(scene1, scene2)
    curve = _region_curve(scene1, probability[scene1.scene_id])
    selected = max(curve, key=_selection_key)
    heldout_region = _region_metric_for_rule(
        scene2, probability[scene2.scene_id], selected
    )
    print(
        json.dumps(
            {
                "phase": "region_evidence_complete_primitive_evidence_pending",
                "selected_on_scene0001": selected,
                "heldout_scene0002": heldout_region,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    primitive1 = _primitive_metrics(
        scene=scene1,
        descriptor=descriptor1,
        probability=probability[scene1.scene_id],
        rule=selected,
    )
    primitive2 = _primitive_metrics(
        scene=scene2,
        descriptor=descriptor2,
        probability=probability[scene2.scene_id],
        rule=selected,
    )
    report = {
        "schema": "radio_gs.region_comembership_v2_scene1_to_scene2_audit.v1",
        "schema_version": 1,
        "status": "source_scene1_selected_scene2_heldout_complete",
        "preregistration": file_record(preregistration),
        "producer": file_record(Path(__file__).resolve()),
        "fit": fit,
        "inputs": {
            "scene0001_authority": base1.record,
            "scene0002_authority": base2.record,
            "scene0001_descriptor": descriptor1["record"],
            "scene0002_descriptor": descriptor2["record"],
        },
        "selected_on_scene0001_region_evidence": selected,
        "selection_curve_scene0001_region_evidence": curve,
        "heldout_scene0002_region_evidence": heldout_region,
        "scene0001_primitive_instance": primitive1,
        "heldout_scene0002_primitive_instance": primitive2,
        "used_for_formal_selection": False,
        "target_execution_authorized": False,
        "benchmark_opened": False,
        "target_metric_computed": False,
    }
    write_frozen_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene0001-authority", required=True)
    parser.add_argument("--scene0001-authority-sha256", required=True)
    parser.add_argument("--scene0002-authority", required=True)
    parser.add_argument("--scene0002-authority-sha256", required=True)
    parser.add_argument("--scene0001-descriptor", required=True)
    parser.add_argument("--scene0001-descriptor-sha256", required=True)
    parser.add_argument("--scene0002-descriptor", required=True)
    parser.add_argument("--scene0002-descriptor-sha256", required=True)
    parser.add_argument("--expected-preregistration-sha256", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(parser.parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
