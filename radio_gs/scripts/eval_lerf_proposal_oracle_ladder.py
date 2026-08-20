#!/usr/bin/env python3
"""Development-only LERF proposal/association/membership/boundary oracle ladder.

The ladder deliberately changes one authority at a time:

O0: the frozen deployable object posterior;
O1: the best single proposal identity, with fixed lifted membership;
O2: the best greedy cross-view proposal union, with fixed membership;
O3: O2 with an oracle membership threshold;
O4: exact-adjoint Gaussian membership without the proposal-set constraint;
O5: O4 with a per-sample oracle pixel threshold (reported by the renderer).

All target-dependent branches are diagnostics and are never deployable methods.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from radio_gs.scripts import eval_lerf_direct_3d_selection as frozen
from radio_gs.scripts.eval_lerf_grounding import (
    load_lerf_ovs_labels,
    load_render_pipeline,
)
from radio_gs.scripts.eval_lerf_support_readout_oracle_d0_d5 import (
    _materialize_target_membership,
    _primitive_query_metrics,
    _render_membership_diagnostic,
)
from radio_gs.scripts.eval_lerf_teacher_view_oracle_diagnostic import _oracle_iou
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_torch_mapping,
    write_frozen_json,
    write_torch_noclobber,
)


SCHEMA = "radio_gs.lerf_proposal_oracle_ladder.v1"
MEMBERSHIP_FLOOR = 0.50
MAXIMUM_ASSOCIATED_VIEWS = 12


def _normalized_sparse_membership(
    payload: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    rows = torch.as_tensor(payload["row_indices"]).long().cpu()
    proposals = torch.as_tensor(payload["proposal_indices"]).long().cpu()
    weights = torch.as_tensor(payload["weights"]).float().cpu()
    num_rows = int(payload["num_rows"])
    num_proposals = int(payload["num_proposals"])
    valid = (
        (rows >= 0)
        & (rows < num_rows)
        & (proposals >= 0)
        & (proposals < num_proposals)
        & torch.isfinite(weights)
        & (weights > 0)
    )
    rows, proposals, weights = rows[valid], proposals[valid], weights[valid]
    maxima = torch.zeros(num_proposals, dtype=torch.float32)
    maxima.scatter_reduce_(
        0, proposals, weights, reduce="amax", include_self=True
    )
    conditional = weights / maxima[proposals].clamp_min(1e-8)
    return rows, proposals, conditional, num_rows, num_proposals


def _support_sets(
    rows: torch.Tensor,
    proposals: torch.Tensor,
    conditional: torch.Tensor,
    num_proposals: int,
) -> list[set[int]]:
    result: list[set[int]] = [set() for _ in range(num_proposals)]
    keep = conditional >= MEMBERSHIP_FLOOR
    for row, proposal in zip(rows[keep].tolist(), proposals[keep].tolist()):
        result[int(proposal)].add(int(row))
    return result


def _set_iou(prediction: set[int], target: set[int]) -> float:
    union = prediction | target
    return float(len(prediction & target) / len(union)) if union else 1.0


def _compose_continuous_membership(
    *,
    selected: Sequence[int],
    rows: torch.Tensor,
    proposals: torch.Tensor,
    conditional: torch.Tensor,
    num_rows: int,
) -> torch.Tensor:
    result = torch.zeros(num_rows, dtype=torch.float32)
    if not selected:
        return result
    selected_tensor = torch.as_tensor(list(selected), dtype=torch.long)
    keep = torch.isin(proposals, selected_tensor)
    if bool(keep.any()):
        result.scatter_reduce_(
            0,
            rows[keep],
            conditional[keep],
            reduce="amax",
            include_self=True,
        )
    return result


def _proposal_oracles(
    *,
    rows: torch.Tensor,
    proposals: torch.Tensor,
    conditional: torch.Tensor,
    proposal_views: torch.Tensor,
    num_rows: int,
    num_proposals: int,
    target: torch.Tensor,
    observed: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    supports = _support_sets(rows, proposals, conditional, num_proposals)
    query_count = int(target.shape[1])
    o1 = torch.zeros((num_rows, query_count), dtype=torch.float32)
    o2 = torch.zeros_like(o1)
    o3 = torch.zeros_like(o1)
    choices: list[dict[str, Any]] = []
    proposal_ids = list(range(num_proposals))

    for query in range(query_count):
        permitted = set(torch.where(observed[:, query])[0].tolist())
        truth = set(torch.where(target[:, query] & observed[:, query])[0].tolist())
        visible_supports = [value & permitted for value in supports]
        single_iou = [_set_iou(value, truth) for value in visible_supports]
        best = min(
            proposal_ids,
            key=lambda index: (-single_iou[index], index),
        )
        single_continuous = _compose_continuous_membership(
            selected=[best],
            rows=rows,
            proposals=proposals,
            conditional=conditional,
            num_rows=num_rows,
        )
        o1[:, query] = (
            single_continuous >= MEMBERSHIP_FLOOR
        ).float()

        selected = [best]
        selected_views = {int(proposal_views[best])}
        union = set(visible_supports[best])
        current_iou = _set_iou(union, truth)
        while len(selected) < MAXIMUM_ASSOCIATED_VIEWS:
            candidates = [
                index
                for index in proposal_ids
                if int(proposal_views[index]) not in selected_views
            ]
            if not candidates:
                break
            scored = [
                (_set_iou(union | visible_supports[index], truth), index)
                for index in candidates
            ]
            candidate_iou, candidate = max(scored, key=lambda value: (value[0], -value[1]))
            if candidate_iou <= current_iou + 1e-12:
                break
            selected.append(candidate)
            selected_views.add(int(proposal_views[candidate]))
            union |= visible_supports[candidate]
            current_iou = candidate_iou

        associated_continuous = _compose_continuous_membership(
            selected=selected,
            rows=rows,
            proposals=proposals,
            conditional=conditional,
            num_rows=num_rows,
        )
        o2[:, query] = (
            associated_continuous >= MEMBERSHIP_FLOOR
        ).float()
        available = observed[:, query]
        oracle_iou, threshold = _oracle_iou(
            associated_continuous[available].numpy().astype(np.float64),
            target[available, query].numpy(),
        )
        o3[:, query] = (associated_continuous >= float(threshold)).float()
        choices.append(
            {
                "query_index": int(query),
                "single_proposal": int(best),
                "single_view": int(proposal_views[best]),
                "single_primitive_iou": float(single_iou[best]),
                "associated_proposals": [int(value) for value in selected],
                "associated_views": [
                    int(proposal_views[value]) for value in selected
                ],
                "associated_fixed_membership_iou": float(current_iou),
                "oracle_membership_threshold": float(threshold),
                "oracle_membership_iou": float(oracle_iou),
            }
        )
    return {"O1": o1, "O2": o2, "O3": o3}, choices


def run(args: argparse.Namespace) -> Path:
    result_source = Path(args.frozen_result).expanduser().resolve()
    frozen_result = json.loads(result_source.read_text(encoding="utf-8"))
    scene = str(args.scene)
    if scene not in frozen.OPEN_GAUSSIAN_LERF_FRAMES:
        raise ValueError(f"unsupported LERF scene: {scene}")
    if str(frozen_result.get("scene", {}).get("scene")) != scene:
        raise ValueError("frozen result scene differs")

    output = Path(args.output_dir).expanduser().resolve()
    result_path = output / "result.json"
    membership_path = output / "oracle_memberships.pt"
    if result_path.exists():
        raise FileExistsError("refuses to clobber oracle ladder result")
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    source_args = frozen_result["args"]
    annotations, categories, height, width = load_lerf_ovs_labels(
        source_args["label_dir"], scene
    )
    official = set(frozen.OPEN_GAUSSIAN_LERF_FRAMES[scene])
    annotations = {
        frame: objects for frame, objects in annotations.items() if frame in official
    }
    model, codec, old_renderer, sharpener, refiner, config, hybrid = (
        load_render_pipeline(
            source_args["config"],
            source_args["checkpoint"],
            device,
            strict_checkpoint_contract=True,
            load_ply_rgb_features=False,
        )
    )
    del codec, old_renderer, sharpener, refiner, hybrid
    gc.collect()
    if device.type == "cuda":
        # The exact-adjoint compositor temporarily materializes packed hits.
        # Return deleted codec/refiner blocks to the driver before that peak.
        torch.cuda.empty_cache()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    dataset = frozen.build_lerf_dataset_for_scene(
        scene,
        config,
        source_args["label_dir"],
        feature_height=height,
        feature_width=width,
    )
    renderer = frozen.build_mask_renderer(
        config, height=height, width=width, device=device
    )

    score_payload, score_sha, score_path = load_torch_mapping(
        args.score_cache, map_location="cpu", label="frozen object posterior"
    )
    membership_payload, proposal_sha, proposal_path = load_torch_mapping(
        args.proposal_membership,
        map_location="cpu",
        label="query-independent proposal membership",
    )
    if str(score_payload.get("scene")) != scene or str(
        membership_payload.get("scene")
    ) != scene:
        raise ValueError("scene cache differs")
    query_ids = list(score_payload["metadata"]["query_names"])
    if query_ids != list(categories):
        raise ValueError("score-cache query order differs from official labels")
    xyz = model.get_xyz().detach().float().cpu()
    if not torch.equal(torch.as_tensor(score_payload["xyz"]).float(), xyz):
        raise ValueError("score cache/model Gaussian rows differ")

    rows, proposals, conditional, num_rows, num_proposals = (
        _normalized_sparse_membership(membership_payload)
    )
    if num_rows != int(xyz.shape[0]):
        raise ValueError("proposal membership/model row count differs")
    resumed_membership = membership_path.is_file()
    if resumed_membership:
        sealed, _, _ = load_torch_mapping(
            membership_path,
            map_location="cpu",
            label="sealed oracle memberships",
        )
        if sealed.get("schema") != SCHEMA or sealed.get("scene") != scene:
            raise ValueError("sealed oracle membership contract differs")
        if list(sealed.get("query_ids", [])) != query_ids:
            raise ValueError("sealed oracle membership query order differs")
        target_probability = torch.as_tensor(
            sealed["target_probability"]
        ).float()
        target_observed = torch.as_tensor(sealed["target_observed"]).bool()
        variants = {
            str(name): torch.as_tensor(value).float()
            for name, value in sealed["variants"].items()
        }
        choices = list(sealed["oracle_choices"])
        adjoint_frames = list(sealed.get("target_adjoint_frames", []))
    else:
        target_probability, target_observed, adjoint_frames = (
            _materialize_target_membership(
                annotations=annotations,
                categories=categories,
                height=height,
                width=width,
                model=model,
                renderer=renderer,
                dataset=dataset,
                device=device,
            )
        )
        target_binary = target_probability >= 0.5
        oracle_memberships, choices = _proposal_oracles(
            rows=rows,
            proposals=proposals,
            conditional=conditional,
            proposal_views=torch.as_tensor(
                membership_payload["proposal_view_indices"]
            ).long(),
            num_rows=num_rows,
            num_proposals=num_proposals,
            target=target_binary,
            observed=target_observed,
        )
        variants = {
            "O0_current": torch.as_tensor(score_payload["query_scores"]).float(),
            "O1_oracle_identity": oracle_memberships["O1"],
            "O2_oracle_association": oracle_memberships["O2"],
            "O3_oracle_membership": oracle_memberships["O3"],
            "O4_exact_adjoint_membership": target_probability,
        }
    target_binary = target_probability >= 0.5
    primitive = {
        name: _primitive_query_metrics(
            value,
            target_binary,
            target_observed,
            torch.ones(num_rows, dtype=torch.bool),
            include_ranking_metrics=False,
        )
        for name, value in variants.items()
    }
    rendered = {
        name: _render_membership_diagnostic(
            name=name,
            membership=value,
            annotations=annotations,
            categories=categories,
            height=height,
            width=width,
            model=model,
            renderer=renderer,
            dataset=dataset,
            device=device,
            include_ranking_metrics=False,
            include_oracle_threshold=(name == "O4_exact_adjoint_membership"),
        )
        for name, value in variants.items()
    }
    if not resumed_membership:
        membership_artifact = {
            "schema": SCHEMA,
            "scene": scene,
            "query_ids": query_ids,
            "target_probability": target_probability,
            "target_observed": target_observed,
            "variants": variants,
            "oracle_choices": choices,
            "target_adjoint_frames": adjoint_frames,
            "metadata": {
                "development_only": True,
                "benchmark_method": False,
                "gt_oracle": True,
                "membership_floor": MEMBERSHIP_FLOOR,
                "maximum_associated_views": MAXIMUM_ASSOCIATED_VIEWS,
            },
        }
        write_torch_noclobber(membership_path, membership_artifact)
    result = {
        "schema": SCHEMA,
        "scene": scene,
        "development_only": True,
        "benchmark_method": False,
        "deployable_candidate": False,
        "ladder": {
            "O0": "frozen deployable object posterior",
            "O1": "oracle single-proposal identity; fixed membership floor",
            "O2": "oracle greedy distinct-view association; fixed membership floor",
            "O3": "O2 association with oracle primitive membership threshold",
            "O4": "exact-adjoint Gaussian membership without proposal constraint",
            "O5": "O4 renderer with per-sample oracle pixel threshold",
        },
        "primitive": primitive,
        "rendered": rendered,
        "oracle_choices": choices,
        "target_adjoint_frames": adjoint_frames,
        "resumed_from_sealed_membership": resumed_membership,
        "artifacts": {
            "frozen_result": file_record(result_source),
            "score_cache": {"path": str(score_path), "sha256": score_sha},
            "proposal_membership": {
                "path": str(proposal_path),
                "sha256": proposal_sha,
            },
            "memberships": file_record(membership_path),
            "producer": file_record(Path(__file__).resolve()),
        },
        "contract_sha256": canonical_json_sha256(
            {
                "official_frames": sorted(official),
                "target_membership": "exact_adjoint_ratio_ge_0p5",
                "proposal_membership_floor": MEMBERSHIP_FLOOR,
                "association": "greedy_iou_improving_distinct_view_union",
                "maximum_associated_views": MAXIMUM_ASSOCIATED_VIEWS,
                "o5": "per_sample_oracle_pixel_threshold_of_exact_adjoint_membership",
            }
        ),
    }
    write_frozen_json(result_path, result)
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--frozen-result", required=True)
    parser.add_argument("--score-cache", required=True)
    parser.add_argument("--proposal-membership", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
