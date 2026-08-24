#!/usr/bin/env python3
"""Gate registered NVOS SAM observations by cross-view mask reconstruction.

The official SAM region in one registered mapping view is exact-adjointed to
the frozen Gaussian carrier and rendered into the independent prompt and
evaluation views.  A mapping observation is authoritative only when both
held-out renders reconstruct the corresponding official SAM regions.  Native
SigLIP appearance remains a tie breaker; it cannot promote a view that fails
the geometric/object-membership gate.  No benchmark mask or metric is opened.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from radio_gs.scripts.build_nvos_native_siglip_view_reliability import (
    resolve_view_roles,
)
from radio_gs.scripts.build_nvos_two_round_exact_consensus import (
    exact_adjoint_probability,
    exact_forward_probability,
)
from radio_gs.scripts.materialize_nvos_synchronous_candidate_marginal import (
    _load_assignment,
    _load_probability,
)
from radio_gs.utils.immutable_artifacts import sha256_file, write_frozen_json


def supported_binary_iou(
    prediction: torch.Tensor,
    target: torch.Tensor,
    supported: torch.Tensor,
    *,
    decision_boundary: float = 0.5,
) -> float:
    """Jaccard overlap on pixels connected by the exact 3D transport."""

    predicted = torch.as_tensor(prediction).float()
    reference = torch.as_tensor(target).float().to(predicted.device)
    valid = torch.as_tensor(supported).bool().to(predicted.device)
    if predicted.shape != reference.shape or predicted.shape != valid.shape:
        raise ValueError("held-out reconstruction raster axes differ")
    if not bool(torch.isfinite(predicted).all()) or not bool(
        torch.isfinite(reference).all()
    ):
        raise ValueError("held-out reconstruction contains nonfinite values")
    left = (predicted > float(decision_boundary)) & valid
    right = (reference > float(decision_boundary)) & valid
    union = int((left | right).sum())
    if union <= 0:
        return 0.0
    return float((left & right).sum()) / float(union)


def select_mapping_views(
    roles: Sequence[str],
    heldout_scores: torch.Tensor,
    appearance_scores: torch.Tensor,
    *,
    minimum_heldout_iou: float,
    top_k: int,
) -> tuple[int, ...]:
    """Keep evaluation plus only source observations passing held-out replay."""

    geometry = torch.as_tensor(heldout_scores).float().reshape(-1)
    appearance = torch.as_tensor(appearance_scores).float().reshape(-1)
    if (
        geometry.numel() != len(roles)
        or appearance.numel() != len(roles)
        or int(top_k) < 0
        or not 0.0 <= float(minimum_heldout_iou) <= 1.0
    ):
        raise ValueError("registered-view reliability axes or constants differ")
    evaluation = [
        index for index, role in enumerate(roles) if str(role) == "evaluation"
    ]
    prompt = [index for index, role in enumerate(roles) if str(role) == "prompt"]
    if len(evaluation) != 1 or len(prompt) != 1:
        raise ValueError("held-out gate requires one prompt and evaluation anchor")
    retained_prompt = [
        prompt[0]
        if float(geometry[prompt[0]]) >= float(minimum_heldout_iou)
        else None
    ]
    eligible = [
        index
        for index, role in enumerate(roles)
        if str(role) == "registered_mapping"
        and float(geometry[index]) >= float(minimum_heldout_iou)
    ]
    ranked = sorted(
        eligible,
        key=lambda index: (
            -float(geometry[index]),
            -float(appearance[index]),
            index,
        ),
    )
    retained = evaluation + [index for index in retained_prompt if index is not None]
    return tuple(sorted(retained + ranked[: min(int(top_k), len(ranked))]))


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be one regular JSON file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain one object")
    return dict(value)


def _assignment_on_device(
    record: Mapping[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    loaded = _load_assignment(record)
    return {
        name: torch.as_tensor(loaded[name]).to(device)
        for name in ("gaussian_ids", "pixel_ids", "weights")
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    inventory_path = Path(args.inventory).expanduser().resolve(strict=True)
    appearance_path = Path(args.appearance_inventory).expanduser().resolve(strict=True)
    plan_path = Path(args.plan).expanduser().resolve(strict=True)
    for path, expected, label in (
        (inventory_path, args.expected_inventory_sha256, "native SAM inventory"),
        (appearance_path, args.expected_appearance_sha256, "appearance inventory"),
        (plan_path, args.expected_plan_sha256, "registered-view plan"),
    ):
        if sha256_file(path) != str(expected):
            raise ValueError(f"{label} SHA-256 differs")

    inventory = _load_json(inventory_path, label="native SAM inventory")
    appearance = _load_json(appearance_path, label="appearance inventory")
    plan = _load_json(plan_path, label="registered-view plan")
    candidates = inventory.get("candidates")
    plan_candidates = plan.get("candidates")
    if (
        not isinstance(candidates, list)
        or len(candidates) != 1
        or not isinstance(plan_candidates, list)
        or not plan_candidates
        or inventory.get("candidate_count") != 1
        or inventory.get("target_mask_opened") is not False
        or inventory.get("target_metric_opened") is not False
        or plan.get("registered_view_contract")
        != "complete_queue_locked_rgb_camera_map"
    ):
        raise ValueError("registered-view held-out contract differs")
    source_record = appearance.get("native_appearance_reliability", {}).get(
        "source_inventory", {}
    )
    if source_record != {
        "path": str(inventory_path),
        "sha256": str(args.expected_inventory_sha256),
    }:
        raise ValueError("appearance inventory source lineage differs")

    plan_views = {
        str(row["view_digest"]): dict(row)
        for row in plan_candidates[0].get("views", [])
        if isinstance(row, Mapping)
    }
    inventory_views = list(candidates[0].get("views", []))
    if (
        not plan_views
        or len(inventory_views) != len(plan_views)
        or {str(row.get("view_digest", "")) for row in inventory_views}
        != set(plan_views)
    ):
        raise ValueError("plan and native SAM registered-view cohorts differ")

    ordered_plan = [plan_views[str(row["view_digest"])] for row in inventory_views]
    dataset_record = plan.get("inputs", {}).get("dataset_manifest", {})
    dataset_path = Path(str(dataset_record.get("path", ""))).resolve(strict=True)
    if sha256_file(dataset_path) != str(dataset_record.get("sha256", "")):
        raise ValueError("NVOS dataset manifest SHA-256 differs")
    dataset = _load_json(dataset_path, label="NVOS dataset manifest")
    scene_rows = [
        row
        for row in dataset.get("scenes", [])
        if isinstance(row, Mapping)
        and str(row.get("scene_id")) == str(plan.get("scene_id"))
    ]
    if len(scene_rows) != 1:
        raise ValueError("NVOS scene is absent from dataset manifest")
    roles = resolve_view_roles(
        [str(row.get("frame_id", "")) for row in ordered_plan],
        prompt_frame_ids=scene_rows[0].get("prompt_frame_ids", []),
        evaluation_frame_ids=scene_rows[0].get("evaluation_frame_ids", []),
        explicit_roles=[str(row.get("role", "")) for row in ordered_plan],
    )

    appearance_rows = appearance.get("native_appearance_reliability", {}).get(
        "all_registered_view_scores", []
    )
    appearance_by_digest = {
        str(row.get("view_digest", "")): float(row.get("score", float("nan")))
        for row in appearance_rows
        if isinstance(row, Mapping)
    }
    if set(appearance_by_digest) != set(plan_views) or not all(
        math.isfinite(value) for value in appearance_by_digest.values()
    ):
        raise ValueError("native appearance reliability cohort differs")
    appearance_scores = torch.tensor(
        [appearance_by_digest[str(row["view_digest"])] for row in ordered_plan]
    )

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("held-out exact transport requires CUDA")
    num_gaussians = int(inventory.get("num_gaussians", 0))
    if num_gaussians <= 0:
        raise ValueError("native SAM carrier size is absent")
    anchor_indices = {
        role: roles.index(role) for role in ("prompt", "evaluation")
    }
    anchor_assignments = {
        role: _assignment_on_device(
            ordered_plan[index]["assignment"], device
        )
        for role, index in anchor_indices.items()
    }
    anchor_probability = {
        role: _load_probability(inventory_views[index]["probability"])
        .reshape(
            np.load(
                Path(inventory_views[index]["probability"]["path"]),
                allow_pickle=False,
                mmap_mode="r",
            ).shape
        )
        .to(device)
        for role, index in anchor_indices.items()
    }
    height, width = map(int, next(iter(anchor_probability.values())).shape)
    if any(tuple(value.shape) != (height, width) for value in anchor_probability.values()):
        raise ValueError("prompt/evaluation SAM raster axes differ")
    fallback = torch.zeros(height * width, dtype=torch.float32, device=device)

    heldout_scores = torch.ones(len(inventory_views), dtype=torch.float32)
    reconstruction_records: list[dict[str, Any]] = []
    for index, (role, view, plan_view) in enumerate(
        zip(roles, inventory_views, ordered_plan)
    ):
        if role == "evaluation":
            reconstruction_records.append(
                {
                    "frame_id": str(plan_view.get("frame_id", "")),
                    "role": role,
                    "view_digest": str(plan_view["view_digest"]),
                    "minimum_anchor_iou": 1.0,
                    "anchor_iou": {},
                    "mandatory_anchor": True,
                }
            )
            continue
        source_assignment = _assignment_on_device(plan_view["assignment"], device)
        source_probability = _load_probability(view["probability"]).to(device)
        primitive, source_mass = exact_adjoint_probability(
            source_assignment["gaussian_ids"],
            source_assignment["pixel_ids"],
            source_assignment["weights"],
            source_probability,
            num_gaussians=num_gaussians,
        )
        # Invisible is unknown, hence zero affirmative support for this
        # source-to-heldout reconstruction test.
        primitive[source_mass <= 0] = 0.0
        per_anchor: dict[str, float] = {}
        for anchor_role, assignment in anchor_assignments.items():
            if anchor_role == role:
                continue
            projected, mass = exact_forward_probability(
                assignment["gaussian_ids"],
                assignment["pixel_ids"],
                assignment["weights"],
                primitive,
                height=height,
                width=width,
                unsupported_fallback=fallback,
            )
            per_anchor[anchor_role] = supported_binary_iou(
                projected,
                anchor_probability[anchor_role],
                mass > 0,
            )
        minimum = min(per_anchor.values())
        heldout_scores[index] = minimum
        reconstruction_records.append(
            {
                "frame_id": str(plan_view.get("frame_id", "")),
                "role": role,
                "view_digest": str(plan_view["view_digest"]),
                "minimum_anchor_iou": minimum,
                "anchor_iou": per_anchor,
                "mandatory_anchor": False,
            }
        )
        del source_assignment, source_probability, primitive, source_mass

    retained = select_mapping_views(
        roles,
        heldout_scores,
        appearance_scores,
        minimum_heldout_iou=float(args.minimum_heldout_iou),
        top_k=int(args.mapping_top_k),
    )
    output_inventory = copy.deepcopy(inventory)
    output_views: list[dict[str, Any]] = []
    for index in retained:
        row = copy.deepcopy(inventory_views[index])
        row["registered_view_role"] = roles[index]
        row["native_siglip2_identity_score"] = float(appearance_scores[index])
        row["source_heldout_minimum_anchor_iou"] = float(heldout_scores[index])
        if roles[index] != "evaluation":
            row["log_precision"] = float(row["log_precision"]) + math.log(
                max(float(heldout_scores[index]), 1e-8)
            )
        output_views.append(row)
    output_inventory["candidates"][0]["views"] = output_views
    output_inventory["view_count"] = len(output_views)
    output_inventory["view_digests"] = sorted(
        str(row["view_digest"]) for row in output_views
    )
    output_inventory["view_selection"] = (
        "source_heldout_two_anchor_mask_reconstruction_then_native_siglip2"
    )
    output_inventory["candidate_selection"] = (
        "official_sam3_extent_plus_source_heldout_membership_authority"
    )
    output_inventory["source_heldout_view_reliability"] = {
        "schema": "radio_gs.nvos_source_heldout_view_reliability.v1",
        "source_inventory": {
            "path": str(inventory_path),
            "sha256": str(args.expected_inventory_sha256),
        },
        "appearance_inventory": {
            "path": str(appearance_path),
            "sha256": str(args.expected_appearance_sha256),
        },
        "source_plan": {
            "path": str(plan_path),
            "sha256": str(args.expected_plan_sha256),
        },
        "minimum_heldout_iou": float(args.minimum_heldout_iou),
        "mapping_top_k": int(args.mapping_top_k),
        "source_precision": "log_exact_precision_plus_log_minimum_heldout_anchor_iou",
        "ranking": "minimum_anchor_iou_then_native_siglip2_identity",
        "per_registered_view": reconstruction_records,
        "access_audit": {
            "protocol_authorized_registered_rgb_sam_opened": True,
            "benchmark_target_mask_opened": False,
            "benchmark_target_metric_opened": False,
        },
    }
    output_path = write_frozen_json(args.output, output_inventory)
    return {
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "registered_views": len(inventory_views),
        "retained_views": len(output_views),
        "retained_mapping_views": sum(
            roles[index] == "registered_mapping" for index in retained
        ),
        "mapping_views_passing_gate": sum(
            role == "registered_mapping"
            and float(heldout_scores[index]) >= float(args.minimum_heldout_iou)
            for index, role in enumerate(roles)
        ),
        "target_mask_opened": False,
        "target_metric_opened": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--expected-inventory-sha256", required=True)
    parser.add_argument("--appearance-inventory", required=True)
    parser.add_argument("--expected-appearance-sha256", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--minimum-heldout-iou", type=float, default=0.5)
    parser.add_argument("--mapping-top-k", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = build(build_parser().parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
