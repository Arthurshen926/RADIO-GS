#!/usr/bin/env python3
"""Reweight sealed NVOS SAM views with native SigLIP2 instance identity.

Official SAM3 continues to own per-view extent.  The prompt-view SAM region is
the physical identity anchor, while the evaluation-view SAM region supplies a
second protocol-authorized observation.  Registered mapping views are ranked
by the minimum native SigLIP2 masked/context similarity to those two anchors.
Only a fixed top-k mapping cohort is retained, and its exact-adjoint precision
is softly downweighted by appearance disagreement.  No target mask, metric, or
scene-specific threshold is reachable.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor

from radio_gs.scripts.build_multiscale_sam_mask_aligned_crop_summary_teacher import (
    NativeSiglip2Runtime,
)
from radio_gs.scripts.build_sam_mask_aligned_language_teacher import (
    build_crop_pairs,
    encode_in_batches,
)
from radio_gs.utils.immutable_artifacts import sha256_file, write_frozen_json


def appearance_reliability(
    prompt: torch.Tensor,
    evaluation: torch.Tensor,
    descriptors: torch.Tensor,
) -> torch.Tensor:
    """Return the conservative two-anchor cosine for every registered view."""

    query = F.normalize(torch.as_tensor(prompt).float().reshape(1, -1), dim=-1)
    target = F.normalize(
        torch.as_tensor(evaluation).float().reshape(1, -1), dim=-1
    )
    values = F.normalize(torch.as_tensor(descriptors).float(), dim=-1)
    if values.ndim != 2 or values.shape[1] != query.shape[1]:
        raise ValueError("native appearance descriptor axes differ")
    return torch.minimum(values @ query.T, values @ target.T).reshape(-1)


def select_mapping_views(
    roles: Sequence[str], scores: torch.Tensor, *, top_k: int
) -> tuple[int, ...]:
    """Keep both protocol views plus a stable fixed-size mapping retrieval."""

    values = torch.as_tensor(scores).float().reshape(-1)
    if len(roles) != values.numel() or int(top_k) < 0:
        raise ValueError("view role/score axis or top-k differs")
    mandatory = [
        index
        for index, role in enumerate(roles)
        if str(role) in {"prompt", "evaluation"}
    ]
    if len(mandatory) != 2:
        raise ValueError("appearance view cohort requires one prompt and evaluation")
    mapping = [
        index
        for index, role in enumerate(roles)
        if str(role) == "registered_mapping"
    ]
    ranked = sorted(mapping, key=lambda index: (-float(values[index]), index))
    return tuple(sorted(mandatory + ranked[: min(int(top_k), len(ranked))]))


def resolve_view_roles(
    frame_ids: Sequence[str],
    *,
    prompt_frame_ids: Sequence[str],
    evaluation_frame_ids: Sequence[str],
    explicit_roles: Sequence[str] | None = None,
) -> list[str]:
    """Recover roles from the sealed benchmark manifest and cross-check v2 plans."""

    prompts = {str(value) for value in prompt_frame_ids}
    evaluations = {str(value) for value in evaluation_frame_ids}
    if len(prompts) != 1 or len(evaluations) != 1 or prompts & evaluations:
        raise ValueError("NVOS prompt/evaluation frame authority differs")
    roles = [
        "prompt"
        if str(frame) in prompts
        else "evaluation"
        if str(frame) in evaluations
        else "registered_mapping"
        for frame in frame_ids
    ]
    if explicit_roles is not None:
        explicit = [str(value) for value in explicit_roles]
        if any(explicit) and explicit != roles:
            raise ValueError("explicit plan roles differ from dataset manifest")
    return roles


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be one regular JSON file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain one object")
    return dict(value)


def _load_mask(record: Mapping[str, Any]) -> np.ndarray:
    path = Path(str(record.get("path", ""))).expanduser().resolve(strict=True)
    if sha256_file(path) != str(record.get("sha256", "")):
        raise ValueError("selected official-SAM probability SHA-256 differs")
    value = np.load(path, allow_pickle=False)
    mask = np.asarray(value, dtype=np.float32)
    if mask.ndim != 2 or not bool(np.isfinite(mask).all()):
        raise ValueError("selected official-SAM probability is malformed")
    return mask >= 0.5


def _box(mask: np.ndarray) -> torch.Tensor:
    rows, columns = np.where(np.asarray(mask, dtype=bool))
    if not rows.size:
        raise ValueError("selected official-SAM region is empty")
    return torch.tensor(
        [[int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1]],
        dtype=torch.long,
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    inventory_path = Path(args.inventory).expanduser().resolve(strict=True)
    plan_path = Path(args.plan).expanduser().resolve(strict=True)
    if sha256_file(inventory_path) != str(args.expected_inventory_sha256):
        raise ValueError("native SAM inventory SHA-256 differs")
    if sha256_file(plan_path) != str(args.expected_plan_sha256):
        raise ValueError("registered-view plan SHA-256 differs")
    inventory = _load_json(inventory_path, label="native SAM inventory")
    plan = _load_json(plan_path, label="registered-view plan")
    if (
        inventory.get("plan")
        != {"path": str(plan_path), "sha256": str(args.expected_plan_sha256)}
        or inventory.get("candidate_count") != 1
        or inventory.get("target_mask_opened") is not False
        or inventory.get("target_metric_opened") is not False
        or plan.get("registered_view_contract")
        != "complete_queue_locked_rgb_camera_map"
    ):
        raise ValueError("native registered-view inventory contract differs")
    plan_candidates = plan.get("candidates")
    candidates = inventory.get("candidates")
    if not isinstance(plan_candidates, list) or not plan_candidates:
        raise ValueError("registered-view plan candidate cohort is absent")
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise ValueError("native SAM inventory candidate cohort differs")
    plan_views = {
        str(record["view_digest"]): dict(record)
        for record in plan_candidates[0].get("views", [])
        if isinstance(record, Mapping)
    }
    inventory_views = list(candidates[0].get("views", []))
    if (
        not plan_views
        or len(inventory_views) != len(plan_views)
        or {str(record.get("view_digest", "")) for record in inventory_views}
        != set(plan_views)
    ):
        raise ValueError("plan and native SAM registered-view cohorts differ")

    device = torch.device(args.device)
    runtime = NativeSiglip2Runtime.load(
        Path(args.native_siglip2_model), device=device
    )
    masked_crops: list[torch.Tensor] = []
    context_crops: list[torch.Tensor] = []
    ordered_plan: list[dict[str, Any]] = []
    for inventory_view in inventory_views:
        digest = str(inventory_view["view_digest"])
        record = plan_views[digest]
        rgb_record = record.get("rgb", {})
        rgb_path = Path(str(rgb_record.get("path", ""))).expanduser().resolve(strict=True)
        if sha256_file(rgb_path) != str(rgb_record.get("sha256", "")):
            raise ValueError("registered RGB SHA-256 differs")
        mask = _load_mask(inventory_view["probability"])
        image = Image.open(rgb_path).convert("RGB").resize(
            (int(mask.shape[1]), int(mask.shape[0])), Image.Resampling.LANCZOS
        )
        image_tensor = pil_to_tensor(image).float().div_(255.0)
        masked, context = build_crop_pairs(
            image_tensor,
            mask[None],
            _box(mask),
            context_expansion=float(args.context_expansion),
            crop_resolution=384,
            masked_background_rgb=(0.5, 0.5, 0.5),
        )
        masked_crops.append(masked)
        context_crops.append(context)
        ordered_plan.append(record)
    masked_descriptor = encode_in_batches(
        runtime,
        torch.cat(masked_crops),
        batch_size=int(args.batch_size),
        device=device,
    ).float()
    context_descriptor = encode_in_batches(
        runtime,
        torch.cat(context_crops),
        batch_size=int(args.batch_size),
        device=device,
    ).float()
    descriptors = F.normalize(
        0.75 * masked_descriptor + 0.25 * context_descriptor,
        dim=-1,
        eps=1e-8,
    )
    dataset_record = plan.get("inputs", {}).get("dataset_manifest", {})
    dataset_path = Path(str(dataset_record.get("path", ""))).resolve(strict=True)
    if sha256_file(dataset_path) != str(dataset_record.get("sha256", "")):
        raise ValueError("NVOS dataset manifest SHA-256 differs")
    dataset_manifest = _load_json(dataset_path, label="NVOS dataset manifest")
    scenes = [
        row
        for row in dataset_manifest.get("scenes", [])
        if isinstance(row, Mapping) and str(row.get("scene_id")) == str(plan.get("scene_id"))
    ]
    if len(scenes) != 1:
        raise ValueError("NVOS scene is absent from dataset manifest")
    roles = resolve_view_roles(
        [str(record.get("frame_id", "")) for record in ordered_plan],
        prompt_frame_ids=scenes[0].get("prompt_frame_ids", []),
        evaluation_frame_ids=scenes[0].get("evaluation_frame_ids", []),
        explicit_roles=[str(record.get("role", "")) for record in ordered_plan],
    )
    prompt_index = roles.index("prompt")
    evaluation_index = roles.index("evaluation")
    scores = appearance_reliability(
        descriptors[prompt_index], descriptors[evaluation_index], descriptors
    )
    retained = select_mapping_views(roles, scores, top_k=int(args.mapping_top_k))
    mapping_retained = [
        index for index in retained if roles[index] == "registered_mapping"
    ]
    maximum_mapping_score = max(
        (float(scores[index]) for index in mapping_retained), default=1.0
    )
    output_inventory = copy.deepcopy(inventory)
    output_views = []
    for index in retained:
        record = copy.deepcopy(inventory_views[index])
        if roles[index] == "registered_mapping":
            record["log_precision"] = float(record["log_precision"]) + float(
                args.appearance_temperature
            ) * (float(scores[index]) - maximum_mapping_score)
        record["native_siglip2_identity_score"] = float(scores[index])
        record["registered_view_role"] = roles[index]
        output_views.append(record)
    output_inventory["candidates"][0]["views"] = output_views
    output_inventory["view_count"] = len(output_views)
    output_inventory["view_digests"] = sorted(
        str(record["view_digest"]) for record in output_views
    )
    output_inventory["view_selection"] = (
        "fixed_topk_native_siglip2_two_anchor_instance_retrieval"
    )
    output_inventory["candidate_selection"] = (
        "official_sam3_extent_plus_native_siglip2_instance_identity"
    )
    output_inventory["native_appearance_reliability"] = {
        "schema": "radio_gs.nvos_native_siglip2_view_reliability.v1",
        "source_inventory": {
            "path": str(inventory_path),
            "sha256": str(args.expected_inventory_sha256),
        },
        "source_plan": {
            "path": str(plan_path),
            "sha256": str(args.expected_plan_sha256),
        },
        "encoder": runtime.bundle,
        "descriptor": "normalize(0.75*masked_crop+0.25*expanded_context_crop)",
        "identity_score": "min(cosine(prompt_anchor),cosine(evaluation_anchor))",
        "mapping_top_k": int(args.mapping_top_k),
        "appearance_temperature": float(args.appearance_temperature),
        "per_scene_or_metric_parameter": False,
        "all_registered_view_scores": [
            {
                "frame_id": str(record.get("frame_id", "")),
                "role": roles[index],
                "view_digest": str(record["view_digest"]),
                "score": float(scores[index]),
                "retained": index in retained,
            }
            for index, record in enumerate(ordered_plan)
        ],
        "access_audit": {
            "protocol_authorized_registered_rgb_opened": True,
            "target_mask_opened": False,
            "target_metric_opened": False,
        },
    }
    output_path = write_frozen_json(args.output, output_inventory)
    return {
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "registered_views": len(inventory_views),
        "retained_views": len(output_views),
        "retained_mapping_views": len(mapping_retained),
        "prompt_evaluation_cosine": float(
            descriptors[prompt_index] @ descriptors[evaluation_index]
        ),
        "target_mask_opened": False,
        "target_metric_opened": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--expected-inventory-sha256", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--native-siglip2-model", required=True)
    parser.add_argument("--mapping-top-k", type=int, default=4)
    parser.add_argument("--appearance-temperature", type=float, default=16.0)
    parser.add_argument("--context-expansion", type=float, default=1.5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = build(build_parser().parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
