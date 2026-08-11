#!/usr/bin/env python3
"""Lift sealed source-view SAM3 proposals into a target-blind region memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image
import torch

from radio_gs.interfaces.prompt_responsibility_cache import tensor_sha256
from radio_gs.querying.multiview_region_memory import (
    METHOD,
    aggregate_proposal_membership,
    method_contract,
    sample_native_mask_at_feature_centers,
)
from radio_gs.scripts.materialize_multiview_region_memory_coarse import (
    ARTIFACT_TYPE as COARSE_ARTIFACT_TYPE,
    DECODER_HEIGHT,
    DECODER_WIDTH,
    SOURCE_VIEW_COUNT,
)
from radio_gs.scripts.refine_multiview_region_memory_official_sam3 import (
    ARTIFACT_TYPE as SOURCE_SAM3_ARTIFACT_TYPE,
    VIEW_ARTIFACT_TYPE,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    load_torch_mapping,
    sha256_file,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


ARTIFACT_TYPE = "multiview_region_memory_primitive_asset_v1"
RECEIPT_ARTIFACT_TYPE = "multiview_region_memory_primitive_receipt_v1"


def normalized_box_observation_domain(
    box_cxcywh: list[float], *, height: int, width: int
) -> torch.Tensor:
    """Return the pixel-centre domain of the already padded SAM3 box."""

    if len(box_cxcywh) != 4 or int(height) <= 0 or int(width) <= 0:
        raise ValueError("normalized box and output shape are malformed")
    values = torch.as_tensor(box_cxcywh, dtype=torch.float64).reshape(-1)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("normalized box must be finite")
    cx, cy, box_width, box_height = (float(value) for value in values)
    if (
        box_width <= 0
        or box_height <= 0
        or cx < 0
        or cx > 1
        or cy < 0
        or cy > 1
        or box_width > 1
        or box_height > 1
    ):
        raise ValueError("normalized box must be positive and lie in [0,1]")
    x0, x1 = max(0.0, cx - box_width / 2), min(1.0, cx + box_width / 2)
    y0, y1 = max(0.0, cy - box_height / 2), min(1.0, cy + box_height / 2)
    xs = (torch.arange(int(width), dtype=torch.float64) + 0.5) / float(width)
    ys = (torch.arange(int(height), dtype=torch.float64) + 0.5) / float(height)
    domain = (
        (ys[:, None] >= y0)
        & (ys[:, None] <= y1)
        & (xs[None, :] >= x0)
        & (xs[None, :] <= x1)
    )
    if not bool(domain.any()):
        raise ValueError("normalized box contains no output pixel centre")
    return domain.contiguous()


def proposal_positive_mass(
    assignment: Mapping[str, torch.Tensor],
    proposal_mask: torch.Tensor,
    observation_domain: torch.Tensor,
    *,
    num_gaussians: int,
    view_reliability: float,
) -> torch.Tensor:
    """Lift one proposal's positive evidence mass to primitive rows."""

    proposal = torch.as_tensor(proposal_mask, device="cpu")
    domain = torch.as_tensor(observation_domain, device="cpu")
    if (
        proposal.ndim != 2
        or proposal.dtype != torch.bool
        or domain.shape != proposal.shape
        or domain.dtype != torch.bool
        or not isinstance(assignment, Mapping)
        or set(assignment) != {"gaussian_ids", "pixel_ids", "weights"}
    ):
        raise ValueError("proposal, domain, or assignment schema differs")
    reliability = float(view_reliability)
    gaussian_ids = torch.as_tensor(assignment["gaussian_ids"]).long().reshape(-1)
    pixel_ids = torch.as_tensor(assignment["pixel_ids"]).long().reshape(-1)
    weights = torch.as_tensor(assignment["weights"]).float().reshape(-1)
    if (
        not 0 <= reliability <= 1
        or gaussian_ids.shape != pixel_ids.shape
        or gaussian_ids.shape != weights.shape
        or gaussian_ids.numel() == 0
        or int(gaussian_ids.min()) < 0
        or int(gaussian_ids.max()) >= int(num_gaussians)
        or int(pixel_ids.min()) < 0
        or int(pixel_ids.max()) >= proposal.numel()
        or not bool(torch.isfinite(weights).all())
        or bool((weights <= 0).any())
    ):
        raise ValueError("proposal positive-mass authority differs")
    positive = proposal.reshape(-1)[pixel_ids] & domain.reshape(-1)[pixel_ids]
    mass = torch.zeros(int(num_gaussians), dtype=torch.float32)
    if bool(positive.any()) and reliability > 0:
        mass.index_add_(
            0, gaussian_ids[positive], weights[positive] * reliability
        )
    return mass


def _load_mask(record: Mapping[str, Any], *, label: str) -> torch.Tensor:
    path = Path(str(record.get("path", ""))).expanduser().resolve(strict=True)
    if sha256_file(path) != str(record.get("sha256", "")):
        raise ValueError(f"{label} changed after sealing")
    with Image.open(path) as handle:
        value = np.asarray(handle.convert("L")) > 127
    if (
        value.shape != (DECODER_HEIGHT, DECODER_WIDTH)
        or int(value.sum()) != int(record.get("pixels", -1))
    ):
        raise ValueError(f"{label} shape or pixel count differs")
    return torch.from_numpy(value.copy()).bool()


def _scene(inventory: Mapping[str, Any], scene_id: str) -> dict[str, Any]:
    scenes = inventory.get("scenes")
    if not isinstance(scenes, Mapping) or not isinstance(scenes.get(scene_id), Mapping):
        raise ValueError(f"source inventory lacks scene {scene_id!r}")
    return dict(scenes[scene_id])


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_sam3, source_sam3_sha, source_sam3_path = load_json_object(
        args.source_sam3_receipt,
        expected_sha256=args.source_sam3_receipt_sha256,
        label="source-view SAM3 receipt",
    )
    if (
        source_sam3.get("schema_version") != 1
        or source_sam3.get("artifact_type") != SOURCE_SAM3_ARTIFACT_TYPE
        or source_sam3.get("status")
        != "all_source_proposals_sealed_before_target_access"
        or source_sam3.get("method") != METHOD
        or source_sam3.get("method_contract") != method_contract()
        or source_sam3.get("view_count") != SOURCE_VIEW_COUNT
        or source_sam3.get("accepted_view_count") != SOURCE_VIEW_COUNT
        or source_sam3.get("source_access")
        != {
            "source_rgb_opened": True,
            "target_rgb_opened": False,
            "target_mask_opened": False,
            "target_metric_opened": False,
            "candidate_selected_with_gt": False,
        }
    ):
        raise ValueError("source-view SAM3 receipt violates the target-blind contract")
    validate_file_record(source_sam3.get("implementation"), label="source SAM3 bridge")

    coarse_record = source_sam3.get("coarse_receipt")
    if not isinstance(coarse_record, Mapping):
        raise ValueError("source-view SAM3 receipt lacks coarse binding")
    coarse, coarse_sha, coarse_path = load_json_object(
        coarse_record.get("path", ""),
        expected_sha256=coarse_record.get("sha256"),
        label="coarse source-view receipt",
    )
    selection = coarse.get("selection")
    if (
        coarse.get("artifact_type") != COARSE_ARTIFACT_TYPE
        or coarse.get("status")
        != "coarse_source_predictions_sealed_before_source_rgb_sam3_or_target_access"
        or coarse.get("method_contract") != method_contract()
        or coarse.get("scene_id") != source_sam3.get("scene_id")
        or not isinstance(selection, list)
        or len(selection) != SOURCE_VIEW_COUNT
        or coarse_sha != coarse_record.get("sha256")
    ):
        raise ValueError("coarse source-view authority differs")

    inventory_record = coarse.get("source_inventory")
    if not isinstance(inventory_record, Mapping):
        raise ValueError("coarse receipt lacks source inventory")
    inventory, inventory_sha, inventory_path = load_json_object(
        inventory_record.get("path", ""),
        expected_sha256=inventory_record.get("sha256"),
        label="source inventory",
    )
    scene_id = str(source_sam3["scene_id"])
    scene = _scene(inventory, scene_id)
    if (
        inventory.get("artifact_type")
        != "nvos_multiview_region_memory_source_inventory_v1"
        or inventory.get("method_contract") != method_contract()
        or inventory.get("global_safety", {}).get("target_rgb_content_opened_or_hashed")
        is not False
        or scene.get("safety", {}).get("target_rgb_content_opened_or_hashed")
        is not False
        or scene.get("safety", {}).get("target_mask_opened") is not False
        or scene.get("safety", {}).get("target_metric_opened") is not False
    ):
        raise ValueError("source inventory violates target isolation")

    responsibility_record = coarse.get("source_responsibility")
    if not isinstance(responsibility_record, Mapping):
        raise ValueError("coarse receipt lacks source responsibility")
    responsibility, responsibility_sha, responsibility_path = load_torch_mapping(
        responsibility_record.get("path", ""),
        expected_sha256=responsibility_record.get("sha256"),
        map_location="cpu",
        label="source responsibility",
    )
    metadata = responsibility.get("metadata")
    assignments = responsibility.get("assignments")
    feature_height, feature_width = map(int, scene["feature_grid_hw"])
    if (
        responsibility.get("schema_version") != 1
        or not isinstance(metadata, Mapping)
        or not isinstance(assignments, list)
        or len(assignments) != int(scene["source_view_count"])
        or metadata.get("assignment_mode") != "raster_gaussian_top1"
        or metadata.get("registration_weight_mode") != "alpha_depth"
        or [int(metadata.get("feature_height", -1)), int(metadata.get("feature_width", -1))]
        != [feature_height, feature_width]
        or metadata.get("benchmark_images_opened") is not False
        or metadata.get("benchmark_masks_opened") is not False
    ):
        raise ValueError("source responsibility contract differs")

    base, base_sha, base_path = load_torch_mapping(
        args.base_primitive_unary,
        expected_sha256=args.base_primitive_unary_sha256,
        map_location="cpu",
        label="frozen base primitive unary",
    )
    valid = base.get("valid")
    valid_rows = base.get("valid_rows")
    if (
        base.get("artifact_type") != "nvos_frozen_k16_primitive_unary_probability_v1"
        or base.get("scene_id") != scene_id
        or not torch.is_tensor(valid)
        or valid.dtype != torch.bool
        or valid.ndim != 1
        or not torch.is_tensor(valid_rows)
        or valid_rows.dtype != torch.long
        or not torch.equal(torch.where(valid)[0], valid_rows)
        or base.get("written_before_target_ground_truth_open") is not True
        or base.get("target_rgb_opened") is not False
        or base.get("target_mask_opened") is not False
    ):
        raise ValueError("frozen base primitive row authority differs")
    num_gaussians = int(valid.numel())

    view_receipt_records = source_sam3.get("view_receipts")
    if not isinstance(view_receipt_records, list) or len(view_receipt_records) != SOURCE_VIEW_COUNT:
        raise ValueError("source-view receipt list differs")
    selected_assignments: list[Mapping[str, torch.Tensor]] = []
    proposal_features: list[torch.Tensor] = []
    domain_features: list[torch.Tensor] = []
    reliabilities: list[float] = []
    view_rows: list[dict[str, Any]] = []
    for rank, (coarse_row, view_record) in enumerate(zip(selection, view_receipt_records)):
        if (
            coarse_row.get("selection_rank") != rank
            or view_record.get("selection_rank") != rank
            or coarse_row.get("frame_id") != view_record.get("frame_id")
        ):
            raise ValueError("selected source-view order changed")
        view_receipt, view_sha, view_path = load_json_object(
            view_record.get("path", ""),
            expected_sha256=view_record.get("sha256"),
            label=f"source SAM3 view {rank}",
        )
        access = view_receipt.get("source_access")
        report = view_receipt.get("sam3_report")
        if (
            view_receipt.get("artifact_type") != VIEW_ARTIFACT_TYPE
            or view_receipt.get("status") != "source_proposal_sealed_before_target_access"
            or view_receipt.get("scene_id") != scene_id
            or view_receipt.get("selection_rank") != rank
            or view_receipt.get("frame_id") != coarse_row.get("frame_id")
            or not isinstance(report, Mapping)
            or report.get("accepted") is not True
            or float(report.get("best_initial_overlap", 0.0)) < 0.05
            or access
            != {
                "source_rgb_opened": True,
                "target_rgb_opened": False,
                "target_mask_opened": False,
                "target_metric_opened": False,
                "candidate_selected_with_gt": False,
            }
        ):
            raise ValueError(f"source SAM3 view {rank} violates the contract")
        assignment_index = int(coarse_row["assignment_view_index"])
        if view_receipt.get("assignment_view_index") != assignment_index:
            raise ValueError("source-view assignment index changed")
        proposal_native = _load_mask(
            view_receipt.get("proposal_mask", {}), label=f"view {rank} proposal"
        )
        decoder_domain = normalized_box_observation_domain(
            report.get("box_prompt_cxcywh_norm", []),
            height=DECODER_HEIGHT,
            width=DECODER_WIDTH,
        )
        proposal_feature = sample_native_mask_at_feature_centers(
            proposal_native,
            feature_height=feature_height,
            feature_width=feature_width,
        )
        domain_feature = sample_native_mask_at_feature_centers(
            decoder_domain,
            feature_height=feature_height,
            feature_width=feature_width,
        )
        proposal_feature &= domain_feature
        if not bool(proposal_feature.any()):
            raise ValueError(f"accepted source proposal {rank} has no feature support")
        reliability = float(coarse_row["assignment_reliability"])
        if not 0 < reliability <= 1:
            raise ValueError("selected source-view reliability differs")
        selected_assignments.append(assignments[assignment_index])
        proposal_features.append(proposal_feature)
        domain_features.append(domain_feature)
        reliabilities.append(reliability)
        view_rows.append(
            {
                "selection_rank": rank,
                "assignment_view_index": assignment_index,
                "frame_id": str(coarse_row["frame_id"]),
                "view_receipt": {"path": str(view_path), "sha256": view_sha},
                "assignment_reliability": reliability,
                "best_initial_overlap": float(report["best_initial_overlap"]),
                "feature_proposal_pixels": int(proposal_feature.sum()),
                "feature_domain_pixels": int(domain_feature.sum()),
            }
        )

    memory = aggregate_proposal_membership(
        selected_assignments,
        proposal_features,
        domain_features,
        reliabilities,
        num_gaussians=num_gaussians,
    )
    positive_mass_by_view = torch.stack(
        [
            proposal_positive_mass(
                assignment,
                proposal,
                domain,
                num_gaussians=num_gaussians,
                view_reliability=reliability,
            )[valid_rows]
            for assignment, proposal, domain, reliability in zip(
                selected_assignments,
                proposal_features,
                domain_features,
                reliabilities,
            )
        ],
        dim=0,
    ).contiguous()
    probability = memory.probability[valid_rows].contiguous()
    confidence = memory.confidence[valid_rows].contiguous()
    observed = memory.observed[valid_rows].contiguous()
    if (
        not bool(observed.any())
        or not bool((positive_mass_by_view.sum(dim=1) > 0).all())
        or not torch.equal(observed, confidence > 0)
    ):
        raise ValueError("source proposals produced no valid primitive memory")

    tensors = {
        "valid_rows": valid_rows.cpu().contiguous(),
        "membership_probability": probability,
        "membership_confidence": confidence,
        "membership_observed": observed,
        "positive_mass_by_view": positive_mass_by_view,
        "proposal_masks_feature": torch.stack(proposal_features).contiguous(),
        "observation_domains_feature": torch.stack(domain_features).contiguous(),
        "view_reliability": torch.tensor(reliabilities, dtype=torch.float32),
    }
    tensor_hashes = {name: tensor_sha256(value) for name, value in tensors.items()}
    payload = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "status": "primitive_region_memory_sealed_before_target_access",
        "scene_id": scene_id,
        "method": METHOD,
        "method_contract": method_contract(),
        "num_gaussians": num_gaussians,
        "valid_primitive_rows": int(valid_rows.numel()),
        "view_count": SOURCE_VIEW_COUNT,
        "views": view_rows,
        "source_sam3_receipt": {
            "path": str(source_sam3_path),
            "sha256": source_sam3_sha,
        },
        "coarse_receipt": {"path": str(coarse_path), "sha256": coarse_sha},
        "source_inventory": {"path": str(inventory_path), "sha256": inventory_sha},
        "source_responsibility": {
            "path": str(responsibility_path),
            "sha256": responsibility_sha,
        },
        "base_primitive_unary": {"path": str(base_path), "sha256": base_sha},
        "capability_cache": {
            "path": str(base["capability_cache"]),
            "sha256": str(base["capability_cache_sha256"]),
        },
        "runtime_fusion_policy": {
            "write_domain": "base_registered_observation_confidence_exactly_zero_and_memory_confidence_positive",
            "observed_base_rows": "bitwise_preserved",
            "positive_negative_hard_anchors": "bitwise_preserved_as_subset_of_observed_base_rows",
            "region_tokens": "one_positive_mass_weighted_token_per_source_view_and_capability_bank",
            "field_prior_and_memory_reliability_separate": True,
        },
        "tensor_sha256": tensor_hashes,
        "source_access": {
            "source_rgb_opened_by_upstream_sam3": True,
            "target_rgb_opened": False,
            "target_mask_opened": False,
            "target_metric_opened": False,
            "candidate_selected_with_gt": False,
        },
        "implementation": file_record(Path(__file__).resolve()),
        **tensors,
    }
    output = Path(args.output).expanduser().absolute()
    write_torch_noclobber(output, payload)
    output_sha = sha256_file(output)
    receipt = {
        "schema_version": 1,
        "artifact_type": RECEIPT_ARTIFACT_TYPE,
        "status": "primitive_region_memory_receipt_sealed_before_target_access",
        "scene_id": scene_id,
        "artifact": {"path": str(output.resolve()), "sha256": output_sha},
        "source_sam3_receipt": {
            "path": str(source_sam3_path),
            "sha256": source_sam3_sha,
        },
        "tensor_sha256": tensor_hashes,
        "gate": {
            "accepted_source_views": SOURCE_VIEW_COUNT,
            "memory_observed_valid_rows": int(observed.sum()),
            "memory_positive_valid_rows": int((observed & (probability >= 0.5)).sum()),
            "memory_confidence_sum": float(confidence.sum()),
            "every_region_token_carrier_nonempty": bool(
                (positive_mass_by_view.sum(dim=1) > 0).all()
            ),
            "exact_runtime_anchor_bitwise_gate_pending": True,
        },
        "source_access": payload["source_access"],
        "implementation": payload["implementation"],
    }
    receipt_path = Path(args.output_receipt).expanduser().absolute()
    write_frozen_json(receipt_path, receipt)
    return {
        "artifact": str(output.resolve()),
        "artifact_sha256": output_sha,
        "receipt": str(receipt_path.resolve()),
        "receipt_sha256": sha256_file(receipt_path),
        "memory_observed_valid_rows": int(observed.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sam3-receipt", required=True)
    parser.add_argument("--source-sam3-receipt-sha256", required=True)
    parser.add_argument("--base-primitive-unary", required=True)
    parser.add_argument("--base-primitive-unary-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-receipt", required=True)
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
