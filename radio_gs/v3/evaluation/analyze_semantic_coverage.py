"""Explain source-only D128 coverage without opening benchmark observations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import torch

from radio_gs.utils.immutable_artifacts import write_frozen_json
from radio_gs.v3.training.instance_upper_bound import (
    align_masks,
    proposal_supports,
    sha256_file,
    unpack_masks,
)


def _exclusive_coverage(
    *,
    visible: torch.Tensor,
    sam_supported: torch.Tensor,
    retained: torch.Tensor,
    mixed: torch.Tensor,
    semantic_valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return an exhaustive first-failure partition of Gaussian rows."""

    values = [
        torch.as_tensor(value).bool().reshape(-1)
        for value in (visible, sam_supported, retained, mixed, semantic_valid)
    ]
    if len({value.numel() for value in values}) != 1:
        raise ValueError("coverage row axes differ")
    visible, sam_supported, retained, mixed, semantic_valid = values
    eligible = visible & sam_supported & retained
    categories = {
        "source_invisible": ~visible,
        "no_sam_proposal": visible & ~sam_supported,
        "mpr_no_retained_support": visible & sam_supported & ~retained,
        "cross_boundary_mixed": eligible & mixed,
        "semantic_conflict_rejected": eligible & ~mixed & ~semantic_valid,
        "valid_clean_d128": eligible & ~mixed & semantic_valid,
    }
    assigned = torch.stack(tuple(categories.values())).sum(0)
    if not bool((assigned == 1).all()):
        raise RuntimeError("coverage partition is not mutually exclusive and exhaustive")
    return categories


def _intersect_sorted(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if not left.numel() or not right.numel():
        return left.new_empty(0)
    if left.numel() > right.numel():
        left, right = right, left
    positions = torch.searchsorted(right, left)
    valid = positions < right.numel()
    return left[valid & (right[positions.clamp_max(right.numel() - 1)] == left)]


def _explicitly_mixed_rows(
    membership: Mapping[str, object], authority: Mapping[str, object]
) -> torch.Tensor:
    rows = int(membership["num_rows"])
    proposals = int(membership["num_proposals"])
    supports = proposal_supports(
        torch.as_tensor(membership["row_indices"]),
        torch.as_tensor(membership["proposal_indices"]),
        torch.as_tensor(membership["weights"]),
        proposals,
    )
    unique_supports = tuple(torch.unique(value[0], sorted=True) for value in supports)
    views = torch.as_tensor(membership["proposal_view_indices"]).long()
    train = (views % 4 == 1) | (views % 4 == 2)
    relation = torch.as_tensor(authority["edge_relation"]).to(torch.int8)
    left = torch.as_tensor(authority["edge_left"]).long()
    right = torch.as_tensor(authority["edge_right"]).long()
    selected = (relation == 0) & train[left] & train[right]
    mixed = torch.zeros(rows, dtype=torch.bool)
    for a, b in zip(left[selected].tolist(), right[selected].tolist()):
        overlap = _intersect_sorted(unique_supports[a], unique_supports[b])
        mixed[overlap] = True
    return mixed


def _sam_supported_rows(membership: Mapping[str, object]) -> torch.Tensor:
    rows = int(membership["num_rows"])
    height = int(membership["metadata"]["feature_height"])
    width = int(membership["metadata"]["feature_width"])
    supported = torch.zeros(rows, dtype=torch.bool)
    for record in membership["metadata"]["source_records"]:
        if int(record["source_view_index"]) % 4 not in (1, 2):
            continue
        mask_payload = torch.load(record["mask_cache"], map_location="cpu")
        responsibility = torch.load(record["responsibility_view"], map_location="cpu")
        mask_height, mask_width = (
            int(value) for value in mask_payload["mask_shape"]
        )
        masks = align_masks(
            unpack_masks(mask_payload["packed_masks"], mask_width), height, width
        )
        if masks.shape[0] != int(record["num_proposals"]):
            raise ValueError("coverage SAM proposal axis differs")
        union = masks.any(0).flatten()
        pixels = torch.as_tensor(responsibility["pixel_ids"]).long()
        gaussian = torch.as_tensor(responsibility["gaussian_ids"]).long()
        if int(responsibility["num_pixels"]) != height * width:
            raise ValueError("coverage MPR raster axis differs")
        supported[gaussian[union[pixels]]] = True
    return supported


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--membership", required=True)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--semantic-memory", required=True)
    parser.add_argument("--cpu-threads", type=int, default=24)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.cpu_threads <= 0:
        raise ValueError("coverage CPU thread budget differs")
    torch.set_num_threads(int(args.cpu_threads))
    paths = {
        name: Path(value).resolve(strict=True)
        for name, value in (
            ("membership", args.membership),
            ("authority", args.authority),
            ("semantic_memory", args.semantic_memory),
        )
    }
    membership = torch.load(paths["membership"], map_location="cpu")
    authority = torch.load(paths["authority"], map_location="cpu")
    semantic = torch.load(paths["semantic_memory"], map_location="cpu")
    if (
        authority.get("schema") != "radio_gs.sugm_v3.native_language_authority.v3"
        or semantic.get("schema")
        != "radio_gs.sugm_v3.conflict_aware_semantic_memory.v1"
        or len({membership["scene"], authority["scene"], semantic["scene"]}) != 1
    ):
        raise ValueError("coverage input lineage differs")
    rows = int(membership["num_rows"])
    proposal_views = torch.as_tensor(membership["proposal_view_indices"]).long()
    proposal_train = (proposal_views % 4 == 1) | (proposal_views % 4 == 2)
    membership_rows = torch.as_tensor(membership["row_indices"]).long()
    membership_proposals = torch.as_tensor(membership["proposal_indices"]).long()
    retained = torch.zeros(rows, dtype=torch.bool)
    retained[membership_rows[proposal_train[membership_proposals]]] = True
    view_observed = torch.as_tensor(membership["view_observed"]).bool()
    train_views = torch.tensor(
        [
            int(record["source_view_index"])
            for record in membership["metadata"]["source_records"]
            if int(record["source_view_index"]) % 4 in (1, 2)
        ],
        dtype=torch.long,
    )
    visible = view_observed[train_views].any(0)
    sam_supported = _sam_supported_rows(membership)
    mixed = _explicitly_mixed_rows(membership, authority)
    semantic_valid = torch.as_tensor(semantic["semantic"]).float().norm(dim=1) > 0
    categories = _exclusive_coverage(
        visible=visible,
        sam_supported=sam_supported,
        retained=retained,
        mixed=mixed,
        semantic_valid=semantic_valid,
    )
    counts = {name: int(mask.sum()) for name, mask in categories.items()}
    payload = {
        "schema": "radio_gs.sugm_v3.semantic_coverage_analysis.v1",
        "scene": membership["scene"],
        "rows": rows,
        "partition_order": list(categories),
        "categories": {
            name: {"rows": count, "fraction": count / rows}
            for name, count in counts.items()
        },
        "diagnostics": {
            "source_visible_rows": int(visible.sum()),
            "sam_supported_rows": int(sam_supported.sum()),
            "retained_train_membership_rows": int(retained.sum()),
            "explicitly_mixed_rows": int(mixed.sum()),
            "semantic_valid_rows": int(semantic_valid.sum()),
            "semantic_valid_outside_eligible_rows": int(
                (semantic_valid & ~(visible & sam_supported & retained)).sum()
            ),
        },
        "definitions": {
            "source_invisible": "no exact-MPR observation in source train residues 1/2",
            "no_sam_proposal": "source-visible but no exact-MPR hit lands inside any source-train SAM proposal",
            "mpr_no_retained_support": "SAM-supported but absent from retained proposal-to-Gaussian memberships",
            "cross_boundary_mixed": "retained by both endpoints of an explicit different-object proposal edge",
            "semantic_conflict_rejected": "eligible non-mixed row rejected by robust D128 directional consensus",
            "valid_clean_d128": "eligible non-mixed row with a nonzero clean D128 write",
        },
        "source_split": {"train_residues": [1, 2]},
        "source_only": True,
        "target_rgb_opened": False,
        "benchmark_metrics_opened": False,
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }
    write_frozen_json(Path(args.output).resolve(), payload)
    print(payload)


if __name__ == "__main__":
    main()


__all__ = ["_exclusive_coverage", "_intersect_sorted"]
