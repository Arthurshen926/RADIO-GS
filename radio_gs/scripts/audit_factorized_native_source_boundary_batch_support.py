#!/usr/bin/env python3
"""Audit source-train class support for direct boundary alignment batches."""

from __future__ import annotations

import argparse
import json

import torch
from torch.nn import functional as F

from radio_gs.interfaces import (
    factorized_native_source_global_margin_calibration as formal,
)
from radio_gs.losses import factorized_native_source_boundary_alignment as boundary
from radio_gs.losses import source_global_response_listwise_loss_v21 as relevance
from radio_gs.scripts import (
    calibrate_factorized_native_contrast_v21_global_margin as calibration,
)
from radio_gs.scripts import (
    train_factorized_native_gauge_state_readout_exact4x2 as legacy,
)
from radio_gs.scripts import train_surface_region_typed_context_residual as sparse_teacher


def audit(args: argparse.Namespace) -> dict[str, object]:
    authority = formal.validate_execution_authority(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
    )
    source = calibration._prepared_source(authority)
    fit = calibration._load_fit_text_bank(authority["fit_text_bank"])
    negatives = relevance.load_frozen_canonical_negative_bank(
        authority["canonical_negative_bank"]["path"],
        expected_file_sha256=authority["canonical_negative_bank"]["sha256"],
    )
    device = torch.device(str(args.device))
    positive_text = F.normalize(fit.embeddings.float(), dim=-1).to(device)
    negative_text = F.normalize(negatives.embeddings.float(), dim=-1).to(device)
    per_scene: dict[str, object] = {}
    for binding in source.train:
        scene = legacy.load_scene(binding)
        rows = legacy._scene_rows(scene)
        if rows.numel() != formal.REGIONS_PER_SCENE:
            raise ValueError("boundary batch audit requires all canonical source rows")
        counts: list[int] = []
        batches: list[torch.Tensor] = []
        for step in range(1, 65):
            selected = legacy._cyclic_batch(rows, step=step)
            batches.append(selected)
            teacher, mask = sparse_teacher.gather_sparse_teacher_batch(
                scene.shard, selected
            )
            probability = boundary.exact_multiview_teacher_probability(
                teacher.to(device),
                mask.to(device),
                positive_text,
                negative_text,
            )
            counts.append(int((probability >= 0.5).sum()))
        covered = torch.cat(batches).unique().numel()
        pair_count = legacy.BATCH_ROWS * formal.FIT_QUERY_ROWS
        per_scene[binding.scene_id] = {
            "batches": len(counts),
            "covered_unique_rows": int(covered),
            "minimum_positive_pairs": min(counts),
            "maximum_positive_pairs": max(counts),
            "mean_positive_pairs": sum(counts) / len(counts),
            "empty_positive_batches": sum(value == 0 for value in counts),
            "empty_negative_batches": sum(value == pair_count for value in counts),
        }
        print(
            json.dumps(
                {binding.scene_id: per_scene[binding.scene_id]}, sort_keys=True
            ),
            flush=True,
        )
    all_supported = all(
        row["empty_positive_batches"] == 0
        and row["empty_negative_batches"] == 0
        and row["covered_unique_rows"] == formal.REGIONS_PER_SCENE
        for row in per_scene.values()
    )
    return {
        "schema": "radio_gs.factorized_native_source_boundary_batch_support.v1",
        "status": "source_batches_supported" if all_supported else "source_batches_unsupported",
        "training_steps": 64,
        "batch_rows": legacy.BATCH_ROWS,
        "fit_query_rows": formal.FIT_QUERY_ROWS,
        "per_scene": per_scene,
        "all_batches_have_both_boundary_classes": all_supported,
        "target_or_benchmark_opened": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-authority", required=True)
    parser.add_argument("--expected-execution-authority-sha256", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    print(json.dumps(audit(build_parser().parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
