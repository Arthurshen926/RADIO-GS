#!/usr/bin/env python3
"""Materialize label-free opacity-volume weights for ScanNet source objectives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from radio_gs.config import load_config
from radio_gs.scripts.eval_scannet_pointcloud_radio_gs import _build_hybrid_model
from radio_gs.scripts.train_scannet_frozen_l512_native_categorical_score_decoder import _load
from radio_gs.utils.immutable_artifacts import file_record, write_torch_noclobber


def build(args: argparse.Namespace) -> dict:
    baseline, baseline_record = _load(
        args.baseline_query_cache, args.expected_baseline_query_cache_sha256,
        "row-domain authority",
    )
    device = torch.device(args.device)
    config_path = Path(args.config).expanduser().resolve(strict=True)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve(strict=True)
    model, _codec = _build_hybrid_model(load_config(config_path), checkpoint_path, device)
    xyz = model.get_xyz().detach().float().cpu().contiguous()
    reference_xyz = torch.as_tensor(baseline["xyz"]).float().contiguous()
    if xyz.shape != reference_xyz.shape or not torch.equal(xyz, reference_xyz):
        raise ValueError("metric-weight Gaussian row domain differs")
    significance = (
        model.get_scaling().detach().float().prod(1)
        * model.get_opacity().detach().float().reshape(-1)
    ).cpu().contiguous()
    if not bool(torch.isfinite(significance).all()) or bool((significance <= 0).any()):
        raise ValueError("metric weights must be finite and positive")
    output = Path(args.output).expanduser().resolve()
    write_torch_noclobber(output, {
        "schema": "radio_gs.scannet_query_independent_metric_weights.v1",
        "schema_version": 1,
        "scene": args.scene,
        "xyz": xyz,
        "significance": significance,
        "metadata": {
            "query_independent": True,
            "source_only": True,
            "benchmark_labels_opened": False,
            "benchmark_masks_opened": False,
            "evaluation_rgb_opened": False,
            "definition": "activated_opacity_times_activated_scale_product",
            "config": file_record(config_path),
            "checkpoint": file_record(checkpoint_path),
            "row_domain": baseline_record,
        },
    })
    return {"status": "complete", "scene": args.scene, "output": file_record(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline-query-cache", required=True)
    parser.add_argument("--expected-baseline-query-cache-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    print(json.dumps(build(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
