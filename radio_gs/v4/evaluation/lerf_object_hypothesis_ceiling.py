"""Cold-load oracle extent evaluation for global object hypotheses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.data.lerf_dataset import _read_cameras_binary
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.v4.contracts.build_lerf_object_hypotheses import (
    validate_hypothesis_memory,
)
from radio_gs.v4.evaluation.lerf_common import load_labels as _load_labels
from radio_gs.v4.evaluation.lerf_object_ceiling import (
    _build_carrier,
    _evaluate_membership_ceiling_at_raster,
    _load_state,
)
from radio_gs.v4.geometry.fuse_lerf_moge3 import _read_images


REPORT_SCHEMA = "radio_gs.surface_object_memory_v4.lerf_object_hypothesis_ceiling.v1"


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.cpu_threads <= 0:
        raise ValueError("cpu thread count must be positive")
    torch.set_num_threads(int(args.cpu_threads))
    state_path = Path(args.scene_state)
    state = _load_state(state_path, expected_sha256=args.expected_scene_state_sha256)
    hypothesis_path = Path(args.hypothesis_memory).resolve(strict=True)
    if sha256_file(hypothesis_path) != args.expected_hypothesis_memory_sha256:
        raise ValueError("object-hypothesis memory SHA256 differs")
    hypothesis = validate_hypothesis_memory(
        torch.load(hypothesis_path, map_location="cpu", weights_only=False)
    )
    if hypothesis["scene_state_sha256"] != state["scene_state_sha256"]:
        raise ValueError("object-hypothesis memory and scene state are not bound")
    if hypothesis["scene_label"] != args.scene_label or state["scene_label"] != args.scene_label:
        raise ValueError("scene labels differ across object-hypothesis evaluation inputs")
    height, width = (int(value) for value in args.raster_shape)
    scene_root = Path(args.scene_root).resolve(strict=True)
    sparse = scene_root / "sparse" / "0"
    views = _read_images(
        sparse / "images.bin", _read_cameras_binary(sparse / "cameras.bin")
    )
    annotations, categories, source_height, source_width = _load_labels(
        Path(args.label_root).resolve(strict=True), args.scene_label
    )
    carrier = _build_carrier(state)
    variants = {
        assignment_mode: _evaluate_membership_ceiling_at_raster(
            carrier=carrier,
            membership=hypothesis["observed_membership"],
            views=views,
            annotations=annotations,
            categories=categories,
            source_height=source_height,
            source_width=source_width,
            height=height,
            width=width,
            pixel_threshold=float(args.pixel_threshold),
            assignment_mode=assignment_mode,
            fragmentation_capacity=3,
        )
        for assignment_mode in ("canonical_top2", "raw_independent_token")
    }
    report = {
        "schema": REPORT_SCHEMA,
        "development_only": True,
        "oracle_identity_used": True,
        "promotion_eligible": False,
        "scene_label": args.scene_label,
        "scene_state": str(state_path.resolve()),
        "scene_state_sha256": state["scene_state_sha256"],
        "hypothesis_memory": str(hypothesis_path),
        "hypothesis_memory_sha256": args.expected_hypothesis_memory_sha256,
        "cold_loaded_in_independent_process": True,
        "association_configuration": hypothesis["association_configuration"],
        "association_audit": hypothesis["association_audit"],
        "appearance_audit": hypothesis["appearance_audit"],
        "raster_shape": [height, width],
        "pixel_threshold": float(args.pixel_threshold),
        "variants": variants,
        "interpretation_contract": {
            "oracle_identity_is_not_deployment_text_result": True,
            "single_hypothesis_tests_global_object_organization": True,
            "top3_minus_single_tests_residual_fragmentation": True,
            "benchmark_labels_written_to_hypothesis_memory": False,
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-state", required=True)
    parser.add_argument("--expected-scene-state-sha256", required=True)
    parser.add_argument("--hypothesis-memory", required=True)
    parser.add_argument("--expected-hypothesis-memory-sha256", required=True)
    parser.add_argument("--scene-root", required=True)
    parser.add_argument("--scene-label", required=True)
    parser.add_argument("--label-root", required=True)
    parser.add_argument("--raster-shape", nargs=2, type=int, required=True, metavar=("H", "W"))
    parser.add_argument("--pixel-threshold", type=float, default=0.05)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({key: value["metrics"] for key, value in report["variants"].items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
