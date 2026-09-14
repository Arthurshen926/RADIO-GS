"""Oracle extent ceiling for the unmerged observed-fragment layer.

This diagnostic cold-loads a query-free fragment memory and asks whether a
fixed set of raw source fragments can cover each LERF category in held-out
annotated views.  Labels select fragments only inside this evaluator and are
never written back to either scene state or fragment memory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.data.lerf_dataset import _read_cameras_binary
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.v4.contracts.lerf_fragment_surface_memory import (
    SCHEMA as FRAGMENT_SCHEMA,
    validate_memory,
)
from radio_gs.v4.evaluation.lerf_common import load_labels as _load_labels
from radio_gs.v4.evaluation.lerf_object_ceiling import (
    _build_carrier,
    _evaluate_membership_ceiling_at_raster,
    _load_state,
)
from radio_gs.v4.geometry.fuse_lerf_moge3 import _read_images


REPORT_SCHEMA = "radio_gs.surface_object_memory_v4.lerf_fragment_ceiling.v1"


def _load_fragment_memory(
    path: Path,
    *,
    expected_sha256: str,
    scene_state_sha256: str,
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if sha256_file(resolved) != expected_sha256:
        raise ValueError("observed fragment memory SHA256 differs")
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    memory = validate_memory(payload)
    if memory.get("schema") != FRAGMENT_SCHEMA:
        raise ValueError("observed fragment memory schema differs")
    if memory.get("scene_state_sha256") != scene_state_sha256:
        raise ValueError("observed fragment memory and scene state are not bound")
    memory["memory_sha256"] = expected_sha256
    return memory


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.cpu_threads <= 0:
        raise ValueError("cpu thread count must be positive")
    torch.set_num_threads(int(args.cpu_threads))
    state_path = Path(args.scene_state)
    state = _load_state(state_path, expected_sha256=args.expected_scene_state_sha256)
    memory_path = Path(args.fragment_memory)
    memory = _load_fragment_memory(
        memory_path,
        expected_sha256=args.expected_fragment_memory_sha256,
        scene_state_sha256=state["scene_state_sha256"],
    )
    if state["scene_label"] != args.scene_label or memory["scene_label"] != args.scene_label:
        raise ValueError("scene labels differ across fragment ceiling inputs")
    height, width = (int(value) for value in args.raster_shape)
    if [height, width] != list(memory["raster_shape"]):
        raise ValueError("fragment ceiling raster differs from fragment construction")
    carrier = _build_carrier(state)
    if memory["fragment_positive"].shape[1] != carrier.num_elements:
        raise ValueError("fragment memory and surface carrier element axes differ")
    scene_root = Path(args.scene_root).resolve(strict=True)
    sparse = scene_root / "sparse" / "0"
    views = _read_images(
        sparse / "images.bin", _read_cameras_binary(sparse / "cameras.bin")
    )
    annotations, categories, source_height, source_width = _load_labels(
        Path(args.label_root).resolve(strict=True), args.scene_label
    )
    ceiling = _evaluate_membership_ceiling_at_raster(
        carrier=carrier,
        membership=memory["fragment_positive"].T.float(),
        views=views,
        annotations=annotations,
        categories=categories,
        source_height=source_height,
        source_width=source_width,
        height=height,
        width=width,
        pixel_threshold=float(args.pixel_threshold),
        assignment_mode="raw_independent_token",
        fragmentation_capacity=int(args.maximum_fragments),
    )
    report = {
        "schema": REPORT_SCHEMA,
        "development_only": True,
        "oracle_identity_used": True,
        "promotion_eligible": False,
        "scene_label": args.scene_label,
        "scene_state": str(state_path.resolve()),
        "scene_state_sha256": state["scene_state_sha256"],
        "fragment_memory": str(memory_path.resolve()),
        "fragment_memory_sha256": memory["memory_sha256"],
        "cold_loaded_in_independent_process": True,
        "fragment_count": int(memory["fragment_positive"].shape[0]),
        "source_view_count": len(memory["source_frames"]),
        "raster_shape": [height, width],
        "maximum_fragments": int(args.maximum_fragments),
        "ceiling": ceiling,
        "interpretation_contract": {
            "raw_fragments_are_not_deployment_object_hypotheses": True,
            "single_fragment_measures_best_complete_source_proposal": True,
            "multi_fragment_measures_association_recoverable_extent": True,
            "benchmark_labels_written_to_fragment_memory": False,
            "text_query_used": False,
            "completion_used": False,
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
    parser.add_argument("--fragment-memory", required=True)
    parser.add_argument("--expected-fragment-memory-sha256", required=True)
    parser.add_argument("--scene-root", required=True)
    parser.add_argument("--scene-label", required=True)
    parser.add_argument("--label-root", required=True)
    parser.add_argument("--raster-shape", nargs=2, type=int, required=True, metavar=("H", "W"))
    parser.add_argument("--pixel-threshold", type=float, default=0.2)
    parser.add_argument("--maximum-fragments", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report["ceiling"]["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
