#!/usr/bin/env python3
"""Train the SPIn-NeRF Truck all-view geometry with pinned original 3DGS.

The source is the preprocessed Truck scene released with Graphdeco's original
3DGS dataset.  Its 979x546 RGBs intentionally accompany centered 1957x1091
PINHOLE metadata.  The common launcher audits and stages that representation
without resampling images, rewriting intrinsics, or modifying source assets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from reproductions.ludvig.train_nvos_all_view_3dgs import (
    ROOT,
    AllViewTrainingSpec,
    NATIVE_SPIN_TRUCK_PINHOLE_CONTRACT,
    TrainingProtocolError,
    _add_common_training_arguments,
    launch_all_view_training,
)


DEFAULT_SPIN_RELEASED_ALL_VIEW_ROOT = (
    ROOT
    / "output"
    / "protocol_audit_20260731"
    / "ludvig"
    / "spin"
    / "released_all_view"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_SPIN_RELEASED_ALL_VIEW_ROOT / "truck" / "training"

SPIN_TRUCK_SPEC = AllViewTrainingSpec(
    benchmark="SPIn-NeRF",
    scene="truck",
    geometry_scene="truck",
    converted_source_relative=(
        Path("SPIn-NeRF")
        / "source_images"
        / "tandt"
        / "extracted"
        / "tandt"
        / "truck"
    ),
    expected_registered_images=251,
    evaluation_render_resolution=(979, 546),
    default_output_root=DEFAULT_OUTPUT_ROOT,
    source_asset_contract=NATIVE_SPIN_TRUCK_PINHOLE_CONTRACT,
)


def launch(args: argparse.Namespace) -> Path:
    return launch_all_view_training(args, SPIN_TRUCK_SPEC)


def parse_args() -> argparse.Namespace:
    parser = _add_common_training_arguments(argparse.ArgumentParser())
    return parser.parse_args()


if __name__ == "__main__":
    try:
        print(launch(parse_args()))
    except TrainingProtocolError as error:
        raise SystemExit(f"protocol error: {error}") from error
