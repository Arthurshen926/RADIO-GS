#!/usr/bin/env python3
"""Train SPIn-NeRF Lego all-view geometry with pinned original 3DGS.

The released Lego archive already includes Graphdeco-compatible COLMAP
undistortion.  The common launcher audits its raw-to-undistorted pose
equivalence and split-prefixed RGB/annotation mapping, then trains from 102
canonical per-attempt aliases without modifying the source dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from reproductions.ludvig.train_nvos_all_view_3dgs import (
    ROOT,
    AllViewTrainingSpec,
    NATIVE_SPIN_LEGO_PINHOLE_CONTRACT,
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
DEFAULT_OUTPUT_ROOT = DEFAULT_SPIN_RELEASED_ALL_VIEW_ROOT / "lego" / "training"

SPIN_LEGO_SPEC = AllViewTrainingSpec(
    benchmark="SPIn-NeRF",
    scene="lego",
    geometry_scene="lego",
    converted_source_relative=(
        Path("SPIn-NeRF")
        / "source_images"
        / "lego_real_night_radial"
        / "lego_real_night_radial"
    ),
    expected_registered_images=102,
    evaluation_render_resolution=(1015, 764),
    default_output_root=DEFAULT_OUTPUT_ROOT,
    source_asset_contract=NATIVE_SPIN_LEGO_PINHOLE_CONTRACT,
)


def launch(args: argparse.Namespace) -> Path:
    return launch_all_view_training(args, SPIN_LEGO_SPEC)


def parse_args() -> argparse.Namespace:
    parser = _add_common_training_arguments(argparse.ArgumentParser())
    return parser.parse_args()


if __name__ == "__main__":
    try:
        print(launch(parse_args()))
    except TrainingProtocolError as error:
        raise SystemExit(f"protocol error: {error}") from error
