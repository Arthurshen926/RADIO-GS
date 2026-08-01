#!/usr/bin/env python3
"""Train SPIn-NeRF LLFF ``room`` all-view geometry with pinned original 3DGS.

The SPIn raw room scene is checked byte-for-byte against the raw source of the
existing NVOS PINHOLE conversion on every launch.  The conversion is reused
only after that identity proof succeeds; no SPIn or NVOS source asset is
modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from reproductions.ludvig.train_nvos_all_view_3dgs import (
    ROOT,
    AllViewTrainingSpec,
    TrainingProtocolError,
    VERIFIED_SPIN_NVOS_PINHOLE_REUSE_CONTRACT,
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
DEFAULT_OUTPUT_ROOT = DEFAULT_SPIN_RELEASED_ALL_VIEW_ROOT / "room" / "training"

SPIN_LLFF_ROOM_SPEC = AllViewTrainingSpec(
    benchmark="SPIn-NeRF",
    scene="room",
    geometry_scene="room",
    converted_source_relative=(
        Path("NVOS") / "llff_undistorted" / "room_undistort"
    ),
    raw_identity_source_relative=(
        Path("SPIn-NeRF")
        / "source_images"
        / "llff_google_drive"
        / "extracted"
        / "nerf_llff_data"
        / "room"
    ),
    expected_registered_images=41,
    evaluation_render_resolution=(1600, 1200),
    default_output_root=DEFAULT_OUTPUT_ROOT,
    source_asset_contract=VERIFIED_SPIN_NVOS_PINHOLE_REUSE_CONTRACT,
)


def launch(args: argparse.Namespace) -> Path:
    return launch_all_view_training(args, SPIN_LLFF_ROOM_SPEC)


def parse_args() -> argparse.Namespace:
    parser = _add_common_training_arguments(argparse.ArgumentParser())
    return parser.parse_args()


if __name__ == "__main__":
    try:
        print(launch(parse_args()))
    except TrainingProtocolError as error:
        raise SystemExit(f"protocol error: {error}") from error
