#!/usr/bin/env python3
"""Train SPIn-NeRF Pinecone all-view geometry with pinned original 3DGS."""

from __future__ import annotations

import argparse
from pathlib import Path

from reproductions.ludvig.train_nvos_all_view_3dgs import (
    ROOT,
    AllViewTrainingSpec,
    NATIVE_SPIN_PINECONE_PINHOLE_CONTRACT,
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
DEFAULT_OUTPUT_ROOT = DEFAULT_SPIN_RELEASED_ALL_VIEW_ROOT / "pinecone" / "training"

SPIN_PINECONE_SPEC = AllViewTrainingSpec(
    benchmark="SPIn-NeRF",
    scene="pinecone",
    geometry_scene="pinecone",
    converted_source_relative=(
        Path("SPIn-NeRF")
        / "protocol_derived"
        / "pinecone_colmap_3p6_undistorted_v2"
    ),
    expected_registered_images=99,
    evaluation_render_resolution=(1600, 1199),
    default_output_root=DEFAULT_OUTPUT_ROOT,
    source_asset_contract=NATIVE_SPIN_PINECONE_PINHOLE_CONTRACT,
    raw_identity_source_relative=(
        Path("SPIn-NeRF")
        / "source_images"
        / "nerf_real_360"
        / "extracted"
        / "pinecone"
    ),
)


def launch(args: argparse.Namespace) -> Path:
    return launch_all_view_training(args, SPIN_PINECONE_SPEC)


def parse_args() -> argparse.Namespace:
    parser = _add_common_training_arguments(argparse.ArgumentParser())
    return parser.parse_args()


if __name__ == "__main__":
    try:
        print(launch(parse_args()))
    except TrainingProtocolError as error:
        raise SystemExit(f"protocol error: {error}") from error
