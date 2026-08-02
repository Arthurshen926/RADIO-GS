#!/usr/bin/env python3
"""Independent scene/GPU authority entrypoint for forward-Beta v2."""

from __future__ import annotations

from typing import Sequence

from radio_gs.scripts.nvos_forward_beta_scene_authority import (
    V2_PROFILE,
    finalize_scene_receipt as _finalize_scene_receipt,
    main_for_profile,
    prepare_scene_command as _prepare_scene_command,
    validate_run_manifest as _validate_run_manifest,
    validate_scene_receipt as _validate_scene_receipt,
    write_scene_postcheck as _write_scene_postcheck,
)


def validate_run_manifest(path, *, scene):
    return _validate_run_manifest(path, scene=scene, profile=V2_PROFILE)


def prepare_scene_command(**kwargs):
    return _prepare_scene_command(**kwargs, profile=V2_PROFILE)


def write_scene_postcheck(**kwargs):
    return _write_scene_postcheck(**kwargs, profile=V2_PROFILE)


def finalize_scene_receipt(**kwargs):
    return _finalize_scene_receipt(**kwargs, profile=V2_PROFILE)


def validate_scene_receipt(path, *, run_manifest, scene, result):
    return _validate_scene_receipt(
        path,
        run_manifest=run_manifest,
        scene=scene,
        result=result,
        profile=V2_PROFILE,
    )


def main(argv: Sequence[str] | None = None) -> int:
    return main_for_profile(V2_PROFILE, argv)


if __name__ == "__main__":
    raise SystemExit(main())
