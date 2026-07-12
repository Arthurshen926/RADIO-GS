#!/usr/bin/env python3
"""Build protocol-locked NVOS/SPIn-NeRF evaluation manifests.

SPIn-NeRF source RGB directories must be supplied explicitly with either a
JSON map or repeated ``--spin-rgb SCENE=PATH`` arguments.  This is intentional:
the ten scenes originate from several upstream datasets and guessing a camera
directory from a shared download root can silently misassociate annotations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from radio_gs.data.promptable_nvs_manifest import (
    ManifestError,
    SPIN_SCENES,
    SPIN_DIAGNOSTIC_SCENES,
    build_nvos_manifest,
    build_spin_manifest,
    load_scene_rgb_map,
    validate_manifest,
    write_manifest,
)


DEFAULT_DATA_ROOT = Path(
    "/mnt/pool/sqy/3d_understanding/segmentation_benchmarks"
)


def _scene_assignment(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected SCENE=PATH")
    scene, path = value.split("=", 1)
    scene = scene.strip().lower()
    path = path.strip()
    if not scene or not path:
        raise argparse.ArgumentTypeError("expected non-empty SCENE=PATH")
    if scene not in SPIN_SCENES:
        raise argparse.ArgumentTypeError(
            f"unknown SPIn-NeRF scene {scene!r}; expected one of {list(SPIN_SCENES)}"
        )
    return scene, path


def _merge_spin_rgb_map(
    map_path: str | None,
    assignments: Sequence[tuple[str, str]],
    *,
    allow_missing_fork_diagnostic: bool = False,
) -> dict[str, str]:
    merged = load_scene_rgb_map(map_path) if map_path else {}
    for scene, path in assignments:
        if scene in merged and Path(merged[scene]).expanduser() != Path(path).expanduser():
            raise ManifestError(
                f"Conflicting SPIn-NeRF RGB directories for {scene}: "
                f"{merged[scene]} versus {path}"
            )
        merged[scene] = path
    expected = SPIN_DIAGNOSTIC_SCENES if allow_missing_fork_diagnostic else SPIN_SCENES
    missing = sorted(set(expected) - set(merged))
    unknown = sorted(set(merged) - set(expected))
    if missing or unknown:
        raise ManifestError(
            "SPIn-NeRF requires an explicit RGB directory for every requested scene; "
            f"missing={missing}, unknown={unknown}. Provide --spin-rgb-map or "
            "the matching --spin-rgb SCENE=PATH arguments."
        )
    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        choices=("nvos", "spin", "all"),
        default="all",
        help="Manifest(s) to build (default: all).",
    )
    parser.add_argument(
        "--allow-incomplete-spin-diagnostic",
        action="store_true",
        help=(
            "Build only the nine available non-Fork scenes. The output is labelled "
            "diagnostic and is never eligible for a formal ten-scene result."
        ),
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--nvos-annotation-root", type=Path)
    parser.add_argument("--nvos-rgb-root", type=Path)
    parser.add_argument("--spin-annotation-root", type=Path)
    parser.add_argument(
        "--spin-rgb-map",
        help="JSON object mapping each canonical SPIn-NeRF scene id to an exact image directory.",
    )
    parser.add_argument(
        "--spin-rgb",
        action="append",
        default=[],
        type=_scene_assignment,
        metavar="SCENE=PATH",
        help="Explicit scene image directory; repeat for all ten scenes.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_root = args.data_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else data_root / "manifests"
    )

    built: list[tuple[str, dict, Path]] = []
    if args.benchmark in {"nvos", "all"}:
        annotation_root = args.nvos_annotation_root or (
            data_root / "NVOS" / "official_annotations" / "llff"
        )
        rgb_root = args.nvos_rgb_root or data_root / "NVOS" / "llff_undistorted"
        manifest = build_nvos_manifest(
            annotation_root,
            rgb_root,
            threshold=args.threshold,
        )
        validate_manifest(manifest, check_files=True)
        built.append(("NVOS", manifest, output_dir / "nvos_strict_unseen_v1.json"))

    if args.benchmark in {"spin", "all"}:
        annotation_root = args.spin_annotation_root or (
            data_root / "SPIn-NeRF" / "multiview_annotations"
        )
        scene_rgb_dirs = _merge_spin_rgb_map(
            args.spin_rgb_map,
            args.spin_rgb,
            allow_missing_fork_diagnostic=args.allow_incomplete_spin_diagnostic,
        )
        manifest = build_spin_manifest(
            annotation_root,
            scene_rgb_dirs,
            threshold=args.threshold,
            diagnostic_missing_fork=args.allow_incomplete_spin_diagnostic,
        )
        validate_manifest(manifest, check_files=True)
        built.append(
            (
                "SPIn-NeRF full-reference-mask diagnostic",
                manifest,
                output_dir
                / (
                    "spin_nerf_full_reference_mask_9scene_diagnostic_v1.json"
                    if args.allow_incomplete_spin_diagnostic
                    else "spin_nerf_full_reference_mask_10scene_v1.json"
                ),
            )
        )

    for _, manifest, path in built:
        write_manifest(manifest, path)
    for benchmark, manifest, path in built:
        print(
            json.dumps(
                {
                    "benchmark": benchmark,
                    "manifest": str(path),
                    "protocol_hash": manifest["protocol_hash"],
                    "scenes": len(manifest["scenes"]),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ManifestError as error:
        raise SystemExit(f"manifest build failed: {error}") from error
