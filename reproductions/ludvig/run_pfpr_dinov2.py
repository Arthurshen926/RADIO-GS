#!/usr/bin/env python3
"""Single fail-closed wrapper for the staged LUDVIG-to-PFPR adaptation.

``phase-a`` binds public inputs and stages cameras without importing PyTorch;
``dino-pca`` executes the separately audited exact-LUDVIG vendored DINO/PCA
phase.  Later phase names remain explicit protocol errors instead of silently
returning an intermediate artifact as an evaluation result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_a import (
    EXPECTED_FIELD_FRAME_COUNT,
    EXPECTED_GAUSSIAN_COUNT,
    FROZEN_METHOD_MANIFEST_SHA256,
    FROZEN_PUBLIC_MANIFEST_SHA256,
    FROZEN_SOURCE_ADAPTER_LEDGER_FILENAME_BY_SCENE,
    FROZEN_SOURCE_ADAPTER_LEDGER_SHA256_BY_SCENE,
    LUDVIG_AUDITED_COMMIT,
    LudvigPFPRPhaseAError,
    OFFICIAL_DINO_CHECKPOINT_SHA256,
    OFFICIAL_DINO_CHECKPOINT_SIZE,
    PhaseAConfig,
    reject_unimplemented_phase,
    run_phase_a,
)
from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_b import (
    PhaseBConfig,
    run_phase_b,
)


PHASES = (
    "phase-a",
    "dino-pca",
    "uplift",
    "score",
    "evaluate",
    "all",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--phase", choices=PHASES, default="phase-a")
    result.add_argument("--scene", required=True)
    result.add_argument("--benchmark-dir", type=Path, required=True)
    result.add_argument("--source-scene", type=Path, required=True)
    result.add_argument("--field-contract-sha256", required=True)
    result.add_argument(
        "--source-adapter-ledger",
        type=Path,
        help="Optional relocation of the repo-frozen, hash-bound scene ledger",
    )
    result.add_argument("--geometry-ply", type=Path, required=True)
    result.add_argument("--geometry-sha256", required=True)
    result.add_argument("--dino-checkpoint", type=Path, required=True)
    result.add_argument(
        "--ludvig-upstream", type=Path, default=Path("/root/baselines/LUDVIG")
    )
    result.add_argument(
        "--phase-a-dir",
        type=Path,
        help="Hash-bound completed Phase-A attempt required by dino-pca",
    )
    result.add_argument(
        "--phase-a-manifest-sha256",
        help="Exact run_manifest.json SHA-256 required by dino-pca",
    )
    result.add_argument(
        "--dinov2-source",
        type=Path,
        default=Path("/root/baselines/LUDVIG"),
        help="Pinned LUDVIG checkout containing its vendored DINOv2 source",
    )
    result.add_argument(
        "--driver-library-dir",
        type=Path,
        default=Path("/root/baselines/LUDVIG/.driver535"),
        help="Process-local libcuda directory matching the loaded kernel driver",
    )
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--views", type=int, default=120)
    return result


def run(args: argparse.Namespace, *, argv=None):
    reject_unimplemented_phase(args.phase)
    try:
        ledger_sha256 = FROZEN_SOURCE_ADAPTER_LEDGER_SHA256_BY_SCENE[args.scene]
        ledger_filename = FROZEN_SOURCE_ADAPTER_LEDGER_FILENAME_BY_SCENE[args.scene]
    except KeyError as error:
        raise LudvigPFPRPhaseAError(
            f"No frozen PFPR/LUDVIG source adapter ledger for {args.scene!r}"
        ) from error
    ledger_path = args.source_adapter_ledger or (
        Path(__file__).resolve().parent / "receipts" / ledger_filename
    )
    if args.phase == "dino-pca":
        if args.phase_a_dir is None or not args.phase_a_manifest_sha256:
            raise LudvigPFPRPhaseAError(
                "dino-pca requires --phase-a-dir and "
                "--phase-a-manifest-sha256"
            )
        config = PhaseBConfig(
            phase_a_dir=args.phase_a_dir,
            expected_phase_a_manifest_sha256=args.phase_a_manifest_sha256,
            dino_checkpoint=args.dino_checkpoint,
            ludvig_upstream=args.ludvig_upstream,
            source_adapter_ledger=ledger_path,
            dinov2_source=args.dinov2_source,
            output_dir=args.output_dir,
            driver_library_dir=args.driver_library_dir,
            device=args.device,
            view_count=args.views,
            expected_source_adapter_ledger_sha256=ledger_sha256,
        )
        return run_phase_b(config, argv=argv)
    config = PhaseAConfig(
        scene_id=args.scene,
        benchmark_dir=args.benchmark_dir,
        source_scene=args.source_scene,
        field_contract_sha256=args.field_contract_sha256,
        source_adapter_ledger=ledger_path,
        expected_source_adapter_ledger_sha256=ledger_sha256,
        geometry_ply=args.geometry_ply,
        geometry_sha256=args.geometry_sha256,
        dino_checkpoint=args.dino_checkpoint,
        ludvig_upstream=args.ludvig_upstream,
        output_dir=args.output_dir,
        view_count=args.views,
        expected_field_frame_count=EXPECTED_FIELD_FRAME_COUNT,
        expected_gaussian_count=EXPECTED_GAUSSIAN_COUNT,
        expected_method_manifest_sha256=FROZEN_METHOD_MANIFEST_SHA256,
        expected_public_manifest_sha256=FROZEN_PUBLIC_MANIFEST_SHA256,
        expected_checkpoint_size=OFFICIAL_DINO_CHECKPOINT_SIZE,
        expected_checkpoint_sha256=OFFICIAL_DINO_CHECKPOINT_SHA256,
        expected_ludvig_commit=LUDVIG_AUDITED_COMMIT,
    )
    return run_phase_a(config, argv=argv)


def main() -> None:
    args = parser().parse_args()
    payload = run(args, argv=sys.argv)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
