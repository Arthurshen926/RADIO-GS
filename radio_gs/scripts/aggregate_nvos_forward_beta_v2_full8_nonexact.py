#!/usr/bin/env python3
"""Aggregate only the independent NVOS forward-Beta v2 full-8 run.

The implementation reuses the score/report checks of the frozen v1
aggregator, but supplies a closed v2 profile.  Consequently a v1 candidate,
forward mode, scene receipt, result report, or unbound reliability sidecar is
rejected rather than silently treated as a v2 result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from radio_gs.scripts.aggregate_nvos_forward_beta_full8_nonexact import (
    aggregate_forward_beta_full8,
    build_arg_parser,
)
from radio_gs.scripts.nvos_forward_beta_v2_scene_authority import (
    validate_scene_receipt,
)
from radio_gs.utils.immutable_artifacts import write_frozen_json


V2_CANDIDATE_ID = "nvos-forward-beta-balanced-residual-v2"
V2_FORWARD_MODE = "beta_balanced_residual_v2"
V2_ARTIFACT_TYPE = (
    "nvos_forward_beta_v2_full8_authority_bound_non_exact_diagnostic"
)


def aggregate_forward_beta_v2_full8(
    *,
    run_manifest_path: str | Path,
    result_root: str | Path,
    receipt_root: str | Path,
) -> dict[str, Any]:
    """Aggregate a complete v2 cohort under the dedicated v2 authority."""

    return aggregate_forward_beta_full8(
        run_manifest_path=run_manifest_path,
        result_root=result_root,
        receipt_root=receipt_root,
        expected_candidate=V2_CANDIDATE_ID,
        expected_forward_mode=V2_FORWARD_MODE,
        require_reliability_bindings=True,
        receipt_validator=validate_scene_receipt,
        artifact_type=V2_ARTIFACT_TYPE,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(
            f"immutable v2 aggregation output already exists: {args.output}"
        )
    summary = aggregate_forward_beta_v2_full8(
        run_manifest_path=args.run_manifest,
        result_root=args.result_root,
        receipt_root=args.receipt_root,
    )
    write_frozen_json(args.output, summary)
    print(json.dumps(summary, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
