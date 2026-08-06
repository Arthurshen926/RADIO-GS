#!/usr/bin/env python3
"""Continue a sealed v1 clean-ScanNet pilot with corrected exact MPR flags.

The first five v1 stage receipts are immutable and bind the byte identity of
``run_full_scalar_scannet_clean_pilot.py``.  This versioned continuation keeps
that implementation untouched, validates the complete v1 prefix, and binds
both the terminal v1 prefix receipt and this v2 implementation into every new
stage input.  It changes one command detail only: exact marginal MPR builders
receive the frozen ``--alpha-threshold 0`` required by their own contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from radio_gs.scripts import run_full_scalar_scannet_clean_pilot as v1
from radio_gs.utils.immutable_artifacts import load_json_object


V1_PREFIX_STAGES = v1.STAGES[:5]
CONTINUATION_STAGES = v1.STAGES[5:]
_V1_MPR_COMMAND = v1._mpr_command
_V1_COMMON_INPUTS = v1._common_inputs


def _mpr_command(**kwargs: Any) -> list[str]:
    command = _V1_MPR_COMMAND(**kwargs)
    if "--alpha-threshold" in command:
        raise ValueError("v1 MPR command unexpectedly declares alpha threshold")
    insertion = command.index("--aggregation-mode")
    command[insertion:insertion] = ["--alpha-threshold", "0"]
    return command


def validate_v1_prefix(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Path]:
    """Fail closed unless receipts 00--04 form one intact v1 authority chain."""

    common = v1.validate_pilot_authorities(args)
    receipt_root = Path(args.pilot_root).expanduser().resolve() / "receipts"
    previous: Path | None = None
    for index, stage in enumerate(V1_PREFIX_STAGES):
        path = receipt_root / f"{index:02d}_{stage}.json"
        if not path.is_file():
            raise ValueError(f"v1 continuation prefix receipt is missing: {path}")
        value, _digest, _source = load_json_object(
            path,
            expected_sha256=(
                args.expected_v1_prefix_receipt_sha256
                if index == len(V1_PREFIX_STAGES) - 1
                else None
            ),
            label=f"v1 {stage} prefix receipt",
        )
        receipt = v1._validate_stage_receipt(value, stage=stage, common=common)
        declared_previous = receipt["inputs"].get("previous_stage_receipt")
        if previous is None:
            if declared_previous is not None:
                raise ValueError("first v1 prefix receipt declares a predecessor")
        elif declared_previous != v1._file_record(previous):
            raise ValueError("v1 continuation prefix chain differs")
        previous = path
    assert previous is not None
    return common, previous


def continuation_inputs(
    common: Mapping[str, Any],
    previous_receipt: Path | None,
    *,
    v1_prefix_receipt: Path,
) -> dict[str, Any]:
    inputs = _V1_COMMON_INPUTS(common, previous_receipt)
    inputs.update(
        {
            "v1_prefix_receipt": v1._file_record(v1_prefix_receipt),
            "v2_continuation_launcher": v1._file_record(Path(__file__).resolve()),
        }
    )
    return inputs


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.stop_after not in CONTINUATION_STAGES:
        raise ValueError("v2 launcher only runs stages after render_contract")
    common, prefix = validate_v1_prefix(args)

    def bound_common_inputs(
        observed_common: Mapping[str, Any], previous_receipt: Path | None
    ) -> dict[str, Any]:
        if observed_common != common:
            raise ValueError("v2 continuation authority changed during execution")
        return continuation_inputs(
            observed_common,
            previous_receipt,
            v1_prefix_receipt=prefix,
        )

    original_mpr = v1._mpr_command
    original_inputs = v1._common_inputs
    v1._mpr_command = _mpr_command
    v1._common_inputs = bound_common_inputs
    try:
        return v1.run(args)
    finally:
        v1._mpr_command = original_mpr
        v1._common_inputs = original_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--scan-archive-part", required=True)
    parser.add_argument("--cohort-authority", required=True)
    parser.add_argument("--expected-cohort-authority-sha256", required=True)
    parser.add_argument("--exclusion-manifest", required=True)
    parser.add_argument("--expected-exclusion-manifest-sha256", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--expected-inventory-sha256", required=True)
    parser.add_argument("--expected-v1-prefix-receipt-sha256", required=True)
    parser.add_argument("--pilot-root", required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument(
        "--stop-after", choices=CONTINUATION_STAGES, default=CONTINUATION_STAGES[-1]
    )
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
