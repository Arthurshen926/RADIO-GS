#!/usr/bin/env python3
"""Build an immutable source-only NVOS completion-abstention receipt."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import torch

from radio_gs.querying.nvos_source_completion_calibration import (
    PREREGISTRATION_RELATIVE_PATH,
    SCHEMA,
    canonical_sha256,
    compute_source_completion_loo_gate,
    file_sha256,
    source_completion_loo_method_contract,
)
from radio_gs.interfaces.prompt_responsibility_cache import tensor_sha256


def _atomic_json(payload: object, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, output)
        temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def build_gate(args: argparse.Namespace) -> dict[str, object]:
    completion = Path(args.source_completion).expanduser().resolve()
    completion_receipt = Path(args.source_completion_receipt).expanduser().resolve()
    if file_sha256(completion) != str(args.source_completion_sha256):
        raise ValueError("source completion SHA256 differs")
    if file_sha256(completion_receipt) != str(
        args.source_completion_receipt_sha256
    ):
        raise ValueError("source completion receipt SHA256 differs")
    upstream_receipt = json.loads(completion_receipt.read_text(encoding="utf-8"))
    payload = torch.load(completion, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("tensors"), dict):
        raise ValueError("source completion payload is malformed")
    authority = payload.get("authority")
    tensors = payload["tensors"]
    if (
        payload.get("artifact_type")
        != "radio_gs.nvos_sam3_reference_completion"
        or payload.get("schema_version") != 1
        or not isinstance(authority, dict)
        or authority.get("scene_id") != str(args.scene_id)
        or authority.get("frame_id") != str(args.prompt_frame_id)
        or authority.get("target_rgb_opened") is not False
        or authority.get("target_mask_opened") is not False
        or upstream_receipt.get("artifact_sha256")
        != str(args.source_completion_sha256)
        or upstream_receipt.get("target_rgb_opened") is not False
        or upstream_receipt.get("target_mask_opened") is not False
        or upstream_receipt.get("target_metric_opened") is not False
    ):
        raise ValueError("source completion authority differs")
    digests = {
        str(name): tensor_sha256(torch.as_tensor(value))
        for name, value in sorted(tensors.items())
    }
    if (
        payload.get("tensor_sha256") != digests
        or upstream_receipt.get("tensor_sha256") != digests
        or payload.get("tensor_bundle_sha256") != canonical_sha256(digests)
        or upstream_receipt.get("tensor_bundle_sha256")
        != payload.get("tensor_bundle_sha256")
    ):
        raise ValueError("source completion tensor authority differs")
    computed = compute_source_completion_loo_gate(tensors["trial_masks"])
    preregistration = (
        Path(__file__).resolve().parents[2] / PREREGISTRATION_RELATIVE_PATH
    ).resolve()
    contract = source_completion_loo_method_contract()
    return {
        "schema": SCHEMA,
        "scene_id": str(args.scene_id),
        "prompt_frame_id": str(args.prompt_frame_id),
        "preregistration": {
            "path": str(preregistration),
            "sha256": file_sha256(preregistration),
        },
        "source_completion": {
            "path": str(completion),
            "sha256": str(args.source_completion_sha256),
            "receipt_path": str(completion_receipt),
            "receipt_sha256": str(args.source_completion_receipt_sha256),
            "tensor_bundle_sha256": payload["tensor_bundle_sha256"],
            "trial_masks_tensor_sha256": digests["trial_masks"],
        },
        "method_contract": contract,
        "method_contract_sha256": canonical_sha256(contract),
        "source_only_metrics": {
            "per_trial": computed["per_trial"],
            "summary": computed["summary"],
        },
        "decision": {
            "accept_source_completion": computed["accept_source_completion"],
            "action": computed["action"],
        },
        "safety": {
            "computed_before_target_rendering": True,
            "target_rgb_opened": False,
            "target_mask_opened": False,
            "target_metric_opened": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--prompt-frame-id", required=True)
    parser.add_argument("--source-completion", required=True)
    parser.add_argument("--source-completion-sha256", required=True)
    parser.add_argument("--source-completion-receipt", required=True)
    parser.add_argument("--source-completion-receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_gate(args)
    output = args.output.expanduser().resolve()
    _atomic_json(payload, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": file_sha256(output),
                "decision": payload["decision"],
                "summary": payload["source_only_metrics"]["summary"],
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
