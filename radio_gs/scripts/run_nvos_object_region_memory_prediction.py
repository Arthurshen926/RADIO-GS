#!/usr/bin/env python3
"""Replay one frozen NVOS scene with a sealed object-region memory only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


PATH_ONLY_ARGS = {
    "output_dir",
    "prediction_receipt_output",
    "primitive_unary_output",
}
OBJECT_ARGS = {
    "object_multiview_region_memory",
    "object_region_memory",
    "object_region_memory_sha256",
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if _sha256(source) != expected_sha256:
        raise ValueError(f"JSON authority changed: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON authority must be an object")
    return value


def _argv(arguments: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    for name in sorted(arguments):
        value = arguments[name]
        flag = f"--{name.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                result.append(flag)
        elif value is not None and str(value) != "":
            result.extend([flag, str(value)])
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    base = _load_json(args.base_prediction_receipt, args.base_prediction_receipt_sha256)
    if (
        base.get("artifact_type") != "nvos_pre_metric_prediction_receipt_v1"
        or base.get("scene_id") != args.scene_id
        or base.get("sealed_before_target_ground_truth_open") is not True
        or base.get("target_rgb_opened") is not False
        or base.get("target_mask_opened") is not False
        or base.get("target_metric_opened") is not False
    ):
        raise ValueError("base prediction receipt differs")
    method = base.get("method_contract")
    if not isinstance(method, Mapping) or not isinstance(
        method.get("candidate_args"), Mapping
    ):
        raise ValueError("base receipt lacks candidate args")
    base_args = dict(method["candidate_args"])
    if (
        str(base_args.get("registered_readout_stage")) != "unary_prior"
        or base_args.get("disable_registered_graph") is not True
        or str(base_args.get("registered_observation_fusion"))
        != "probability_mixture"
        or str(base_args.get("registered_query_likelihood_calibration", "none"))
        != "none"
        or str(base_args.get("registered_forward_unary", "none")) != "none"
    ):
        raise ValueError("base is not the frozen hierarchical unary mainline")
    memory = Path(args.object_region_memory).expanduser().resolve()
    if _sha256(memory) != args.object_region_memory_sha256:
        raise ValueError("object region-memory asset changed")
    evaluator = Path(args.evaluator).expanduser().resolve()
    if _sha256(evaluator) != args.evaluator_sha256:
        raise ValueError("candidate evaluator changed")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"candidate output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    candidate = dict(base_args)
    candidate.update(
        {
            "scene_id": str(args.scene_id),
            "device": "cuda:0",
            "output_dir": str(output),
            "prediction_only": True,
            "prediction_receipt_output": str(
                output / "pre_metric_prediction_receipt.json"
            ),
            "primitive_unary_output": str(output / "primitive_unary.pt"),
            "source_only_correspondence_completion": "none",
            "source_correspondence_support_graph": "",
            "source_correspondence_support_graph_sha256": "",
            "source_multiview_responsibility_cache": "",
            "source_multiview_responsibility_cache_sha256": "",
            "object_multiview_region_memory": (
                "source_only_object_multiview_region_memory_v1"
            ),
            "object_region_memory": str(memory),
            "object_region_memory_sha256": args.object_region_memory_sha256,
        }
    )
    command = [sys.executable, str(evaluator), *_argv(candidate)]
    subprocess.run(command, check=True)
    receipt = output / "pre_metric_prediction_receipt.json"
    primitive = output / "primitive_unary.pt"
    if not receipt.is_file() or not primitive.is_file():
        raise RuntimeError("candidate did not seal prediction artifacts")
    sealed = _load_json(receipt, _sha256(receipt))
    if not (
        sealed.get("sealed_before_target_ground_truth_open") is True
        and sealed.get("target_rgb_opened") is False
        and sealed.get("target_mask_opened") is False
        and sealed.get("target_metric_opened") is False
    ):
        raise ValueError("candidate prediction safety barrier differs")
    return {
        "scene_id": args.scene_id,
        "receipt": str(receipt),
        "receipt_sha256": _sha256(receipt),
        "primitive_unary": str(primitive),
        "primitive_unary_sha256": _sha256(primitive),
        "output_dir": str(output),
        "target_rgb_mask_or_metric_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--base-prediction-receipt", required=True)
    parser.add_argument("--base-prediction-receipt-sha256", required=True)
    parser.add_argument("--object-region-memory", required=True)
    parser.add_argument("--object-region-memory-sha256", required=True)
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--evaluator-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    print(json.dumps(run(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
