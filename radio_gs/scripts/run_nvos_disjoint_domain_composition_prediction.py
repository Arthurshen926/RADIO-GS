#!/usr/bin/env python3
"""Replay one NVOS scene with two sealed inputs on disjoint primitive domains."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import torch

from radio_gs.querying.disjoint_domain_composition import MODE
from radio_gs.querying.multiview_region_memory_runtime import RUNTIME_MODE


LIKELIHOOD_MODE = "balanced_reference_source_reconstruction_v1"
PATH_ONLY_ARGS = {"output_dir", "prediction_receipt_output", "primitive_unary_output"}
FACTOR_ARGS = {
    "registered_query_likelihood_calibration",
    "registered_disjoint_domain_composition",
    "object_multiview_region_memory",
    "object_region_memory",
    "object_region_memory_sha256",
    "source_only_correspondence_completion",
    "source_correspondence_support_graph",
    "source_correspondence_support_graph_sha256",
    "source_multiview_responsibility_cache",
    "source_multiview_responsibility_cache_sha256",
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


def _load_primitive(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if _sha256(source) != expected_sha256:
        raise ValueError(f"primitive authority changed: {source}")
    value = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise ValueError("primitive authority must be a mapping")
    return dict(value)


def _normalized(arguments: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(arguments)
    for name in PATH_ONLY_ARGS | FACTOR_ARGS:
        result.pop(name, None)
    return result


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


def _safe_receipt(value: Mapping[str, Any], scene_id: str) -> bool:
    return bool(
        value.get("scene_id") == scene_id
        and value.get("sealed_before_target_ground_truth_open") is True
        and value.get("target_rgb_opened") is False
        and value.get("target_mask_opened") is False
        and value.get("target_metric_opened") is False
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    prereg = _load_json(args.preregistration, args.preregistration_sha256)
    if (
        prereg.get("artifact_type")
        != "nvos_disjoint_domain_composition_fern_trex_preregistration_v1"
        or prereg.get("frozen_implementation", {}).get("runner", {}).get("sha256")
        != _sha256(Path(__file__).resolve())
    ):
        raise ValueError("disjoint-domain preregistration differs")
    base = _load_json(args.base_prediction_receipt, args.base_prediction_receipt_sha256)
    likelihood = _load_json(
        args.likelihood_prediction_receipt,
        args.likelihood_prediction_receipt_sha256,
    )
    memory_receipt = _load_json(
        args.memory_prediction_receipt,
        args.memory_prediction_receipt_sha256,
    )
    if not all(
        _safe_receipt(value, args.scene_id)
        for value in (base, likelihood, memory_receipt)
    ):
        raise ValueError("sealed input prediction safety barrier differs")
    base_args = dict(base["method_contract"]["candidate_args"])
    likelihood_args = dict(likelihood["method_contract"]["candidate_args"])
    memory_args = dict(memory_receipt["method_contract"]["candidate_args"])
    if not (
        likelihood_args.get("registered_query_likelihood_calibration")
        == LIKELIHOOD_MODE
        and memory_args.get("object_multiview_region_memory") == RUNTIME_MODE
        and _normalized(base_args) == _normalized(likelihood_args)
        and _normalized(base_args) == _normalized(memory_args)
    ):
        raise ValueError("sealed input methods do not share one frozen base")

    base_primitive = _load_primitive(
        base["method_contract"]["primitive_unary_artifact"]["path"],
        base["method_contract"]["primitive_unary_artifact"]["file_sha256"],
    )
    likelihood_primitive = _load_primitive(
        args.likelihood_primitive,
        args.likelihood_primitive_sha256,
    )
    memory_primitive = _load_primitive(
        args.memory_primitive,
        args.memory_primitive_sha256,
    )
    shared = (base_primitive, likelihood_primitive, memory_primitive)
    if not (
        len({str(value["protocol_hash"]) for value in shared}) == 1
        and len({str(value["capability_cache"]) for value in shared}) == 1
        and len({str(value["capability_cache_sha256"]) for value in shared}) == 1
        and all(
            torch.equal(base_primitive["valid"], value["valid"])
            and torch.equal(base_primitive["valid_rows"], value["valid_rows"])
            for value in shared[1:]
        )
    ):
        raise ValueError("base/field/global-row authority differs")
    likelihood_diagnostics = likelihood_primitive["compiler_contract"][
        "registered_query_likelihood"
    ]
    memory_diagnostics = memory_primitive["compiler_contract"][
        "object_multiview_region_memory"
    ]["diagnostics"]
    if not (
        int(likelihood_diagnostics["observed_rows"])
        == int(memory_diagnostics["base_observed_rows"])
        and int(likelihood_diagnostics["abstained_rows"])
        == int(memory_diagnostics["base_abstained_rows"])
    ):
        raise ValueError("sealed input original-observation counts differ")

    memory_asset = Path(args.object_region_memory).expanduser().resolve()
    if _sha256(memory_asset) != args.object_region_memory_sha256:
        raise ValueError("sealed object memory changed")
    memory_payload = torch.load(memory_asset, map_location="cpu", weights_only=True)
    memory_base = memory_payload.get("base_primitive_unary", {})
    memory_capability = memory_payload.get("capability_cache", {})
    if not (
        memory_base.get("path")
        == base["method_contract"]["primitive_unary_artifact"]["path"]
        and memory_base.get("sha256")
        == base["method_contract"]["primitive_unary_artifact"]["file_sha256"]
        and memory_capability.get("path") == base_primitive["capability_cache"]
        and memory_capability.get("sha256")
        == base_primitive["capability_cache_sha256"]
        and torch.equal(
            memory_payload["valid_rows"], base_primitive["valid_rows"]
        )
    ):
        raise ValueError("object memory base/field/global-row authority differs")
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
            "scene_id": args.scene_id,
            "device": "cuda:0",
            "output_dir": str(output),
            "prediction_only": True,
            "prediction_receipt_output": str(output / "pre_metric_prediction_receipt.json"),
            "primitive_unary_output": str(output / "primitive_unary.pt"),
            "registered_query_likelihood_calibration": LIKELIHOOD_MODE,
            "registered_disjoint_domain_composition": MODE,
            "object_multiview_region_memory": RUNTIME_MODE,
            "object_region_memory": str(memory_asset),
            "object_region_memory_sha256": args.object_region_memory_sha256,
            "source_only_correspondence_completion": "none",
            "source_correspondence_support_graph": "",
            "source_correspondence_support_graph_sha256": "",
            "source_multiview_responsibility_cache": "",
            "source_multiview_responsibility_cache_sha256": "",
        }
    )
    subprocess.run([sys.executable, str(evaluator), *_argv(candidate)], check=True)
    receipt_path = output / "pre_metric_prediction_receipt.json"
    primitive_path = output / "primitive_unary.pt"
    sealed = _load_json(receipt_path, _sha256(receipt_path))
    if not _safe_receipt(sealed, args.scene_id) or not primitive_path.is_file():
        raise ValueError("composed prediction did not seal safely")
    return {
        "scene_id": args.scene_id,
        "receipt": str(receipt_path),
        "receipt_sha256": _sha256(receipt_path),
        "primitive_unary": str(primitive_path),
        "primitive_unary_sha256": _sha256(primitive_path),
        "target_rgb_mask_or_metric_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--base-prediction-receipt", required=True)
    parser.add_argument("--base-prediction-receipt-sha256", required=True)
    parser.add_argument("--likelihood-prediction-receipt", required=True)
    parser.add_argument("--likelihood-prediction-receipt-sha256", required=True)
    parser.add_argument("--likelihood-primitive", required=True)
    parser.add_argument("--likelihood-primitive-sha256", required=True)
    parser.add_argument("--memory-prediction-receipt", required=True)
    parser.add_argument("--memory-prediction-receipt-sha256", required=True)
    parser.add_argument("--memory-primitive", required=True)
    parser.add_argument("--memory-primitive-sha256", required=True)
    parser.add_argument("--object-region-memory", required=True)
    parser.add_argument("--object-region-memory-sha256", required=True)
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--evaluator-sha256", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    print(json.dumps(run(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
