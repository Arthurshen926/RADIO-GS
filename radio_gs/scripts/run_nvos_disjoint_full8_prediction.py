#!/usr/bin/env python3
"""Run one preregistered NVOS full8 disjoint-domain prediction scene."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import torch

from radio_gs.interfaces.prompt_responsibility_cache import tensor_sha256
from radio_gs.querying.disjoint_domain_composition import MODE
from radio_gs.querying.multiview_region_memory_runtime import RUNTIME_MODE


ARTIFACT_TYPE = "nvos_disjoint_domain_composition_full8_preregistration_v1"
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


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_record(record: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    path = Path(str(record.get("path", ""))).expanduser().resolve(strict=True)
    if sha256_file(path) != str(record.get("sha256", "")):
        raise ValueError(f"{label} changed after sealing")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def load_torch_record(record: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    path = Path(str(record.get("path", ""))).expanduser().resolve(strict=True)
    if sha256_file(path) != str(record.get("sha256", "")):
        raise ValueError(f"{label} changed after sealing")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def normalized_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(arguments)
    for name in PATH_ONLY_ARGS | FACTOR_ARGS:
        result.pop(name, None)
    return result


def arguments_to_argv(arguments: Mapping[str, Any]) -> list[str]:
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


def safe_prediction_receipt(value: Mapping[str, Any], scene_id: str) -> bool:
    return bool(
        value.get("scene_id") == scene_id
        and value.get("sealed_before_target_ground_truth_open") is True
        and value.get("target_rgb_opened") is False
        and value.get("target_mask_opened") is False
        and value.get("target_metric_opened") is False
    )


def verify_partition_artifact(
    primitive: Mapping[str, Any],
    base_primitive: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the same-runtime causal partition without opening target labels."""

    partition = primitive.get("disjoint_domain_partition")
    compiler = primitive.get("compiler_contract")
    if not isinstance(partition, Mapping) or not isinstance(compiler, Mapping):
        raise ValueError("candidate lacks disjoint-domain authority")
    diagnostics = compiler.get("registered_disjoint_domain_composition")
    likelihood = compiler.get("registered_query_likelihood")
    memory = compiler.get("object_multiview_region_memory")
    if not all(isinstance(value, Mapping) for value in (diagnostics, likelihood, memory)):
        raise ValueError("candidate lacks factor diagnostics")
    memory_diagnostics = memory.get("diagnostics")
    if not isinstance(memory_diagnostics, Mapping):
        raise ValueError("candidate lacks region-memory diagnostics")

    names = ("observed_rows", "memory_rows", "abstained_rows", "hard_anchor_rows")
    masks = {
        name: torch.as_tensor(partition.get(name)).bool().reshape(-1).contiguous()
        for name in names
    }
    tensor_hashes = partition.get("tensor_sha256")
    if not isinstance(tensor_hashes, Mapping) or any(
        tensor_sha256(masks[name]) != tensor_hashes.get(name) for name in names
    ):
        raise ValueError("candidate partition tensor changed")
    observed, memory_rows, abstained, anchors = (masks[name] for name in names)
    if not (
        torch.equal(primitive.get("valid"), base_primitive.get("valid"))
        and torch.equal(primitive.get("valid_rows"), base_primitive.get("valid_rows"))
        and primitive.get("protocol_hash") == base_primitive.get("protocol_hash")
        and primitive.get("capability_cache") == base_primitive.get("capability_cache")
        and primitive.get("capability_cache_sha256")
        == base_primitive.get("capability_cache_sha256")
        and torch.equal(partition.get("global_rows"), primitive.get("valid_rows"))
        and all(mask.shape == observed.shape for mask in masks.values())
        and observed.numel() == int(torch.as_tensor(primitive["valid"]).sum())
        and torch.equal(observed | memory_rows | abstained, torch.ones_like(observed))
        and not bool(
            (observed & memory_rows).any()
            or (observed & abstained).any()
            or (memory_rows & abstained).any()
            or (anchors & ~observed).any()
        )
        and diagnostics.get("partition_exhaustive") is True
        and diagnostics.get("partition_pairwise_disjoint") is True
        and diagnostics.get("assignment_commutative") is True
        and diagnostics.get("same_row_double_counted") is False
        and diagnostics.get("probability_average_or_product_of_experts_used") is False
        and diagnostics.get("observed_unary_bitwise_equal_to_learned") is True
        and diagnostics.get("hard_anchor_unary_bitwise_equal_to_learned") is True
        and int(diagnostics.get("observed_rows", -1)) == int(observed.sum())
        and int(diagnostics.get("memory_rows", -1)) == int(memory_rows.sum())
        and int(diagnostics.get("abstained_rows", -1)) == int(abstained.sum())
        and diagnostics.get("observed_mask_sha256") == tensor_sha256(observed)
        and int(likelihood.get("observed_rows", -1)) == int(observed.sum())
        and int(likelihood.get("abstained_rows", -1)) == int((~observed).sum())
        and int(memory_diagnostics.get("base_observed_rows", -1))
        == int(observed.sum())
        and int(memory_diagnostics.get("base_abstained_rows", -1))
        == int((~observed).sum())
        and int(memory_diagnostics.get("completed_rows", -1))
        == int(memory_rows.sum())
        and memory_diagnostics.get("observed_values_bitwise_equal") is True
        and memory_diagnostics.get("observed_confidence_bitwise_equal") is True
        and memory_diagnostics.get("observed_unary_bitwise_equal_to_likelihood") is True
    ):
        raise ValueError("candidate partition causal contract differs")
    probability = torch.as_tensor(primitive["primitive_unary_probability"]).float()
    if (
        probability.shape != torch.as_tensor(primitive["valid"]).shape
        or not bool(torch.isfinite(probability).all())
        or bool(((probability < 0) | (probability > 1)).any())
    ):
        raise ValueError("candidate primitive probability differs")
    return {
        "observed_rows": int(observed.sum()),
        "memory_rows": int(memory_rows.sum()),
        "abstained_rows": int(abstained.sum()),
        "hard_anchor_rows": int(anchors.sum()),
        "observed_mask_sha256": tensor_sha256(observed),
        "partition_exhaustive": True,
        "partition_pairwise_disjoint": True,
        "same_row_double_counted": False,
        "observed_and_hard_anchor_unary_bitwise_preserved": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    prereg_path = Path(args.preregistration).expanduser().resolve(strict=True)
    if sha256_file(prereg_path) != args.preregistration_sha256:
        raise ValueError("full8 preregistration changed")
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    runner_path = Path(__file__).resolve()
    if not (
        isinstance(prereg, Mapping)
        and prereg.get("artifact_type") == ARTIFACT_TYPE
        and prereg.get("status")
        == "frozen_after_six_scene_source_memory_seal_before_composed_predictions_or_target_scoring"
        and prereg.get("frozen_implementation", {}).get("runner", {}).get("sha256")
        == sha256_file(runner_path)
        and args.scene_id in prereg.get("new_prediction_scene_order", [])
    ):
        raise ValueError("full8 preregistration contract differs")
    inputs = prereg["sealed_inputs"][args.scene_id]
    base_receipt = load_json_record(
        inputs["base_prediction_receipt"], label="base prediction receipt"
    )
    if not safe_prediction_receipt(base_receipt, args.scene_id):
        raise ValueError("base prediction safety barrier differs")
    base_primitive = load_torch_record(inputs["base_primitive"], label="base primitive")
    memory_receipt = load_json_record(inputs["memory_receipt"], label="memory receipt")
    memory_asset = load_torch_record(inputs["memory_asset"], label="memory asset")
    if not (
        memory_receipt.get("artifact_type")
        == "multiview_region_memory_primitive_receipt_v1"
        and memory_receipt.get("status")
        == "primitive_region_memory_receipt_sealed_before_target_access"
        and memory_receipt.get("scene_id") == args.scene_id
        and memory_receipt.get("artifact") == inputs["memory_asset"]
        and memory_receipt.get("source_access", {}).get("target_rgb_opened") is False
        and memory_receipt.get("source_access", {}).get("target_mask_opened") is False
        and memory_receipt.get("source_access", {}).get("target_metric_opened") is False
        and memory_asset.get("scene_id") == args.scene_id
        and memory_asset.get("base_primitive_unary") == inputs["base_primitive"]
        and memory_asset.get("capability_cache", {}).get("path")
        == base_primitive.get("capability_cache")
        and memory_asset.get("capability_cache", {}).get("sha256")
        == base_primitive.get("capability_cache_sha256")
        and torch.equal(memory_asset.get("valid_rows"), base_primitive.get("valid_rows"))
    ):
        raise ValueError("memory/base/field/global-row authority differs")

    evaluator_record = prereg["frozen_implementation"]["evaluator"]
    evaluator = Path(str(evaluator_record["path"])).resolve(strict=True)
    if sha256_file(evaluator) != evaluator_record["sha256"]:
        raise ValueError("candidate evaluator changed")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"candidate output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    base_arguments = dict(base_receipt["method_contract"]["candidate_args"])
    if not (
        base_arguments.get("disable_registered_graph") is True
        and base_arguments.get("registered_readout_stage") == "unary_prior"
        and float(base_receipt["method_contract"]["score_threshold"]) == 0.5
    ):
        raise ValueError("frozen graph/readout/threshold differs")
    candidate = dict(base_arguments)
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
            "object_region_memory": inputs["memory_asset"]["path"],
            "object_region_memory_sha256": inputs["memory_asset"]["sha256"],
            "source_only_correspondence_completion": "none",
            "source_correspondence_support_graph": "",
            "source_correspondence_support_graph_sha256": "",
            "source_multiview_responsibility_cache": "",
            "source_multiview_responsibility_cache_sha256": "",
        }
    )
    subprocess.run([sys.executable, str(evaluator), *arguments_to_argv(candidate)], check=True)
    receipt_path = output / "pre_metric_prediction_receipt.json"
    primitive_path = output / "primitive_unary.pt"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not safe_prediction_receipt(receipt, args.scene_id):
        raise ValueError("candidate prediction safety barrier differs")
    method = receipt["method_contract"]
    candidate_arguments = dict(method["candidate_args"])
    if not (
        method.get("evaluator_sha256") == evaluator_record["sha256"]
        and candidate_arguments.get("registered_query_likelihood_calibration")
        == LIKELIHOOD_MODE
        and candidate_arguments.get("registered_disjoint_domain_composition") == MODE
        and candidate_arguments.get("object_multiview_region_memory") == RUNTIME_MODE
        and normalized_arguments(candidate_arguments) == normalized_arguments(base_arguments)
        and float(method.get("score_threshold", -1)) == 0.5
    ):
        raise ValueError("candidate method differs from frozen recipe")
    primitive_record = method["primitive_unary_artifact"]
    if Path(primitive_record["path"]).resolve() != primitive_path:
        raise ValueError("candidate primitive output path differs")
    primitive = load_torch_record(
        {"path": str(primitive_path), "sha256": primitive_record["file_sha256"]},
        label="candidate primitive",
    )
    partition = verify_partition_artifact(primitive, base_primitive)
    return {
        "scene_id": args.scene_id,
        "prediction_receipt": {"path": str(receipt_path), "sha256": sha256_file(receipt_path)},
        "primitive_unary": {"path": str(primitive_path), "sha256": sha256_file(primitive_path)},
        "partition": partition,
        "target_rgb_mask_or_metric_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    print(json.dumps(run(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
