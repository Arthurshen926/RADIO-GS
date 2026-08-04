#!/usr/bin/env python3
"""Apply only the v2 continuous solver to a sealed v1 NVOS unary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from radio_gs.interfaces.prompt_responsibility_cache import (
    PromptResponsibilityAuthority,
    sha256_file,
    tensor_sha256,
)
from radio_gs.querying.support_solver import (
    PrimitiveSupportGraph,
    SupportSolverConfig,
    solve_seeded_random_walker,
)
from radio_gs.scripts.build_nvos_strict_query_conditioned_support import (
    validate_nvos_strict_support_payload,
)


ARTIFACT_TYPE = "nvos_strict_v1_unary_v2_continuous_solver"
SCHEMA_VERSION = 1
REGISTRATION_SHA256 = "4c51f0f0bf660f9fdda6fbcc6171baf28e603be3a5843f349a4b1ac3b21b8414"
FIXED_INPUTS = {
    "horns_left": {
        "selector": "f90d723fa03abb916312f2510430661e47feee0b13df1eb2251a276b55e6a469",
        "query": "577055394928b07737c05eda716ae63ffb133999d1cd352376c3c04f155b269b",
        "signed": "b4628ed951f0336138c6c7985f0e0172c8799f18baaafc8585826fb7dc2139ac",
        "graph": "cb8c0384d0951e9a4347935402e762a4c0c25fa4dcf9e493de8465054d47b7ba",
    },
    "fern": {
        "selector": "ad04b55f17aefefba9b144ce13000fb885100b9aa3cb1d41fa38f8825d99f610",
        "query": "bca7d2cdc1d4624dda1d0d95067ee3d8ade9001d398fc56be8ef14eb94d33761",
        "signed": "012a7cf991506080f09cbe4f2c100592dc60ee5120d83ec785652e48c275e017",
        "graph": "2d570d098e36dabb7fc82b06944f75def6ebc13598e97afce232f008cea34eeb",
    },
}
METHOD_CONTRACT = {
    "track": "strict_source_only",
    "unary_y_and_query_gate_g": "sealed_v1_hashed_query_compatibility",
    "unary_confidence_c": "abs(2*y-1)_clamp_min_0.05",
    "graph": "sealed_v1_bound_canonical_shared_k16_raw_affinity",
    "query_edge": "w_ij*sqrt(g_i*g_j)_symmetric_degree_renormalized",
    "hard_seeds": "per_sign_max_normalized_relu_signed_evidence_threshold_0.2_exclusive_relative",
    "laplacian_weight": 0.25,
    "cg_iterations": 128,
    "cg_tolerance": 1e-6,
    "connected_selection": "none",
    "classifier_changed": False,
    "target_dependent_tuning": False,
}
TENSOR_KEYS = {
    "primitive_probability",
    "query_compatibility",
    "unary_confidence",
    "signed_reference_evidence",
    "capability_valid",
    "hard_positive",
    "hard_negative",
}
ARTIFACT_KEYS = {
    "schema_version",
    "artifact_type",
    "scene_id",
    "method_contract",
    "method_contract_sha256",
    "experiment_registration_path",
    "experiment_registration_sha256",
    "base_selector_path",
    "base_selector_sha256",
    "base_selector_method_contract_sha256",
    "responsibility_report_path",
    "responsibility_report_sha256",
    "responsibility_file_sha256",
    "responsibility_authority_sha256",
    "responsibility_tensor_bundle_sha256",
    "support_graph_path",
    "support_graph_sha256",
    "solver_diagnostics",
    "tensors",
    "tensor_sha256",
    "tensor_bundle_sha256",
    "target_rgb_opened",
    "target_mask_opened",
    "target_metric_computed",
}


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    value = str(value)
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def validate_nvos_v1_unary_v2_solver_payload(
    payload: object,
    *,
    authority: PromptResponsibilityAuthority,
    expected_responsibility_file_sha256: str,
    expected_primitive_sha256: str | None = None,
) -> torch.Tensor:
    if not isinstance(payload, dict) or set(payload) != ARTIFACT_KEYS:
        raise ValueError("NVOS v1-unary/v2-solver artifact schema differs")
    if (
        payload["schema_version"] != SCHEMA_VERSION
        or isinstance(payload["schema_version"], bool)
        or payload["artifact_type"] != ARTIFACT_TYPE
        or payload["scene_id"] != authority.scene_id
        or payload["method_contract"] != METHOD_CONTRACT
        or payload["method_contract_sha256"] != _json_sha256(METHOD_CONTRACT)
        or payload["experiment_registration_sha256"] != REGISTRATION_SHA256
        or payload["responsibility_file_sha256"] != expected_responsibility_file_sha256
        or payload["responsibility_authority_sha256"] != authority.digest
        or payload["target_rgb_opened"] is not False
        or payload["target_mask_opened"] is not False
        or payload["target_metric_computed"] is not False
    ):
        raise ValueError("NVOS v1-unary/v2-solver method or authority differs")
    for name in (
        "base_selector_sha256",
        "base_selector_method_contract_sha256",
        "responsibility_report_sha256",
        "responsibility_tensor_bundle_sha256",
        "support_graph_sha256",
    ):
        if not _is_sha256(payload[name]):
            raise ValueError(f"NVOS v1-unary/v2-solver {name} is not a SHA-256")
    tensors = payload["tensors"]
    if not isinstance(tensors, dict) or set(tensors) != TENSOR_KEYS:
        raise ValueError("NVOS v1-unary/v2-solver tensor schema differs")
    count = int(authority.num_gaussians)
    for name in (
        "primitive_probability",
        "query_compatibility",
        "unary_confidence",
        "signed_reference_evidence",
    ):
        value = tensors[name]
        if (
            not torch.is_tensor(value)
            or value.device.type != "cpu"
            or value.dtype != torch.float32
            or tuple(value.shape) != (count,)
            or not value.is_contiguous()
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"NVOS v1-unary/v2-solver tensor {name} is malformed")
    for name in ("primitive_probability", "query_compatibility", "unary_confidence"):
        if bool(((tensors[name] < 0) | (tensors[name] > 1)).any()):
            raise ValueError(f"NVOS v1-unary/v2-solver {name} is outside [0,1]")
    for name in ("capability_valid", "hard_positive", "hard_negative"):
        value = tensors[name]
        if (
            not torch.is_tensor(value)
            or value.device.type != "cpu"
            or value.dtype != torch.bool
            or tuple(value.shape) != (count,)
            or not value.is_contiguous()
        ):
            raise ValueError(f"NVOS v1-unary/v2-solver mask {name} is malformed")
    hard_positive = tensors["hard_positive"]
    hard_negative = tensors["hard_negative"]
    if bool((hard_positive & hard_negative).any()):
        raise ValueError("NVOS v1-unary/v2-solver hard constraints overlap")
    if not bool((tensors["primitive_probability"][hard_positive] == 1).all()):
        raise ValueError("NVOS v1-unary/v2-solver positive constraints differ")
    if not bool((tensors["primitive_probability"][hard_negative] == 0).all()):
        raise ValueError("NVOS v1-unary/v2-solver negative constraints differ")
    digests = {name: tensor_sha256(value) for name, value in sorted(tensors.items())}
    if payload["tensor_sha256"] != digests or payload["tensor_bundle_sha256"] != _json_sha256(digests):
        raise ValueError("NVOS v1-unary/v2-solver tensor digest differs")
    if expected_primitive_sha256 is not None and digests["primitive_probability"] != expected_primitive_sha256:
        raise ValueError("NVOS v1-unary/v2-solver primitive probability differs")
    return tensors["primitive_probability"]


@torch.inference_mode()
def build(args: argparse.Namespace) -> dict[str, object]:
    registration_path = Path(args.experiment_registration).resolve()
    if sha256_file(registration_path) != REGISTRATION_SHA256:
        raise ValueError("NVOS v1-unary/v2-solver registration differs")
    if args.scene_id not in FIXED_INPUTS:
        raise ValueError("scene is outside registered sentinels")
    expected = FIXED_INPUTS[args.scene_id]
    report_path = Path(args.cache_report).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    authority = PromptResponsibilityAuthority.from_dict(report["authority"])
    if authority.scene_id != args.scene_id:
        raise ValueError("responsibility authority scene differs")

    base_path = Path(args.base_selector).resolve()
    if sha256_file(base_path) != expected["selector"]:
        raise ValueError("sealed v1 selector SHA-256 differs")
    base = torch.load(base_path, map_location="cpu", weights_only=True)
    validate_nvos_strict_support_payload(
        base,
        authority=authority,
        expected_responsibility_file_sha256=str(report["file_sha256"]),
    )
    base_tensors = base["tensors"]
    if (
        tensor_sha256(base_tensors["query_compatibility"]) != expected["query"]
        or tensor_sha256(base_tensors["signed_reference_evidence"]) != expected["signed"]
    ):
        raise ValueError("sealed v1 unary or signed evidence differs")
    valid = base_tensors["capability_valid"]
    global_rows = torch.where(valid)[0]
    y_global = base_tensors["query_compatibility"].float()
    signed = base_tensors["signed_reference_evidence"].float()
    y = y_global[global_rows].contiguous()
    confidence = (2.0 * y - 1.0).abs().clamp_min(0.05)
    positive = signed[global_rows].clamp_min(0.0)
    negative = (-signed[global_rows]).clamp_min(0.0)
    positive = positive / positive.max().clamp_min(1e-12)
    negative = negative / negative.max().clamp_min(1e-12)
    threshold = 0.20
    local_hard_positive = (positive >= threshold) & (positive > negative)
    local_hard_negative = (negative >= threshold) & (negative > positive)

    graph_path = Path(args.support_graph).resolve()
    if sha256_file(graph_path) != expected["graph"] or base["support_graph_sha256"] != expected["graph"]:
        raise ValueError("registered shared support graph differs")
    graph_payload = torch.load(graph_path, map_location="cpu", weights_only=True)
    if not torch.equal(torch.as_tensor(graph_payload["global_rows"]).long(), global_rows):
        raise ValueError("support graph rows differ from sealed v1 unary")
    graph = PrimitiveSupportGraph(
        edge_index=graph_payload["edge_index"],
        edge_weight=torch.as_tensor(graph_payload["edge_weight"]).float(),
        raw_affinity=torch.as_tensor(graph_payload["raw_affinity"]).float(),
        local_sigma=graph_payload["local_sigma"],
        num_nodes=int(global_rows.numel()),
        edge_channels={
            str(name): torch.as_tensor(values).float()
            for name, values in dict(graph_payload.get("edge_channels", {})).items()
        },
    )
    device = torch.device(args.device)
    config = SupportSolverConfig(
        solver_type="random_walker",
        laplacian_weight=0.25,
        cg_iterations=128,
        cg_tolerance=1e-6,
        hard_seed_threshold=threshold,
        hard_seed_conflict_policy="exclusive_relative",
    )
    solved = solve_seeded_random_walker(
        graph.to(device),
        y.to(device),
        positive.to(device),
        negative.to(device),
        config=config,
        unary_confidence=confidence.to(device),
        query_gate=y.to(device),
    ).float().cpu().contiguous()

    primitive = torch.zeros(authority.num_gaussians, dtype=torch.float32)
    unary_confidence = torch.zeros_like(primitive)
    hard_positive = torch.zeros_like(valid)
    hard_negative = torch.zeros_like(valid)
    primitive[global_rows] = solved
    unary_confidence[global_rows] = confidence
    hard_positive[global_rows] = local_hard_positive
    hard_negative[global_rows] = local_hard_negative
    primitive[hard_positive] = 1.0
    primitive[hard_negative] = 0.0
    tensors = {
        "primitive_probability": primitive.contiguous(),
        "query_compatibility": y_global.contiguous(),
        "unary_confidence": unary_confidence.contiguous(),
        "signed_reference_evidence": signed.contiguous(),
        "capability_valid": valid.contiguous(),
        "hard_positive": hard_positive.contiguous(),
        "hard_negative": hard_negative.contiguous(),
    }
    digests = {name: tensor_sha256(value) for name, value in sorted(tensors.items())}
    diagnostics = {
        "valid_rows": int(global_rows.numel()),
        "hard_positive_rows": int(hard_positive.sum()),
        "hard_negative_rows": int(hard_negative.sum()),
        "query_compatibility_mean": float(y.mean()),
        "unary_confidence_mean": float(confidence.mean()),
        "primitive_probability_mean": float(primitive.mean()),
        "primitive_support_fraction_at_0_5": float((primitive >= 0.5).double().mean()),
    }
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "scene_id": args.scene_id,
        "method_contract": METHOD_CONTRACT,
        "method_contract_sha256": _json_sha256(METHOD_CONTRACT),
        "experiment_registration_path": str(registration_path),
        "experiment_registration_sha256": REGISTRATION_SHA256,
        "base_selector_path": str(base_path),
        "base_selector_sha256": expected["selector"],
        "base_selector_method_contract_sha256": base["method_contract_sha256"],
        "responsibility_report_path": str(report_path),
        "responsibility_report_sha256": sha256_file(report_path),
        "responsibility_file_sha256": str(report["file_sha256"]),
        "responsibility_authority_sha256": authority.digest,
        "responsibility_tensor_bundle_sha256": str(report["tensor_bundle_sha256"]),
        "support_graph_path": str(graph_path),
        "support_graph_sha256": expected["graph"],
        "solver_diagnostics": diagnostics,
        "tensors": tensors,
        "tensor_sha256": digests,
        "tensor_bundle_sha256": _json_sha256(digests),
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    torch.save(artifact, output)
    output_sha256 = sha256_file(output)
    frozen = torch.load(output, map_location="cpu", weights_only=True)
    validate_nvos_v1_unary_v2_solver_payload(
        frozen,
        authority=authority,
        expected_responsibility_file_sha256=str(report["file_sha256"]),
        expected_primitive_sha256=digests["primitive_probability"],
    )
    receipt = {
        "scene_id": args.scene_id,
        "artifact_type": ARTIFACT_TYPE,
        "method_contract_sha256": artifact["method_contract_sha256"],
        "experiment_registration_sha256": REGISTRATION_SHA256,
        "output": str(output),
        "output_sha256": output_sha256,
        "primitive_probability_sha256": digests["primitive_probability"],
        "solver_diagnostics": diagnostics,
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--experiment-registration", required=True)
    parser.add_argument("--base-selector", required=True)
    parser.add_argument("--cache-report", required=True)
    parser.add_argument("--support-graph", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
