#!/usr/bin/env python3
"""Apply one frozen canonical support-graph solve to a dense prompt selector."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from radio_gs.scripts.analyze_nvos_dense_prompt_adjoint_cycle import (
    validate_dino_completion_payload,
)
from radio_gs.interfaces.prompt_responsibility_cache import (
    PromptResponsibilityAuthority,
    sha256_file,
    tensor_sha256,
)
from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline
from radio_gs.querying.support_solver import (
    PrimitiveSupportGraph,
    SupportSolverConfig,
    solve_primitive_support,
)


SOLVER_CONFIG = SupportSolverConfig(
    iterations=12,
    residual=0.30,
    unary_temperature=0.10,
    support_threshold=0.50,
    component_edge_threshold=1e-5,
    seeded_component_min_weight=0.20,
    top_k_components=3,
    solver_type="confidence_random_walker",
    laplacian_weight=1.0,
    cg_iterations=64,
    cg_tolerance=1e-5,
    hard_seed_threshold=0.20,
    hard_seed_conflict_policy="exclusive_relative",
    hard_seed_conflict_margin=0.0,
    unary_edge_contrast=0.0,
)


METHOD_CONTRACT = {
    "input": "frozen_dino_dense_prompt_exact_adjoint_primitive_probability",
    "graph": "frozen_canonical_mpr_v3_shared_support_graph_k16",
    "random_walker_weight_semantics": (
        "symmetric_normalized_affinity_derived_from_raw_affinity; stored "
        "edge_weight is solver-inert for confidence_random_walker when "
        "unary_edge_contrast=0"
    ),
    "solver_config": asdict(SOLVER_CONFIG),
    "prior_transform": "unary=0.1*logit(clamp(q,1e-6,1-1e-6)); sigmoid(unary/0.1)=q",
    "hard_positive_seeds": "none",
    "hard_negative_seeds": "none",
    "invalid_capability_rows": "preserve_input_probability",
    "threshold": 0.5,
    "connected_selection": "none",
    "target_dependent_tuning": False,
}

GRAPH_RECEIPT_KEYS = {
    "schema_version", "source", "capability_cache", "capability_metadata",
    "valid_mask_source", "feature_hash", "graph_config", "edge_channels",
    "legacy_edge_weight", "typed_edge_weight", "benchmark_images_opened",
    "benchmark_masks_opened", "text_queries_opened", "output", "num_nodes",
    "num_edges",
}
GRAPH_METADATA_KEYS = GRAPH_RECEIPT_KEYS - {"output", "num_nodes", "num_edges"}
CAPABILITY_RECEIPT_TAIL_KEYS = {
    "output", "num_gaussians", "valid_gaussians", "appearance_dim",
    "boundary_dim",
}
GRAPH_ARTIFACT_KEYS = {
    "schema_version", "artifact_type", "scene_id", "method_contract",
    "method_contract_sha256", "responsibility_authority_sha256",
    "responsibility_file_sha256", "source_completion_sha256",
    "source_primitive_probability_sha256", "render_manifest_path",
    "render_manifest_sha256", "support_graph_path", "support_graph_sha256",
    "support_graph_receipt_path", "support_graph_receipt_sha256",
    "support_graph_config_sha256", "support_graph_capability_metadata_sha256",
    "capability_receipt_path", "capability_receipt_sha256",
    "geometry_xyz_sha256", "support_graph_xyz_sha256", "tensors",
    "tensor_sha256", "tensor_bundle_sha256", "target_rgb_opened",
    "target_mask_opened",
}


def _float32_rows_sha256(value: torch.Tensor) -> str:
    array = value.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _load_json_bound(path: Path, expected_sha256: str) -> tuple[dict, str]:
    before = sha256_file(path)
    if before != str(expected_sha256):
        raise ValueError(f"JSON receipt SHA-256 differs: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or sha256_file(path) != before:
        raise ValueError(f"JSON receipt changed across trusted load: {path}")
    return value, before


def _validate_graph_receipt(
    *,
    graph_payload: dict,
    graph_path: Path,
    receipt_path: Path,
    expected_receipt_sha256: str,
    render_manifest: dict,
    authority: PromptResponsibilityAuthority,
) -> dict[str, object]:
    receipt, receipt_sha256 = _load_json_bound(
        receipt_path, expected_receipt_sha256
    )
    if set(receipt) != GRAPH_RECEIPT_KEYS:
        raise ValueError("support graph receipt schema differs")
    metadata = graph_payload["metadata"]
    expected_metadata = {
        key: value for key, value in receipt.items() if key in GRAPH_METADATA_KEYS
    }
    edge_index = graph_payload["edge_index"]
    global_rows = graph_payload["global_rows"]
    if (
        receipt["schema_version"] != 1
        or isinstance(receipt["schema_version"], bool)
        or receipt["source"]
        != "canonical_official_dino_sam3_multichannel_support_graph"
        or Path(str(receipt["output"])).resolve() != graph_path
        or receipt["num_nodes"] != int(torch.as_tensor(global_rows).numel())
        or receipt["num_edges"] != int(torch.as_tensor(edge_index).shape[1])
        or metadata != expected_metadata
        or receipt["benchmark_images_opened"] is not False
        or receipt["benchmark_masks_opened"] is not False
        or receipt["text_queries_opened"] is not False
    ):
        raise ValueError("support graph receipt or target-blind authority differs")
    capability = receipt["capability_metadata"]
    signatures = capability.get("capability_signatures", {})
    expected_field_sha = render_manifest.get("canonical_field_checkpoint_sha256")
    expected_radio_sha = capability.get("radio_checkpoint_sha256")
    if (
        not isinstance(capability, dict)
        or capability.get("schema_version") != 1
        or isinstance(capability.get("schema_version"), bool)
        or capability.get("source")
        != "canonical_radio_field_official_frozen_capability_views"
        or capability.get("query_independent") is not True
        or capability.get("custom_adaptor_head") is not False
        or capability.get("benchmark_images_opened") is not False
        or capability.get("benchmark_masks_opened") is not False
        or capability.get("text_queries_opened") is not False
        or capability.get("field_checkpoint_sha256") != expected_field_sha
        or Path(str(capability.get("field_checkpoint"))).resolve()
        != Path(str(render_manifest.get("canonical_field_checkpoint"))).resolve()
        or not isinstance(signatures, dict)
        or set(signatures) != {"appearance", "boundary"}
        or any(
            not isinstance(signature, dict)
            or signature.get("field_checkpoint_sha256") != expected_field_sha
            or signature.get("radio_checkpoint_sha256") != expected_radio_sha
            for signature in signatures.values()
        )
    ):
        raise ValueError("support graph capability authority differs")
    capability_path = Path(str(receipt["capability_cache"])).resolve()
    capability_receipt_path = Path(str(capability_path) + ".json")
    capability_receipt_sha256 = sha256_file(capability_receipt_path)
    capability_receipt, _ = _load_json_bound(
        capability_receipt_path, capability_receipt_sha256
    )
    if (
        set(capability_receipt)
        != set(capability) | CAPABILITY_RECEIPT_TAIL_KEYS
        or {
            key: value
            for key, value in capability_receipt.items()
            if key not in CAPABILITY_RECEIPT_TAIL_KEYS
        }
        != capability
        or Path(str(capability_receipt["output"])).resolve() != capability_path
        or int(capability_receipt["num_gaussians"]) != authority.num_gaussians
        or int(capability_receipt["valid_gaussians"])
        != int(torch.as_tensor(global_rows).numel())
    ):
        raise ValueError("capability cache receipt differs from graph authority")
    return {
        "support_graph_receipt_sha256": receipt_sha256,
        "support_graph_config_sha256": _json_sha256(receipt["graph_config"]),
        "support_graph_capability_metadata_sha256": _json_sha256(capability),
        "capability_receipt_path": str(capability_receipt_path),
        "capability_receipt_sha256": capability_receipt_sha256,
    }


def validate_graph_selector_payload(
    payload: object,
    *,
    authority: PromptResponsibilityAuthority,
    expected_responsibility_file_sha256: str,
    expected_completion_sha256: str,
    expected_source_primitive_sha256: str,
    expected_primitive_sha256: str | None = None,
) -> torch.Tensor:
    if not isinstance(payload, dict) or set(payload) != GRAPH_ARTIFACT_KEYS:
        raise ValueError("graph selector artifact schema differs")
    if (
        payload["schema_version"] != 1
        or isinstance(payload["schema_version"], bool)
        or payload["artifact_type"] != "nvos_dino_dense_prompt_fixed_graph_selector"
        or payload["scene_id"] != authority.scene_id
        or payload["method_contract"] != METHOD_CONTRACT
        or payload["method_contract_sha256"] != _json_sha256(METHOD_CONTRACT)
        or payload["responsibility_authority_sha256"] != authority.digest
        or payload["responsibility_file_sha256"]
        != expected_responsibility_file_sha256
        or payload["source_completion_sha256"] != expected_completion_sha256
        or payload["source_primitive_probability_sha256"]
        != expected_source_primitive_sha256
        or payload["target_rgb_opened"] is not False
        or payload["target_mask_opened"] is not False
    ):
        raise ValueError("graph selector method or authority differs")
    render_manifest_path = Path(str(payload["render_manifest_path"])).resolve()
    graph_path = Path(str(payload["support_graph_path"])).resolve()
    receipt_path = Path(str(payload["support_graph_receipt_path"])).resolve()
    capability_receipt_path = Path(str(payload["capability_receipt_path"])).resolve()
    if (
        sha256_file(render_manifest_path) != payload["render_manifest_sha256"]
        or sha256_file(graph_path) != payload["support_graph_sha256"]
        or sha256_file(receipt_path) != payload["support_graph_receipt_sha256"]
        or sha256_file(capability_receipt_path)
        != payload["capability_receipt_sha256"]
        or payload["geometry_xyz_sha256"] != authority.geometry_xyz_sha256
    ):
        raise ValueError("graph selector transitive file authority differs")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        not isinstance(receipt, dict)
        or set(receipt) != GRAPH_RECEIPT_KEYS
        or Path(str(receipt["output"])).resolve() != graph_path
        or _json_sha256(receipt["graph_config"])
        != payload["support_graph_config_sha256"]
        or _json_sha256(receipt["capability_metadata"])
        != payload["support_graph_capability_metadata_sha256"]
        or Path(str(receipt["capability_cache"]) + ".json").resolve()
        != capability_receipt_path
        or any(
            not isinstance(payload[name], str)
            or len(payload[name]) != 64
            or any(char not in "0123456789abcdef" for char in payload[name])
            for name in (
                "render_manifest_sha256", "support_graph_sha256",
                "support_graph_receipt_sha256", "support_graph_config_sha256",
                "support_graph_capability_metadata_sha256",
                "capability_receipt_sha256", "geometry_xyz_sha256",
                "support_graph_xyz_sha256",
            )
        )
    ):
        raise ValueError("graph selector receipt authority differs")
    tensors = payload["tensors"]
    if not isinstance(tensors, dict) or set(tensors) != {"primitive_probability"}:
        raise ValueError("graph selector tensor schema differs")
    probability = tensors["primitive_probability"]
    if (
        not torch.is_tensor(probability)
        or probability.device.type != "cpu"
        or probability.dtype != torch.float32
        or tuple(probability.shape) != (authority.num_gaussians,)
        or not probability.is_contiguous()
        or not bool(torch.isfinite(probability).all())
        or bool(((probability < 0.0) | (probability > 1.0)).any())
    ):
        raise ValueError("graph selector probability is malformed")
    actual = tensor_sha256(probability)
    if (
        payload["tensor_sha256"] != {"primitive_probability": actual}
        or payload["tensor_bundle_sha256"] != _json_sha256(payload["tensor_sha256"])
        or (expected_primitive_sha256 is not None and actual != expected_primitive_sha256)
    ):
        raise ValueError("graph selector tensor digest differs")
    return probability


def _load_base_selector(
    args: argparse.Namespace,
    authority: PromptResponsibilityAuthority,
    *,
    expected_responsibility_file_sha256: str,
):
    path = Path(args.completion).resolve()
    before_sha256 = sha256_file(path)
    if before_sha256 != str(args.expected_completion_sha256):
        raise ValueError("base completion file SHA-256 differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    tensors = validate_dino_completion_payload(
        payload,
        authority=authority,
        expected_responsibility_file_sha256=expected_responsibility_file_sha256,
        expected_primitive_sha256=str(args.expected_primitive_sha256),
    )
    if sha256_file(path) != before_sha256:
        raise ValueError("base completion changed across trusted load")
    probability = tensors["primitive_probability"]
    return probability, payload


@torch.inference_mode()
def propagate(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    cache_report = json.loads(Path(args.cache_report).resolve().read_text(encoding="utf-8"))
    authority = PromptResponsibilityAuthority.from_dict(cache_report["authority"])
    if authority.scene_id != args.scene_id:
        raise ValueError("scene differs from prompt responsibility authority")
    probability, base_payload = _load_base_selector(
        args,
        authority,
        expected_responsibility_file_sha256=str(cache_report["file_sha256"]),
    )

    render_manifest_path = Path(args.render_manifest).resolve()
    render_manifest_sha256 = sha256_file(render_manifest_path)
    render_manifest = json.loads(render_manifest_path.read_text(encoding="utf-8"))
    geometry = render_manifest.get("canonical_field_geometry_fingerprint", {})
    if (
        render_manifest.get("scene_id") != args.scene_id
        or geometry.get("xyz_sha256") != authority.geometry_xyz_sha256
        or int(geometry.get("num_gaussians", -1)) != authority.num_gaussians
    ):
        raise ValueError("canonical render geometry differs from selector authority")
    if sha256_file(render_manifest_path) != render_manifest_sha256:
        raise ValueError("canonical render manifest changed across trusted load")

    graph_path = Path(args.support_graph).resolve()
    graph_sha256 = sha256_file(graph_path)
    if graph_sha256 != str(args.support_graph_sha256):
        raise ValueError("support graph SHA-256 differs from frozen authority")
    graph_payload = torch.load(graph_path, map_location="cpu", weights_only=True)
    required = {
        "schema_version", "global_rows", "num_global_rows", "xyz", "edge_index",
        "edge_weight", "raw_affinity", "edge_channels", "local_sigma", "metadata",
    }
    if not isinstance(graph_payload, dict) or set(graph_payload) != required:
        raise ValueError("support graph payload schema differs")
    if sha256_file(graph_path) != graph_sha256:
        raise ValueError("support graph changed across trusted load")
    metadata = graph_payload["metadata"]
    capability_metadata = metadata.get("capability_metadata", {})
    if (
        metadata.get("source") != "canonical_official_dino_sam3_multichannel_support_graph"
        or metadata.get("benchmark_images_opened") is not False
        or metadata.get("benchmark_masks_opened") is not False
        or metadata.get("text_queries_opened") is not False
        or capability_metadata.get("field_checkpoint_sha256")
        != render_manifest.get("canonical_field_checkpoint_sha256")
        or int(graph_payload["num_global_rows"]) != authority.num_gaussians
    ):
        raise ValueError("support graph target-blind/canonical authority differs")
    global_rows = torch.as_tensor(graph_payload["global_rows"]).long().cpu().contiguous()
    stored_global_rows = graph_payload["global_rows"]
    if (
        not torch.is_tensor(stored_global_rows)
        or stored_global_rows.device.type != "cpu"
        or stored_global_rows.dtype != torch.int64
        or not stored_global_rows.is_contiguous()
        or global_rows.ndim != 1
        or global_rows.numel() == 0
        or int(global_rows.min()) < 0
        or int(global_rows.max()) >= authority.num_gaussians
        or torch.unique(global_rows).numel() != global_rows.numel()
    ):
        raise ValueError("support graph global rows are invalid")
    graph_xyz = torch.as_tensor(graph_payload["xyz"]).float().cpu()
    edge_index = graph_payload["edge_index"]
    edge_count = int(torch.as_tensor(edge_index).shape[1])
    if (
        not torch.is_tensor(graph_payload["xyz"])
        or graph_payload["xyz"].dtype != torch.float32
        or not graph_payload["xyz"].is_contiguous()
        or graph_xyz.shape != (global_rows.numel(), 3)
        or not bool(torch.isfinite(graph_xyz).all())
        or not torch.is_tensor(edge_index)
        or edge_index.dtype != torch.int64
        or tuple(edge_index.shape) != (2, edge_count)
        or not edge_index.is_contiguous()
    ):
        raise ValueError("support graph xyz rows are invalid")
    for name in ("edge_weight", "raw_affinity"):
        value = graph_payload[name]
        if (
            not torch.is_tensor(value)
            or value.dtype != torch.float16
            or tuple(value.shape) != (edge_count,)
            or not value.is_contiguous()
            or not bool(torch.isfinite(value).all())
            or bool(((value < 0.0) | (value > 1.0)).any())
        ):
            raise ValueError(f"support graph {name} is malformed")
    if set(graph_payload["edge_channels"]) != {"geometry", "appearance", "boundary"}:
        raise ValueError("support graph edge-channel schema differs")
    for name, value in graph_payload["edge_channels"].items():
        if (
            not torch.is_tensor(value)
            or value.dtype != torch.float16
            or tuple(value.shape) != (edge_count,)
            or not value.is_contiguous()
            or not bool(torch.isfinite(value).all())
            or bool(((value < 0.0) | (value > 1.0)).any())
        ):
            raise ValueError(f"support graph edge channel {name} is malformed")
    local_sigma = graph_payload["local_sigma"]
    if (
        not torch.is_tensor(local_sigma)
        or local_sigma.dtype != torch.float32
        or tuple(local_sigma.shape) != (global_rows.numel(),)
        or not local_sigma.is_contiguous()
        or not bool(torch.isfinite(local_sigma).all())
        or bool((local_sigma <= 0.0).any())
    ):
        raise ValueError("support graph local sigma is malformed")
    receipt_path = Path(args.support_graph_receipt).resolve()
    receipt_authority = _validate_graph_receipt(
        graph_payload=graph_payload,
        graph_path=graph_path,
        receipt_path=receipt_path,
        expected_receipt_sha256=str(args.support_graph_receipt_sha256),
        render_manifest=render_manifest,
        authority=authority,
    )

    config_path = Path(str(render_manifest["config"])).resolve()
    checkpoint_path = Path(str(render_manifest["checkpoint"])).resolve()
    if (
        sha256_file(config_path) != render_manifest.get("config_sha256")
        or sha256_file(checkpoint_path) != render_manifest.get("checkpoint_sha256")
        or render_manifest.get("config_sha256")
        != authority.source_sha256["gaussfm_config"]
        or render_manifest.get("checkpoint_sha256")
        != authority.geometry_checkpoint_sha256
    ):
        raise ValueError("canonical render pipeline source authority differs")
    model, _codec, _renderer, _sharpener, refiner, _field_config, _is_hybrid = (
        load_render_pipeline(
            str(config_path), str(checkpoint_path), device,
            strict_checkpoint_contract=True, load_ply_rgb_features=False,
        )
    )
    if refiner is not None:
        raise ValueError("graph authority binding forbids RGB screen refiners")
    full_xyz = model.get_xyz().detach().float().cpu().contiguous()
    full_xyz_sha256 = _float32_rows_sha256(full_xyz)
    if (
        tuple(full_xyz.shape) != (authority.num_gaussians, 3)
        or full_xyz_sha256 != authority.geometry_xyz_sha256
        or not torch.equal(graph_xyz, full_xyz[global_rows])
    ):
        raise ValueError("support graph xyz does not equal canonical geometry rows")
    del model, full_xyz
    if device.type == "cuda":
        torch.cuda.empty_cache()
    graph = PrimitiveSupportGraph(
        edge_index=graph_payload["edge_index"],
        edge_weight=torch.as_tensor(graph_payload["edge_weight"]).float(),
        raw_affinity=torch.as_tensor(graph_payload["raw_affinity"]).float(),
        local_sigma=torch.as_tensor(graph_payload["local_sigma"]).float(),
        num_nodes=int(global_rows.numel()),
        edge_channels={
            str(name): torch.as_tensor(values).float()
            for name, values in dict(graph_payload["edge_channels"]).items()
        },
    ).to(device)
    prior = probability[global_rows].to(device).clamp(1e-6, 1 - 1e-6)
    unary = 0.1 * torch.logit(prior)
    propagated_valid = solve_primitive_support(
        graph,
        unary,
        config=SOLVER_CONFIG,
    ).cpu().contiguous()
    propagated = probability.clone()
    propagated[global_rows] = propagated_valid
    if not bool(torch.isfinite(propagated).all()) or bool(
        ((propagated < 0) | (propagated > 1)).any()
    ):
        raise ValueError("graph-propagated selector is not a finite probability")

    artifact = {
        "schema_version": 1,
        "artifact_type": "nvos_dino_dense_prompt_fixed_graph_selector",
        "scene_id": args.scene_id,
        "method_contract": METHOD_CONTRACT,
        "method_contract_sha256": _json_sha256(METHOD_CONTRACT),
        "responsibility_authority_sha256": authority.digest,
        "responsibility_file_sha256": str(cache_report["file_sha256"]),
        "source_completion_sha256": str(args.expected_completion_sha256),
        "source_primitive_probability_sha256": str(args.expected_primitive_sha256),
        "render_manifest_path": str(render_manifest_path),
        "render_manifest_sha256": render_manifest_sha256,
        "support_graph_path": str(graph_path),
        "support_graph_sha256": graph_sha256,
        "support_graph_receipt_path": str(receipt_path),
        **receipt_authority,
        "geometry_xyz_sha256": full_xyz_sha256,
        "support_graph_xyz_sha256": tensor_sha256(graph_xyz.contiguous()),
        "tensors": {"primitive_probability": propagated},
        "tensor_sha256": {"primitive_probability": tensor_sha256(propagated)},
        "target_rgb_opened": False,
        "target_mask_opened": False,
    }
    artifact["tensor_bundle_sha256"] = _json_sha256(artifact["tensor_sha256"])
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(output_path)
    torch.save(artifact, output_path)
    output_sha256 = sha256_file(output_path)
    frozen = torch.load(output_path, map_location="cpu", weights_only=True)
    validate_graph_selector_payload(
        frozen,
        authority=authority,
        expected_responsibility_file_sha256=str(cache_report["file_sha256"]),
        expected_completion_sha256=str(args.expected_completion_sha256),
        expected_source_primitive_sha256=str(args.expected_primitive_sha256),
        expected_primitive_sha256=artifact["tensor_sha256"]["primitive_probability"],
    )
    if sha256_file(output_path) != output_sha256:
        raise ValueError("graph selector changed across freeze/reload")
    report = {
        "scene_id": args.scene_id,
        "method_contract": METHOD_CONTRACT,
        "method_contract_sha256": artifact["method_contract_sha256"],
        "source_completion_sha256": str(args.expected_completion_sha256),
        "source_primitive_probability_sha256": str(args.expected_primitive_sha256),
        "support_graph": str(graph_path),
        "support_graph_sha256": graph_sha256,
        "support_graph_nodes": int(global_rows.numel()),
        "support_graph_edges": int(graph.edge_index.shape[1]),
        "output": str(output_path),
        "output_sha256": output_sha256,
        "primitive_probability_sha256": artifact["tensor_sha256"]["primitive_probability"],
        "base_mean_graph_rows": float(probability[global_rows].mean()),
        "propagated_mean_graph_rows": float(propagated_valid.mean()),
        "mean_absolute_change_graph_rows": float(
            (probability[global_rows] - propagated_valid).abs().mean()
        ),
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--cache-report", required=True)
    parser.add_argument("--completion", required=True)
    parser.add_argument("--expected-completion-sha256", required=True)
    parser.add_argument("--expected-primitive-sha256", required=True)
    parser.add_argument("--render-manifest", required=True)
    parser.add_argument("--support-graph", required=True)
    parser.add_argument("--support-graph-sha256", required=True)
    parser.add_argument("--support-graph-receipt", required=True)
    parser.add_argument("--support-graph-receipt-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(propagate(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
