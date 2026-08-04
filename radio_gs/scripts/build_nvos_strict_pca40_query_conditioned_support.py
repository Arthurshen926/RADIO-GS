#!/usr/bin/env python3
"""Build strict NVOS support with a frozen C-RADIO primitive PCA40 relation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from radio_gs.evaluation.promptable_segmentation import load_ground_truth_mask
from radio_gs.interfaces.prompt_responsibility_cache import (
    PromptResponsibilityAuthority,
    load_prompt_responsibility_cache,
    sha256_file,
    tensor_sha256,
)
from radio_gs.interfaces.query_diffusion_cache import load_query_diffusion_knn_cache
from radio_gs.interfaces.query_diffusion_relation_cache import (
    load_query_diffusion_relation_cache,
)
from radio_gs.querying.query_conditioned_diffusion import (
    QueryConditionedDiffusionConfig,
    compute_query_conditioned_support,
)


ARTIFACT_TYPE = "nvos_strict_cradio_dino_v3_primitive_pca40_query_conditioned_support"
SCHEMA_VERSION = 1
REGISTRATION_SHA256 = (
    "46cf89efa1de5a26518b6d80082b17db6a16d94572448e6b8772152e30f4555a"
)
RELATION_REGISTRATION_SHA256 = (
    "4728cfc6cfc5028bcb4a8d966afb7914ca2a24e2a753b5494407c8a346c2bc4c"
)
METHOD_CONTRACT = {
    "track": "strict_raw_positive_negative_scribble",
    "initial_evidence": "signed_exact_W_transpose_scribble_divided_by_W_transpose_one",
    "classifier": "balanced_logistic_C_0.01_signed_nonzero_rows_weighted_by_W_transpose_one",
    "relation_feature": "query_independent_official_C_RADIO_DINOv3_primitive_standardized_PCA40_singular_weighted",
    "relation_feature_role": "LUDVIG_inspired_adaptation_not_native_DINOv2_or_exact_LUDVIG_feature_match",
    "relation_fit_population": "all_capability_valid_primitive_rows_target_blind",
    "topology": "exact_euclidean_num_neighbors_200_plus_retained_self_equals_K201",
    "feature_bandwidth": 2.0,
    "regularizer_bandwidth": 4.0,
    "bandwidth_selection": "frozen_before_PCA40_target_readout_no_target_tuning",
    "query_edge_gate": "sqrt(P_i*P_j)",
    "diffusion_kernel": "released_N_by_K_slotwise_normalization_then_symmetrize_binarize_1e-5",
    "iterations": 100,
    "direct_anchor_rule": "exact_positive_exclusive_rows_set_1_negative_exclusive_rows_set_0",
    "invalid_capability_rows": "zero_except_exact_direct_exclusive_anchors",
    "threshold": 0.5,
    "connected_selection": "none",
    "target_dependent_tuning": False,
}
TENSOR_KEYS = {
    "primitive_probability",
    "query_compatibility",
    "signed_reference_evidence",
    "reference_weight",
    "capability_valid",
    "positive_exclusive",
    "negative_exclusive",
    "conflict",
}
ARTIFACT_KEYS = {
    "schema_version",
    "artifact_type",
    "scene_id",
    "method_contract",
    "method_contract_sha256",
    "experiment_registration_path",
    "experiment_registration_sha256",
    "responsibility_report_path",
    "responsibility_report_sha256",
    "responsibility_file_sha256",
    "responsibility_authority_sha256",
    "responsibility_tensor_bundle_sha256",
    "relation_cache_path",
    "relation_cache_sha256",
    "relation_cache_receipt_sha256",
    "relation_feature_sha256",
    "source_feature_sha256",
    "source_xyz_sha256",
    "capability_sidecar_sha256",
    "field_checkpoint_sha256",
    "support_graph_path",
    "support_graph_sha256",
    "knn_cache_path",
    "knn_cache_sha256",
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
    digest = str(value)
    return len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)


def validate_nvos_strict_pca40_support_payload(
    payload: object,
    *,
    authority: PromptResponsibilityAuthority,
    expected_registration_sha256: str = REGISTRATION_SHA256,
    expected_responsibility_file_sha256: str,
    expected_primitive_sha256: str | None = None,
) -> torch.Tensor:
    """Fail closed on the PCA40 selector authority and every stored tensor."""

    if not isinstance(payload, dict) or set(payload) != ARTIFACT_KEYS:
        raise ValueError("NVOS strict PCA40 support artifact schema differs")
    if (
        payload["schema_version"] != SCHEMA_VERSION
        or isinstance(payload["schema_version"], bool)
        or payload["artifact_type"] != ARTIFACT_TYPE
        or payload["scene_id"] != authority.scene_id
        or payload["method_contract"] != METHOD_CONTRACT
        or payload["method_contract_sha256"] != _json_sha256(METHOD_CONTRACT)
        or payload["experiment_registration_sha256"] != expected_registration_sha256
        or payload["responsibility_file_sha256"] != expected_responsibility_file_sha256
        or payload["responsibility_authority_sha256"] != authority.digest
        or payload["target_rgb_opened"] is not False
        or payload["target_mask_opened"] is not False
        or payload["target_metric_computed"] is not False
    ):
        raise ValueError("NVOS strict PCA40 support method or authority differs")
    for name in (
        "responsibility_report_sha256",
        "responsibility_tensor_bundle_sha256",
        "relation_cache_sha256",
        "relation_cache_receipt_sha256",
        "relation_feature_sha256",
        "source_feature_sha256",
        "source_xyz_sha256",
        "capability_sidecar_sha256",
        "field_checkpoint_sha256",
        "support_graph_sha256",
        "knn_cache_sha256",
    ):
        if not _is_sha256(payload[name]):
            raise ValueError(f"NVOS strict PCA40 support {name} is not a SHA-256")
    tensors = payload["tensors"]
    if not isinstance(tensors, dict) or set(tensors) != TENSOR_KEYS:
        raise ValueError("NVOS strict PCA40 support tensor schema differs")
    count = int(authority.num_gaussians)
    for name in (
        "primitive_probability",
        "query_compatibility",
        "signed_reference_evidence",
        "reference_weight",
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
            raise ValueError(f"NVOS strict PCA40 support tensor {name} is malformed")
    for name in ("primitive_probability", "query_compatibility"):
        if bool(((tensors[name] < 0) | (tensors[name] > 1)).any()):
            raise ValueError(f"NVOS strict PCA40 probability {name} is outside [0,1]")
    if bool(
        (
            (tensors["signed_reference_evidence"] < -1)
            | (tensors["signed_reference_evidence"] > 1)
        ).any()
    ):
        raise ValueError("NVOS strict PCA40 signed evidence is outside [-1,1]")
    if bool((tensors["reference_weight"] < 0).any()):
        raise ValueError("NVOS strict PCA40 reference weight is negative")
    for name in ("capability_valid", "positive_exclusive", "negative_exclusive", "conflict"):
        value = tensors[name]
        if (
            not torch.is_tensor(value)
            or value.device.type != "cpu"
            or value.dtype != torch.bool
            or tuple(value.shape) != (count,)
            or not value.is_contiguous()
        ):
            raise ValueError(f"NVOS strict PCA40 support mask {name} is malformed")
    positive = tensors["positive_exclusive"]
    negative = tensors["negative_exclusive"]
    conflict = tensors["conflict"]
    if bool((positive & negative).any()) or bool((positive & conflict).any()) or bool(
        (negative & conflict).any()
    ):
        raise ValueError("NVOS strict PCA40 exact prompt row partitions overlap")
    if not bool((tensors["primitive_probability"][positive] == 1).all()):
        raise ValueError("NVOS strict PCA40 positive anchors are not hard one")
    if not bool((tensors["primitive_probability"][negative] == 0).all()):
        raise ValueError("NVOS strict PCA40 negative anchors are not hard zero")
    if bool((tensors["query_compatibility"][~tensors["capability_valid"]] != 0).any()):
        raise ValueError("NVOS strict PCA40 invalid rows have compatibility")
    digests = {name: tensor_sha256(value) for name, value in sorted(tensors.items())}
    if payload["tensor_sha256"] != digests or payload["tensor_bundle_sha256"] != _json_sha256(
        digests
    ):
        raise ValueError("NVOS strict PCA40 support tensor digest differs")
    primitive_digest = digests["primitive_probability"]
    if expected_primitive_sha256 is not None and primitive_digest != expected_primitive_sha256:
        raise ValueError("NVOS strict PCA40 primitive probability differs")
    return tensors["primitive_probability"]


@torch.inference_mode()
def build(args: argparse.Namespace) -> dict[str, object]:
    registration_path = Path(args.experiment_registration).resolve()
    registration_sha256 = sha256_file(registration_path)
    if registration_sha256 != REGISTRATION_SHA256:
        raise ValueError("PCA40 experiment registration differs")
    report_path = Path(args.cache_report).resolve()
    report_sha256 = sha256_file(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    authority = PromptResponsibilityAuthority.from_dict(report["authority"])
    if authority.scene_id != args.scene_id:
        raise ValueError("responsibility authority scene differs")
    cache_path = Path(args.cache).resolve()
    if cache_path != Path(str(report["artifact_path"])).resolve():
        raise ValueError("responsibility cache path differs from receipt")
    cache = load_prompt_responsibility_cache(
        cache_path,
        expected_authority=authority,
        expected_file_sha256=str(report["file_sha256"]),
    )
    if cache.tensor_bundle_sha256 != str(report["tensor_bundle_sha256"]):
        raise ValueError("responsibility tensor bundle differs from receipt")
    positive_path = Path(args.positive_scribble).resolve()
    negative_path = Path(args.negative_scribble).resolve()
    if (
        sha256_file(positive_path) != authority.source_sha256["positive_scribble"]
        or sha256_file(negative_path) != authority.source_sha256["negative_scribble"]
    ):
        raise ValueError("official scribbles differ from responsibility authority")
    positive_pixels = torch.from_numpy(load_ground_truth_mask(positive_path))
    negative_pixels = torch.from_numpy(load_ground_truth_mask(negative_path))
    native_shape = (int(authority.height), int(authority.width))
    if tuple(positive_pixels.shape) != native_shape or tuple(negative_pixels.shape) != native_shape:
        raise ValueError("official scribbles differ from responsibility native shape")
    if bool((positive_pixels & negative_pixels).any()):
        raise ValueError("official positive and negative scribbles overlap")
    positive_mass = cache.adjoint(positive_pixels).weighted_sum
    negative_mass = cache.adjoint(negative_pixels).weighted_sum
    visible_mass = cache.visible_mass
    del cache
    signed = torch.zeros_like(visible_mass)
    visible = visible_mass > 0
    signed[visible] = (positive_mass[visible] - negative_mass[visible]) / visible_mass[visible]
    if bool(((signed < -1.0 - 1e-12) | (signed > 1.0 + 1e-12)).any()):
        raise ValueError("exact signed prompt evidence exceeds its probability bound")
    signed = signed.clamp(-1.0, 1.0)
    positive_observed = positive_mass > 0
    negative_observed = negative_mass > 0
    positive_exclusive = positive_observed & ~negative_observed
    negative_exclusive = negative_observed & ~positive_observed
    conflict = positive_observed & negative_observed

    graph_path = Path(args.support_graph).resolve()
    graph_sha256 = sha256_file(graph_path)
    graph_payload = torch.load(graph_path, map_location="cpu", weights_only=True)
    global_rows = torch.as_tensor(graph_payload["global_rows"]).long().cpu()
    graph_xyz = torch.as_tensor(graph_payload["xyz"]).float().cpu()
    if int(graph_payload["num_global_rows"]) != authority.num_gaussians:
        raise ValueError("support graph global row count differs")
    relation_path = Path(args.relation_cache).resolve()
    relation_sha256 = sha256_file(relation_path)
    relation_receipt_path = Path(str(relation_path) + ".json")
    relation_receipt_sha256 = sha256_file(relation_receipt_path)
    relation_receipt = json.loads(relation_receipt_path.read_text(encoding="utf-8"))
    if (
        relation_receipt.get("output_sha256") != relation_sha256
        or relation_receipt.get("scene_id") != args.scene_id
    ):
        raise ValueError("PCA40 relation receipt differs")
    relation = load_query_diffusion_relation_cache(
        relation_path,
        expected_scene_id=args.scene_id,
        expected_global_rows=global_rows,
        expected_num_global_rows=authority.num_gaussians,
        expected_source_xyz_sha256=authority.geometry_xyz_sha256,
        expected_registration_sha256=RELATION_REGISTRATION_SHA256,
        expected_capability_sidecar_sha256=str(
            relation_receipt["metadata"]["capability_sidecar_sha256"]
        ),
        expected_field_checkpoint_sha256=str(
            relation_receipt["metadata"]["field_checkpoint_sha256"]
        ),
    )
    relation_feature_sha256 = tensor_sha256(relation.relation_features)
    if relation_receipt.get("relation_feature_sha256") != relation_feature_sha256:
        raise ValueError("PCA40 relation feature receipt differs")
    if relation_receipt.get("source_feature_sha256") != relation.source_feature_sha256:
        raise ValueError("PCA40 relation source feature receipt differs")

    knn_path = Path(args.knn_cache).resolve()
    knn_sha256 = sha256_file(knn_path)
    knn = load_query_diffusion_knn_cache(
        knn_path,
        expected_global_rows=global_rows,
        expected_xyz=graph_xyz,
        expected_source_graph_sha256=graph_sha256,
        expected_num_neighbors=200,
    )
    if knn.metadata.get("experiment_registration_sha256") != registration_sha256:
        raise ValueError("PCA40 kNN registration differs")
    config = QueryConditionedDiffusionConfig(
        kernel="ludvig_release_compat",
        feature_bandwidth=2.0,
        regularizer_bandwidth=4.0,
        logistic_c=0.01,
        logistic_fit_population="signed_nonzero",
        iterations=100,
        edge_binarize_threshold=1e-5,
        distance_chunk_size=int(args.distance_chunk_size),
    )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("NVOS PCA40 query-conditioned support requires CUDA")
    local_signed = signed[global_rows].float().contiguous()
    local_weight = visible_mass[global_rows].float().contiguous()
    support_local, compatibility_local = compute_query_conditioned_support(
        relation.relation_features.to(device),
        knn.neighbor_indices.to(device),
        local_signed.to(device),
        local_weight,
        config=config,
    )
    support_local = support_local.float().cpu().clamp(0.0, 1.0).contiguous()
    compatibility_local = compatibility_local.float().cpu().clamp(0.0, 1.0).contiguous()
    primitive = torch.zeros(authority.num_gaussians, dtype=torch.float32)
    compatibility = torch.zeros_like(primitive)
    primitive[global_rows] = support_local
    compatibility[global_rows] = compatibility_local
    primitive[positive_exclusive] = 1.0
    primitive[negative_exclusive] = 0.0
    tensors = {
        "primitive_probability": primitive.contiguous(),
        "query_compatibility": compatibility.contiguous(),
        "signed_reference_evidence": signed.float().contiguous(),
        "reference_weight": visible_mass.float().contiguous(),
        "capability_valid": torch.zeros(authority.num_gaussians, dtype=torch.bool),
        "positive_exclusive": positive_exclusive.contiguous(),
        "negative_exclusive": negative_exclusive.contiguous(),
        "conflict": conflict.contiguous(),
    }
    tensors["capability_valid"][global_rows] = True
    digests = {name: tensor_sha256(value) for name, value in sorted(tensors.items())}
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "scene_id": args.scene_id,
        "method_contract": METHOD_CONTRACT,
        "method_contract_sha256": _json_sha256(METHOD_CONTRACT),
        "experiment_registration_path": str(registration_path),
        "experiment_registration_sha256": registration_sha256,
        "responsibility_report_path": str(report_path),
        "responsibility_report_sha256": report_sha256,
        "responsibility_file_sha256": str(report["file_sha256"]),
        "responsibility_authority_sha256": authority.digest,
        "responsibility_tensor_bundle_sha256": str(report["tensor_bundle_sha256"]),
        "relation_cache_path": str(relation_path),
        "relation_cache_sha256": relation_sha256,
        "relation_cache_receipt_sha256": relation_receipt_sha256,
        "relation_feature_sha256": relation_feature_sha256,
        "source_feature_sha256": relation.source_feature_sha256,
        "source_xyz_sha256": relation.source_xyz_sha256,
        "capability_sidecar_sha256": str(relation.metadata["capability_sidecar_sha256"]),
        "field_checkpoint_sha256": str(relation.metadata["field_checkpoint_sha256"]),
        "support_graph_path": str(graph_path),
        "support_graph_sha256": graph_sha256,
        "knn_cache_path": str(knn_path),
        "knn_cache_sha256": knn_sha256,
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
    validate_nvos_strict_pca40_support_payload(
        frozen,
        authority=authority,
        expected_responsibility_file_sha256=str(report["file_sha256"]),
        expected_primitive_sha256=digests["primitive_probability"],
    )
    if sha256_file(output) != output_sha256:
        raise ValueError("NVOS strict PCA40 support changed across freeze and reload")
    report_payload = {
        "scene_id": args.scene_id,
        "artifact_type": ARTIFACT_TYPE,
        "method_contract": METHOD_CONTRACT,
        "method_contract_sha256": artifact["method_contract_sha256"],
        "experiment_registration_sha256": registration_sha256,
        "output": str(output),
        "output_sha256": output_sha256,
        "primitive_probability_sha256": digests["primitive_probability"],
        "relation_cache_sha256": relation_sha256,
        "relation_feature_sha256": relation_feature_sha256,
        "source_feature_sha256": relation.source_feature_sha256,
        "diffusion_config": asdict(config),
        "num_global_rows": authority.num_gaussians,
        "capability_valid_rows": int(global_rows.numel()),
        "effective_knn_columns": knn.effective_k,
        "signed_observed_valid_rows": int((local_signed != 0).sum()),
        "positive_exclusive_rows": int(positive_exclusive.sum()),
        "negative_exclusive_rows": int(negative_exclusive.sum()),
        "conflict_rows": int(conflict.sum()),
        "support_fraction_at_0_5": float((primitive >= 0.5).double().mean()),
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
    }
    Path(str(output) + ".json").write_text(
        json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--experiment-registration", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--cache-report", required=True)
    parser.add_argument("--positive-scribble", required=True)
    parser.add_argument("--negative-scribble", required=True)
    parser.add_argument("--relation-cache", required=True)
    parser.add_argument("--support-graph", required=True)
    parser.add_argument("--knn-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--distance-chunk-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
