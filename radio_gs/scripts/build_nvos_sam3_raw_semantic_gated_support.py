#!/usr/bin/env python3
"""Build the registered posthoc raw-semantic-gated SAM3 NVOS support."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from radio_gs.evaluation.promptable_segmentation import load_ground_truth_mask
from radio_gs.interfaces.capability_cache import load_canonical_capability_bank
from radio_gs.interfaces.prompt_responsibility_cache import (
    PromptResponsibilityAuthority,
    load_prompt_responsibility_cache,
    sha256_file,
    tensor_sha256,
)
from radio_gs.interfaces.query_diffusion_cache import load_query_diffusion_knn_cache
from radio_gs.querying.query_conditioned_diffusion import (
    QueryConditionedDiffusionConfig,
    normalize_node_features,
    rbf_knn_feature_similarity,
    run_query_conditioned_diffusion,
    weighted_logistic_query_compatibility,
)
from radio_gs.querying.sam3_reference_completion import (
    entropy_reliability_soft_observation,
)
from radio_gs.scripts.build_canonical_support_graph import deterministic_feature_hash
from radio_gs.scripts.build_nvos_sam3_reference_support import (
    _load_reference_completion,
    bank_metadata_field_hash,
)
from radio_gs.scripts.build_nvos_sam3_soft_reliability_support import (
    _float_rows_sha256,
    _json_sha256,
    _validate_pixel_tensors,
)


ARTIFACT_TYPE = "nvos_sam3_raw_semantic_gated_query_conditioned_support"
SCHEMA_VERSION = 1
REGISTRATION_SHA256 = (
    "c0942e8cb3b04754e3fd5c0b9f6c490d32c0174a9b2dfcfeff12e3a26b4f000a"
)
METHOD_CONTRACT = {
    "claim": "posthoc_registered_followup_not_independent_validation",
    "track": "raw_signed_semantic_compatibility_gates_soft_SAM3_completion",
    "phase1": "reuse_frozen_ten_binary_mask_mean_q_without_new_SAM3_call",
    "reliability": "c(q)=1-H2(q)/ln2_parameter_free_float64_then_float32",
    "soft_positive_observation": "q_times_c(q)_raw_positive_overwrite_1_raw_negative_overwrite_0",
    "raw_semantic_evidence": "signed_exact_W_transpose_raw_positive_minus_raw_negative_divided_by_visible_mass",
    "raw_semantic_weight": "W_transpose_raw_positive_plus_W_transpose_raw_negative",
    "classifier": "balanced_logistic_C_0.01_raw_signed_nonzero_rows_weighted_only_by_raw_semantic_weight",
    "compatibility": "P_raw_predict_proba_power_one_over_regularizer_bandwidth_4",
    "compatibility_refit_from_completed_evidence": "forbidden",
    "completed_positive_probability": "W_transpose_soft_positive_divided_by_visible_mass",
    "initial_unary": "completed_positive_probability_times_P_raw_then_raw_hard_anchors",
    "relation_feature": "query_independent_C_RADIO_DINO_signed_multiplicative_hash_256",
    "relation_feature_role": "diagnostic_not_native_DINO_or_exact_LUDVIG_feature_match",
    "topology": "exact_euclidean_num_neighbors_200_plus_retained_self_equals_K201",
    "feature_bandwidth": 2.0,
    "regularizer_bandwidth": 4.0,
    "query_edge_gate": "sqrt(P_raw_i_times_P_raw_j)",
    "diffusion_kernel": "released_N_by_K_slotwise_normalization_then_symmetrize_binarize_1e-5",
    "iterations": 100,
    "direct_anchor_rule": "raw_positive_exclusive_rows_set_1_raw_negative_exclusive_rows_set_0_before_and_after_diffusion",
    "invalid_capability_rows": "zero_except_raw_direct_exclusive_anchors",
    "threshold": 0.5,
    "connected_selection": "none",
    "new_scalar": "forbidden",
    "target_dependent_tuning": False,
}
TENSOR_KEYS = {
    "primitive_probability",
    "raw_query_compatibility",
    "raw_signed_reference_evidence",
    "raw_reference_weight",
    "completed_positive_probability",
    "semantic_completed_initial_unary",
    "completed_positive_mass",
    "raw_positive_mass",
    "raw_negative_mass",
    "visible_mass",
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
    "reference_completion_path",
    "reference_completion_sha256",
    "reference_completion_receipt_path",
    "reference_completion_receipt_sha256",
    "reference_completion_tensor_bundle_sha256",
    "source_rgb_path",
    "source_rgb_sha256",
    "capability_cache_path",
    "capability_sidecar_sha256",
    "field_checkpoint_sha256",
    "support_graph_path",
    "support_graph_sha256",
    "knn_cache_path",
    "knn_cache_sha256",
    "feature_hash_sha256",
    "pixel_tensors",
    "pixel_tensor_sha256",
    "pixel_tensor_bundle_sha256",
    "tensors",
    "tensor_sha256",
    "tensor_bundle_sha256",
    "target_rgb_opened",
    "target_mask_opened",
    "target_metric_computed",
}


def form_semantic_completed_initial_unary(
    completed_positive_probability: torch.Tensor,
    raw_query_compatibility: torch.Tensor,
    capability_valid: torch.Tensor,
    positive_exclusive: torch.Tensor,
    negative_exclusive: torch.Tensor,
) -> torch.Tensor:
    """Apply the registered parameter-free primitive semantic gate and anchors."""

    completed = torch.as_tensor(completed_positive_probability).detach().float().cpu()
    compatibility = torch.as_tensor(raw_query_compatibility).detach().float().cpu()
    valid = torch.as_tensor(capability_valid).detach().bool().cpu()
    positive = torch.as_tensor(positive_exclusive).detach().bool().cpu()
    negative = torch.as_tensor(negative_exclusive).detach().bool().cpu()
    if not (
        completed.ndim == 1
        and compatibility.shape == completed.shape
        and valid.shape == completed.shape
        and positive.shape == completed.shape
        and negative.shape == completed.shape
    ):
        raise ValueError("semantic completed unary inputs do not align")
    if (
        not bool(torch.isfinite(completed).all())
        or not bool(torch.isfinite(compatibility).all())
        or bool(((completed < 0) | (completed > 1)).any())
        or bool(((compatibility < 0) | (compatibility > 1)).any())
        or bool((positive & negative).any())
    ):
        raise ValueError("semantic completed unary inputs are invalid")
    output = torch.zeros_like(completed)
    output[valid] = completed[valid] * compatibility[valid]
    output[positive] = 1.0
    output[negative] = 0.0
    return output.contiguous()


def validate_nvos_sam3_raw_semantic_gated_payload(
    payload: object,
    *,
    authority: PromptResponsibilityAuthority,
    expected_responsibility_file_sha256: str,
    expected_completion_sha256: str,
    expected_primitive_sha256: str | None = None,
) -> torch.Tensor:
    if not isinstance(payload, dict) or set(payload) != ARTIFACT_KEYS:
        raise ValueError("NVOS raw-semantic-gated artifact schema differs")
    if (
        payload["schema_version"] != SCHEMA_VERSION
        or payload["artifact_type"] != ARTIFACT_TYPE
        or payload["scene_id"] != authority.scene_id
        or payload["method_contract"] != METHOD_CONTRACT
        or payload["method_contract_sha256"] != _json_sha256(METHOD_CONTRACT)
        or payload["experiment_registration_sha256"] != REGISTRATION_SHA256
        or payload["responsibility_file_sha256"]
        != expected_responsibility_file_sha256
        or payload["responsibility_authority_sha256"] != authority.digest
        or payload["reference_completion_sha256"] != expected_completion_sha256
        or payload["target_rgb_opened"] is not False
        or payload["target_mask_opened"] is not False
        or payload["target_metric_computed"] is not False
    ):
        raise ValueError("NVOS raw-semantic-gated method or authority differs")
    pixel_digests = _validate_pixel_tensors(
        payload["pixel_tensors"], authority=authority
    )
    if (
        payload["pixel_tensor_sha256"] != pixel_digests
        or payload["pixel_tensor_bundle_sha256"] != _json_sha256(pixel_digests)
    ):
        raise ValueError("NVOS raw-semantic-gated pixel tensor digests differ")
    tensors = payload["tensors"]
    if not isinstance(tensors, dict) or set(tensors) != TENSOR_KEYS:
        raise ValueError("NVOS raw-semantic-gated tensor schema differs")
    count = int(authority.num_gaussians)
    float_keys = TENSOR_KEYS - {
        "capability_valid", "positive_exclusive", "negative_exclusive", "conflict"
    }
    for name in float_keys:
        value = tensors[name]
        if (
            not torch.is_tensor(value)
            or value.device.type != "cpu"
            or value.dtype != torch.float32
            or tuple(value.shape) != (count,)
            or not value.is_contiguous()
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"NVOS raw-semantic-gated tensor {name} is malformed")
    probability_keys = {
        "primitive_probability",
        "raw_query_compatibility",
        "completed_positive_probability",
        "semantic_completed_initial_unary",
    }
    for name in probability_keys:
        if bool(((tensors[name] < 0) | (tensors[name] > 1)).any()):
            raise ValueError(f"NVOS raw-semantic-gated {name} is outside [0,1]")
    signed = tensors["raw_signed_reference_evidence"]
    if bool(((signed < -1) | (signed > 1)).any()):
        raise ValueError("NVOS raw signed evidence is outside [-1,1]")
    for name in (
        "raw_reference_weight", "completed_positive_mass", "raw_positive_mass",
        "raw_negative_mass", "visible_mass",
    ):
        if bool((tensors[name] < 0).any()):
            raise ValueError(f"NVOS raw-semantic-gated {name} is negative")
    for name in (
        "capability_valid", "positive_exclusive", "negative_exclusive", "conflict"
    ):
        value = tensors[name]
        if (
            not torch.is_tensor(value)
            or value.device.type != "cpu"
            or value.dtype != torch.bool
            or tuple(value.shape) != (count,)
            or not value.is_contiguous()
        ):
            raise ValueError(f"NVOS raw-semantic-gated mask {name} is malformed")
    positive = tensors["positive_exclusive"]
    negative = tensors["negative_exclusive"]
    conflict = tensors["conflict"]
    if bool(
        (positive & negative).any()
        or (positive & conflict).any()
        or (negative & conflict).any()
    ):
        raise ValueError("NVOS raw semantic anchor partitions overlap")
    valid = tensors["capability_valid"]
    if bool((tensors["raw_query_compatibility"][~valid] != 0).any()):
        raise ValueError("invalid capability rows have raw compatibility")
    recomputed_initial = form_semantic_completed_initial_unary(
        tensors["completed_positive_probability"],
        tensors["raw_query_compatibility"],
        valid,
        positive,
        negative,
    )
    if not torch.equal(tensors["semantic_completed_initial_unary"], recomputed_initial):
        raise ValueError("stored semantic completed unary differs from formula")
    if not bool((tensors["primitive_probability"][positive] == 1).all()):
        raise ValueError("NVOS raw semantic positive anchors are not one")
    if not bool((tensors["primitive_probability"][negative] == 0).all()):
        raise ValueError("NVOS raw semantic negative anchors are not zero")
    if bool(
        (tensors["primitive_probability"][~valid & ~positive & ~negative] != 0).any()
    ):
        raise ValueError("invalid non-anchor rows have primitive support")
    expected_weight = tensors["raw_positive_mass"] + tensors["raw_negative_mass"]
    if not torch.allclose(
        tensors["raw_reference_weight"], expected_weight, rtol=1e-6, atol=1e-7
    ):
        raise ValueError("raw semantic weight differs from raw signed masses")
    visible = tensors["visible_mass"] > 0
    expected_completed = torch.zeros(count, dtype=torch.float32)
    expected_completed[visible] = (
        tensors["completed_positive_mass"][visible]
        / tensors["visible_mass"][visible]
    )
    if not torch.allclose(
        tensors["completed_positive_probability"],
        expected_completed.clamp(0, 1),
        rtol=1e-6,
        atol=1e-7,
    ):
        raise ValueError("completed positive probability differs from exact masses")
    digests = {name: tensor_sha256(value) for name, value in sorted(tensors.items())}
    if (
        payload["tensor_sha256"] != digests
        or payload["tensor_bundle_sha256"] != _json_sha256(digests)
    ):
        raise ValueError("NVOS raw-semantic-gated tensor digests differ")
    if (
        expected_primitive_sha256 is not None
        and digests["primitive_probability"] != expected_primitive_sha256
    ):
        raise ValueError("NVOS raw-semantic-gated primitive differs")
    return tensors["primitive_probability"]


@torch.inference_mode()
def build(args: argparse.Namespace) -> dict[str, object]:
    registration_path = Path(args.experiment_registration).resolve()
    registration_sha256 = sha256_file(registration_path)
    if registration_sha256 != REGISTRATION_SHA256:
        raise ValueError("raw-semantic-gated registration differs")
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
        raise ValueError("responsibility tensor bundle differs")

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
    if (
        tuple(positive_pixels.shape) != native_shape
        or tuple(negative_pixels.shape) != native_shape
        or bool((positive_pixels & negative_pixels).any())
    ):
        raise ValueError("official signed scribbles are malformed")

    completion_path = Path(args.reference_completion).resolve()
    completion_receipt_path = Path(args.completion_receipt).resolve()
    _, completion_payload, completion_sha256, receipt_sha256 = (
        _load_reference_completion(
            completion_path,
            completion_receipt_path,
            authority=authority,
            positive_pixels=positive_pixels,
            negative_pixels=negative_pixels,
        )
    )
    aggregate = completion_payload["tensors"]["aggregate_probability"].float().contiguous()
    reliability_numpy, observation_numpy = entropy_reliability_soft_observation(
        aggregate.numpy(), positive_pixels.numpy(), negative_pixels.numpy()
    )
    pixel_tensors = {
        "aggregate_probability": aggregate,
        "binary_entropy_reliability": torch.from_numpy(reliability_numpy).contiguous(),
        "soft_positive_observation": torch.from_numpy(observation_numpy).contiguous(),
        "raw_positive": positive_pixels.contiguous(),
        "raw_negative": negative_pixels.contiguous(),
    }
    pixel_digests = _validate_pixel_tensors(pixel_tensors, authority=authority)

    completed_mass = cache.adjoint(
        pixel_tensors["soft_positive_observation"]
    ).weighted_sum.float().contiguous()
    raw_positive_mass = cache.adjoint(positive_pixels).weighted_sum.float().contiguous()
    raw_negative_mass = cache.adjoint(negative_pixels).weighted_sum.float().contiguous()
    visible_mass = cache.visible_mass.float().contiguous()
    del cache
    visible = visible_mass > 0
    raw_signed = torch.zeros_like(visible_mass)
    raw_signed[visible] = (
        raw_positive_mass[visible] - raw_negative_mass[visible]
    ) / visible_mass[visible]
    if bool(((raw_signed < -1.0 - 2e-6) | (raw_signed > 1.0 + 2e-6)).any()):
        raise ValueError("raw signed evidence exceeds probability bounds")
    raw_signed = raw_signed.clamp(-1, 1).contiguous()
    raw_weight = (raw_positive_mass + raw_negative_mass).contiguous()
    completed_probability = torch.zeros_like(visible_mass)
    completed_probability[visible] = completed_mass[visible] / visible_mass[visible]
    if bool(
        ((completed_probability < -2e-6) | (completed_probability > 1 + 2e-6)).any()
    ):
        raise ValueError("completed positive probability exceeds bounds")
    completed_probability = completed_probability.clamp(0, 1).contiguous()
    positive_observed = raw_positive_mass > 0
    negative_observed = raw_negative_mass > 0
    positive_exclusive = positive_observed & ~negative_observed
    negative_exclusive = negative_observed & ~positive_observed
    conflict = positive_observed & negative_observed

    capability_path = Path(args.capability_cache).resolve()
    capability_sidecar = Path(str(capability_path) + ".json")
    bank = load_canonical_capability_bank(capability_path)
    if bank.num_gaussians != authority.num_gaussians:
        raise ValueError("capability and responsibility row counts differ")
    if _float_rows_sha256(bank.xyz) != authority.geometry_xyz_sha256:
        raise ValueError("capability and responsibility geometry differ")
    graph_path = Path(args.support_graph).resolve()
    graph_sha256 = sha256_file(graph_path)
    graph_payload = torch.load(graph_path, map_location="cpu", weights_only=True)
    global_rows = torch.as_tensor(graph_payload["global_rows"]).long().cpu()
    if not torch.equal(global_rows, bank.global_rows):
        raise ValueError("support graph and capability rows differ")
    if int(graph_payload["num_global_rows"]) != authority.num_gaussians:
        raise ValueError("support graph global row count differs")
    if not torch.equal(
        torch.as_tensor(graph_payload["xyz"]).float().cpu(), bank.xyz[global_rows]
    ):
        raise ValueError("support graph and capability xyz differ")
    knn_path = Path(args.knn_cache).resolve()
    knn_sha256 = sha256_file(knn_path)
    knn = load_query_diffusion_knn_cache(
        knn_path,
        expected_global_rows=global_rows,
        expected_xyz=bank.xyz[global_rows],
        expected_source_graph_sha256=graph_sha256,
        expected_num_neighbors=200,
    )
    appearance = bank.valid_feature_banks()["appearance"]
    hashed = deterministic_feature_hash(
        appearance, 256, batch_size=int(args.hash_batch_size)
    ).contiguous()
    feature_hash_sha256 = tensor_sha256(hashed)
    del appearance, bank, graph_payload

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
        raise ValueError("raw-semantic-gated support requires CUDA")
    normalized = normalize_node_features(hashed.to(device), eps=config.eps)
    local_raw_signed = raw_signed[global_rows].float().contiguous()
    local_raw_weight = raw_weight[global_rows].float().contiguous()
    compatibility_local = weighted_logistic_query_compatibility(
        normalized,
        local_raw_signed,
        local_raw_weight,
        logistic_c=config.logistic_c,
        regularizer_bandwidth=config.regularizer_bandwidth,
        fit_population="signed_nonzero",
    ).to(device)
    capability_valid = torch.zeros(authority.num_gaussians, dtype=torch.bool)
    capability_valid[global_rows] = True
    compatibility = torch.zeros(authority.num_gaussians, dtype=torch.float32)
    compatibility[global_rows] = compatibility_local.float().cpu().clamp(0, 1)
    initial = form_semantic_completed_initial_unary(
        completed_probability,
        compatibility,
        capability_valid,
        positive_exclusive,
        negative_exclusive,
    )
    local_initial = initial[global_rows].to(device)
    similarities = rbf_knn_feature_similarity(
        normalized,
        knn.neighbor_indices.to(device),
        feature_bandwidth=config.feature_bandwidth,
        positive_reference_mask=local_initial > 0,
        eps=config.eps,
        distance_chunk_size=config.distance_chunk_size,
    )
    support_local = run_query_conditioned_diffusion(
        local_initial[:, None],
        knn.neighbor_indices.to(device),
        similarities,
        compatibility_local,
        config=config,
    ).squeeze(1)
    support_local = support_local.float().cpu().clamp(0, 1).contiguous()
    primitive = torch.zeros(authority.num_gaussians, dtype=torch.float32)
    primitive[global_rows] = support_local
    primitive[positive_exclusive] = 1.0
    primitive[negative_exclusive] = 0.0
    tensors = {
        "primitive_probability": primitive.contiguous(),
        "raw_query_compatibility": compatibility.contiguous(),
        "raw_signed_reference_evidence": raw_signed.contiguous(),
        "raw_reference_weight": raw_weight.contiguous(),
        "completed_positive_probability": completed_probability.contiguous(),
        "semantic_completed_initial_unary": initial.contiguous(),
        "completed_positive_mass": completed_mass.contiguous(),
        "raw_positive_mass": raw_positive_mass.contiguous(),
        "raw_negative_mass": raw_negative_mass.contiguous(),
        "visible_mass": visible_mass.contiguous(),
        "capability_valid": capability_valid.contiguous(),
        "positive_exclusive": positive_exclusive.contiguous(),
        "negative_exclusive": negative_exclusive.contiguous(),
        "conflict": conflict.contiguous(),
    }
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
        "reference_completion_path": str(completion_path),
        "reference_completion_sha256": completion_sha256,
        "reference_completion_receipt_path": str(completion_receipt_path),
        "reference_completion_receipt_sha256": receipt_sha256,
        "reference_completion_tensor_bundle_sha256": completion_payload[
            "tensor_bundle_sha256"
        ],
        "source_rgb_path": completion_payload["authority"]["source_rgb_path"],
        "source_rgb_sha256": completion_payload["authority"]["source_rgb_sha256"],
        "capability_cache_path": str(capability_path),
        "capability_sidecar_sha256": sha256_file(capability_sidecar),
        "field_checkpoint_sha256": bank_metadata_field_hash(capability_path),
        "support_graph_path": str(graph_path),
        "support_graph_sha256": graph_sha256,
        "knn_cache_path": str(knn_path),
        "knn_cache_sha256": knn_sha256,
        "feature_hash_sha256": feature_hash_sha256,
        "pixel_tensors": pixel_tensors,
        "pixel_tensor_sha256": pixel_digests,
        "pixel_tensor_bundle_sha256": _json_sha256(pixel_digests),
        "tensors": tensors,
        "tensor_sha256": digests,
        "tensor_bundle_sha256": _json_sha256(digests),
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    torch.save(artifact, output)
    output_sha256 = sha256_file(output)
    frozen = torch.load(output, map_location="cpu", weights_only=True)
    validate_nvos_sam3_raw_semantic_gated_payload(
        frozen,
        authority=authority,
        expected_responsibility_file_sha256=str(report["file_sha256"]),
        expected_completion_sha256=completion_sha256,
        expected_primitive_sha256=digests["primitive_probability"],
    )
    if sha256_file(output) != output_sha256:
        raise ValueError("raw-semantic-gated support changed across freeze and reload")
    report_payload = {
        "scene_id": args.scene_id,
        "artifact_type": ARTIFACT_TYPE,
        "method_contract": METHOD_CONTRACT,
        "method_contract_sha256": artifact["method_contract_sha256"],
        "experiment_registration_sha256": registration_sha256,
        "reference_completion_sha256": completion_sha256,
        "reference_completion_receipt_sha256": receipt_sha256,
        "output": str(output),
        "output_sha256": output_sha256,
        "primitive_probability_sha256": digests["primitive_probability"],
        "raw_query_compatibility_sha256": digests["raw_query_compatibility"],
        "semantic_completed_initial_unary_sha256": digests[
            "semantic_completed_initial_unary"
        ],
        "pixel_tensor_sha256": pixel_digests,
        "pixel_tensor_bundle_sha256": _json_sha256(pixel_digests),
        "feature_hash_sha256": feature_hash_sha256,
        "diffusion_config": asdict(config),
        "num_global_rows": authority.num_gaussians,
        "capability_valid_rows": int(global_rows.numel()),
        "effective_knn_columns": knn.effective_k,
        "raw_signed_observed_valid_rows": int((local_raw_signed != 0).sum()),
        "positive_exclusive_rows": int(positive_exclusive.sum()),
        "negative_exclusive_rows": int(negative_exclusive.sum()),
        "conflict_rows": int(conflict.sum()),
        "mean_raw_query_compatibility_valid": float(
            compatibility[global_rows].double().mean()
        ),
        "mean_completed_positive_probability_valid": float(
            completed_probability[global_rows].double().mean()
        ),
        "mean_semantic_initial_unary_valid": float(
            initial[global_rows].double().mean()
        ),
        "nonzero_semantic_initial_fraction_valid": float(
            (initial[global_rows] > 0).double().mean()
        ),
        "support_fraction_at_0_5": float((primitive >= 0.5).double().mean()),
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--experiment-registration", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--cache-report", required=True)
    parser.add_argument("--reference-completion", required=True)
    parser.add_argument("--completion-receipt", required=True)
    parser.add_argument("--positive-scribble", required=True)
    parser.add_argument("--negative-scribble", required=True)
    parser.add_argument("--capability-cache", required=True)
    parser.add_argument("--support-graph", required=True)
    parser.add_argument("--knn-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hash-batch-size", type=int, default=8192)
    parser.add_argument("--distance-chunk-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
