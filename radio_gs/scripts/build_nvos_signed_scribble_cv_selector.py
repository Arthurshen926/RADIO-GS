#!/usr/bin/env python3
"""Select strong unary vs fixed hash256 diffusion using reference-only OOF CV."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from radio_gs.interfaces.capability_cache import load_canonical_capability_bank
from radio_gs.interfaces.prompt_responsibility_cache import (
    PromptResponsibilityAuthority,
    sha256_file,
    tensor_sha256,
)
from radio_gs.interfaces.query_diffusion_cache import load_query_diffusion_knn_cache
from radio_gs.querying.query_conditioned_diffusion import (
    QueryConditionedDiffusionConfig,
    knn_feature_distances,
    normalize_node_features,
    rbf_similarity_from_distances,
    run_query_conditioned_diffusion,
    weighted_logistic_query_compatibility,
)
from radio_gs.querying.query_specific_propagation_cv import (
    ACTION_HASH256_DIFFUSION,
    ACTION_STRONG_UNARY,
    audit_signed_cv_population,
    responsibility_balanced_log_loss,
    responsibility_weighted_auc,
    run_signed_scribble_cross_validation,
    select_registered_action,
    stable_primitive_folds,
)
from radio_gs.scripts.build_canonical_support_graph import deterministic_feature_hash
from radio_gs.scripts.build_nvos_strict_query_conditioned_support import (
    _float_rows_sha256,
    validate_nvos_strict_support_payload,
)


ARTIFACT_TYPE = "nvos_strict_signed_scribble_cv_selected_propagation_support"
SCHEMA_VERSION = 1
REGISTRATION_SHA256 = (
    "224984d1afe764b8b13d568b5f63504ca4b8570dec2c76635de2b9a7da925874"
)
METHOD_CONTRACT = {
    "track": "strict_raw_positive_negative_scribble",
    "selection": "deterministic_3fold_signed_scribble_OOF_reference_only",
    "candidate_actions": [ACTION_STRONG_UNARY, ACTION_HASH256_DIFFUSION],
    "primary_metric": "responsibility_weighted_balanced_binary_log_loss",
    "secondary_metric": "responsibility_weighted_ROC_AUC",
    "tie_break": "strong_unary",
    "fold_key": "SplitMix64_global_primitive_row_modulo_3",
    "minimum_class_rows_per_training_or_heldout_fold": 32,
    "heldout_evidence": "exact_zero_before_fit_bandwidth_gate_and_diffusion",
    "classifier": "balanced_logistic_C_0.01_signed_nonzero_rows_weighted_by_W_transpose_one",
    "strong_unary": "full_refit_logistic_probability_power_one_quarter",
    "relation_feature": "query_independent_C_RADIO_DINO_signed_multiplicative_hash_256",
    "relation_feature_role": "diagnostic_not_native_DINO_or_exact_LUDVIG_feature_match",
    "topology": "exact_euclidean_num_neighbors_200_plus_retained_self_equals_K201",
    "feature_bandwidth": 2.0,
    "regularizer_bandwidth": 4.0,
    "iterations": 100,
    "edge_binarization_threshold": 1e-5,
    "direct_anchor_rule": "exact_positive_exclusive_rows_set_1_negative_exclusive_rows_set_0",
    "invalid_capability_rows": "zero_except_exact_direct_exclusive_anchors",
    "threshold": 0.5,
    "connected_selection": "none",
    "target_dependent_tuning": False,
}
CV_CONTRACT = {
    "num_folds": 3,
    "minimum_class_rows": 32,
    "metric_round_decimals": 12,
    "probability_epsilon": 1e-7,
    "selection_order": "min_balanced_logloss_max_weighted_auc_strong_unary_tie",
}
BASE_TENSOR_KEYS = {
    "primitive_probability",
    "query_compatibility",
    "signed_reference_evidence",
    "reference_weight",
    "capability_valid",
    "positive_exclusive",
    "negative_exclusive",
    "conflict",
    "cv_fold_ids",
    "cv_oof_strong_unary",
    "cv_oof_hash256_diffusion",
}
ARTIFACT_KEYS = {
    "schema_version",
    "artifact_type",
    "scene_id",
    "method_contract",
    "method_contract_sha256",
    "cv_contract",
    "cv_contract_sha256",
    "selected_action",
    "cv_metrics",
    "cv_fold_reports",
    "experiment_registration_path",
    "experiment_registration_sha256",
    "base_selector_path",
    "base_selector_sha256",
    "base_selector_method_contract_sha256",
    "base_primitive_probability_sha256",
    "base_query_compatibility_sha256",
    "responsibility_report_path",
    "responsibility_report_sha256",
    "responsibility_file_sha256",
    "responsibility_authority_sha256",
    "responsibility_tensor_bundle_sha256",
    "capability_cache_path",
    "capability_sidecar_sha256",
    "field_checkpoint_sha256",
    "support_graph_path",
    "support_graph_sha256",
    "knn_cache_path",
    "knn_cache_sha256",
    "feature_hash_sha256",
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


def _metrics_from_payload_tensors(tensors: dict[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    observed = tensors["capability_valid"] & (tensors["signed_reference_evidence"] != 0)
    labels = tensors["signed_reference_evidence"][observed] > 0
    weights = tensors["reference_weight"][observed]
    predictions = {
        ACTION_STRONG_UNARY: tensors["cv_oof_strong_unary"][observed],
        ACTION_HASH256_DIFFUSION: tensors["cv_oof_hash256_diffusion"][observed],
    }
    return {
        action: {
            "responsibility_balanced_log_loss": responsibility_balanced_log_loss(
                labels,
                predictions[action],
                weights,
                probability_epsilon=float(CV_CONTRACT["probability_epsilon"]),
            ),
            "responsibility_weighted_auc": responsibility_weighted_auc(
                labels, predictions[action], weights
            ),
        }
        for action in (ACTION_STRONG_UNARY, ACTION_HASH256_DIFFUSION)
    }


def validate_nvos_cv_selector_payload(
    payload: object,
    *,
    authority: PromptResponsibilityAuthority,
    expected_responsibility_file_sha256: str,
    expected_primitive_sha256: str | None = None,
) -> torch.Tensor:
    """Recompute the reference-only decision and fail closed on every tensor."""

    if not isinstance(payload, dict) or set(payload) != ARTIFACT_KEYS:
        raise ValueError("NVOS CV selector artifact schema differs")
    if (
        payload["schema_version"] != SCHEMA_VERSION
        or isinstance(payload["schema_version"], bool)
        or payload["artifact_type"] != ARTIFACT_TYPE
        or payload["scene_id"] != authority.scene_id
        or payload["method_contract"] != METHOD_CONTRACT
        or payload["method_contract_sha256"] != _json_sha256(METHOD_CONTRACT)
        or payload["cv_contract"] != CV_CONTRACT
        or payload["cv_contract_sha256"] != _json_sha256(CV_CONTRACT)
        or payload["experiment_registration_sha256"] != REGISTRATION_SHA256
        or payload["responsibility_file_sha256"] != expected_responsibility_file_sha256
        or payload["responsibility_authority_sha256"] != authority.digest
        or payload["target_rgb_opened"] is not False
        or payload["target_mask_opened"] is not False
        or payload["target_metric_computed"] is not False
    ):
        raise ValueError("NVOS CV selector method or authority differs")
    for name in (
        "base_selector_sha256",
        "base_selector_method_contract_sha256",
        "base_primitive_probability_sha256",
        "base_query_compatibility_sha256",
        "responsibility_report_sha256",
        "responsibility_tensor_bundle_sha256",
        "capability_sidecar_sha256",
        "field_checkpoint_sha256",
        "support_graph_sha256",
        "knn_cache_sha256",
        "feature_hash_sha256",
    ):
        if not _is_sha256(payload[name]):
            raise ValueError(f"NVOS CV selector {name} is not a SHA-256")
    tensors = payload["tensors"]
    if not isinstance(tensors, dict) or set(tensors) != BASE_TENSOR_KEYS:
        raise ValueError("NVOS CV selector tensor schema differs")
    count = int(authority.num_gaussians)
    for name in (
        "primitive_probability",
        "query_compatibility",
        "signed_reference_evidence",
        "reference_weight",
        "cv_oof_strong_unary",
        "cv_oof_hash256_diffusion",
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
            raise ValueError(f"NVOS CV selector tensor {name} is malformed")
    for name in (
        "primitive_probability",
        "query_compatibility",
        "cv_oof_strong_unary",
        "cv_oof_hash256_diffusion",
    ):
        if bool(((tensors[name] < 0) | (tensors[name] > 1)).any()):
            raise ValueError(f"NVOS CV selector probability {name} is outside [0,1]")
    if bool(
        (
            (tensors["signed_reference_evidence"] < -1)
            | (tensors["signed_reference_evidence"] > 1)
        ).any()
    ) or bool((tensors["reference_weight"] < 0).any()):
        raise ValueError("NVOS CV selector evidence or responsibility is invalid")
    for name in ("capability_valid", "positive_exclusive", "negative_exclusive", "conflict"):
        value = tensors[name]
        if (
            not torch.is_tensor(value)
            or value.device.type != "cpu"
            or value.dtype != torch.bool
            or tuple(value.shape) != (count,)
            or not value.is_contiguous()
        ):
            raise ValueError(f"NVOS CV selector mask {name} is malformed")
    folds_global = tensors["cv_fold_ids"]
    if (
        not torch.is_tensor(folds_global)
        or folds_global.device.type != "cpu"
        or folds_global.dtype != torch.int64
        or tuple(folds_global.shape) != (count,)
        or not folds_global.is_contiguous()
    ):
        raise ValueError("NVOS CV selector fold ids are malformed")
    valid = tensors["capability_valid"]
    global_rows = torch.where(valid)[0]
    expected_folds, expected_reports = audit_signed_cv_population(
        global_rows,
        tensors["signed_reference_evidence"][global_rows],
        tensors["reference_weight"][global_rows],
        num_folds=int(CV_CONTRACT["num_folds"]),
        minimum_class_rows=int(CV_CONTRACT["minimum_class_rows"]),
    )
    if not bool((folds_global[~valid] == -1).all()) or not torch.equal(
        folds_global[global_rows], expected_folds
    ):
        raise ValueError("NVOS CV selector fold assignment differs")
    observed = valid & (tensors["signed_reference_evidence"] != 0)
    for name in ("cv_oof_strong_unary", "cv_oof_hash256_diffusion"):
        if bool((tensors[name][~observed] != 0).any()):
            raise ValueError("NVOS CV selector OOF prediction leaks outside held-out anchors")
    if payload["cv_fold_reports"] != expected_reports:
        raise ValueError("NVOS CV selector fold population receipt differs")
    recomputed_metrics = _metrics_from_payload_tensors(tensors)
    if payload["cv_metrics"] != recomputed_metrics:
        raise ValueError("NVOS CV selector metrics differ from OOF tensors")
    selected = select_registered_action(
        recomputed_metrics,
        metric_round_decimals=int(CV_CONTRACT["metric_round_decimals"]),
    )
    if payload["selected_action"] != selected:
        raise ValueError("NVOS CV selector action differs from registered decision")
    positive = tensors["positive_exclusive"]
    negative = tensors["negative_exclusive"]
    conflict = tensors["conflict"]
    if bool((positive & negative).any()) or bool((positive & conflict).any()) or bool(
        (negative & conflict).any()
    ):
        raise ValueError("NVOS CV selector exact prompt row partitions overlap")
    if not bool((tensors["primitive_probability"][positive] == 1).all()) or not bool(
        (tensors["primitive_probability"][negative] == 0).all()
    ):
        raise ValueError("NVOS CV selector direct anchors differ")
    if bool((tensors["query_compatibility"][~valid] != 0).any()):
        raise ValueError("NVOS CV selector invalid rows have compatibility")
    if selected == ACTION_STRONG_UNARY:
        expected = tensors["query_compatibility"].clone()
        expected[positive] = 1
        expected[negative] = 0
        if not torch.equal(tensors["primitive_probability"], expected):
            raise ValueError("NVOS CV strong-unary output differs from full-refit compatibility")
    digests = {name: tensor_sha256(value) for name, value in sorted(tensors.items())}
    if digests["query_compatibility"] != payload["base_query_compatibility_sha256"]:
        raise ValueError("NVOS CV full-refit compatibility differs from frozen base")
    if (
        selected == ACTION_HASH256_DIFFUSION
        and digests["primitive_probability"]
        != payload["base_primitive_probability_sha256"]
    ):
        raise ValueError("NVOS CV selected diffusion differs from frozen base")
    if payload["tensor_sha256"] != digests or payload["tensor_bundle_sha256"] != _json_sha256(
        digests
    ):
        raise ValueError("NVOS CV selector tensor digest differs")
    primitive_digest = digests["primitive_probability"]
    if expected_primitive_sha256 is not None and primitive_digest != expected_primitive_sha256:
        raise ValueError("NVOS CV selector primitive probability differs")
    return tensors["primitive_probability"]


@torch.inference_mode()
def build(args: argparse.Namespace) -> dict[str, object]:
    registration_path = Path(args.experiment_registration).resolve()
    registration_sha256 = sha256_file(registration_path)
    if registration_sha256 != REGISTRATION_SHA256:
        raise ValueError("signed-scribble CV experiment registration differs")
    report_path = Path(args.cache_report).resolve()
    report_sha256 = sha256_file(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    authority = PromptResponsibilityAuthority.from_dict(report["authority"])
    if authority.scene_id != args.scene_id:
        raise ValueError("responsibility authority scene differs")
    base_path = Path(args.base_selector).resolve()
    base_sha256 = sha256_file(base_path)
    base_payload = torch.load(base_path, map_location="cpu", weights_only=True)
    validate_nvos_strict_support_payload(
        base_payload,
        authority=authority,
        expected_responsibility_file_sha256=str(report["file_sha256"]),
    )
    if (
        base_payload["responsibility_tensor_bundle_sha256"]
        != str(report["tensor_bundle_sha256"])
        or base_payload["target_rgb_opened"] is not False
        or base_payload["target_mask_opened"] is not False
        or base_payload["target_metric_computed"] is not False
    ):
        raise ValueError("base strict selector authority differs")
    base_tensors = base_payload["tensors"]

    capability_path = Path(args.capability_cache).resolve()
    capability_sidecar = Path(str(capability_path) + ".json")
    capability_receipt = json.loads(capability_sidecar.read_text(encoding="utf-8"))
    bank = load_canonical_capability_bank(capability_path)
    if bank.num_gaussians != authority.num_gaussians or _float_rows_sha256(
        bank.xyz
    ) != authority.geometry_xyz_sha256:
        raise ValueError("capability and strict responsibility geometry differ")
    if (
        str(capability_receipt.get("field_checkpoint_sha256"))
        != str(base_payload["field_checkpoint_sha256"])
        or bool(capability_receipt.get("benchmark_images_opened"))
        or bool(capability_receipt.get("benchmark_masks_opened"))
        or bool(capability_receipt.get("text_queries_opened"))
    ):
        raise ValueError("capability receipt authority differs from frozen base")
    graph_path = Path(args.support_graph).resolve()
    graph_sha256 = sha256_file(graph_path)
    graph_payload = torch.load(graph_path, map_location="cpu", weights_only=True)
    global_rows = torch.as_tensor(graph_payload["global_rows"]).long().cpu()
    graph_xyz = torch.as_tensor(graph_payload["xyz"]).float().cpu()
    if (
        not torch.equal(global_rows, bank.global_rows)
        or int(graph_payload["num_global_rows"]) != authority.num_gaussians
        or not torch.equal(graph_xyz, bank.xyz[global_rows])
    ):
        raise ValueError("support graph and capability rows differ")
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
        raise ValueError("signed-scribble CV kNN registration differs")
    base_knn = load_query_diffusion_knn_cache(
        base_payload["knn_cache_path"],
        expected_global_rows=global_rows,
        expected_xyz=graph_xyz,
        expected_source_graph_sha256=graph_sha256,
        expected_num_neighbors=200,
    )
    if not torch.equal(knn.neighbor_indices, base_knn.neighbor_indices):
        raise ValueError("signed-scribble CV topology differs from frozen hash candidate")
    appearance = bank.valid_feature_banks()["appearance"]
    hashed = deterministic_feature_hash(
        appearance, 256, batch_size=int(args.hash_batch_size)
    ).contiguous()
    feature_hash_sha256 = tensor_sha256(hashed)
    if feature_hash_sha256 != base_payload["feature_hash_sha256"]:
        raise ValueError("signed-scribble CV hash relation differs from frozen candidate")
    del appearance, bank, graph_payload

    local_signed = base_tensors["signed_reference_evidence"][global_rows].float().contiguous()
    local_weight = base_tensors["reference_weight"][global_rows].float().contiguous()
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
        raise ValueError("signed-scribble CV support requires CUDA")
    normalized_cpu = normalize_node_features(hashed, eps=config.eps).cpu().contiguous()
    normalized = normalized_cpu.to(device)
    neighbors = knn.neighbor_indices.to(device)
    distances = knn_feature_distances(
        normalized,
        neighbors,
        distance_chunk_size=int(args.distance_chunk_size),
    )

    def predictor(training_evidence: torch.Tensor, fold: int):
        del fold
        compatibility_cpu = weighted_logistic_query_compatibility(
            normalized_cpu,
            training_evidence,
            local_weight,
            logistic_c=config.logistic_c,
            regularizer_bandwidth=config.regularizer_bandwidth,
            fit_population="signed_nonzero",
        )
        compatibility = compatibility_cpu.to(device)
        training_device = training_evidence.to(device)
        similarities = rbf_similarity_from_distances(
            distances,
            feature_bandwidth=config.feature_bandwidth,
            positive_reference_mask=training_device > 0,
            eps=config.eps,
        )
        support = run_query_conditioned_diffusion(
            training_device[:, None],
            neighbors,
            similarities,
            compatibility,
            config=config,
        ).squeeze(1)
        support_cpu = support.float().cpu().clamp(0, 1).contiguous()
        del compatibility, training_device, similarities, support
        torch.cuda.empty_cache()
        return compatibility_cpu.float().contiguous(), support_cpu

    cv = run_signed_scribble_cross_validation(
        global_rows,
        local_signed,
        local_weight,
        predictor,
        num_folds=int(CV_CONTRACT["num_folds"]),
        minimum_class_rows=int(CV_CONTRACT["minimum_class_rows"]),
        metric_round_decimals=int(CV_CONTRACT["metric_round_decimals"]),
        probability_epsilon=float(CV_CONTRACT["probability_epsilon"]),
    )
    count = authority.num_gaussians
    compatibility = base_tensors["query_compatibility"].clone().float().contiguous()
    if cv.selected_action == ACTION_STRONG_UNARY:
        primitive = compatibility.clone()
        primitive[base_tensors["positive_exclusive"]] = 1
        primitive[base_tensors["negative_exclusive"]] = 0
    else:
        primitive = base_tensors["primitive_probability"].clone().float().contiguous()
    cv_fold_ids = torch.full((count,), -1, dtype=torch.int64)
    cv_fold_ids[global_rows] = cv.folds
    oof_unary = torch.zeros(count, dtype=torch.float32)
    oof_diffusion = torch.zeros(count, dtype=torch.float32)
    observed_local = cv.observed
    observed_global = global_rows[observed_local]
    oof_unary[observed_global] = cv.oof_predictions[ACTION_STRONG_UNARY][observed_local]
    oof_diffusion[observed_global] = cv.oof_predictions[
        ACTION_HASH256_DIFFUSION
    ][observed_local]
    tensors = {
        "primitive_probability": primitive,
        "query_compatibility": compatibility,
        "signed_reference_evidence": base_tensors["signed_reference_evidence"].clone(),
        "reference_weight": base_tensors["reference_weight"].clone(),
        "capability_valid": base_tensors["capability_valid"].clone(),
        "positive_exclusive": base_tensors["positive_exclusive"].clone(),
        "negative_exclusive": base_tensors["negative_exclusive"].clone(),
        "conflict": base_tensors["conflict"].clone(),
        "cv_fold_ids": cv_fold_ids,
        "cv_oof_strong_unary": oof_unary,
        "cv_oof_hash256_diffusion": oof_diffusion,
    }
    tensors = {name: value.cpu().contiguous() for name, value in tensors.items()}
    digests = {name: tensor_sha256(value) for name, value in sorted(tensors.items())}
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "scene_id": args.scene_id,
        "method_contract": METHOD_CONTRACT,
        "method_contract_sha256": _json_sha256(METHOD_CONTRACT),
        "cv_contract": CV_CONTRACT,
        "cv_contract_sha256": _json_sha256(CV_CONTRACT),
        "selected_action": cv.selected_action,
        "cv_metrics": cv.metrics,
        "cv_fold_reports": cv.fold_reports,
        "experiment_registration_path": str(registration_path),
        "experiment_registration_sha256": registration_sha256,
        "base_selector_path": str(base_path),
        "base_selector_sha256": base_sha256,
        "base_selector_method_contract_sha256": base_payload["method_contract_sha256"],
        "base_primitive_probability_sha256": base_payload["tensor_sha256"][
            "primitive_probability"
        ],
        "base_query_compatibility_sha256": base_payload["tensor_sha256"][
            "query_compatibility"
        ],
        "responsibility_report_path": str(report_path),
        "responsibility_report_sha256": report_sha256,
        "responsibility_file_sha256": str(report["file_sha256"]),
        "responsibility_authority_sha256": authority.digest,
        "responsibility_tensor_bundle_sha256": str(report["tensor_bundle_sha256"]),
        "capability_cache_path": str(capability_path),
        "capability_sidecar_sha256": sha256_file(capability_sidecar),
        "field_checkpoint_sha256": str(base_payload["field_checkpoint_sha256"]),
        "support_graph_path": str(graph_path),
        "support_graph_sha256": graph_sha256,
        "knn_cache_path": str(knn_path),
        "knn_cache_sha256": knn_sha256,
        "feature_hash_sha256": feature_hash_sha256,
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
    validate_nvos_cv_selector_payload(
        frozen,
        authority=authority,
        expected_responsibility_file_sha256=str(report["file_sha256"]),
        expected_primitive_sha256=digests["primitive_probability"],
    )
    if sha256_file(output) != output_sha256:
        raise ValueError("NVOS CV selector changed across freeze and reload")
    report_payload = {
        "scene_id": args.scene_id,
        "artifact_type": ARTIFACT_TYPE,
        "method_contract": METHOD_CONTRACT,
        "method_contract_sha256": artifact["method_contract_sha256"],
        "cv_contract": CV_CONTRACT,
        "cv_contract_sha256": artifact["cv_contract_sha256"],
        "selected_action": cv.selected_action,
        "cv_metrics": cv.metrics,
        "cv_fold_reports": cv.fold_reports,
        "experiment_registration_sha256": registration_sha256,
        "output": str(output),
        "output_sha256": output_sha256,
        "primitive_probability_sha256": digests["primitive_probability"],
        "feature_hash_sha256": feature_hash_sha256,
        "diffusion_config": asdict(config),
        "num_global_rows": count,
        "capability_valid_rows": int(global_rows.numel()),
        "signed_observed_valid_rows": int(observed_local.sum()),
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
    parser.add_argument("--base-selector", required=True)
    parser.add_argument("--cache-report", required=True)
    parser.add_argument("--capability-cache", required=True)
    parser.add_argument("--support-graph", required=True)
    parser.add_argument("--knn-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hash-batch-size", type=int, default=8192)
    parser.add_argument("--distance-chunk-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
