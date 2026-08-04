#!/usr/bin/env python3
"""Build the preregistered strict-source-only NVOS E2S-v2 selector.

The builder may read the official positive/negative scribbles, their sealed
exact raster-adjoint cache, and the query-independent canonical field assets.
It cannot read a target RGB image, target mask, score, or metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from radio_gs.evaluation.promptable_segmentation import load_ground_truth_mask
from radio_gs.interfaces.capability_cache import (
    load_canonical_capability_bank,
    load_canonical_support_graph,
)
from radio_gs.interfaces.prompt_responsibility_cache import (
    PromptResponsibilityAuthority,
    load_prompt_responsibility_cache,
    sha256_file,
    tensor_sha256,
)
from radio_gs.querying.support_solver import (
    SupportSolverConfig,
    solve_seeded_random_walker,
)
from radio_gs.scripts.build_nvos_strict_query_conditioned_support import (
    _float_rows_sha256,
)


ARTIFACT_TYPE = "nvos_strict_evidence_to_support_v2"
SCHEMA_VERSION = 1
REGISTRATION_SHA256 = (
    "3d4085c634f89f527e7cacf8ea5a073f20de73ea7e3c14a77d7c0e77095ef8f9"
)
METHOD_CONTRACT = {
    "track": "strict_source_only_raw_positive_negative_scribbles",
    "classifier": "all_scribble_exact_DINOv3_diagonal_shrinkage_LDA",
    "classifier_population": "all_capability_valid_nonzero_signed_exact_adjoint_rows",
    "classifier_sample_weight": "abs(signed_evidence)*W_transpose_one_class_normalized",
    "classifier_ridge": "one_over_feature_dimension",
    "unary": "continuous_sigmoid_LDA_probability",
    "unary_confidence": "abs(2*probability-1)_clamp_min_0.05",
    "graph": "frozen_canonical_shared_k16_symmetric_raw_affinity",
    "query_gate": "sqrt(P_i*P_j)_then_recompute_symmetric_degree_normalization",
    "laplacian_weight": 0.25,
    "cg_iterations": 128,
    "cg_tolerance": 1e-6,
    "hard_seed_rule": "per_sign_max_normalized_exact_adjoint_mass_at_least_0.2_exclusive_relative",
    "connected_selection": "none",
    "target_dependent_tuning": False,
}
TENSOR_KEYS = {
    "primitive_probability",
    "lda_probability",
    "unary_confidence",
    "signed_reference_evidence",
    "reference_weight",
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
    "classifier_diagnostics",
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
    digest = str(value)
    return len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)


def diagonal_shrinkage_lda(
    features: torch.Tensor,
    signed_evidence: torch.Tensor,
    responsibility: torch.Tensor,
    *,
    device: torch.device,
    chunk_size: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Fit and apply the registered class-balanced diagonal LDA."""

    rows = torch.as_tensor(features).detach().cpu()
    signed = torch.as_tensor(signed_evidence).detach().float().cpu().reshape(-1)
    mass = torch.as_tensor(responsibility).detach().float().cpu().reshape(-1)
    if rows.ndim != 2 or signed.shape != (rows.shape[0],) or mass.shape != signed.shape:
        raise ValueError("LDA inputs do not align")
    if not bool(torch.isfinite(signed).all()) or not bool(torch.isfinite(mass).all()):
        raise ValueError("LDA evidence contains NaN or infinity")
    positive = signed > 0
    negative = signed < 0
    if int(positive.sum()) < 32 or int(negative.sum()) < 32:
        raise ValueError("all-scribble LDA requires at least 32 rows per sign")
    weights = signed.abs() * mass.clamp_min(0.0)
    positive_mass = weights[positive].sum()
    negative_mass = weights[negative].sum()
    if float(positive_mass) <= 0 or float(negative_mass) <= 0:
        raise ValueError("all-scribble LDA requires positive responsibility per sign")
    weights[positive] *= 0.5 / positive_mass
    weights[negative] *= 0.5 / negative_mass

    dimension = int(rows.shape[1])
    observed = positive | negative
    observed_rows = rows[observed].to(device=device, dtype=torch.float32)
    observed_rows = F.normalize(observed_rows, dim=-1, eps=1e-8)
    observed_weights = weights[observed].to(device)
    observed_positive = positive[observed].to(device)
    observed_negative = negative[observed].to(device)
    positive_weights = observed_weights * observed_positive
    negative_weights = observed_weights * observed_negative
    # Each sign has total weight one half.  Divide by that total explicitly
    # so the means and second moments retain the usual probabilistic scale.
    mu_positive = (positive_weights[:, None] * observed_rows).sum(dim=0) * 2.0
    mu_negative = (negative_weights[:, None] * observed_rows).sum(dim=0) * 2.0
    second_positive = (
        positive_weights[:, None] * observed_rows.square()
    ).sum(dim=0) * 2.0
    second_negative = (
        negative_weights[:, None] * observed_rows.square()
    ).sum(dim=0) * 2.0
    variance = 0.5 * (
        (second_positive - mu_positive.square()).clamp_min(0.0)
        + (second_negative - mu_negative.square()).clamp_min(0.0)
    )
    ridge = 1.0 / float(dimension)
    direction = (mu_positive - mu_negative) / (variance + ridge)
    midpoint = 0.5 * torch.dot(direction, mu_positive + mu_negative)
    del observed_rows, positive_weights, negative_weights

    probability = torch.empty(rows.shape[0], dtype=torch.float32)
    raw_minimum = float("inf")
    raw_maximum = float("-inf")
    for start in range(0, rows.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), rows.shape[0])
        batch = F.normalize(
            rows[start:stop].to(device=device, dtype=torch.float32),
            dim=-1,
            eps=1e-8,
        )
        logits = batch @ direction - midpoint
        raw_minimum = min(raw_minimum, float(logits.min()))
        raw_maximum = max(raw_maximum, float(logits.max()))
        probability[start:stop] = torch.sigmoid(logits).cpu()
    diagnostics = {
        "feature_dimension": dimension,
        "positive_training_rows": int(positive.sum()),
        "negative_training_rows": int(negative.sum()),
        "positive_raw_weight": float(positive_mass),
        "negative_raw_weight": float(negative_mass),
        "ridge": ridge,
        "direction_l2_norm": float(direction.norm()),
        "logit_minimum": raw_minimum,
        "logit_maximum": raw_maximum,
        "probability_mean": float(probability.mean()),
    }
    return probability.contiguous(), diagnostics


def validate_nvos_e2s_v2_payload(
    payload: object,
    *,
    authority: PromptResponsibilityAuthority,
    expected_responsibility_file_sha256: str,
    expected_primitive_sha256: str | None = None,
) -> torch.Tensor:
    if not isinstance(payload, dict) or set(payload) != ARTIFACT_KEYS:
        raise ValueError("NVOS E2S-v2 artifact schema differs")
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
        raise ValueError("NVOS E2S-v2 method or authority differs")
    for name in (
        "responsibility_report_sha256",
        "responsibility_tensor_bundle_sha256",
        "capability_sidecar_sha256",
        "field_checkpoint_sha256",
        "support_graph_sha256",
    ):
        if not _is_sha256(payload[name]):
            raise ValueError(f"NVOS E2S-v2 {name} is not a SHA-256")
    tensors = payload["tensors"]
    if not isinstance(tensors, dict) or set(tensors) != TENSOR_KEYS:
        raise ValueError("NVOS E2S-v2 tensor schema differs")
    count = int(authority.num_gaussians)
    for name in (
        "primitive_probability",
        "lda_probability",
        "unary_confidence",
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
            raise ValueError(f"NVOS E2S-v2 tensor {name} is malformed")
    for name in ("primitive_probability", "lda_probability", "unary_confidence"):
        if bool(((tensors[name] < 0) | (tensors[name] > 1)).any()):
            raise ValueError(f"NVOS E2S-v2 probability/confidence {name} is invalid")
    for name in ("capability_valid", "hard_positive", "hard_negative"):
        value = tensors[name]
        if (
            not torch.is_tensor(value)
            or value.device.type != "cpu"
            or value.dtype != torch.bool
            or tuple(value.shape) != (count,)
            or not value.is_contiguous()
        ):
            raise ValueError(f"NVOS E2S-v2 mask {name} is malformed")
    hard_positive = tensors["hard_positive"]
    hard_negative = tensors["hard_negative"]
    if bool((hard_positive & hard_negative).any()):
        raise ValueError("NVOS E2S-v2 hard constraints overlap")
    if not bool((tensors["primitive_probability"][hard_positive] == 1).all()):
        raise ValueError("NVOS E2S-v2 positive constraints are not exact one")
    if not bool((tensors["primitive_probability"][hard_negative] == 0).all()):
        raise ValueError("NVOS E2S-v2 negative constraints are not exact zero")
    digests = {name: tensor_sha256(value) for name, value in sorted(tensors.items())}
    if (
        payload["tensor_sha256"] != digests
        or payload["tensor_bundle_sha256"] != _json_sha256(digests)
    ):
        raise ValueError("NVOS E2S-v2 tensor digest differs")
    if (
        expected_primitive_sha256 is not None
        and digests["primitive_probability"] != expected_primitive_sha256
    ):
        raise ValueError("NVOS E2S-v2 primitive probability differs")
    return tensors["primitive_probability"]


@torch.inference_mode()
def build(args: argparse.Namespace) -> dict[str, object]:
    registration_path = Path(args.experiment_registration).resolve()
    if sha256_file(registration_path) != REGISTRATION_SHA256:
        raise ValueError("NVOS E2S-v2 experiment registration differs")
    report_path = Path(args.cache_report).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    authority = PromptResponsibilityAuthority.from_dict(report["authority"])
    if authority.scene_id != args.scene_id:
        raise ValueError("responsibility authority scene differs")
    cache_path = Path(args.cache).resolve()
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
    if bool((positive_pixels & negative_pixels).any()):
        raise ValueError("official scribbles overlap")
    positive_mass = cache.adjoint(positive_pixels).weighted_sum.float()
    negative_mass = cache.adjoint(negative_pixels).weighted_sum.float()
    visible_mass = cache.visible_mass.float()
    del cache
    signed = torch.zeros_like(visible_mass)
    visible = visible_mass > 0
    signed[visible] = (
        positive_mass[visible] - negative_mass[visible]
    ) / visible_mass[visible]
    signed.clamp_(-1.0, 1.0)

    capability_path = Path(args.capability_cache).resolve()
    bank = load_canonical_capability_bank(capability_path)
    if bank.num_gaussians != authority.num_gaussians:
        raise ValueError("capability and responsibility row counts differ")
    if _float_rows_sha256(bank.xyz) != authority.geometry_xyz_sha256:
        raise ValueError("capability and responsibility geometry differ")
    graph_path = Path(args.support_graph).resolve()
    graph = load_canonical_support_graph(graph_path, bank)
    global_rows = bank.global_rows
    features = bank.valid_feature_banks()["appearance"]
    local_probability, classifier_diagnostics = diagonal_shrinkage_lda(
        features,
        signed[global_rows],
        visible_mass[global_rows],
        device=torch.device(args.device),
        chunk_size=int(args.feature_chunk_size),
    )

    positive_seed = positive_mass[global_rows]
    negative_seed = negative_mass[global_rows]
    positive_seed = positive_seed / positive_seed.max().clamp_min(1e-12)
    negative_seed = negative_seed / negative_seed.max().clamp_min(1e-12)
    hard_threshold = 0.20
    local_hard_positive = (positive_seed >= hard_threshold) & (
        positive_seed > negative_seed
    )
    local_hard_negative = (negative_seed >= hard_threshold) & (
        negative_seed > positive_seed
    )
    confidence = (2.0 * local_probability - 1.0).abs().clamp_min(0.05)
    device = torch.device(args.device)
    solver_config = SupportSolverConfig(
        solver_type="random_walker",
        laplacian_weight=0.25,
        cg_iterations=128,
        cg_tolerance=1e-6,
        hard_seed_threshold=hard_threshold,
        hard_seed_conflict_policy="exclusive_relative",
    )
    solved = solve_seeded_random_walker(
        graph.to(device),
        local_probability.to(device),
        positive_seed.to(device),
        negative_seed.to(device),
        config=solver_config,
        unary_confidence=confidence.to(device),
        query_gate=local_probability.to(device),
    ).float().cpu().contiguous()
    if not bool((solved[local_hard_positive] == 1).all()) or not bool(
        (solved[local_hard_negative] == 0).all()
    ):
        raise ValueError("continuous solver failed exact hard constraints")

    primitive = torch.zeros(authority.num_gaussians, dtype=torch.float32)
    lda_probability = torch.zeros_like(primitive)
    unary_confidence = torch.zeros_like(primitive)
    capability_valid = torch.zeros(authority.num_gaussians, dtype=torch.bool)
    hard_positive = torch.zeros_like(capability_valid)
    hard_negative = torch.zeros_like(capability_valid)
    primitive[global_rows] = solved
    lda_probability[global_rows] = local_probability
    unary_confidence[global_rows] = confidence
    capability_valid[global_rows] = True
    hard_positive[global_rows] = local_hard_positive
    hard_negative[global_rows] = local_hard_negative
    # Preserve exact constraints for high-confidence observed primitives that
    # abstain from the canonical capability bank.
    positive_all = positive_mass / positive_mass.max().clamp_min(1e-12)
    negative_all = negative_mass / negative_mass.max().clamp_min(1e-12)
    hard_positive |= (positive_all >= hard_threshold) & (positive_all > negative_all)
    hard_negative |= (negative_all >= hard_threshold) & (negative_all > positive_all)
    primitive[hard_positive] = 1.0
    primitive[hard_negative] = 0.0

    tensors = {
        "primitive_probability": primitive.contiguous(),
        "lda_probability": lda_probability.contiguous(),
        "unary_confidence": unary_confidence.contiguous(),
        "signed_reference_evidence": signed.contiguous(),
        "reference_weight": visible_mass.contiguous(),
        "capability_valid": capability_valid.contiguous(),
        "hard_positive": hard_positive.contiguous(),
        "hard_negative": hard_negative.contiguous(),
    }
    digests = {name: tensor_sha256(value) for name, value in sorted(tensors.items())}
    field_hash = str(bank.metadata.get("field_checkpoint_sha256", ""))
    if not _is_sha256(field_hash):
        raise ValueError("capability metadata lacks field checkpoint SHA-256")
    solver_diagnostics = {
        "valid_rows": int(global_rows.numel()),
        "hard_positive_valid_rows": int(local_hard_positive.sum()),
        "hard_negative_valid_rows": int(local_hard_negative.sum()),
        "hard_positive_global_rows": int(hard_positive.sum()),
        "hard_negative_global_rows": int(hard_negative.sum()),
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
        "responsibility_report_path": str(report_path),
        "responsibility_report_sha256": sha256_file(report_path),
        "responsibility_file_sha256": str(report["file_sha256"]),
        "responsibility_authority_sha256": authority.digest,
        "responsibility_tensor_bundle_sha256": str(report["tensor_bundle_sha256"]),
        "capability_cache_path": str(capability_path),
        "capability_sidecar_sha256": sha256_file(Path(str(capability_path) + ".json")),
        "field_checkpoint_sha256": field_hash,
        "support_graph_path": str(graph_path),
        "support_graph_sha256": sha256_file(graph_path),
        "classifier_diagnostics": classifier_diagnostics,
        "solver_diagnostics": solver_diagnostics,
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
    validate_nvos_e2s_v2_payload(
        frozen,
        authority=authority,
        expected_responsibility_file_sha256=str(report["file_sha256"]),
        expected_primitive_sha256=digests["primitive_probability"],
    )
    if sha256_file(output) != output_sha256:
        raise ValueError("NVOS E2S-v2 selector changed across freeze and reload")
    receipt = {
        "scene_id": args.scene_id,
        "artifact_type": ARTIFACT_TYPE,
        "method_contract_sha256": artifact["method_contract_sha256"],
        "experiment_registration_sha256": REGISTRATION_SHA256,
        "output": str(output),
        "output_sha256": output_sha256,
        "primitive_probability_sha256": digests["primitive_probability"],
        "classifier_diagnostics": classifier_diagnostics,
        "solver_diagnostics": solver_diagnostics,
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
    parser.add_argument("--cache", required=True)
    parser.add_argument("--cache-report", required=True)
    parser.add_argument("--positive-scribble", required=True)
    parser.add_argument("--negative-scribble", required=True)
    parser.add_argument("--capability-cache", required=True)
    parser.add_argument("--support-graph", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-chunk-size", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
