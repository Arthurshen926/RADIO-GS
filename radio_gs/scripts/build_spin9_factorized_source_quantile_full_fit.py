#!/usr/bin/env python3
"""Seal one full9 factorized SPIn source-only exact-W quantile gauge.

This entrypoint is intentionally separate from the frozen Lego v2 builder.  It
binds the full-nine-scene preregistration, the source-footprint OOF deployment
gate, the native exact responsibility matrix and the matched target-score
receipt.  Only the *source* reference mask is decoded; target rasters remain
sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch

from radio_gs.evaluation.promptable_segmentation import load_ground_truth_mask
from radio_gs.interfaces.prompt_responsibility_cache import (
    PromptResponsibilityAuthority,
    load_prompt_responsibility_cache,
    tensor_sha256,
)
from radio_gs.interfaces.query_diffusion_cache import (
    load_query_diffusion_knn_cache,
    load_query_diffusion_relation_cache,
)
from radio_gs.querying.spin_source_footprint_quantile_calibration import (
    FULL_FIT_GAUGE_ARTIFACT_TYPE,
    compute_full_fit_quantile_gauge,
    quantile_method_contract,
)
from radio_gs.scripts.build_spin_source_footprint_quantile_oof import (
    file_sha256,
    json_sha256,
)


PREREGISTRATION = "spin9_factorized_source_quantile_full9_expansion_v1"
SOURCE_FOLD_TYPE = "source_observation_surface_safe_footprint_oof_fold_v1"
SOURCE_GATE_TYPE = "source_observation_surface_safe_footprint_oof_gate_v1"
PREMETRIC_TYPE = "nvos_pre_metric_prediction_receipt_v1"
RAW_SEEN_THRESHOLD = 0.71
COMPLETION_QUANTILE = 0.96
# The OOF evaluator and the native exact-W exporter traverse the same accepted
# splats through independent floating-point accumulation paths.  Preserve the
# frozen relative tolerance while allowing two absolute micro-units for a
# near-zero-mass row.  The deployed gauge below is still constructed solely
# from native exact-W; this tolerance only authenticates the OOF gate lineage.
SOURCE_EXACT_W_PARITY_RTOL = 1e-3
SOURCE_EXACT_W_PARITY_ATOL = 2e-6


def _require_file(path: str | Path, expected: str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    actual = file_sha256(resolved)
    if actual != str(expected):
        raise ValueError(f"{label} SHA-256 differs: {actual}")
    return resolved


def _raw_tensor_sha256(value: torch.Tensor) -> str:
    """Replay the legacy source-OOF raw-byte tensor digest."""

    array = torch.as_tensor(value).detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _require_target_blind(payload: Mapping[str, object], *, label: str) -> None:
    names = (
        "target_rgb_opened",
        "target_mask_opened",
        "target_metric_opened",
        "target_metric_computed",
    )
    for name in names:
        if name in payload and payload.get(name) is not False:
            raise ValueError(f"{label} target-blind flag differs: {name}")


def _load_preregistration(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("registration") != PREREGISTRATION:
        raise ValueError("unexpected full9 factorized quantile preregistration")
    development = payload.get("development_authority")
    gauge = payload.get("source_quantile_gauge")
    prediction = payload.get("prediction_contract")
    if not isinstance(development, Mapping) or (
        float(development.get("raw_seen_threshold", float("nan")))
        != RAW_SEEN_THRESHOLD
        or float(development.get("fixed_completion_quantile", float("nan")))
        != COMPLETION_QUANTILE
    ):
        raise ValueError("full9 raw/completion thresholds differ")
    if not isinstance(gauge, Mapping) or (
        gauge.get("tie_semantics") != "right-continuous"
        or gauge.get("target_distribution_fit") is not False
        or gauge.get("quantile_or_threshold_scan") is not False
        or float(gauge.get("completion_quantile", float("nan")))
        != COMPLETION_QUANTILE
    ):
        raise ValueError("full9 source quantile-gauge contract differs")
    if not isinstance(prediction, Mapping) or prediction.get("parameter_scan") is not False:
        raise ValueError("full9 prediction contract permits a parameter scan")
    return payload


def _load_source_fold(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("artifact_type") != SOURCE_FOLD_TYPE:
        raise ValueError("unexpected source-footprint OOF fold")
    _require_target_blind(payload, label="source fold")
    hashes = payload.get("tensor_sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError("source fold lacks tensor hashes")
    for name, expected in hashes.items():
        if name not in payload or _raw_tensor_sha256(payload[name]) != expected:
            raise ValueError(f"source-fold tensor changed: {name}")
    valid = torch.as_tensor(payload.get("valid")).detach().bool().cpu().reshape(-1)
    rows = torch.as_tensor(payload.get("global_rows")).detach().long().cpu().reshape(-1)
    if not torch.equal(rows, torch.where(valid)[0]):
        raise ValueError("source fold rows differ from its valid-row authority")
    return payload


def _load_gate(
    path: Path,
    *,
    source_fold: Mapping[str, object],
    source_fold_sha256: str,
) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("artifact_type") != SOURCE_GATE_TYPE:
        raise ValueError("unexpected source-footprint OOF gate")
    _require_target_blind(payload, label="source gate")
    if (
        payload.get("gate_mode") != "eligible_source_oof"
        or payload.get("selected_action") != "surface_safe_propagated"
        or payload.get("full_fit_predictions_used_as_oof") is not False
        or payload.get("scene_id") != source_fold.get("scene_id")
        or payload.get("protocol_hash") != source_fold.get("protocol_hash")
    ):
        raise ValueError("source-footprint OOF gate did not authorize deployment")
    heldout_fold = str(int(source_fold.get("heldout_fold", -1)))
    folds = payload.get("fold_artifacts")
    record = folds.get(heldout_fold) if isinstance(folds, Mapping) else None
    if not isinstance(record, Mapping) or record.get("sha256") != source_fold_sha256:
        raise ValueError("source gate does not bind the supplied source fold")
    return payload


def _load_exact_w(
    *,
    exact_path: Path,
    exact_sha256: str,
    report_path: Path,
    source_mask_path: Path,
    source_mask_sha256: str,
    scene_id: str,
):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("exact-W report is not an object")
    _require_target_blind(report, label="exact-W report")
    if report.get("historical_top1_responsibility_opened") is not False:
        raise ValueError("exact-W report opened historical top-1 responsibility")
    if report.get("file_sha256") != exact_sha256:
        raise ValueError("exact-W report binds a different artifact")
    authority = PromptResponsibilityAuthority.from_dict(report.get("authority"))
    if authority.scene_id != scene_id or report.get("authority_sha256") != authority.digest:
        raise ValueError("exact-W scene or authority digest differs")
    source_hashes = authority.source_sha256
    if not isinstance(source_hashes, Mapping) or source_hashes.get(
        "reference_binary_mask"
    ) != source_mask_sha256:
        raise ValueError("exact-W authority binds a different source mask")
    header = report.get("reference_mask_header_authority")
    if not isinstance(header, Mapping) or header.get(
        "reference_binary_mask_sha256"
    ) != source_mask_sha256:
        raise ValueError("exact-W header binds a different source mask")
    cache = load_prompt_responsibility_cache(
        exact_path,
        expected_authority=authority,
        expected_file_sha256=exact_sha256,
    )
    if report.get("tensor_bundle_sha256") != cache.tensor_bundle_sha256:
        raise ValueError("exact-W tensor bundle differs from its report")
    mask = torch.from_numpy(load_ground_truth_mask(source_mask_path)).bool()
    if tuple(mask.shape) != (authority.height, authority.width):
        raise ValueError("source mask shape differs from exact-W authority")
    return report, authority, cache, cache.adjoint(mask)


def _load_premetric_receipt(
    path: Path,
    *,
    scene_id: str,
    protocol_hash: str,
    relation_sha256: str,
    knn_sha256: str,
) -> dict[str, object]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or receipt.get("artifact_type") != PREMETRIC_TYPE:
        raise ValueError("unexpected matched pre-metric receipt")
    _require_target_blind(receipt, label="matched pre-metric receipt")
    if (
        receipt.get("sealed_before_target_ground_truth_open") is not True
        or receipt.get("scene_id") != scene_id
        or receipt.get("protocol_hash") != protocol_hash
    ):
        raise ValueError("matched pre-metric receipt scene/protocol/safety differs")
    method = receipt.get("method_contract")
    query = method.get("query_conditioned_diffusion") if isinstance(method, Mapping) else None
    if not isinstance(query, Mapping) or (
        query.get("kernel") != "ludvig_release_compat"
        or float(query.get("feature_bandwidth", float("nan"))) != 0.5
        or float(query.get("regularizer_bandwidth", float("nan"))) != 1.0
        or float(query.get("logistic_c", float("nan"))) != 0.01
        or int(query.get("iterations", -1)) != 100
        or float(query.get("edge_binarize_threshold", float("nan"))) != 1e-5
        or float(query.get("max_positive_fraction", float("nan"))) != 0.1
        or query.get("reference_calibration") is not False
    ):
        raise ValueError("matched pre-metric receipt K201 contract differs")
    relation = query.get("relation_cache")
    knn = query.get("knn_cache")
    if not isinstance(relation, Mapping) or relation.get("sha256") != relation_sha256:
        raise ValueError("matched pre-metric receipt relation cache differs")
    if not isinstance(knn, Mapping) or knn.get("sha256") != knn_sha256:
        raise ValueError("matched pre-metric receipt KNN cache differs")
    stages = receipt.get("stage_target_scores")
    propagated = stages.get("propagated") if isinstance(stages, Mapping) else None
    final = receipt.get("target_scores")
    if not isinstance(propagated, Mapping) or not isinstance(final, Mapping) or (
        set(propagated) != set(final)
    ):
        raise ValueError("matched receipt final/propagated frame identities differ")
    for frame_id, record in propagated.items():
        final_record = final.get(frame_id)
        if not isinstance(record, Mapping) or not isinstance(final_record, Mapping) or (
            record.get("sha256") != final_record.get("sha256")
        ):
            raise ValueError("matched receipt final score is not propagated score")
        _require_file(record.get("path", ""), record.get("sha256", ""), f"score {frame_id}")
    return receipt


def _max_relative_error(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    denominator = reference.abs().clamp_min(1e-12)
    return float(((candidate - reference).abs() / denominator).max())


def build(args: argparse.Namespace) -> dict[str, object]:
    prereg_path = _require_file(
        args.preregistration, args.preregistration_sha256, "full9 preregistration"
    )
    _load_preregistration(prereg_path)
    source_path = _require_file(args.source_fold, args.source_fold_sha256, "source fold")
    source = _load_source_fold(source_path)
    scene_id = str(source.get("scene_id", ""))
    protocol_hash = str(source.get("protocol_hash", ""))
    gate_path = _require_file(args.source_gate, args.source_gate_sha256, "source gate")
    gate = _load_gate(
        gate_path,
        source_fold=source,
        source_fold_sha256=args.source_fold_sha256,
    )
    exact_path = _require_file(args.exact_w, args.exact_w_sha256, "exact W")
    report_path = _require_file(
        args.exact_w_report, args.exact_w_report_sha256, "exact-W report"
    )
    source_mask_path = _require_file(
        args.source_mask, args.source_mask_sha256, "source reference mask"
    )
    exact_report, exact_authority, exact_cache, adjoint = _load_exact_w(
        exact_path=exact_path,
        exact_sha256=args.exact_w_sha256,
        report_path=report_path,
        source_mask_path=source_mask_path,
        source_mask_sha256=args.source_mask_sha256,
        scene_id=scene_id,
    )
    valid = torch.as_tensor(source["valid"]).detach().bool().cpu().reshape(-1)
    rows = torch.as_tensor(source["global_rows"]).detach().long().cpu().reshape(-1)
    fold_positive = (
        torch.as_tensor(source["population_positive_weight"])
        .detach()
        .double()
        .cpu()
        .reshape(-1)
    )
    fold_reference = (
        torch.as_tensor(source["reference_weight"]).detach().double().cpu().reshape(-1)
    )
    exact_positive = adjoint.primitive_probability.detach().double().cpu().reshape(-1)
    exact_reference = exact_cache.visible_mass.detach().double().cpu().reshape(-1)
    if any(value.shape != valid.shape for value in (
        fold_positive,
        fold_reference,
        exact_positive,
        exact_reference,
    )) or valid.numel() != exact_authority.num_gaussians:
        raise ValueError("source fold and exact-W global row domains differ")
    exact_visible = exact_reference > 0
    fold_visible = fold_reference > 0
    if not torch.equal(exact_visible, fold_visible):
        raise ValueError("source-fold and exact-W visible support differ")
    visible_reference_error = _max_relative_error(
        exact_reference[exact_visible], fold_reference[exact_visible]
    )
    visible_positive_error = float(
        (exact_positive[exact_visible] - fold_positive[exact_visible]).abs().max()
    )
    reference_absolute_error = (
        fold_reference[exact_visible] - exact_reference[exact_visible]
    ).abs()
    reference_failed_rows = reference_absolute_error > (
        SOURCE_EXACT_W_PARITY_ATOL
        + SOURCE_EXACT_W_PARITY_RTOL * exact_reference[exact_visible].abs()
    )
    if bool(reference_failed_rows.any()):
        raise ValueError("source-fold reference mass differs from exact W")
    if not torch.allclose(
        fold_reference[exact_visible],
        exact_reference[exact_visible],
        rtol=SOURCE_EXACT_W_PARITY_RTOL,
        atol=SOURCE_EXACT_W_PARITY_ATOL,
    ):
        raise ValueError("source-fold reference mass differs from exact W")
    if not torch.allclose(
        fold_positive[exact_visible],
        exact_positive[exact_visible],
        rtol=SOURCE_EXACT_W_PARITY_RTOL,
        atol=SOURCE_EXACT_W_PARITY_ATOL,
    ):
        raise ValueError("source-fold positive fraction differs from exact-W adjoint")

    relation_path = _require_file(
        args.relation_cache, args.relation_cache_sha256, "query relation cache"
    )
    knn_path = _require_file(args.knn_cache, args.knn_cache_sha256, "query KNN cache")
    graph_sha256 = str(source.get("support_graph_sha256", ""))
    relation = load_query_diffusion_relation_cache(
        relation_path,
        expected_global_rows=rows,
        expected_source_graph_sha256=graph_sha256,
    )
    knn = load_query_diffusion_knn_cache(
        knn_path,
        expected_global_rows=rows,
        expected_source_graph_sha256=graph_sha256,
        expected_num_neighbors=200,
    )
    if (
        relation.num_global_rows != valid.numel()
        or knn.num_global_rows != valid.numel()
        or relation.xyz_sha256 != knn.xyz_sha256
    ):
        raise ValueError("query caches differ from the source primitive domain")
    premetric_path = _require_file(
        args.matched_premetric_receipt,
        args.matched_premetric_receipt_sha256,
        "matched pre-metric receipt",
    )
    _load_premetric_receipt(
        premetric_path,
        scene_id=scene_id,
        protocol_hash=protocol_hash,
        relation_sha256=args.relation_cache_sha256,
        knn_sha256=args.knn_cache_sha256,
    )
    result = compute_full_fit_quantile_gauge(
        relation.features,
        knn.neighbor_indices,
        exact_positive[rows],
        exact_reference[rows],
        t_seen_raw=RAW_SEEN_THRESHOLD,
        device=args.device,
    )
    full_probability = torch.zeros(valid.shape, dtype=torch.float32)
    full_compatibility = torch.zeros(valid.shape, dtype=torch.float32)
    capped_positive = torch.zeros(valid.shape, dtype=torch.float32)
    full_probability[rows] = result.probability
    full_compatibility[rows] = result.query_compatibility
    capped_positive[rows] = result.capped_positive_weight
    tensors = {
        "valid": valid,
        "global_rows": rows,
        "reference_weight": exact_reference,
        "population_positive_weight": exact_positive,
        "full_fit_probability": full_probability,
        "full_fit_query_compatibility": full_compatibility,
        "full_fit_capped_positive_weight": capped_positive,
        "source_ecdf_support": result.ecdf.support,
        "source_ecdf_cumulative": result.ecdf.cumulative,
    }
    tensor_hashes = {name: tensor_sha256(value) for name, value in tensors.items()}
    parity = {
        "visible_support_exact": True,
        "visible_rows": int(exact_visible.sum()),
        "exact_visible_mass_sum": float(exact_reference.sum()),
        "fold_visible_mass_sum": float(fold_reference.sum()),
        "maximum_reference_mass_relative_error": visible_reference_error,
        "maximum_reference_mass_absolute_error": float(
            reference_absolute_error.max()
        ),
        "reference_mass_failed_rows": int(reference_failed_rows.sum()),
        "total_reference_mass_relative_error": float(
            (fold_reference.sum() - exact_reference.sum()).abs()
            / exact_reference.sum().abs().clamp_min(1e-12)
        ),
        "maximum_positive_fraction_absolute_error": visible_positive_error,
        "allclose_rtol": SOURCE_EXACT_W_PARITY_RTOL,
        "allclose_atol": SOURCE_EXACT_W_PARITY_ATOL,
        "numeric_reconciliation": (
            "independent_oof_and_native_exact_w_accumulation_paths_v1"
        ),
    }
    authority = {
        "schema_version": 1,
        "artifact_type": FULL_FIT_GAUGE_ARTIFACT_TYPE,
        "status": "sealed_full9_factorized_exact_w_source_only_quantile_gauge",
        "scene_id": scene_id,
        "protocol_hash": protocol_hash,
        "preregistration": str(prereg_path),
        "preregistration_sha256": str(args.preregistration_sha256),
        "source_fold": str(source_path),
        "source_fold_sha256": str(args.source_fold_sha256),
        "source_gate": str(gate_path),
        "source_gate_sha256": str(args.source_gate_sha256),
        "source_gate_selected_action": gate["selected_action"],
        "exact_w": str(exact_path),
        "exact_w_sha256": str(args.exact_w_sha256),
        "exact_w_report": str(report_path),
        "exact_w_report_sha256": str(args.exact_w_report_sha256),
        "exact_w_authority_sha256": exact_authority.digest,
        "exact_w_tensor_bundle_sha256": exact_cache.tensor_bundle_sha256,
        "source_mask": str(source_mask_path),
        "source_mask_sha256": str(args.source_mask_sha256),
        "source_mask_positive_pixels": int(
            load_ground_truth_mask(source_mask_path).astype(bool).sum()
        ),
        "source_exact_w_parity": parity,
        "query_diffusion_relation": str(relation_path),
        "query_diffusion_relation_sha256": str(args.relation_cache_sha256),
        "query_diffusion_knn": str(knn_path),
        "query_diffusion_knn_sha256": str(args.knn_cache_sha256),
        "query_diffusion_xyz_sha256": relation.xyz_sha256,
        "matched_premetric_receipt": str(premetric_path),
        "matched_premetric_receipt_sha256": str(
            args.matched_premetric_receipt_sha256
        ),
        "method_contract": quantile_method_contract(),
        "method_contract_sha256": json_sha256(quantile_method_contract()),
        "t_seen_raw": result.t_seen_raw,
        "t_seen_quantile": result.t_seen_quantile,
        "t_completion_quantile": COMPLETION_QUANTILE,
        "source_ecdf_total_weight": result.ecdf.total_weight,
        "source_ecdf_rows": result.ecdf.source_rows,
        "source_ecdf_unique_support_values": int(result.ecdf.support.numel()),
        "source_ecdf_definition": "sum_i(w_i*1[p_i<=s])/sum_i(w_i)",
        "source_ecdf_tie_semantics": "right_continuous_include_equal_mass",
        "source_unary_authority": "native_exact_W_adjoint_source_mask_probability",
        "source_weight_authority": "native_exact_W_visible_mass",
        "target_distribution_opened": False,
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
        "tensor_sha256": tensor_hashes,
        "device": str(torch.device(args.device)),
    }
    content_sha256 = json_sha256(authority)
    payload = {**authority, "content_sha256": content_sha256, **tensors}
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = torch.load(output, map_location="cpu", weights_only=False)
        if not isinstance(existing, Mapping) or existing.get(
            "content_sha256"
        ) != content_sha256:
            raise FileExistsError(f"refusing to overwrite different full9 gauge: {output}")
    else:
        temporary = output.with_suffix(output.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(output)
    receipt = {
        **authority,
        "content_sha256": content_sha256,
        "artifact_path": str(output),
        "artifact_sha256": file_sha256(output),
    }
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    encoded = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if receipt_path.exists() and receipt_path.read_text(encoding="utf-8") != encoded:
        raise FileExistsError(f"refusing to overwrite different full9 receipt: {receipt_path}")
    if not receipt_path.exists():
        receipt_path.write_text(encoded, encoding="utf-8")
    return receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--preregistration", required=True)
    result.add_argument("--preregistration-sha256", required=True)
    result.add_argument("--source-fold", required=True)
    result.add_argument("--source-fold-sha256", required=True)
    result.add_argument("--source-gate", required=True)
    result.add_argument("--source-gate-sha256", required=True)
    result.add_argument("--exact-w", required=True)
    result.add_argument("--exact-w-sha256", required=True)
    result.add_argument("--exact-w-report", required=True)
    result.add_argument("--exact-w-report-sha256", required=True)
    result.add_argument("--source-mask", required=True)
    result.add_argument("--source-mask-sha256", required=True)
    result.add_argument("--relation-cache", required=True)
    result.add_argument("--relation-cache-sha256", required=True)
    result.add_argument("--knn-cache", required=True)
    result.add_argument("--knn-cache-sha256", required=True)
    result.add_argument("--matched-premetric-receipt", required=True)
    result.add_argument("--matched-premetric-receipt-sha256", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--device", default="cpu")
    return result


def main() -> None:
    print(json.dumps(build(parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
