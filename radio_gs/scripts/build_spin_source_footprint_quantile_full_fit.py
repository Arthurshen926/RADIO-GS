#!/usr/bin/env python3
"""Seal the full-source K201 score gauge for SPIn quantile calibration v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import torch

from radio_gs.interfaces.prompt_responsibility_cache import tensor_sha256
from radio_gs.interfaces.query_diffusion_cache import (
    load_query_diffusion_knn_cache,
    load_query_diffusion_relation_cache,
)
from radio_gs.querying.spin_source_footprint_quantile_calibration import (
    FULL_FIT_GAUGE_ARTIFACT_TYPE,
    QUANTILE_OOF_ARTIFACT_TYPE,
    compute_full_fit_quantile_gauge,
    quantile_method_contract,
)
from radio_gs.scripts.build_spin_source_footprint_quantile_oof import (
    file_sha256,
    json_sha256,
)


def _require_file(path: str | Path, expected: str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    actual = file_sha256(resolved)
    if actual != str(expected):
        raise ValueError(f"{label} SHA-256 differs: {actual}")
    return resolved


def _load_tensor_authority(path: Path, *, artifact_type: str) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("artifact_type") != artifact_type:
        raise ValueError(f"unexpected tensor authority: {path}")
    hashes = payload.get("tensor_sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError("tensor authority lacks hashes")
    for name, expected in hashes.items():
        if name not in payload or tensor_sha256(torch.as_tensor(payload[name])) != expected:
            raise ValueError(f"tensor authority changed: {name}")
    return payload


def _load_seen_threshold(
    path: Path,
    *,
    scene_id: str,
    protocol_hash: str,
) -> float:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(receipt, Mapping) or receipt.get("artifact_type") != (
        "nvos_pre_metric_prediction_receipt_v1"
    ):
        raise ValueError("unexpected full-fit pre-metric receipt")
    if receipt.get("scene_id") != scene_id or receipt.get("protocol_hash") != protocol_hash:
        raise ValueError("full-fit receipt scene/protocol differs")
    if receipt.get("sealed_before_target_ground_truth_open") is not True or any(
        receipt.get(key) is not False
        for key in ("target_rgb_opened", "target_mask_opened", "target_metric_opened")
    ):
        raise ValueError("full-fit pre-metric receipt is not target blind")
    method = receipt.get("method_contract")
    if not isinstance(method, Mapping):
        raise ValueError("full-fit receipt lacks method contract")
    threshold = float(method.get("score_threshold", float("nan")))
    if not 0 < threshold < 1:
        raise ValueError("full-fit receipt seen threshold is invalid")
    query = method.get("query_conditioned_diffusion")
    if not isinstance(query, Mapping) or query.get("kernel") != "ludvig_release_compat":
        raise ValueError("full-fit receipt is not the matched K201 interface")
    return threshold


def build(args: argparse.Namespace) -> dict[str, object]:
    prereg_path = _require_file(
        args.preregistration, args.preregistration_sha256, "v2 preregistration"
    )
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    if not isinstance(prereg, Mapping) or prereg.get("registration") != (
        "spin_source_footprint_crossfit_quantile_calibration_v2"
    ):
        raise ValueError("unexpected SPIn quantile preregistration")
    oof_path = _require_file(
        args.quantile_oof_authority,
        args.quantile_oof_authority_sha256,
        "quantile OOF authority",
    )
    oof = _load_tensor_authority(
        oof_path, artifact_type=QUANTILE_OOF_ARTIFACT_TYPE
    )
    if oof.get("stable") is not True or oof.get("status") != (
        "pass_stable_source_only_quantile_gauge"
    ):
        raise ValueError("quantile OOF authority did not pass stability")
    if dict(oof.get("method_contract", {})) != quantile_method_contract():
        raise ValueError("quantile OOF method contract differs")
    source_path = _require_file(
        args.source_authority, args.source_authority_sha256, "source authority"
    )
    source = _load_tensor_authority(
        source_path,
        artifact_type="spin_source_footprint_crossfit_matched_oof_fold_v1",
    )
    scene_id = str(source.get("scene_id", ""))
    protocol_hash = str(source.get("protocol_hash", ""))
    if scene_id != oof.get("scene_id") or protocol_hash != oof.get("protocol_hash"):
        raise ValueError("source and quantile OOF scene/protocol differ")
    if any(
        source.get(key) is not False
        for key in ("target_rgb_opened", "target_mask_opened", "target_metric_computed")
    ):
        raise ValueError("source authority violates target-blind safety")
    valid = torch.as_tensor(source["valid"]).detach().bool().cpu().reshape(-1)
    rows = torch.as_tensor(source["global_rows"]).detach().long().cpu().reshape(-1)
    positive = (
        torch.as_tensor(source["population_positive_weight"])
        .detach()
        .float()
        .cpu()
        .reshape(-1)
    )
    reference = (
        torch.as_tensor(source["reference_weight"])
        .detach()
        .float()
        .cpu()
        .reshape(-1)
    )
    if not torch.equal(rows, torch.where(valid)[0]) or any(
        value.shape != valid.shape for value in (positive, reference)
    ):
        raise ValueError("full source evidence does not align with valid rows")
    relation_path = _require_file(
        args.relation_cache, args.relation_cache_sha256, "query relation cache"
    )
    knn_path = _require_file(
        args.knn_cache, args.knn_cache_sha256, "query KNN cache"
    )
    if source.get("query_diffusion_relation_sha256") != args.relation_cache_sha256 or (
        source.get("query_diffusion_knn_sha256") != args.knn_cache_sha256
    ):
        raise ValueError("source authority binds different query caches")
    graph_sha256 = str(source.get("support_graph_sha256", ""))
    relation = load_query_diffusion_relation_cache(
        relation_path,
        expected_global_rows=rows,
        expected_source_graph_sha256=graph_sha256,
    )
    # The relation/KNN loaders already bind global rows, graph lineage and xyz.
    # Capability lineage is additionally fixed by the input source tensor hash;
    # older relation caches do not expose a stable path in the matched fold.
    knn = load_query_diffusion_knn_cache(
        knn_path,
        expected_global_rows=rows,
        expected_source_graph_sha256=graph_sha256,
        expected_num_neighbors=200,
    )
    if relation.num_global_rows != valid.numel() or knn.num_global_rows != valid.numel():
        raise ValueError("query caches differ from the primitive domain")
    if relation.xyz_sha256 != knn.xyz_sha256:
        raise ValueError("query relation and KNN geometry differ")
    premetric_path = _require_file(
        args.full_fit_pre_metric_receipt,
        args.full_fit_pre_metric_receipt_sha256,
        "full-fit pre-metric receipt",
    )
    t_seen_raw = _load_seen_threshold(
        premetric_path,
        scene_id=scene_id,
        protocol_hash=protocol_hash,
    )
    result = compute_full_fit_quantile_gauge(
        relation.features,
        knn.neighbor_indices,
        positive[rows],
        reference[rows],
        t_seen_raw=t_seen_raw,
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
        "reference_weight": reference,
        "population_positive_weight": positive,
        "full_fit_probability": full_probability,
        "full_fit_query_compatibility": full_compatibility,
        "full_fit_capped_positive_weight": capped_positive,
        "source_ecdf_support": result.ecdf.support,
        "source_ecdf_cumulative": result.ecdf.cumulative,
    }
    tensor_hashes = {name: tensor_sha256(value) for name, value in tensors.items()}
    authority = {
        "schema_version": 1,
        "artifact_type": FULL_FIT_GAUGE_ARTIFACT_TYPE,
        "status": "sealed_full_fit_source_only_quantile_gauge",
        "scene_id": scene_id,
        "protocol_hash": protocol_hash,
        "preregistration": str(prereg_path),
        "preregistration_sha256": str(args.preregistration_sha256),
        "quantile_oof_authority": str(oof_path),
        "quantile_oof_authority_sha256": str(args.quantile_oof_authority_sha256),
        "source_authority": str(source_path),
        "source_authority_sha256": str(args.source_authority_sha256),
        "query_diffusion_relation": str(relation_path),
        "query_diffusion_relation_sha256": str(args.relation_cache_sha256),
        "query_diffusion_knn": str(knn_path),
        "query_diffusion_knn_sha256": str(args.knn_cache_sha256),
        "query_diffusion_xyz_sha256": relation.xyz_sha256,
        "full_fit_pre_metric_receipt": str(premetric_path),
        "full_fit_pre_metric_receipt_sha256": str(
            args.full_fit_pre_metric_receipt_sha256
        ),
        "method_contract": quantile_method_contract(),
        "method_contract_sha256": json_sha256(quantile_method_contract()),
        "t_seen_raw": result.t_seen_raw,
        "t_seen_quantile": result.t_seen_quantile,
        "t_completion_quantile": float(oof["t_completion_quantile"]),
        "source_ecdf_total_weight": result.ecdf.total_weight,
        "source_ecdf_rows": result.ecdf.source_rows,
        "source_ecdf_unique_support_values": int(result.ecdf.support.numel()),
        "source_ecdf_definition": "sum_i(w_i*1[p_i<=s])/sum_i(w_i)",
        "source_ecdf_tie_semantics": "right_continuous_include_equal_mass",
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
        if not isinstance(existing, Mapping) or existing.get("content_sha256") != content_sha256:
            raise FileExistsError(f"refusing to overwrite different full-fit gauge: {output}")
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
        raise FileExistsError(f"refusing to overwrite different full-fit receipt: {receipt_path}")
    if not receipt_path.exists():
        receipt_path.write_text(encoded, encoding="utf-8")
    return receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--preregistration", required=True)
    result.add_argument("--preregistration-sha256", required=True)
    result.add_argument("--quantile-oof-authority", required=True)
    result.add_argument("--quantile-oof-authority-sha256", required=True)
    result.add_argument("--source-authority", required=True)
    result.add_argument("--source-authority-sha256", required=True)
    result.add_argument("--relation-cache", required=True)
    result.add_argument("--relation-cache-sha256", required=True)
    result.add_argument("--knn-cache", required=True)
    result.add_argument("--knn-cache-sha256", required=True)
    result.add_argument("--full-fit-pre-metric-receipt", required=True)
    result.add_argument("--full-fit-pre-metric-receipt-sha256", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--device", default="cpu")
    return result


def main() -> None:
    print(json.dumps(build(parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
