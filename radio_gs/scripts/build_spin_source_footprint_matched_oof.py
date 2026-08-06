#!/usr/bin/env python3
"""Build one target-blind SPIn matched-interface footprint OOF fold."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch

from radio_gs.interfaces.prompt_responsibility_cache import tensor_sha256
from radio_gs.interfaces.query_diffusion_cache import (
    load_query_diffusion_knn_cache,
    load_query_diffusion_relation_cache,
)
from radio_gs.querying.spin_source_footprint_visibility_calibration import (
    MATCHED_OOF_ARTIFACT_TYPE,
    compute_matched_oof_support,
    matched_oof_method_contract,
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _raw_tensor_sha256(value: torch.Tensor) -> str:
    """Replay the legacy source-OOF tensor digest (raw contiguous bytes)."""

    array = torch.as_tensor(value).detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _require_file_sha(path: str | Path, expected: str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    actual = file_sha256(resolved)
    if actual != str(expected):
        raise ValueError(f"{label} SHA-256 differs: {actual}")
    return resolved


def _load_preregistration(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("registration") != (
        "spin_source_footprint_crossfit_visibility_calibration_v1"
    ):
        raise ValueError("unexpected SPIn visibility-calibration preregistration")
    execution = payload.get("execution")
    if not isinstance(execution, Mapping) or execution.get("cpu_first") is not True:
        raise ValueError("SPIn visibility preregistration does not require CPU first")
    return payload


def _validate_source_fold(
    payload: Mapping[str, object], *, expected_fold: int
) -> tuple[torch.Tensor, ...]:
    if payload.get("artifact_type") != (
        "source_observation_surface_safe_footprint_oof_fold_v1"
    ):
        raise ValueError("source fold is not the frozen footprint OOF artifact")
    if int(payload.get("heldout_fold", -1)) != int(expected_fold):
        raise ValueError("source fold identity differs")
    if any(
        payload.get(key) is not False
        for key in ("target_rgb_opened", "target_mask_opened", "target_metric_computed")
    ):
        raise ValueError("source fold is not target blind")
    required = (
        "valid",
        "global_rows",
        "fold_ids",
        "observed",
        "heldout",
        "reference_weight",
        "population_positive_weight",
        "population_negative_weight",
        "surface_safe_propagated_probability",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"source fold lacks tensors: {missing}")
    hashes = payload.get("tensor_sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError("source fold lacks tensor hash authority")
    for name, expected in hashes.items():
        if name not in payload or _raw_tensor_sha256(torch.as_tensor(payload[name])) != expected:
            raise ValueError(f"source fold tensor changed: {name}")
    valid = torch.as_tensor(payload["valid"]).detach().bool().cpu().reshape(-1)
    rows = torch.as_tensor(payload["global_rows"]).detach().long().cpu().reshape(-1)
    fold_ids = torch.as_tensor(payload["fold_ids"]).detach().long().cpu().reshape(-1)
    observed = torch.as_tensor(payload["observed"]).detach().bool().cpu().reshape(-1)
    heldout = torch.as_tensor(payload["heldout"]).detach().bool().cpu().reshape(-1)
    reference = torch.as_tensor(payload["reference_weight"]).detach().float().cpu().reshape(-1)
    positive = (
        torch.as_tensor(payload["population_positive_weight"])
        .detach()
        .float()
        .cpu()
        .reshape(-1)
    )
    negative = (
        torch.as_tensor(payload["population_negative_weight"])
        .detach()
        .float()
        .cpu()
        .reshape(-1)
    )
    k16 = (
        torch.as_tensor(payload["surface_safe_propagated_probability"])
        .detach()
        .float()
        .cpu()
        .reshape(-1)
    )
    shape = valid.shape
    if any(value.shape != shape for value in (fold_ids, observed, heldout, reference, positive, negative, k16)):
        raise ValueError("source fold tensors do not align globally")
    if not torch.equal(rows, torch.where(valid)[0]):
        raise ValueError("source fold global rows differ from valid rows")
    if not torch.equal(heldout, valid & (fold_ids == int(expected_fold))):
        raise ValueError("source fold heldout rows differ from footprint folds")
    if not bool((heldout & observed).any()):
        raise ValueError("source fold has no observed held-out rows")
    return valid, rows, fold_ids, observed, heldout, reference, positive, negative, k16


def build(args: argparse.Namespace) -> dict[str, object]:
    prereg_path = _require_file_sha(
        args.preregistration, args.preregistration_sha256, "preregistration"
    )
    _load_preregistration(prereg_path)
    source_path = _require_file_sha(
        args.source_fold, args.source_fold_sha256, "source footprint OOF fold"
    )
    relation_path = _require_file_sha(
        args.relation_cache, args.relation_cache_sha256, "query relation cache"
    )
    knn_path = _require_file_sha(
        args.knn_cache, args.knn_cache_sha256, "query KNN cache"
    )
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(source, Mapping):
        raise ValueError("source footprint OOF fold is not a mapping")
    fold = int(args.fold)
    (
        valid,
        rows,
        fold_ids,
        observed,
        heldout,
        reference,
        positive,
        negative,
        k16,
    ) = _validate_source_fold(source, expected_fold=fold)
    graph_sha256 = str(source.get("support_graph_sha256", ""))
    capability_path = str(source.get("capability_cache", ""))
    if len(graph_sha256) != 64 or not capability_path:
        raise ValueError("source fold lacks graph/capability lineage")
    relation = load_query_diffusion_relation_cache(
        relation_path,
        expected_global_rows=rows,
        expected_source_graph_sha256=graph_sha256,
        expected_source_capability_cache=capability_path,
    )
    knn = load_query_diffusion_knn_cache(
        knn_path,
        expected_global_rows=rows,
        expected_source_graph_sha256=graph_sha256,
        expected_num_neighbors=200,
    )
    if relation.num_global_rows != valid.numel() or knn.num_global_rows != valid.numel():
        raise ValueError("matched caches differ from the global primitive domain")
    if relation.xyz_sha256 != knn.xyz_sha256:
        raise ValueError("matched relation and KNN geometry authorities differ")
    local_heldout = heldout[rows]
    matched = compute_matched_oof_support(
        relation.features,
        knn.neighbor_indices,
        positive[rows],
        reference[rows],
        local_heldout,
        device=args.device,
    )
    probability = torch.zeros(valid.shape, dtype=torch.float32)
    compatibility = torch.zeros(valid.shape, dtype=torch.float32)
    training_positive = torch.zeros(valid.shape, dtype=torch.float32)
    training_reference = torch.zeros(valid.shape, dtype=torch.float32)
    probability[rows] = matched.probability
    compatibility[rows] = matched.query_compatibility
    training_positive[rows] = matched.training_positive_weight
    training_reference[rows] = matched.training_reference_weight
    heldout_positive_sum = float(training_positive[heldout].sum())
    heldout_reference_sum = float(training_reference[heldout].sum())
    if heldout_positive_sum != 0.0 or heldout_reference_sum != 0.0:
        raise RuntimeError("held-out matched evidence survived global expansion")

    tensors = {
        "valid": valid,
        "global_rows": rows,
        "fold_ids": fold_ids,
        "observed": observed,
        "heldout": heldout,
        "reference_weight": reference,
        "population_positive_weight": positive,
        "population_negative_weight": negative,
        "surface_safe_k16_propagated_probability": k16,
        "matched_query_diffusion_probability": probability,
        "matched_query_compatibility": compatibility,
        "training_positive_weight": training_positive,
        "training_reference_weight": training_reference,
    }
    tensor_hashes = {name: tensor_sha256(value) for name, value in tensors.items()}
    method_contract = matched_oof_method_contract()
    authority = {
        "schema_version": 1,
        "artifact_type": MATCHED_OOF_ARTIFACT_TYPE,
        "scene_id": str(source.get("scene_id", "")),
        "protocol_hash": str(source.get("protocol_hash", "")),
        "heldout_fold": fold,
        "num_folds": 3,
        "source_fold": str(source_path),
        "source_fold_sha256": str(args.source_fold_sha256),
        "capability_cache_sha256": str(source.get("capability_cache_sha256", "")),
        "support_graph_sha256": graph_sha256,
        "source_evidence_authority_sha256": str(
            source.get("source_evidence_authority_sha256", "")
        ),
        "source_footprint_fold_authority_sha256": str(
            source.get("source_footprint_fold_authority_sha256", "")
        ),
        "source_footprint_fold_authority_tensor_bundle_sha256": str(
            source.get("source_footprint_fold_authority_tensor_bundle_sha256", "")
        ),
        "query_diffusion_knn": str(knn_path),
        "query_diffusion_knn_sha256": str(args.knn_cache_sha256),
        "query_diffusion_relation": str(relation_path),
        "query_diffusion_relation_sha256": str(args.relation_cache_sha256),
        "query_diffusion_xyz_sha256": relation.xyz_sha256,
        "preregistration": str(prereg_path),
        "preregistration_sha256": str(args.preregistration_sha256),
        "method_contract": method_contract,
        "method_contract_sha256": json_sha256(method_contract),
        "heldout_training_positive_weight_sum": heldout_positive_sum,
        "heldout_training_reference_weight_sum": heldout_reference_sum,
        "tensor_sha256": tensor_hashes,
        "device": str(torch.device(args.device)),
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
    }
    content_sha256 = json_sha256(authority)
    payload = {**authority, "content_sha256": content_sha256, **tensors}
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = torch.load(output, map_location="cpu", weights_only=False)
        if not isinstance(existing, Mapping) or existing.get("content_sha256") != content_sha256:
            raise FileExistsError(f"refusing to overwrite different matched OOF fold: {output}")
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
        raise FileExistsError(f"refusing to overwrite different matched OOF receipt: {receipt_path}")
    if not receipt_path.exists():
        receipt_path.write_text(encoded, encoding="utf-8")
    return receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--preregistration", required=True)
    result.add_argument("--preregistration-sha256", required=True)
    result.add_argument("--source-fold", required=True)
    result.add_argument("--source-fold-sha256", required=True)
    result.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    result.add_argument("--relation-cache", required=True)
    result.add_argument("--relation-cache-sha256", required=True)
    result.add_argument("--knn-cache", required=True)
    result.add_argument("--knn-cache-sha256", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--device", default="cpu")
    return result


def main() -> None:
    receipt = build(parser().parse_args())
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
