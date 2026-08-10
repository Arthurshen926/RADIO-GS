#!/usr/bin/env python3
"""Materialize the source-confirmed LERF copula candidate before metrics.

The accepted O2 cache owns the primitive axis, valid domain, score marginal,
and operating point.  The genuine-MPR cache is only a ranking proposal.  A
candidate-missing primitive is filled from the accepted score and assigned
zero reliability, making it a fixed-rank barrier instead of dropping it from
the output domain.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping

import torch

import radio_gs.interfaces.lerf_marginal_preserving_copula_residual as interface
import radio_gs.scripts.eval_lerf_source_marginal_copula_residual_grid as source_eval
import radio_gs.scripts.select_lerf_source_marginal_copula_residual as source_select
from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces.lerf_marginal_preserving_copula_residual import (
    marginal_preserving_primitive_query_scores,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    load_torch_payload,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


SCHEMA = "radio_gs.lerf_marginal_copula_external_scores.v1"
REPORT_SCHEMA = "radio_gs.lerf_marginal_copula_materialization_report.v1"
IMPLEMENTATION = file_record(Path(__file__).resolve())


def _load_selection(
    path: str | Path, digest: str, *, expected_split: str
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    value, actual, source = load_json_object(
        path,
        expected_sha256=digest,
        label=f"LERF {expected_split} marginal-copula selection",
    )
    if (
        value.get("schema") != source_select.RESULT_SCHEMA
        or value.get("schema_version") != 1
        or value.get("status") != "passed"
        or value.get("implementation") != source_select.IMPLEMENTATION
        or value.get("access_audit", {}).get("target_metric_executed") is not False
        or value.get("selection", {}).get("fallback_control") is not False
        or value.get("selection", {}).get("eligible_for_reserved_audit90") is not True
        or value.get("selection", {}).get("eligible_for_target_metric") is not False
    ):
        raise ValueError(f"LERF {expected_split} source selection differs")
    prereg_record = value.get("preregistration")
    validate_file_record(prereg_record, label=f"{expected_split} preregistration")
    prereg, observed_record = source_eval._load_preregistration(
        prereg_record["path"], prereg_record["sha256"]
    )
    if observed_record != prereg_record or prereg.get("split") != expected_split:
        raise ValueError(f"LERF {expected_split} source split differs")
    return (
        dict(value),
        {"path": str(source), "sha256": actual},
        prereg,
    )


def _confirmed_policy(
    dev: Mapping[str, Any],
    dev_record: Mapping[str, str],
    dev_prereg: Mapping[str, Any],
    audit: Mapping[str, Any],
    audit_prereg: Mapping[str, Any],
) -> dict[str, Any]:
    dev_policy = dict(dev["selection"]["selected_policy"])
    audit_policy = dict(audit["selection"]["selected_policy"])
    if dev_policy != audit_policy or audit_policy != {
        "policy_id": "b40_s05",
        "strength": 0.5,
        "maximum_rank_fraction": 0.4,
    }:
        raise ValueError("dev101 and audit90 selected policies differ")
    lock = audit_prereg.get("candidate_lock", {})
    if (
        lock.get("development_selection") != dict(dev_record)
        or lock.get("candidate_selected_before_audit_embedding_open") is not True
        or lock.get("one_shot_reserved_confirmation") is not True
        or len(audit_prereg.get("policies", [])) != 1
        or dict(audit_prereg["policies"][0]) != audit_policy
        or dev_prereg.get("interface_implementation")
        != audit_prereg.get("interface_implementation")
        or audit_prereg.get("interface_implementation") != file_record(Path(interface.__file__).resolve())
        or audit_prereg.get("interface_contract") != interface.CONTRACT
    ):
        raise ValueError("audit90 candidate lock or interface binding differs")
    return audit_policy


def _load_score_cache(
    path: str | Path, digest: str, *, label: str
) -> tuple[dict[str, Any], dict[str, str]]:
    payload, actual, source = load_torch_payload(
        path, expected_sha256=digest, map_location="cpu", label=label
    )
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain a mapping")
    scores = payload.get("query_scores")
    valid = payload.get("valid")
    xyz = payload.get("xyz")
    metadata = payload.get("metadata")
    if (
        not isinstance(scores, torch.Tensor)
        or scores.ndim != 2
        or not scores.is_floating_point()
        or not isinstance(valid, torch.Tensor)
        or valid.dtype != torch.bool
        or valid.shape != scores.shape[:1]
        or not isinstance(xyz, torch.Tensor)
        or xyz.shape != (scores.shape[0], 3)
        or not isinstance(metadata, Mapping)
        or len(metadata.get("query_names", [])) != scores.shape[1]
    ):
        raise ValueError(f"{label} score-cache contract differs")
    if not bool(torch.isfinite(scores[valid]).all()):
        raise ValueError(f"{label} has non-finite valid scores")
    return dict(payload), {"path": str(source), "sha256": actual}


def materialize(
    *,
    scene_id: str,
    accepted_path: str | Path,
    accepted_sha256: str,
    candidate_path: str | Path,
    candidate_sha256: str,
    dev_selection_path: str | Path,
    dev_selection_sha256: str,
    audit_selection_path: str | Path,
    audit_selection_sha256: str,
    output_path: str | Path,
) -> dict[str, Any]:
    dev, dev_record, dev_prereg = _load_selection(
        dev_selection_path, dev_selection_sha256, expected_split="dev"
    )
    audit, audit_record, audit_prereg = _load_selection(
        audit_selection_path, audit_selection_sha256, expected_split="audit"
    )
    policy = _confirmed_policy(dev, dev_record, dev_prereg, audit, audit_prereg)
    accepted, accepted_record = _load_score_cache(
        accepted_path, accepted_sha256, label="accepted O2 final score cache"
    )
    candidate, candidate_record = _load_score_cache(
        candidate_path, candidate_sha256, label="genuine-MPR final score cache"
    )
    accepted_scores = accepted["query_scores"].detach().float().cpu().contiguous()
    candidate_scores = candidate["query_scores"].detach().float().cpu().contiguous()
    accepted_valid = accepted["valid"].detach().bool().cpu().contiguous()
    candidate_valid = candidate["valid"].detach().bool().cpu().contiguous()
    if (
        accepted_scores.shape != candidate_scores.shape
        or not torch.equal(accepted["xyz"], candidate["xyz"])
        or list(accepted["metadata"]["query_names"])
        != list(candidate["metadata"]["query_names"])
    ):
        raise ValueError("accepted and genuine-MPR primitive/query axes differ")
    reliable = accepted_valid & candidate_valid
    filled_candidate = torch.where(
        reliable[:, None], candidate_scores, accepted_scores
    ).contiguous()
    fused = marginal_preserving_primitive_query_scores(
        accepted_scores,
        filled_candidate,
        accepted_valid,
        strength=float(policy["strength"]),
        maximum_rank_fraction=float(policy["maximum_rank_fraction"]),
        reliability=reliable.float(),
    )
    protected = accepted_valid & ~candidate_valid
    if not torch.equal(fused.scores[protected], accepted_scores[protected]):
        raise AssertionError("candidate-missing fixed-rank barrier changed")
    if not torch.equal(fused.scores[~accepted_valid], accepted_scores[~accepted_valid]):
        raise AssertionError("accepted invalid-domain row changed")

    output = Path(output_path).expanduser().resolve()
    if str(output) != str(output_path) or output.exists() or output.is_symlink():
        raise FileExistsError("LERF marginal-copula output must be new and canonical")
    metadata = dict(accepted["metadata"])
    metadata.update(
        {
            "construction": "accepted_o2_marginal_preserving_genuine_mpr_copula_v1",
            "accepted_valid_domain_authoritative": True,
            "candidate_missing_policy": "accepted_score_zero_reliability_fixed_rank_barrier",
            "policy": dict(policy),
            "source_dev101_and_audit90_confirmed": True,
            "per_scene_or_per_query_parameter": False,
        }
    )
    payload = {
        "schema": SCHEMA,
        "schema_version": 1,
        "query_scores": fused.scores.contiguous(),
        "valid": accepted_valid,
        "xyz": accepted["xyz"].detach().float().cpu().contiguous(),
        "metadata": metadata,
        "authority": {
            "implementation": IMPLEMENTATION,
            "interface": file_record(Path(interface.__file__).resolve()),
            "accepted": accepted_record,
            "candidate": candidate_record,
            "dev_selection": dev_record,
            "audit_selection": audit_record,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_torch_noclobber(output, payload)
    report = {
        "schema": REPORT_SCHEMA,
        "schema_version": 1,
        "status": "complete_source_confirmed_premetric_candidate",
        "scene_id": str(scene_id),
        "policy": dict(policy),
        "implementation": IMPLEMENTATION,
        "interface_contract": interface.CONTRACT,
        "output": file_record(output),
        "inputs": {
            "accepted": accepted_record,
            "candidate": candidate_record,
            "dev_selection": dev_record,
            "audit_selection": audit_record,
        },
        "diagnostics": {
            **asdict(fused),
            "scores": None,
            "accepted_valid_count": int(accepted_valid.sum()),
            "candidate_valid_count": int(candidate_valid.sum()),
            "reliable_candidate_count": int(reliable.sum()),
            "candidate_missing_protected_count": int(protected.sum()),
            "query_scores_sha256": tensor_sha256(fused.scores),
            "protected_rows_exact": True,
            "invalid_rows_exact": True,
        },
        "access_audit": {
            "benchmark_query_score_caches_opened": True,
            "benchmark_masks_or_labels_opened": False,
            "target_metric_opened": False,
            "target_metric_executed": False,
            "per_scene_or_per_query_tuning": False,
        },
    }
    # ``asdict`` includes the tensor; keep the report JSON compact and typed.
    report["diagnostics"].pop("scores")
    report_path = output.with_suffix(output.suffix + ".json")
    write_frozen_json(report_path, report)
    return {**report, "report": file_record(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--accepted-score-cache", required=True)
    parser.add_argument("--accepted-score-cache-sha256", required=True)
    parser.add_argument("--candidate-score-cache", required=True)
    parser.add_argument("--candidate-score-cache-sha256", required=True)
    parser.add_argument("--dev-selection", required=True)
    parser.add_argument("--dev-selection-sha256", required=True)
    parser.add_argument("--audit-selection", required=True)
    parser.add_argument("--audit-selection-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = materialize(
        scene_id=args.scene_id,
        accepted_path=args.accepted_score_cache,
        accepted_sha256=args.accepted_score_cache_sha256,
        candidate_path=args.candidate_score_cache,
        candidate_sha256=args.candidate_score_cache_sha256,
        dev_selection_path=args.dev_selection,
        dev_selection_sha256=args.dev_selection_sha256,
        audit_selection_path=args.audit_selection,
        audit_selection_sha256=args.audit_selection_sha256,
        output_path=args.output,
    )
    print(json.dumps({"output": result["output"], "report": result["report"]}, indent=2))


if __name__ == "__main__":
    main()

