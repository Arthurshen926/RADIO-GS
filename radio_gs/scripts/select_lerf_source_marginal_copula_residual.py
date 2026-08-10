#!/usr/bin/env python3
"""Select one global LERF copula residual from source-only scene results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from radio_gs.evaluation.lerf_source_text_response_ranking import paired_source_gate
from radio_gs.scripts import eval_lerf_source_marginal_copula_residual_grid as evaluator
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    write_frozen_json,
)


RESULT_SCHEMA = "radio_gs.lerf_source_marginal_copula_selection.v1"
IMPLEMENTATION = file_record(Path(__file__).resolve())


def _load_result(path: str | Path, digest: str) -> tuple[dict[str, Any], dict[str, str]]:
    value, actual, source = load_json_object(
        path, expected_sha256=digest, label="source marginal-copula scene result"
    )
    access = value.get("access_audit") if isinstance(value, Mapping) else None
    if (
        value.get("schema") != evaluator.RESULT_SCHEMA
        or value.get("schema_version") != 1
        or value.get("status") != "complete_source_only_no_target_access"
        or value.get("implementation") != evaluator.IMPLEMENTATION
        or value.get("interface_contract") != evaluator.CONTRACT
        or not isinstance(access, Mapping)
        or access.get("benchmark_queries_opened") is not False
        or access.get("benchmark_masks_or_labels_opened") is not False
        or access.get("target_metric_executed") is not False
        or access.get("target_metric_execution_authorized") is not False
    ):
        raise ValueError("source marginal-copula scene result differs")
    return dict(value), {"path": str(source), "sha256": actual}


def _topology(scene: Mapping[str, Any], policy_id: str) -> dict[str, Any]:
    control = scene["control_support_diagnostics"]
    candidate = scene["diagnostics"][policy_id]
    boundary_delta = float(candidate["top_decile_boundary_f_mean"]) - float(
        control["top_decile_boundary_f_mean"]
    )
    component_improvement = float(
        control["top_decile_component_abs_error_mean"]
    ) - float(candidate["top_decile_component_abs_error_mean"])
    exact = (
        candidate["marginal_exact_every_frame"] is True
        and int(candidate["selected_count_error_sum"]) == 0
    )
    return {
        "control_boundary_f": float(control["top_decile_boundary_f_mean"]),
        "candidate_boundary_f": float(candidate["top_decile_boundary_f_mean"]),
        "boundary_f_delta": boundary_delta,
        "control_component_abs_error": float(
            control["top_decile_component_abs_error_mean"]
        ),
        "candidate_component_abs_error": float(
            candidate["top_decile_component_abs_error_mean"]
        ),
        "component_abs_error_improvement": component_improvement,
        "marginal_and_selected_count_exact": exact,
        "topology_no_harm": boundary_delta >= 0.0 and component_improvement >= 0.0,
        "support_units": int(candidate["support_units"]),
    }


def _candidate_is_eligible(
    response_gate: Mapping[str, Any], topology: Mapping[str, Mapping[str, Any]]
) -> bool:
    decision = response_gate.get("decision")
    return bool(
        isinstance(decision, Mapping)
        and decision.get("candidate_eligible_for_next_source_gate") is True
        and topology
        and all(
            row.get("topology_no_harm") is True
            and row.get("marginal_and_selected_count_exact") is True
            for row in topology.values()
        )
    )


def _select_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [row for row in rows if row.get("eligible") is True]
    return max(
        eligible,
        key=lambda row: float(
            row["response_gate"]["pooled"]["candidate"]["ranking_spearman_mean"]
        ),
        default=None,
    )


def select(
    preregistration_path: str | Path,
    preregistration_sha256: str,
    *,
    ramen_path: str | Path,
    ramen_sha256: str,
    teatime_path: str | Path,
    teatime_sha256: str,
) -> dict[str, Any]:
    prereg, prereg_record = evaluator._load_preregistration(
        preregistration_path, preregistration_sha256
    )
    scenes: dict[str, dict[str, Any]] = {}
    records: dict[str, dict[str, str]] = {}
    for scene_id, path, digest in (
        ("ramen", ramen_path, ramen_sha256),
        ("teatime", teatime_path, teatime_sha256),
    ):
        scene, record = _load_result(path, digest)
        if (
            scene.get("scene_id") != scene_id
            or scene.get("split") != prereg["split"]
            or scene.get("preregistration") != prereg_record
        ):
            raise ValueError("source marginal-copula scene/preregistration differs")
        scenes[scene_id] = scene
        records[scene_id] = record

    candidate_rows: list[dict[str, Any]] = []
    for policy in prereg["policies"]:
        policy_id = str(policy["policy_id"])
        response_gate = paired_source_gate(
            [scenes[name]["control_summary"] for name in ("ramen", "teatime")],
            [
                scenes[name]["candidate_summaries"][policy_id]
                for name in ("ramen", "teatime")
            ],
            required_scene_ids=["ramen", "teatime"],
        )
        topology = {
            name: _topology(scenes[name], policy_id)
            for name in ("ramen", "teatime")
        }
        topology_pass = all(
            row["topology_no_harm"] and row["marginal_and_selected_count_exact"]
            for row in topology.values()
        )
        eligible = _candidate_is_eligible(response_gate, topology)
        candidate_rows.append(
            {
                "policy": dict(policy),
                "response_gate": response_gate,
                "topology": topology,
                "topology_and_exactness_pass": topology_pass,
                "eligible": eligible,
            }
        )
    selected = _select_candidate(candidate_rows)
    return {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "status": "passed" if selected is not None else "rejected_fallback_control",
        "implementation": IMPLEMENTATION,
        "preregistration": prereg_record,
        "input_results": records,
        "candidate_rows": candidate_rows,
        "selection": {
            "selected_policy": selected["policy"] if selected is not None else None,
            "fallback_control": selected is None,
            "eligible_for_reserved_audit90": selected is not None,
            "eligible_for_target_metric": False,
        },
        "access_audit": {
            "source_summary_results_opened": True,
            "benchmark_queries_opened": False,
            "benchmark_masks_or_labels_opened": False,
            "target_metric_executed": False,
            "target_metric_execution_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--ramen-result", required=True)
    parser.add_argument("--ramen-result-sha256", required=True)
    parser.add_argument("--teatime-result", required=True)
    parser.add_argument("--teatime-result-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    if str(output) != args.output or output.exists() or output.is_symlink():
        raise FileExistsError("source marginal-copula selection output must be new")
    result = select(
        args.preregistration,
        args.preregistration_sha256,
        ramen_path=args.ramen_result,
        ramen_sha256=args.ramen_result_sha256,
        teatime_path=args.teatime_result,
        teatime_sha256=args.teatime_result_sha256,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_frozen_json(output, result)
    print(json.dumps({**result["selection"], "output": file_record(output)}, indent=2))


if __name__ == "__main__":
    main()
