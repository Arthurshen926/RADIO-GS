#!/usr/bin/env python3
"""Freeze a query-free alpha selection from one formal interpolation diagnostic.

This selector is intentionally incapable of accepting another diagnostic.  It
uses only per-seed Surface metrics and aggregate target-blind ``fit`` metrics.
Held-out ``dev_posthoc`` values are never extracted, copied, compared, or
included in the output, and no audit artifact is opened.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from radio_gs.utils.immutable_artifacts import (
    load_json_object,
    sha256_file,
    validate_file_record,
    write_frozen_json,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "surface_readout_weight_interpolation_frozen_selection"
FORMAL_DIAGNOSTIC_PATH = Path(
    "/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260801/"
    "surface_readout_weight_interpolation_pareto_joint_c1024_src50c48dfab98e.json"
)
FORMAL_DIAGNOSTIC_SHA256 = (
    "a7bc7b6715cfd840a18a80c09f936af193fdb7c9572671a0ca02ea87b098c225"
)
FORMAL_DIAGNOSTIC_CONTRACT_SHA256 = (
    "1c283cfb8c79dc56dbd1136062cb0c090fdb8811750d56cf1996583b403c7152"
)
FORMAL_DIAGNOSTIC_ARTIFACT_TYPE = (
    "surface_readout_weight_interpolation_pareto_diagnostic"
)
FIXED_ALPHAS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
REQUIRED_SEEDS = (0, 1, 2)
SURFACE_METRICS = (
    "summary_token_cosine",
    "mean_descriptor_cosine",
    "all_view_descriptor_cosine",
    "relation_fidelity",
)
SURFACE_NONINFERIORITY_TOLERANCE = 0.002
FIT_SUPPORT_METRIC = "text_support_top1_agreement"
FIT_ERROR_METRIC = "text_response_smooth_l1"
EXPECTED_FORMAL_SELECTED_ALPHA = 0.1
SELECTION_POLICY = "minimum_positive_feasible_alpha_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: object, *, label: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label} must be finite numeric",
    )
    return float(value)


def _cpu_only_preflight() -> None:
    _require(
        os.environ.get("CUDA_VISIBLE_DEVICES") in {"", "-1"},
        "set CUDA_VISIBLE_DEVICES='' (or -1) for this CPU-only selector",
    )


def _sha_record(value: object, *, label: str) -> dict[str, str]:
    _require(isinstance(value, Mapping), f"{label} is not a mapping")
    record = {"path": value.get("path"), "sha256": value.get("sha256")}
    path = validate_file_record(record, label=label)
    return {"path": str(path), "sha256": str(record["sha256"])}


def _validate_formal_diagnostic_identity(
    path: Path,
    expected_sha256: str,
) -> dict[str, Any]:
    _require(path.is_absolute(), "diagnostic path must be absolute")
    _require(
        path == FORMAL_DIAGNOSTIC_PATH,
        "selector accepts only the frozen formal interpolation diagnostic path",
    )
    _require(
        expected_sha256 == FORMAL_DIAGNOSTIC_SHA256,
        "diagnostic SHA argument differs from the frozen formal SHA",
    )
    payload, observed_sha, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="formal interpolation diagnostic",
    )
    _require(source == path, "formal diagnostic resolved another path")
    _require(observed_sha == FORMAL_DIAGNOSTIC_SHA256, "formal diagnostic SHA drifted")
    _require(
        payload.get("schema_version") == 1
        and payload.get("artifact_type") == FORMAL_DIAGNOSTIC_ARTIFACT_TYPE
        and payload.get("status")
        == "diagnostic_only_no_alpha_selected_no_checkpoint_created"
        and payload.get("device") == "cpu"
        and payload.get("fixed_alphas") == list(FIXED_ALPHAS)
        and payload.get("diagnostic_contract_sha256")
        == FORMAL_DIAGNOSTIC_CONTRACT_SHA256,
        "formal diagnostic contract differs",
    )
    diagnostic_selection = payload.get("selection_contract")
    _require(
        isinstance(diagnostic_selection, Mapping)
        and diagnostic_selection.get("selected_alpha") is None
        and diagnostic_selection.get("checkpoint_emitted") is False
        and diagnostic_selection.get("promotion_decision_emitted") is False
        and diagnostic_selection.get("audit_opened") is False,
        "diagnostic was already selected/promoted/audited",
    )
    return payload


def _selection_view(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Extract only selection-eligible fields; never touch ``dev_posthoc``."""

    raw_seeds = payload.get("per_seed")
    raw_aggregate = payload.get("aggregate_seed_mean")
    _require(isinstance(raw_seeds, list), "diagnostic per_seed must be a list")
    _require(isinstance(raw_aggregate, list), "diagnostic aggregate must be a list")
    _require(len(raw_seeds) == len(REQUIRED_SEEDS), "diagnostic seed count differs")

    per_seed: dict[int, dict[str, Any]] = {}
    for raw_seed in raw_seeds:
        _require(isinstance(raw_seed, Mapping), "diagnostic seed row is invalid")
        seed = raw_seed.get("seed")
        _require(seed in REQUIRED_SEEDS and seed not in per_seed, "diagnostic seed differs")
        compatibility = raw_seed.get("interpolation_compatibility")
        _require(
            isinstance(compatibility, Mapping)
            and compatibility.get("seed") == seed
            and compatibility.get("tensor_count") == 24,
            "diagnostic interpolation compatibility differs",
        )
        control = _sha_record(raw_seed.get("control"), label=f"seed {seed} control checkpoint")
        candidate = _sha_record(
            raw_seed.get("candidate"), label=f"seed {seed} candidate checkpoint"
        )
        raw_points = raw_seed.get("points")
        _require(isinstance(raw_points, list), f"seed {seed} points must be a list")
        points: dict[float, dict[str, dict[str, float]]] = {}
        for raw_point in raw_points:
            _require(isinstance(raw_point, Mapping), f"seed {seed} point is invalid")
            alpha = _finite(raw_point.get("alpha"), label=f"seed {seed} alpha")
            _require(alpha in FIXED_ALPHAS and alpha not in points, f"seed {seed} alpha differs")
            raw_surface = raw_point.get("surface")
            raw_fit = raw_point.get("fit")
            _require(isinstance(raw_surface, Mapping), "point Surface metrics are missing")
            _require(isinstance(raw_fit, Mapping), "point fit metrics are missing")
            points[alpha] = {
                "surface": {
                    metric: _finite(
                        raw_surface.get(metric),
                        label=f"seed {seed} alpha {alpha} Surface {metric}",
                    )
                    for metric in SURFACE_METRICS
                },
                "fit": {
                    FIT_SUPPORT_METRIC: _finite(
                        raw_fit.get(FIT_SUPPORT_METRIC),
                        label=f"seed {seed} alpha {alpha} fit support",
                    ),
                    FIT_ERROR_METRIC: _finite(
                        raw_fit.get(FIT_ERROR_METRIC),
                        label=f"seed {seed} alpha {alpha} fit SmoothL1",
                    ),
                },
            }
        _require(set(points) == set(FIXED_ALPHAS), f"seed {seed} alpha grid differs")
        per_seed[int(seed)] = {
            "control": control,
            "candidate": candidate,
            "points": points,
        }
    _require(set(per_seed) == set(REQUIRED_SEEDS), "diagnostic seeds do not cover 0/1/2")

    aggregate: dict[float, dict[str, float]] = {}
    for raw_row in raw_aggregate:
        _require(isinstance(raw_row, Mapping), "aggregate alpha row is invalid")
        alpha = _finite(raw_row.get("alpha"), label="aggregate alpha")
        _require(alpha in FIXED_ALPHAS and alpha not in aggregate, "aggregate alpha differs")
        raw_fit = raw_row.get("fit")
        _require(isinstance(raw_fit, Mapping), "aggregate fit metrics are missing")
        aggregate[alpha] = {
            FIT_SUPPORT_METRIC: _finite(
                raw_fit.get(FIT_SUPPORT_METRIC), label=f"alpha {alpha} aggregate fit support"
            ),
            FIT_ERROR_METRIC: _finite(
                raw_fit.get(FIT_ERROR_METRIC), label=f"alpha {alpha} aggregate fit SmoothL1"
            ),
        }
    _require(set(aggregate) == set(FIXED_ALPHAS), "aggregate alpha grid differs")
    for alpha in FIXED_ALPHAS:
        for metric in (FIT_SUPPORT_METRIC, FIT_ERROR_METRIC):
            recomputed = sum(
                per_seed[seed]["points"][alpha]["fit"][metric]
                for seed in REQUIRED_SEEDS
            ) / len(REQUIRED_SEEDS)
            _require(
                math.isclose(recomputed, aggregate[alpha][metric], rel_tol=1e-9, abs_tol=1e-9),
                f"aggregate alpha {alpha} {metric} does not match per-seed mean",
            )
    return {"per_seed": per_seed, "aggregate_fit": aggregate}


def select_from_view(view: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the predeclared no-dev feasibility and minimum-positive rule."""

    per_seed = view["per_seed"]
    aggregate = view["aggregate_fit"]
    base_fit = aggregate[0.0]
    per_alpha = []
    feasible_alphas: list[float] = []
    for alpha in FIXED_ALPHAS:
        seed_checks: dict[int, dict[str, bool]] = {}
        for seed in REQUIRED_SEEDS:
            baseline = per_seed[seed]["points"][0.0]["surface"]
            current = per_seed[seed]["points"][alpha]["surface"]
            seed_checks[seed] = {
                metric: current[metric]
                >= baseline[metric] - SURFACE_NONINFERIORITY_TOLERANCE
                for metric in SURFACE_METRICS
            }
        surface_passes = all(
            all(checks.values()) for checks in seed_checks.values()
        )
        fit_support_passes = (
            alpha == 0.0
            or aggregate[alpha][FIT_SUPPORT_METRIC] > base_fit[FIT_SUPPORT_METRIC]
        )
        fit_error_passes = (
            alpha == 0.0
            or aggregate[alpha][FIT_ERROR_METRIC] <= base_fit[FIT_ERROR_METRIC]
        )
        feasible = surface_passes and fit_support_passes and fit_error_passes
        if feasible:
            feasible_alphas.append(alpha)
        per_alpha.append(
            {
                "alpha": alpha,
                "surface_checks_by_seed": [
                    {"seed": seed, "checks": seed_checks[seed]}
                    for seed in REQUIRED_SEEDS
                ],
                "surface_all_seed_passes": surface_passes,
                "aggregate_fit_support_strictly_improves": fit_support_passes,
                "aggregate_fit_smooth_l1_noninferior": fit_error_passes,
                "feasible": feasible,
            }
        )
    positive = [alpha for alpha in feasible_alphas if alpha > 0.0]
    _require(positive, "no positive alpha satisfies the frozen feasibility rule")
    selected = min(positive)
    _require(
        positive == [EXPECTED_FORMAL_SELECTED_ALPHA]
        and selected == EXPECTED_FORMAL_SELECTED_ALPHA,
        "formal feasible set no longer uniquely selects alpha=0.1",
    )
    return {
        "per_alpha": per_alpha,
        "feasible_alphas": feasible_alphas,
        "positive_feasible_alphas": positive,
        "selected_alpha": selected,
    }


def freeze_selection(args: argparse.Namespace) -> dict[str, Any]:
    _cpu_only_preflight()
    diagnostic_path = Path(args.diagnostic)
    diagnostic = _validate_formal_diagnostic_identity(
        diagnostic_path,
        str(args.diagnostic_sha256),
    )
    view = _selection_view(diagnostic)
    decision = select_from_view(view)
    alpha = decision["selected_alpha"]
    per_seed_output = []
    for seed in REQUIRED_SEEDS:
        record = view["per_seed"][seed]
        baseline = record["points"][0.0]["surface"]
        selected = record["points"][alpha]["surface"]
        deltas = {
            metric: selected[metric] - baseline[metric] for metric in SURFACE_METRICS
        }
        checks = {
            metric: selected[metric]
            >= baseline[metric] - SURFACE_NONINFERIORITY_TOLERANCE
            for metric in SURFACE_METRICS
        }
        per_seed_output.append(
            {
                "seed": seed,
                "selected_alpha": alpha,
                "control_checkpoint": record["control"],
                "candidate_checkpoint": record["candidate"],
                "interpolation": {
                    "formula": "theta_alpha=(1-alpha)*theta_control+alpha*theta_candidate",
                    "pairing": "same_seed_only",
                    "materialized_checkpoint": None,
                },
                "alpha0_surface": baseline,
                "selected_surface": selected,
                "selected_minus_alpha0": deltas,
                "minimum_allowed_surface": {
                    metric: baseline[metric] - SURFACE_NONINFERIORITY_TOLERANCE
                    for metric in SURFACE_METRICS
                },
                "surface_checks": checks,
                "surface_passes": all(checks.values()),
            }
        )
    base_fit = view["aggregate_fit"][0.0]
    selected_fit = view["aggregate_fit"][alpha]
    aggregate_fit = {
        "scope": "aggregate_seed_mean",
        "alpha0": base_fit,
        "selected": selected_fit,
        "selected_minus_alpha0": {
            metric: selected_fit[metric] - base_fit[metric]
            for metric in (FIT_SUPPORT_METRIC, FIT_ERROR_METRIC)
        },
        "checks": {
            "support_top1_strictly_improves": (
                selected_fit[FIT_SUPPORT_METRIC] > base_fit[FIT_SUPPORT_METRIC]
            ),
            "smooth_l1_noninferior": (
                selected_fit[FIT_ERROR_METRIC] <= base_fit[FIT_ERROR_METRIC]
            ),
        },
    }
    selector_path = Path(__file__).resolve()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": "frozen_query_free_selection_audit_unopened",
        "diagnostic": {
            "path": str(diagnostic_path),
            "sha256": FORMAL_DIAGNOSTIC_SHA256,
            "artifact_type": FORMAL_DIAGNOSTIC_ARTIFACT_TYPE,
            "diagnostic_contract_sha256": FORMAL_DIAGNOSTIC_CONTRACT_SHA256,
        },
        "selection_contract": {
            "policy": SELECTION_POLICY,
            "fixed_alphas": list(FIXED_ALPHAS),
            "surface_scope": "every_seed",
            "surface_metrics": list(SURFACE_METRICS),
            "surface_noninferiority_tolerance": SURFACE_NONINFERIORITY_TOLERANCE,
            "surface_rule": "each_metric_gte_same_seed_alpha0_minus_tolerance",
            "fit_scope": "aggregate_seed_mean",
            "fit_support_rule": "strictly_greater_than_alpha0",
            "fit_smooth_l1_rule": "less_than_or_equal_to_alpha0",
            "selection_rule": "minimum_strictly_positive_feasible_alpha",
            "formal_unique_positive_alpha_required": EXPECTED_FORMAL_SELECTED_ALPHA,
            "dev_fields_read": False,
            "dev_values_copied": False,
            "audit_opened": False,
        },
        "feasible_alphas": decision["feasible_alphas"],
        "positive_feasible_alphas": decision["positive_feasible_alphas"],
        "selected_alpha": alpha,
        "per_alpha": decision["per_alpha"],
        "per_seed": per_seed_output,
        "aggregate_fit": aggregate_fit,
        "selected_interpolation": {
            "alpha": alpha,
            "formula": "theta_alpha=(1-alpha)*theta_control+alpha*theta_candidate",
            "pairing": "same_seed_only",
            "checkpoint_materialized": False,
        },
        "audit": {"opened": False, "status": "unopened", "artifact": None},
        "selector_implementation": {
            "path": str(selector_path),
            "sha256": sha256_file(selector_path),
        },
    }
    output = Path(args.output)
    _require(output.is_absolute(), "output must be an absolute path")
    write_frozen_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", required=True)
    parser.add_argument("--diagnostic-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = freeze_selection(args)
    print(
        {
            "output": str(Path(args.output).resolve()),
            "status": payload["status"],
            "selected_alpha": payload["selected_alpha"],
            "audit": payload["audit"],
        }
    )


if __name__ == "__main__":
    main()
