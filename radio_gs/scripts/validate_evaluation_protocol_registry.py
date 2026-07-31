#!/usr/bin/env python3
"""Validate the fail-closed cross-benchmark evaluation protocol registry."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


COMPLETION_STATES = {"complete", "partial", "blocked", "not_attempted"}
EVIDENCE_CLASSES = {
    "exact_reproduction",
    "released_code_reproduction",
    "protocol_aligned",
    "compatibility_reproduction",
    "diagnostic",
    "published_context",
}
PAPER_COMPARISON_STATES = {"strict", "diagnostic_only", "forbidden"}
PROTOCOL_MATCH_DIMENSIONS = {
    "cohort",
    "prompt_or_query",
    "target_visibility",
    "metric_domain",
    "aggregation",
    "calibration",
    "implementation",
}


class RegistryError(ValueError):
    """Raised when a protocol registry could permit an invalid comparison."""


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RegistryError(f"{path} must be a mapping")
    return value


def _require_nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RegistryError(f"{path} must be a non-empty string")
    return value


def _validate_reported_metrics(row_id: str, row: Mapping[str, Any]) -> int:
    metrics = row.get("reported_metrics", [])
    if not isinstance(metrics, list):
        raise RegistryError(f"{row_id}.reported_metrics must be a list")
    names: set[str] = set()
    comparison_count = 0
    for index, raw_metric in enumerate(metrics):
        path = f"{row_id}.reported_metrics[{index}]"
        metric = _require_mapping(raw_metric, path)
        name = _require_nonempty_string(metric.get("name"), f"{path}.name")
        if name in names:
            raise RegistryError(f"{row_id} repeats reported metric {name}")
        names.add(name)
        role = metric.get("role")
        if role not in {"primary", "secondary", "diagnostic"}:
            raise RegistryError(f"{path}.role must be primary, secondary, or diagnostic")
        present = [key in metric for key in ("local", "paper", "delta_points")]
        if any(present) and not all(present):
            raise RegistryError(
                f"{path} must provide local, paper, and delta_points together"
            )
        if all(present):
            comparison_count += 1
            local = float(metric["local"])
            paper = float(metric["paper"])
            delta = float(metric["delta_points"])
            if not all(math.isfinite(value) for value in (local, paper, delta)):
                raise RegistryError(f"{path} contains a non-finite comparison")
            if not math.isclose(local - paper, delta, abs_tol=1e-8, rel_tol=1e-8):
                raise RegistryError(
                    f"{path}.delta_points={delta} does not equal local-paper={local - paper}"
                )
        if bool(metric.get("oracle_selected", False)):
            raise RegistryError(f"{path} cannot report an oracle-selected metric")
    return comparison_count


def _validate_oracle(row_id: str, row: Mapping[str, Any]) -> None:
    oracle = row.get("oracle_diagnostics")
    if oracle is None:
        return
    oracle_map = _require_mapping(oracle, f"{row_id}.oracle_diagnostics")
    if oracle_map.get("diagnostic_only") is not True:
        raise RegistryError(f"{row_id}.oracle_diagnostics must be diagnostic_only")
    if oracle_map.get("used_for_reported_metric") is not False:
        raise RegistryError(
            f"{row_id}.oracle_diagnostics.used_for_reported_metric must be false"
        )
    if oracle_map.get("used_for_model_or_threshold_selection") is not False:
        raise RegistryError(
            f"{row_id}.oracle_diagnostics cannot select a model or threshold"
        )


def validate_registry(payload: Mapping[str, Any]) -> None:
    """Validate registry structure and strict-comparison eligibility."""

    if payload.get("schema_version") != 1:
        raise RegistryError("schema_version must equal 1")
    policy = _require_mapping(payload.get("reporting_policy"), "reporting_policy")
    if policy.get("oracle_metrics_are_diagnostic_only") is not True:
        raise RegistryError("reporting_policy must ban oracle metrics from reported rows")
    if policy.get("incomplete_cohorts_are_strictly_comparable") is not False:
        raise RegistryError("reporting_policy must reject incomplete strict comparisons")

    rows = _require_mapping(payload.get("evaluations"), "evaluations")
    if not rows:
        raise RegistryError("evaluations must not be empty")
    for raw_row_id, raw_row in rows.items():
        row_id = _require_nonempty_string(raw_row_id, "evaluation id")
        row = _require_mapping(raw_row, row_id)
        _require_nonempty_string(row.get("benchmark_family"), f"{row_id}.benchmark_family")
        _require_nonempty_string(row.get("task"), f"{row_id}.task")
        _require_nonempty_string(row.get("method"), f"{row_id}.method")

        completion = row.get("completion")
        if completion not in COMPLETION_STATES:
            raise RegistryError(f"{row_id}.completion has unknown value {completion!r}")
        evidence_class = row.get("evidence_class")
        if evidence_class not in EVIDENCE_CLASSES:
            raise RegistryError(
                f"{row_id}.evidence_class has unknown value {evidence_class!r}"
            )

        cohort = _require_mapping(row.get("cohort"), f"{row_id}.cohort")
        if not isinstance(cohort.get("complete"), bool):
            raise RegistryError(f"{row_id}.cohort.complete must be boolean")
        protocol = _require_mapping(row.get("protocol"), f"{row_id}.protocol")
        _require_nonempty_string(protocol.get("aggregation"), f"{row_id}.protocol.aggregation")
        _require_nonempty_string(protocol.get("metric_domain"), f"{row_id}.protocol.metric_domain")
        _require_nonempty_string(protocol.get("calibration"), f"{row_id}.protocol.calibration")

        comparison = _require_mapping(row.get("comparability"), f"{row_id}.comparability")
        paper_comparison = comparison.get("paper_comparison")
        if paper_comparison not in PAPER_COMPARISON_STATES:
            raise RegistryError(
                f"{row_id}.comparability.paper_comparison has unknown value "
                f"{paper_comparison!r}"
            )
        if not isinstance(comparison.get("strict_table_eligible"), bool):
            raise RegistryError(
                f"{row_id}.comparability.strict_table_eligible must be boolean"
            )
        reasons = comparison.get("reasons", [])
        if not isinstance(reasons, list) or not all(
            isinstance(reason, str) and reason.strip() for reason in reasons
        ):
            raise RegistryError(f"{row_id}.comparability.reasons must be strings")
        if paper_comparison != "strict" and not reasons:
            raise RegistryError(
                f"{row_id}: non-strict paper comparison requires explicit reasons"
            )

        match = _require_mapping(
            comparison.get("protocol_match_to_paper"),
            f"{row_id}.comparability.protocol_match_to_paper",
        )
        if set(match) != PROTOCOL_MATCH_DIMENSIONS:
            missing = sorted(PROTOCOL_MATCH_DIMENSIONS - set(match))
            extra = sorted(set(match) - PROTOCOL_MATCH_DIMENSIONS)
            raise RegistryError(
                f"{row_id}.protocol_match_to_paper dimensions mismatch; "
                f"missing={missing}, extra={extra}"
            )
        if not all(isinstance(value, bool) for value in match.values()):
            raise RegistryError(f"{row_id}.protocol_match_to_paper values must be boolean")

        comparison_metric_count = _validate_reported_metrics(row_id, row)
        if paper_comparison == "strict":
            if completion != "complete":
                raise RegistryError(f"{row_id}: strict comparison requires completion=complete")
            if not bool(cohort["complete"]):
                raise RegistryError(f"{row_id}: strict comparison requires a complete cohort")
            if not all(bool(value) for value in match.values()):
                failed = sorted(key for key, value in match.items() if not value)
                raise RegistryError(
                    f"{row_id}: strict comparison has protocol mismatches {failed}"
                )
            if comparison["strict_table_eligible"] is not True:
                raise RegistryError(
                    f"{row_id}: strict comparison must be strict_table_eligible"
                )
            if evidence_class != "exact_reproduction":
                raise RegistryError(
                    f"{row_id}: strict comparison requires exact_reproduction evidence"
                )
            if comparison_metric_count == 0:
                raise RegistryError(
                    f"{row_id}: strict comparison requires local/paper/delta metrics"
                )
        elif comparison["strict_table_eligible"] is not False:
            raise RegistryError(
                f"{row_id}: non-strict comparisons cannot be strict_table_eligible"
            )
        if paper_comparison == "diagnostic_only" and comparison_metric_count == 0:
            raise RegistryError(
                f"{row_id}: diagnostic paper comparison requires a local/paper/delta metric"
            )
        if paper_comparison == "forbidden" and comparison_metric_count:
            raise RegistryError(
                f"{row_id}: forbidden paper comparison cannot contain paper deltas"
            )

        if not bool(cohort["complete"]) and paper_comparison == "strict":
            raise RegistryError(f"{row_id}: incomplete cohort cannot be strict")
        if completion in {"blocked", "not_attempted"} and row.get("reported_metrics"):
            raise RegistryError(
                f"{row_id}: blocked/not-attempted rows cannot contain local reported metrics"
            )
        _validate_oracle(row_id, row)


def load_and_validate(path: Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    mapping = _require_mapping(payload, str(path))
    validate_registry(mapping)
    return mapping


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "registry",
        type=Path,
        nargs="?",
        default=Path("paper/artifacts/evaluation_protocol_registry_20260731.yaml"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = load_and_validate(args.registry)
    print(f"validated {len(payload['evaluations'])} evaluation rows: {args.registry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
