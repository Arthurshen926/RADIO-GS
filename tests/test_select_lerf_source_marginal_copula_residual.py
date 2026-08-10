from __future__ import annotations

from radio_gs.scripts.select_lerf_source_marginal_copula_residual import (
    _candidate_is_eligible,
    _select_candidate,
    _topology,
)


def _scene(
    *, boundary: float = 0.7, component: float = 2.0, exact: bool = True
) -> dict:
    return {
        "control_support_diagnostics": {
            "top_decile_boundary_f_mean": 0.6,
            "top_decile_component_abs_error_mean": 3.0,
        },
        "diagnostics": {
            "p": {
                "top_decile_boundary_f_mean": boundary,
                "top_decile_component_abs_error_mean": component,
                "marginal_exact_every_frame": exact,
                "selected_count_error_sum": 0 if exact else 1,
                "support_units": 10,
            }
        },
    }


def _response_gate(score: float, *, passed: bool = True) -> dict:
    return {
        "decision": {"candidate_eligible_for_next_source_gate": passed},
        "pooled": {"candidate": {"ranking_spearman_mean": score}},
    }


def test_any_scene_boundary_or_component_regression_rejects() -> None:
    good = _topology(_scene(), "p")
    bad_boundary = _topology(_scene(boundary=0.59), "p")
    bad_component = _topology(_scene(component=3.01), "p")
    assert _candidate_is_eligible(_response_gate(0.5), {"ramen": good, "teatime": good})
    assert not _candidate_is_eligible(
        _response_gate(0.5), {"ramen": good, "teatime": bad_boundary}
    )
    assert not _candidate_is_eligible(
        _response_gate(0.5), {"ramen": bad_component, "teatime": good}
    )


def test_marginal_or_selected_count_inexact_rejects() -> None:
    good = _topology(_scene(), "p")
    inexact = _topology(_scene(exact=False), "p")
    assert not _candidate_is_eligible(
        _response_gate(0.5), {"ramen": good, "teatime": inexact}
    )


def test_selects_highest_pooled_spearman_only_among_eligible() -> None:
    rows = [
        {
            "policy": {"policy_id": "safe_low"},
            "eligible": True,
            "response_gate": _response_gate(0.6),
        },
        {
            "policy": {"policy_id": "unsafe_high"},
            "eligible": False,
            "response_gate": _response_gate(0.9),
        },
        {
            "policy": {"policy_id": "safe_high"},
            "eligible": True,
            "response_gate": _response_gate(0.8),
        },
    ]
    selected = _select_candidate(rows)
    assert selected is not None
    assert selected["policy"]["policy_id"] == "safe_high"
    # Target authorization is intentionally not a selector output; only the
    # caller's reserved-audit handoff may be enabled.
    assert "eligible_for_target_metric" not in selected
