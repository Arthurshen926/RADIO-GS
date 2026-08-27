from radio_gs.v3.evaluation.gates import (
    CapabilityMetric,
    capability_pareto_gate,
    source_heldout_gate,
)
from radio_gs.v3.evaluation.source_heldout import SourceHeldoutMetrics


def test_gate_requires_every_scene_and_all_proper_metrics():
    baseline = {"easy": SourceHeldoutMetrics(0.4, 0.2, 0.5, 0.1), "hard": SourceHeldoutMetrics(0.3, 0.3, 0.4, 0.2)}
    candidate = {"easy": SourceHeldoutMetrics(0.46, 0.18, 0.55, 0.09), "hard": SourceHeldoutMetrics(0.36, 0.25, 0.45, 0.19)}
    assert source_heldout_gate(baseline, candidate).passed
    candidate["hard"] = SourceHeldoutMetrics(0.29, 0.25, 0.45, 0.19)
    decision = source_heldout_gate(baseline, candidate)
    assert not decision.passed
    assert any("hard" in value for value in decision.failures)


def test_capability_gate_uses_direction_and_preregistered_tolerance():
    baseline = {
        "text_locacc": CapabilityMetric(0.7, True, 0.01),
        "render_error": CapabilityMetric(0.2, False, 0.005),
    }
    assert capability_pareto_gate(
        baseline, {"text_locacc": 0.691, "render_error": 0.204}
    ).passed
    decision = capability_pareto_gate(
        baseline, {"text_locacc": 0.68, "render_error": 0.204}
    )
    assert not decision.passed
    assert "text_locacc" in decision.failures[0]
