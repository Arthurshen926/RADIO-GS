from radio_gs.v3.evaluation.evaluate_text_anchor_instance_gate import _gate_decision


def test_text_anchor_gate_allows_equal_perfect_rank_with_positive_margin() -> None:
    identity = {"recall_at_1": 1.0, "mrr": 1.0, "margin": 0.2}
    instance = {"recall_at_1": 1.0, "mrr": 1.0, "margin": 0.1}
    passed, failures = _gate_decision(identity, instance, identity_exact=True)
    assert passed
    assert not failures


def test_text_anchor_gate_rejects_ranking_regression() -> None:
    identity = {"recall_at_1": 0.8, "mrr": 0.9, "margin": 0.2}
    instance = {"recall_at_1": 0.7, "mrr": 0.85, "margin": 0.1}
    passed, failures = _gate_decision(identity, instance, identity_exact=True)
    assert not passed
    assert len(failures) == 2
