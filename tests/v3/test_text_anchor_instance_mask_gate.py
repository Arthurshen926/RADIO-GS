from radio_gs.v3.evaluation.evaluate_text_anchor_instance_mask_gate import (
    _gate_decision,
)


def test_text_mask_gate_requires_directional_improvement_over_both_controls() -> None:
    zero = {"mask_iou": 0.1, "brier": 0.3}
    random = {"mask_iou": 0.2, "brier": 0.25}
    candidate = {"mask_iou": 0.3, "brier": 0.2}
    passed, failures = _gate_decision(
        zero, random, candidate, identity_exact=True
    )
    assert passed
    assert not failures


def test_text_mask_gate_rejects_identity_change() -> None:
    baseline = {"mask_iou": 0.1, "brier": 0.3}
    candidate = {"mask_iou": 0.2, "brier": 0.2}
    passed, failures = _gate_decision(
        baseline, baseline, candidate, identity_exact=False
    )
    assert not passed
    assert failures == ["candidate changed clean D128 identity scores or anchors"]
