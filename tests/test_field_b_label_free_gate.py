from radio_gs.scripts.eval_field_b_label_free_gate import evaluate_field_b_gate


def _metrics() -> dict[str, float]:
    return {
        "mean_cosine": 0.96,
        "p05_cosine": 0.91,
        "dino_v3_target_mean_cosine": 0.93,
        "dino_v3_target_p05_cosine": 0.70,
        "sam3_target_mean_cosine": 0.93,
        "sam3_target_p05_cosine": 0.77,
        "relation_teacher_margin_hinge": 0.12,
    }


def test_field_b_gate_requires_relation_improvement_and_capability_preservation() -> None:
    initial = _metrics()
    final = dict(initial)
    final["mean_cosine"] -= 0.004
    final["dino_v3_target_p05_cosine"] -= 0.009
    final["relation_teacher_margin_hinge"] -= 0.01
    result = evaluate_field_b_gate(initial, final)
    assert result["passed"] is True

    failed = dict(final)
    failed["relation_teacher_margin_hinge"] = initial[
        "relation_teacher_margin_hinge"
    ]
    assert evaluate_field_b_gate(initial, failed)["passed"] is False
