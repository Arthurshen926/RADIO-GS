from radio_gs.scripts.eval_field_c_training_gate import evaluate_gate


def _metrics(*, raw_mean: float, raw_p05: float, dino: float, sam: float):
    return {
        "mean_cosine": raw_mean,
        "p05_cosine": raw_p05,
        "dino_v3_target_mean_cosine": dino,
        "sam3_target_mean_cosine": sam,
    }


def _thresholds():
    return {
        "raw_mean_cosine_delta_minimum": -0.005,
        "raw_p05_cosine_delta_minimum": -0.01,
        "dino_v3_mean_cosine_delta_minimum": 0.0,
        "sam3_mean_cosine_delta_minimum": 0.0,
    }


def test_field_c_training_gate_accepts_registered_boundary_values():
    initial = _metrics(raw_mean=0.8, raw_p05=0.6, dino=0.7, sam=0.65)
    final = _metrics(raw_mean=0.795, raw_p05=0.59, dino=0.7, sam=0.65)

    result = evaluate_gate(initial, final, _thresholds())

    assert result["passed"] is True
    assert all(result["checks"].values())


def test_field_c_training_gate_rejects_capability_regression():
    initial = _metrics(raw_mean=0.8, raw_p05=0.6, dino=0.7, sam=0.65)
    final = _metrics(raw_mean=0.81, raw_p05=0.61, dino=0.699, sam=0.66)

    result = evaluate_gate(initial, final, _thresholds())

    assert result["passed"] is False
    assert result["checks"]["dino_v3_mean_cosine"] is False
