from radio_gs.v3.evaluation.aggregate_source_multimodal_gate import _image_gate


def test_image_gate_uses_directional_calibration_and_relaxed_iou():
    reports = [{
        "identity_bitwise_preserved": True,
        "uncalibrated_metrics": {
            "mask_iou": 0.4,
            "brier": 0.25,
            "boundary_f": 0.3,
            "unknown_fp_mass": 0.4,
        },
        "metrics": {
            "mask_iou": 0.391,
            "brier": 0.20,
            "boundary_f": 0.3,
            "unknown_fp_mass": 0.2,
        },
    } for _ in range(4)]
    passed, failures, _metrics = _image_gate(reports)
    assert passed
    assert not failures
