import pytest
import torch

from radio_gs.scripts.build_lerf_anchor_conditioned_text_posterior import (
    _validate_source_gate,
    _validated_query_validity,
)


def _passed_report():
    return {
        "status": "source_image_gate_pass",
        "checkpoint_selection": {"all_scenes_noninferior": True},
        "image_audit": {"all_scenes_noninferior": True},
    }


def _extent():
    return {"metadata": {
        "decision_calibrated_identity": True,
        "fixed_decision_thresholds": True,
        "identity_scale": 0.05,
    }}


def test_rejects_historical_checkpoint_that_failed_validation_noninferiority():
    report = _passed_report()
    report["checkpoint_selection"]["all_scenes_noninferior"] = False
    with pytest.raises(ValueError, match="validation noninferiority"):
        _validate_source_gate(
            _extent(), report, {"status": "source_text_anchor_gate_pass"}
        )


def test_rejects_mixed_identity_gauge_and_threshold_contract():
    extent = _extent()
    extent["metadata"]["fixed_decision_thresholds"] = False
    with pytest.raises(ValueError, match="gauge and fixed decision"):
        _validate_source_gate(
            extent, _passed_report(), {"status": "source_text_anchor_gate_pass"}
        )


def test_query_validity_is_not_silently_replaced_by_all_rows():
    expected = torch.tensor([True, False, True])
    assert torch.equal(_validated_query_validity({"valid": expected}, 3), expected)
    with pytest.raises(ValueError, match="validity row domain"):
        _validated_query_validity({"valid": torch.ones(4, dtype=torch.bool)}, 3)
