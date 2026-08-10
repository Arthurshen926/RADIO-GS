from __future__ import annotations

from radio_gs.scripts.validate_source_monotone_missing_core_selector_scene0004 import (
    fixed_validation,
    source_access,
)


def test_heldout_validation_never_refits_and_has_fixed_gate() -> None:
    fixed = fixed_validation()
    assert fixed["threshold_or_model_refit_on_scene0004"] is False
    assert fixed["selector_probability"] == "minimum_probability_across_three_fold_models"
    assert fixed["selector_gate"]["minimum_hard_precision_Wilson95_lower"] == 0.75
    assert fixed["report_non_gate_scene0001_train_Wilson_bar_0p80"] is True
    access = source_access()
    assert access["source_validation_instance_labels_opened_exactly_once"] is True
    assert access["benchmark_labels_opened"] is False
    assert access["target_benchmark_opened"] is False
