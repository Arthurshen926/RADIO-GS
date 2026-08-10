from __future__ import annotations

from radio_gs.scripts.fit_source_monotone_missing_core_selector import (
    fixed_fit,
    source_access,
)


def test_fit_is_low_capacity_monotone_and_target_blind() -> None:
    fit = fixed_fit()
    assert fit["model"] == "six_feature_monotone_additive_logistic"
    assert fit["positive_weight_parameterization"] == "softplus_nonnegative"
    assert fit["scene_or_query_identifiers_as_features"] is False
    assert fit["instance_labels_as_features"] is False
    assert fit["deep_network"] is False
    assert fit["target_probability"] == "minimum_probability_across_three_fold_models"
    access = source_access()
    assert access["source_validation_instance_labels_opened"] is False
    assert access["benchmark_labels_opened"] is False
    assert access["target_heldout_opened"] is False
