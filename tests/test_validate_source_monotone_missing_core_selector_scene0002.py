from radio_gs.scripts.validate_source_monotone_missing_core_selector_scene0002 import (
    EXTERNAL_SPLIT,
    REGISTERED_MEMBERSHIP_SPLIT,
    fixed_validation,
    source_access,
)


def test_scene0002_selector_validation_is_external_and_never_refits() -> None:
    assert EXTERNAL_SPLIT == "source_external_validation_selector_only"
    assert REGISTERED_MEMBERSHIP_SPLIT == "source_train"
    fixed = fixed_validation()
    assert fixed["threshold_or_model_refit_on_scene0002"] is False
    assert fixed["selector_gate"]["minimum_hard_precision_Wilson95_lower"] == 0.75
    assert fixed["positive_query_count"] == "bound_from_label_blind_subset_authority"
    access = source_access()
    assert access["scene0002_was_original_source_train_but_not_selector_fit_scene"]
    assert access["benchmark_labels_opened"] is False
