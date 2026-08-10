from radio_gs.scripts.materialize_source_same_axis_o0_external_query_subset_scene0003 import (
    SCENE_ID,
    SPLIT,
    _access,
    _method,
)


def test_scene0003_subset_is_label_blind_and_fixed() -> None:
    assert SCENE_ID == "scene0003_00"
    assert SPLIT == "source_external_validation_multisource_selector_only"
    assert _method()["per_scene_hyperparameters"] is False
    assert _access()["scene0003_membership_payload_opened"] is False
