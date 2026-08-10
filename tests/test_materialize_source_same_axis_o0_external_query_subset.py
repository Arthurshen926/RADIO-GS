from radio_gs.scripts.materialize_source_same_axis_o0_external_query_subset import (
    SCENE_ID,
    SPLIT,
)


def test_external_subset_identity_is_explicit() -> None:
    assert SCENE_ID == "scene0002_00"
    assert SPLIT == "source_external_validation_selector_only"
