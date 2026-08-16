from radio_gs.scripts.build_universal_field_v1_inventory import (
    LERF_SCENES,
    SCANNET_SCENES,
)


def test_universal_field_inventory_covers_existing_d512_l512_cohort_once():
    scenes = (*LERF_SCENES, *SCANNET_SCENES)

    assert len(scenes) == 12
    assert len(set(scenes)) == 12
