from radio_gs.scripts.build_universal_field_v1_inventory import (
    LERF_SCENES,
    NVOS_SCENES,
    SCANNET_SCENES,
)


def test_universal_field_inventory_covers_existing_d512_l512_cohort_once():
    scenes = (*LERF_SCENES, *SCANNET_SCENES)

    assert len(scenes) == 12
    assert len(set(scenes)) == 12


def test_live_inventory_adds_namespaced_nvos_cohort_once():
    scene_keys = (
        *LERF_SCENES,
        *SCANNET_SCENES,
        *(f"nvos/{scene}" for scene in NVOS_SCENES),
    )

    assert len(NVOS_SCENES) == 8
    assert len(scene_keys) == 20
    assert len(set(scene_keys)) == 20
