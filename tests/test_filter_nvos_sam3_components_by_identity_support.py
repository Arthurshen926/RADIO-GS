import numpy as np

from radio_gs.scripts.filter_nvos_sam3_components_by_identity_support import (
    identity_supported_components,
    identity_supported_components_local_density,
)


def test_keeps_anchored_and_identity_supported_components_only():
    extent = np.zeros((7, 12), dtype=bool)
    extent[1:3, 1:3] = True
    extent[1:3, 5:7] = True
    extent[4:6, 9:11] = True
    coarse = np.zeros_like(extent)
    coarse[1:3, 1:3] = True
    coarse[1, 5] = True
    # The first component explains 4/5 coarse pixels, the second only 1/5 but
    # is explicitly point anchored; the disconnected third is rejected.
    result = identity_supported_components(
        extent,
        coarse,
        np.array([[5, 1]], dtype=np.float32),
        minimum_coarse_support_fraction=0.5,
    )
    assert bool(result[1:3, 1:3].all())
    assert bool(result[1:3, 5:7].all())
    assert not bool(result[4:6, 9:11].any())


def test_filter_never_adds_extent():
    extent = np.zeros((5, 5), dtype=bool)
    extent[1:3, 1:3] = True
    coarse = np.ones_like(extent)
    result = identity_supported_components(
        extent,
        coarse,
        np.array([[1, 1]], dtype=np.float32),
    )
    assert not bool((result & ~extent).any())


def test_local_density_preserves_small_locally_supported_component():
    extent = np.zeros((8, 12), dtype=bool)
    extent[1:5, 1:5] = True
    extent[1:3, 8:10] = True
    coarse = np.zeros_like(extent)
    coarse[1:5, 1:5] = True
    coarse[1, 8] = True  # 1/4 local density, but only 1/17 global support.

    global_result = identity_supported_components(
        extent, coarse, np.empty((0, 2), dtype=np.float32),
        minimum_coarse_support_fraction=0.1,
    )
    local_result = identity_supported_components_local_density(
        extent, coarse, np.empty((0, 2), dtype=np.float32),
        minimum_local_identity_density=0.1,
    )

    assert not bool(global_result[1:3, 8:10].any())
    assert bool(local_result[1:3, 8:10].all())
    assert not bool((local_result & ~extent).any())
