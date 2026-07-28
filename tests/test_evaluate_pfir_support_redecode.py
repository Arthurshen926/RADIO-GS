import numpy as np

from radio_gs.scripts.evaluate_pfir_support_redecode import (
    interpolate_scores,
    prepare_mesh_interpolator,
)


def test_precomputed_mesh_interpolation_matches_inverse_distance_readout() -> None:
    source = np.array([[0.0, 0, 0], [1.0, 0, 0]], dtype=np.float32)
    target = np.array([[0.25, 0, 0], [0.75, 0, 0], [3.0, 0, 0]], dtype=np.float32)
    index, weight, valid = prepare_mesh_interpolator(
        source, target, neighbors=2, maximum_distance_m=1.0
    )
    mapped = interpolate_scores(
        np.array([0.0, 1.0], dtype=np.float32), index, weight, valid
    )

    np.testing.assert_allclose(mapped[:2], [0.25, 0.75], atol=1e-6)
    assert not valid[2]
    assert np.isneginf(mapped[2])
