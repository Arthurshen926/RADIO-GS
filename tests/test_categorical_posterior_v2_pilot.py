import numpy as np
import torch

from radio_gs.scripts.train_categorical_posterior_v2_pilot import (
    _balanced_weights,
    encode_background_targets,
    posterior_prediction_to_raw,
    scene_macro,
)


def test_encode_background_targets_preserves_frozen_column_order():
    encoded = encode_background_targets(np.array([1, 2, 13, 36, 40]))

    assert encoded.tolist() == [0, 1, 19, 18, 19]


def test_posterior_prediction_maps_abstention_to_nyu40_zero():
    raw = posterior_prediction_to_raw(torch.tensor([0, 1, 18, -1]))

    assert raw.tolist() == [1, 2, 36, 0]


def test_scene_macro_is_unweighted_across_scenes():
    row = {
        "a": {split: {"miou": 0.2, "macc": 0.4} for split in ("19", "15", "10")},
        "b": {split: {"miou": 0.6, "macc": 0.8} for split in ("19", "15", "10")},
    }

    macro = scene_macro(row, ("a", "b"))

    assert macro["19"] == {"miou": 0.4, "macc": 0.6000000000000001}


def test_balanced_weights_allow_class_absent_from_development_scenes():
    weights = _balanced_weights(
        torch.tensor([0, 0, 2]),
        torch.tensor([1.0, 3.0, 2.0]),
    )

    assert torch.isfinite(weights).all()
    assert torch.isclose(weights.mean(), torch.tensor(1.0))
