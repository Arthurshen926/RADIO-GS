from __future__ import annotations

import torch

from radio_gs.v4.training.diagnose_scannet_ridge_mass_spatial_slots import (
    fit_scene_balanced_ridge,
    predict_log_correction,
)


def test_scene_balanced_ridge_recovers_finite_linear_correction():
    records = [
        {
            "features": torch.tensor([[0.0], [1.0], [2.0]]),
            "target_log_correction": torch.tensor([1.0, 3.0, 5.0]),
        },
        {
            "features": torch.tensor([[3.0], [4.0]]),
            "target_log_correction": torch.tensor([7.0, 9.0]),
        },
    ]
    all_features = torch.cat([record["features"] for record in records])
    mean = all_features.mean(0)
    scale = all_features.std(0, unbiased=False)
    coefficient = fit_scene_balanced_ridge(
        records, feature_mean=mean, feature_scale=scale, ridge=1e-6
    )
    prediction = predict_log_correction(
        all_features, coefficient, feature_mean=mean, feature_scale=scale
    )

    assert torch.isfinite(coefficient).all()
    torch.testing.assert_close(
        prediction, torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0]), atol=2e-4, rtol=0
    )
