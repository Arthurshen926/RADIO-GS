from __future__ import annotations

from types import SimpleNamespace

import torch

from radio_gs.v4.training.diagnose_scannet_learned_mass_calibration import (
    UnaryMassCalibrator,
    _source_view_coverage_features,
)


def test_unary_mass_calibrator_starts_from_frozen_missing_mass():
    model = UnaryMassCalibrator(input_dimension=5, hidden_dimension=8)
    features = torch.randn(3, 5)
    observed = torch.tensor([4.0, 7.0, 2.0])
    frozen = torch.tensor([9.0, 6.0, 2.0])

    predicted = model(features, observed, frozen)

    # When the frozen posterior undershoots an exact observed fact, the model
    # still starts from the physical lower bound rather than reducing support.
    torch.testing.assert_close(predicted, torch.tensor([9.0, 7.0001, 2.0001]))
    assert bool((predicted >= observed).all())


def test_unary_mass_calibrator_is_positive_finite_and_trainable():
    model = UnaryMassCalibrator(input_dimension=4, hidden_dimension=8)
    features = torch.randn(6, 4)
    observed = torch.arange(1, 7, dtype=torch.float32)
    frozen = observed + torch.linspace(0.5, 3.0, 6)

    predicted = model(features, observed, frozen)
    loss = torch.log1p(predicted).sum()
    loss.backward()

    assert torch.isfinite(predicted).all()
    assert bool((predicted >= observed).all())
    assert model.network[-1].weight.grad is not None
    assert float(model.network[-1].weight.grad.abs().sum()) > 0


def test_source_view_coverage_uses_only_observed_support_and_source_receipt():
    class Carrier:
        def project(self, camera):
            ids = {
                "view_a": torch.tensor([0, 1, 2]),
                "view_b": torch.tensor([1, 2, 3, 3]),
            }[camera.key]
            return SimpleNamespace(element_ids=ids)

    cameras = [
        {
            "key": key,
            "intrinsic": torch.eye(3),
            "camera_to_world": torch.eye(4),
            "height": 2,
            "width": 2,
        }
        for key in ("view_a", "view_b")
    ]
    records = [
        {"frame_id": view, "object_id": object_id, "kept": kept}
        for view, values in (
            ("view_a", (True, False)),
            ("view_b", (False, True)),
        )
        for object_id, kept in zip((10, 20), values)
    ]
    runtime = {
        "partial": SimpleNamespace(
            positive=torch.tensor(
                [[True, False], [True, False], [False, True], [False, True]]
            )
        ),
        "payload": {
            "observation_cameras": cameras,
            "object_ids": [10, 20],
            "mask_dropout_receipt": {"records": records},
        },
        "carrier": Carrier(),
        # Complete labels deliberately disagree with observed support.  The
        # feature extractor must not inspect this target-only field.
        "labels": torch.tensor([1, 1, 0, 0]),
    }

    before = _source_view_coverage_features(runtime)
    runtime["labels"] = torch.tensor([-1, -1, -1, -1])
    after = _source_view_coverage_features(runtime)

    assert before.shape == (2, 21)
    assert torch.isfinite(before).all()
    torch.testing.assert_close(before, after)
