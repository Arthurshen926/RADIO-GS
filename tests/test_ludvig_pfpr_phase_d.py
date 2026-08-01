from pathlib import Path

import numpy as np
import pytest
import torch

from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_d import (
    LudvigPFPRPhaseDError,
    center_3x3_descriptor,
    gaussian_covariances,
)


def test_center_3x3_descriptor_uses_geometric_center() -> None:
    values = np.zeros((1, 9, 9, 2), dtype=np.float32)
    values[0, 3:6, 3:6, 0] = 2.0
    values[0, 0, 0, 1] = 100.0
    assert np.allclose(center_3x3_descriptor(values), [1.0, 0.0])


def test_center_descriptor_rejects_too_small_grid() -> None:
    with pytest.raises(LudvigPFPRPhaseDError, match="token grid"):
        center_3x3_descriptor(np.ones((1, 2, 2, 40), dtype=np.float32))


def test_identity_quaternion_covariance_is_squared_scale() -> None:
    scale = torch.tensor([[2.0, 3.0, 4.0]])
    rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    covariance = gaussian_covariances(scale, rotation)
    assert torch.allclose(covariance, torch.diag(torch.tensor([4.0, 9.0, 16.0]))[None])


def test_quaternion_covariance_is_symmetric() -> None:
    scale = torch.tensor([[0.2, 0.3, 0.4]])
    rotation = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    covariance = gaussian_covariances(scale, rotation)
    assert torch.allclose(covariance, covariance.transpose(1, 2), atol=1e-7)
    assert torch.all(torch.linalg.eigvalsh(covariance) > 0)
