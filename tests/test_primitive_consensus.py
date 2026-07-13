import torch

from radio_gs.training.primitive_consensus import robust_multiview_consensus


def test_robust_mpr_rejects_cross_view_outlier():
    observations = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.9, 0.1], [0.1, 0.9]],
            [[-1.0, 0.0], [0.0, -1.0]],
        ]
    )
    valid = torch.ones(3, 2, dtype=torch.bool)
    result = robust_multiview_consensus(
        observations, valid, robust_temperature=0.1, iterations=3
    )
    normalized = torch.nn.functional.normalize(result.targets, dim=-1)
    assert normalized[0, 0] > 0.95
    assert normalized[1, 1] > 0.95
    assert result.reliability.shape == (2, 3)
    assert torch.equal(result.observation_count, torch.tensor([3, 3]))


def test_mpr_marks_unseen_primitive_invalid():
    observations = torch.randn(2, 3, 4)
    valid = torch.tensor([[True, False, True], [True, False, False]])
    result = robust_multiview_consensus(observations, valid)
    assert result.valid.tolist() == [True, False, True]
    assert torch.equal(result.targets[1], torch.zeros(4))
