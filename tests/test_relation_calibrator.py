import torch

from radio_gs.interfaces.relation_calibrator import MonotonicRelationCalibrator
from radio_gs.scripts.build_scannet_relation_edge_cache import accumulate_relation_votes
from radio_gs.scripts.train_relation_calibrator import binary_metrics
from radio_gs.scripts.train_scene_relation_private_code import SceneRelationPrivateCode


def test_relation_votes_require_coobservation_and_mask_coverage() -> None:
    edge = torch.tensor([[0, 1, 0], [1, 2, 2]])
    membership = torch.tensor([[1, 1, 0], [0, 0, 1]], dtype=torch.bool)
    observed = torch.ones(3, dtype=torch.bool)
    same, cannot, seen = accumulate_relation_votes([membership], [observed], edge)
    assert same.tolist() == [1, 0, 0]
    assert cannot.tolist() == [0, 1, 1]
    assert seen.tolist() == [1, 1, 1]


def test_relation_calibrator_is_monotonic_in_declared_directions() -> None:
    model = MonotonicRelationCalibrator()
    base = torch.tensor([[-2.0, -2.0, -2.0, 1.0, 1.0]])
    better_affinity = base.clone(); better_affinity[0, 1] = -1.0
    worse_distance = base.clone(); worse_distance[0, 3] = 2.0
    assert model(better_affinity) > model(base)
    assert model(worse_distance) < model(base)


def test_binary_relation_auc_is_one_for_perfect_ranking() -> None:
    metrics = binary_metrics(torch.tensor([-2.0, -1.0, 1.0, 2.0]),
                             torch.tensor([0.0, 0.0, 1.0, 1.0]))
    assert metrics["auc"] == 1.0


def test_relation_private_code_is_symmetric_and_relation_only() -> None:
    model = SceneRelationPrivateCode(num_nodes=3, dimension=8)
    forward = torch.tensor([[0, 1], [1, 2]])
    reverse = forward.flip(0)
    assert model.code.weight.shape == (3, 8)
    assert torch.allclose(model.residual(forward), model.residual(reverse))
    assert model.residual(forward).shape == (2,)
