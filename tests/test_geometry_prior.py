import pytest
import torch
from radio_gs.field.geometry_prior import depth_conditioned_weights


def test_identity_unknown_and_zero_strength():
    w = torch.tensor([0.2, 0.3, 0.4, 0.5])
    z = torch.tensor([1., 2., 3., 4.])
    d = torch.tensor([1., float('nan'), 0., 100.])
    c = torch.tensor([1., 1., 1., 0.])
    out, known = depth_conditioned_weights(w, z, d, c, absolute_tolerance=.08)
    assert torch.equal(out, w)
    assert known.tolist() == [True, False, False, False]
    out, _ = depth_conditioned_weights(w, z, z + 1, torch.ones(4), absolute_tolerance=.08, strength=0)
    assert torch.equal(out, w)


def test_cross_surface_writes_reduced_without_deleting_semantic_support():
    w = torch.ones(3)
    out, _ = depth_conditioned_weights(w, torch.tensor([1., 2., 3.]), torch.ones(3)*2,
                                      torch.ones(3), absolute_tolerance=.08)
    assert out[1] == 1
    assert torch.equal(out[[0, 2]], torch.tensor([.5, .5]))
    # A two-view feature write prefers the geometrically matching source;
    # this is not merely multiplying the final query response.
    feature = (out[:2, None] * torch.eye(2)).sum(0) / out[:2].sum()
    assert feature[1] > feature[0]


@pytest.mark.parametrize('strength', [-1, 1, float('nan')])
def test_invalid_policy(strength):
    with pytest.raises(ValueError):
        depth_conditioned_weights(torch.ones(1), torch.ones(1), torch.ones(1),
                                  torch.ones(1), absolute_tolerance=.08, strength=strength)


def test_invalid_confidence():
    with pytest.raises(ValueError):
        depth_conditioned_weights(torch.ones(1), torch.ones(1), torch.ones(1),
                                  torch.tensor([float('nan')]), absolute_tolerance=.08)


def test_depth_prior_changes_actual_multiview_teacher_fusion():
    from radio_gs.scripts.build_gaussian_multiview_teacher_cache import accumulate_contribution_mean_channel_chunked
    sums, counts = torch.zeros(1, 2), torch.zeros(1)
    for feature, depth in (([1., 0.], 1.), ([0., 1.], 2.)):
        weight, _ = depth_conditioned_weights(torch.ones(1), torch.tensor([2.]),
            torch.tensor([depth]), torch.ones(1), absolute_tolerance=.08)
        accumulate_contribution_mean_channel_chunked(
            torch.tensor(feature).reshape(2, 1, 1), torch.tensor([0]), torch.tensor([0]),
            weight, sums, counts, channel_chunk_size=1)
    assert torch.allclose(sums / counts[:, None], torch.tensor([[1/3, 2/3]]))
    assert counts[0] > 0


def test_prior_rejects_wrong_report_digest(tmp_path):
    from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
    path = tmp_path / 'report.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='digest'):
        MogeGeometryPrior(str(path), '0' * 64)


def test_prior_rejects_camera_mismatch_before_depth_access():
    from types import SimpleNamespace
    from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
    prior = object.__new__(MogeGeometryPrior)
    prior.records = {1: {}}
    prior.cameras = {1: SimpleNamespace(intrinsic=torch.eye(3), width=1, height=1,
                                       world_to_camera=torch.eye(4))}
    pose = torch.eye(4)
    pose[0, 3] = 1
    with pytest.raises(ValueError, match='coordinate frames'):
        prior.apply(frame_index=1, xyz=torch.zeros(1, 3), world_to_camera=pose,
                    intrinsic=torch.eye(3), gaussian_ids=torch.tensor([0]),
                    pixel_ids=torch.tensor([0]), weights=torch.ones(1), height=1, width=1)
