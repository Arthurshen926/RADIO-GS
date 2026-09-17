import pytest
import torch

from radio_gs.v4.carrier import Camera, SurfaceVoxelCarrier
from radio_gs.v4.geometry import DepthObservation, SparseSurfaceFusion
from radio_gs.v4.contracts.surface_scene_bundle import SurfaceCarrierConfiguration
from radio_gs.v4.evaluation.lerf_text_cold_evaluator import compose_object_queries
from radio_gs.v4.object_memory.fragment_to_object import _pairwise_visibility_conflict, _compose_observed_membership
from radio_gs.v4.query.overlapping_candidates import compose_selected_candidates


def carrier(points, **kwargs):
    return SurfaceVoxelCarrier(points, .04, maximum_splat_radius=0,
        surface_band_voxels=1.5, maximum_contributors_per_pixel=32, **kwargs)


@pytest.mark.parametrize('shape', [(1, 8), (7, 9), (12, 17)])
@pytest.mark.parametrize('stride', [1, 2, 4, 7])
def test_every_pixel_returns_to_its_pixel_for_sliced_and_resized_cameras(shape, stride):
    h, w = shape
    k = torch.tensor([[17.3, 0, w*.41], [0, 21.7, h*.63], [0, 0, 1.]])
    k[:2] *= 1.7  # Nonintegral resize, then sample every stride pixel.
    k[0, 0] /= stride
    k[1, 1] /= stride
    k[:2, 2] = (k[:2, 2] - .5) / stride + .5
    c = Camera('roundtrip', k, torch.eye(4), h, w)
    z = torch.linspace(1, 3, h*w).reshape(h, w)
    o = DepthObservation(c, z, torch.ones_like(z, dtype=torch.bool), torch.ones_like(z))
    points, _, _, _ = SparseSurfaceFusion._backproject(o)
    p = carrier(points).project(c)
    assert torch.equal(p.pixel_ids, p.element_ids)
    assert p.element_ids.numel() == h*w


def test_nearest_surface_blocks_background_but_legacy_remains_replayable():
    c = Camera('c', torch.eye(3), torch.eye(4), 1, 1)
    points = torch.tensor([[0., 0., 1.], [0., 0., 1.01]])
    assert carrier(points).render_posterior(torch.tensor([0., 1.]), c).item() == 0
    legacy = carrier(points, projection_contract='legacy_round_band_v1')
    assert legacy.render_posterior(torch.tensor([0., 1.]), c).item() == pytest.approx(.4378235)


def test_unknown_confidence_cannot_occlude_trusted_surface():
    c = Camera('c', torch.eye(3), torch.eye(4), 1, 1)
    p = carrier(torch.tensor([[0., 0., 1.], [0., 0., 1.01]]), confidence=torch.tensor([0., 1.]))
    assert p.render_posterior(torch.tensor([0., 1.]), c).item() == 1


def test_serialized_projection_contracts_keep_old_and_new_distinct():
    old = dict(voxel_size=.04, maximum_splat_radius=0, surface_band_voxels=1.5,
        maximum_contributors_per_pixel=32, camera_convention='colmap_world_opencv_camera_feature_raster')
    assert SurfaceCarrierConfiguration.from_dict(old).to_dict() == old
    new = SurfaceCarrierConfiguration(**old)
    assert SurfaceCarrierConfiguration.from_dict(new.to_dict()) == new
    assert new != SurfaceCarrierConfiguration.from_dict(old)


def test_uncompressed_categorical_composition_retains_all_known_mass():
    membership = torch.full((1, 16), 1/16)
    result, audit = compose_object_queries(membership, torch.ones(1, 16))
    assert result.item() == 1
    assert audit['assignment_compression'] == 'none'
    with pytest.raises(ValueError, match='not categorical'):
        compose_object_queries(torch.ones(1, 16), torch.ones(1, 16))


def test_selected_candidate_union_is_duplicate_invariant():
    values = torch.tensor([[1., 0.], [0., .7], [.5, 0.]])
    choice = torch.tensor([[True, True], [False, False]])
    expected, _ = compose_selected_candidates(values, choice)
    actual, _ = compose_selected_candidates(values.repeat(1, 16), choice.repeat(1, 16))
    torch.testing.assert_close(expected, actual)
    assert torch.equal(expected[:, 1], torch.zeros(3))
    with pytest.raises(ValueError, match='boolean'):
        compose_selected_candidates(values, choice.float())


def test_same_object_parts_are_unknown_without_explicit_negative_evidence():
    positive = torch.eye(2)
    visibility = torch.ones(2, 2, dtype=torch.bool)
    args = (positive, torch.arange(2), visibility)
    assert not _pairwise_visibility_conflict(*args, evidence_threshold=.5).any()
    conflict = _pairwise_visibility_conflict(*args, evidence_threshold=.5, negative=1-positive)
    assert conflict[0, 1] == 1
    with pytest.raises(ValueError, match='contradict'):
        _pairwise_visibility_conflict(*args, evidence_threshold=.5, negative=positive)


def test_duplicate_same_view_masks_do_not_amplify_weak_support():
    for n in [1, 5, 20]:
        result = _compose_observed_membership(torch.full((n, 1), .05), torch.ones(n),
            torch.ones(n, 1), torch.zeros(n, dtype=torch.long))
        assert result.item() == pytest.approx(.05)


def test_fusion_counts_free_space_as_conflict_and_occluded_depth_as_unknown():
    def obs(z, key):
        k = torch.tensor([[1.,0,.5],[0,1.,.5],[0,0,1.]])
        return DepthObservation(Camera(key,k,torch.eye(4),1,1), torch.tensor([[z]]),
            torch.ones(1,1,dtype=torch.bool), torch.ones(1,1))
    result = SparseSurfaceFusion(.04).fuse([obs(1.,'a'),obs(1.,'b'),obs(2.,'c')])
    assert result.agreement_count.tolist() == [2]
    assert result.free_space_conflict_count.tolist() == [1]
    assert result.confidence.item() == pytest.approx(2/3)
    # A foreground observation occludes the retained farther surface; it must
    # not count as free-space evidence against that surface.
    result = SparseSurfaceFusion(.04).fuse([obs(2.,'a'),obs(2.,'b'),obs(1.,'c')])
    assert result.confidence.item() == 1
    assert result.free_space_conflict_count.tolist() == [0]


def test_compression_loss_is_distinct_from_unobserved_mass():
    from radio_gs.v4.object_memory.sparse_assignment import SparseObjectAssignments
    compressed = SparseObjectAssignments.from_dense(torch.tensor([[.4, .3, .2]]))
    assert compressed.precompression_unknown_weight.item() == pytest.approx(.1, abs=1e-6)
    assert compressed.compression_discarded_weight.item() == pytest.approx(.2, abs=1e-6)


def test_fragment_assignment_can_preserve_more_than_two_hypotheses():
    from radio_gs.v4.object_memory.fragment_to_object import _soft_assignments
    known, null = _soft_assignments(torch.ones(4, 4), [[0],[1],[2],[3]],
        torch.arange(4), torch.full((4,), -1), temperature=.1, null_affinity=.45,
        top_k=2, inherit_hierarchy_root=False, normalization='full_uncompressed',
        cannot_link=torch.zeros(4, 4, dtype=torch.bool))
    assert (known > 0).sum(-1).tolist() == [4]*4
    torch.testing.assert_close(known.sum(-1)+null, torch.ones(4))


def test_evaluation_exposes_sample_and_category_means_separately(monkeypatch):
    from radio_gs.v4.evaluation import lerf_text_cold_evaluator as evaluator
    monkeypatch.setattr(evaluator, '_camera', lambda *a: a[1])
    targets = {0: {'a':torch.ones(1,1,dtype=torch.bool),'b':torch.ones(1,1,dtype=torch.bool)},
               1: {'a':torch.zeros(1,1,dtype=torch.bool),'b':torch.ones(1,1,dtype=torch.bool)}}
    monkeypatch.setattr(evaluator, '_ground_truth_masks', lambda frame,*a: targets[frame])
    class Carrier:
        def render_posterior(self, values, camera):
            return torch.tensor([[[1., 0.]]])
    result = evaluator._evaluate_masks(carrier=Carrier(), element_probability=torch.tensor([[1.,0.]]),
        views={0:None,1:None},annotations={0:0,1:1},categories=['a','b'],
        source_height=1,source_width=1,height=1,width=1,pixel_threshold=.2)
    assert result['macro_category_miou'] == .5
    assert result['object_frame_mean_iou'] == pytest.approx(1/3)
    assert result['object_frame_count'] == 3
    assert not result['absent_target_rejection_validated']


def test_legacy_projection_preserves_zero_weight_samples_for_exact_replay():
    c = Camera('c', torch.eye(3), torch.eye(4), 1, 1)
    points = torch.tensor([[0.,0.,1.],[0.,0.,1.01]])
    legacy = carrier(points, confidence=torch.tensor([0.,1.]), projection_contract='legacy_round_band_v1')
    assert legacy.project(c).element_ids.tolist() == [0,1]
