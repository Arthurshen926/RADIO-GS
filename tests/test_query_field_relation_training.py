import torch
import pytest

from radio_gs.query_field.relation_training import compatible_regions, weighted_partial_loss


def test_missing_joint_visibility_does_not_become_negative():
    masks = torch.tensor([[True, False], [True, True]])
    ids, weights = compatible_regions(masks, torch.ones(2), torch.empty(0, dtype=torch.long), torch.empty(0), 1.)
    assert not len(ids) and not len(weights)


def test_nested_masks_share_a_view_and_incompatible_scales_are_unknown():
    masks = torch.tensor([[True, False, False], [True, True, False], [False, True, True]])
    ids, mass = compatible_regions(masks, torch.tensor([1., 2., 4.]),
                                   torch.tensor([0, 1]), torch.tensor([.8, .2]), 1.)
    assert ids.tolist() == [0, 1]
    torch.testing.assert_close(mass, torch.tensor([.8, 1.]) / 1.8)
    assert mass.sum() == 1


def test_soft_partial_loss_has_no_unknown_gradient_and_retains_disagreement():
    prediction = torch.tensor([.2, .8, .4, .5], requires_grad=True)
    positive = torch.tensor([1., 0., 0., .5])
    negative = torch.tensor([0., 1., 0., .5])
    weighted_partial_loss(prediction, positive, negative).backward()
    assert prediction.grad[0] < 0 and prediction.grad[1] > 0
    assert prediction.grad[2] == 0 and abs(prediction.grad[3]) < 1e-6


def test_duplicate_evidence_mass_does_not_create_extra_view_confidence():
    p = torch.tensor([.3, .7, .8])
    pos = torch.tensor([.8, .2, 0.]); neg = torch.tensor([0., .3, .7])
    a = weighted_partial_loss(p, pos, neg)
    b = weighted_partial_loss(p, 3 * pos, 3 * neg)
    torch.testing.assert_close(a, b)
    with pytest.raises(ValueError, match='no known'):
        weighted_partial_loss(p, torch.zeros(3), torch.zeros(3))


def test_source_relations_exclude_development_and_replay_pair_conditions(tmp_path):
    import json
    from radio_gs.query_field.relation_training import SourceRelations
    from radio_gs.query_field.storage import sha256
    count = 108; pixels = torch.arange(count)
    masks = torch.zeros(2, 9, 12, dtype=torch.bool)
    masks[0, 1:8, 1:5] = True; masks[1, 1:8, 8:12] = True
    operator = torch.sparse_coo_tensor(torch.stack([pixels, pixels]), torch.ones(count), (count, count)).coalesce()
    entries = []
    for frame in [1, 2]:
        view = dict(masks=masks, scales_m=torch.ones(2), raw_mass=torch.ones(count), best_element=pixels,
                    operator=operator, intrinsic=torch.eye(3), world_to_camera=torch.eye(4), original_hw=[9, 12])
        path = tmp_path / f'view_{frame:05d}.pt'; torch.save(view, path)
        entries.append(dict(frame=frame, split='train', compiled={'sha256': sha256(path)}))
    entries.append(dict(frame=3, split='development', compiled={'sha256': 'must-not-be-read'}))
    (tmp_path / 'observations.json').write_text(json.dumps(dict(frames=entries,
          splits=dict(train_frames=[1, 2], development_frames=[3]))))
    relations = SourceRelations(tmp_path, torch.ones(count, dtype=torch.bool))
    assert len(relations.views) == 2 and len(relations.examples) == 4
    a = relations.pair(0, torch.Generator().manual_seed(3))
    b = relations.pair(0, torch.Generator().manual_seed(3))
    assert a['cross_view'] and a['primary_view'] != a['secondary_view']
    assert a['anchor'] == b['anchor'] and a['secondary_anchor'] == b['secondary_anchor']
    assert torch.equal(a['positive'], b['positive']) and torch.equal(a['negative'], b['negative'])
