import json

import pytest
import torch
from torch.nn import functional as F

from radio_gs.query_field.contracts import Support
from radio_gs.query_field.field import CapabilityField
from radio_gs.query_field.functional_training import (
    FunctionalSource, NativeResidual, fit, preference_loss, projected_query_scores)
from radio_gs.query_field.relation_training import training_parameters
from radio_gs.query_field.storage import load_field, save_field, sha256


def small_field(n=108):
    generator = torch.Generator().manual_seed(9)
    geometry = dict(xyz=torch.randn(n, 3, generator=generator), normals=torch.zeros(n, 3),
                    quaternions=torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1),
                    scales=torch.ones(n, 3), opacities=torch.full((n, 1), .8))
    return CapabilityField(geometry, torch.randn(n, 3, generator=generator),
                           torch.randn(n, 3, generator=generator), torch.randn(n, 4, generator=generator),
                           torch.ones(n, dtype=torch.bool), world_units_per_metre=1.)


def test_group_policy_preserves_acceptance_under_actual_optimizer_update():
    field = small_field()
    parameters, protected = training_parameters(field, 'group_only')
    group = field.group.detach().clone()
    optimizer = torch.optim.Adam(parameters, lr=.1)
    logits = field.member_logits(Support(torch.tensor([0]), torch.ones(1)), .5)
    F.binary_cross_entropy_with_logits(logits[1:], torch.ones(107)).backward(); optimizer.step()
    assert not torch.equal(group, field.group)
    assert set(protected) == {'scale_cutoffs', 'log_temperature', 'affinity_bias'}
    for key, value in protected.items():
        assert torch.equal(value, dict(field.named_parameters())[key])
        assert dict(field.named_parameters())[key].grad is None
    joint, frozen = training_parameters(field, 'joint')
    assert len(joint) == 4 and not frozen
    with pytest.raises(ValueError, match='policy'):
        training_parameters(field, 'invalid')


def test_teacher_preference_is_stationary_and_ignores_absolute_offset():
    teacher = torch.tensor([.7, .4, .3, .5])
    student = teacher.clone().requires_grad_()
    loss = preference_loss(student, teacher); loss.backward()
    assert abs(float(loss)) < 1e-7 and student.grad.abs().max() < 1e-6
    torch.testing.assert_close(preference_loss(teacher + 10, teacher), loss, atol=1e-6, rtol=0)


def test_native_scores_normalize_before_observation_integration():
    codes = torch.tensor([[10., 0.], [0., 1.]])
    query = torch.tensor([1., 0.])
    operator = torch.tensor([[.5, .5]]).to_sparse()
    actual = projected_query_scores(codes, query, operator)
    torch.testing.assert_close(actual, torch.tensor([.5]))
    assert abs(float(actual) - float(F.normalize(operator @ codes, dim=-1) @ query)) > .4


def source_fixture(root, field):
    root.mkdir()
    masks = torch.zeros(2, 9, 12, dtype=torch.bool)
    masks[0, 1:8, 1:5] = True; masks[1, 1:8, 8:12] = True
    teacher = torch.tensor([[1., 0., 0.]]).repeat(108, 1)
    teacher[masks[1].flatten()] = torch.tensor([0., 1., 0.])
    native = teacher @ torch.tensor([[.8, .6, 0.], [.6, .8, 0.], [0., 0., 1.]])
    field.semantic.codes.copy_(native); field.local.codes.copy_(native)
    field.observed[0] = False
    entries = []
    for frame, split in [(1, 'train'), (2, 'train'), (3, 'development')]:
        path = root / f'view_{frame:05d}.pt'
        torch.save(dict(masks=masks, scales_m=torch.ones(2), raw_mass=torch.ones(108),
            best_element=torch.arange(108), operator=torch.eye(108).to_sparse(),
            semantic_pixels=teacher, local_pixels=teacher, semantic_descriptors=torch.eye(3)[:2]), path)
        entries.append(dict(frame=frame, split=split, compiled={'sha256': sha256(path)}))
    (root / 'observations.json').write_text(json.dumps(dict(frames=entries, world_units_per_metre=1.,
          splits=dict(train_frames=[1, 2], development_frames=[3]))))


def test_source_training_never_opens_development_and_rejects_tampering(tmp_path):
    field = small_field(); source_fixture(tmp_path / 'source', field)
    development = tmp_path / 'source/view_00003.pt'; development.unlink()
    source = FunctionalSource(tmp_path / 'source', field.observed)
    assert [v['frame'] for v in source.views] == [1, 2]
    assert len(source.examples) == 4
    with (tmp_path / 'source/view_00001.pt').open('ab') as f:
        f.write(b'tampered')
    with pytest.raises(ValueError, match='digest'):
        FunctionalSource(tmp_path / 'source', field.observed)


def test_functional_fit_improves_heldout_queries_and_bakes_standalone_field(tmp_path):
    field = small_field(); source = tmp_path / 'source'; source_fixture(source, field)
    save_field(field, tmp_path / 'parent', {'observations_sha256': sha256(source / 'observations.json')})
    report = fit(tmp_path / 'parent', source, tmp_path / 'fit', epochs=10, pairs=8,
                 learning_rate=.01, device='cpu')
    loaded, manifest = load_field(tmp_path / 'fit/field')
    assert manifest['external_scene_dependencies'] == []
    assert report['protected_state_unchanged'] and report['steps'] == 40
    for channel in ['semantic', 'local']:
        metric = report['source_development']['metrics'][channel]
        assert metric['refined_kl'] < metric['parent_kl']
        assert torch.equal(getattr(field, channel).codes[0], getattr(loaded, channel).codes[0])
        assert getattr(loaded, channel).weight is None
    for key, value in field.state_dict().items():
        if key not in {'semantic.codes', 'local.codes'}:
            assert torch.equal(value, loaded.state_dict()[key]), key
    # A registered member query is identical after native-only construction.
    support = Support(torch.tensor([14]), torch.ones(1))
    torch.testing.assert_close(field.member_logits(support, .5), loaded.member_logits(support, .5), rtol=0, atol=0)
