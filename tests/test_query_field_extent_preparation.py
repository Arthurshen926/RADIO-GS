import pytest
import torch
from unittest.mock import patch
from radio_gs.query_field.extent_preparation import ExtentPreparation


def fixture(count=4):
    # Row 0 mixes two members, row 1 supports member 2; member 3 is unknown.
    state = dict(index=torch.tensor([[0, 0, 1], [0, 1, 2]]), weight=torch.tensor([.25, .75, 1.]),
                 valid=torch.ones(2, dtype=torch.bool), footprint=torch.arange(1, count+1).float())
    return ExtentPreparation({f'views.{i}.'+k: v.clone() for i in range(2) for k, v in state.items()}, count)


def scopes():
    return [dict(target=torch.tensor([.8, .2]), positive=torch.tensor([True, False]),
                 negative=torch.tensor([False, True]), confidence=torch.tensor(.25)),
            dict(target=torch.tensor([.2, .9]), positive=torch.tensor([False, True]),
                 negative=torch.tensor([True, False]), confidence=torch.tensor(.75))]


def test_lift_confidence_and_initializer_normalized_cost_without_io():
    module = fixture(); prior = torch.tensor([.1, .1, .1, .6])
    with patch('builtins.open', side_effect=AssertionError('runtime file access')), patch('torch.load', side_effect=AssertionError('runtime load')):
        initial, cost, report = module(prior, scopes())
    expected = torch.tensor([.35, .35, .725, .6])
    torch.testing.assert_close(initial, expected, atol=1e-7, rtol=0)
    raw = torch.arange(1, 5).float()
    torch.testing.assert_close(cost, raw / torch.dot(raw, expected))
    assert abs(float(cost @ initial) - 1) < 1e-6 and report['active_views'] == [0, 1]


def test_inactive_complement_and_unknown_target_do_not_create_evidence():
    module = fixture(); prior = torch.tensor([.1, .3, .4, .6]); scope = scopes()
    for s in scope: s['positive'].zero_()
    initial, cost, report = module(prior, scope)
    assert torch.equal(initial, prior) and not cost.any() and report['fallback']
    scope = scopes(); scope[0]['negative'].zero_(); scope[1]['confidence'].zero_()
    first = module(prior, scope)[0]
    scope[0]['target'][1] = 1
    assert torch.equal(first, module(prior, scope)[0]) and first[2] == prior[2]
    scope[0]['negative'][0] = True
    with pytest.raises(ValueError): module(prior, scope)


def test_full_scene_visibility_precedes_member_domain_filter():
    from radio_gs.querying.typed_evidence import visible_operator
    # An unavailable foreground occluder owns 90% of a ray. Dropping it before
    # visibility would falsely admit the ray as reliable background evidence.
    hit = dict(pixel_ids=torch.tensor([0, 0, 1]), gaussian_ids=torch.tensor([0, 1, 1]),
               base_weights=torch.tensor([.9, .1, .8]))
    available = torch.tensor([False, True])
    op, _, valid = visible_operator(hit, available, 2, False)
    assert valid[:, 0].tolist() == [False, True]
    assert torch.equal(op.to_dense(), torch.tensor([[0., 1.], [0., 1.]]))


def test_preparation_roundtrip_and_optical_binding(tmp_path):
    from test_query_preserving_field import example
    from radio_gs.query_field.extent_factors import ExtentFactors
    from radio_gs.query_field.storage import save_field, load_field
    field = example(); previous = {k: v.clone() for k, v in field.state_dict().items()}
    single = dict(pointer=torch.tensor([0, 1, 2]), index=torch.tensor([0, 2]),
                  alpha=torch.tensor([.8, .8]), visibility=torch.ones(2), atom=torch.zeros(2, dtype=torch.long))
    field.extent_factors = ExtentFactors({f'views.{i}.'+k: v for i in range(2) for k, v in single.items()}, 12, field.opacities)
    field.extent_preparation = fixture(12)
    save_field(field, tmp_path/'field', {})
    restored, manifest = load_field(tmp_path/'field')
    for k, v in previous.items(): assert torch.equal(v, restored.state_dict()[k])
    initial, cost, _ = restored.extent_preparation(torch.full((12,), .2), scopes())
    assert torch.isfinite(cost).all() and initial.shape == (12,)
    assert manifest['external_scene_dependencies'] == []
