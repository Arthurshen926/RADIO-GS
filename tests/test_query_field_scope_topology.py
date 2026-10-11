from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from torch.nn import functional as F
from radio_gs.query_field.scope_topology import ScopeTopologyView, ScopeTopology, compile_scope_topology_view, assemble_scope
from radio_gs.query_field.extent_preparation import ExtentPreparationView


def make(masks, pixels=None):
    if pixels is None: pixels = torch.arange(masks.shape[1] * masks.shape[2])
    return ScopeTopologyView(compile_scope_topology_view(masks, torch.ones(len(masks), 3),
                             torch.ones(len(masks), dtype=torch.bool), pixels))


def test_union_before_erosion_and_border_exact():
    masks = torch.zeros(2, 5, 6, dtype=torch.bool)
    masks[0, :, :3] = True; masks[1, :, 3:] = True
    v = make(masks)
    with patch('builtins.open', side_effect=AssertionError('IO')), patch('torch.load', side_effect=AssertionError('IO')):
        y, p, n = v.compose(torch.ones(2, dtype=torch.bool))
    assert p.all() and y.all() and not n.any()
    individual = torch.stack([v.compose(torch.arange(2) == i)[1] for i in range(2)]).any(0)
    assert int((p & ~individual).sum()) == 10


def test_random_compositions_equal_raster_oracle():
    g = torch.Generator().manual_seed(391)
    masks = torch.rand(7, 9, 12, generator=g) > .6
    pixels = torch.arange(108).reshape(9, 12)[1::2, ::2].flatten()
    v = make(masks, pixels)
    for _ in range(20):
        a = torch.rand(7, generator=g) > .6
        u = (torch.rand(7, generator=g) > .6) & ~a
        y, p, n = v.compose(a, unknown=u)
        binary = masks[a].any(0).float()[None, None]
        expected_p = -F.max_pool2d(-binary, 3, 1, 1) > .5
        expected_n = (F.max_pool2d(binary, 3, 1, 1) < .5) & ~masks[u].any(0)
        if not a.any(): expected_n.zero_()
        assert torch.equal(y, binary.flatten()[pixels])
        assert torch.equal(p, expected_p.flatten()[pixels])
        assert torch.equal(n, expected_n.flatten()[pixels])


def test_no_admission_no_exclusion_and_unknown_is_not_negative():
    masks = torch.zeros(2, 5, 8, dtype=torch.bool)
    masks[0, :, :2] = True; masks[1, :, 5:] = True
    v = make(masks)
    assert not v.compose(torch.zeros(2, dtype=torch.bool))[2].any()
    y, p, n = v.compose(torch.tensor([True, False]), unknown=torch.tensor([False, True]))
    assert not n.reshape(5, 8)[:, 5:].any()
    assert p.any() and y.any()
    with pytest.raises(ValueError): v.compose(torch.ones(2, dtype=torch.bool), unknown=torch.ones(2, dtype=torch.bool))


def test_visibility_and_native_permission_are_separate_from_members():
    masks = torch.zeros(1, 3, 5, dtype=torch.bool); masks[0, :, :2] = True
    v = make(masks); count = 15
    valid = torch.ones(count, dtype=torch.bool); valid[0] = False
    p = ExtentPreparationView(dict(index=torch.arange(count).repeat(2, 1), weight=torch.ones(count),
                                  valid=valid, footprint=torch.ones(count)), count)
    prior = torch.zeros(count); prior[4] = 1
    s = assemble_scope(v, p, prior, torch.ones(1, dtype=torch.bool), torch.tensor(.7))
    assert not s['positive'][0] and not s['negative'][4] and s['negative'][9]
    assert s['target'][0] == 1  # geometric target survives visibility restriction


def test_state_rejects_wrong_ray_binding_and_unknown_payload():
    v = make(torch.ones(1, 2, 3, dtype=torch.bool))
    state = {'views.0.' + k: x for k, x in v.state_dict().items()}
    with pytest.raises(ValueError): ScopeTopology(state, SimpleNamespace(views=[SimpleNamespace(valid=torch.ones(5))]))
    with pytest.raises(ValueError): ScopeTopologyView(dict(v.state_dict(), masks=torch.ones(3)))


def test_single_field_cold_roundtrip(tmp_path):
    from test_query_preserving_field import example
    from test_query_field_extent_preparation import fixture
    from radio_gs.query_field.extent_factors import ExtentFactors
    from radio_gs.query_field.storage import save_field, load_field
    field = example()
    optical = dict(pointer=torch.tensor([0,1,2]), index=torch.tensor([0,2]), alpha=torch.tensor([.8,.8]),
                   visibility=torch.ones(2), atom=torch.zeros(2,dtype=torch.long))
    field.extent_factors = ExtentFactors({f'views.{i}.'+k:v for i in range(2) for k,v in optical.items()},12,field.opacities)
    field.extent_preparation = fixture(12)
    v = make(torch.tensor([[[True,False]],[[False,True]]]))
    field.scope_topology = ScopeTopology({f'views.{i}.'+k:x for i in range(2) for k,x in v.state_dict().items()},field.extent_preparation)
    save_field(field,tmp_path/'field',{})
    cold,_ = load_field(tmp_path/'field')
    for k,x in field.state_dict().items(): assert torch.equal(x,cold.state_dict()[k])
    assert cold.scope_topology.views[0].compose(torch.ones(2,dtype=torch.bool))[1].all()
