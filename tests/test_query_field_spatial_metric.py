import json

import pytest
import torch

from radio_gs.query_field import CapabilityField, QueryPacket, Support
from radio_gs.query_field.storage import save_field, load_field
from test_query_preserving_field import example


def spatial(field, **kwargs):
    geometry = {k: getattr(field, k) for k in ['xyz', 'normals', 'quaternions', 'scales', 'opacities']}
    return CapabilityField(geometry, field.semantic.codes, field.local.codes,
                           field.group, field.observed, spatial_metric='scale_normalized', **kwargs)


def test_spatial_energy_suppresses_distant_identical_codes_without_changing_native():
    base = example()
    with torch.no_grad():
        base.group[:] = base.group[0].clone()
    field = spatial(base, world_units_per_metre=1.)
    support = Support(torch.tensor([0]), torch.ones(1))
    assert (base.member_logits(support, .5) > 0).all()
    logits = field.member_logits(support, .5)
    assert logits[0] > 0 and logits[-1] < 0
    assert torch.equal(field.semantic.score(base.semantic.codes[0]), base.semantic.score(base.semantic.codes[0]))
    assert torch.equal(logits[[1, 5]], field.member_logits(support, .5, rows=torch.tensor([1, 5])))
    field.member_logits(support, 1.)[1:3].sum().backward()
    assert field.group.grad is not None and torch.isfinite(field.group.grad).all()


def test_metric_is_rigid_and_world_unit_invariant_and_preserves_physical_scope():
    field = spatial(example(), world_units_per_metre=2.)
    support = Support(torch.tensor([0, 2]), torch.tensor([.25, .75]))
    reference = field.member_logits(support, .7).detach()
    # A change of reconstruction units and a rigid pose must not change members.
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    field.xyz = (field.xyz @ rotation.T) * 3 + torch.tensor([10., -5., 7.])
    field.world_units_per_metre *= 3
    torch.testing.assert_close(field.member_logits(support, .7), reference)
    # Keep the learned feature gate fixed to isolate the spatial effect of scale.
    with torch.no_grad():
        field.group[:] = field.group[0].clone()
    assert field.member_logits(support, 2.)[-1] > field.member_logits(support, .5)[-1]


def test_spatial_units_are_required_and_export_replays_signed_and_native_queries(tmp_path):
    base = example()
    for units in [None, 0., float('nan')]:
        with pytest.raises(ValueError, match='world units'):
            spatial(base, world_units_per_metre=units)
    field = spatial(base, world_units_per_metre=4.)
    positive = Support(torch.tensor([0, 1]), torch.tensor([.2, .8]))
    negative = Support(torch.tensor([10]), torch.ones(1))
    packets = [QueryPacket('text', semantic=field.semantic.codes[0], task='union'),
               QueryPacket('image', semantic=field.semantic.codes[0], local=field.local.codes[0]),
               QueryPacket('point_2d', positive=positive, negative=negative),
               QueryPacket('point_3d', positive=positive, negative=negative, exact_positive=True)]
    expected = [field.query(p) for p in packets]
    save_field(field, tmp_path/'scene', {})
    restored, manifest = load_field(tmp_path/'scene')
    assert manifest['spec']['world_units_per_metre'] == 4.
    for packet, reference in zip(packets, expected):
        assert torch.equal(restored.query(packet).membership_scores, reference.membership_scores)
    assert expected[-1].selected[[0, 1]].all() and not expected[-1].selected[10]


def test_legacy_export_without_geometry_units_keeps_original_member_behavior(tmp_path):
    field = example()
    save_field(field, tmp_path/'legacy', {})
    path = tmp_path/'legacy/manifest.json'
    manifest = json.loads(path.read_text())
    del manifest['spec']['spatial_metric'], manifest['spec']['world_units_per_metre']
    path.write_text(json.dumps(manifest))
    restored, _ = load_field(tmp_path/'legacy')
    packet = QueryPacket('text', semantic=field.semantic.codes[0], task='union')
    assert restored.spatial_metric == 'none' and restored.world_units_per_metre is None
    assert torch.equal(field.query(packet).membership_scores, restored.query(packet).membership_scores)
