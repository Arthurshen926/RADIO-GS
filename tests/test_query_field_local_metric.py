import json

import pytest
import torch
from torch.nn import functional as F

from radio_gs.query_field import QueryPacket, Support
from radio_gs.query_field.field import NativeCoordinates
from radio_gs.query_field.storage import load_field, save_field
from test_query_preserving_field import example


def test_native_local_metric_separates_nearby_identical_group_codes():
    field = example()
    with torch.no_grad():
        field.group[:] = field.group[0].clone()
        field.xyz[:] = 0
        field.local.codes[1] = -field.local.codes[0]
    support = Support(torch.tensor([0]), torch.ones(1))
    original = field.member_logits(support, .5)
    assert original[1] > 0
    field.local_metric = 'native_cosine'
    changed = field.member_logits(support, .5)
    assert changed[0] > 0 and changed[1] < 0
    assert torch.isfinite(changed).all()


def test_affine_local_metric_matches_full_native_directions_and_row_selection():
    field = example()
    torch.manual_seed(32)
    weight, bias = torch.randn(7, 6), torch.randn(7)
    field.local = NativeCoordinates(field.local.codes, weight, bias)
    support = Support(torch.tensor([0, 2]), torch.tensor([.2, .8]))
    original = field.member_logits(support, .7)
    native = F.normalize(field.local.decode(torch.arange(12)), dim=-1)
    proto = F.normalize((native[support.indices] * support.weights[:, None]).sum(0), dim=0)
    expected = original - field.log_temperature.exp() * (1 - native @ proto).clamp_min(0)
    field.local_metric = 'native_cosine'
    torch.testing.assert_close(field.member_logits(support, .7), expected)
    ids = torch.tensor([1, 5, 9])
    torch.testing.assert_close(field.member_logits(support, .7, ids), expected[ids])
    frozen = {k: v.clone() for k, v in field.named_buffers()}
    optimizer = torch.optim.Adam(field.parameters(), lr=.01)
    F.binary_cross_entropy_with_logits(field.member_logits(support, .7), torch.ones(12)).backward()
    assert field.group.grad.abs().sum() > 0
    optimizer.step()
    for key, value in field.named_buffers():
        assert torch.equal(value, frozen[key])


def test_all_adapters_export_and_legacy_local_metric_default(tmp_path):
    field = example()
    field.local_metric = 'native_cosine'
    field.spatial_metric, field.world_units_per_metre = 'scale_normalized', 2.
    support = Support(torch.tensor([0]), torch.ones(1))
    negative = Support(torch.tensor([4]), torch.ones(1))
    packets = [QueryPacket('text', semantic=field.semantic.codes[0], task='union'),
        QueryPacket('image', semantic=field.semantic.codes[0], local=field.local.codes[:2]),
        QueryPacket('point_2d', positive=support, negative=negative),
        QueryPacket('point_3d', positive=support, negative=negative, exact_positive=True)]
    save_field(field, tmp_path/'scene', {})
    restored, _ = load_field(tmp_path/'scene')
    for packet in packets:
        assert torch.equal(field.query(packet).membership_scores, restored.query(packet).membership_scores)
    field.local_metric = 'none'
    save_field(field, tmp_path/'legacy', {})
    path = tmp_path/'legacy/manifest.json'
    manifest = json.loads(path.read_text())
    del manifest['spec']['local_metric']
    path.write_text(json.dumps(manifest))
    restored, _ = load_field(tmp_path/'legacy')
    for packet in packets:
        assert torch.equal(field.query(packet).membership_scores, restored.query(packet).membership_scores)
