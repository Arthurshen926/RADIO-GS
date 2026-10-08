import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csc_matrix
import torch

from radio_gs.query_field.conditional_relation import enable, residual
from radio_gs.query_field.contracts import Support, QueryPacket
from radio_gs.query_field.element_relation_training import ElementRelations, view_transfer, element_loss, parent_logits, fit
from radio_gs.query_field.storage import save_field, load_field, sha256
from radio_gs.query_field.query_program import query, category_partition
from test_query_preserving_field import example


def metric_field():
    field = example(); field.spatial_metric = 'scale_normalized'; field.world_units_per_metre = 2.
    return field


def test_zero_initialization_preserves_old_formula_and_native_scores():
    field = metric_field(); support = Support(torch.tensor([1, 3]), torch.tensor([.2, .8]))
    before = field.member_logits(support, 1.).detach()
    native = field.semantic.score(field.semantic.codes[2]).clone()
    enable(field, torch.linspace(0, 1, 12))
    assert torch.equal(before, field.member_logits(support, 1.))
    assert torch.equal(native, field.semantic.score(field.semantic.codes[2]))


@pytest.mark.parametrize('rows', [None, [4, 2, 7, 4]])
def test_native_local_metric_keeps_group_and_local_relation_features_distinct(rows):
    field = metric_field(); field.local_metric = 'native_cosine'
    support = Support(torch.tensor([1, 3]), torch.tensor([.2, .8]))
    rows = None if rows is None else torch.tensor(rows)
    base = field.member_logits(support, .7, rows).detach()
    enable(field, torch.ones(12))
    with torch.no_grad():
        field.relation.layers[-1].weight.fill_(.2)
    captured = []
    hook = field.relation.register_forward_pre_hook(lambda _, args: captured.append(args[0].detach().clone()))
    try:
        actual = field.member_logits(support, .7, rows)
    finally:
        hook.remove()
    features = torch.cat(captured)
    group = field.grouping_features(.7)
    prototype = torch.nn.functional.normalize((group[support.indices] * support.weights[:, None]).sum(0), dim=0)
    group_cosine = (group if rows is None else group[rows]) @ prototype
    assert not torch.allclose(group_cosine, features[:, 2])
    torch.testing.assert_close(features[:, 0], group_cosine)
    torch.testing.assert_close(actual, base + residual(field, support, .7, group_cosine, rows))


def test_parent_source_reference_matches_deployed_logits():
    field = metric_field(); rows = torch.tensor([1, 7, 2, 1])
    expected = field.member_logits(Support(torch.tensor([2]), torch.ones(1)), .7, rows)
    torch.testing.assert_close(parent_logits(field, field.group, 2, .7, rows), expected, rtol=1e-6, atol=1e-6)


def test_relation_chunking_weighted_support_and_geometry_frame_invariance():
    field = metric_field(); enable(field, torch.linspace(0, 1, 12))
    with torch.no_grad():
        field.relation.layers[-1].weight.fill_(.3)
    support = Support(torch.tensor([1, 3]), torch.tensor([.25, .75]))
    features = field.grouping_features(.6)
    prototype = torch.nn.functional.normalize((features[support.indices] * support.weights[:, None]).sum(0), dim=0)
    cosine = features @ prototype
    a = residual(field, support, .6, cosine, chunk_size=3)
    b = residual(field, Support(torch.tensor([1, 3, 3]), torch.tensor([.25, .25, .5])), .6, cosine)
    torch.testing.assert_close(a, b)
    with torch.no_grad():
        field.xyz.mul_(7).add_(torch.tensor([17., -3., 4.])); field.scales.mul_(7)
        field.world_units_per_metre *= 7
    torch.testing.assert_close(a, residual(field, support, .6, cosine), rtol=1e-5, atol=1e-6)


def test_relation_is_trainable_and_protects_native_geometry(tmp_path):
    field = metric_field(); enable(field, torch.ones(12))
    before = {k: v.clone() for k, v in field.named_buffers()}
    optimizer = torch.optim.Adam(field.relation.parameters(), lr=.01)
    support = Support(torch.tensor([0]), torch.ones(1))
    initial = field.member_logits(support, 4.).detach()
    for _ in range(5):
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(field.member_logits(support, 4.), torch.ones(12))
        loss.backward(); optimizer.step()
    assert not torch.equal(initial, field.member_logits(support, 4.))
    for k, value in field.named_buffers(): assert torch.equal(before[k], value)
    save_field(field, tmp_path / 'field', {})
    loaded, manifest = load_field(tmp_path / 'field')
    assert manifest['external_scene_dependencies'] == []
    for packet in [QueryPacket('text', semantic=field.semantic.codes[2], task='union'),
                   QueryPacket('image', semantic=field.semantic.codes[2], local=field.local.codes[2:4]),
                   QueryPacket('point_2d', positive=support),
                   QueryPacket('point_3d', positive=support, exact_positive=True)]:
        a, b = query(field, packet), query(loaded, packet)
        assert torch.equal(a.membership_scores, b.membership_scores)
        assert torch.equal(a.selected, b.selected)
    a, _ = category_partition(field, field.semantic.codes[:3])
    b, _ = category_partition(loaded, field.semantic.codes[:3])
    assert torch.equal(a, b)


def test_visibility_witness_rejects_tiny_occluded_and_unobserved_rows():
    columns = csc_matrix(np.array([[.98, .02, 0., 0.], [0., 0., .1, .9]], np.float32))
    transfer, peak = view_transfer(columns, torch.tensor([True, True, True, False]))
    assert np.allclose(transfer.toarray(), [[1, 0], [0, 0], [0, 1], [0, 0]])
    assert peak[1] < .1
    scaled, _ = view_transfer(columns * .001, torch.tensor([True, True, True, False]))
    assert np.allclose(scaled.toarray(), transfer.toarray())


def test_missing_views_and_conflicts_remain_partial():
    source = SimpleNamespace(views=[
        dict(columns=csc_matrix(np.eye(3, dtype=np.float32)),
             positive=torch.tensor([[True, False, False]]), negative=torch.tensor([[False, True, False]])),
        dict(columns=csc_matrix(np.eye(3, dtype=np.float32)),
             positive=torch.tensor([[False, True, False]]), negative=torch.tensor([[True, False, False]]))])
    evidence = ElementRelations(source, torch.ones(3, dtype=torch.bool))
    pair = dict(primary_view=0, primary_region=0, secondary_view=1, regions=[0], weights=[1.])
    rows, target, confidence = evidence.combine(pair)
    assert rows.tolist() == [0, 1]  # unseen third element is not a negative
    torch.testing.assert_close(target, torch.tensor([.5, .5]))
    torch.testing.assert_close(confidence, torch.ones(2))
    pair.update(secondary_view=0)
    rows, target, confidence = evidence.combine(pair)
    torch.testing.assert_close(target, torch.tensor([1., 0.]))
    torch.testing.assert_close(confidence, torch.ones(2))  # repeated view is not extra confidence


def test_soft_element_loss_has_no_unknown_gradient():
    logits = torch.zeros(4, requires_grad=True)
    loss = element_loss(logits, torch.tensor([1., 0., .5, 1.]),
                        torch.tensor([1., 1., 1., 0.]), torch.tensor([1, 0, 1, 1]))
    loss.backward()
    assert logits.grad[0] < 0 and logits.grad[1] > 0
    assert logits.grad[2] == 0 and logits.grad[3] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_conditional_rows_match_full_decode_and_receive_gradients():
    field = metric_field().cuda(); enable(field, torch.ones(12))
    with torch.no_grad():
        field.relation.layers[-1].weight.fill_(.1)
    support = Support(torch.tensor([1, 3]), torch.tensor([.2, .8]))
    rows = torch.tensor([4, 2, 7, 4], device='cuda')
    expected = field.member_logits(support, .8)[rows]
    actual = field.member_logits(support, .8, rows)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    actual.mean().backward()
    assert torch.isfinite(field.group.grad).all() and field.relation.layers[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize('arm', ['element_relation', 'conditional_relation', 'joint_conditional_relation'])
def test_tiny_complete_fit_keeps_source_splits_and_export_contract(tmp_path, arm, monkeypatch):
    from radio_gs.query_field.field import CapabilityField
    count = 108; pixels = torch.arange(count); observations = tmp_path / 'source'; observations.mkdir()
    masks = torch.zeros(2, 9, 12, dtype=torch.bool)
    masks[0, 1:8, 1:5] = True; masks[1, 1:8, 8:12] = True
    entries = []
    for frame in [1, 2, 3]:
        view = dict(masks=masks, scales_m=torch.ones(2), raw_mass=torch.ones(count), best_element=pixels,
            operator=torch.sparse_coo_tensor(torch.stack([pixels, pixels]), torch.ones(count), (count, count)).coalesce(),
            intrinsic=torch.eye(3), world_to_camera=torch.eye(4), original_hw=[9, 12])
        path = observations / f'view_{frame:05d}.pt'; torch.save(view, path)
        entries.append(dict(frame=frame, split='development' if frame == 3 else 'train', compiled={'sha256': sha256(path)}))
    manifest = observations / 'observations.json'
    manifest.write_text(json.dumps(dict(frames=entries, world_units_per_metre=1.,
        raster_hw=[9, 12], splits=dict(train_frames=[1, 2], development_frames=[3]))))
    geometry = dict(xyz=torch.randn(count, 3), normals=torch.zeros(count, 3), quaternions=torch.zeros(count, 4),
                    scales=torch.ones(count, 3), opacities=torch.full((count, 1), .8))
    field = CapabilityField(geometry, torch.randn(count, 9), torch.randn(count, 6), torch.randn(count, 4),
        torch.ones(count, dtype=torch.bool), spatial_metric='scale_normalized', world_units_per_metre=1.)
    save_field(field, tmp_path / 'parent', dict(observations_sha256=sha256(manifest)))
    if arm == 'joint_conditional_relation':
        # CPU orchestration check. The separate CUDA case exercises the actual
        # subset rasterizer and full-row gradients beyond the element sample.
        monkeypatch.setattr('radio_gs.query_field.element_relation_training.relaxed_subset_alpha',
                            lambda field, membership, *args: membership * field.opacities.flatten())
    report = fit(tmp_path / 'parent', observations, tmp_path / arm, arm=arm, per_class=8, device='cpu')
    assert report['source_frames'] == [1, 2] and report['source_development']['frames'] == [3]
    assert report['steps'] == 8 and report['skipped_steps'] == 0 and report['protected_state_unchanged']
    loaded, _ = load_field(tmp_path / arm / 'field')
    assert loaded.relation_decoder == ('none' if arm == 'element_relation' else 'native_geometry_v1')
    if arm == 'joint_conditional_relation':
        assert report['rendered_weight'] == report['element_weight'] == .5
        assert report['rendered_logit_gradient']['nonzero_rows'] > 0
        assert report['rendered_logit_gradient']['absolute_gradient_total'] > 0
        assert 'refined_rendered_bce' in report['source_development']['metrics']


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA rasterization required')
def test_actual_subset_renderer_supervises_full_logits_outside_element_sample():
    from radio_gs.query_field.geometry import relaxed_subset_alpha
    from radio_gs.query_field.relation_training import weighted_partial_loss
    field = metric_field().cuda()
    with torch.no_grad():
        field.xyz[:, 0] = torch.linspace(-.4, .4, 12, device='cuda')
        field.xyz[:, 1] = 0; field.xyz[:, 2] = 2.
        field.scales.fill_(.07); field.opacities.fill_(.8)
        field.quaternions.zero_(); field.quaternions[:, 0] = 1.
    enable(field, torch.ones(12))
    camera = SimpleNamespace(intrinsic=torch.tensor([[50., 0., 32.], [0., 50., 32.], [0., 0., 1.]]),
                             world_to_camera=torch.eye(4), width=64, height=64)
    support = Support(torch.tensor([0], device='cuda'), torch.ones(1, device='cuda'))
    logits = field.support_logits(support, 1.); logits.retain_grad()
    rendered = relaxed_subset_alpha(field, logits.sigmoid(), camera, 64, 64, field.world_units_per_metre)
    pixels = torch.arange(64*64, device='cuda')
    positive = (pixels % 64 < 32).float(); negative = 1 - positive
    (.25 * weighted_partial_loss(rendered, positive, negative)).backward()
    assert torch.isfinite(logits.grad).all() and logits.grad[2:].abs().sum() > 0
    assert field.relation.layers[-1].weight.grad.abs().sum() > 0
    assert all(value.grad is None for value in field.buffers())
