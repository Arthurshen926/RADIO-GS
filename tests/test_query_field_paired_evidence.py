import json

import numpy as np
import pytest
from scipy import sparse
import torch

from radio_gs.query_field.compile_paired_evidence import csr_state, compile_field
from radio_gs.query_field.contracts import Support, QueryPacket
from radio_gs.query_field.paired_evidence import PairedEvidence, enable
from radio_gs.query_field.storage import load_field, save_field, sha256
from test_query_preserving_field import example


def state(positive, negative, views, scales=None):
    return dict(**csr_state(sparse.csr_matrix(np.array(positive, np.float32)), 'positive'),
                **csr_state(sparse.csr_matrix(np.array(negative, np.float32)), 'negative'),
                views=torch.tensor(views), scales=torch.ones(len(views)) if scales is None else torch.tensor(scales))


def fixture():
    return state([[1, 1, 0], [1, 0, 1], [0, 0, 0], [0, 0, 0]],
                 [[0, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 0]], [0, 0, 1])


def test_partial_positive_exclusion_and_unknown_are_distinct():
    evidence = PairedEvidence(fixture(), 4)
    support = Support(torch.tensor([0]), torch.ones(1))
    pos, neg = evidence.masses(support, 1.)
    torch.testing.assert_close(pos, torch.tensor([1., .5, 0., 0.]))
    torch.testing.assert_close(neg, torch.tensor([0., .5, 1., 0.]))
    logits = torch.tensor([0., 0., 0., -100.])
    answer = evidence.refine(logits, support, 1.)
    torch.testing.assert_close(answer[:3].sigmoid(), torch.tensor([.75, .5, .25]))
    assert answer[-1] == logits[-1]  # exact absence, including saturated logits


def test_nested_duplicates_do_not_create_views_or_change_confidence():
    original = PairedEvidence(fixture(), 4)
    duplicated = PairedEvidence(state([[1, 1, 1, 1, 0], [1, 0, 1, 0, 1], [0]*5, [0]*5],
        [[0]*5, [0, 1, 0, 1, 0], [1, 1, 1, 1, 0], [0]*5], [0, 0, 0, 0, 1]), 4)
    support = Support(torch.tensor([0]), torch.ones(1))
    for a, b in zip(original.masses(support, 1.), duplicated.masses(support, 1.)):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA required'))])
def test_weighted_repeated_support_rows_and_partial_decode(device):
    evidence = PairedEvidence(fixture(), 4).to(device)
    a = Support(torch.tensor([0, 1]), torch.tensor([.25, .75]))
    b = Support(torch.tensor([0, 1, 1]), torch.tensor([.25, .25, .5]))
    rows = torch.tensor([3, 1, 0, 1], device=device)
    for full, sub, duplicate in zip(evidence.masses(a, 1.), evidence.masses(a, 1., rows), evidence.masses(b, 1.)):
        torch.testing.assert_close(full[rows], sub)
        torch.testing.assert_close(full, duplicate)


def test_scale_mismatch_is_unknown_and_no_support_is_forced_positive():
    evidence = PairedEvidence(fixture(), 4)
    logits = torch.randn(4)
    actual = evidence.refine(logits, Support(torch.tensor([0]), torch.ones(1)), 10.)
    assert torch.equal(logits, actual)
    unknown = Support(torch.tensor([3]), torch.ones(1))
    assert torch.equal(logits, evidence.refine(logits, unknown, 1.))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_soft_relation_function_matches_cpu_on_cuda():
    rng = np.random.default_rng(20261009)
    positive = rng.uniform(0, 1, (512, 64)).astype(np.float32)
    negative = 1 - positive
    unknown = rng.uniform(0, 1, positive.shape) < .35
    positive[unknown] = 0; negative[unknown] = 0
    data = state(positive, negative, np.repeat(np.arange(8), 8).tolist())
    cpu = PairedEvidence(data, 512)
    gpu = PairedEvidence({k: v.clone() for k, v in data.items()}, 512).cuda()
    support = Support(torch.tensor([2, 51, 14]), torch.tensor([.2, .7, .1]))
    logits = torch.from_numpy(rng.normal(size=512).astype(np.float32))
    expected = cpu.refine(logits, support, 1.)
    actual = gpu.refine(logits.cuda(), support, 1.).cpu()
    assert not torch.equal(expected, logits)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)


def test_export_replay_all_input_types_and_legacy_native_state(tmp_path):
    field = example(); before = {k: v.clone() for k, v in field.state_dict().items()}
    pos = np.zeros((12, 3), np.float32); neg = pos.copy()
    pos[:4, :2] = 1; neg[4:8, :2] = 1; pos[4:8, 2] = 1
    enable(field, state(pos, neg, [0, 0, 1]))
    save_field(field, tmp_path / 'field', {})
    loaded, manifest = load_field(tmp_path / 'field')
    assert manifest['spec']['evidence_kind'] == field.evidence_kind
    assert manifest['external_scene_dependencies'] == []
    for key, value in before.items(): assert torch.equal(value, loaded.state_dict()[key])
    positive = Support(torch.tensor([0]), torch.ones(1)); negative = Support(torch.tensor([5]), torch.ones(1))
    for packet in [QueryPacket('text', task='union', semantic=field.semantic.codes[0]),
                   QueryPacket('image', semantic=field.semantic.codes[0], local=field.local.codes[:2]),
                   QueryPacket('point_2d', positive=positive, negative=negative),
                   QueryPacket('point_3d', positive=positive, negative=negative, exact_positive=True)]:
        a, b = field.query(packet), loaded.query(packet)
        assert torch.equal(a.membership_scores, b.membership_scores)
        assert torch.equal(a.selected, b.selected)
    for a, b in zip(field.category_partition(field.semantic.codes[:3]), loaded.category_partition(field.semantic.codes[:3])):
        assert torch.equal(a, b)


def test_compiler_excludes_development_and_replays_source_functions(tmp_path):
    from radio_gs.query_field.field import CapabilityField
    count = 108; pixels = torch.arange(count); source = tmp_path / 'source'; source.mkdir()
    masks = torch.zeros(2, 9, 12, dtype=torch.bool)
    masks[0, 1:8, 1:5] = True; masks[1, 1:8, 8:12] = True
    records = []
    for frame in [1, 2, 3]:
        path = source / f'view_{frame:05d}.pt'
        torch.save(dict(masks=masks, scales_m=torch.ones(2), raw_mass=torch.ones(count), best_element=pixels,
            operator=torch.sparse_coo_tensor(torch.stack([pixels, pixels]), torch.ones(count), (count, count)).coalesce(),
            intrinsic=torch.eye(3), world_to_camera=torch.eye(4), original_hw=[9, 12]), path)
        records.append(dict(frame=frame, split='development' if frame == 3 else 'train', compiled={'sha256': sha256(path)}))
    manifest = source / 'observations.json'
    manifest.write_text(json.dumps(dict(frames=records, world_units_per_metre=1., raster_hw=[9, 12],
        splits=dict(train_frames=[1, 2], development_frames=[3]))))
    geometry = dict(xyz=torch.randn(count, 3), normals=torch.zeros(count, 3), quaternions=torch.zeros(count, 4),
        scales=torch.ones(count, 3), opacities=torch.full((count, 1), .8))
    field = CapabilityField(geometry, torch.randn(count, 9), torch.randn(count, 6), torch.randn(count, 4),
        torch.ones(count, dtype=torch.bool), world_units_per_metre=1.)
    save_field(field, tmp_path / 'parent', dict(observations_sha256=sha256(manifest)))
    # Unreadable held-out features must not impede construction.
    (source / 'view_00003.pt').unlink()
    report = compile_field(tmp_path / 'parent', source, tmp_path / 'out')
    assert report['source_frames'] == [1, 2] and report['source_regions'] == 4
    assert report['protected_state_unchanged'] and report['training_steps'] == 0
    assert len(report['replay']) == 4 and max(r['positive_error'] for r in report['replay']) == 0


def test_invalid_export_mass_is_rejected():
    data = fixture(); data['positive_value'][0] = float('nan')
    with pytest.raises(ValueError, match='mass'): PairedEvidence(data, 4)
