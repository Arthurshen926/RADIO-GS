import json

import torch

from radio_gs.query_field.contracts import Support, QueryPacket
from radio_gs.query_field.paired_evidence import PairedEvidence
from radio_gs.query_field.soft_paired_evidence import SoftPairedEvidence, enable
from radio_gs.query_field.storage import load_field, save_field
from test_query_field_paired_evidence import state, fixture
from test_query_preserving_field import example


def soft_fixture():
    hard = fixture()
    soft = state([[1, 1, 0], [1, 0, 1], [0, 0, 0], [.8, .8, 0]],
                 [[0, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 0]], [0, 0, 1])
    soft.update({k.replace('positive_', 'association_'): v for k, v in hard.items() if k.startswith('positive_')})
    return soft


def test_soft_targets_do_not_change_any_hard_support_coefficients():
    hard = PairedEvidence(fixture(), 4); soft = SoftPairedEvidence(soft_fixture(), 4)
    for support in [Support(torch.tensor([0]), torch.ones(1)), Support(torch.tensor([3]), torch.ones(1)),
                    Support(torch.tensor([0, 3]), torch.tensor([.25, .75]))]:
        assert torch.equal(hard.coefficients(support, 1.), soft.coefficients(support, 1.))
    assert soft.coefficients(Support(torch.tensor([3]), torch.ones(1)), 1.).sum() == 0


def test_soft_targets_keep_strong_rows_and_add_fractional_evidence():
    hard = PairedEvidence(fixture(), 4); soft = SoftPairedEvidence(soft_fixture(), 4)
    support = Support(torch.tensor([0]), torch.ones(1)); logits = torch.zeros(4)
    before, after = hard.refine(logits, support, 1.), soft.refine(logits, support, 1.)
    assert torch.equal(before[:3], after[:3])
    torch.testing.assert_close(after[3].sigmoid(), torch.tensor(1.3 / 1.8))
    assert before[3] == 0


def test_soft_export_replays_same_common_field(tmp_path):
    field = example(); data = soft_fixture()
    for prefix in ['positive', 'negative', 'association']:
        pointer = data[prefix + '_pointer']
        data[prefix + '_pointer'] = torch.cat([pointer, pointer[-1].expand(8)])
    enable(field, data); save_field(field, tmp_path / 'field', {})
    loaded, manifest = load_field(tmp_path / 'field')
    packet = QueryPacket('point_3d', positive=Support(torch.tensor([0]), torch.ones(1)), exact_positive=True)
    assert torch.equal(field.query(packet).membership_scores, loaded.query(packet).membership_scores)
    assert manifest['external_scene_dependencies'] == []


def test_compiler_adds_weak_target_rows_without_changing_associations(tmp_path):
    from radio_gs.query_field.field import CapabilityField
    from radio_gs.query_field.storage import sha256
    from radio_gs.query_field.compile_paired_evidence import compile_field as hard_compile
    from radio_gs.query_field.compile_soft_paired_evidence import compile_field as soft_compile
    pixels = torch.arange(108); count = 216; source = tmp_path / 'source'; source.mkdir()
    masks = torch.zeros(2, 9, 12, dtype=torch.bool)
    masks[0, 1:8, 1:5] = True; masks[1, 1:8, 8:12] = True
    operator = torch.sparse_coo_tensor(torch.stack([pixels.repeat_interleave(2), torch.arange(count)]),
        torch.tensor([.98, .02]).repeat(108), (108, count)).coalesce()
    records = []
    for frame in [1, 2]:
        path = source / f'view_{frame:05d}.pt'
        torch.save(dict(masks=masks, scales_m=torch.ones(2), raw_mass=torch.ones(108), best_element=2*pixels,
            operator=operator, intrinsic=torch.eye(3), world_to_camera=torch.eye(4), original_hw=[9, 12]), path)
        records.append(dict(frame=frame, split='train', compiled={'sha256': sha256(path)}))
    manifest = source / 'observations.json'
    manifest.write_text(json.dumps(dict(frames=records, world_units_per_metre=1., raster_hw=[9, 12],
        splits=dict(train_frames=[1, 2], development_frames=[3]))))
    geometry = dict(xyz=torch.randn(count, 3), normals=torch.zeros(count, 3), quaternions=torch.zeros(count, 4),
        scales=torch.ones(count, 3), opacities=torch.full((count, 1), .8))
    field = CapabilityField(geometry, torch.randn(count, 9), torch.randn(count, 6), torch.randn(count, 4),
        torch.ones(count, dtype=torch.bool), world_units_per_metre=1.)
    save_field(field, tmp_path / 'parent', dict(observations_sha256=sha256(manifest)))
    hard_compile(tmp_path / 'parent', source, tmp_path / 'hard')
    report = soft_compile(tmp_path / 'parent', source, tmp_path / 'soft', tmp_path / 'hard/field')
    coverage = report['relation_availability']
    assert report['exact_hard_association_coordinates'] and report['protected_state_unchanged']
    assert coverage['available_rows'] == 2 * coverage['hard_available_rows'] > 0
    assert report['source_frames'] == [1, 2] and len(report['replay']) == 4
    from radio_gs.query_field.compile_mixture_evidence import compile_field as mixture_compile
    mixture = mixture_compile(tmp_path / 'soft/field', source, tmp_path / 'mixture', confidence_weighted=True)
    assert mixture['exact_tensor_payload'] and mixture['additional_tensor_bytes'] == 0
    assert mixture['parent_sha256'] == mixture['export_sha256']
    assert len(mixture['replay']) == 4 and all(r['known_elements'] for r in mixture['replay'])
    assert {r['kind'] for r in mixture['replay']} == {'point', 'two_full_rays'}
    loaded, _ = load_field(tmp_path / 'mixture/field')
    assert loaded.evidence_kind == 'confidence_mixture_partial_view_evidence_v1'
