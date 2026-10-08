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
