import torch

from radio_gs.query_field.contracts import QueryPacket, Support
from radio_gs.query_field.query_program import query, category_partition
from radio_gs.query_field.storage import save_field, load_field
from test_query_preserving_field import example


def test_nonimage_inputs_retain_exact_shared_member_behavior():
    field = example()
    packets = [QueryPacket('text', semantic=field.semantic.codes[1], task='union'),
        QueryPacket('point_2d', positive=Support(torch.tensor([1, 2]), torch.tensor([.3, .7]))),
        QueryPacket('point_3d', positive=Support(torch.tensor([1]), torch.ones(1)), exact_positive=True)]
    for p in packets:
        a, b = query(field, p), field.query(p)
        assert torch.equal(a.membership_scores, b.membership_scores)
        assert torch.equal(a.selected, b.selected)


def test_image_program_uses_one_common_member_hypothesis_and_replays(tmp_path):
    field = example()
    p = QueryPacket('image', semantic=field.semantic.codes[2], local=field.local.codes[2:5])
    before = {k: v.clone() for k, v in field.state_dict().items()}
    result = query(field, p)
    assert result.diagnostics['shared_member_core'] and not result.diagnostics['source_access']
    if len(result.candidate_indices):
        anchor = result.candidate_indices[result.candidate_weights.argmax()]
        expected = field.member_logits(Support(anchor.reshape(1), torch.ones(1)), result.scale).sigmoid()
        assert torch.equal(result.membership_scores, expected)
    for k, v in before.items(): assert torch.equal(v, field.state_dict()[k])
    save_field(field, tmp_path / 'field', {})
    loaded, _ = load_field(tmp_path / 'field')
    assert torch.equal(query(loaded, p).membership_scores, result.membership_scores)


def test_category_program_keeps_unobserved_void():
    field = example(); field.observed[0] = False
    scores, labels = category_partition(field, field.semantic.codes[1:4])
    assert scores.shape == (12, 3) and labels[0] == -1
