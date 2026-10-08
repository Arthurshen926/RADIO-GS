import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from radio_gs.query_field.field import NativeCoordinates
from radio_gs.query_field.identity_evidence import bidirectional_local_match, native_rank_fusion
from radio_gs.querying.local_appearance_match import match_regions
from radio_gs.querying.observation_readout import mean_native_identity_ranks
from radio_gs.query_field.identity_evidence import IDENTITY_CONTROLS, image_identity_controls
from radio_gs.query_field.storage import save_field, load_field
from radio_gs.query_field.contracts import QueryPacket
from test_query_preserving_field import example


def test_ordinal_fusion_matches_v10_ties_and_excludes_unavailable():
    first = torch.tensor([.2, .2, 100., .8, -.1])
    second = torch.tensor([.9, .1, 100., .1, .2])
    available = torch.tensor([True, True, False, True, True])
    expected = mean_native_identity_ranks(first.numpy(), second.numpy(), np.flatnonzero(available.numpy()))
    actual = native_rank_fusion(first, second, available)
    torch.testing.assert_close(actual, torch.from_numpy(expected), rtol=0, atol=0)
    torch.testing.assert_close(actual, native_rank_fusion(first * 2 + 7, second * 3 - 9, available))
    assert actual[2] == 0


def test_bidirectional_matching_matches_v10_with_weighted_patch_sets():
    torch.manual_seed(4)
    source = torch.randn(13, 7)
    query = torch.randn(5, 7)
    weights = torch.tensor([1., 0., 3., 2., 0., 4., 1., 2., 1., 0., 1., 2., 3.])
    query_weights = torch.tensor([1., 2., 0., 3., 1.])
    expected = match_regions(source.numpy(), csr_matrix(weights.numpy()[None]), [0, 13], [0, 1],
                             [query.numpy()], [query_weights.numpy()], device='cpu')['chamfer'][0, 0]
    for chunk in [1, 4, 30]:
        actual = bidirectional_local_match(NativeCoordinates(source), torch.arange(13), query,
                    field_weights=weights, query_weights=query_weights, chunk_size=chunk)
        assert abs(float(actual.score) - expected) < 1e-6


def test_reverse_coverage_rejects_single_shared_patch_shortcut():
    source = NativeCoordinates(torch.eye(3))
    query = torch.eye(3)[:2]
    partial = bidirectional_local_match(source, torch.tensor([0]), query)
    complete = bidirectional_local_match(source, torch.tensor([0, 1]), query)
    # Anchor maximum is one for both; complete query coverage distinguishes them.
    assert partial.reverse == complete.reverse == 1
    assert partial.forward == .5 and complete.score == 1
    assert partial.score < complete.score


def test_affine_codec_and_split_duplicate_mass_preserve_correspondence():
    torch.manual_seed(10)
    codes, weight, bias = torch.randn(6, 3), torch.randn(8, 3), torch.randn(8)
    codec = NativeCoordinates(codes, weight, bias)
    explicit = NativeCoordinates(codes @ weight.T + bias)
    query = torch.randn(4, 8)
    a = bidirectional_local_match(codec, torch.arange(6), query)
    b = bidirectional_local_match(explicit, torch.arange(6), query)
    c = bidirectional_local_match(codec, torch.arange(6).repeat_interleave(2), query,
                                  field_weights=torch.full((12,), .5))
    torch.testing.assert_close(a.score, b.score)
    torch.testing.assert_close(a.score, c.score)


@pytest.mark.parametrize('weights', [torch.zeros(2), torch.tensor([1., -1.]), torch.tensor([1., float('nan')])])
def test_invalid_correspondence_mass_is_rejected(weights):
    with pytest.raises(ValueError, match='weights'):
        bidirectional_local_match(NativeCoordinates(torch.eye(2)), torch.arange(2), torch.eye(2),
                                  field_weights=weights)


def test_controls_preserve_parent_field_and_replay_from_standalone_export(tmp_path):
    field = example()
    before = {k: v.clone() for k, v in field.state_dict().items()}
    semantic, local = field.semantic.codes[0], field.local.codes[:3]
    expected = field.query(QueryPacket('image', semantic=semantic, local=local))
    outputs, diagnostics = image_identity_controls(field, semantic, local)
    assert torch.equal(outputs['parent']['membership'], expected.membership_scores)
    assert torch.equal(outputs['rank_same_candidates']['candidates'], expected.candidate_indices)
    assert set(outputs) == set(IDENTITY_CONTROLS)
    for k, v in before.items():
        assert torch.equal(field.state_dict()[k], v)
    save_field(field, tmp_path / 'export', {'source': '/nonexistent/source'})
    loaded, _ = load_field(tmp_path / 'export')
    replay, _ = image_identity_controls(loaded, semantic, local)
    for arm in IDENTITY_CONTROLS:
        assert torch.equal(outputs[arm]['membership'], replay[arm]['membership'])
    assert diagnostics['shared_member_core'] and not diagnostics['source_access']


def test_empty_predicted_scope_does_not_force_inferred_anchor_positive():
    field = example()
    with torch.no_grad():
        field.affinity_bias.fill_(2.)
    outputs, _ = image_identity_controls(field, field.semantic.codes[0], field.local.codes[:3])
    result = outputs['bidirectional_same_candidates']
    assert result['chosen'] == -1 and not result['membership'].any()
    assert not len(result['candidates'])
