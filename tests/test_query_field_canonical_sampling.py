import torch
from scripts.research.run_query_field_canonical_sampling import sampling_mask, sampled_state


def test_sampling_never_leaves_its_canonical_phase_and_spans_the_view():
    for phase in range(4):
        mask = sampling_mask(phase); y, x = mask.nonzero(as_tuple=True)
        assert len(y) == 256 and len(y.unique()) == len(x.unique()) == 16
        assert (y % 2 == phase // 2).all() and (x % 2 == phase % 2).all()
        assert int(y.min()) < 24 and int(y.max()) > 360
        assert int(x.min()) < 32 and int(x.max()) > 480


def test_sampling_only_censors_negatives_and_fills_no_source_state():
    prior = torch.tensor([[.8, .6], [.2, .9]])
    negative = torch.ones(192*256, dtype=torch.bool); negative[:20000] = False
    state = {0: dict(view_indices=[2], confidence=[.7], negative=[negative],
                    start_membership=torch.tensor([.1, .9]), support_cost=torch.tensor([1., 1.]))}
    new = sampled_state(prior, state)
    assert not (new[0]['negative'][0] & ~negative).any()
    assert 0 < int(new[0]['negative'][0].sum()) < 256
    assert torch.equal(new[0]['start_membership'], state[0]['start_membership'])
    assert torch.equal(new[0]['support_cost'], state[0]['support_cost'])
    assert new[1]['view_indices'] == [] and torch.equal(new[1]['start_membership'], prior[:, 1])
    assert int(negative.sum()) == 192*256-20000
