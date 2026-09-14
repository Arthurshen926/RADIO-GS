import torch
from radio_gs.v4.query.anchored_fragments import anchored_fragment_extent


def test_anchor_completion_excludes_unrelated_high_similarity_fragment():
    positive = torch.tensor([[1., 1., 0., 0.], [0., 1., 1., 0.], [0., 0., 0., 1.]])
    sim = torch.tensor([[.9, .89, .895]])
    out, audit = anchored_fragment_extent(positive, sim, torch.tensor([0, 1, 1]))
    assert torch.equal(out[:, 0], torch.tensor([1., 1., 1., 0.]))
    assert audit["selected_fragments"] == [[0, 1]]


def test_duplicate_view_evidence_does_not_accumulate_and_empty_stays_empty():
    p = torch.tensor([[1., .4], [1., .4], [0., 0.]])
    out, _ = anchored_fragment_extent(p, torch.tensor([[.9, .9, 1.]]), torch.tensor([0, 0, 1]))
    assert torch.equal(out[:, 0], p[0])
    out, _ = anchored_fragment_extent(torch.zeros_like(p), torch.ones(1, 3), torch.tensor([0, 0, 1]))
    assert not out.any()
