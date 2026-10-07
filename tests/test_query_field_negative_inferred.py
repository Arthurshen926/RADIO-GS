"""Shared signed evidence must also preserve inferred text/image identities."""
import pytest
import torch
from torch.nn import functional as F

from radio_gs.query_field.contracts import QueryPacket, Support
from test_query_preserving_field import example


@pytest.mark.parametrize('modality', ['text', 'image'])
@pytest.mark.parametrize('task', ['instance', 'union'])
def test_signed_inferred_query_competes_after_original_identity_selection(modality, task):
    field = example()
    field.negative_rule = 'competitive_pooled'
    semantic = field.semantic.codes[0]
    local = None if modality == 'text' else field.local.codes[1:3]
    negative = Support(torch.tensor([4, 5]), torch.tensor([.3, .7]))
    unsigned = field.query(QueryPacket(modality, task=task, semantic=semantic, local=local))
    signed = field.query(QueryPacket(modality, task=task, semantic=semantic, local=local, negative=negative))
    assert torch.equal(unsigned.semantic_scores, signed.semantic_scores)
    assert torch.equal(unsigned.candidate_indices, signed.candidate_indices)
    assert torch.equal(unsigned.candidate_weights, signed.candidate_weights)
    identity = field.semantic.score(semantic).flatten()
    if local is not None:
        identity = (identity+field.local.score(local, patch_max=True))*.5
    ids = unsigned.candidate_indices
    logits = torch.stack([field.member_logits(Support(i.reshape(1), torch.ones(1)), unsigned.scale) for i in ids], 1)
    if task == 'instance':
        positive = logits[:, unsigned.candidate_weights.argmax()]
    else:
        admitted = identity[ids] >= identity[ids].max()-.02
        positive = logits[:, admitted].max(1).values
    expected = (positive-F.softplus(field.member_logits(negative, unsigned.scale))).sigmoid()
    expected = expected.masked_fill(~field.observed, 0)
    assert torch.equal(signed.membership_scores, expected)
    assert bool((signed.membership_scores <= unsigned.membership_scores).all())
