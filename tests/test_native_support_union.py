import torch
from radio_gs.models.grounded_semantic_transfer import greedy_native_support_union,GroundedSemanticTransfer


def test_union_adds_complementary_support_and_rejects_duplicates_and_background():
    masks=torch.tensor([[1,0,1,0],[1,0,1,0],[0,1,0,0],[0,0,0,1]],dtype=torch.bool)
    support=torch.tensor([[1,0],[1,0],[1,0],[0,0]],dtype=torch.bool)
    union,quality,chosen=greedy_native_support_union(masks,support)
    assert chosen==[[0,1],[]]
    assert torch.equal(union,support)
    torch.testing.assert_close(quality,torch.tensor([1.,0.]))
    for qi in range(2):
        one,q,_=greedy_native_support_union(masks,support[:,qi:qi+1])
        assert torch.equal(one,union[:,qi:qi+1]);assert q==quality[qi]


def test_union_does_not_trade_coverage_for_lower_native_iou():
    masks=torch.tensor([[1,1],[1,1],[0,1],[0,1],[0,1],[0,1]],dtype=torch.bool)
    support=torch.tensor([[1],[1],[1],[0],[0],[0]],dtype=torch.bool)
    union,quality,chosen=greedy_native_support_union(masks,support)
    assert chosen==[[0]]
    torch.testing.assert_close(quality,torch.tensor([2/3]))


def test_union_preserves_source_accepted_head_and_full_scope_on_row_subsets():
    head=GroundedSemanticTransfer(3,2,4,1,num_proposals=3,identity_selection={
        'metric':'support_iou','trusted_concept_start':2,'combine_candidates':'greedy_union'})
    with torch.no_grad():
        head.prototypes.copy_(torch.eye(4)[:1]);head.acceptance.fill_(.9)
        head.decoder.weight.copy_(torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,0.]]));head.decoder.bias.copy_(torch.tensor([0.,0.,.7]))
    latent=torch.tensor([[1.,0.,1.],[0.,1.,1.],[-1.,0.,1.]])
    observed=torch.tensor([True,True,False]);support=torch.ones(3,2,dtype=torch.bool);q=torch.eye(4)[:2]
    logits,accepted=head(q,latent,native_support=support,observed=observed)
    assert accepted.all();torch.testing.assert_close(logits[:,0].sigmoid(),torch.full((3,),.7))
    sub=head(q,latent,torch.tensor([0,2]),native_support=support,observed=observed)[0]
    torch.testing.assert_close(sub,logits[[0,2]])


def test_precision_guard_rejects_iou_gain_that_dilutes_native_evidence():
    masks=torch.tensor([[1,0],[0,1],[0,1],[0,1],[0,1]],dtype=torch.bool)
    target=torch.tensor([[1],[1],[1],[1],[0]],dtype=torch.bool)
    _,_,plain=greedy_native_support_union(masks,target)
    _,_,guarded=greedy_native_support_union(masks,target,preserve_precision=True)
    # Best single is candidate 1 (IoU .6). Candidate 0 adds a true positive,
    # improving both precision and IoU, so remains admissible.
    assert plain==guarded==[[1,0]]
    masks=torch.tensor([[1,0],[1,0],[0,1],[0,1],[0,1]],dtype=torch.bool)
    target=torch.tensor([[1],[1],[1],[1],[0]],dtype=torch.bool)
    _,_,plain=greedy_native_support_union(masks,target)
    _,_,guarded=greedy_native_support_union(masks,target,preserve_precision=True)
    assert plain==[[0,1]] and guarded==[[0]]
