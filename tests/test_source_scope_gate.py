import torch
from radio_gs.models.grounded_semantic_transfer import GroundedSemanticTransfer
from scripts.audits.fit_source_scope_gate import coobserved_scope


def test_missing_scope_is_unknown_and_overlap_is_not_semantic_identity():
    masks=torch.tensor([[[1,1,0]],[[0,1,1]],[[0,0,0]]],dtype=torch.bool)
    overlap,known=coobserved_scope(masks,torch.tensor([True,True,False]))
    assert not known[2].any() and not known[:,2].any()
    torch.testing.assert_close(overlap[0,1],torch.tensor(1/3))
    assert overlap[0,0]==1


def test_source_scope_veto_does_not_reroute_or_admit_rejected_queries():
    config=dict(latent_dim=6,native_dim=4,query_dim=4,num_groups=2,routing={'scope_gate':True})
    head=GroundedSemanticTransfer(**config)
    with torch.no_grad():
        head.prototypes.copy_(torch.eye(4)[:2]);head.acceptance.fill_(.9)
        head.routing_scope_images.copy_(torch.eye(4)[[2,1]])
        head.routing_scope_weights.copy_(torch.tensor([1.,1.]));head.routing_scope_bias.fill_(-1.5)
    query=torch.eye(4)[:3]
    index,accepted,_=head.route(query)
    assert index.tolist()==[0,1,0] and accepted.tolist()==[False,True,False]
    cold=GroundedSemanticTransfer(**config);cold.load_state_dict(head.state_dict())
    for i in range(3):
        assert torch.equal(cold.route(query[i:i+1])[1],accepted[i:i+1])


def test_hard_scope_pairs_choose_similar_known_negatives_and_nonself_positives():
    from scripts.audits.fit_source_scope_gate import select_hard_scope_pairs
    similarity=torch.tensor([[1.,.9,.8,.99],[.5,.4,.9,.99]])
    overlap=torch.tensor([[1.,.8,0.,0.],[.8,0.,0.,0.]])
    known=torch.tensor([[True,True,True,False],[True,True,True,False]])
    self_pair=torch.tensor([[True,False,False,False],[False,False,False,False]])
    chosen=select_hard_scope_pairs(similarity,overlap,known,self_pair)
    assert chosen.tolist()==[[False,True,True,False],[True,False,True,False]]


def test_scope_veto_releases_accepted_concept_to_structural_fallback():
    h=GroundedSemanticTransfer(3,2,4,1,num_proposals=2,
        routing={'scope_gate':True},identity_selection={'metric':'support_iou','trusted_concept_start':1})
    with torch.no_grad():
        h.prototypes.copy_(torch.eye(4)[:1]);h.acceptance.fill_(.9)
        h.routing_scope_bias.fill_(-1)
        h.decoder.weight.zero_();h.decoder.bias.copy_(torch.tensor([.8,.2]))
    logits,accepted=h(torch.eye(4)[:1],torch.randn(5,3),native_support=torch.ones(5,1,dtype=torch.bool),observed=torch.ones(5,dtype=torch.bool))
    assert accepted.all()
    torch.testing.assert_close(logits.sigmoid(),torch.full((5,1),.8))
