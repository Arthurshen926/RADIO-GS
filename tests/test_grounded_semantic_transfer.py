import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField


def fields(hidden_dim=0):
    kwargs=dict(num_elements=12,latent_dim=10,query_dims={'text':16,'image':16,'prompt':8},
                shared_query_groups=[['text','image']],native_siglip_readout=True,native_semantic_dim=6,
                calibrated_readout=True,native_query_thresholds={'text':{'lower':-.2,'upper':.2}})
    original=ElementRetrievalField(**kwargs);original.observed[:]=True
    updated=ElementRetrievalField(**kwargs,grounded_transfer={'num_groups':2,'hidden_dim':hidden_dim})
    updated.load_state_dict(original.state_dict(),strict=False)
    with torch.no_grad():
        updated.grounded_transfer.prototypes.copy_(torch.eye(16)[:2]);updated.grounded_transfer.acceptance.fill_(.9)
    return original,updated


def test_unknown_concepts_fall_back_exactly_and_parent_capabilities_survive():
    old,new=fields();query=torch.eye(16)[2:4];prompt=torch.randn(3,8)
    assert torch.equal(old.logits(query,'text'),new.logits(query,'text'))
    assert torch.equal(old.native_cosine(query),new.native_cosine(query))
    assert torch.equal(old.logits(query,'image'),new.logits(query,'image'))
    assert torch.equal(old.logits(prompt,'prompt'),new.logits(prompt,'prompt'))
    for key,value in old.state_dict().items():assert torch.equal(value,new.state_dict()[key])


def test_accepted_extent_is_query_conditioned_and_batch_row_invariant():
    old,new=fields();query=torch.eye(16)[:3];rows=torch.tensor([0,3,9])
    proposed,accepted=new.grounded_transfer(query,new.latent)
    assert accepted.tolist()==[True,True,False]
    logits=new.logits(query,'text')
    torch.testing.assert_close(logits[:,:2],proposed[:,:2])
    assert torch.equal(logits[:,2:],old.logits(query,'text')[:,2:])
    torch.testing.assert_close(logits[:,2:],old.logits(query[2:],'text'))
    torch.testing.assert_close(new.logits(query,'text',rows),logits[rows])
    torch.testing.assert_close(new.logits(query[:1],'text'),logits[:,:1])
    new.observed[0]=False;assert not new.retrieve(query,'text')['scores'][0].any()


def test_cold_state_loading_reproduces_gated_readout():
    _,new=fields();_,other=fields();other.load_state_dict(new.state_dict(),strict=True)
    query=torch.eye(16)[:4]
    assert torch.equal(new.logits(query,'text'),other.logits(query,'text'))


def test_nonlinear_head_keeps_fallback_and_uses_selected_logits():
    old,new=fields(hidden_dim=7);query=torch.eye(16)[:3]
    head=new.grounded_transfer;all_logits=head.decoder(head.features(new.latent))
    torch.testing.assert_close(new.logits(query,'text')[:,:2],all_logits)
    assert torch.equal(new.logits(query[2:],'text'),old.logits(query[2:],'text'))
    rows=torch.tensor([1,4,9]);torch.testing.assert_close(new.logits(query,'text',rows),new.logits(query,'text')[rows])
    _,cold=fields(hidden_dim=7);cold.load_state_dict(new.state_dict(),strict=True)
    assert torch.equal(new.logits(query,'text'),cold.logits(query,'text'))


def test_source_competitors_abstain_without_changing_extent_or_batch_decisions():
    from radio_gs.models.grounded_semantic_transfer import GroundedSemanticTransfer
    head=GroundedSemanticTransfer(10,6,4,2,routing={'num_competitors':1})
    with torch.no_grad():
        head.prototypes.copy_(torch.eye(4)[:2]);head.routing_competitors.copy_(torch.eye(4)[2:3])
        head.acceptance.fill_(-1)
    query=torch.eye(4)[:3]
    indices,accepted,_=head.route(query)
    assert accepted.tolist()==[True,True,False]
    for i in range(3):
        index,single,_=head.route(query[i:i+1])
        assert int(index)==int(indices[i]) and bool(single)==bool(accepted[i])
    latent=torch.randn(5,10)
    proposed,_=head(query,latent)
    torch.testing.assert_close(head(query,latent,torch.tensor([1,3]))[0],proposed[[1,3]])
    cold=GroundedSemanticTransfer(10,6,4,2,routing={'num_competitors':1})
    cold.load_state_dict(head.state_dict())
    assert torch.equal(head(query,latent)[0],cold(query,latent)[0])


def test_centering_only_changes_semantic_routing_metric():
    from radio_gs.models.grounded_semantic_transfer import GroundedSemanticTransfer
    head=GroundedSemanticTransfer(10,6,4,2,routing={'num_competitors':0})
    with torch.no_grad():
        head.prototypes.copy_(torch.tensor([[1.,1.,0.,0.],[1.,0.,1.,0.]]))
        head.routing_center.copy_(torch.tensor([.7,0.,0.,0.]));head.acceptance.fill_(.5)
    index,accepted,score=head.route(head.prototypes)
    assert index.tolist()==[0,1] and accepted.all()
    torch.testing.assert_close(score,torch.ones(2))


def test_residual_initializes_to_original_extent_and_restores_from_cold_state():
    from radio_gs.models.grounded_semantic_transfer import GroundedSemanticTransfer
    old=GroundedSemanticTransfer(10,6,4,2)
    new=GroundedSemanticTransfer(10,6,4,2,residual_dim=8)
    with torch.no_grad():old.prototypes.copy_(torch.eye(4)[:2])
    new.load_state_dict(old.state_dict(),strict=False)
    query=torch.eye(4)[:2];latent=torch.randn(9,10)
    assert torch.equal(old(query,latent)[0],new(query,latent)[0])
    with torch.no_grad():new.residual[2].weight.normal_(std=.01)
    cold=GroundedSemanticTransfer(10,6,4,2,residual_dim=8);cold.load_state_dict(new.state_dict())
    assert torch.equal(new(query,latent)[0],cold(query,latent)[0])
    torch.testing.assert_close(new(query,latent,torch.tensor([0,5]))[0],new(query,latent)[0][[0,5]])


def test_region_selection_is_concept_conditioned_and_empty_groups_fall_back():
    from radio_gs.models.grounded_semantic_transfer import GroundedSemanticTransfer
    head=GroundedSemanticTransfer(10,6,4,2,num_proposals=3,residual_dim=8)
    with torch.no_grad():
        head.prototypes.copy_(torch.eye(4)[:2]);head.acceptance.fill_(.9)
        head.proposal_embeddings.copy_(torch.tensor([[.8,0.,.6,0.],[1.,0.,0.,0.],[1.,0.,0.,0.]]))
        head.proposal_groups[:,0]=torch.tensor([True,True,False])
        head.decoder.weight.zero_();head.decoder.bias.copy_(torch.tensor([.2,.7,.9]))
    query=torch.eye(4)[:2];latent=torch.randn(5,10)
    logits,accepted=head(query,latent)
    assert accepted.tolist()==[True,False]
    torch.testing.assert_close(logits[:,0].sigmoid(),torch.full((5,),.7))
    cold=GroundedSemanticTransfer(10,6,4,2,num_proposals=3,residual_dim=8)
    cold.load_state_dict(head.state_dict());assert torch.equal(head(query,latent)[0],cold(query,latent)[0])


def test_source_ranker_uses_only_eligible_proposals_and_preserves_cold_state():
    from radio_gs.models.grounded_semantic_transfer import GroundedSemanticTransfer
    head=GroundedSemanticTransfer(10,6,4,2,num_proposals=3,proposal_ranker=True)
    with torch.no_grad():
        head.prototypes.copy_(torch.eye(4)[:2]);head.acceptance.fill_(.9)
        head.proposal_embeddings.copy_(torch.tensor([[1.,0.,0.,0.],[.8,0.,.6,0.],[1.,0.,0.,0.]]))
        head.proposal_groups[:,0]=torch.tensor([True,True,False])
        head.proposal_quality[0]=torch.tensor([.1,.8,1.])
        head.proposal_rank_weights.copy_(torch.tensor([1.,1.]))
        head.decoder.weight.zero_();head.decoder.bias.copy_(torch.tensor([.2,.7,.9]))
    query=torch.eye(4)[:2];latent=torch.randn(4,10)
    logits,accepted=head(query,latent)
    assert accepted.tolist()==[True,False]
    torch.testing.assert_close(logits[:,0].sigmoid(),torch.full((4,),.7))
    cold=GroundedSemanticTransfer(10,6,4,2,num_proposals=3,proposal_ranker=True)
    cold.load_state_dict(head.state_dict());assert torch.equal(logits,cold(query,latent)[0])


def test_native_extent_mean_preserves_purity_and_native_threshold():
    from radio_gs.models.grounded_semantic_transfer import GroundedSemanticTransfer
    head=GroundedSemanticTransfer(10,6,4,2,num_proposals=3,identity_selection={})
    with torch.no_grad():
        # Candidate 0 has higher normalized direction cosine, but weak mean evidence.
        head.native_extent_means.copy_(torch.tensor([[.1,0.,0.,0.],[.6,.4,0.,0.],[0.,0.,.7,0.]]))
        head.decoder.weight.zero_();head.decoder.bias.copy_(torch.tensor([.2,.7,.9]))
    query=torch.eye(4)[:3];latent=torch.randn(5,10)
    logits,accepted=head(query,latent,native_threshold=torch.tensor([.5,.5,.8]))
    assert accepted.tolist()==[True,False,False]
    torch.testing.assert_close(logits[:,0].sigmoid(),torch.full((5,),.7))
    rows=torch.tensor([1,3])
    torch.testing.assert_close(head(query,latent,rows,native_threshold=.5)[0],logits[rows])
    for i in range(3):
        torch.testing.assert_close(head(query[i:i+1],latent,native_threshold=.5)[0],logits[:,i:i+1])
    cold=GroundedSemanticTransfer(10,6,4,2,num_proposals=3,identity_selection={})
    cold.load_state_dict(head.state_dict())
    assert torch.equal(cold(query,latent,native_threshold=.5)[0],logits)


def test_native_identity_restriction_keeps_original_concept_gate():
    from radio_gs.models.grounded_semantic_transfer import GroundedSemanticTransfer
    head=GroundedSemanticTransfer(10,6,4,2,num_proposals=3,
        identity_selection={'restrict_concepts':True,'require_native_support':False})
    with torch.no_grad():
        head.prototypes.copy_(torch.eye(4)[:2]);head.acceptance.fill_(.9)
        head.proposal_groups[0,0]=True
        head.native_extent_means.copy_(torch.tensor([[.1,0.,0.,0.],[.8,0.,0.,0.],[0.,0.,.7,0.]]))
        head.decoder.weight.zero_();head.decoder.bias.copy_(torch.tensor([.2,.7,.9]))
    logits,accepted=head(torch.eye(4)[:3],torch.randn(5,10))
    assert accepted.tolist()==[True,False,False]
    torch.testing.assert_close(logits[:,0].sigmoid(),torch.full((5,),.2))


def test_native_identity_abstention_is_exact_field_fallback():
    old,_=fields()
    kwargs=dict(num_elements=12,latent_dim=10,query_dims={'text':16,'image':16,'prompt':8},
                shared_query_groups=[['text','image']],native_siglip_readout=True,native_semantic_dim=6,
                calibrated_readout=True,native_query_thresholds={'text':{'lower':-.2,'upper':.2}},
                grounded_transfer={'num_groups':2,'num_proposals':2,'identity_selection':{}})
    new=ElementRetrievalField(**kwargs);new.load_state_dict(old.state_dict(),strict=False)
    with torch.no_grad():new.grounded_transfer.native_extent_means.fill_(-.9)
    query=torch.eye(16)[:3]
    assert torch.equal(old.logits(query,'text'),new.logits(query,'text'))
    assert torch.equal(old.logits(query,'image'),new.logits(query,'image'))
    prompt=torch.randn(3,8)
    assert torch.equal(old.logits(prompt,'prompt'),new.logits(prompt,'prompt'))


def test_native_overlap_accounts_for_missing_support_and_ignores_unobserved():
    from radio_gs.models.grounded_semantic_transfer import GroundedSemanticTransfer
    head=GroundedSemanticTransfer(3,2,4,2,num_proposals=2,
        identity_selection={'metric':'support_iou','require_native_support':False})
    latent=torch.tensor([[1.,0.,1.],[1.,0.,0.],[0.,1.,0.],[0.,1.,1.]])
    with torch.no_grad():
        head.decoder.weight.copy_(torch.tensor([[0.,0.,1.],[1.,0.,0.]]));head.decoder.bias.zero_()
    support=torch.tensor([[True,False],[True,False],[False,False],[False,False]])
    observed=torch.tensor([True,True,True,False])
    torch.testing.assert_close(head.support_overlap(latent,support,observed),torch.tensor([[.5,1.],[0.,0.]]))
    query=torch.eye(4)[:2]
    logits,accepted=head(query,latent,native_support=support,observed=observed)
    assert accepted.tolist()==[True,False]
    assert logits[:,0].sigmoid().gt(.5).tolist()==[True,True,False,False]
    sub=head(query,latent,torch.tensor([1,3]),native_support=support,observed=observed)[0]
    torch.testing.assert_close(sub,logits[[1,3]])


def test_empty_native_identity_cannot_pass_negative_threshold():
    from radio_gs.models.grounded_semantic_transfer import GroundedSemanticTransfer
    head=GroundedSemanticTransfer(10,6,4,2,num_proposals=2,identity_selection={})
    _,accepted=head(torch.eye(4)[:2],torch.randn(5,10),native_threshold=-.2)
    assert not accepted.any()


def test_overlap_fallback_preserves_source_accepted_concept_extent():
    from radio_gs.models.grounded_semantic_transfer import GroundedSemanticTransfer
    head=GroundedSemanticTransfer(3,2,4,2,num_proposals=3,
        identity_selection={'metric':'support_iou','trusted_concept_start':1})
    with torch.no_grad():
        head.prototypes.copy_(torch.eye(4)[:2]);head.acceptance.fill_(.9)
        head.decoder.weight.zero_();head.decoder.bias.copy_(torch.tensor([.8,.2,.3]))
    latent=torch.randn(5,3);support=torch.ones(5,2,dtype=torch.bool);observed=torch.ones(5,dtype=torch.bool)
    # The native overlap winner is candidate 0, but the supported concept uses 1.
    logits,accepted=head(torch.eye(4)[[0,2]],latent,native_support=support,observed=observed)
    assert accepted.all()
    torch.testing.assert_close(logits.sigmoid(),torch.tensor([[.2,.8]]).expand(5,-1))
