import torch
from radio_gs.models.grounded_semantic_transfer import rank_preserving_extent_threshold
from scripts.audits.fit_completion_admission import choose_admission_threshold


def test_native_threshold_matches_exhaustive_sets_and_never_splits_ties():
    score=torch.tensor([[.9,.1],[.8,.5],[.8,.5],[.2,.9],[.1,1.]])
    target=torch.tensor([[0,1],[1,1],[0,0],[1,0],[1,1]],dtype=torch.bool)
    observed=torch.tensor([True,True,True,True,False])
    cutoff=rank_preserving_extent_threshold(score,target,observed)
    for j in range(2):
        pred=(score[:,j]>cutoff[j])&observed;t=target[:,j]&observed
        actual=(pred&t).sum()/(pred|t).sum()
        best=0.
        for c in torch.cat([score[observed,j],torch.tensor([-1.])]):
            p=(score[:,j]>c)&observed;best=max(best,float((p&t).sum()/(p|t).sum().clamp_min(1)))
        torch.testing.assert_close(actual,torch.tensor(best))
    assert bool(score[1,0]>cutoff[0])==bool(score[2,0]>cutoff[0])


def test_all_positive_extent_can_select_all_observed_elements():
    score=torch.tensor([[.8],[.5],[.1]])
    cutoff=rank_preserving_extent_threshold(score,torch.ones_like(score,dtype=torch.bool),torch.ones(3,dtype=torch.bool))
    assert (score>cutoff).all()


def test_accepted_concept_can_calibrate_cutoff_without_replacing_native_ranks():
    from radio_gs.models.element_retrieval_field import ElementRetrievalField
    torch.manual_seed(42)
    kwargs=dict(num_elements=12,latent_dim=10,query_dims={'text':16,'image':16},
        shared_query_groups=[['text','image']],native_siglip_readout=True,native_semantic_dim=6,
        calibrated_readout=True,native_query_thresholds={'text':{'lower':-1.,'upper':1.}},
        grounded_transfer={'num_groups':1,'num_proposals':2,'identity_selection':{
            'metric':'support_iou','trusted_concept_start':1,'completion_mode':'native_threshold',
            'calibrate_accepted_concepts':True}})
    field=ElementRetrievalField(**kwargs);field.observed[:]=True
    with torch.no_grad():
        field.grounded_transfer.acceptance.fill_(-2)
        field.grounded_transfer.decoder.weight.zero_();field.grounded_transfer.decoder.bias.fill_(.8)
    q=torch.eye(16)[:2]
    assert field.grounded_transfer.route(q)[1].all()
    actual=field.logits(q,'text');native=field.native_cosine(q)
    torch.testing.assert_close(actual-actual[:1],field.readout_log_scale['text'].exp()*(native-native[:1]))
    rows=torch.tensor([1,4,7]);torch.testing.assert_close(field.logits(q,'text',rows),actual[rows])
    torch.testing.assert_close(field.logits(q[:1],'text'),actual[:,:1])
    cold=ElementRetrievalField(**kwargs);cold.load_state_dict(field.state_dict())
    assert torch.equal(cold.logits(q,'text'),actual)
    legacy_kwargs=dict(kwargs,grounded_transfer=dict(kwargs['grounded_transfer'],identity_selection=dict(
        kwargs['grounded_transfer']['identity_selection'],calibrate_accepted_concepts=False)))
    legacy=ElementRetrievalField(**legacy_kwargs);legacy.load_state_dict(field.state_dict())
    assert not torch.allclose(legacy.logits(q,'text'),actual)
    assert torch.equal(legacy.logits(q,'image'),field.logits(q,'image'))


def test_admission_is_fit_from_benefit_and_ties_preserve_native():
    t,_=choose_admission_threshold(torch.tensor([.1,.8]),torch.tensor([.9,.2]),torch.tensor([.1,.8]))
    assert abs(t-.1)<1e-6
    t,_=choose_admission_threshold(torch.tensor([.1,.8]),torch.tensor([.2,.3]),torch.tensor([.2,.3]))
    assert t==1.


def test_field_completion_preserves_native_order_rows_and_cold_state():
    from radio_gs.models.element_retrieval_field import ElementRetrievalField
    kwargs=dict(num_elements=12,latent_dim=10,query_dims={'text':16,'image':16,'prompt':8},
        shared_query_groups=[['text','image']],native_siglip_readout=True,native_semantic_dim=6,
        calibrated_readout=True,native_query_thresholds={'text':{'lower':-1.,'upper':1.}},
        grounded_transfer={'num_groups':1,'num_proposals':2,'identity_selection':{
            'metric':'support_iou','trusted_concept_start':1,'completion_mode':'native_threshold'}})
    field=ElementRetrievalField(**kwargs);field.observed[:]=True
    with torch.no_grad():
        field.native_query_thresholds['text'].linear.weight.zero_();field.native_query_thresholds['text'].linear.bias.fill_(-1)
        field.grounded_transfer.acceptance.fill_(2)
        field.grounded_transfer.decoder.weight.zero_();field.grounded_transfer.decoder.bias.fill_(.8)
    q=torch.eye(16)[:2];logits=field.logits(q,'text');cosine=field.native_cosine(q)
    scale=field.readout_log_scale['text'].exp()
    torch.testing.assert_close(logits-logits[:1],scale*(cosine-cosine[:1]))
    rows=torch.tensor([1,4,7]);torch.testing.assert_close(field.logits(q,'text',rows),logits[rows])
    torch.testing.assert_close(field.logits(q[:1],'text'),logits[:,:1])
    cold=ElementRetrievalField(**kwargs);cold.load_state_dict(field.state_dict())
    assert torch.equal(cold.logits(q,'text'),logits)


def test_within_extent_keeps_element_membership_and_native_order_in_each_stratum():
    from radio_gs.models.element_retrieval_field import ElementRetrievalField
    kwargs=dict(num_elements=8,latent_dim=6,query_dims={'text':4,'image':4},shared_query_groups=[['text','image']],
        native_siglip_readout=True,native_semantic_dim=4,calibrated_readout=True,
        native_query_thresholds={'text':{'lower':-1.,'upper':1.}},
        grounded_transfer={'num_groups':1,'num_proposals':2,'identity_selection':{
            'metric':'support_iou','trusted_concept_start':1,'completion_mode':'native_within_extent'}})
    field=ElementRetrievalField(**kwargs);field.observed[:]=True
    with torch.no_grad():
        field.native_query_thresholds['text'].linear.weight.zero_();field.native_query_thresholds['text'].linear.bias.fill_(-1)
        field.grounded_transfer.acceptance.fill_(2)
        field.grounded_transfer.decoder.weight.zero_();field.grounded_transfer.decoder.weight[:,4]=1
        field.grounded_transfer.decoder.bias.zero_()
        field.latent[:4,4:]=torch.tensor([1.,0.]);field.latent[4:,4:]=torch.tensor([-1.,0.])
    q=torch.eye(4)[:1];scores=field.logits(q,'text').sigmoid();native=field.native_cosine(q)
    assert (scores[:,0]>.5).tolist()==[True]*4+[False]*4
    for section in [slice(0,4),slice(4,8)]:
        expected=.5*(torch.sigmoid(10*(native[section]+1))+(1 if section.start==0 else 0))
        torch.testing.assert_close(scores[section],expected)
    rows=torch.tensor([1,6]);torch.testing.assert_close(field.logits(q,'text',rows).sigmoid(),scores[rows])
