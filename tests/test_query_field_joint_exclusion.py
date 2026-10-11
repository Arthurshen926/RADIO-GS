import numpy as np
import torch
import pytest
from scipy import sparse
from types import SimpleNamespace
from radio_gs.query_field.joint_exclusion import _prune,_initial_weights,CUTOFF


def prune(matrix,positive,negative,score):
    matrix=sparse.csr_matrix(matrix);columns=matrix.tocsc();s=np.asarray(score);selected=s>.5
    return _prune(columns.indptr,columns.indices,columns.data,np.argsort(s,kind='stable'),selected,np.array(positive),np.array(negative),np.asarray(matrix@selected.astype(float)).ravel(),.5,.5001)[0]


def test_joint_redundancy_never_deletes_both_alternatives():
    # G0,G1 jointly explain P, but only G0 also leaks into N.
    assert prune([[1.,1.],[1.,0.]],[True,False],[False,True],[.6,.9]).tolist()==[False,True]
    # If BOTH leak into N, sequential protection retains one, not zero.
    assert prune([[1.,1.],[1.,1.]],[True,False],[False,True],[.6,.9]).tolist()==[False,True]


def test_complementary_positive_members_and_unknowns_are_retained():
    assert prune([[1.,0.],[0.,1.],[1.,1.]],[True,True,False],[False,False,True],[.6,.9]).all()
    assert prune([[1.,1.]],[False],[False],[.6,.9]).all()


def test_empty_and_already_satisfied_negatives_are_noop():
    assert not prune([[1.,1.]],[False],[True],[.1,.2]).any()
    assert prune([[.1,.1]],[False],[True],[.6,.9]).all()


def test_frozen_initial_visibility_underestimates_after_front_removal():
    ptr=np.array([0,2]);ids=np.array([0,1]);alpha=np.array([.8,.8]);selected=np.array([True,True])
    weights=_initial_weights(ptr,ids,alpha,selected)
    np.testing.assert_allclose(weights,[.8,.16])
    assert weights[1]<.8


def test_field_decoder_labels_and_public_api(monkeypatch):
    from radio_gs.query_field.coverage_factors import CoverageView
    from radio_gs.query_field.joint_exclusion import JointExclusionDecoder,query_text_joint_exclusion
    from radio_gs.query_field.scope_readout import ScopeReadout
    from radio_gs.query_field import scope_confidence
    state=dict(pointer=torch.tensor([0,2,3]),index=torch.tensor([0,1,0]),alpha=torch.tensor([.8,.8,.8]),atom=torch.tensor([0,1]))
    factors=CoverageView(state,2,2)
    constraints=SimpleNamespace(regions=torch.tensor([0]),candidates=torch.tensor([0]),
        positive=torch.tensor([[True],[False]]),negative=torch.tensor([[False],[True]]),interior=torch.zeros(2,1,dtype=torch.bool))
    class Permissions:
        accepted_views=torch.tensor([0])
        def __call__(self,a):return torch.tensor([False])
    field=SimpleNamespace(xyz=torch.zeros(2,3),scope_permissions=Permissions(),scope_constraints=SimpleNamespace(views=[constraints]),coverage_factors=SimpleNamespace(views=[factors]))
    decoder=JointExclusionDecoder(field);initial=torch.tensor([.6,.9]);accepted=torch.tensor([True])
    result,detail=decoder.decode(initial,accepted)
    assert result.tolist()==[0.,initial[1].item()] and detail['positive_coverage_losses']==0
    assert torch.equal(initial,torch.tensor([.6,.9]))
    assert torch.equal(decoder.decode(initial,torch.tensor([False]))[0],initial)
    def native(*args):return ScopeReadout(torch.tensor([.3,.7]),initial.clone(),initial>.5,accepted,{'native_view_confidence_preserved':True})
    monkeypatch.setattr(scope_confidence,'query_text_scope_confident',native)
    output=query_text_joint_exclusion(field,torch.ones(1),torch.ones(1,1),decoder=decoder)
    assert torch.equal(output.membership_score,result) and output.selected.tolist()==[False,True]
    assert not output.diagnostics['native_view_confidence_preserved']
    assert output.diagnostics['initializer_view_confidence_preserved']
    assert not output.diagnostics['coverage_view_confidence_preserved']
    assert not detail['view_confidence_weighted_constraints']
    semantic=query_text_joint_exclusion(field,torch.ones(1),torch.ones(1,1),decoder=decoder,initializer='semantic')
    assert torch.equal(semantic.membership_score,torch.tensor([.3,.7]))
    assert not semantic.diagnostics['native_view_confidence_preserved']
    assert not semantic.diagnostics['initializer_view_confidence_preserved']
    assert not semantic.diagnostics['coverage_view_confidence_preserved']
    with pytest.raises(ValueError):query_text_joint_exclusion(field,torch.ones(1),torch.ones(1,1),decoder=decoder,initializer='invalid')
    with pytest.raises(ValueError):decoder.decode(torch.ones(2,dtype=torch.bool),accepted)

    from radio_gs.query_field.member_energy import query_text_member_energy
    from radio_gs.query_field.support_footprint import SupportFootprint
    from radio_gs.query_field import part_conditioning
    field.support_footprint=SupportFootprint({'cost':torch.tensor([[1.,1.]])},2)
    field.evidence=object()
    monkeypatch.setattr(scope_confidence,'native_text_evidence',lambda *args:(torch.tensor([.3,.7]),torch.tensor([.9])))
    monkeypatch.setattr(scope_confidence,'native_scope_confidence',lambda *args:(accepted,torch.tensor([.8])))
    import builtins
    def denied(*args,**kwargs):raise AssertionError('query attempted file access')
    with monkeypatch.context() as guard:
        guard.setattr(builtins,'open',denied)
        def finite_forbidden(*args,**kwargs):raise AssertionError('unused finite readout requested')
        guard.setattr(scope_confidence,'query_text_scope_confident',finite_forbidden)
        energy=query_text_member_energy(field,torch.ones(1),torch.ones(1,1),decoder=decoder)
    assert energy.selected.tolist()==[False,True]
    assert energy.diagnostics['coverage_view_confidence_preserved']
    assert not energy.diagnostics['member_energy']['query_parameter_fitting']
    field.support_footprint=None
    with pytest.raises(ValueError,match='footprint'):
        query_text_member_energy(field,torch.ones(1),torch.ones(1,1),decoder=decoder)
