from copy import deepcopy
import pytest
import torch
from radio_gs.querying.source_refitted_category import source_refitted_category_logits


def receipts(accepted=True):
    scope=dict(schema='bounded_fallback_scope.v1',source_only=True,benchmark_data_used=False,
        field_sha256='a'*64,source_acceptance=accepted,score_order=['retained',1.],selected_strength=1.,validation_iou=[.5,.6 if accepted else .4])
    parent=dict(schema='counterfactual_extent_benefit_admission.v1',source_only=True,benchmark_data_used=False,
        field_sha256='b'*64,parent_field_sha256='a'*64,source_routed_validation_passed=True,
        selected=dict(validation_routed_iou=.6,validation_routed_parent_iou=.5))
    refit=dict(schema='registered_visibility_qualified_selected_scope.v1',source_only=True,benchmark_data_used=False,
        field_sha256='c'*64,parent_field_sha256='b'*64,geometry_unchanged=True,native_field_unchanged=True,
        query_encoder_unchanged=True,steps=400,selected_step=100,mode='visible_selected',
        history=[dict(step=0,validation=dict(full_iou=.5,selected_iou=.4,observations=10)),dict(step=100,validation=dict(full_iou=.5,selected_iou=.45,observations=10))])
    return scope,parent,refit


def test_refit_cannot_reenable_parent_rejected_scope():
    scope,parent,refit=receipts(False);native=torch.ones(1,2)
    result,decision=source_refitted_category_logits(native,lambda:1/0,scope_receipt=scope,
        parent_admission=parent,refit_receipt=refit,field_sha256='c'*64)
    assert result is native and not decision['use_scope'] and decision['parent_scope_policy_retained']


def test_accepted_refit_keeps_original_correction_bound():
    scope,parent,refit=receipts();native=torch.tensor([[5.,0.]])
    result,_=source_refitted_category_logits(native,lambda:torch.tensor([[-100.,100.]]),scope_receipt=scope,
        parent_admission=parent,refit_receipt=refit,field_sha256='c'*64)
    assert result.argmax()==native.argmax() and (result-native).abs().max()<=1


@pytest.mark.parametrize('bad',['parent','field','native','source','full','selected','step','cohort','mode'])
def test_invalid_refit_cannot_inherit_admission(bad):
    scope,parent,refit=deepcopy(receipts())
    if bad=='parent':refit['parent_field_sha256']='d'*64
    if bad=='field':refit['field_sha256']='d'*64
    if bad=='native':refit['native_field_unchanged']=False
    if bad=='source':refit['benchmark_data_used']=True
    if bad=='full':refit['history'][1]['validation']['full_iou']=.49
    if bad=='selected':refit['history'][1]['validation']['selected_iou']=.39
    if bad=='step':refit['selected_step']=200
    if bad=='cohort':refit['history'][1]['validation']['observations']=9
    if bad=='mode':refit['mode']='target_winner'
    with pytest.raises(ValueError):
        source_refitted_category_logits(torch.ones(1,2),lambda:torch.zeros(1,2),scope_receipt=scope,
            parent_admission=parent,refit_receipt=refit,field_sha256='c'*64)
