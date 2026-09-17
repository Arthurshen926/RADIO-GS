import torch
import pytest
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.models.signed_query_refinement import refine_signed_query,retrieve_latent_query


def make_field():
    field=ElementRetrievalField(3,3,{'prompt':3});field.observed.fill_(True)
    with torch.no_grad():
        field.latent.copy_(torch.eye(3));field.query_adapters['prompt'].weight.copy_(torch.eye(3))
    return field


def test_signed_refinement_changes_query_only_and_satisfies_seeds():
    field=make_field();before={k:v.clone() for k,v in field.state_dict().items()}
    q=torch.tensor([[.2,1.,.1]])
    result=refine_signed_query(field,q,[[0],[1]],[[1.],[1.]],[1.,0.])
    assert result['constraints_satisfied'].all()
    assert result['seed_after'][0]>result['seed_before'][0]
    assert result['seed_after'][1]<result['seed_before'][1]
    assert all(torch.equal(before[k],v) for k,v in field.state_dict().items())
    assert all(v.grad is None for v in field.parameters())
    posterior=retrieve_latent_query(field,result['latent_query'])['scores'][:,0]
    assert torch.allclose(posterior[:2],result['seed_after'])


def test_inconsistent_seed_mixtures_are_reported_not_overridden():
    field=make_field()
    result=refine_signed_query(field,torch.tensor([[1.,0.,0.]]),[[0],[0]],[[1.],[1.]],[1.,0.])
    assert not result['constraints_satisfied'].all()
    assert result['seed_after'][0]==result['seed_after'][1]


def test_unobserved_seed_is_rejected():
    field=make_field();field.observed[0]=False
    with pytest.raises(ValueError,match='lacks observed'):
        refine_signed_query(field,torch.ones(1,3),[[0],[1]],[[1.],[1.]],[1.,0.])


def test_zero_steps_preserves_original_posterior():
    field=make_field();q=torch.tensor([[1.,.2,.1]])
    result=refine_signed_query(field,q,[[0],[1]],[[1.],[1.]],[1.,0.],steps=0)
    assert torch.allclose(retrieve_latent_query(field,result['latent_query'])['scores'],field.retrieve(q,'prompt')['scores'])


def test_already_correct_signed_evidence_preserves_query_without_confidence_chasing():
    field=make_field();q=torch.tensor([[.8,0.,.6]])
    result=refine_signed_query(field,q,[[0],[1]],[[1.],[1.]],[1.,0.])
    assert result['constraints_satisfied'].all()
    assert not result['refinement_applied']
    assert result['steps']==0
    assert torch.equal(result['seed_before'],result['seed_after'])
    assert torch.allclose(retrieve_latent_query(field,result['latent_query'])['scores'],field.retrieve(q,'prompt')['scores'])


def test_unavailable_seed_is_typed_and_does_not_write_the_map():
    from radio_gs.models.signed_query_refinement import SignedEvidenceUnavailable
    field=make_field();field.observed[1]=False
    before=field.latent.clone()
    with pytest.raises(SignedEvidenceUnavailable) as exc:
        refine_signed_query(field,torch.ones(1,3),[[0],[1]],[[1.],[1.]],[1.,0.])
    assert exc.value.seed_index==1
    assert torch.equal(field.latent,before)
    assert all(p.grad is None for p in field.parameters())


def test_unavailable_query_is_preserved_and_registration_errors_still_fail():
    from radio_gs.models.signed_query_refinement import condition_signed_query
    field=make_field();field.observed[1]=False;q=torch.tensor([[1.,.2,.1]])
    result=condition_signed_query(field,q,[[0],[1]],[[1.],[1.]],[1.,0.])
    assert result['available'] is False
    assert result['constraints_satisfied'] is None
    assert torch.allclose(retrieve_latent_query(field,result['latent_query'])['scores'],field.retrieve(q,'prompt')['scores'])
    with pytest.raises(ValueError,match='outside field'):
        condition_signed_query(field,q,[[99],[0]],[[1.],[1.]],[1.,0.])


def test_stop_at_feasibility_moves_less_than_confidence_fitting():
    field=make_field();q=torch.tensor([[.2,1.,.1]])
    confidence=refine_signed_query(field,q,[[0],[1]],[[1.],[1.]],[1.,0.])
    minimal=refine_signed_query(field,q,[[0],[1]],[[1.],[1.]],[1.,0.],stop_when_satisfied=True)
    assert minimal['constraints_satisfied'].all()
    assert minimal['steps']<confidence['steps']
    assert minimal['cosine_to_initial']>confidence['cosine_to_initial']
    assert min(abs(float(p)-.5) for p in minimal['seed_after']) < .002
    assert all(p.grad is None for p in field.parameters())


def test_feasibility_stopping_does_not_claim_inconsistent_seeds_solved():
    result=refine_signed_query(make_field(),torch.tensor([[1.,0.,0.]]),[[0],[0]],[[1.],[1.]],[1.,0.],stop_when_satisfied=True)
    assert not result['constraints_satisfied'].all()
    assert result['steps']==64

@pytest.mark.parametrize('mass',[.1,.5])
def test_minority_or_tied_observed_mass_is_unavailable_even_with_valid_rows(mass):
    from radio_gs.models.signed_query_refinement import condition_signed_query
    field=make_field();field.observed[2]=False
    q=torch.tensor([[1.,.2,.1]])
    result=condition_signed_query(field,q,[[0,2],[1]],[[mass,1-mass],[1.]],[1.,0.])
    assert result['available'] is False and result['missing_seed_index']==0
    assert torch.allclose(retrieve_latent_query(field,result['latent_query'])['scores'],field.retrieve(q,'prompt')['scores'])


def test_majority_observed_mixture_matches_rendering_and_tiny_alpha_is_unknown():
    from radio_gs.models.signed_query_refinement import observed_seed_contributions, SignedEvidenceUnavailable
    field=make_field();field.observed[2]=False
    rows,w=observed_seed_contributions(field,[0,1,2],[.4,.2,.4],0,device='cpu',dtype=torch.float32)
    assert rows.tolist()==[0,1] and torch.allclose(w,torch.tensor([2/3,1/3]))
    with pytest.raises(SignedEvidenceUnavailable):
        observed_seed_contributions(field,[0],[1e-9],0,device='cpu',dtype=torch.float32)
    with pytest.raises(ValueError,match='integer'):
        observed_seed_contributions(field,[.5],[1.],0,device='cpu',dtype=torch.float32)
