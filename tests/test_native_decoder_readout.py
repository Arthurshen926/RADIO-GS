import torch
import pytest
from torch.nn import functional as F
from radio_gs.models.native_decoder_readout import affine_decoder_cosine


def test_affine_native_cosine_and_gradients_match_explicit_decoding():
 torch.manual_seed(0);items=[torch.randn(11,3,dtype=torch.double,requires_grad=True),torch.randn(5,7,dtype=torch.double,requires_grad=True),torch.randn(7,3,dtype=torch.double,requires_grad=True),torch.randn(7,dtype=torch.double,requires_grad=True)]
 z,q,w,b=items;actual=affine_decoder_cosine(*items);expected=F.normalize(F.linear(z,w,b),dim=-1)@F.normalize(q,dim=-1).T
 assert torch.allclose(actual,expected,atol=1e-10,rtol=1e-10)
 g=torch.randn_like(actual);a=torch.autograd.grad((actual*g).sum(),items,retain_graph=True);e=torch.autograd.grad((expected*g).sum(),items)
 assert all(torch.allclose(x,y,atol=1e-9,rtol=1e-9) for x,y in zip(a,e))


def test_tied_transpose_with_latent_cosine_is_not_native_cosine():
 z=torch.tensor([[1.,1.]]);q=torch.tensor([[1.,0.,0.]]);w=torch.tensor([[10.,0.],[0.,1.],[0.,0.]]);b=torch.tensor([0.,0.,3.])
 actual=affine_decoder_cosine(z,q,w,b);naive=F.normalize(z,dim=-1)@F.normalize(q@w,dim=-1).T
 assert not torch.allclose(actual,naive,atol=.1)


def test_near_cancellation_and_zero_decoding_are_stable():
 w=torch.eye(3);z=torch.tensor([[1.,2.,3.],[1.,2.,3.+1e-6]]);b=torch.tensor([-1.,-2.,-3.]);q=torch.eye(3)
 actual=affine_decoder_cosine(z,q,w,b);expected=F.normalize(F.linear(z,w,b),dim=-1)@q.T
 assert torch.isfinite(actual).all() and torch.allclose(actual,expected,atol=1e-6)


def test_bad_axes_rejected():
 with pytest.raises(ValueError,match='axes'):
  affine_decoder_cosine(torch.ones(2,3),torch.ones(2,4),torch.ones(4,3),torch.ones(3))

def test_field_native_readout_round_trip_and_legacy_prompt_are_consistent():
 from radio_gs.models.element_retrieval_field import ElementRetrievalField
 from radio_gs.models.signed_query_refinement import retrieve_latent_query
 args=dict(num_elements=5,latent_dim=3,query_dims={'text':7,'image':7,'prompt':4},shared_query_groups=[['text','image']],native_siglip_readout=True)
 f=ElementRetrievalField(**args);f.observed[:]=True;q=torch.randn(2,7)
 expected=F.normalize(f.native_decoder(f.latent),dim=-1)@F.normalize(q,dim=-1).T
 assert torch.allclose(f.logits(q,'text'),10*expected-5,atol=1e-5)
 copy=ElementRetrievalField(**args);copy.load_state_dict(f.state_dict());assert torch.equal(copy.retrieve(q,'image')['scores'],f.retrieve(q,'image')['scores'])
 assert not f.query_adapters['text'].weight.requires_grad and f.query_adapters['prompt'].weight.requires_grad
 with pytest.raises(ValueError,match='original native query'):retrieve_latent_query(f,torch.ones(1,3),modality='image')
