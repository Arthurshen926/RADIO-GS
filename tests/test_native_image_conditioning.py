import torch
import pytest
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.querying.native_image_conditioning import condition_native_image_query
from radio_gs.models.signed_query_refinement import SignedEvidenceUnavailable

@pytest.mark.parametrize('native',[False,True])
def test_conditioned_query_uses_identical_deployed_image_readout_without_field_updates(native):
 f=ElementRetrievalField(2,2,{'text':3,'image':3},shared_query_groups=[['text','image']],native_siglip_readout=native)
 with torch.no_grad():
  f.latent.copy_(torch.eye(2));f.observed[:]=True
  if native:f.native_decoder.weight.copy_(torch.tensor([[1.,0.],[0.,1.],[0.,0.]]));f.native_decoder.bias.zero_()
  else:f.query_adapters['image'].weight.copy_(torch.tensor([[1.,0.,0.],[0.,1.,0.]]))
 before={k:v.clone() for k,v in f.state_dict().items()};r=condition_native_image_query(f,torch.tensor([[0.,1.,0.]]),[[0],[1]],[[1.],[1.]],[1.,0.],steps=64)
 actual=f.retrieve(r['native_query'],'image')['scores'][:,0]
 assert torch.allclose(actual,r['seed_after'],atol=1e-6) and r['constraints_satisfied'].all()
 assert all(p.grad is None for p in f.parameters()) and all(torch.equal(before[k],v) for k,v in f.state_dict().items())
 r2=condition_native_image_query(f,r['native_query'],[[0],[1]],[[1.],[1.]],[1.,0.]);assert r2['steps']==0

def test_unknown_registered_support_is_not_invented():
 f=ElementRetrievalField(2,2,{'image':3})
 with pytest.raises(SignedEvidenceUnavailable):condition_native_image_query(f,torch.ones(1,3),[[0],[1]],[[1.],[1.]],[1.,0.])

@pytest.mark.parametrize('native',[False,True])
def test_image_conditioning_rejects_minority_observed_support(native):
 f=ElementRetrievalField(3,2,{'text':3,'image':3},shared_query_groups=[['text','image']],native_siglip_readout=native)
 f.observed[:2]=True
 with pytest.raises(SignedEvidenceUnavailable):
  condition_native_image_query(f,torch.ones(1,3),[[0,2],[1]],[[.1,.9],[1.]],[1.,0.])
