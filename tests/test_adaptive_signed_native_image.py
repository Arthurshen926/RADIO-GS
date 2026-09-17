import torch
import pytest
from radio_gs.querying.adaptive_signed_native_image import condition_signed_native_image_adaptive

class Field(torch.nn.Module):
 def __init__(self):
  super().__init__();self.latent=torch.nn.Parameter(torch.eye(3));self.native_decoder=torch.nn.Linear(3,3);self.native_semantic_dim=None
 def logits(self,q,m):return torch.tensor([[-30.],[30.],[0.]])
 def observed_for(self,m):return torch.ones(3,dtype=torch.bool)

def test_confident_wrong_native_signs_are_feasible_and_trainable_without_map_updates():
 f=Field();before={k:v.clone() for k,v in f.state_dict().items()}
 def render(s):return s[:,0],torch.tensor([True,True,False])
 r=condition_signed_native_image_adaptive(f,torch.ones(1,3),render,torch.tensor([True,False,True]),torch.tensor([False,True,False]),steps=600)
 assert r['report']['maximum_logit_correction']==32
 assert r['report']['all_supported_signs_satisfied']
 assert not r['report']['all_input_signs_satisfied']
 assert r['report']['unsupported_positive_pixels']==1
 assert all(torch.equal(v,f.state_dict()[k]) for k,v in before.items())
 assert all(p.grad is None for p in f.parameters())

def test_mixed_pixel_constraints_do_not_claim_joint_feasibility():
 f=Field()
 def render(s):return s[:2].mean().repeat(2),torch.ones(2,dtype=torch.bool)
 r=condition_signed_native_image_adaptive(f,torch.ones(1,3),render,torch.tensor([True,False]),torch.tensor([False,True]),steps=3)
 assert not r['report']['all_supported_signs_satisfied']
 assert r['report']['final_balanced_error']==.5

def test_bad_bounds_rejected():
 with pytest.raises(ValueError,match='bounded signed'):
  condition_signed_native_image_adaptive(Field(),torch.ones(1,3),None,torch.tensor([True,False]),torch.tensor([False,True]),bounds=(64,8))
