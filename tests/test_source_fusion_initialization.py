import torch
from torch.nn import functional as F
from radio_gs.training.source_fusion_initialization import fit_source_fusion,apply_source_fusion
from radio_gs.models.element_retrieval_field import ElementRetrievalField


def test_source_fusion_projects_after_native_mixture_and_preserves_unknowns():
 torch.manual_seed(8);t=torch.randn(5,8);weights=torch.rand(5,4);weights[:,3]=0;b=fit_source_fusion(t,weights,torch.ones(4,dtype=torch.bool),3)
 expected=weights.T@F.normalize(t,dim=-1)@b['basis'].T/weights.sum(0)[:,None].clamp_min(1e-8)
 assert torch.allclose(b['latent'],expected,atol=1e-6) and b['valid'].tolist()==[True,True,True,False]
 f=ElementRetrievalField(4,3,{'text':8,'image':8},shared_query_groups=[['text','image']],native_siglip_readout=True);f.observed[:]=True;unknown=f.latent[3].detach().clone();apply_source_fusion(f,b,freeze_decoder=True)
 assert torch.equal(f.latent[3],unknown) and not f.native_decoder.weight.requires_grad
 q=F.normalize(torch.randn(2,8),dim=-1);actual=f.logits(q,'text',rows=torch.arange(3));native=F.normalize((weights.T@F.normalize(t,dim=-1))@b['basis'].T@b['basis'],dim=-1)[:3]@q.T
 assert torch.allclose(actual,10*native-5,atol=1e-5)


def test_initialization_rejects_nonorthogonal_basis_and_unobserved_writes():
 import pytest
 b=fit_source_fusion(torch.eye(3),torch.eye(3),torch.ones(3,dtype=torch.bool),2)
 f=ElementRetrievalField(3,2,{'text':3,'image':3},shared_query_groups=[['text','image']],native_siglip_readout=True)
 with pytest.raises(ValueError,match='unobserved'):apply_source_fusion(f,b)
 f.observed[:]=True
 with pytest.raises(ValueError,match='orthonormal'):apply_source_fusion(f,{**b,'basis':b['basis']*2})


def test_trainable_source_decoder_and_latent_remain_connected_to_retrieval():
 b=fit_source_fusion(torch.eye(3),torch.eye(3),torch.ones(3,dtype=torch.bool),2)
 f=ElementRetrievalField(3,2,{'text':3,'image':3},shared_query_groups=[['text','image']],native_siglip_readout=True);f.observed[:]=True;apply_source_fusion(f,b)
 f.logits(torch.tensor([[1.,.5,.2]]),'text').sum().backward()
 assert f.native_decoder.weight.grad.norm()>0 and f.latent.grad.norm()>0


def test_frozen_teacher_reconstruction_has_element_gradients_without_head_gradients():
 b=fit_source_fusion(torch.eye(3),torch.eye(3),torch.ones(3,dtype=torch.bool),2)
 f=ElementRetrievalField(3,2,{'text':3,'image':3},shared_query_groups=[['text','image']],native_siglip_readout=True);f.observed[:]=True;apply_source_fusion(f,b,freeze_decoder=True)
 loss=1-F.cosine_similarity(f.native_decoder(f.latent[:2]).mean(0),torch.tensor([1.,.2,.3]),dim=0)
 gradient=torch.autograd.grad(loss,f.latent)[0]
 assert gradient.norm()>0 and f.native_decoder.weight.grad is None


def test_bounded_updates_preserve_unsupported_rows_and_native_displacement_bound():
 from radio_gs.training.source_fusion_initialization import project_source_update
 initial=torch.tensor([[1.,.2],[.1,1.]]);latent=torch.nn.Parameter(torch.tensor([[3.,4.],[5.,6.],[7.,8.]]));last=latent[-1].clone();rows=torch.tensor([0,1]);project_source_update(latent,rows,initial,.02)
 assert (((latent[:2]-initial).norm(dim=-1)/initial.norm(dim=-1))<=.020001).all()
 assert torch.equal(last,latent[-1])
 q=F.normalize(torch.randn(100,2),dim=-1)
 change=(F.normalize(latent[:2],dim=-1)-F.normalize(initial,dim=-1))@q.T
 assert change.abs().max()<=2*.02/(1-.02)
