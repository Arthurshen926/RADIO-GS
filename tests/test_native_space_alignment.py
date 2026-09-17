import torch
from torch.nn import functional as F
from radio_gs.training.native_space_alignment import native_region_alignment

def test_same_direction_means_same_query_scores():
 projection=torch.tensor([[1.,0.,0.],[0.,1.,0.]])
 teacher=torch.tensor([[2.,1.,9.]])
 mean=F.normalize(teacher@projection.T,dim=-1).requires_grad_()
 assert native_region_alignment(mean,teacher,projection).item()==0
 queries=F.normalize(torch.tensor([[1.,2.],[3.,1.]]),dim=-1)
 assert torch.allclose(queries@mean.T,queries@F.normalize(teacher@projection.T,dim=-1).T)

def test_targets_cannot_absorb_alignment_error():
 mean=torch.tensor([[0.,1.]],requires_grad=True)
 teacher=torch.tensor([[1.,0.,0.]],requires_grad=True)
 projection=torch.tensor([[1.,0.,0.],[0.,1.,0.]],requires_grad=True)
 loss=native_region_alignment(mean,teacher,projection);loss.backward()
 assert mean.grad.abs().sum()>0
 assert teacher.grad is None and projection.grad is None

def test_mean_magnitude_is_not_renormalized_away():
 projection=torch.eye(2);teacher=torch.tensor([[1.,0.]])
 assert native_region_alignment(torch.tensor([[.2,0.]]),teacher,projection)>.6


def test_overlapping_independent_unit_region_targets_cannot_both_be_satisfied():
 # Two regions share the same single element but demand orthogonal directions.
 target=torch.eye(2);z=F.normalize(torch.ones(1,2),dim=-1).expand(2,-1)
 assert native_region_alignment(z,target,torch.eye(2))>.5


def test_one_element_teacher_is_jointly_realizable_under_different_mixtures():
 from radio_gs.training.native_space_alignment import native_element_alignment
 teacher=F.normalize(torch.tensor([[1.,1.],[0.,1.],[1.,0.]]),dim=-1).requires_grad_()
 z=teacher.detach().clone().requires_grad_()
 mixtures=torch.tensor([[.2,.3,.5],[.8,.2,0.]])
 assert native_element_alignment(z,teacher)<1e-12
 assert torch.allclose(mixtures@F.normalize(z,dim=-1),mixtures@F.normalize(teacher,dim=-1))
 native_element_alignment(z,teacher).backward();assert teacher.grad is None


def test_native_cone_preserves_feasible_values_norms_and_handles_antipodes():
 from radio_gs.training.native_space_alignment import project_native_direction_cone
 target=torch.tensor([[1.,0.]]).expand(4,-1)
 values=torch.tensor([[2.,0.],[0.,3.],[-4.,0.],[0.,0.]])
 result=project_native_direction_cone(values,target,.99)
 assert torch.equal(result[0],values[0])
 assert torch.allclose(result[:3].norm(dim=-1),values[:3].norm(dim=-1),atol=1e-6)
 assert (F.normalize(result,dim=-1)[:,0]>=.99-1e-6).all()
 assert torch.isfinite(result).all()


def test_fixed_direction_constraint_retains_teacher_exactly():
 from radio_gs.training.native_space_alignment import project_native_direction_cone
 target=F.normalize(torch.tensor([[1.,2.],[3.,4.]]),dim=-1)
 result=project_native_direction_cone(torch.tensor([[4.,3.],[-2.,1.]]),target,1.)
 assert torch.allclose(F.normalize(result,dim=-1),target,atol=1e-6)


def test_native_cone_does_not_inflate_small_nonzero_features():
 from radio_gs.training.native_space_alignment import project_native_direction_cone
 value=torch.tensor([[0.,1e-4]])
 result=project_native_direction_cone(value,torch.tensor([[1.,0.]]),.99)
 assert torch.allclose(result.norm(dim=-1),value.norm(dim=-1),atol=1e-9,rtol=0)
