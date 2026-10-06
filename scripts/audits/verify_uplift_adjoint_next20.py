"""Numerical renderer-adjoint and source-unknown preservation checks."""
import json
from pathlib import Path
import torch
from gsplat import rasterization
from scripts.audits.source_uplift_scope_next20 import local_correction

torch.manual_seed(42)
means=torch.tensor([[0.,0.,2.],[0.,0.,3.],[.4,0.,2.]],device='cuda')
kw=dict(means=means,quats=torch.tensor([[1.,0.,0.,0.]]*3,device='cuda'),scales=torch.full((3,3),.12,device='cuda'),opacities=torch.full((3,),.8,device='cuda'),viewmats=torch.eye(4,device='cuda')[None],Ks=torch.tensor([[[20.,0.,8.],[0.,20.,8.],[0.,0.,1.]]],device='cuda'),width=16,height=16,packed=False)
colors=torch.rand(3,2,device='cuda',requires_grad=True);rhs=torch.rand(16,16,2,device='cuda')
im,_,_=rasterization(colors=colors,**kw);lhs=(im[0]*rhs).sum();adj=torch.autograd.grad(lhs,colors)[0]
error=float((lhs.detach()-(colors.detach()*adj).sum()).abs());assert error<1e-5
ones=torch.zeros(3,2,device='cuda',requires_grad=True);render,_,_=rasterization(colors=ones,**kw);rhs=torch.stack([torch.full((16,16),.7,device='cuda'),torch.ones(16,16,device='cuda')],-1);grad=torch.autograd.grad((render[0]*rhs).sum(),ones)[0]
mean=grad[:,0]/grad[:,1];constant_error=float((mean-.7).abs().max());assert constant_error<1e-6
assert grad[0,1]>grad[1,1], 'front/back transmittance must not be equal center projection'
out=Path('output/source_uplift_scope_next20/adjoint_verification.json');assert not out.exists();out.write_text(json.dumps(dict(complete=True,adjoint_inner_product_abs_error=error,constant_reconstruction_max_error=constant_error,front_weight=float(grad[0,1]),back_weight=float(grad[1,1]),device='cuda',seed=42),indent=2)+'\n');print(out.read_text())
