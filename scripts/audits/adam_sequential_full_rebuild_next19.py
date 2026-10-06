"""Bounded launcher: same Adam update, sequential tensors to avoid foreach peak."""
import json,runpy,sys
import torch
OriginalAdam=torch.optim.Adam
class SequentialAdam(OriginalAdam):
 def __init__(self,*args,**kwargs):
  assert kwargs.get('foreach') in (None,False)
  kwargs['foreach']=False
  super().__init__(*args,**kwargs)
if sys.argv[1:]==['--verify']:
 torch.manual_seed(19)
 params=[torch.randn(1024,96,device='cuda'),torch.randn(64,96,device='cuda')]
 a=[p.clone().requires_grad_() for p in params];b=[p.clone().requires_grad_() for p in params]
 x=OriginalAdam(a,lr=.003,foreach=True);y=SequentialAdam(b,lr=.003)
 for i in range(80):
  for p,q in zip(a,b):
   g=torch.randn_like(p);p.grad=g;q.grad=g.clone()
  x.step();y.step()
 d=max(float((p-q).abs().max()) for p,q in zip(a,b))
 assert d<1e-6,d
 print(json.dumps({'steps':80,'max_abs_parameter_difference':d,'foreach_before':True,'foreach_after':False,'same_hyperparameters':True}))
else:
 torch.optim.Adam=SequentialAdam
 entry=sys.argv[1];sys.argv=sys.argv[1:]
 runpy.run_path(entry,run_name='__main__')
