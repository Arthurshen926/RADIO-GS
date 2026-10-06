"""Same upstream Adam arithmetic on bounded flattened views; no full-size denom."""
import importlib,json,runpy,sys
import torch
adam=importlib.import_module('torch.optim.adam');original=adam._single_tensor_adam;OriginalAdam=torch.optim.Adam
CHUNK=8388608
def chunked(params,grads,exp_avgs,exp_avg_sqs,max_exp_avg_sqs,state_steps,grad_scale,found_inf,**kw):
 assert not kw['capturable'] and not kw['differentiable']
 for i,p in enumerate(params):
  arrays=[p,grads[i],exp_avgs[i],exp_avg_sqs[i]]
  assert all(t.is_contiguous() and not t.is_complex() for t in arrays)
  flat=[t.view(-1) for t in arrays];start_step=state_steps[i].clone()
  for start in range(0,p.numel(),CHUNK):
   slices=[[t[start:start+CHUNK].to(p.device)] for t in flat];local_step=start_step.clone()
   maximum=[max_exp_avg_sqs[i].view(-1)[start:start+CHUNK]] if kw['amsgrad'] else []
   original(*slices,maximum,[local_step],grad_scale,found_inf,**kw)
   for j in [2,3]:flat[j][start:start+CHUNK].copy_(slices[j][0])
  state_steps[i].copy_(local_step)
class ChunkedAdam(OriginalAdam):
 def __init__(self,*a,**kw):
  assert kw.get('foreach') in (None,False);kw['foreach']=False;super().__init__(*a,**kw)
 def step(self,closure=None):
  for group in self.param_groups:
   assert not group['amsgrad'] and not group['capturable']
   for p in group['params']:
    if p.grad is None or p.numel()<65536:continue
    state=self.state[p]
    if not state:
     state['step']=torch.tensor(0.);state['exp_avg']=torch.zeros(p.shape,dtype=p.dtype,device='cpu');state['exp_avg_sq']=torch.zeros(p.shape,dtype=p.dtype,device='cpu')
  return super().step(closure)
if sys.argv[1:]==['--verify']:
 torch.manual_seed(19);CHUNK=16384
 a=[torch.randn(1024,96,device='cuda',requires_grad=True),torch.randn(64,96,device='cuda',requires_grad=True)];b=[p.detach().clone().requires_grad_() for p in a]
 x=OriginalAdam(a,lr=.003,foreach=False);y=ChunkedAdam(b,lr=.003)
 for i in range(80):
  for p,q in zip(a,b):g=torch.randn_like(p);p.grad=g;q.grad=g.clone()
  adam._single_tensor_adam=original;x.step();adam._single_tensor_adam=chunked;y.step()
 d=max(float((p-q).abs().max()) for p,q in zip(a,b));mom=max(float((x.state[p][k]-y.state[q][k].to(x.state[p][k].device)).abs().max()) for p,q in zip(a,b) for k in ['exp_avg','exp_avg_sq','step'])
 assert d==0 and mom==0,(d,mom)
 print(json.dumps(dict(steps=80,parameter_max_abs=d,moments_and_step_max_abs=mom,training_chunk_elements=8388608,test_chunk_elements=16384)))
else:
 adam._single_tensor_adam=chunked;torch.optim.Adam=ChunkedAdam
 entry=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(entry,run_name='__main__')
