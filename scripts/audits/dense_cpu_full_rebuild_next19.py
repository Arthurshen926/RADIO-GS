"""Exact GPU interpolation, CPU backing, selected rows restored to same GPU."""
import hashlib,runpy,sys
from pathlib import Path
import torch
EXPECTED='4dd0e22c64b0035046c072855529ca1263aee8e3b51b0cdb52df2e0e6bd85753'
class DenseCPU:
 def __init__(self,t):self.device=t.device;self.backing=t.cpu()
 def __getitem__(self,key):
  key=tuple(x.cpu() if isinstance(x,torch.Tensor) else x for x in key)
  return self.backing[key].to(self.device)
if sys.argv[1:]==['--verify']:
 import json
 t=torch.nn.functional.interpolate(torch.randn(3,64,16,20,device='cuda'),size=(96,128),mode='bilinear',align_corners=False).flatten(2).transpose(1,2)
 c=DenseCPU(t);rows=torch.randperm(96*128,device='cuda')[:1024]
 assert torch.equal(t[1,rows],c[1,rows])
 print(json.dumps({'selected_rows':1024,'max_abs_difference':float((t[1,rows]-c[1,rows]).abs().max()),'same_gpu_interpolation':True}))
else:
 original=runpy.run_module
 def module(name,*args,**kwargs):
  if name!='radio_gs.scripts.train_three_query_field':return original(name,*args,**kwargs)
  path=Path('radio_gs/scripts/train_three_query_field.py');raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==EXPECTED
  source=raw.decode();old="dense={k:F.interpolate(teachers[k].float().to(device),size=(h,w),mode='bilinear',align_corners=False).flatten(2).transpose(1,2) for k in ('radio','dino') if k in names}"
  new="dense={k:DenseCPU(F.interpolate(teachers[k].float().to(device),size=(h,w),mode='bilinear',align_corners=False).flatten(2).transpose(1,2)) for k in ('radio','dino') if k in names}"
  assert source.count(old)==1;source=source.replace(old,new)
  scope={'__name__':'__main__','__file__':str(path),'__package__':'radio_gs.scripts','DenseCPU':DenseCPU};exec(compile(source,str(path),'exec'),scope);return scope
 runpy.run_module=module;entry=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(entry,run_name='__main__')
