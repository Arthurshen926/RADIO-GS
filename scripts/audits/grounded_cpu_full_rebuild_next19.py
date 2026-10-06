"""Keep frozen proposal probability table on CPU; copy exact selected rows."""
import hashlib,runpy,sys
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file
class RowCPU:
 def __init__(self,t,device):self.t=t;self.device=device
 def __getitem__(self,k):return self.t[k].float().to(self.device)
original=runpy.run_path
expected=Path('scripts/audits/grounded_cpu_full_rebuild_next19.source_sha256').read_text().strip()
def patched(path,*args,**kwargs):
 if str(path)!='scripts/audits/fit_grounded_semantic_transfer.py':return original(path,*args,**kwargs)
 p=Path(path);assert sha256_file(p)==expected;source=p.read_text();old="probability=evidence['region_probability'].float().to(a.device);seen=evidence['seen'].to(a.device)";new="probability=RowCPU(evidence['region_probability'],a.device);seen=evidence['seen'].to(a.device)";assert source.count(old)==1
 scope={'__name__':'__main__','__file__':str(p),'RowCPU':RowCPU};exec(compile(source.replace(old,new),str(p),'exec'),scope);return scope
runpy.run_path=patched;entry=sys.argv[1];sys.argv=sys.argv[1:];patched(entry,run_name='__main__')
