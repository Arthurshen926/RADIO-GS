"""Competing A+C propagation can withhold exclusions, never assert positives.
This is a conservative scope hypothesis, not verified correspondence for C.
"""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();key='lerf_'+a.scene;state=load(Path('output/core_scope_next104/state')/(key+'.pt'));frames=json.loads((Path('output/source_coverage_next72')/a.scene/'cohorts.json').read_text())['frames']['all'];rows=json.loads((Path('output/core_seeds_next104')/key/'report.json').read_text());report=[]
for row in rows:
 if not row['changed']:continue
 qi=row['query_index']
 if qi not in state:continue
 competing=load(Path('output/reference_component_merged_next103/union')/key/f'{qi}.pt');fixed=state[qi];before=after=0
 for j,vi in enumerate(fixed['view_indices']):
  r=competing['frames'].get(frames[vi],{});neg=fixed['negative'][j];before+=int(neg.sum())
  if r.get('present',False):
   mask=(r['probability']>.5).float();phase=vi%4;outside=(F.max_pool2d(mask[None,None],3,1,1)[0,0]<.5)[phase//2::2,phase%2::2].flatten();neg=neg&outside
  fixed['negative'][j]=neg;after+=int(neg.sum())
 report.append(dict(query=row['query'],negative_before=before,negative_after=after))
out=Path('output/core_scope_next104/pending_state');out.mkdir(parents=True,exist_ok=True);torch.save(state,out/(key+'.pt'));(out/(key+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(report)
