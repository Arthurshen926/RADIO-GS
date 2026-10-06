"""Change only fixed negative masks using independently anchored source tracking."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();key='lerf_'+a.scene;frames=json.loads((Path('output/source_coverage_next72')/a.scene/'cohorts.json').read_text())['frames']['all'];report=[]
for arm in ['agreement','veto']:
 state=load(Path('output/fixed_measure_next100/state')/(key+'.pt'))
 for qi,fixed in state.items():
  partner=load(Path('output/partner_tracks_next102')/key/f'{qi}.pt');original=load(Path('output/uncertain_tracks_next99/all')/key/f'{qi}.pt');before=after=missing=0;viewrows=[]
  for j,vi in enumerate(fixed['view_indices']):
   frame=frames[vi];r=partner['frames'].get(frame,{});neg=fixed['negative'][j];before+=int(neg.sum());present=r.get('present',False)
   if present:
    mask=r['probability'].float()>.5;h,w=mask.shape;phase=vi%4;outside=(F.max_pool2d(mask.float()[None,None],3,1,1)[0,0]<.5)[phase//2::2,phase%2::2].flatten();new=neg&outside
    old=original['frames'].get(frame,{});om=old['probability']>.5 if old.get('present',False) else torch.zeros_like(mask);iou=float((mask&om).sum()/(mask|om).sum().clamp_min(1))
   else:
    missing+=1;new=torch.zeros_like(neg) if arm=='agreement' else neg;iou=None
   fixed['negative'][j]=new;after+=int(new.sum());viewrows.append(dict(frame=frame,partner_present=present,track_iou=iou,negative_before=int(neg.sum()),negative_after=int(new.sum())))
  report.append(dict(arm=arm,query=partner['query'],query_index=qi,negative_before=before,negative_after=after,missing_partner_views=missing,views=viewrows))
 out=Path('output/partner_exclusion_next102/state')/arm;out.mkdir(parents=True,exist_ok=True);torch.save(state,out/(key+'.pt'))
root=Path('output/partner_exclusion_next102');(root/(key+'_source.json')).write_text(json.dumps(report,indent=2)+'\n');print([(r['arm'],r['query'],r['negative_before'],r['negative_after'],r['missing_partner_views']) for r in report])
