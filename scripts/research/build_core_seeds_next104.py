"""Confirmed-majority core anchors; uncertain support remains a separate hypothesis.
Do not initialize tracking from a tiny accepted speck amid an uncertain object.
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np,torch
from scipy.ndimage import label
load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();key='lerf_'+a.scene;out=Path('output/core_seeds_next104')/key;out.mkdir(parents=True,exist_ok=True);trackout=Path('output/core_tracks_next104')/key;trackout.mkdir(parents=True,exist_ok=True);report=[]
for r in json.loads((Path('output/grounded_seeds_next88')/key/'proposals.json').read_text()):
 qi=r['query_index'];original=load(Path('output/grounded_seeds_next88')/key/f'{qi}.pt');base=load(Path('output/reference_component_seeds_next103')/key/f'{qi}.pt');d=json.loads((Path('output/reference_component_seeds_next103')/key/f'{qi}.json').read_text());accepted=[];uncertain=[]
 for mi,m in enumerate(original['masks']):
  components,n=label(m.numpy(),structure=np.ones((3,3)));js=[j for j in d['judgments'] if j['mask']==mi];aa=[j['component'] for j in js if j['answer']=='A'];cc=[j['component'] for j in js if j['answer']=='C'];accepted.append(torch.from_numpy(np.isin(components,aa)));uncertain.append(torch.from_numpy(np.isin(components,cc)))
 core=torch.stack(accepted);pending=torch.stack(uncertain);core_pixels=int(core.sum());pending_pixels=int(pending.sum());eligible=core_pixels>0 and core_pixels>=pending_pixels
 masks=core if eligible else base['masks'];changed=not torch.equal(masks,base['masks']);seed=dict(base,masks=masks,source='majority confirmed source core anchors tracker; uncertain components stored separately',uncertain_masks=pending,core_eligible=eligible);torch.save(seed,out/f'{qi}.pt');reuse=None
 for seeds,tracks in [('output/grounded_seeds_next88','output/boundary_tracks_next95/high_primary'),('output/reference_component_seeds_next103','output/reference_component_tracks_next103'),('output/clean_seeds_next102','output/clean_tracks_next102')]:
  cached=load(Path(seeds)/key/f'{qi}.pt')
  if cached['seed_frame']==seed['seed_frame'] and torch.equal(cached['masks'],masks):
   reuse=str(Path(tracks)/key/f'{qi}.pt');t=load(reuse);t['seed_evidence']=seed;torch.save(t,trackout/f'{qi}.pt');break
 report.append(dict(query=seed['query'],query_index=qi,core_pixels=core_pixels,pending_pixels=pending_pixels,eligible=eligible,changed=changed,reused_track=reuse))
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
