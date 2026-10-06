"""Bounded seed-contamination control: dominant connected region per hypothesis.
Not a general text-union rule; disconnected true parts can be lost.
"""
import argparse,json
from pathlib import Path
import torch,numpy as np
from scipy.ndimage import label
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();key='lerf_'+a.scene;root=Path('output/clean_seeds_next102')/key;root.mkdir(parents=True,exist_ok=True);report=[]
for r in json.loads((Path('output/grounded_seeds_next88')/key/'proposals.json').read_text()):
 qi=r['query_index'];s=torch.load(Path('output/grounded_seeds_next88')/key/f'{qi}.pt',map_location='cpu',weights_only=False);masks=[];areas=[]
 for mask in s['masks']:
  labels,n=label(mask.numpy(),structure=np.ones((3,3)));counts=np.bincount(labels.ravel());counts[0]=0;largest=int(counts.argmax());masks.append(torch.from_numpy(labels==largest));areas.append(dict(components=n,before=int(mask.sum()),after=int(counts[largest])))
 s['masks']=torch.stack(masks);s['source']='dominant connected component per original single-scope seed hypothesis; optional control';torch.save(s,root/f'{qi}.pt');report.append(dict(query=s['query'],query_index=qi,areas=areas))
(root/'components.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
