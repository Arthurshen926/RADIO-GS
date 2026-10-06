"""Source-only mask retention before/after resolution change; no target labels."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
torch.set_num_threads(4);rows=[]
for scene in ['figurines','ramen','teatime','waldo_kitchen']:
 key='lerf_'+scene;proposals=json.loads((Path('output/grounded_seeds_next88')/key/'proposals.json').read_text());items=[(r['query_index'],'primary') for r in proposals]
 if scene=='waldo_kitchen':items.append((14,'secondary'))
 for qi,role in items:
  row=dict(scene=scene,query_index=qi,role=role,arms={})
  for arm in ['low','high']:
   path=Path('output/boundary_tracks_next95')/(arm+'_'+role)/key/f'{qi}.pt';t=load(path);row['query']=t['query'];masks=[r['probability'].float() for f,r in t['frames'].items() if r['present'] and f!=t['seed_frame']]
   x=torch.stack(masks)[:,None];binary=(x>.5).float();core=-F.max_pool2d(-binary,3,1,1)>.5;p=binary.flatten(1).sum(1);c=core.flatten(1).sum(1)
   row['arms'][arm]=dict(raster=list(x.shape[-2:]),nonempty=int((p>0).sum()),core_nonempty=int((c>0).sum()),positive_pixels=int(p.sum()),core_pixels=int(c.sum()),core_retention=float(c.sum()/p.sum().clamp_min(1)))
  oldbase='output/grounded_tracks_next88' if role=='primary' else 'output/secondary_tracks_next89'
  old=load(Path(oldbase)/key/f'{qi}.pt');low=load(Path('output/boundary_tracks_next95/low_'+role)/key/f'{qi}.pt');same=[torch.equal(v['probability'],old['frames'][f]['probability']) and v['present']==old['frames'][f]['present'] for f,v in low['frames'].items()];row['low_frames_exact_to_old']=sum(same);row['frame_count']=len(same);rows.append(row)
Path('output/next95/source_evidence.json').write_text(json.dumps(rows,indent=2)+'\n');print([(r['query'],r['role'],round(r['arms']['low']['core_retention'],3),round(r['arms']['high']['core_retention'],3),r['low_frames_exact_to_old'],r['frame_count']) for r in rows])
