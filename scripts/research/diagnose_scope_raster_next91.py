"""Source-only loss of positive support at the consumed scope raster."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
torch.set_num_threads(4);rows=[]
for scene in ['figurines','ramen','teatime','waldo_kitchen']:
 key='lerf_'+scene;proposals=json.loads((Path('output/grounded_seeds_next88')/key/'proposals.json').read_text());items=[(p['query_index'],'primary',Path('output/grounded_tracks_next88')/key/f"{p['query_index']}.pt") for p in proposals]
 if scene=='waldo_kitchen':
  qi=next(p['query_index'] for p in proposals if p['query']=='spatula');items.append((qi,'secondary',Path('output/secondary_tracks_next89')/key/f'{qi}.pt'))
 for qi,role,path in items:
  t=load(path);masks=[v['probability'].float() for f,v in t['frames'].items() if v['present'] and f!=t['seed_frame']];x=torch.stack(masks)[:,None];binary=(x>.5).float();eroded=-F.max_pool2d(-binary,3,1,1)>.5;positive=binary.flatten(1).sum(1);inner=eroded.flatten(1).sum(1);rows.append(dict(scene=scene,query=t['query'],role=role,raster=list(x.shape[-2:]),present_nonseed_views=len(masks),nonempty_views=int((positive>0).sum()),eroded_nonempty_views=int((inner>0).sum()),positive_pixels=int(positive.sum()),eroded_positive_pixels=int(inner.sum()),retained_positive_fraction=float(inner.sum()/positive.sum().clamp_min(1))))
Path('output/next91/source_raster_diagnostic.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
