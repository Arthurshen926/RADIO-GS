"""Post-hoc camera-coverage diagnostic; never consumed by source selection."""
import json,argparse
p=argparse.ArgumentParser();p.add_argument("--seed-base",default="output/grounded_seeds_next85");p.add_argument("--output-base",default="output/next86");a=p.parse_args()
from pathlib import Path
from functools import lru_cache
import torch
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
torch.set_num_threads(4);key='lerf_waldo_kitchen';scene='waldo_kitchen';row=next(r for r in json.loads((Path(a.seed_base)/key/'proposals.json').read_text()) if r['query']=='spatula');relation=load(Path('output/region_relations_next79')/scene/'relations.pt');manifest=json.loads(Path(relation['source_manifest']).read_text());nodes=[]
for e in manifest['entries']:nodes.extend((e['frame'],i,e['path']) for i in range(e['regions']))
r=load(Path('output/grounded_scope_next85/source')/key/'typed_readout.pt');binding=r['selection_geometry'];geo=FrozenGaussianGeometry.load(binding['path'],expected_sha256=binding['sha256'],device='cpu');data=load(Path('output/grounded_scope_next85/source')/key/'field.pt');gr=json.loads(Path(data['geometry_report']).read_text());prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256']);cam=prior.cameras[66];xyz=geo.xyz@cam.world_to_camera[:3,:3].T+cam.world_to_camera[:3,3];uv=xyz@cam.intrinsic.T;uv=uv[:,:2]/xyz[:,2:].clamp_min(1e-8);inside=(xyz[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<cam.width)&(uv[:,1]>=0)&(uv[:,1]<cam.height)
@lru_cache(maxsize=8)
def bank(path):return load(path)
results=[]
for i,c in enumerate(row['candidates']):
 ids=set()
 for node in c['nodes']:
  frame,ri,path=nodes[node];b=bank(path);take=(b['columns']==ri)&(b['values']>.5);ids.update(b['ids'][b['rows'][take].long()].tolist())
 indices=torch.tensor(sorted(ids),dtype=torch.long);results.append(dict(rank=i,native_score=c['native_score'],grounded_margin=min(row['candidate_margin'][i]),support=len(ids),centers_in_scored_camera=int(inside[indices].sum())))
report=dict(query='spatula',camera=66,interpretation='centre projection only, not visibility or IoU; diagnostic does not select predictions',candidates=results);out=Path(a.output_base)/'candidate_coverage.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
