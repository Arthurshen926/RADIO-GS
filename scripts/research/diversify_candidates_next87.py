"""Same 16-pair budget, source-support diversity instead of Jaccard near-dedup."""
import argparse,json
from pathlib import Path
from functools import lru_cache
import torch
from torch.nn import functional as F

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();torch.set_num_threads(4);scene=a.scene;key='lerf_'+scene;relation=load(Path('output/region_relations_next79')/scene/'relations.pt');manifest=json.loads(Path(relation['source_manifest']).read_text());nodes=[];features=[]
for e in manifest['entries']:
 b=load(e['path']);features.append(F.normalize(b['descriptors'].float(),dim=1));nodes.extend((i,e['path']) for i in range(e['regions']))
field=load(Path('output/native_rebuild_next54/mixture_masscal')/key/'field.pt');root=Path(field['root']);q=F.normalize(load(root.parent.parent/'evaluation/text_online.encodings.pt')[0]['encoding'].float(),dim=-1);scores=torch.cat(features)@q.T;ij=torch.tensor(relation['equivalence_edges'])[:,:2].long();frames=[]
for e in manifest['entries']:frames.extend([e['frame']]*e['regions'])
@lru_cache(maxsize=16)
def bank(path):return load(path)
@lru_cache(maxsize=8192)
def support(node):
 ri,path=nodes[node];r=bank(path);take=(r['columns']==ri)&(r['values']>.5);return set(r['ids'][r['rows'][take].long()].tolist())
rows=[]
for old in json.loads((Path('output/content_seeds_next83')/key/'proposals.json').read_text()):
 qi=old['query_index'];native=scores[ij,qi].amin(1);chosen=[];supports=[]
 for ei in native.argsort(descending=True).tolist():
  i,j=ij[ei].tolist();sig=support(i)|support(j)
  if not sig:continue
  if any(len(sig&s)/max(1,min(len(sig),len(s)))>.2 for s in supports):continue
  supports.append(sig);chosen.append(dict(nodes=[i,j],frames=[frames[i],frames[j]],native_score=float(native[ei])))
  if len(chosen)==16:break
 rows.append(dict(query=old['query'],query_index=qi,candidates=chosen));print(old['query'],len(chosen),flush=True)
out=Path('output/diverse_candidates_next87')/key;out.mkdir(parents=True,exist_ok=True);(out/'proposals.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'contract.json').write_text(json.dumps(dict(candidate_budget=16,source_frames=manifest['source_frames'],overlap='intersection/min support <= 0.2',target_camera_rgb_or_labels_used=False,scope='candidate diversity only; disjoint support is not verified instance identity'),indent=2)+'\n')
