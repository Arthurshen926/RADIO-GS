"""Use positive geometric overlap as proposal association, not equivalence admission."""
import argparse,json
from pathlib import Path
from functools import lru_cache
import torch
from torch.nn import functional as F

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();torch.set_num_threads(4);scene=a.scene;key='lerf_'+scene;relation=load(Path('output/region_relations_next79')/scene/'relations.pt');manifest=json.loads(Path(relation['source_manifest']).read_text());nodes=[];features=[]
for e in manifest['entries']:
 b=load(e['path']);features.append(F.normalize(b['descriptors'].float(),dim=1));nodes.extend((e['frame'],i,e['path']) for i in range(e['regions']))
field=load(Path('output/native_rebuild_next54/mixture_masscal')/key/'field.pt');root=Path(field['root']);q=F.normalize(load(root.parent.parent/'evaluation/text_online.encodings.pt')[0]['encoding'].float(),dim=-1);native_scores=torch.cat(features)@q.T;scores=native_scores.clone();term_data=load('output/literal_terms_next105/terms.pt');term_scores=torch.cat(features)@term_data['features'].T
def standardize(v):return (v-v.mean())/v.std().clamp_min(1e-6)
for row in term_data['rows']:
 if row['scene']==scene and len(row['terms'])>1:
  ti=[term_data['terms'].index(t) for t in row['terms']];conjunction=torch.stack([standardize(term_scores[:,i]) for i in ti],1).amin(1);scores[:,row['query_index']]=.5*(standardize(native_scores[:,row['query_index']])+conjunction)
@lru_cache(maxsize=16)
def bank(path):return load(path)
@lru_cache(maxsize=4096)
def support(node):
 _,ri,path=nodes[node];r=bank(path);take=(r['columns']==ri)&(r['values']>.5);return set(r['ids'][r['rows'][take].long()].tolist())
rows=[]
for old in json.loads((Path('output/content_seeds_next83')/key/'proposals.json').read_text()):
 qi=old['query_index'];anchors=list(dict.fromkeys(native_scores[:,qi].argsort(descending=True)[:64].tolist()+scores[:,qi].argsort(descending=True)[:64].tolist()))
 for node in native_scores[:,qi].argsort(descending=True).tolist():
  if len(anchors)>=128:break
  if node not in anchors:anchors.append(node)
 supports={i:support(i) for i in anchors};pairs={}
 for i in anchors:
  others=[j for j in anchors if nodes[j][0]!=nodes[i][0]]
  j=max(others,key=lambda j:(len(supports[i]&supports[j])/max(1,len(supports[i]|supports[j])),float(scores[j,qi])))
  overlap=len(supports[i]&supports[j])/max(1,len(supports[i]|supports[j]))
  if overlap>0:pairs[tuple(sorted([i,j]))]=overlap
 ranked=sorted(pairs,key=lambda ij:float(scores[list(ij),qi].min()),reverse=True);chosen=[];used=[]
 for i,j in ranked:
  sig=supports[i]|supports[j]
  if any(len(sig&s)/max(1,len(sig|s))>.8 for s in used):continue
  used.append(sig);chosen.append(dict(nodes=[i,j],frames=[nodes[i][0],nodes[j][0]],native_score=float(native_scores[[i,j],qi].min()),retrieval_score=float(scores[[i,j],qi].min()),support_jaccard=pairs[(i,j)]))
  if len(chosen)==16:break
 rows.append(dict(query=old['query'],query_index=qi,candidates=chosen));print(old['query'],len(chosen),flush=True)
out=Path('output/literal_candidates_next105')/key;out.mkdir(parents=True,exist_ok=True);(out/'proposals.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'contract.json').write_text(json.dumps(dict(source_frames=manifest['source_frames'],native_region_shortlist=128,verifier_pair_budget=16,association='highest Jaccard in another source frame among native shortlist; positive overlap only',pair_ranking='minimum mean of phrase z-score and minimum literal-term z-score; single-word queries unchanged',deduplication_jaccard=.8,hard_equivalence_or_third_view_admission=False,target_camera_rgb_or_labels_used=False,scope='bounded query-native proposal control, not confirmed physical correspondence'),indent=2)+'\n')
