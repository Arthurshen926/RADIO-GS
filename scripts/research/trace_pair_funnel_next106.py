"""Replay source pair proposal decisions before any query verification or target evaluation."""
import argparse,json
from pathlib import Path
from functools import lru_cache
import torch
from torch.nn import functional as F
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--query-index',required=True,type=int);a=p.parse_args();torch.set_num_threads(4);load=lambda p:torch.load(p,map_location='cpu',weights_only=False);scene=a.scene;key='lerf_'+scene;qi=a.query_index
relation=load(Path('output/region_relations_next79')/scene/'relations.pt');manifest=json.loads(Path(relation['source_manifest']).read_text());nodes=[];features=[]
for e in manifest['entries']:
 b=load(e['path']);features.append(F.normalize(b['descriptors'].float(),dim=1));nodes.extend((e['frame'],i,e['path']) for i in range(e['regions']))
f=load(Path('output/native_rebuild_next54/mixture_masscal')/key/'field.pt');root=Path(f['root']);q=F.normalize(load(root.parent.parent/'evaluation/text_online.encodings.pt')[0]['encoding'].float(),dim=-1);scores=torch.cat(features)@q[qi];anchors=scores.argsort(descending=True)[:128].tolist();ranks={n:i+1 for i,n in enumerate(anchors)}
@lru_cache(maxsize=16)
def bank(path):return load(path)
def support(n):
 _,ri,path=nodes[n];b=bank(path);take=(b['columns']==ri)&(b['values']>.5);return set(b['ids'][b['rows'][take].long()].tolist())
supports={n:support(n) for n in anchors};pairs={};records=[]
for n in anchors:
 others=[j for j in anchors if nodes[j][0]!=nodes[n][0]];j=max(others,key=lambda j:(len(supports[n]&supports[j])/max(1,len(supports[n]|supports[j])),float(scores[j])));overlap=len(supports[n]&supports[j])/max(1,len(supports[n]|supports[j]));pair=tuple(sorted([n,j]));records.append(dict(node=n,rank=ranks[n],frame=nodes[n][0],support=len(supports[n]),best_partner=j,partner_rank=ranks[j],support_jaccard=overlap,pair=pair))
 if overlap>0:pairs[pair]=overlap
ranked=sorted(pairs,key=lambda ij:float(scores[list(ij)].min()),reverse=True);used=[];chosen=[];status={}
for i,pair in enumerate(ranked):
 union=supports[pair[0]]|supports[pair[1]]
 if len(chosen)>=16:reason='pair_budget'
 elif any(len(union&s)/max(1,len(union|s))>.8 for s in used):reason='support_deduplication'
 else:reason='selected';used.append(union);chosen.append(pair)
 status[pair]=dict(pair_rank=i+1,pair_outcome=reason)
for r in records:r.update(status.get(tuple(r['pair']),dict(pair_rank=None,pair_outcome='no_positive_overlap')));r['node_admitted']=any(r['node'] in pair for pair in chosen)
old=next(r for r in json.loads((Path('output/grounded_seeds_next88')/key/'proposals.json').read_text()) if r['query_index']==qi);report=dict(query=old['query'],records=records,pairs=len(pairs),chosen=[list(x) for x in chosen],reproduces_old=chosen==[tuple(c['nodes']) for c in old['candidates']],outcomes={reason:sum(r['pair_outcome']==reason for r in records) for reason in ['selected','pair_budget','support_deduplication','no_positive_overlap']},target_rgb_or_gt_used=False)
out=Path('output/next106')/key;(out/f'{qi}_pair_funnel.json').write_text(json.dumps(report,indent=2)+'\n');print(key,report['reproduces_old'],report['outcomes']);print([r for r in records if r['node'] in [8032,4218]])
