"""Separate native retrieval ordering from categorical source admission.
Same candidate bank and per-pair seed view as next88. No new teacher calls/GT.
"""
import argparse,json
from pathlib import Path
from functools import lru_cache
import torch
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();torch.set_num_threads(4);key='lerf_'+a.scene;relation=load(Path('output/region_relations_next79')/a.scene/'relations.pt');manifest=json.loads(Path(relation['source_manifest']).read_text());nodes=[]
for e in manifest['entries']:nodes.extend((e['frame'],i,e['path']) for i in range(e['regions']))
@lru_cache(maxsize=4)
def bank(path):
 b=load(path);r=load(b['source']);s=load(r['sam_path']);return r,unpack_masks(s['packed_masks'],s['mask_shape'][1])
rows=json.loads((Path('output/grounded_seeds_next88')/key/'proposals.json').read_text())
for arm in ['native','qualified_native']:
 out=Path('output/native_order_seeds_next90')/arm/key;out.mkdir(parents=True,exist_ok=True);report=[]
 for row in rows:
  candidates=row['candidates'];eligible=[i for i,c in enumerate(candidates) if arm=='native' or min(row['candidate_margin'][i])>0];fallback=not eligible
  if fallback:seed=load(Path('output/region_seeds_next80')/key/f"{row['query_index']}.pt");winner=None
  else:
   winner=max(eligible,key=lambda i:candidates[i]['native_score']);c=candidates[winner];side=int(torch.tensor(row['candidate_margin'][winner]).argmax());node=c['nodes'][side];frame,ri,path=nodes[node];r,masks=bank(path);mask=masks[int(r['native_proposal_indices'][ri])].copy();seed=dict(query=row['query'],query_index=row['query_index'],seed_frame=frame,seed_confidence=max(0.,c['native_score']),masks=torch.from_numpy(mask)[None].bool(),pair_nodes=c['nodes'],pair_frames=c['frames'],single_scope_hypothesis=True)
  seed=dict(seed,source='native query orders scope candidates; categorical admission separated',policy=arm,native_fallback=fallback,view_choice='unchanged next88 higher VLM margin within pair, including native arm')
  torch.save(seed,out/f"{row['query_index']}.pt");report.append(dict(row,selected_rank=winner,native_fallback=fallback,seed_frame=seed['seed_frame'],selection_policy=arm));print(arm,row['query'],winner,flush=True)
 (out/'proposals.json').write_text(json.dumps(report,indent=2)+'\n');(out/'contract.json').write_text(json.dumps(dict(candidate_source='output/grounded_seeds_next88/'+key,ranking='minimum native cosine across pair',admission='none' if arm=='native' else 'A greater than B and C in both views',seed_view='same cached larger VLM margin within chosen pair; not teacher-free',geometry_and_field_frozen=True,target_rgb_or_gt_used=False,new_teacher_calls=0),indent=2)+'\n')
