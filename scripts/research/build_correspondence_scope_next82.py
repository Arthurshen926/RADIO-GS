"""Independent automatic-region observations for a fixed seed's equivalence component.
No containment edges, query reranking, tracker masks or evaluation labels enter
component construction. Connected components are hypotheses, not certified IDs.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();torch.set_num_threads(4);key='lerf_'+a.scene
 relation=load(Path('output/region_relations_next79')/a.scene/'relations.pt');manifest=json.loads(Path(relation['source_manifest']).read_text());n=sum(e['regions'] for e in manifest['entries']);parent=list(range(n))
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 for i,j,_ in relation['equivalence_edges']:parent[find(i)]=find(j)
 components=np.array([find(i) for i in range(n)]);seeds=json.loads((Path('output/region_seeds_next80')/key/'proposals.json').read_text());tracks={r['query_index']:load(Path('output/region_seed_tracks_next80')/key/f"{r['query_index']}.pt") for r in seeds};observations={r['query_index']:{} for r in seeds};diagnostic={r['query_index']:[] for r in seeds}
 roots={r['query_index']:find(r['pair_nodes'][0]) for r in seeds};h,w=next(iter(next(iter(tracks.values()))['frames'].values()))['probability'].shape
 for entry in manifest['entries']:
  frame=entry['frame'];offset=relation['offsets'][frame];labels=components[offset:offset+entry['regions']];selected={qi:np.where(labels==root)[0] for qi,root in roots.items()};selected={qi:ids for qi,ids in selected.items() if len(ids)}
  if not selected:continue
  row=load(Path('output/source_coverage_next72')/a.scene/'native_regions'/f'frame_{frame:05d}.pt');sam=load(row['sam_path']);masks=unpack_masks(sam['packed_masks'],sam['mask_shape'][1])
  for qi,ids in selected.items():
   mask=masks[row['native_proposal_indices'][ids].numpy()].any(0);prob=F.interpolate(torch.from_numpy(mask.copy()).float()[None,None],size=(h,w),mode='nearest-exact')[0,0];observations[qi][frame]=dict(probability=prob.half(),present=bool(prob.any()))
   t=tracks[qi]['frames'].get(frame);binary=t['probability']>.5 if t and t['present'] else torch.zeros(h,w,dtype=torch.bool);m=prob>.5
   diagnostic[qi].append(dict(frame=frame,regions=len(ids),iou=float((m&binary).sum()/(m|binary).sum().clamp_min(1)),region_area=int(m.sum()),track_area=int(binary.sum()),track_to_region_area=float(binary.sum()/m.sum().clamp_min(1))))
 out=Path('output/correspondence_scope_next82')/key;out.mkdir(parents=True,exist_ok=True);report=[]
 for r in seeds:
  qi=r['query_index'];t=tracks[qi];torch.save(dict(query=t['query'],frames=observations[qi],seed_frame=t['seed_frame'],seed_evidence=t['seed_evidence'],source='automatic regions in same frozen equivalence component; no temporal masks'),out/f'{qi}.pt');v=diagnostic[qi];report.append(dict(query=r['query'],query_index=qi,component_regions=int((components==roots[qi]).sum()),source_views=len(v),mean_track_region_iou=float(np.mean([x['iou'] for x in v])),median_track_area_ratio=float(np.median([x['track_to_region_area'] for x in v])),frames=v))
 (out/'agreement.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps([{k:v for k,v in r.items() if k!='frames'} for r in report],indent=2))
if __name__=='__main__':main()
