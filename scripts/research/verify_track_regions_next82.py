"""Mark disagreements with separately generated per-view regions as unknown.
Region choice maximizes mask IoU, without target labels or query reranking.
This verifies scope consistency, not semantic referent correctness.
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
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();torch.set_num_threads(4);key='lerf_'+a.scene;source=Path('output/source_coverage_next72')/a.scene
 seeds=json.loads((Path('output/region_seeds_next80')/key/'proposals.json').read_text());tracks={r['query_index']:load(Path('output/region_seed_tracks_next80')/key/f"{r['query_index']}.pt") for r in seeds};diag={qi:[] for qi in tracks};h,w=next(iter(next(iter(tracks.values()))['frames'].values()))['probability'].shape
 frames=json.loads((source/'cohorts.json').read_text())['frames']['all']
 for frame in frames:
  active=[qi for qi,t in tracks.items() if t['frames'][frame]['present']]
  if not active:continue
  row=load(source/'native_regions'/f'frame_{frame:05d}.pt');sam=load(row['sam_path']);raw=unpack_masks(sam['packed_masks'],sam['mask_shape'][1]);rh,rw=raw.shape[-2:];yy=np.minimum(((np.arange(h)+.5)*rh/h).astype(int),rh-1);xx=np.minimum(((np.arange(w)+.5)*rw/w).astype(int),rw-1);m=torch.from_numpy(raw[:,yy[:,None],xx[None,:]][row['native_proposal_indices'].numpy()].copy()).bool();del raw
  for qi in active:
   r=tracks[qi]['frames'][frame];b=r['probability']>.5;inter=(m&b).flatten(1).sum(1);union=(m|b).flatten(1).sum(1);iou=inter/union.clamp_min(1);idx=int(iou.argmax());mask=m[idx];v=mask.float()[None,None]
   r['positive_allowed']=(-F.max_pool2d(-v,3,1,1)[0,0]>.5);r['negative_allowed']=(F.max_pool2d(v,3,1,1)[0,0]<.5);r['scope_reliability']=float(iou[idx]);diag[qi].append(dict(frame=frame,iou=float(iou[idx]),track_area=int(b.sum()),region_area=int(mask.sum()),region_index=idx,source_proposal=int(row['native_proposal_indices'][idx])))
 out=Path('output/verified_track_regions_next82')/key;out.mkdir(parents=True,exist_ok=True);report=[]
 for qi,t in tracks.items():
  t['scope_verification']='best-IoU query-independent automatic region per source view; disagreement unknown; IoU confidence';torch.save(t,out/f'{qi}.pt');v=diag[qi];report.append(dict(query=t['query'],query_index=qi,views=len(v),mean_iou=float(np.mean([x['iou'] for x in v])),low_overlap_views=sum(x['iou']<.5 for x in v),frames=v))
 (out/'agreement.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps([{k:v for k,v in r.items() if k!='frames'} for r in report],indent=2))
if __name__=='__main__':main()
