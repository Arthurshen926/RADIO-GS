"""Measure native teacher-mask loss at the consumed coarse scope resolution."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from radio_gs.utils.immutable_artifacts import sha256_file
torch.set_num_threads(4);result=dict(schema='scope_resolution_audit.v1',source_only=True,benchmark_data_used=False,teacher_masks_are_not_gt=True,scenes={})
for scene in ['figurines','ramen']:
 root=Path('output/pixel_evidence_rebuild_20260917')/scene/'field';p=root/'caption_grounded_regions.pt';assert sha256_file(p)==json.loads(p.with_suffix('.json').read_text())['sha256'];d=torch.load(p,weights_only=False);ep=root/'caption_region_evidence.pt';assert sha256_file(ep)==json.loads(ep.with_suffix('.json').read_text())['sha256'];e=torch.load(ep,weights_only=False);assert e['grounding_sha256']==sha256_file(p) and torch.equal(e['masks'],d['masks']);rows=[]
 for i,r in enumerate(d['records']):
  if r['confidence']<.7 or int(d['masks'][i].sum())<20:continue
  h,w=r['native_shape'];native=torch.from_numpy(np.unpackbits(r['packed_native'].numpy(),axis=-1)[:,:w].copy()).bool();coarse=d['masks'][i].bool();up=F.interpolate(coarse[None,None].float(),size=(h,w),mode='nearest-exact')[0,0].bool();inter=(up&native).sum();rows.append(dict(region=i,frame=r['frame'],coarse_pixels=int(coarse.sum()),iou=float(inter/(up|native).sum().clamp_min(1)),recall=float(inter/native.sum().clamp_min(1)),precision=float(inter/up.sum().clamp_min(1)),phrase_aliases=len(r['phrase_indices'])))
 arr=torch.tensor([r['iou'] for r in rows]);result['scenes'][scene]=dict(grounding_sha256=sha256_file(p),evidence_sha256=sha256_file(ep),native_resolution=d['records'][0]['native_shape'],coarse_resolution=list(d['masks'].shape[-2:]),regions=len(rows),mean_iou=float(arr.mean()),median_iou=float(arr.median()),p10_iou=float(arr.quantile(.1)),below_50_percent=int((arr<.5).sum()),rows=rows);print(scene,{k:v for k,v in result['scenes'][scene].items() if k!='rows'},flush=True)
result['implementation_sha256']=sha256_file(__file__);result['limitation']='Compares retained native teacher masks to coarse masks; prior coarse-key dedup discarded alternative native masks, so their aliasing loss is unidentifiable from this cache.'
Path('paper/artifacts/scope_resolution_audit_20260917.json').write_text(json.dumps(result,indent=2)+'\n')
