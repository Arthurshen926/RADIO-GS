"""Source-only description of overlapping native teacher observations."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks
torch.set_num_threads(4)
rows={}
for scene in ['figurines','ramen']:
 root=Path('output/source_coverage_next72')/scene
 frames=json.loads((root/'cohorts.json').read_text())['frames']['old']
 seen=overlap=0;cosine_sum=0.;cosine_count=0;area_ratio_sum=0.
 for frame in frames:
  row=torch.load(root/'native_regions'/f'frame_{frame:05d}.pt',map_location='cpu',weights_only=False)
  sam=torch.load(row['sam_path'],map_location='cpu',weights_only=False)
  native=unpack_masks(sam['packed_masks'],sam['mask_shape'][1])[row['native_proposal_indices'].numpy()]
  area=torch.from_numpy(native.sum(axis=(1,2))).float()
  mask=F.interpolate(torch.from_numpy(native.copy()).float()[:,None],size=(192,256),mode='nearest-exact')[:,0].flatten(1)>0
  count=mask.sum(0);multi=count>1;seen+=int((count>0).sum());overlap+=int(multi.sum())
  leaf=area[:,None].expand_as(mask).masked_fill(~mask,float('inf')).argmin(0)
  coarse=area[:,None].expand_as(mask).masked_fill(~mask,-1).argmax(0)
  desc=F.normalize(row['descriptors'].float(),dim=1)
  similarity=(desc[leaf[multi]]*desc[coarse[multi]]).sum(1)
  cosine_sum+=float(similarity.sum());cosine_count+=len(similarity)
  area_ratio_sum+=float((area[coarse[multi]]/area[leaf[multi]].clamp_min(1)).sum())
 rows[scene]=dict(source_frames=frames,covered_pixels=seen,multiply_covered_fraction=overlap/seen,leaf_coarse_native_cosine=cosine_sum/cosine_count,coarse_leaf_area_ratio=area_ratio_sum/cosine_count,scope='old authorized source cohort; no target annotations; area rank does not prove identity')
 print(scene,rows[scene],flush=True)
out=Path('output/next76');out.mkdir(exist_ok=True);(out/'source_overlap.json').write_text(json.dumps(rows,indent=2)+'\n')
