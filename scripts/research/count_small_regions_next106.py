import json
from pathlib import Path
import torch,numpy as np
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks
torch.set_num_threads(4)
for scene in ['ramen','waldo_kitchen']:
 root=Path('output/source_coverage_next72')/scene;frames=json.loads((root/'cohorts.json').read_text())['frames']['all'];rows=[]
 for f in frames:
  s=torch.load(root/'source_sam'/f'frame_{f:05d}.pt',weights_only=False,map_location='cpu');ms=unpack_masks(s['packed_masks'],s['mask_shape'][1]);pixels=ms.sum((1,2));area=pixels/(ms.shape[1]*ms.shape[2]);rows.append(dict(frame=f,raw=len(ms),old=int(((area>=.002)&(area<=.6)).sum()),small=int(((area<.002)&(pixels>=64)).sum()),tiny=int((pixels<64).sum())))
 out=Path('output/next106')/('lerf_'+scene);out.mkdir(parents=True,exist_ok=True);report=dict(scene=scene,source_frames=len(frames),counts={k:sum(r[k] for r in rows) for k in ['raw','old','small','tiny']},rows=rows);(out/'small_regions.json').write_text(json.dumps(report,indent=2)+'\n');print(scene,report['counts'],flush=True)
