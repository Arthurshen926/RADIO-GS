"""Full-cohort coverage and positive-core comparison; no per-query selection."""
import json
import torch,numpy as np,cv2
from PIL import Image
from pathlib import Path
root=Path('output/reference_evidence_coverage');scenes=['orchids','leaves','fern','flower','fortress','horns_center','horns_left','trex'];rows=[]
for s in scenes:
    d=json.loads((root/s/'diagnosis.json').read_text());baseline=json.loads(Path(f'output/reference_integration/fresh/{s}/evaluation/evaluation.json').read_text())['metrics'];arms={'eroded':baseline}
    for arm in ['medial','raw']:arms[arm]=json.loads((root/'candidates'/arm/s/'evaluation/evaluation.json').read_text())['metrics']
    ev=json.loads(Path(f'configs/consolidated/nvos/{s}.evaluation.json').read_text());truth=np.array(Image.open(ev['gt_mask']));truth=np.any(truth>127,axis=2) if truth.ndim==3 else truth>127
    for arm in arms:
        path=Path(f'output/reference_integration/fresh/{s}/evaluation/response.pt') if arm=='eroded' else root/'candidates'/arm/s/'evaluation/response.pt'
        pred=torch.load(path,map_location='cpu',weights_only=False)['response'].numpy();pred=cv2.resize(pred,(truth.shape[1],truth.shape[0]),interpolation=cv2.INTER_LINEAR)>=ev['threshold'];tp=int((pred&truth).sum());arms[arm]['precision']=tp/max(1,int(pred.sum()));arms[arm]['recall']=tp/max(1,int(truth.sum()))
    rows.append(dict(scene=s,coverage=d,arms=arms))
result=dict(scenes=rows,macro={arm:{m:sum(r['arms'][arm][m] for r in rows)/8 for m in ['iou','average_precision','precision','recall']} for arm in ['eroded','medial','raw']},development_only=True,source_witnesses_are_pseudolabels=True,target_rgb_used=False,default_changed=False)
(root/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(result['macro'])
for r in rows:print(r['scene'],{a:round(v['iou']*100,3) for a,v in r['arms'].items()})
