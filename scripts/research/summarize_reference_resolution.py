"""Compare all eight queries; fixed baseline error regions for conflict analysis."""
import json,torch,cv2,numpy as np
from pathlib import Path
from PIL import Image
root=Path('output/reference_resolution');scenes=['orchids','leaves','fern','flower','fortress','horns_center','horns_left','trex'];rows=[]
for s in scenes:
    ev=json.loads(Path(f'configs/consolidated/nvos/{s}.evaluation.json').read_text());truth=np.asarray(Image.open(ev['gt_mask']));truth=np.any(truth>127,axis=2) if truth.ndim==3 else truth>127;size=(truth.shape[1],truth.shape[0]);arms={}
    for arm in ['baseline','double_raster','long_solver','double_long']:
        dest=Path('output/reference_integration/fresh')/s if arm=='baseline' else root/arm/s
        d=json.loads((dest/'evaluation/evaluation.json').read_text())['metrics'];pred=torch.load(dest/'evaluation/response.pt',map_location='cpu',weights_only=False)['response'].numpy();pred=cv2.resize(pred,size)>=ev['threshold'];tp=int((truth&pred).sum());d.update(precision=tp/max(1,int(pred.sum())),recall=tp/max(1,int(truth.sum())),fit_seconds=json.loads((dest/'fit.json').read_text())['seconds']);arms[arm]=d
    conflict={}
    for arm,path in [('baseline',Path('output/reference_conflict')/s),('double_raster',root/'double_raster'/s/'conflict')]:
        data=np.load(path/'target_evidence.npz');f=cv2.resize(data['fractions'],size);response=cv2.resize(data['response'],size);fn=truth&(response<ev['threshold']);conflict[arm]=dict(raw_within_mean_on_fixed_fn=float(f[...,4][fn].mean()),raw_between_mean_on_fixed_fn=float(f[...,5][fn].mean()),fixed_fn_pixels=int(fn.sum()))
    rows.append(dict(scene=s,arms=arms,conflict=conflict))
result=dict(scenes=rows,macro={arm:{m:sum(r['arms'][arm][m] for r in rows)/8 for m in ['iou','average_precision','precision','recall','fit_seconds']} for arm in ['baseline','double_raster','long_solver','double_long']},default_changed=False,development_only=True,geometry_changed=False,teacher_changed=False,boundary_radius_scaled_with_raster=True)
(root/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(result['macro'])
for r in rows:print(r['scene'],{k:round(v['iou']*100,3) for k,v in r['arms'].items()},r['conflict'])
