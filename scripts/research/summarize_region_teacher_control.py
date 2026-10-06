"""Compare completed predictions, with labels used only for retrospective diagnosis."""
import argparse,json
from pathlib import Path
import numpy as np
from plyfile import PlyData
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--layout',choices=['source','map'],required=True);a=p.parse_args()
root=Path(a.root);summary=json.loads((root/'summary.json').read_text());result={}
for mode,scenes in summary['scenes'].items():
    rows={};totals={arm:dict(corrected=0,spoiled=0,changed=0,valid=0) for arm in ['map','source']}
    for scene,metrics in scenes.items():
        after=(root/mode/scene if a.layout=='source' else root/scene/mode)/'evaluation'
        before=Path('output/system_v5/scannet')/scene/'evaluation'
        base=json.loads((before/'report.json').read_text())['point_metrics']
        rows[scene]={arm:{s:100*(metrics[arm][s]['miou']-base[arm][s]['miou']) for s in ['19','15','10']} for arm in ['map','source']}
        spec=json.loads(Path(f'configs/consolidated/scannet_native/{scene}.evaluation.json').read_text())
        truth=np.asarray(PlyData.read(spec['ground_truth'])['vertex'].data['label']);valid=np.isin(truth,spec['splits']['19'])
        for arm in ['map','source']:
            b=np.load(before/f'{arm}_point_predictions.npz')['19'];v=np.load(after/f'{arm}_point_predictions.npz')['19']
            counts=dict(corrected=int((valid&(b!=truth)&(v==truth)).sum()),spoiled=int((valid&(b==truth)&(v!=truth)).sum()),changed=int((valid&(b!=v)).sum()),valid=int(valid.sum()))
            for k,val in counts.items():totals[arm][k]+=val
    result[mode]=dict(scene_delta_pp=rows,paired_points_not_scene_macro=totals,
        scene_counts={arm:dict(improve=sum(r[arm]['19']>1e-9 for r in rows.values()),regress=sum(r[arm]['19']< -1e-9 for r in rows.values()),unchanged=sum(abs(r[arm]['19'])<=1e-9 for r in rows.values())) for arm in ['map','source']})
(root/'comparison.json').write_text(json.dumps(dict(conditions=result,labels_read_only_after_prediction=True,independent_test=False),indent=2)+'\n')
for mode,r in result.items():print(mode,r['scene_counts'],r['paired_points_not_scene_macro'])
