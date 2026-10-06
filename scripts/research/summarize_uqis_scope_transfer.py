"""Summarize the complete three-readout cohort, including regressions."""
import argparse
import json
from pathlib import Path
import numpy as np
from radio_gs.benchmarks.uqis.metrics import scene_clustered_bootstrap


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('output/uqis_scope_transfer_v1'))
    p.add_argument('--base',type=Path,default=Path('output/uqis_map_only_v1'))
    a=p.parse_args();read=lambda f:json.loads(f.read_text())
    complete=read(a.root/'predictions_complete.json')
    names=['native_semantic','anchor_comembership','anchor_correspondence']
    result={n:read(a.root/n/'result.json') for n in names}
    changes={}
    for parent in names[:2]:
        changes[parent]={}
        for m in result[parent]['modalities']:
            old=result[parent]['modalities'][m];new=result[names[2]]['modalities'][m]
            gains={s:100*(r['average_precision']-old['per_scene'][s]['average_precision']) for s,r in new['per_scene'].items()}
            changes[parent][m]=dict(ap_gain_pp=scene_clustered_bootstrap(gains),per_scene_ap_gain_pp=gains,
                fixed_iou_gain_pp=100*(new['scene_macro']['fixed_iou_0.5']-old['scene_macro']['fixed_iou_0.5']),
                scenes_improved=sum(v>1e-8 for v in gains.values()),scenes_regressed=sum(v< -1e-8 for v in gains.values()))
    parity={}
    for name in names[:2]:
        maximum=0.;count=0
        for file in (a.root/name/'predictions').glob('*.npy'):
            old=a.base/name/'predictions'/file.name
            if old.exists():
                maximum=max(maximum,float(np.max(np.abs(np.load(file)-np.load(old)))));count+=1
        parity[name]=dict(shared_query_count=count,maximum_absolute_difference=maximum)
    all_cases=[]
    index={n:{r['query_id']:r for r in d['per_query']} for n,d in result.items()}
    for row in result[names[2]]['per_query']:
        if not row['modality'].startswith('point_'):continue
        qid=row['query_id'];before=index[names[1]][qid]
        all_cases.append({k:row[k] for k in ['scene_id','instance_id','query_id','modality']}|dict(
            ap_gain_pp=100*(row['average_precision']-before['average_precision']),
            fixed_iou_gain_pp=100*(row['fixed_iou_0.5']-before['fixed_iou_0.5'])))
    summary=dict(scope='opened UQIS development; identical maps/evidence; no independent generalization claim',
        prediction_receipt=complete,benchmark_version=result[names[0]]['benchmark_version'],
        modalities={n:{m:v['scene_macro'] for m,v in d['modalities'].items()} for n,d in result.items()},
        gains_over=changes,control_parity=parity,
        decision='retain one-hop as optional ranking candidate; fixed-threshold selection regresses, no default promotion',
        primary_cases=all_cases)
    (a.root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['primary_cases','modalities']},indent=2))


if __name__=='__main__':main()
