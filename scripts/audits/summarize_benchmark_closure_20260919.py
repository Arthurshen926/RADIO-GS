"""Report full frozen cohorts; never turn a partial scene mean into a headline."""
import argparse,json
from pathlib import Path
import numpy as np
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    bindings={}
    def read(path):
        path=Path(path);bindings[str(path)]=sha256_file(path);return json.loads(path.read_text())
    lerf=['figurines','ramen','teatime','waldo_kitchen'];scan=['scene0000_00','scene0062_00','scene0070_00','scene0097_00','scene0140_00','scene0347_00','scene0400_00','scene0590_00']
    result=dict(independent_test=False,unified_complete_method_across_all_benchmarks=False,benchmarks={})
    rows={}
    for scene in lerf:
        path=Path('output/frozen_method_lerf2d_20260918')/scene/'report.json'
        if path.exists():
            d=read(path);rows[scene]={k:d[k] for k in ['miou','localization_accuracy','observation_count','field_sha256']}
    full=len(rows)==4
    if full:assert sum(r['observation_count'] for r in rows.values())==208
    result['benchmarks']['LERF2D']=dict(complete=full,missing=[s for s in lerf if s not in rows],scenes=rows,
        macro={k:float(np.mean([r[k] for r in rows.values()])) for k in ['miou','localization_accuracy']} if full else None)
    old=read('paper/artifacts/frozen_method_lerf3d_partial_20260918.json');rows={s:old['scenes'][s] for s in ['figurines','ramen']}
    for scene in lerf[2:]:
        path=Path('output/bounded_fallback_repair_20260918')/scene/'frozen_jpeg_evaluation.json'
        if path.exists():rows[scene]=read(path)
    rows={s:{k:d[k] for k in ['miou','acc025','acc05','count','missing']} for s,d in rows.items()};full=len(rows)==4
    if full:assert sum(r['count'] for r in rows.values())==208 and not any(r['missing'] for r in rows.values())
    result['benchmarks']['LERF3D']=dict(complete=full,missing=[s for s in lerf if s not in rows],scenes=rows,
        macro={k:float(np.mean([r[k] for r in rows.values()])) for k in ['miou','acc025','acc05']} if full else None)
    rows={}
    for scene in scan:
        root=Path('output/scannet_frozen_field_20260918')/scene;path=root/'official_protocol_evaluation/report.json'
        if not path.exists():continue
        d=read(path);fit=read(root/'reconstruction/field/bounded_fallback_withheld/report.json')
        assert sha256_file(root/'official_protocol_evaluation/predictions.npz')==d['prediction_sha256']
        rows[scene]=dict(metrics={s:{k:v[k] for k in ['miou','macc']} for s,v in d['metrics'].items()},
            scope_control={s:{k:v[k] for k in ['miou','macc']} for s,v in d['object_scope_control_metrics'].items()},source_acceptance=fit['source_acceptance'],field_sha256=d['field_sha256'])
    full=len(rows)==8
    result['benchmarks']['ScanNet_OVS']=dict(complete=full,missing=[s for s in scan if s not in rows],scenes=rows,
        macro={s:{k:float(np.mean([r['metrics'][s][k] for r in rows.values()])) for k in ['miou','macc']} for s in ['19','15','10']} if full else None,
        source_rejected_scenes=[s for s,r in rows.items() if not r['source_acceptance']])
    nvos=read('paper/artifacts/nvos_positive_field_eight_task_20260918.json')
    for f,h in nvos['bindings'].items():assert sha256_file(f)==h
    result['benchmarks']['NVOS']=dict(complete=nvos['complete_eight_tasks'],tasks=len(nvos['per_task']),macro=nvos['macro'],full_lerf_caption_method=False,method='shared frozen positive-only image branch')
    result['all_requested_cohorts_evaluated']=all(r['complete'] for r in result['benchmarks'].values());result['bindings']=bindings
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print({k:dict(complete=v['complete'],macro=v['macro']) for k,v in result['benchmarks'].items()})


if __name__=='__main__':main()
