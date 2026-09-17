"""Aggregate all eight fixed tasks for each common branch, never scene-wise best."""
import argparse,json,shutil
from pathlib import Path
from statistics import mean
from radio_gs.utils.immutable_artifacts import sha256_file
TASKS=('fern','flower','fortress','trex','orchids','horns_center','horns_left','leaves')
RUNS=('nvos96_protected_imagecal_balanced','image_supervised_native_residual','nvos96_protected_imagecal_balanced_measured_support512','image_supervised_native_residual_measured_support512')

def aggregate(base):
    rows={};identities={};protocols={}
    for task in TASKS:
        root=base/('horns_center' if task=='horns_left' else task)
        marker=root/'execution.json'
        if not marker.exists() or not json.loads(marker.read_text()).get('finished'):raise RuntimeError('Incomplete full pipeline: '+task)
        rows[task]={}
        for mode in ('image','signed'):
            p=root/'evaluation'/(task+'_'+mode)/'report.json';r=json.loads(p.read_text())
            assert r['scene']==task and r['sam_reference_complex_rope_preserved']
            assert not r['target_rgb_opened'] and r['target_gt_used_after_prediction_only']
            assert r['signed_refinement']==(mode=='signed')
            assert mode!='signed' or r['signed_policy']=='adaptive'
            assert set(r['results'])==set(RUNS)
            protocol={k:r[k] for k in ('threshold','threshold_comparison','score_semantics','score_resize','signed_refinement','signed_policy','signed_bound_control')}
            if mode in protocols:assert protocol==protocols[mode]
            protocols[mode]=protocol
            for run in RUNS:
                assert sha256_file(root/'field'/run/'field.pt')==r['fields'][run]
                assert sha256_file(p.parent/(run+'.pt'))==r['results'][run]['prediction_sha256']
            identities[str(p)]=sha256_file(p);rows[task][mode]=r['results']
    metrics={}
    for mode in ('image','signed'):
        metrics[mode]={}
        for run in RUNS:
            m={k:mean(rows[t][mode][run][k] for t in TASKS) for k in ('iou','average_precision','pixel_accuracy','positive_support_recall')}
            m['reference_constraints_required_by_this_route']=(mode=='signed')
            m['reference_constraints_not_fully_satisfied']=[t for t in TASKS if any(v<1 for v in rows[t][mode][run]['reference_signed_agreement'].values())]
            metrics[mode][run]=m
    return dict(schema='clean_nvos_eight_task_summary.v1',independent_final_test=False,complete_joint_lerf_scannet_method=False,aggregation='equal task macro mean; both horns tasks intentionally included; no per-task branch selection',tasks=TASKS,protocols=protocols,reports=identities,metrics=metrics,per_task=rows,implementation_sha256=sha256_file(__file__))

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,default=Path('output/nvos_ropefix_pipeline_20260917'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    r=aggregate(a.base);a.output.write_text(json.dumps(r,indent=2)+'\n');shutil.copy2(__file__,a.output.with_suffix('.source.py'));print(json.dumps(r['metrics'],indent=2))
if __name__=='__main__':main()
