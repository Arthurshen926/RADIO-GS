"""Finish frozen evaluations serially on one user-assigned GPU.

Failures remain explicit and do not remove scenes from the requested cohort.
ScanNet rejected source candidates are evaluated, never silently promoted.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--wait-pid',type=int,required=True)
    p.add_argument('--lerf-scene',required=True);p.add_argument('--scannet-scenes',nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ['GPU-888bd14c-aa37-e1f4-dd2d-54385991d363','GPU-b3fc88ce-5b9d-dfaa-6d9b-2268a1394aab']
    a.output.mkdir(parents=True,exist_ok=False)
    record=dict(pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],state='waiting_for_lerf',complete=False,stages=[],failures=[])
    def save():
        tmp=a.output/'execution.tmp';tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(a.output/'execution.json')
    def run(name,entry,args):
        cmd=[sys.executable,entry,*map(str,args)];row=dict(name=name,command=cmd,status='running');record['stages'].append(row);record['state']=name;save()
        with (a.output/(name+'.log')).open('x') as log:ret=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        row.update(status='failed' if ret.returncode else 'complete',returncode=ret.returncode);save()
        if ret.returncode:raise RuntimeError(name+' failed')
    save()
    while Path(f'/proc/{a.wait_pid}').exists():time.sleep(15)
    try:
        scene=a.lerf_scene;base=Path('output/depth_opacity_pipeline_20260918')/scene
        evidence=Path('output/bounded_fallback_repair_20260918')/scene
        assert json.loads((evidence/'execution.json').read_text())['complete'],'LERF continuation did not finish'
        run('lerf2d','scripts/audits/evaluate_frozen_lerf2d_native_grid.py',['--scene',scene,'--device','cuda:0','--output',Path('output/frozen_method_lerf2d_20260918')/scene])
        selected=evidence/'selected_output'
        run('lerf3d_render','scripts/audits/evaluate_current_selected_lerf.py',['--root',base/'reconstruction/field',
            '--geometry-report',Path('output/depth_opacity_repair_20260918')/scene/'geometry_report.json','--scene',scene,
            '--run','bounded_fallback_admission','--encoding-bank',evidence/'text_online.encodings.pt',
            '--encoding-receipt',evidence/'text_online.routes.json','--device','cuda:0','--output',selected])
        from radio_gs.scripts.eval_opengaussian_lerf_baseline import evaluate_scene
        gt=Path('output/protocol_audit_20260801/vala/lerf3d_occam_geometry_v1/eval_gt')/scene/'gt'
        result=asdict(evaluate_scene(gt,selected,scene,threshold=10))
        assert result['missing']==0
        result.update(contract='lerf3d-field-only-four-scene-v1',independent_test=False,
            selected_report_sha256=sha256_file(selected/'report.json'),gt_files={str(p):sha256_file(p) for p in gt.rglob('*.jpg')})
        (evidence/'frozen_jpeg_evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    except Exception as error:
        record['failures'].append(dict(scene=a.lerf_scene,error=str(error)));save()
    for scene in a.scannet_scenes:
        try:
            run('build_'+scene,'scripts/audits/run_scannet_full_cohort.py',['--scene',scene,'--device','cuda:0','--resume'])
            root=Path('output/scannet_frozen_field_20260918')/scene
            run('evaluate_'+scene,'scripts/audits/evaluate_scannet_frozen_field.py',['--scene',scene,'--device','cuda:0','--output',root/'official_protocol_evaluation'])
            acceptance=json.loads((root/'execution.json').read_text())['source_acceptance']
            record.setdefault('source_acceptance',{})[scene]=acceptance;save()
        except Exception as error:
            record['failures'].append(dict(scene=scene,error=str(error)));save()
    record.update(complete=not record['failures'],state='finished_with_failures' if record['failures'] else 'complete');save()


if __name__=='__main__':main()
