"""Run the predeclared stage diagnostics after a complete pose-coverage rebuild."""
import argparse,json,subprocess,sys,time,shutil
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);a=p.parse_args()
root=Path('output/lerf_pose_covered_pipeline_20260917')/a.scene
while True:
    marker=root/'execution.json'
    if marker.exists():
        r=json.loads(marker.read_text())
        if any(s['status']=='failed' for s in r['stages']):raise RuntimeError('Upstream pipeline failed')
        if r.get('finished'):break
    time.sleep(10)
out=root/'diagnostics';out.mkdir(exist_ok=False);sources=out/'sources';sources.mkdir()
record=dict(pipeline_execution_sha256=sha256_file(marker),stages=[])
for name,entry,source,extra in [('native_compression','diagnose_current_native_compression',False,['--run','joint_supervised_image_residual']),('source_semantic_stages','diagnose_source_semantic_stages',True,[])]:
    script=Path('scripts/audits')/(entry+'.py');shutil.copy2(script,sources/script.name)
    result=out/(name+'.json');cmd=[sys.executable]+(['scripts/audits/run_lerf_source_stage.py'] if source else [])+[str(script),'--root',str(root/'field'),'--scene',a.scene,'--device',a.device,'--output',str(result),*extra]
    row=dict(name=name,command=cmd,implementation_sha256=sha256_file(script),status='running');record['stages'].append(row);(out/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
    with (out/(name+'.log')).open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    row.update(status='failed' if r.returncode else 'complete',returncode=r.returncode)
    if not r.returncode:row['output_sha256']=sha256_file(result)
    (out/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
    if r.returncode:raise RuntimeError(name+' failed')
record['finished']=True;(out/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
