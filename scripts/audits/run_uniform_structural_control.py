"""Four fixed clean-field prompt controls; no model/scale selection on labels."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);a=p.parse_args();torch.set_num_threads(4)
base=Path('output');new=base/'lerf_pose_covered_pipeline_20260917'/a.scene
while True:
    path=new/'diagnostics/execution.json'
    if path.exists():
        x=json.loads(path.read_text())
        if any(s['status']=='failed' for s in x['stages']):raise RuntimeError('Pose diagnostics failed')
        if x.get('finished'):break
    time.sleep(10)
for recipe,dirname in [('matched','lerf_ropefix_multiscale_pipeline_20260917'),('pose','lerf_pose_covered_pipeline_20260917')]:
    root=base/dirname/a.scene;fieldroot=root/'field';out=root/'uniform_structural_validation';out.mkdir(exist_ok=False);snap=out/'sources';snap.mkdir()
    entries=[Path(__file__),Path('scripts/audits/export_uniform_structural_affinity.py'),Path('scripts/audits/evaluate_functional_queries.py'),Path('scripts/audits/evaluate_real_reference_queries.py'),Path('radio_gs/models/scale_affinity_readout.py')]
    r=dict(scene=a.scene,recipe=recipe,implementations={},stages=[]);marker=out/'execution.json'
    for f in entries:shutil.copy2(f,snap/f.name);r['implementations'][str(f)]=sha256_file(f)
    control=fieldroot/'uniform_structural_affinity';parent=fieldroot/'joint_supervised_image_residual'
    if not control.exists():
        cmd=[sys.executable,'scripts/audits/run_lerf_source_stage.py','scripts/audits/export_uniform_structural_affinity.py','--root',str(fieldroot)]
        with (out/'export.log').open('x') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    rec=json.loads((control/'report.json').read_text());assert sha256_file(control/'field.pt')==rec['field_sha256'];assert sha256_file(parent/'field.pt')==rec['parent_field_sha256'];assert sha256_file(control/'source.py')==rec['implementation_sha256']
    ps=torch.load(parent/'field.pt',weights_only=False,map_location='cpu')['state_dict'];cs=torch.load(control/'field.pt',weights_only=False,map_location='cpu')['state_dict'];assert ps.keys()==cs.keys()
    assert all(torch.equal(ps[k],cs[k]) for k in ps if k!='affinity.cutoffs');assert not cs['affinity.cutoffs'].any();del ps,cs
    r['control_report_sha256']=sha256_file(control/'report.json');r['all_other_state_exact']=True
    cmd=[sys.executable,'scripts/audits/evaluate_functional_queries.py','--task','reference','--root',str(fieldroot),'--scene',a.scene,'--geometry-report',str(base/'geometry_only_repair_final_20260914'/a.scene/'report.json'),'--runs','uniform_structural_affinity','--device',a.device,'--output',str(out/'reference.json')]
    row=dict(name='reference',command=cmd,status='running');r['stages'].append(row);marker.write_text(json.dumps(r,indent=2)+'\n')
    with (out/'reference.log').open('x') as log:run=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    row.update(status='failed' if run.returncode else 'complete',returncode=run.returncode)
    if not run.returncode:row['output_sha256']=sha256_file(out/'reference.json')
    marker.write_text(json.dumps(r,indent=2)+'\n')
    if run.returncode:raise RuntimeError('reference failed')
    oldpath=root/'scoped_joint_image_validation/reference.json' if recipe=='matched' else root/'reference_evaluation.json'
    before=json.loads(oldpath.read_text())['runs']['joint_supervised_image_residual'];after=json.loads((out/'reference.json').read_text())['runs']['uniform_structural_affinity'];assert len(before['observations'])==len(after['observations'])
    for u,v in zip(before['observations'],after['observations']):
        for k in ('reference_frame','target_frame','category','text','image'):assert u[k]==v[k],(recipe,k)
    r.update(finished=True,text_image_observations_exact=True,parent_evaluation_sha256=sha256_file(oldpath),before_macro=before['macro'],after_macro=after['macro']);marker.write_text(json.dumps(r,indent=2)+'\n');print(recipe,a.scene,before['macro']['prompt'],after['macro']['prompt'],flush=True)
