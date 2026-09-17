"""Repeat the predeclared structural controls on corrected SAM source inputs."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);a=p.parse_args()
base=Path('output');root=base/'lerf_ropefix_multiscale_pipeline_20260917'/a.scene/'field';out=root.parent/'clean_bounded_structure_validation';out.mkdir(exist_ok=False);snap=out/'sources';snap.mkdir()
script=Path('scripts/audits/train_bounded_structure_exploration.py');sources=[Path(__file__),script,Path('scripts/audits/evaluate_functional_queries.py'),Path('scripts/audits/evaluate_real_reference_queries.py'),Path('radio_gs/models/scale_affinity_readout.py'),Path('radio_gs/models/hybrid_gaussian.py'),Path('radio_gs/training/geometry_relation_weight.py')]
r=dict(scene=a.scene,source_only_training=True,independent_final_test=False,comparison='same corrected sources, base field, two region passes, fixed seed, native prefix and geometry frozen; source-validation scale; no new benchmark-tuned variants',implementations={},stages=[]);marker=out/'execution.json'
for f in sources:shutil.copy2(f,snap/f.name);r['implementations'][str(f)]=sha256_file(f)
variants=[(v,[]) for v in ('explicit','mlp','hash','moge_output','moge_refiner','moge_encoder')]+[('relations_measured',['--geometry-relations','measured']),('relations_shuffled',['--geometry-relations','shuffled'])]
runs=[]
for name,extra in variants:
 variant='explicit' if name.startswith('relations_') else name;runname='clean_bounded_'+name;runs.append(runname)
 cmd=[sys.executable,'scripts/audits/run_lerf_source_stage.py',str(script),'--root',str(root),'--geometry',str(base/'geometry_only_repair_final_20260914'/a.scene/'independent_scale.pt'),'--parent','ropefix96_protected_imagecal_balanced','--name',runname,'--variant',variant,'--passes','2','--device',a.device,*extra]
 if variant.startswith('moge_'):cmd+=['--auxiliary-loss','cosine']
 if variant in ('moge_refiner','moge_encoder'):cmd+=['--moge-root',str(base/'moge_feature_teacher_20260917'/a.scene)]
 row=dict(name=name,command=cmd,status='running');r['stages'].append(row);marker.write_text(json.dumps(r,indent=2)+'\n');print(a.scene,name,flush=True)
 with (out/(name+'.log')).open('x') as log:proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 row.update(status='failed' if proc.returncode else 'complete',returncode=proc.returncode)
 if not proc.returncode:row['outputs']={str(root/runname/'report.json'):sha256_file(root/runname/'report.json'),str(root/runname/'field.pt'):sha256_file(root/runname/'field.pt')}
 marker.write_text(json.dumps(r,indent=2)+'\n')
 if proc.returncode:raise RuntimeError(name+' failed')
cmd=[sys.executable,'scripts/audits/evaluate_functional_queries.py','--task','reference','--root',str(root),'--scene',a.scene,'--geometry-report',str(base/'geometry_only_repair_final_20260914'/a.scene/'report.json'),'--runs','clean_bounded_explicit_direct_control',*runs,'--device',a.device,'--output',str(out/'reference.json')]
row=dict(name='reference',command=cmd,status='running');r['stages'].append(row);marker.write_text(json.dumps(r,indent=2)+'\n')
with (out/'reference.log').open('x') as log:proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
row.update(status='failed' if proc.returncode else 'complete',returncode=proc.returncode)
if not proc.returncode:row['outputs']={str(out/'reference.json'):sha256_file(out/'reference.json')}
marker.write_text(json.dumps(r,indent=2)+'\n')
if proc.returncode:raise RuntimeError('reference failed')
x=json.loads((out/'reference.json').read_text())['runs'];control=x['clean_bounded_explicit_direct_control']['observations']
for name in runs:
 for u,v in zip(control,x[name]['observations']):
  assert all(u[k]==v[k] for k in ('reference_frame','target_frame','category','text','image'))
r.update(finished=True,text_image_observations_exact=True,metrics={k:v['macro'] for k,v in x.items()});marker.write_text(json.dumps(r,indent=2)+'\n');print(a.scene,r['metrics'],flush=True)
