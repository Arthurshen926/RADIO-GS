"""Repeat only the previously fixed strengths 0/2 on corrected source evidence."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);a=p.parse_args();root=Path('output/lerf_ropefix_multiscale_pipeline_20260917')/a.scene;field=root/'field';out=root/'clean_unified_extent_validation';out.mkdir(exist_ok=False);snap=out/'sources';snap.mkdir();r=dict(scene=a.scene,predeclared_strengths=[0,2],benchmark_selected_strength=False,stages=[],implementations={});marker=out/'execution.json'
for f in [Path(__file__),Path('scripts/audits/export_unified_bounded_extent.py'),Path('scripts/audits/evaluate_functional_queries.py'),Path('radio_gs/models/bounded_extent_readout.py')]:shutil.copy2(f,snap/f.name);r['implementations'][str(f)]=sha256_file(f)
commands=[('export',[sys.executable,'scripts/audits/run_lerf_source_stage.py','scripts/audits/export_unified_bounded_extent.py','--root',str(field),'--scene',a.scene,'--parent','source_precision_gate96','--prefix','clean_unified_extent'],[field/('clean_unified_extent'+str(i))/'report.json' for i in (0,2)])]
for task in ('text','reference'):
 commands.append((task,[sys.executable,'scripts/audits/evaluate_functional_queries.py','--task',task,'--root',str(field),'--scene',a.scene,'--geometry-report','output/geometry_only_repair_final_20260914/'+a.scene+'/report.json','--runs','clean_unified_extent0','clean_unified_extent2','--device',a.device,'--output',str(out/(task+'.json'))],[out/(task+'.json')]))
for name,cmd,outputs in commands:
 row=dict(name=name,command=cmd,status='running');r['stages'].append(row);marker.write_text(json.dumps(r,indent=2)+'\n')
 with (out/(name+'.log')).open('x') as log:proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 row.update(status='failed' if proc.returncode else 'complete',returncode=proc.returncode)
 if not proc.returncode:row['outputs']={str(p):sha256_file(p) for p in outputs}
 marker.write_text(json.dumps(r,indent=2)+'\n')
 if proc.returncode:raise RuntimeError(name+' failed')
old=json.loads((root/'precision_gate_validation/reference.json').read_text())['runs']['source_precision_gate96']['observations']
for run,v in json.loads((out/'reference.json').read_text())['runs'].items():
 assert len(v['observations'])==len(old)
 for x,y in zip(v['observations'],old):assert all(x[k]==y[k] for k in ('reference_frame','target_frame','category','prompt'))
r.update(finished=True,prompt_observations_exact=True);marker.write_text(json.dumps(r,indent=2)+'\n')
