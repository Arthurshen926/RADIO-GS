"""Cold decision consistency and complete real-query validation of a source fix."""
import argparse,json,subprocess,sys,shutil
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);p.add_argument('--recipe',default='lerf_view_balanced_rebuild_20260917',choices=['lerf_view_balanced_rebuild_20260917','lerf_ropefix_multiscale_pipeline_20260917']);p.add_argument('--candidate',default='source_precision_gate96');p.add_argument('--validation-directory',default='precision_gate_validation');a=p.parse_args()
root=Path('/mnt/pool/sqy/results/RADIO-GS/output')/a.recipe/a.scene;field=root/'field';candidate=field/a.candidate;receipt=json.loads((candidate/'report.json').read_text());fp=candidate/'field.pt';assert sha256_file(fp)==receipt['field_sha256']
out=root/a.validation_directory;out.mkdir(exist_ok=False);snap=out/'sources';snap.mkdir()
sources=[Path(__file__),Path('scripts/audits/cold_three_query_native.py'),Path('scripts/audits/evaluate_functional_queries.py'),Path('scripts/audits/evaluate_real_reference_queries.py'),Path('scripts/audits/evaluate_frozen_field_lerf_labels.py'),Path('radio_gs/models/grounded_semantic_transfer.py'),Path('radio_gs/models/element_retrieval_field.py')]
identities={}
for src in sources:shutil.copy2(src,snap/src.name);identities[str(src)]=sha256_file(src)
record=dict(source_repair_report_sha256=sha256_file(candidate/'report.json'),implementations=identities,stages=[],finished=False);marker=out/'execution.json'
commands=[('cold',[sys.executable,'scripts/audits/run_lerf_source_stage.py','scripts/audits/cold_three_query_native.py','--fixture',str(field/'cold_query_fixture.pt'),'--field',str(fp),'--field-sha256',receipt['field_sha256'],'--device',a.device,'--require-concept-route-consistency','--output',str(out/'cold.json')])]
for task in ['text','reference']:
 commands.append((task,[sys.executable,'scripts/audits/evaluate_functional_queries.py','--task',task,'--root',str(field),'--scene',a.scene,'--geometry-report',str(Path('/mnt/pool/sqy/results/RADIO-GS/output/geometry_only_repair_final_20260914')/a.scene/'report.json'),'--runs',a.candidate,'--device',a.device,'--output',str(out/(task+'.json'))]))
for name,cmd in commands:
 row=dict(name=name,command=cmd,status='running');record['stages'].append(row);marker.write_text(json.dumps(record,indent=2)+'\n')
 with (out/(name+'.log')).open('x') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 row.update(status='complete' if r.returncode==0 else 'failed',returncode=r.returncode)
 if not r.returncode:row['outputs']={str(x):sha256_file(x) for x in out.glob(name+'*.json')}
 marker.write_text(json.dumps(record,indent=2)+'\n')
 if r.returncode:raise RuntimeError(name+' failed; inspect source fix without bypassing cold gate')
 print(a.scene,name,'complete',flush=True)
record['finished']=True;marker.write_text(json.dumps(record,indent=2)+'\n')
