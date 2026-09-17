"""Close the existing shared-native residual control on corrected teacher inputs."""
import argparse,json,subprocess,sys,time,shutil
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);a=p.parse_args()
root=Path('/mnt/pool/sqy/results/RADIO-GS/output/lerf_ropefix_multiscale_pipeline_20260917')/a.scene;field=root/'field';training=field/'joint_native_retrieval_residual/report.json'
for _ in range(720):
 if training.exists():break
 log=Path('/tmp')/(a.scene+'_clean_joint_residual.log')
 if log.exists() and 'Traceback (most recent call last)' in log.read_text():raise RuntimeError('source residual fitting failed')
 time.sleep(5)
else:raise TimeoutError('source fitting did not finish in one hour')
r=json.loads(training.read_text());assert sha256_file(training.parent/'field.pt')==r['field_sha256']
out=root/'clean_joint_validation';out.mkdir(exist_ok=False);snap=out/'sources';snap.mkdir();identities={}
for name,digest in r['implementation_sources'].items():
 src=Path(name);assert sha256_file(src)==digest;shutil.copy2(src,snap/src.name);identities[name]=digest
record=dict(training_report_sha256=sha256_file(training),training_sources=identities,stages=[],finished=False);marker=out/'execution.json'
def stage(name,entry,args,*,source=True,outputs):
 sources=[Path(__file__),Path(entry),Path('scripts/audits/run_lerf_source_stage.py')];directory=snap/name;directory.mkdir();ids={}
 for f in sources:shutil.copy2(f,directory/f.name);ids[str(f)]=sha256_file(f)
 cmd=[sys.executable]+(['scripts/audits/run_lerf_source_stage.py'] if source else [])+[entry]+list(map(str,args));row=dict(name=name,command=cmd,implementations=ids,status='running');record['stages'].append(row);marker.write_text(json.dumps(record,indent=2)+'\n')
 with (out/(name+'.log')).open('x') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 row.update(returncode=p.returncode,status='failed' if p.returncode else 'complete')
 if not p.returncode:row['outputs']={str(f):sha256_file(f) for f in outputs}
 marker.write_text(json.dumps(record,indent=2)+'\n')
 if p.returncode:raise RuntimeError(name+' failed')
 print(a.scene,name,'complete',flush=True)
for parent in ['source_precision_gate96','joint_native_retrieval_residual']:
 stage('fallback_'+parent,'scripts/audits/export_native_fallback_control.py',['--root',field,'--parent',parent],outputs=[field/(parent+'_native_fallback_control')/'report.json'])
name='joint_native_recalibrated'
stage('recalibrate','scripts/audits/fit_exact_render_query_threshold.py',['--root',field,'--scene',a.scene,'--device',a.device,'--parent','joint_native_retrieval_residual','--output-name',name],outputs=[field/name/'report.json'])
fp=field/name/'field.pt'
stage('cold','scripts/audits/cold_three_query_native.py',['--fixture',field/'cold_query_fixture.pt','--field',fp,'--field-sha256',sha256_file(fp),'--device',a.device,'--require-concept-route-consistency','--output',out/'cold.json'],outputs=[out/'cold.json'])
for task in ['text','reference']:
 runs=['source_precision_gate96','joint_native_retrieval_residual','joint_native_recalibrated']
 if task=='text':runs+=['source_precision_gate96_native_fallback_control','joint_native_retrieval_residual_native_fallback_control']
 stage(task,'scripts/audits/evaluate_functional_queries.py',['--task',task,'--root',field,'--scene',a.scene,'--geometry-report',Path('/mnt/pool/sqy/results/RADIO-GS/output/geometry_only_repair_final_20260914')/a.scene/'report.json','--runs',*runs,'--device',a.device,'--output',out/(task+'.json')],source=False,outputs=[out/(task+'.json'),out/(task+'.protocol.json'),out/(task+'.dispatch.json')])
record['finished']=True;marker.write_text(json.dumps(record,indent=2)+'\n')
