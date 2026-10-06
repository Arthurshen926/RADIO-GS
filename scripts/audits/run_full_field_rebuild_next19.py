"""Complete semantic map rebuild and cold/replay evaluation, isolated from frozen maps."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args()
root=Path('output/full_field_rebuild_next19')/a.scene;root.mkdir(parents=True,exist_ok=True);gp=Path('output/depth_opacity_repair_20260918')/a.scene/'geometry_report.json';base=root/'base';recon=root/'reconstruction';field=recon/'field';marker=root/'execution.json';record=json.loads(marker.read_text()) if marker.exists() else {'scene':a.scene,'complete':False,'geometry_report_sha256':sha256_file(gp),'full_source_asset_and_field_rebuild':True,'stages':[]}
assert record['geometry_report_sha256']==sha256_file(gp)
def save():marker.write_text(json.dumps(record,indent=2)+'\n')
def run(name,entry,args,outputs,source=False):
 cmd=[sys.executable]+(['scripts/audits/run_lerf_source_stage.py'] if source else [])+[entry]+list(map(str,args));old=next((r for r in record['stages'] if r['name']==name),None)
 if old:
  assert old['status']=='complete' and old['command']==cmd,(name,'failed attempt must be inspected and preserved before retry')
  for f,h in old['outputs'].items():assert sha256_file(f)==h
  return
 row={'name':name,'command':cmd,'status':'running','implementation_sha256':sha256_file(entry)};record['stages'].append(row);save();print(a.scene,name,flush=True)
 with (root/(name+'.log')).open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4'))
 row.update(returncode=r.returncode,status='complete' if r.returncode==0 else 'failed')
 if r.returncode==0:row['outputs']={str(f):sha256_file(f) for f in outputs}
 save();assert r.returncode==0,(name,str(root/(name+'.log')))
run('prepare','scripts/audits/prepare_full_field_rebuild_next19.py',['--scene',a.scene],[base/'source_plan.json',base/'source_sam_execution.json'])
run('base','scripts/audits/run_pose_covered_lerf_pipeline.py',['--scene',a.scene,'--device','cuda:0','--output-root',base,'--geometry-report',gp,'--until-base'],[base/'execution.json'])
run('pixel_reconstruction','scripts/audits/run_pixel_evidence_rebuild.py',['--scene',a.scene,'--device','cuda:0','--parent',base/'field','--output-root',recon,'--geometry-report',gp,'--skip-evaluation'],[recon/'execution.json'])
run('initial_admission','scripts/audits/fit_counterfactual_extent_admission.py',['--root',field,'--scene',a.scene,'--device','cuda:0'],[field/'counterfactual_extent_benefit_admission/report.json'],source=True)
run('bounded_scope','scripts/audits/fit_bounded_fallback_scope.py',['--root',field,'--device','cuda:0','--withheld-concepts'],[field/'bounded_fallback_withheld/report.json'],source=True)
record['source_acceptance']=json.loads((field/'bounded_fallback_withheld/report.json').read_text())['source_acceptance'];save()
run('final_admission','scripts/audits/fit_counterfactual_extent_admission.py',['--root',field,'--scene',a.scene,'--device','cuda:0','--parent','bounded_fallback_withheld','--name','bounded_fallback_admission','--withheld-fallback'],[field/'bounded_fallback_admission/report.json'],source=True)
for task in ['text','reference']:
 for mode in ['online','replay']:
  dst=root/(task+'_'+mode+'.json');args=['--task',task,'--root',field,'--scene',a.scene,'--geometry-report',gp,'--runs','bounded_fallback_admission','--device','cuda:0','--output',dst]
  if mode=='replay':args+=['--encoding-replay',root/(task+'_online.encodings.pt')]
  run(task+'_'+mode,'scripts/audits/audit_current_query_routes.py',args,[dst])
 assert json.loads((root/(task+'_online.json')).read_text())['runs']==json.loads((root/(task+'_replay.json')).read_text())['runs']
record['complete']=True;record['promoted']=False;record['independent_test']=False;save();print(a.scene,'COMPLETE',flush=True)
