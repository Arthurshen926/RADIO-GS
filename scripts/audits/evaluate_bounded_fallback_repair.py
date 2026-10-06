"""Complete cold development evaluation of the source-selected fallback repair."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--positive-only',action='store_true');a=p.parse_args()
 base=Path('output/depth_opacity_pipeline_20260918')/a.scene;field=base/'reconstruction/field';gp=Path('output/depth_opacity_repair_20260918')/a.scene/'geometry_report.json'
 run='positive_fallback_admission' if a.positive_only else 'bounded_fallback_admission'
 fitname='positive_fallback_source_control' if a.positive_only else 'bounded_fallback_withheld'
 root=Path('output/positive_fallback_repair_20260918' if a.positive_only else 'output/bounded_fallback_repair_20260918')/a.scene;root.mkdir(parents=True,exist_ok=False)
 report=json.loads((field/run/'report.json').read_text());assert report['withheld_fallback_supervised']
 fitted=json.loads((field/fitname/'report.json').read_text());assert fitted['source_acceptance']
 record=dict(scene=a.scene,complete=False,source_selected=True,stages=[],field_sha256=report['field_sha256'],fit_report_sha256=sha256_file(field/fitname/'report.json'))
 lognames=([f'positive_fallback_{a.scene}.log',f'positive_admission_{a.scene}.log'] if a.positive_only else [f'bounded_{label}_{a.scene}.log' for label in ['scope','withheld','admission']])
 for logname in lognames:
  path=Path('/tmp')/logname
  if path.exists():shutil.copy2(path,root/path.name)
 snap=root/'sources';snap.mkdir()
 for f in [Path(__file__),Path('radio_gs/models/element_retrieval_field.py'),Path('radio_gs/models/bounded_scope_evidence.py')]:shutil.copy2(f,snap/f.name)
 for task in ['text','reference']:
  for mode in ['online','replay']:
   out=root/(task+'_'+mode+'.json');cmd=[sys.executable,'scripts/audits/audit_current_query_routes.py','--task',task,'--root',str(field),'--scene',a.scene,'--geometry-report',str(gp),'--runs',run,'--device','cuda:0','--output',str(out)]
   if mode=='replay':cmd+=['--encoding-replay',str(root/(task+'_online.encodings.pt'))]
   row=dict(task=task,mode=mode,command=cmd,status='running');record['stages'].append(row);(root/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
   with (root/(task+'_'+mode+'.log')).open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4'))
   row.update(returncode=r.returncode,status='complete' if not r.returncode else 'failed');(root/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
   if r.returncode:raise RuntimeError(str(out))
   row['output_sha256']=sha256_file(out)
  online=json.loads((root/(task+'_online.json')).read_text());replay=json.loads((root/(task+'_replay.json')).read_text());assert online['runs']==replay['runs']
  old=json.loads((base/(task+'_online.json')).read_text())['runs']['counterfactual_extent_benefit_admission'];new=online['runs'][run]
  if task=='text': print(a.scene,'text',old['observation_miou'],new['observation_miou'],flush=True)
  else: print(a.scene,'reference',old['macro'],new['macro'],flush=True)
 record['complete']=True;(root/'execution.json').write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__':main()
