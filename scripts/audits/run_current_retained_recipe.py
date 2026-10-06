"""Execute the declared retained recipe; refuse altered inputs and overwritten runs."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file

def main():
 p=argparse.ArgumentParser();p.add_argument('--recipe',type=Path,default=Path('configs/recipes/current_retained_96d.json'));p.add_argument('--scene',required=True);p.add_argument('--device',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 recipe=json.loads(a.recipe.read_text());scene=recipe['scenes'][a.scene]
 for f,h in scene['input_sha256'].items():
  if sha256_file(f)!=h:raise ValueError('retained recipe input changed: '+f)
 a.output.mkdir(parents=True,exist_ok=False)
 receipt={'recipe_sha256':sha256_file(a.recipe),'scene':a.scene,'training':False,'stages':[]}
 for task in ['text','reference']:
  for mode in ['online','replay']:
   out=a.output/(task+'_'+mode+'.json');cmd=[sys.executable,'scripts/audits/audit_current_query_routes.py','--task',task,'--root',scene['root'],'--scene',a.scene,'--geometry-report',scene['geometry_report'],'--runs',scene['run'],'--device',a.device,'--output',str(out)]
   if mode=='replay':cmd+=['--encoding-replay',str(a.output/(task+'_online.encodings.pt'))]
   row={'task':task,'mode':mode,'command':cmd,'status':'running'};receipt['stages'].append(row);(a.output/'execution.json').write_text(json.dumps(receipt,indent=2)+'\n')
   with (a.output/(task+'_'+mode+'.log')).open('x') as log:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONPATH='.'))
   row.update(status='complete' if result.returncode==0 else 'failed',returncode=result.returncode)
   if result.returncode==0:row['result_sha256']=sha256_file(out)
   (a.output/'execution.json').write_text(json.dumps(receipt,indent=2)+'\n')
   if result.returncode:raise RuntimeError('evaluation failed; inspect '+str(a.output))
 for f,h in scene['input_sha256'].items():assert sha256_file(f)==h
 receipt['complete']=True;(a.output/'execution.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
