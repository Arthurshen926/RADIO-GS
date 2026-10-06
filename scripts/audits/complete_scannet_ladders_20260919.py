"""CPU-only ladder audit as each frozen ScanNet scene finishes evaluation."""
import json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from radio_gs.utils.immutable_artifacts import sha256_file
root=Path('output/semantic_ladder_20260919');marker=root/'full_cohort_execution.json';assert not marker.exists();scenes=['scene0000_00','scene0062_00','scene0070_00','scene0347_00','scene0097_00','scene0400_00','scene0140_00','scene0590_00'];record=dict(pid=os.getpid(),cpu_only=True,complete=False,scenes={},failures=[])
def save():
 tmp=marker.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(marker)
save()
while len(record['scenes'])+len(record['failures'])<len(scenes):
 ready=False
 for scene in scenes:
  if scene in record['scenes'] or any(r['scene']==scene for r in record['failures']):continue
  base=Path('output/scannet_frozen_field_20260918')/scene/'official_protocol_evaluation'
  if not (base/'report.json').exists():continue
  ready=True;out=root/scene
  try:
   if not (out/'report.json').exists():
    with (root/(scene+'_ladder.log')).open('x') as log:subprocess.run([sys.executable,'scripts/audits/diagnose_scannet_semantic_ladder.py','--scene',scene,'--output',str(out)],stdout=log,stderr=subprocess.STDOUT,check=True)
   if not (out/'error_attribution.json').exists():
    with (root/(scene+'_attribution.log')).open('x') as log:subprocess.run([sys.executable,'scripts/audits/diagnose_scannet_missing_vs_confusion.py','--scene',scene,'--output',str(out/'error_attribution.json')],stdout=log,stderr=subprocess.STDOUT,check=True)
   pred=np.load(out/'element_predictions.npz');differences={k:int((pred[k]!=np.load(base/f)['19']).sum()) for k,f in [('compact_fusion','predictions.npz'),('final_scope','object_scope_control.npz')]};assert not any(differences.values()),differences
   record['scenes'][scene]=dict(report_sha256=sha256_file(out/'report.json'),attribution_sha256=sha256_file(out/'error_attribution.json'),prediction_mismatches=differences)
  except Exception as err:record['failures'].append(dict(scene=scene,error=str(err)))
  save()
 if not ready:
  closure=json.loads(Path('output/benchmark_closure_20260919/liveness.json').read_text())
  if not any(r['alive'] for r in closure['workers']):break
  time.sleep(30)
record['complete']=len(record['scenes'])==len(scenes) and not record['failures'];record['missing']=[s for s in scenes if s not in record['scenes']];save()
