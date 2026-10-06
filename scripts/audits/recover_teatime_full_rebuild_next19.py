"""Preserve first OOM attempt; replay identical field task using sequential Adam."""
import json,os,shutil,subprocess,sys,time
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file
root=Path('output/full_field_rebuild_next19');tea=root/'teatime';base=tea/'base';archive=tea/'failed_attempt1_foreach_oom';archive.mkdir()
for src in [tea/'execution.json',tea/'base.log',base/'execution.json',base/'field.log']:
 dst=archive/('base_'+src.name if src.parent==base else src.name);shutil.copy2(src,dst)
brec=json.loads((base/'execution.json').read_text());row=next(r for r in brec['stages'] if r['name']=='field');assert row['status']=='failed'
hook=Path('scripts/audits/adam_sequential_full_rebuild_next19.py');cmd=[sys.executable,str(hook),*row['command'][1:]]
receipt=tea/'recovery.json';rec={'reason':'first Adam foreach temporary OOM before successful optimizer step','original_command':row['command'],'execution_command':cmd,'hook_sha256':sha256_file(hook),'verification_sha256':sha256_file(root/'adam_sequential_verification.json'),'stages':[],'complete':False}
def save():receipt.write_text(json.dumps(rec,indent=2)+'\n')
save();shutil.copy2(hook,base/'sources/field'/hook.name)
with (tea/'sequential_field.log').open('x') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
rec['field_returncode']=p.returncode;save();assert p.returncode==0
row.update(status='complete',returncode=0,outputs={str(base/'field/ropefix96_protected/report.json'):sha256_file(base/'field/ropefix96_protected/report.json')},runtime_override={'command':cmd,'implementation_sha256':sha256_file(hook),'foreach':False})
row['implementations'][str(hook)]=sha256_file(hook)
(base/'execution.json').write_text(json.dumps(brec,indent=2)+'\n')
trec=json.loads((tea/'execution.json').read_text());assert trec['stages'][-1]['name']=='base' and trec['stages'][-1]['status']=='failed';trec['stages'].pop();(tea/'execution.json').write_text(json.dumps(trec,indent=2)+'\n');(tea/'base.log').unlink()
for scene in ['teatime','waldo_kitchen']:
 cmd=[sys.executable,'scripts/audits/run_full_field_rebuild_next19.py','--scene',scene];r={'scene':scene,'command':cmd,'status':'running','start':time.time()};rec['stages'].append(r);save()
 with (root/(scene+'_recovery_driver.log')).open('x') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 r.update(status='complete' if p.returncode==0 else 'failed',returncode=p.returncode,seconds=time.time()-r['start']);save();assert p.returncode==0
rec['complete']=True;save()
