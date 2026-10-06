"""Adopt running tea training without interrupting it; never enqueue waldo."""
import json,os,shutil,subprocess,sys,time
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file
root=Path('output/full_field_rebuild_next19');tea=root/'teatime';base=tea/'base';hook=Path('scripts/audits/dense_cpu_full_rebuild_next19.py');receipt=tea/'recovery_v5_adopted.json';rec=json.loads((tea/'recovery_v5.json').read_text());rec['adopted_field_pid']=2457979;rec['supervisor_handoff']='old waiting supervisor terminated; training untouched';rec['complete']=False
brec=json.loads((base/'execution.json').read_text());row=next(r for r in brec['stages'] if r['name']=='field');cmd=rec['execution_command']
def save():receipt.write_text(json.dumps(rec,indent=2)+'\n')
save()
while Path('/proc/2457979/cmdline').exists() and b'dense_cpu_full_rebuild_next19.py' in Path('/proc/2457979/cmdline').read_bytes():time.sleep(10)
report=base/'field/ropefix96_protected/report.json';assert report.exists(),'adopted training terminated without final report'
assert 'Traceback (most recent call last)' not in (tea/'dense_cpu_field_v5.log').read_text()
rec['field_returncode']=None;rec['exit_code_unavailable_after_process_reparenting']=True;rec['completion_evidence']='final report+no traceback+outputs SHA';save()
row.update(status='complete',returncode=None,exit_code_unavailable_after_process_reparenting=True,completion_evidence='full 35856 steps; final report; no traceback; output SHA',outputs={str(base/'field/ropefix96_protected/report.json'):sha256_file(base/'field/ropefix96_protected/report.json')},runtime_override={'command':cmd,'implementation_sha256':sha256_file(hook),'dense_cpu_selected_rows':True,'optimizer':'original unmodified Adam'})
row['implementations'][str(hook)]=sha256_file(hook)
(base/'execution.json').write_text(json.dumps(brec,indent=2)+'\n')
trec=json.loads((tea/'execution.json').read_text());assert trec['stages'][-1]['name']=='base' and trec['stages'][-1]['status']=='failed';trec['stages'].pop();(tea/'execution.json').write_text(json.dumps(trec,indent=2)+'\n');(tea/'base.log').unlink()
for scene in ['teatime']:
 cmd=[sys.executable,'scripts/audits/run_full_field_rebuild_next19.py','--scene',scene];r={'scene':scene,'command':cmd,'status':'running','start':time.time()};rec['stages'].append(r);save()
 with (root/(scene+'_recovery_driver.log')).open('x') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 r.update(status='complete' if p.returncode==0 else 'failed',returncode=p.returncode,seconds=time.time()-r['start']);save();assert p.returncode==0
rec['complete']=True;save()
