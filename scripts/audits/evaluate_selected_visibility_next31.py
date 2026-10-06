"""Complete two-arm LERF evaluations with frozen rendering and query contracts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from radio_gs.utils.immutable_artifacts import sha256_file

p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args()
base=Path('output/selected_visibility_next31_v2')
registration=base/'evaluation_registration.json'
reg=json.loads(registration.read_text())
entry='scripts/audits/evaluate_full_field_rebuild_next19.py'
assert sha256_file(entry)==reg['evaluator_sha256']
for mode in reg['modes']:
    train=json.loads((base/a.scene/(mode+'.execution.json')).read_text())
    assert train['status']=='complete'
    report=base/a.scene/'reconstruction/field'/(mode+'_next31')/'report.json'
    assert sha256_file(report)==train['report_sha256']
    cmd=[sys.executable,entry,'--scene',a.scene,'--base-root',str(base),'--device','cuda:0',
         '--field-name',mode+'_next31','--evaluation-name','evaluation_'+mode]
    receipt=base/a.scene/('evaluate_'+mode+'.execution.json')
    if receipt.exists():
        old=json.loads(receipt.read_text());assert old['status']=='complete' and old['command']==cmd
        assert sha256_file(base/a.scene/('evaluation_'+mode)/'report.json')==old['report_sha256']
        continue
    record=dict(command=cmd,status='running',registration_sha256=sha256_file(registration))
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    with (base/a.scene/('evaluate_'+mode+'.log')).open('x') as log:
        result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONPATH='.'))
    record.update(status='failed' if result.returncode else 'complete',returncode=result.returncode)
    if not result.returncode:record['report_sha256']=sha256_file(base/a.scene/('evaluation_'+mode)/'report.json')
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    if result.returncode:raise RuntimeError(str(receipt))
    print(a.scene,mode,'evaluation complete',flush=True)
