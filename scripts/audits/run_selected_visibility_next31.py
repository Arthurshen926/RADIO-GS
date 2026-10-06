"""Resume-safe source-only paired visibility-loss control."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from radio_gs.utils.immutable_artifacts import sha256_file

p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--base',type=Path,default=Path('output/selected_visibility_next31'));a=p.parse_args()
out=a.base
reg=json.loads((out/'registration.json').read_text())
entry=Path('scripts/audits/train_visibility_selected_scope_next31.py')
assert sha256_file(entry)==reg['implementation_sha256']
assert sha256_file('radio_gs/training/selected_output_supervision.py')==reg['visibility_implementation_sha256']
row=next(r for r in reg['scenes'] if r['scene']==a.scene)
root=Path(row['root'])
assert sha256_file(root/reg['parent_name']/'field.pt')==row['parent_field_sha256']
for mode in reg['modes']:
    name=mode+'_next31'
    receipt=out/a.scene/(mode+'.execution.json')
    cmd=[sys.executable,'scripts/audits/run_lerf_source_stage.py',str(entry),'--root',str(root),
         '--geometry-report',row['geometry_report'],'--device','cuda:0','--mode',mode,
         '--steps',str(reg['steps']),'--parent-name',reg['parent_name'],'--name',name]
    if receipt.exists():
        old=json.loads(receipt.read_text())
        assert old['command']==cmd and old['status']=='complete','Resolve incomplete attempt explicitly'
        assert sha256_file(root/name/'report.json')==old['report_sha256']
        continue
    record=dict(command=cmd,registration_sha256=sha256_file(out/'registration.json'),status='running')
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    with (out/a.scene/(mode+'.log')).open('x') as log:
        proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONPATH='.'))
    record.update(status='failed' if proc.returncode else 'complete',returncode=proc.returncode)
    if not proc.returncode:record['report_sha256']=sha256_file(root/name/'report.json')
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    if proc.returncode:raise RuntimeError(str(receipt))
    print(a.scene,mode,'complete',flush=True)
