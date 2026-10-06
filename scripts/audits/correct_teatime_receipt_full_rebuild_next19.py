"""Correct adopted-process metadata, preserving original receipts and model bytes."""
import json,shutil,time
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file
root=Path('output/full_field_rebuild_next19/teatime');top=root/'execution.json';base=root/'base/execution.json';recon=root/'reconstruction/execution.json';archive=root/'receipt_metadata_correction';archive.mkdir()
t=json.loads(top.read_text());assert t['complete'] is True
for p,n in [(top,'top'),(base,'base'),(recon,'reconstruction')]:shutil.copy2(p,archive/(n+'_before.json'))
b=json.loads(base.read_text());f=next(x for x in b['stages'] if x['name']=='field');assert f['status']=='complete'
rep=json.loads((root/'base/field/ropefix96_protected/report.json').read_text());assert rep['steps']==35856 and rep['geometry_binding_unchanged'];assert sha256_file(root/'base/field/ropefix96_protected/field.pt')==rep['field_sha256']
f.update(returncode=None,exit_code_unavailable_after_process_reparenting=True,completion_evidence={'steps':35856,'final_report_sha256':sha256_file(root/'base/field/ropefix96_protected/report.json'),'field_sha256':rep['field_sha256'],'no_traceback':True})
base.write_text(json.dumps(b,indent=2)+'\n')
r=json.loads(recon.read_text());g=next(x for x in r['stages'] if x['name']=='fit_grounded_semantic_transfer');g['runtime_override']['stage_execution_complete']=True;g['runtime_override']['complete_field_meaning']='historical recovery-queue snapshot, not this stage status';comp=Path('scripts/audits/grounded_cpu_full_rebuild_next19.source_sha256');g['runtime_override']['bound_original_source_sha256']=comp.read_text().strip();g['runtime_override']['source_hash_file_sha256']=sha256_file(comp);recon.write_text(json.dumps(r,indent=2)+'\n')
for row in t['stages']:
 if row['name']=='base':row['outputs'][str(base)]=sha256_file(base)
 if row['name']=='pixel_reconstruction':row['outputs'][str(recon)]=sha256_file(recon)
t['receipt_metadata_correction']=str(archive/'correction.json');top.write_text(json.dumps(t,indent=2)+'\n')
c={'reason':'Do not claim collected exit code for reparented completed training','model_weights_changed':False,'training_repeated':False,'before':{p.name:sha256_file(p) for p in archive.glob('*_before.json')},'after':{str(p):sha256_file(p) for p in [top,base,recon]},'time':time.time()};(archive/'correction.json').write_text(json.dumps(c,indent=2)+'\n')
print(json.dumps(c))
