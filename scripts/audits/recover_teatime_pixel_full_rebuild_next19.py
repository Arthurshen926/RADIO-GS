import json,subprocess,shutil,sys
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file
root=Path('output/full_field_rebuild_next19/teatime');recon=root/'reconstruction';archive=root/'failed_pixel_grounded_probability_oom';archive.mkdir();marker=recon/'execution.json';rec=json.loads(marker.read_text());row=next(x for x in rec['stages'] if x['name']=='fit_grounded_semantic_transfer');assert row['status']=='failed'
for p in [marker,root/'execution.json',recon/'fit_grounded_semantic_transfer.log',root/'pixel_reconstruction.log']:shutil.copy2(p,archive/('recon_'+p.name if p.parent==recon else p.name))
hook=Path('scripts/audits/grounded_cpu_full_rebuild_next19.py');cmd=[sys.executable,str(hook),*row['command'][1:]]
r={'complete':False,'command':cmd,'implementation_sha256':sha256_file(hook)};receipt=root/'pixel_memory_recovery.json'
def save():receipt.write_text(json.dumps(r,indent=2)+'\n')
save()
with (root/'grounded_cpu_recovery.log').open('x') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
r['grounded_returncode']=p.returncode;save();assert p.returncode==0
out=recon/'field/grounded_semantic_transfer96/report.json';row.update(status='complete',returncode=0,outputs={str(out):sha256_file(out)},runtime_override=r.copy());row['implementations'][str(hook)]=sha256_file(hook);shutil.copy2(hook,recon/'sources/fit_grounded_semantic_transfer'/hook.name);marker.write_text(json.dumps(rec,indent=2)+'\n')
top=json.loads((root/'execution.json').read_text());failed=top['stages'][-1];assert failed['name']=='pixel_reconstruction' and failed['status']=='failed';cmd=failed['command']+['--resume'];r['resume_command']=cmd;save()
with (root/'pixel_reconstruction_resume.log').open('x') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
r['pixel_resume_returncode']=p.returncode;save();assert p.returncode==0
failed.update(status='complete',returncode=0,outputs={str(marker):sha256_file(marker)},recovery_command=cmd);(root/'execution.json').write_text(json.dumps(top,indent=2)+'\n')
with (root/'remaining_pipeline.log').open('x') as log:p=subprocess.run([sys.executable,'scripts/audits/run_full_field_rebuild_next19.py','--scene','teatime'],stdout=log,stderr=subprocess.STDOUT)
r['remaining_returncode']=p.returncode;r['complete']=p.returncode==0;save();assert p.returncode==0
