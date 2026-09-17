"""Fixed extent-to-native-cutoff control, including source-accepted concepts."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);a=p.parse_args();torch.set_num_threads(4)
root=Path('output/pixel_evidence_rebuild_20260917')/a.scene;fieldroot=root/'field';parent=fieldroot/'joint_supervised_image_residual';pr=json.loads((parent/'report.json').read_text());assert sha256_file(parent/'field.pt')==pr['field_sha256'];payload=torch.load(parent/'field.pt',weights_only=False,map_location='cpu')
arch=dict(payload['architecture']);gt=dict(arch['grounded_transfer']);gt['identity_selection']=dict(gt['identity_selection'],completion_mode='native_threshold',calibrate_accepted_concepts=True);arch['grounded_transfer']=gt
name='all_concept_native_cutoff';out=fieldroot/name;out.mkdir(exist_ok=False);torch.save(dict(payload,architecture=arch),out/'field.pt')
report=dict(schema='all_concept_native_cutoff.v1',parent_field_sha256=pr['field_sha256'],field_sha256=sha256_file(out/'field.pt'),all_parent_tensors_exact=True,benchmark_data_used=False,source_only=True,policy='Selected extent calibrates cutoff for all accepted queries; original native element order retained; no guarantee for pixel ranking after nonlinear rendering.',implementation_sha256=sha256_file(__file__),readout_sha256=sha256_file('radio_gs/models/element_retrieval_field.py'));(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
v=root/'all_concept_cutoff_validation';v.mkdir();snap=v/'sources';snap.mkdir();r=dict(stages=[],implementations={})
for f in [Path(__file__),Path('radio_gs/models/element_retrieval_field.py'),Path('radio_gs/models/grounded_semantic_transfer.py'),Path('scripts/audits/evaluate_functional_queries.py')]:shutil.copy2(f,snap/f.name);r['implementations'][str(f)]=sha256_file(f)
for task in ['text','reference']:
 command=[sys.executable,'scripts/audits/evaluate_functional_queries.py','--task',task,'--root',str(fieldroot),'--scene',a.scene,'--geometry-report','output/geometry_only_repair_final_20260914/'+a.scene+'/report.json','--runs',name,'--device',a.device,'--output',str(v/(task+'.json'))];row=dict(name=task,command=command,status='running');r['stages'].append(row);(v/'execution.json').write_text(json.dumps(r,indent=2)+'\n')
 with (v/(task+'.log')).open('x') as log:proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
 row.update(status='failed' if proc.returncode else 'complete',returncode=proc.returncode)
 if not proc.returncode:row['output_sha256']=sha256_file(v/(task+'.json'))
 (v/'execution.json').write_text(json.dumps(r,indent=2)+'\n')
 if proc.returncode:raise RuntimeError(task+' failed')
old=json.loads((root/'reference_evaluation.json').read_text())['runs']['joint_supervised_image_residual'];new=json.loads((v/'reference.json').read_text())['runs'][name]
assert len(old['observations'])==len(new['observations'])
for x,y in zip(old['observations'],new['observations']):assert all(x[k]==y[k] for k in ('reference_frame','target_frame','category','image','prompt'))
r.update(finished=True,image_prompt_observations_exact=True);(v/'execution.json').write_text(json.dumps(r,indent=2)+'\n')
