"""Fail-closed full native-path reconstruction for one source-selected scene."""
import argparse,json,os,subprocess,sys,shutil
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file
parser=argparse.ArgumentParser();parser.add_argument('--scene',required=True);args=parser.parse_args()
scene=args.scene
if not os.environ.get('CUDA_VISIBLE_DEVICES'):
 raise RuntimeError('Explicit CUDA_VISIBLE_DEVICES required')
if not (Path('output/scannet_visibility_audit_next28')/scene/'report.json').exists():
 raise FileNotFoundError('Source-only visibility selection required')
root=Path('output/scannet_coverage_native_next28')/scene;marker=root/'execution.json';record=json.loads(marker.read_text());record['pid']=os.getpid()
def save():
 temp=marker.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(marker)
def run(name,command):
 command=list(map(str,command));row=dict(name=name,command=command,status='running');record['stages'].append(row);save()
 with (root/(name+'.log')).open('x') as log:ret=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
 row.update(status='failed' if ret.returncode else 'complete',returncode=ret.returncode);save()
 if ret.returncode:raise RuntimeError(scene+' '+name+' failed')
try:
 base=root/'base';old=Path('output/scannet_frozen_field_20260918')/scene;queries=old/'official_protocol_evaluation/queries.pt';equiv=Path('output/native_primary_readout_equivalence_20260919')/scene
 if not (equiv/'report.json').exists():run('equivalence',[sys.executable,'scripts/audits/materialize_native_source_readout.py','--root',old/'base/field','--queries',queries,'--reference',old/'official_protocol_evaluation/predictions.npz','--output',equiv])
 eq=json.loads((equiv/'report.json').read_text());assert not any(eq['reference_prediction_mismatches'].values());record['baseline_equivalence_report_sha256']=sha256_file(equiv/'report.json');save()
 guard='scripts/audits/run_scannet_source_stage.py';sam='/root/miniconda3/envs/sam3/bin/python';plan=json.loads((base/'source_plan.json').read_text())
 run('radio',[sys.executable,guard,'--module','radio_gs.scripts.extract_radio_features','--scene',scene,'--image_dir',root/'source_rgb','--output_dir',root/'radio','--radio_checkpoint','/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar','--batch_size','1','--skip_pca_stats','--resume-partial','--device','cuda:0'])
 command=[sam,guard,'--module','radio_gs.scripts.build_sam3_multiscale_hierarchy_cache','--source-authority',plan['source_authority']['path'],'--source-authority-sha256',plan['source_authority']['sha256'],'--image-ids',','.join(f'frame_{f:05d}' for f in plan['frames']),'--output-root',base/'source_sam','--checkpoint-path','checkpoints/sam3_modelscope/sam3.pt','--checkpoint-sha256','9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e','--device','cuda:0'];run('sam',command);snap=base/'sam_sources';snap.mkdir();implementations={}
 for name in ['radio_gs/scripts/build_sam3_multiscale_hierarchy_cache.py','radio_gs/scripts/build_sam3_foundation_cache.py','radio_gs/models/sam3_multiscale_hierarchy.py',guard]:shutil.copy2(name,snap/Path(name).name);implementations[name]=sha256_file(name)
 (base/'source_sam_execution.json').write_text(json.dumps(dict(status='complete',command=list(map(str,command)),implementations=implementations),indent=2)+'\n')
 run('native_inputs',[sys.executable,'scripts/audits/run_registered_mesh_native_probe.py','--scene',scene,'--device','cuda:0','--output-root',base,'--geometry-report',root/'geometry_report.json','--until-base'])
 run('native_materialization',[sys.executable,'scripts/audits/materialize_native_source_readout.py','--root',base/'field','--queries',queries,'--output',root/'native_readout'])
 run('evaluate',[sys.executable,'scripts/audits/evaluate_coverage_native_probe_next28.py','--scene',scene]);record.update(complete=True,evaluation_sha256=sha256_file(root/'evaluation.json'));save()
except Exception as err:
 record['error']=repr(err);save()
 raise
