"""Extend the existing semantic sources within the predeclared source32 cohort.

Selection uses only registered source cameras, retains the original train/val
split, and never opens benchmark annotations, metrics, or query images.
"""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
import torch
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);p.add_argument('--train-views',type=int,default=16);p.add_argument('--continue-pipeline',action='store_true');p.add_argument('--reuse-sam-root',type=Path);a=p.parse_args();torch.set_num_threads(4)
base=Path('/mnt/pool/sqy/results/RADIO-GS/output');parent=base/'lerf_ropefix_multiscale_pipeline_20260917'/a.scene;root=base/'lerf_pose_covered_pipeline_20260917'/a.scene;root.mkdir(parents=True,exist_ok=False)
info=json.loads((parent/'field/teachers.json').read_text());m=json.loads((parent/'source_sam/manifest.json').read_text());authority=Path(m['source_authority']['path']);assert sha256_file(authority)==m['source_authority']['sha256'];auth=json.loads(authority.read_text());assert not auth['information_policy']['benchmark_ground_truth_used'] and not auth['information_policy']['query_text_used']
gp=base/'geometry_only_repair_final_20260914'/a.scene/'report.json';g=json.loads(gp.read_text());prior=MogeGeometryPrior(g['source_report'],g['source_report_sha256'])
radio=base/'canonical_teacher_features'/('figurines' if a.scene=='figurines' else 'ramen_fix4c_exact120_v1');radio_manifest=json.loads((radio/'frame_manifest.json').read_text());radio_by_frame={r['frame_idx']:r for r in radio_manifest['frames']}
by_frame={int(r['image_id'].split('_')[-1]):r for r in auth['images']};available=sorted(set(by_frame)&set(prior.cameras)&set(radio_by_frame)&set(g['mapping_frames']));validation=info['validation_frames'];selected=list(info['train_frames']);assert set(selected+validation)<=set(available);eligible=[f for f in available if f not in validation];assert len(selected)<=a.train_views<=len(eligible)
poses=torch.stack([prior.cameras[f].world_to_camera for f in eligible]);centers=-torch.einsum('nij,nj->ni',poses[:,:3,:3].transpose(1,2),poses[:,:3,3]);diameter=torch.cdist(centers,centers).max().clamp_min(1e-8)
# Translation and viewing direction are dimensionless and independently bounded.
# Roll alone does not qualify an otherwise identical viewpoint as a new view.
distance=torch.cdist(centers,centers).square()/diameter.square()+torch.cdist(poses[:,2,:3],poses[:,2,:3]).square()
added=[]
while len(selected)<a.train_views:
 indices=[eligible.index(f) for f in selected];nearest=distance[:,indices].amin(1);nearest[indices]=-1;index=int(nearest.argmax());selected.append(eligible[index]);added.append(dict(frame=eligible[index],minimum_pose_distance=float(nearest[index].sqrt())))
frames=sorted(selected)+validation
for f in frames:
 src=by_frame[f];assert radio_by_frame[f]['source_file']==Path(src['path']).name and radio_by_frame[f]['source_sha256']==src['sha256'];assert sha256_file(Path(src['path']))==src['sha256']
plan=dict(scene=a.scene,frames=frames,train_frames=sorted(selected),validation_frames=validation,original_train_frames=info['train_frames'],added=added,eligible_frames=eligible,source_authority=m['source_authority'],radio_root=str(radio),radio_manifest_sha256=sha256_file(radio/'frame_manifest.json'),geometry_report_sha256=sha256_file(gp),benchmark_data_used=False,mapping_split_enforced=True,excluded_source32_nonmapping_frames=sorted(set(by_frame)-set(g['mapping_frames'])),selection='retain original sources, then farthest source32 camera by squared normalized translation plus squared unit viewing-direction distance; deterministic frame-id tie break',implementation_sha256=sha256_file(__file__))
(root/'source_plan.json').write_text(json.dumps(plan,indent=2)+'\n');print(a.scene,'source plan',frames,flush=True)
old_execution=json.loads((parent/'source_sam_execution.json').read_text());command=old_execution['command'].copy();command[command.index('--output-root')+1]=str(root/'source_sam');command[command.index('--image-ids')+1]=','.join(f'frame_{f:05d}' for f in frames);command[command.index('--device')+1]=a.device;command+=['--skip-existing']
sam=root/'source_sam';sam.mkdir();reused={}
reuse_manifest=m if a.reuse_sam_root is None else json.loads((a.reuse_sam_root/'manifest.json').read_text())
assert reuse_manifest['generation_contract']==m['generation_contract'] and reuse_manifest['source_authority']==m['source_authority']
for record in reuse_manifest['images']:
 f=int(record['image_id'].split('_')[-1])
 if f in frames:
  src=Path(record['output']) if a.reuse_sam_root is None else a.reuse_sam_root/Path(record['output']).name;assert sha256_file(src)==record['output_sha256'];os.link(src,sam/src.name);reused[src.name]=record['output_sha256']
snap=root/'sam_sources';snap.mkdir();sources={}
for f in [__file__,'radio_gs/scripts/build_sam3_multiscale_hierarchy_cache.py','radio_gs/scripts/build_sam3_foundation_cache.py','radio_gs/models/sam3_multiscale_hierarchy.py','scripts/audits/run_lerf_source_stage.py']:
 shutil.copy2(f,snap/Path(f).name);sources[f]=sha256_file(f)
r=dict(command=command,source_frames=frames,source_plan_sha256=sha256_file(root/'source_plan.json'),parent_manifest_sha256=sha256_file(parent/'source_sam/manifest.json'),reused_payloads=reused,implementations=sources,status='running');marker=root/'source_sam_execution.json';marker.write_text(json.dumps(r,indent=2)+'\n')
with (root/'source_sam.log').open('x') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
r.update(status='complete' if not result.returncode else 'failed',returncode=result.returncode)
if not result.returncode:
 new=json.loads((sam/'manifest.json').read_text());assert new['generation_contract']==m['generation_contract'];r['output_sha256']=sha256_file(sam/'manifest.json')
marker.write_text(json.dumps(r,indent=2)+'\n');assert result.returncode==0
if a.continue_pipeline:subprocess.run([sys.executable,'scripts/audits/run_pose_covered_lerf_pipeline.py','--scene',a.scene,'--device',a.device],check=True)
