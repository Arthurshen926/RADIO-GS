"""Complete the same v2 method on the two remaining historical LERF scenes.

The source cohort is selected from sealed mapping cameras without labels/query
text. Historical use means these scenes remain development, not a new test set.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import torch
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.utils.immutable_artifacts import sha256_file

p=argparse.ArgumentParser();p.add_argument('--scene',choices=['teatime','waldo_kitchen'],required=True);p.add_argument('--device',required=True);p.add_argument('--resume',action='store_true');p.add_argument('--output-root',type=Path);p.add_argument('--source-plan',type=Path,help='Sealed source plan for fresh reproduction, including recovered splits');a=p.parse_args();torch.set_num_threads(4)
root=a.output_root or Path('output/lerf_complete_v2_20260918')/a.scene;source_root=root/'source';marker=root/'execution.json'
if marker.exists():
 assert a.resume;record=json.loads(marker.read_text())
else:
 source_root.mkdir(parents=True,exist_ok=False);record=dict(schema='complete_lerf_v2.v1',scene=a.scene,source_only_training=True,independent_final_test=False,stages=[])
 marker.write_text(json.dumps(record,indent=2)+'\n')
def execute(name,command,outputs):
 old=next((r for r in record['stages'] if r['name']==name),None)
 if old:
  assert old['status']=='complete' and old['command']==list(map(str,command))
  for path,digest in old['outputs'].items():assert sha256_file(path)==digest
  return
 stage=dict(name=name,command=list(map(str,command)),status='running');record['stages'].append(stage);marker.write_text(json.dumps(record,indent=2)+'\n');print(a.scene,name,flush=True)
 with (root/(name+'.log')).open('x') as log:r=subprocess.run(stage['command'],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONPATH='.'))
 stage.update(status='complete' if r.returncode==0 else 'failed',returncode=r.returncode)
 if not r.returncode:stage['outputs']={str(f):sha256_file(f) for f in outputs}
 marker.write_text(json.dumps(record,indent=2)+'\n');assert r.returncode==0,(name,root/(name+'.log'))
planpath=source_root/'source_plan.json'
if a.source_plan is not None:
 supplied=json.loads(a.source_plan.read_text());assert supplied['scene']==a.scene and supplied['benchmark_data_used'] is False
 assert len(supplied['frames'])==len(set(supplied['frames']))
 assert set(supplied['train_frames']).isdisjoint(supplied['validation_frames'])
 assert set(supplied['frames'])==set(supplied['train_frames'])|set(supplied['validation_frames'])
 if planpath.exists():assert sha256_file(planpath)==sha256_file(a.source_plan),'Existing source plan differs'
 else:shutil.copy2(a.source_plan,planpath)
 record['source_plan_input']=dict(path=str(a.source_plan.resolve()),sha256=sha256_file(a.source_plan));marker.write_text(json.dumps(record,indent=2)+'\n')
if not planpath.exists():
 authority=Path(f'paper/artifacts/lerf_{a.scene}_source32_exact_mpr_rgb_authority_20260817.json');auth=json.loads(authority.read_text());assert not auth['information_policy']['benchmark_ground_truth_used'] and not auth['information_policy']['query_text_used']
 gp=Path('output/geometry_only_repair_final_20260914')/a.scene/'report.json';gr=json.loads(gp.read_text());prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256'])
 radio=Path('output/canonical_teacher_features')/a.scene;rm=json.loads((radio/'frame_manifest.json').read_text());rby={r['frame_idx']:r for r in rm['frames']};by={int(r['image_id'].split('_')[-1]):r for r in auth['images']}
 available=sorted(set(by)&set(rby)&set(gr['mapping_frames'])&set(prior.cameras));assert len(available)>=18
 validation=[available[len(available)//3],available[2*len(available)//3]];eligible=[f for f in available if f not in validation]
 poses=torch.stack([prior.cameras[f].world_to_camera for f in eligible]);centers=-torch.einsum('nij,nj->ni',poses[:,:3,:3].transpose(1,2),poses[:,:3,3]);diam=torch.cdist(centers,centers).max().clamp_min(1e-8);distance=torch.cdist(centers,centers).square()/diam.square()+torch.cdist(poses[:,2,:3],poses[:,2,:3]).square();indices=[0]
 while len(indices)<16:
  closest=distance[:,indices].amin(1);closest[indices]=-1;indices.append(int(closest.argmax()))
 train=sorted(eligible[i] for i in indices);frames=train+validation
 for frame in frames:
  assert Path(by[frame]['path']).name==rby[frame]['source_file'];assert sha256_file(by[frame]['path'])==by[frame]['sha256']==rby[frame]['source_sha256']
 plan=dict(scene=a.scene,frames=frames,train_frames=train,validation_frames=validation,radio_root=str(radio.resolve()),geometry_report_sha256=sha256_file(gp),source_authority=dict(path=str(authority.resolve()),sha256=sha256_file(authority)),benchmark_data_used=False,selection='Within sealed source32/mapping/radio intersection. Validation fixed at sorted one-third/two-thirds; train16 greedy camera coverage seeded at first remaining frame. No benchmark queries or labels.',implementation_sha256=sha256_file(__file__))
 planpath.write_text(json.dumps(plan,indent=2)+'\n')
plan=json.loads(planpath.read_text());gp=Path('output/geometry_only_repair_final_20260914')/a.scene/'report.json';assert sha256_file(gp)==plan['geometry_report_sha256'];assert sha256_file(plan['source_authority']['path'])==plan['source_authority']['sha256']
command=json.loads(Path('output/lerf_pose_covered_pipeline_20260917/figurines/source_sam_execution.json').read_text())['command'].copy()
for key,value in {'--source-authority':plan['source_authority']['path'],'--source-authority-sha256':plan['source_authority']['sha256'],'--image-ids':','.join(f'frame_{f:05d}' for f in plan['frames']),'--output-root':str(source_root/'source_sam'),'--device':a.device}.items():command[command.index(key)+1]=value
snap=source_root/'sam_sources';snap.mkdir(exist_ok=True);impl={}
for f in ['radio_gs/scripts/build_sam3_multiscale_hierarchy_cache.py','radio_gs/scripts/build_sam3_foundation_cache.py','radio_gs/models/sam3_multiscale_hierarchy.py','scripts/audits/run_lerf_source_stage.py']:
 if (snap/Path(f).name).exists():assert sha256_file(snap/Path(f).name)==sha256_file(f)
 else:shutil.copy2(f,snap/Path(f).name)
 impl[f]=sha256_file(f)
execute('source_sam',command,[source_root/'source_sam/manifest.json'])
sam_record=source_root/'source_sam_execution.json'
if not sam_record.exists():sam_record.write_text(json.dumps(dict(status='complete',command=command,implementations=impl,source_frames=plan['frames'],source_plan_sha256=sha256_file(planpath)),indent=2)+'\n')
execute('base_field',[sys.executable,'scripts/audits/run_pose_covered_lerf_pipeline.py','--scene',a.scene,'--device',a.device,'--output-root',source_root,'--until-base'],[source_root/'field/scale_affinity_v1_direct_control/report.json',source_root/'field/caption_region_evidence.json'])
rebuild=root/'reconstruction';field=rebuild/'field'
execute('pixel_fusion_and_readouts',[sys.executable,'scripts/audits/run_pixel_evidence_rebuild.py','--scene',a.scene,'--device',a.device,'--parent',source_root/'field','--output-root',rebuild],[field/'joint_supervised_image_residual/report.json'])
execute('admission',[sys.executable,'scripts/audits/run_lerf_source_stage.py','scripts/audits/fit_counterfactual_extent_admission.py','--root',field,'--scene',a.scene,'--device',a.device],[field/'counterfactual_extent_benefit_admission/report.json'])
for task in ['text','reference']:
 execute('evaluate_'+task,[sys.executable,'scripts/audits/audit_current_query_routes.py','--task',task,'--root',field,'--scene',a.scene,'--geometry-report',gp,'--runs','counterfactual_extent_benefit_admission','--device',a.device,'--output',root/(task+'_online.json')],[root/(task+'_online.json')])
record['finished']=True;marker.write_text(json.dumps(record,indent=2)+'\n')
