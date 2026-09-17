"""Rebuild exactly the historical source crop/grid recipe with corrected SAM."""
import argparse,json,subprocess,shutil,sys
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file

p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);p.add_argument('--continue-pipeline',action='store_true');a=p.parse_args();torch.set_num_threads(4)
base=Path('/mnt/pool/sqy/results/RADIO-GS/output');index=torch.load(base/'v4_native_fragment_repair_20260914'/(a.scene+'.pt'),weights_only=False,map_location='cpu');assert len(index['sam_manifests'])==1;rec=index['sam_manifests'][0];assert sha256_file(rec['path'])==rec['sha256'];old=json.loads(Path(rec['path']).read_text());del index
info=json.loads((base/'preserved_source_inventory_20260916'/a.scene/'teachers.json').read_text());root=base/'lerf_ropefix_multiscale_pipeline_20260917'/a.scene;root.mkdir(parents=True,exist_ok=False)
c=old['generation_contract'];authority=old['source_authority'];checkpoint=Path(old['checkpoint']['path']).resolve()
cmd=['/root/miniconda3/envs/sam3/bin/python','scripts/audits/run_lerf_source_stage.py','--module','radio_gs.scripts.build_sam3_multiscale_hierarchy_cache','--source-authority',authority['path'],'--source-authority-sha256',authority['sha256'],'--image-ids',','.join(f'frame_{f:05d}' for f in info['frames']),'--output-root',str(root/'source_sam'),'--checkpoint-path',str(checkpoint),'--checkpoint-sha256',old['checkpoint']['sha256'],'--device',a.device,'--dtype',c['dtype']]
values={'resolution':c['resolution'],'crop-layers':c['crop_pyramid']['layers_after_full_image'],'crop-overlap-ratio':c['crop_pyramid']['overlap_ratio'],'points-per-side':c['point_grid']['points_per_side_at_full_image'],'point-grid-downscale-factor':c['point_grid']['downscale_factor_per_layer'],'dedup-iou':c['deduplication']['iou_threshold'],'dedup-near-equal-area-ratio':c['deduplication']['near_equal_area_ratio'],'maximum-masks':c['deduplication']['maximum_masks'],'containment-threshold':c['hierarchy']['containment_threshold'],'minimum-parent-area-ratio':c['hierarchy']['minimum_parent_area_ratio']}
for key in ['minimum_quality','minimum_stability','stability_offset','minimum_crop_area_fraction','minimum_full_image_area_fraction','maximum_full_image_area_fraction','crop_edge_tolerance_pixels']:values[key.replace('_','-')]=c['filtering'][key]
for key,value in values.items():cmd+=['--'+key,str(value)]
files=[__file__,'radio_gs/scripts/build_sam3_multiscale_hierarchy_cache.py','radio_gs/scripts/build_sam3_foundation_cache.py','radio_gs/models/sam3_multiscale_hierarchy.py','scripts/audits/run_lerf_source_stage.py'];snap=root/'sam_sources';snap.mkdir();sources={}
for f in files:shutil.copy2(f,snap/Path(f).name);sources[f]=sha256_file(f)
r=dict(command=cmd,source_frames=info['frames'],historical_manifest_sha256=rec['sha256'],implementations=sources,status='running');p=root/'source_sam_execution.json';p.write_text(json.dumps(r,indent=2)+'\n')
with (root/'source_sam.log').open('x') as log:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
r.update(returncode=result.returncode,status='failed' if result.returncode else 'complete')
if not result.returncode:
 new=json.loads((root/'source_sam/manifest.json').read_text())['generation_contract'];ignore={'digest','complex_rope_buffers_preserved','model_cast_implementation_sha256'}
 assert {k:v for k,v in c.items() if k not in ignore}=={k:v for k,v in new.items() if k not in ignore}
 r['sampling_and_filtering_contract_exact']=True;r['output_sha256']=sha256_file(root/'source_sam/manifest.json')
p.write_text(json.dumps(r,indent=2)+'\n');assert result.returncode==0
print(a.scene,'matched multiscale source complete',flush=True)
if a.continue_pipeline:subprocess.run([sys.executable,'scripts/audits/run_rope_fixed_matched_lerf_pipeline.py','--scene',a.scene,'--device',a.device],check=True)
