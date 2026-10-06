"""Same native observations and geometry; localize response loss across pooling.

region: category response before overlap pooling and cross-view lifting.
pixel: original normalized overlap feature, response before cross-view lifting.
map: retained element feature, response after both pooling stages (v2 comparator).
The first two are query-conditioned diagnostic readouts of original observations,
not learned19-class maps or compact deployable replacements.
"""
import argparse,concurrent.futures,json,os,subprocess,time
from pathlib import Path
from types import SimpleNamespace
PYTHON='/root/miniconda3/envs/cybersim_agent/bin/python'
ROOT=Path('output/native_pooling_order')
CONFIG=Path('configs/pipeline/scannet_native_complete')


def run_scene(name,gpu):
 os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
 import torch
 from torch.nn import functional as F
 from radio_gs.models.native_query_encoders import load_siglip_text
 from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
 from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
 from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks
 from radio_gs.pipeline.native_field import score_scene_field
 from radio_gs.querying.cold_category import categorical_prior
 from radio_gs.querying.scannet_runtime import refine_from_declared_sources
 torch.set_num_threads(4);start=time.monotonic()
 scene=json.loads((CONFIG/f'{name}.scene.json').read_text())
 queries=json.loads(Path(f'configs/consolidated/scannet_native/{name}.queries.json').read_text())['queries'];texts=[q['text'] for q in queries]
 load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
 out=ROOT/name;out.mkdir(parents=True,exist_ok=True)
 with torch.inference_mode():
  encoder,tokenizer=load_siglip_text(scene['text_encoder'],'cuda')
  tokens=tokenizer(texts,padding='max_length',max_length=64,truncation=True,return_tensors='pt').to('cuda')
  q=F.normalize(F.normalize(encoder(**tokens).pooler_output.float(),dim=-1).half().float(),dim=-1);del encoder
  field=load(scene['native_field']);observed=field['supported'].reshape(-1).cuda()
  base=categorical_prior(score_scene_field(scene,field,q,'cuda').cuda(),100.)
  geometry=FrozenGaussianGeometry.load(scene['geometry']['path'],expected_sha256=scene['geometry']['sha256'],device='cuda')
  cameras={c['frame_index']:c for c in json.loads(Path(scene['source_cameras']).read_text())['records']}
  sums={arm:torch.zeros(len(observed),len(texts),device='cuda') for arm in ['region','pixel']}
  mass=torch.zeros(len(observed),1,device='cuda');records=[]
  for frame in scene['source_frames']:
   c=cameras[frame];cache=load(Path(scene['native_region_directories'][0])/f'frame_{frame:05d}.pt')
   sam=load(cache['sam_path']);masks=unpack_masks(sam['packed_masks'],sam['mask_shape'][1])[cache['native_proposal_indices'].numpy()].copy()
   image_h,image_w=cache['native_mask_hw'];h,w=round(image_h*256/image_w),256
   if not len(masks):continue
   m=F.interpolate(torch.as_tensor(masks).float()[:,None],size=(h,w),mode='nearest-exact')[:,0].flatten(1).cuda()
   count=m.sum(0);f=F.normalize(cache['descriptors'].half().float().cuda(),dim=-1)
   pixel_feature=F.normalize(m.T@f/count[:,None].clamp_min(1),dim=-1)
   targets={'region':m.T@torch.softmax(100*(f@q.T),dim=-1)/count[:,None].clamp_min(1),
            'pixel':torch.softmax(100*(pixel_feature@q.T),dim=-1)}
   k=torch.as_tensor(c['intrinsic'],device='cuda',dtype=torch.float32).clone();k[0]*=w/c['width'];k[1]*=h/c['height']
   renderer=SimpleNamespace(scaled_intrinsics=lambda width,height:k,near_plane=.001,far_plane=100.)
   hit=rasterize_single_view_contributions(geometry,renderer,torch.as_tensor(c['world_to_camera'],device='cuda',dtype=torch.float32),height=h,width=w,include_depths=False)
   pixel,element=hit['pixel_ids'].long(),hit['gaussian_ids'].long();weight=hit['weights'].float()*(count[pixel]>0)
   adjoint=torch.sparse_coo_tensor(torch.stack([element,pixel]),weight,(len(observed),h*w)).coalesce()
   mass+=torch.sparse.mm(adjoint,torch.ones(h*w,1,device='cuda'))
   for arm in sums:sums[arm]+=torch.sparse.mm(adjoint,targets[arm])
   records.append(dict(frame=frame,regions=len(f),raster_hw=[h,w]));print(name,frame,round(time.monotonic()-start,1),flush=True)
  support_difference=int(((mass[:,0]>0)!=observed).sum())
  for arm,total in sums.items():
   probability=total/mass.clamp_min(1e-8)
   prior=torch.where(mass>0,torch.logit(probability.clamp(torch.finfo(probability.dtype).tiny,1-torch.finfo(probability.dtype).eps)),base)
   values,stats=refine_from_declared_sources(prior,observed,scene,texts)
   destination=out/arm;destination.mkdir(parents=True,exist_ok=True)
   torch.save(dict(source_evidence_used=True,queries=queries,texts=texts,category_logits=values.cpu(),scores=values.cpu(),native_logits=prior.cpu(),semantic_geometry=scene['geometry'],semantic_observed=observed.cpu(),source_stats=stats,arm=arm),destination/'typed_readout.pt')
   subprocess.run([PYTHON,'scripts/research/evaluate_scannet_runtime.py','--readout',str(destination/'typed_readout.pt'),'--evaluation',f'configs/consolidated/scannet_native/{name}.evaluation.json','--output',str(destination/'evaluation'),'--point-only'],env=dict(os.environ,PYTHONPATH='.'),check=True)
  (out/'construction.json').write_text(json.dumps(dict(records=records,support_difference=support_difference,seconds=time.monotonic()-start,query_conditioned=True,extra_teacher_calls=0,labels_read_by_inference=False,original_region_assets_required=True),indent=2)+'\n')


def main():
 p=argparse.ArgumentParser();p.add_argument('--scene');p.add_argument('--gpu',default='0');a=p.parse_args()
 if a.scene:run_scene(a.scene,a.gpu);return
 names=sorted(p.name.removesuffix('.scene.json') for p in CONFIG.glob('*.scene.json'))
 def worker(gpu):
  for name in names[gpu::6]:
   out=ROOT/name;out.mkdir(parents=True,exist_ok=True)
   with (out/'run.log').open('w') as log:subprocess.run([PYTHON,__file__,'--scene',name,'--gpu',str(gpu)],env=dict(os.environ,PYTHONPATH='.'),stdout=log,stderr=subprocess.STDOUT,check=True)
   print(name,'complete',flush=True)
 with concurrent.futures.ThreadPoolExecutor(6) as pool:list(pool.map(worker,range(6)))
 rows={}
 for name in names:
  rows[name]={'map':json.loads(Path('output/system_v2/scannet',name,'evaluation/report.json').read_text())['point_metrics']}
  for arm in ['region','pixel']:rows[name][arm]=json.loads((ROOT/name/arm/'evaluation/report.json').read_text())['point_metrics']
 macro={arm:{access:{s:sum(r[arm][access][s]['miou'] for r in rows.values())/len(rows) for s in ['19','15','10']} for access in ['map','source']} for arm in ['map','pixel','region']}
 (ROOT/'summary.json').write_text(json.dumps(dict(scenes=rows,macro=macro,independent_test=False,source_frames=1158,extra_teacher_calls=0),indent=2)+'\n');print(json.dumps(macro,indent=2),flush=True)

if __name__=='__main__':main()
