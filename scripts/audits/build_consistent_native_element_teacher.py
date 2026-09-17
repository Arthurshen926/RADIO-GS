"""Source-only native teacher on the exact frozen element domain.

A single fused direction per element yields jointly realizable feature targets.
This is a training artifact, never an additional deployed semantic map.
"""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.training.native_query_basis import native_query_basis
from radio_gs.training.native_region_mixing import mix_region_descriptors
from radio_gs.training.pipeline_contracts import validate_training_axes,validate_contributions
from radio_gs.utils.immutable_artifacts import sha256_file

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--dimensions',type=int,default=64);p.add_argument('--dense-fallback',action='store_true');p.add_argument('--region-inventory',type=Path);p.add_argument('--inverse-area-mixing',action='store_true');p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(4)
 if a.output.exists() or a.output.with_suffix('.json').exists():raise FileExistsError(a.output)
 root=a.root;info=json.loads((root/'teachers.json').read_text());receipt=json.loads((root/'episodes.json').read_text());auth=json.loads((root/'responsibility.json').read_text());assert sha256_file(root/'episodes.pt')==receipt['sha256'];assert sha256_file(root/'responsibility.json')==info['authority_sha256']
 data=torch.load(root/'episodes.pt',weights_only=False);assert sha256_file(root/'teachers.json')==data['teacher_manifest_sha256'];validate_training_axes(data,auth,info)
 rec=info['teacher_files']['siglip'];path=root/rec['path'];assert sha256_file(path)==rec['sha256'];teacher=torch.load(path,weights_only=False)['descriptors'].float()
 inventory_sha256=None
 if a.region_inventory is not None:
  inventory_sha256=sha256_file(a.region_inventory);ir=json.loads(a.region_inventory.with_suffix('.json').read_text());assert ir['sha256']==inventory_sha256;inventory=torch.load(a.region_inventory,weights_only=False)
  if inventory['schema']!='grounded_native_region_inventory.v1' or inventory['teacher_manifest_sha256']!=data['teacher_manifest_sha256'] or inventory['geometry_binding']!=data['geometry_binding'] or inventory['train_frames']!=info['train_frames'] or inventory['benchmark_data_used'] is not False:raise ValueError('region inventory scope differs')
  data=dict(data,masks=inventory['masks'],proposal_views=inventory['proposal_views']);teacher=inventory['descriptors'].float()
 trainviews=[info['frames'].index(f) for f in info['train_frames']];regions=torch.tensor([i for i,v in enumerate(data['proposal_views']) if int(v) in trainviews]);basis,energy=native_query_basis(torch.cat([data['queries']['text'].float(),(teacher[regions] if a.region_inventory is not None else data['queries']['image'][regions].float())]),a.dimensions)
 h,w=auth['metadata']['feature_height'],auth['metadata']['feature_width'];n=auth['num_gaussians'];dev=a.device;masks=data['masks'].flatten(1).to(dev);dirs=F.normalize(F.linear(teacher.to(dev),basis.to(dev)),dim=-1);numerator=torch.zeros(n,a.dimensions,device=dev);mass=torch.zeros(n,device=dev);view_counts=torch.zeros(n,dtype=torch.int32,device=dev);region_hits=[]
 if a.region_inventory is not None:data['region_probability']=torch.zeros(len(masks),n,device=dev)
 runtime=None;source_records=[]
 if a.dense_fallback:
  from radio_gs.scripts.build_multiscale_sam_mask_aligned_crop_summary_teacher import NativeSiglip2Runtime
  from radio_gs.training.source_native_grid import native_grid_features
  identity=json.loads((root/'native_query_encoder.json').read_text());source=json.loads((root/'source_frames.json').read_text());source_index={x['frame_idx']:x for x in source['frames']}
  runtime=NativeSiglip2Runtime.load(Path(identity['siglip_root']),device=torch.device(dev))
 # Equal weighting of covering masks within a view; exact compositor contribution
 # weighting across pixels/views. No GT, vocabulary matches, or query thresholds.
 for vi in trainviews:
  ids=torch.where(data['proposal_views']==vi)[0].to(dev);m=masks[ids].float();pixels,supported=mix_region_descriptors(m,dirs[ids],inverse_area=a.inverse_area_mixing)
  if runtime is not None:
   sr=source_index[info['frames'][vi]];image_path=Path(source['image_dir'])/sr['source_file'];assert sha256_file(image_path)==sr['source_sha256'];source_records.append({'frame':info['frames'][vi],'sha256':sr['source_sha256']})
   fallback=F.normalize(F.linear(native_grid_features(image_path,runtime,h,w),basis.to(dev)),dim=-1)
   pixels=torch.where(supported[:,None],pixels,fallback);supported=torch.ones_like(supported)
  record=auth['views'][vi];path=root/record['relative_path'];assert sha256_file(path)==record['sha256'];hit=torch.load(path,map_location=dev,weights_only=False);pid=hit['pixel_ids'].long();gid=hit['gaussian_ids'].long();weight=hit['base_weights'].float();validate_contributions(pid,gid,weight,num_pixels=h*w,num_elements=n)
  if a.region_inventory is not None:
   pixelmass=torch.zeros(h*w,device=dev).scatter_add_(0,pid,weight);normalized=weight/pixelmass[pid].clamp_min(1e-8);mapping=torch.sparse_coo_tensor(torch.stack([gid,pid]),normalized,(n,h*w)).coalesce();elementmass=torch.zeros(n,device=dev).scatter_add_(0,gid,normalized)
   data['region_probability'][ids]=(torch.sparse.mm(mapping,m.T)/elementmass[:,None].clamp_min(1e-8)).T
  weight=weight*supported[pid];mat=torch.sparse_coo_tensor(torch.stack([gid,pid]),weight,(n,h*w)).coalesce();numerator+=torch.sparse.mm(mat,pixels);localmass=torch.zeros_like(mass).scatter_add_(0,gid,weight);mass+=localmass;view_counts+=(localmass>1e-8).int()
  print('source view',info['frames'][vi],flush=True)
 observed=data['seen'][trainviews].any(0).to(dev);mean=numerator/mass[:,None].clamp_min(1e-8);agreement=mean.norm(dim=-1);valid=observed&(mass>1e-8)&(agreement>1e-8);directions=F.normalize(mean,dim=-1)*valid[:,None]
 # Audit the old independent region-unit target premise on overlapping elements.
 hard=(data['region_probability'][regions].to(dev)>.7)&observed[None];inter=hard.float()@hard.float().T;area=hard.sum(1);fraction=inter/torch.minimum(area[:,None],area[None]).clamp_min(1);cos=dirs[regions.to(dev)]@dirs[regions.to(dev)].T;upper=torch.triu(torch.ones_like(inter,dtype=torch.bool),diagonal=1);pairs=upper&(inter>=20)&(fraction>=.1);incompatible=pairs&(cos<.95)
 audit={'observed_elements':int(observed.sum()),'teacher_elements':int(valid.sum()),'teacher_coverage_fraction':float(valid.sum()/observed.sum()),'multiple_view_elements':int((valid&(view_counts>=2)).sum()),'mean_resultant_norm':float(agreement[valid].mean()),'overlapping_region_pairs':int(pairs.sum()),'overlapping_pairs_cosine_below_095':int(incompatible.sum()),'old_target_implication':'mean of unit vectors equals a unit vector only when every positively weighted element has that direction; overlapping differing targets cannot both be exact','native_basis_energy':energy}
 bundle={'schema':'consistent_native_element_teacher.v1','inverse_area_mixing':a.inverse_area_mixing,'mixing_implementation_sha256':sha256_file(Path(__file__).resolve().parents[2]/'radio_gs/training/native_region_mixing.py'),'region_inventory_sha256':inventory_sha256,'dense_fallback':a.dense_fallback,'source_rgb_records':source_records,'grid_implementation_sha256':sha256_file(Path(__file__).resolve().parents[2]/'radio_gs/training/source_native_grid.py') if a.dense_fallback else None,'episode_sha256':receipt['sha256'],'teacher_manifest_sha256':data['teacher_manifest_sha256'],'siglip_teacher_sha256':rec['sha256'],'geometry_binding':data['geometry_binding'],'train_frames':info['train_frames'],'basis':basis,'directions':directions.cpu().half(),'valid':valid.cpu(),'agreement':agreement.cpu().half(),'view_counts':view_counts.cpu(),'audit':audit,'construction':'inverse-mask-area normalized descriptor mixture per pixel; exact frozen contribution mean across views; normalize once per element' if a.inverse_area_mixing else 'equal-mask descriptor mixture per pixel; exact frozen contribution mean across views; normalize once per element','benchmark_data_used':False,'implementation_sha256':sha256_file(__file__)}
 torch.save(bundle,a.output);a.output.with_suffix('.json').write_text(json.dumps({'schema':bundle['schema'],'sha256':sha256_file(a.output),'audit':audit,'benchmark_data_used':False,'implementation_sha256':bundle['implementation_sha256']},indent=2)+'\n');print(audit,flush=True)
if __name__=='__main__':main()
