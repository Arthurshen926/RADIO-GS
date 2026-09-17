"""Measure native source features only for previously unsupported image rows.

Fixed 4x4 source RGB tiles, source-training views only, at least two contributing
views. Original supported rows remain exact; text/prompt keep their old domain.
"""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.scripts.build_multiscale_sam_mask_aligned_crop_summary_teacher import NativeSiglip2Runtime
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',default='image_supervised_native_residual');p.add_argument('--authority-root',type=Path);p.add_argument('--output-name');p.add_argument('--device',required=True);a=p.parse_args();torch.set_num_threads(4)
 root=a.root;parent=root/a.parent;report=json.loads((parent/'report.json').read_text());assert sha256_file(parent/'field.pt')==report['field_sha256'];payload=torch.load(parent/'field.pt',weights_only=False)
 info=json.loads((root/'teachers.json').read_text());source=json.loads((root/'source_frames.json').read_text());auth=json.loads((root/'responsibility.json').read_text());identity=json.loads((root/'native_query_encoder.json').read_text())
 assert sha256_file(root/'responsibility.json')==info['authority_sha256'] and payload['query_encoder']==identity and payload['geometry_binding']==auth['metadata']['frozen_geometry_binding']
 assert sha256_file(root/'source_frames.json')==info['source_manifest_sha256']
 for record in identity['files']:assert sha256_file(Path(record['path']).resolve())==record['sha256']
 arch=dict(payload['architecture'],image_support_extension=True);field=ElementRetrievalField(**arch).to(a.device);missing=field.load_state_dict(payload['state_dict'],strict=False)
 assert missing.missing_keys==['image_observed'] and not missing.unexpected_keys;field.requires_grad_(False)
 authority_root=a.authority_root or root
 if a.authority_root is not None:
  oldauth=auth;auth=json.loads((authority_root/'responsibility.json').read_text());build=json.loads((authority_root/'build.json').read_text())
  assert sha256_file(authority_root/'responsibility.json')==build['sha256']
  assert auth['metadata']['frozen_geometry_binding']==oldauth['metadata']['frozen_geometry_binding'] and auth['frame_indices']==oldauth['frame_indices']
  assert json.loads((authority_root/'source_frames.json').read_text())['frames']==source['frames']
  for frame in info['frames']:
   oldcam=oldauth['metadata']['cameras'][str(frame)];newcam=auth['metadata']['cameras'][str(frame)]
   k=torch.tensor(oldcam['intrinsic']);k[0]*=auth['metadata']['feature_width']/oldauth['metadata']['feature_width'];k[1]*=auth['metadata']['feature_height']/oldauth['metadata']['feature_height']
   assert torch.allclose(k,torch.tensor(newcam['intrinsic']),atol=1e-4) and oldcam['world_to_camera']==newcam['world_to_camera']
 dim=field.native_decoder.in_features;basis=field.native_decoder.weight.T
 assert torch.allclose(basis@basis.T,torch.eye(dim,device=a.device),atol=1e-4) and torch.count_nonzero(field.native_decoder.bias)==0
 src={s['frame_idx']:s for s in source['frames']};h,w=auth['metadata']['feature_height'],auth['metadata']['feature_width']
 runtime=NativeSiglip2Runtime.load(Path(identity['siglip_root']),device=torch.device(a.device))
 n=len(field.latent);total=torch.zeros(n,dim,device=a.device);mass=torch.zeros(n,device=a.device);views=torch.zeros(n,dtype=torch.int32,device=a.device);rows=[];energies=[]
 yy,xx=torch.meshgrid(torch.arange(h,device=a.device),torch.arange(w,device=a.device),indexing='ij');tile_id=((yy*4//h)*4+xx*4//w).flatten()
 with torch.inference_mode():
  for frame in info['train_frames']:
   entry=src[frame];path=Path(source['image_dir'])/entry['source_file'];assert sha256_file(path)==entry['source_sha256'];image=Image.open(path).convert('RGB');crops=[]
   for y in range(4):
    for x in range(4):
     box=(image.width*x//4,image.height*y//4,image.width*(x+1)//4,image.height*(y+1)//4)
     crops.append(pil_to_tensor(image.crop(box).resize((384,384),Image.Resampling.BICUBIC)).float()/255)
   native=F.normalize(torch.cat([runtime.encode(torch.stack(crops[i:i+8])).float() for i in range(0,16,8)]),dim=-1)
   compact=native@basis.T;energies.extend(compact.square().sum(-1).cpu().tolist())
   vi=info['frames'].index(frame);record=auth['views'][vi];hp=authority_root/record['relative_path'];assert sha256_file(hp)==record['sha256'];hit=torch.load(hp,map_location=a.device,weights_only=False)
   gid,pid,weight=hit['gaussian_ids'].long(),hit['pixel_ids'].long(),hit['base_weights'].float()
   pixelmass=torch.zeros(h*w,device=a.device).scatter_add_(0,pid,weight);weight=weight/pixelmass[pid].clamp_min(1e-8)
   keep=~field.observed[gid];gid,pid,weight=gid[keep],pid[keep],weight[keep]
   contribution=torch.zeros_like(mass).scatter_add_(0,gid,weight)
   total.index_add_(0,gid,compact[tile_id[pid]]*weight[:,None]);mass+=contribution;views+=(contribution>1e-6).int()
   rows.append(dict(frame=frame,rgb_sha256=entry['source_sha256'],contribution_sha256=record['sha256'],unsupported_rows_measured=int((contribution>1e-6).sum())))
   print(frame,rows[-1]['unsupported_rows_measured'],flush=True)
  measured=(~field.observed)&(views>=2)&(mass>1e-6)&(total.norm(dim=-1)>1e-8)
  field.latent[measured,:dim]=total[measured]/mass[measured,None]
  field.latent[measured,dim:]=0 # Explicitly untrained structural suffix, bypassed by image residual.
  field.image_observed.copy_(field.observed|measured)
 state={k:v.detach().cpu() for k,v in field.state_dict().items()}
 for k,v in payload['state_dict'].items():
  if k=='latent':assert torch.equal(v[~measured.cpu()],state[k][~measured.cpu()])
  else:assert torch.equal(v,state[k]),k
 out=root/(a.output_name or a.parent+'_measured_support');out.mkdir(exist_ok=False);torch.save(dict(payload,architecture=arch,state_dict=state),out/'field.pt')
 receipt=dict(schema='measured_native_image_support.v1',field_sha256=sha256_file(out/'field.pt'),parent_field_sha256=report['field_sha256'],
 source_only=True,benchmark_data_used=False,geometry_unchanged=True,original_supported_latent_exact=True,all_other_parent_tensors_exact=True,
 original_support=int(field.observed.sum()),added_native_measured_rows=int(measured.sum()),image_support=int(field.image_observed.sum()),
 image_only_support=True,text_prompt_support_unchanged=True,minimum_source_views=2,source_grid=[4,4],
 crop_policy='native RGB disjoint 4x4 tiles; actual original native pooler; do not claim precise SAM object identity',
 field_policy='write native prefix only on unsupported measured rows; suffix zero; learned residual bypassed on new rows; no source-free nearest-neighbor fill',
 mean_projection_energy=sum(energies)/len(energies),records=rows,parent_authority_sha256=info['authority_sha256'],authority_sha256=sha256_file(authority_root/'responsibility.json'),source_raster_hw=[h,w],
 implementation_sources={f:sha256_file(f) for f in [__file__,'radio_gs/models/element_retrieval_field.py']})
 (out/'report.json').write_text(json.dumps(receipt,indent=2)+'\n');print('added',int(measured.sum()),flush=True)


if __name__=='__main__':main()
