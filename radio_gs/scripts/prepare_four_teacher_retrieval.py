"""Prepare source-only teacher evidence on a fresh frozen-geometry authority.

SAM masks are pseudo annotations, not benchmark ground truth. Native feature
spaces remain separate. No historical per-element feature table is consumed.
"""
import argparse,json,gc,shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.utils.mask_resampling import resize_boolean_masks_nearest_exact
from radio_gs.training.source_view_split import source_view_split
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks
from radio_gs.scripts.build_native_dinov2_exact_mpr_teacher import _load_model,_extract_feature_map
from radio_gs.scripts.build_multiscale_sam_mask_aligned_crop_summary_teacher import NativeSiglip2Runtime

# Fixed, scene-independent development vocabulary. No benchmark query file read.
WORDS=("person animal bird cat dog horse bear plant flower tree grass rock water "
"table chair sofa bed shelf cabinet desk lamp clock mirror picture book paper "
"cup mug bowl plate bottle glass vase jar box basket bag shoe hat shirt "
"computer keyboard mouse phone camera television speaker remote "
"toy doll sculpture statue ball vehicle car truck bicycle airplane boat "
"food fruit apple banana orange bread cake utensil knife spoon fork "
"door window wall floor ceiling curtain carpet cushion blanket").split()

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--scene',default='figurines');p.add_argument('--sam-manifest',type=Path);p.add_argument('--allow-legacy-sam-source',action='store_true',help='Historical corrupted-SAM control only; never a corrected teacher build');p.add_argument('--radio-root',type=Path);p.add_argument('--without-radio',action='store_true');p.add_argument('--validation-frame-ids',type=int,nargs='+');p.add_argument('--device',default='cuda:0')
 a=p.parse_args();root=a.root
 if a.sam_manifest is None and not a.allow_legacy_sam_source:raise ValueError('Explicit corrected --sam-manifest required; old fragment fallback needs --allow-legacy-sam-source')
 if (root/'teachers.json').exists():raise FileExistsError(root/'teachers.json')
 torch.set_num_threads(4);device=torch.device(a.device)
 authority=json.loads((root/'responsibility.json').read_text());source=json.loads((root/'source_frames.json').read_text())
 frames=authority['frame_indices'];train_frames,validation_frames=source_view_split(frames,a.validation_frame_ids);h=authority['metadata']['feature_height'];w=authority['metadata']['feature_width']
 sources={r['frame_idx']:r for r in source['frames']}
 if a.sam_manifest is not None:
  from radio_gs.training.source_sam_evidence import read_manifest,load_source_sam
  sam_manifest,samrecords=read_manifest(a.sam_manifest,allow_legacy=a.allow_legacy_sam_source)
 else:
  old=torch.load(f'/mnt/pool/sqy/results/RADIO-GS/output/v4_native_fragment_repair_20260914/{a.scene}.pt',map_location='cpu',weights_only=False)
  samrecords={r['frame_id']:r for r in old['frame_records']}
  for record in old['sam_manifests']:
   assert sha256_file(record['path'])==record['sha256']
 masks=[];receipts=[];images=[];crops=[];propviews=[];qualities=[];native_indices=[]
 for vi,frame in enumerate(frames):
  src=sources[frame];imagepath=Path(source['image_dir'])/src['source_file']
  assert sha256_file(imagepath)==src['source_sha256']
  image=Image.open(imagepath).convert('RGB');images.append(imagepath)
  if a.sam_manifest is not None:
   sam,sam_sha=load_source_sam(sam_manifest,samrecords,imagepath,src['source_sha256'])
   quality=sam['scores'].float()
  else:
   record=samrecords[frame];assert sha256_file(record['sam_fragment_payload'])==record['sam_fragment_payload_sha256']
   sam=torch.load(record['sam_fragment_payload'],map_location='cpu',weights_only=False)
   binding=sam['metadata']['source_image'];assert binding['sha256']==src['source_sha256']
   assert sam['metadata']['generation_contract']['query_free'] is True
   quality=sam['quality'].float();sam_sha=record['sam_fragment_payload_sha256']
  native=unpack_masks(sam['packed_masks'],sam['mask_shape'][1])
  if tuple(native.shape[1:])!=(image.height,image.width):raise ValueError('SAM mask/image grid differs')
  resized=resize_boolean_masks_nearest_exact(torch.from_numpy(native.copy()),(h,w))
  rgb=np.array(image)
  for j,mask in enumerate(resized):
   area=float(mask.float().mean())
   if not .002 <= area <= .6:continue
   ys,xs=np.where(native[j]);x0,x1=max(0,int(xs.min())-3),min(image.width,int(xs.max())+4);y0,y1=max(0,int(ys.min())-3),min(image.height,int(ys.max())+4)
   masked=rgb[y0:y1,x0:x1].copy();masked[~native[j,y0:y1,x0:x1]]=127
   crop=Image.fromarray(masked).resize((384,384),Image.Resampling.BICUBIC)
   crops.append(pil_to_tensor(crop).float()/255)
   masks.append(mask);propviews.append(vi);qualities.append(float(quality[j]));native_indices.append(j)
  receipts.append({'frame':frame,'rgb_sha256':src['source_sha256'],'sam_sha256':sam_sha})
 print('SAM regions',len(masks),flush=True)
 # DINO is extracted natively; no RADIO adaptor is used.
 dino_checkpoint='/root/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth'
 model=_load_model(SimpleNamespace(checkpoint=dino_checkpoint,torchhub_repo='/root/.cache/torch/hub/facebookresearch_dinov2_main',model_name='dinov2_vitb14'),device)
 dino=[]
 for image in images:dino.append(_extract_feature_map(model,image,device=device,grid_height=46,grid_width=62,patch_size=14).cpu().half())
 torch.save(torch.stack(dino),root/'teacher_dino.pt')
 del model;gc.collect();torch.cuda.empty_cache()
 # Raw RADIO is an independent optional auxiliary target.
 radio_root=None
 if not a.without_radio:
  radio_root=a.radio_root or Path('/mnt/pool/sqy/results/RADIO-GS/output/canonical_teacher_features')/a.scene
  manifest=json.loads((radio_root/'frame_manifest.json').read_text())
  records={r['frame']['frame_idx']:r for r in manifest['output_bundle']['frames']}
  radio=[]
  for frame in frames:
   if records[frame]['frame']['source_sha256']!=sources[frame]['source_sha256']:
    raise ValueError('RADIO source RGB differs from geometric source')
   rec=next(r for r in records[frame]['tensors'] if r['relative_path'].startswith('backbone/'))
   path=radio_root/rec['relative_path'];assert sha256_file(path)==rec['sha256']
   radio.append(torch.load(path,map_location='cpu',weights_only=True))
  torch.save(torch.stack(radio),root/'teacher_radio.pt');del radio
 sigroot=next(Path('/root/.cache/huggingface/hub/models--google--siglip2-giant-opt-patch16-384/snapshots').iterdir())
 vision=NativeSiglip2Runtime.load(sigroot,device=device)
 sig=[]
 for start in range(0,len(crops),8):sig.append(vision.encode(torch.stack(crops[start:start+8])).cpu().half())
 sig=torch.cat(sig);sigidentity=vision.bundle
 del vision,crops;gc.collect();torch.cuda.empty_cache()
 from transformers import SiglipTextModel,AutoTokenizer
 from transformers import SiglipTextConfig
 from safetensors import safe_open
 config=SiglipTextConfig.from_pretrained(str(sigroot),local_files_only=True)
 textmodel=SiglipTextModel(config)
 state={}
 for shard in sorted(sigroot.glob('model-*-of-*.safetensors')):
  with safe_open(str(shard),framework='pt',device='cpu') as f:
   for key in f.keys():
    if key.startswith('text_model.'):state[key]=f.get_tensor(key)
 out_dim,in_dim=state['text_model.head.weight'].shape
 textmodel.text_model.head=torch.nn.Linear(in_dim,out_dim)
 textmodel.load_state_dict(state,strict=True)
 textmodel=textmodel.to(device=device,dtype=torch.float16).eval()
 textmodel.requires_grad_(False)
 del state

 tokenizer=AutoTokenizer.from_pretrained(str(sigroot),local_files_only=True)
 queries=['a photo of a '+word for word in WORDS]
 text=[]
 with torch.inference_mode():
  for start in range(0,len(queries),16):
   inputs=tokenizer(queries[start:start+16],padding='max_length',max_length=64,truncation=True,return_tensors='pt').to(device)
   text.append(F.normalize(textmodel(**inputs).pooler_output.float(),dim=-1).cpu().half())
 text=torch.cat(text);assert text.shape[1]==sig.shape[1]
 torch.save({'descriptors':sig,'text_embeddings':text,'queries':queries,'encoder':sigidentity},root/'teacher_siglip.pt')
 torch.save({'masks':torch.stack(masks),'view_indices':torch.tensor(propviews),'quality':torch.tensor(qualities),'native_proposal_indices':torch.tensor(native_indices),'source_receipts':receipts},root/'teacher_sam.pt')
 # DINO query features remain query-encoder evidence even when its AUX loss is off.
 m=torch.stack(masks);dino=torch.stack(dino)
 region_dino=[];prompt=[];seedpixels=[]
 for mask,vi in zip(m,propviews):
  dense=F.interpolate(dino[vi:vi+1].float(),size=(h,w),mode='bilinear',align_corners=False)[0]
  region_dino.append(F.normalize(dense[:,mask].mean(1),dim=0))
  # Interior positive and distant background point; negative evidence is explicit.
  from scipy.ndimage import distance_transform_edt
  pos=int(np.argmax(distance_transform_edt(mask.numpy())))
  neg=int(np.argmax(distance_transform_edt((~mask).numpy())))
  prompt.append(torch.cat([dense.reshape(dense.shape[0],-1)[:,pos],dense.reshape(dense.shape[0],-1)[:,neg],
   torch.tensor([pos%w/w,pos//w/h,neg%w/w,neg//w/h])]))
  seedpixels.append([pos,neg])
 torch.save({'image':sig,'prompt':torch.stack(prompt).half(),'region_dino':torch.stack(region_dino).half(),
  'text':text,'queries':queries,'seed_pixels':torch.tensor(seedpixels)},root/'queries.pt')
 resampling_source=Path(__file__).resolve().parents[1]/'utils/mask_resampling.py'
 shutil.copy2(resampling_source,root/'mask_resampling_source.py')
 result={'schema':'radio_gs.four_teacher_source_evidence.v1','frames':frames,'train_frames':train_frames,'validation_frames':validation_frames,
  'mask_resampling_implementation_sha256':sha256_file(root/'mask_resampling_source.py'),
  'sam_source_manifest_sha256':sha256_file(a.sam_manifest) if a.sam_manifest else None,
  'legacy_sam_source_explicitly_allowed':a.allow_legacy_sam_source,
  'sam_complex_rope_preserved':sam_manifest['generation_contract'].get('complex_rope_buffers_preserved',False) if a.sam_manifest else False,
  'authority_sha256':sha256_file(root/'responsibility.json'),'source_manifest_sha256':sha256_file(root/'source_frames.json'),
  'teacher_files':{name:{'path':f'teacher_{name}.pt','sha256':sha256_file(root/f'teacher_{name}.pt')} for name in (['dino','sam','siglip'] if a.without_radio else ['radio','dino','sam','siglip'])},
  'query_file':{'path':'queries.pt','sha256':sha256_file(root/'queries.pt')},
  'teacher_roles':{'radio':'raw_backbone_feature','dino':'native_dinov2_feature_and_relations','sam':'native_sam3_region_boundary_pseudo_labels','siglip':'native_siglip2_region_identity_and_text'},
  'dino_checkpoint_sha256':sha256_file(dino_checkpoint),'radio_manifest_sha256':sha256_file(radio_root/'frame_manifest.json') if radio_root is not None else None,
  'benchmark_queries_opened':False,'benchmark_masks_opened':False,'semantic_ground_truth':False,
  'text_policy':'fixed_scene_independent_development_vocabulary','num_regions':len(masks)}
 (root/'teachers.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
