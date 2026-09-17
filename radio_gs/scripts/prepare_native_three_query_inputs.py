"""Seal native encoder identity and raw heldout query inputs before training."""
import argparse
from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.utils.mask_resampling import resize_boolean_masks_nearest_exact
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True);p.add_argument('--sam-source-index',type=Path);p.add_argument('--sam-manifest',type=Path);p.add_argument('--allow-legacy-sam-source',action='store_true');p.add_argument('--siglip-root',type=Path,required=True);p.add_argument('--dino-checkpoint',type=Path,required=True);p.add_argument('--dino-repo',type=Path,required=True);p.add_argument('--allow-training-encoding-fixture',action='store_true');a=p.parse_args();root=a.root
if (a.sam_source_index is None)==(a.sam_manifest is None):raise ValueError('exactly one SAM evidence source required')
if a.sam_source_index is not None and not a.allow_legacy_sam_source:raise ValueError('Old source-index controls require --allow-legacy-sam-source')
if (root/'native_query_encoder.json').exists() or (root/'cold_query_fixture.pt').exists():raise FileExistsError('native query artifacts already exist')
info=json.loads((root/'teachers.json').read_text());source=json.loads((root/'source_frames.json').read_text());sources={r['frame_idx']:r for r in source['frames']}
episode_receipt=json.loads((root/'episodes.json').read_text());assert sha256_file(root/'episodes.pt')==episode_receipt['sha256']
data=torch.load(root/'episodes.pt',weights_only=False);assert data['teacher_manifest_sha256']==sha256_file(root/'teachers.json')
sig,dino,repo=a.siglip_root,a.dino_checkpoint,a.dino_repo
files=[p for p in sig.iterdir() if p.suffix in ('.json','.model','.safetensors')]+[dino]+sorted((repo/'dinov2').rglob('*.py'))+[repo/'hubconf.py',Path('radio_gs/models/native_query_encoders.py').resolve(),Path('radio_gs/scripts/build_native_dinov2_exact_mpr_teacher.py').resolve(),Path('radio_gs/scripts/build_multiscale_sam_mask_aligned_crop_summary_teacher.py').resolve()]
identity={'schema':'radio_gs.native_three_query_encoder.v1','siglip_root':str(sig),'dino_checkpoint':str(dino),'dino_repo':str(repo),'files':[{'path':str(p),'sha256':sha256_file(p.resolve())} for p in files],'text':'siglip2_native_pooler_max_length64_normalized','image':'siglip2_native_pooler_rgb_bicubic384_normalized','prompt':'dinov2_b14_grid46x62_fp16_bilinear_to_authority_signed_features_plus_normalized_world_xyz'}
(root/'native_query_encoder.json').write_text(json.dumps(identity,indent=2)+'\n')
geometry_path=str(a.geometry);auth=json.loads((root/'responsibility.json').read_text());geomsha=auth['metadata']['frozen_geometry_sha256']
if sha256_file(a.geometry)!=geomsha:raise ValueError('geometry identity')
if a.sam_manifest is not None:
 from radio_gs.training.source_sam_evidence import read_manifest,load_source_sam
 sam_manifest,samrecords=read_manifest(a.sam_manifest,allow_legacy=a.allow_legacy_sam_source)
else:
 old=torch.load(a.sam_source_index,weights_only=False);samrecords={r['frame_id']:r for r in old['frame_records']}
f={'encoder_identity':identity,'geometry_path':geometry_path,'geometry_sha256':geomsha,'geometry_binding':data['geometry_binding'],'height':auth['metadata']['feature_height'],'width':auth['metadata']['feature_width'],'expected_encodings':{}}
from radio_gs.training.query_encoding_fixture import select_encoding_fixture
f['encoding_input_provenance']={}
for modality in ('text','image','prompt'):
 e,provenance=select_encoding_fixture(data['episodes'],modality,allow_training_input=a.allow_training_encoding_fixture);f['encoding_input_provenance'][modality]=provenance;q=e['query_region'];vi=int(data['proposal_views'][q]);frame=info['frames'][vi];src=sources[frame];imagepath=Path(source['image_dir'])/src['source_file'];assert sha256_file(imagepath)==src['source_sha256']
 idx=e.get('text_index',q);f['expected_encodings'][modality]=data['queries'][modality][idx:idx+1]
 if modality=='text':f['text']=data['queries']['queries'][idx]
 if modality=='image':
  if a.sam_manifest is not None:sam,_=load_source_sam(sam_manifest,samrecords,imagepath,src['source_sha256'])
  else:
   record=samrecords[frame];assert sha256_file(record['sam_fragment_payload'])==record['sam_fragment_payload_sha256'];sam=torch.load(record['sam_fragment_payload'],weights_only=False)
  native=unpack_masks(sam['packed_masks'],sam['mask_shape'][1]);image=Image.open(imagepath).convert('RGB')
  resized=resize_boolean_masks_nearest_exact(torch.from_numpy(native.copy()),(f['height'],f['width']))
  sr=info['teacher_files']['sam'];assert sha256_file(root/sr['path'])==sr['sha256'];teacher_sam=torch.load(root/sr['path'],weights_only=False)
  if 'native_proposal_indices' in teacher_sam:
   candidates=teacher_sam['native_proposal_indices'][q:q+1]
   assert 0<=int(candidates[0])<len(native) and torch.equal(resized[int(candidates[0])],data['masks'][q])
  else:
   candidates=torch.where((resized==data['masks'][q]).flatten(1).all(1))[0];assert len(candidates)==1
  mask=native[int(candidates[0])];ys,xs=np.where(mask);rgb=np.array(image);rgb[~mask]=127
  crop=Image.fromarray(rgb).crop((max(0,int(xs.min())-3),max(0,int(ys.min())-3),min(image.width,int(xs.max())+4),min(image.height,int(ys.max())+4)))
  path=root/'cold_query_crop.png';crop.save(path);f['image']={'path':str(path),'sha256':sha256_file(path)}
 if modality=='prompt':
  f['prompt_image']={'path':str(imagepath),'sha256':src['source_sha256']};f['seed_rows']=data['queries']['seed_rows'][q];f['seed_pixels']=data['queries']['seed_pixels'][q];f['prompt_world_normalization']=data['queries']['prompt_world_normalization']
path=root/'cold_query_fixture.pt'
if path.exists():raise FileExistsError(path)
torch.save(f,path)
