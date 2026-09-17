"""Re-encode source regions and language with one functional replacement teacher."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from radio_gs.models.open_clip_native import OpenClipNativeRuntime
from radio_gs.models.teacher_capabilities import TeacherCapabilities,validate_teacher_plan,validate_semantic_space
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.utils.mask_resampling import resize_boolean_masks_nearest_exact


def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--source-root',type=Path);p.add_argument('--output-root',type=Path);p.add_argument('--device',default='cpu');a=p.parse_args();torch.set_num_threads(4)
 base=Path('/mnt/pool/sqy/results/RADIO-GS/output');src=a.source_root or base/'preserved_source_inventory_20260916'/a.scene;out=a.output_root or base/'openclip_replacement_20260917'/a.scene;out.mkdir(parents=True,exist_ok=False)
 info=json.loads((src/'teachers.json').read_text());auth=json.loads((src/'responsibility.json').read_text());sources=json.loads((src/'source_frames.json').read_text());records={r['frame_idx']:r for r in sources['frames']}
 checkpoint=next(Path('/root/.cache/huggingface/hub/models--laion--CLIP-ViT-B-16-laion2B-s34B-b88K/snapshots').glob('*/open_clip_model.safetensors'))
 runtime=Path('/mnt/pool/sqy/results/RADIO-GS/runtime/open_clip_2_32');configuration=dict(runtime_root=str(runtime),checkpoint_path=str(checkpoint),model_name='ViT-B-16')
 oldidentity=json.loads((src/'native_query_encoder.json').read_text());files=[r for r in oldidentity['files'] if 'siglip' not in r['path']]
 for path in [checkpoint.resolve(),*sorted((runtime/'open_clip').rglob('*.py')),runtime/'open_clip/model_configs/ViT-B-16.json',runtime/'open_clip/bpe_simple_vocab_16e6.txt.gz',Path('radio_gs/models/open_clip_native.py').resolve(),Path('radio_gs/models/functional_query_encoder.py').resolve()]:
  files.append(dict(path=str(path.resolve()),sha256=sha256_file(path.resolve())))
 identity={k:v for k,v in oldidentity.items() if k not in ['siglip_root','files','text','image']};identity.update(semantic_backend='open_clip',open_clip=configuration,files=files,text='OpenCLIP ViT-B-16 native text, normalized',image='OpenCLIP ViT-B-16 native masked crop, official preprocessing, normalized')
 capability=TeacherCapabilities('open_clip',frozenset({'text_image_alignment'}),sha256_file(checkpoint.resolve()),512);validate_semantic_space(capability,capability)
 plan=validate_teacher_plan([capability,TeacherCapabilities('dino',frozenset({'visual_correspondence'})),TeacherCapabilities('sam',frozenset({'boundary'})),TeacherCapabilities('moge',frozenset({'geometry'}))],['text','image','prompt'])
 sam=torch.load(src/info['teacher_files']['sam']['path'],weights_only=False);queries=torch.load(src/info['query_file']['path'],weights_only=False)
 native=OpenClipNativeRuntime(configuration,a.device);descriptors=torch.empty(len(sam['masks']),512);views=sam['view_indices']
 for vi,frame in enumerate(info['frames']):
  rec=records[frame];path=Path(sources['image_dir'])/rec['source_file'];assert sha256_file(path)==rec['source_sha256'];image=Image.open(path).convert('RGB');rgb=np.asarray(image)
  ids=torch.where(views==vi)[0];masks=resize_boolean_masks_nearest_exact(sam['masks'][ids],(image.height,image.width)).numpy();crops=[]
  for mask in masks:
   ys,xs=np.where(mask);assert len(ys);masked=rgb.copy();masked[~mask]=127;box=(max(0,int(xs.min())-3),max(0,int(ys.min())-3),min(image.width,int(xs.max())+4),min(image.height,int(ys.max())+4));crops.append(Image.fromarray(masked).crop(box))
  for start in range(0,len(ids),8):descriptors[ids[start:start+8]]=native.encode_images(crops[start:start+8]).cpu()
  print(a.scene,frame,len(ids),flush=True)
 text=native.encode_text(queries['queries']).cpu().half();descriptors=descriptors.half()
 names=['responsibility.json','source_frames.json',*['teacher_'+n+'.pt' for n in ['radio','dino','sam']],*[r['relative_path'] for r in auth['views']]]
 for name in names:
  path=out/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src/name,path)
 torch.save(dict(descriptors=descriptors,text_embeddings=text,queries=queries['queries'],encoder=identity,functional_role='text_image_alignment',provider='open_clip'),out/'teacher_siglip.pt')
 queries=dict(queries,image=descriptors,text=text);torch.save(queries,out/'queries.pt');(out/'native_query_encoder.json').write_text(json.dumps(identity,indent=2)+'\n')
 info['teacher_files']['siglip']={'path':'teacher_siglip.pt','sha256':sha256_file(out/'teacher_siglip.pt')};info['query_file']={'path':'queries.pt','sha256':sha256_file(out/'queries.pt')};info['functional_semantic_provider']='open_clip';info['legacy_siglip_slot_is_functional_alignment']=True
 (out/'teachers.json').write_text(json.dumps(info,indent=2)+'\n')
 # SAM3 RGB/string predictions do not consume the replaced embedding space.
 ground=torch.load(src/'source_grounded_text.pt',weights_only=False);assert ground['queries']==queries['queries'];assert ground['frames']==info['train_frames'];assert ground['teacher_manifest_sha256']==sha256_file(src/'teachers.json')
 ground=dict(ground,teacher_manifest_sha256=sha256_file(out/'teachers.json'),functional_teacher_rebinding=dict(original_path=str(src/'source_grounded_text.pt'),original_sha256=sha256_file(src/'source_grounded_text.pt'),rgb_text_sam_predictions_unchanged=True,native_embedding_not_input_to_sam_prediction=True))
 torch.save(ground,out/'source_grounded_text.pt');(out/'source_grounded_text.json').write_text(json.dumps({'sha256':sha256_file(out/'source_grounded_text.pt')},indent=2)+'\n')
 report=dict(schema='functional_openclip_source.v1',source_only=True,benchmark_data_used=False,capability_plan=plan,semantic_dimension=512,source_masks_unchanged=True,crop_contract='stored 96x128 source masks upsampled nearest-exact; masked gray127, 3px padded box; official OpenCLIP processing',comparison_limitation='old SigLIP inventory used original-resolution masks before storage; not a pure backbone-quality ablation',implementation_sha256=sha256_file(__file__),checkpoint_sha256=sha256_file(checkpoint.resolve()),teacher_manifest_sha256=sha256_file(out/'teachers.json'),query_identity_sha256=sha256_file(out/'native_query_encoder.json'))
 (out/'replacement_source.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
