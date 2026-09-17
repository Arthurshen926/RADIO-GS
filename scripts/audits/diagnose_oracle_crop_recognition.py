"""GT crop recognition oracle: isolates query adapter without localization claims."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
import numpy as np
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from radio_gs.models.native_query_encoders import NativeThreeQueryEncoder
from radio_gs.models.element_retrieval_field import load_element_retrieval_field
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.v4.evaluation.lerf_common import load_labels,ground_truth_masks
from radio_gs.utils.immutable_artifacts import sha256_file

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--scene',required=True);p.add_argument('--run',required=True);p.add_argument('--device',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
 if a.output.exists() or a.output.with_suffix('.protocol.json').exists():raise FileExistsError(a.output)
 root=a.root;info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());source=json.loads((root/'source_frames.json').read_text());identity=json.loads((root/'native_query_encoder.json').read_text());r=json.loads((root/a.run/'report.json').read_text());fixture=torch.load(root/'cold_query_fixture.pt',map_location='cpu',weights_only=False)
 geometry=FrozenGaussianGeometry.load(fixture['geometry_path'],expected_sha256=fixture['geometry_sha256'],device=a.device);field,_=load_element_retrieval_field(root/a.run/'field.pt',expected_sha256=r['field_sha256'],geometry=geometry,expected_query_encoder=identity,device=a.device)
 if getattr(field,'native_decoder',None) is not None:raise ValueError('independent query-adapter diagnostic is not applicable to native decoder readout; compare decoded elements in native space')
 labelroot=Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label');annotations,categories,sh,sw=load_labels(labelroot,a.scene);assert not set(annotations)&set(info['train_frames'])
 protocol={'schema':'oracle_crop_recognition.v3','random_projection_control_seeds':[0,1,2,3,4],'random_dimensions':[64,128,256,512,1024],'target_rgb_and_gt_crop_used':True,'localization_or_deployment_metric':False,'training_updates':False,'query_template':'verbatim','crop':'GT category union bbox3px; paired masked gray127 and unmasked identical bbox','field_sha256':r['field_sha256'],'implementation_sha256':sha256_file(__file__),'labels':{str(x):sha256_file(x) for x in sorted((labelroot/a.scene).glob('*.json'))}}
 a.output.with_suffix('.protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');enc=NativeThreeQueryEncoder(identity,a.device);q=enc.encode_text(categories).half().float();crops=[];meta=[]
 for frame,objects in sorted(annotations.items()):
  path=Path(source['image_dir'])/f'frame_{frame:05d}.jpg';im=Image.open(path).convert('RGB');targets=ground_truth_masks(objects,categories,im.height,im.width,sh,sw)
  for ci,c in enumerate(categories):
   mask=targets[c].numpy()
   if not mask.any():continue
   yy,xx=np.where(mask);box=(max(0,int(xx.min())-3),max(0,int(yy.min())-3),min(im.width,int(xx.max())+4),min(im.height,int(yy.max())+4));rgb=np.asarray(im).copy();rgb[~mask]=127
   for kind,img in [('masked',Image.fromarray(rgb)),('unmasked',im)]:
    crops.append(pil_to_tensor(img.crop(box).resize((384,384),Image.Resampling.BICUBIC)).float()/255);meta.append({'frame':frame,'category':c,'category_index':ci,'crop':kind,'rgb_sha256':sha256_file(path)})
 from radio_gs.scripts.build_multiscale_sam_mask_aligned_crop_summary_teacher import NativeSiglip2Runtime
 vision=NativeSiglip2Runtime.load(Path(identity['siglip_root']),device=torch.device(a.device))
 with torch.no_grad():
  image=torch.cat([vision.encode(torch.stack(crops[i:i+8]).to(a.device)).float() for i in range(0,len(crops),8)]);native=F.normalize(image,dim=-1)@q.T;adapted=F.normalize(field.query_adapters['image'](image),dim=-1)@F.normalize(field.query_adapters['text'](q),dim=-1).T
  stages={'native':native,'adapted':adapted}
  for dim in (64,128,256,512,1024):
   for seed in range(5):
    gen=torch.Generator(device=a.device).manual_seed(seed);projection=torch.randn(image.shape[1],dim,generator=gen,device=a.device)/(dim**.5)
    stages[f'random{dim}_seed{seed}']=F.normalize(image@projection,dim=-1)@F.normalize(q@projection,dim=-1).T
  # Persist explicitly privileged diagnostic encodings for subsequent read-only
  # controls, never into the field or source training assets.
  torch.save({'schema':'privileged_oracle_diagnostic_encodings.v1','queries':q.cpu(),'images':image.cpu(),'metadata':meta,'categories':categories,'not_training_or_deployment':True},a.output.with_suffix('.encodings.pt'))
  for stage,scores in stages.items():
   for row,score in zip(meta,scores):
    order=score.argsort(descending=True);ci=row['category_index'];rank=int(torch.where(order==ci)[0][0])+1;other=score.clone();other[ci]=-float('inf');row[stage]={'rank':rank,'top_category':categories[int(order[0])],'target_cosine':float(score[ci]),'margin':float(score[ci]-other.max())}
 macro={}
 for kind in ('masked','unmasked'):
  selected=[x for x in meta if x['crop']==kind];macro[kind]={stage:{'top1':sum(x[stage]['rank']==1 for x in selected)/len(selected),'mrr':sum(1/x[stage]['rank'] for x in selected)/len(selected),'tasks':len(selected)} for stage in stages}
 a.output.write_text(json.dumps({'protocol':protocol,'macro':macro,'observations':meta},indent=2)+'\n');print(macro,flush=True)
if __name__=='__main__':main()
