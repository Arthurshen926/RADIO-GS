"""Encode whole source-grounded masks instead of legacy fragment crops."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from radio_gs.scripts.build_multiscale_sam_mask_aligned_crop_summary_teacher import NativeSiglip2Runtime
from radio_gs.utils.immutable_artifacts import sha256_file

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--grounding',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',required=True);p.add_argument('--split-components',action='store_true');a=p.parse_args();torch.set_num_threads(4)
 if a.output.exists() or a.output.with_suffix('.json').exists():raise FileExistsError(a.output)
 info=json.loads((a.root/'teachers.json').read_text());source=json.loads((a.root/'source_frames.json').read_text());identity=json.loads((a.root/'native_query_encoder.json').read_text());rec=json.loads(a.grounding.with_suffix('.json').read_text());assert sha256_file(a.grounding)==rec['sha256'];g=torch.load(a.grounding,weights_only=False);assert g['frames']==info['train_frames'] and g['teacher_manifest_sha256']==sha256_file(a.root/'teachers.json');src={x['frame_idx']:x for x in source['frames']}
 runtime=NativeSiglip2Runtime.load(Path(identity['siglip_root']),device=torch.device(a.device));masks=[];views=[];descriptors=[];records=[]
 with torch.inference_mode():
  for i,frame in enumerate(g['frames']):
   path=Path(source['image_dir'])/src[frame]['source_file'];assert sha256_file(path)==src[frame]['source_sha256'];im=Image.open(path).convert('RGB');crops=[]
   for qi,p in enumerate(g['probability'][i]):
    m=p>.8
    if m.sum()<8:continue
    from radio_gs.training.source_region_components import connected_source_regions
    components=connected_source_regions(m) if a.split_components else [m]
    for component_index,m in enumerate(components):
     # De-duplicate identical supports, never merge disconnected appearances.
     if crops and any(torch.equal(m,x) for x in masks[-len(crops):]):continue
     native=np.asarray(Image.fromarray(m.numpy().astype('uint8')*255).resize(im.size,Image.Resampling.NEAREST))>0
     yy,xx=np.where(native);rgb=np.asarray(im).copy();rgb[~native]=127;box=(max(0,int(xx.min())-3),max(0,int(yy.min())-3),min(im.width,int(xx.max())+4),min(im.height,int(yy.max())+4));crop=Image.fromarray(rgb).crop(box).resize((384,384),Image.Resampling.BICUBIC);crops.append(pil_to_tensor(crop).float()/255);masks.append(m);views.append(info['frames'].index(frame));records.append({'frame':frame,'grounding_query_index':qi,'component_index':component_index,'source_sha256':src[frame]['source_sha256']})
   if crops:descriptors.append(torch.cat([runtime.encode(torch.stack(crops[k:k+8]).to(a.device)).float().cpu() for k in range(0,len(crops),8)]))
   print('frame',frame,'masks',len(crops),flush=True)
 if not masks:raise ValueError('no grounded source regions')
 bundle={'schema':'grounded_native_region_inventory.v1','split_components':a.split_components,'component_semantics':'connected appearance support, not verified physical instance','masks':torch.stack(masks),'proposal_views':torch.tensor(views),'descriptors':torch.cat(descriptors).half(),'geometry_binding':g['geometry_binding'],'teacher_manifest_sha256':g['teacher_manifest_sha256'],'train_frames':g['frames'],'grounding_sha256':rec['sha256'],'source_records':records,'benchmark_data_used':False,'implementation_sha256':sha256_file(__file__)};torch.save(bundle,a.output);a.output.with_suffix('.json').write_text(json.dumps({'schema':bundle['schema'],'sha256':sha256_file(a.output),'regions':len(masks),'source_records':records,'benchmark_data_used':False},indent=2)+'\n')
if __name__=='__main__':main()
