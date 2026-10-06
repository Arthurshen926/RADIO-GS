"""Build a query-independent native map from explicit source cameras and masks.

No source-count cap, old query prediction, calibration vocabulary or label mesh
is used. Native descriptors and visibility-weighted writes follow the LERF map
representation; the supplied unlabelled mesh remains a separate geometry track.
"""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks
from radio_gs.scripts.build_multiscale_sam_mask_aligned_crop_summary_teacher import NativeSiglip2Runtime
from radio_gs.pipeline.native_reconstruction import NativeFieldAccumulator

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--output-root',default='output/scannet_native_completion');p.add_argument('--source-root',default='output/scannet_public_next72');p.add_argument('--width',type=int,default=256);p.add_argument('--reuse-region-descriptors',action='store_true');a=p.parse_args();torch.set_num_threads(4);start=time.monotonic();out=Path(a.output_root)/a.scene;src=Path(a.source_root)/a.scene
 cameras=json.loads((src/'source_cameras.json').read_text())['records'];gr=json.loads((src/'geometry_report.json').read_text());binding=dict(path=gr['geometry'],sha256=gr['geometry_sha256']);geo=FrozenGaussianGeometry.load(binding['path'],expected_sha256=binding['sha256'],device='cuda')
 encoder='/root/.cache/huggingface/hub/models--google--siglip2-giant-opt-patch16-384/snapshots/a713301b217d38485fb2204c808367d10bc3cc40';runtime=None if a.reuse_region_descriptors else NativeSiglip2Runtime.load(Path(encoder),device=torch.device('cuda'))
 accumulator=NativeFieldAccumulator(geo,1536);regions=out/'native_regions';regions.mkdir(parents=True,exist_ok=True);records=[]
 for camera in cameras:
  frame=camera['frame_index'];sam_path=out/'source_sam'/f'frame_{frame:05d}.pt';sam=load(sam_path);masks=unpack_masks(sam['packed_masks'],sam['mask_shape'][1]);rgb=np.array(Image.open(camera['source_image']).convert('RGB'));image_h,image_w=rgb.shape[:2];native_ids=[];crops=[]
  descriptor_path=regions/f'frame_{frame:05d}.pt'
  if a.reuse_region_descriptors:
   cached=load(descriptor_path);descriptors=cached['descriptors'];native_ids=cached['native_proposal_indices'].tolist()
  else:
   for index,mask in enumerate(masks):
    ys,xs=np.where(mask)
    if not len(ys):continue
    x0,x1=max(0,int(xs.min())-3),min(image_w,int(xs.max())+4);y0,y1=max(0,int(ys.min())-3),min(image_h,int(ys.max())+4);crop=rgb[y0:y1,x0:x1].copy();crop[~mask[y0:y1,x0:x1]]=127
    crops.append(torch.from_numpy(np.array(Image.fromarray(crop).resize((384,384),Image.Resampling.BICUBIC),copy=True)).permute(2,0,1).float()/255);native_ids.append(index)
   descriptors=F.normalize(torch.cat([runtime.encode(torch.stack(crops[i:i+8])).float().cpu() for i in range(0,len(crops),8)]),dim=1) if crops else torch.empty(0,1536)
   torch.save(dict(frame=frame,descriptors=descriptors.half(),native_proposal_indices=torch.tensor(native_ids),sam_path=str(sam_path),native_mask_hw=[image_h,image_w],encoder=dict(path=encoder,native=True),crop='masked RGB bbox+3pixels gray127 background,384; all nonempty generated masks',query_independent=True),descriptor_path)
  record=accumulator.add_view(masks[native_ids].copy(),descriptors,camera['intrinsic'],camera['world_to_camera'],height=round(image_h*a.width/image_w),width=a.width,image_height=camera['height'],image_width=camera['width'])
  records.append(dict(frame=frame,**record));print(a.scene,frame,len(native_ids),'regions',record['touched_elements'],'elements',flush=True)
 field=accumulator.field(binding,'siglip')
 torch.save(field,out/'native_field.pt');scene=dict(scene='scannet_'+a.scene,native_field=str(out/'native_field.pt'),text_encoder=encoder,runtime_provider='native_category_map',input_track='provided_unlabelled_mesh',geometry=binding,source_cameras=str(src/'source_cameras.json'),source_frames=[c['frame_index'] for c in cameras],source_views=len(cameras),source_rgb_authority=str(src/'source_rgb.json'),native_region_directories=[str(regions)],calibration=dict(kind='none_category_ranking_only',query_independent_training=True),map_contract=dict(native_dimension=1536,geometry_trainable=False,all_nonempty_generated_masks=True,teacher_native=True,query_labels_used=False))
 (out/'scene.json').write_text(json.dumps(scene,indent=2)+'\n');config=Path('configs/consolidated/scannet_native');config.mkdir(parents=True,exist_ok=True);(config/(a.scene+'.scene.json')).write_text(json.dumps(scene,indent=2)+'\n');(out/'build.json').write_text(json.dumps(dict(scene=a.scene,records=records,source_count=len(cameras),supported_elements=int((accumulator.mass>0).sum()),feature_writer='radio_gs.pipeline.native_reconstruction.NativeFieldAccumulator',descriptors_reused=a.reuse_region_descriptors,elements=len(geo.xyz),seconds=time.monotonic()-start,historical_predictions_read=False,semantic_labels_read=False,query_independent=True),indent=2)+'\n')
if __name__=='__main__':main()
