"""Same evidence/geometry/dimension; change only overlapping region ownership."""
import argparse,json,os,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--gpu',required=True);p.add_argument('--output',required=True);a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
 import torch
 from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
 from radio_gs.pipeline.native_reconstruction import NativeFieldAccumulator
 from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks
 torch.set_num_threads(4);start=time.monotonic()
 scene=json.loads(Path(f'configs/pipeline/scannet_native_calibrated/{a.scene}.scene.json').read_text());g=scene['geometry'];geo=FrozenGaussianGeometry.load(g['path'],expected_sha256=g['sha256'],device='cuda')
 acc=NativeFieldAccumulator(geo,1536,region_policy='finest');cameras=json.loads(Path(scene['source_cameras']).read_text())['records'];records=[]
 with torch.inference_mode():
  for c in cameras:
   row=torch.load(Path(scene['native_region_directories'][0])/f"frame_{c['frame_index']:05d}.pt",map_location='cpu',weights_only=False)
   sam=torch.load(row['sam_path'],map_location='cpu',weights_only=False);masks=unpack_masks(sam['packed_masks'],sam['mask_shape'][1])[row['native_proposal_indices'].numpy()]
   height,width=round(c['height']*256/c['width']),256
   record=acc.add_view(masks.copy(),row['descriptors'],c['intrinsic'],c['world_to_camera'],height=height,width=width,image_height=c['height'],image_width=c['width']);records.append(dict(frame=c['frame_index'],**record))
   print(a.scene,c['frame_index'],flush=True)
  field=acc.field(g,'siglip')
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);torch.save(field,out/'native_field.pt');scene.update(native_field=str(out/'native_field.pt'),region_policy='finest');(out/'scene.json').write_text(json.dumps(scene,indent=2)+'\n');(out/'build.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,records=records,query_independent=True,old_features_read=False,descriptors_reencoded=False),indent=2)+'\n')
if __name__=='__main__':main()
