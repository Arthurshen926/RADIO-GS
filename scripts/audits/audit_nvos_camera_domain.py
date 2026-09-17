"""Measure projection disagreement using source-only sparse tracks, not GT depth."""
import argparse,json
from pathlib import Path
import torch
from radio_gs.v4.geometry.fuse_lerf_moge3 import _read_images,_read_cameras_binary,_read_strict_source_points
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 p=argparse.ArgumentParser();p.add_argument('--sources',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 results=[]
 for row in json.loads(a.sources.read_text())['scenes']:
  authority=json.loads(Path(row['source_authority']).read_text());names=authority['camera_name_to_frame_index'];root=Path(row['scene_root']);dense=Path(row['matching_sparse'])
  views=_read_images(dense/'images.bin',_read_cameras_binary(dense/'cameras.bin'),names)
  original=_read_images(root/'sparse/0/images.bin',_read_cameras_binary(root/'sparse/0/cameras.bin'),names)
  source={int(i) for i in authority['frame_indices']};points,counts=_read_strict_source_points(dense/'points3D.bin',{views[i].image_id for i in source})
  errors=[];shifts=[]
  for i in sorted(source):
   v=views[i];old=original[i];selected=[j for j,pid in enumerate(v.point_ids.tolist()) if pid in points]
   if not selected:continue
   selected=selected[::max(1,len(selected)//1000)]
   xyz=torch.stack([points[int(v.point_ids[j])] for j in selected]);cam=xyz@v.world_to_camera[:3,:3].T+v.world_to_camera[:3,3]
   valid=cam[:,2]>0;cam=cam[valid];xy=v.xy[selected][valid]
   pix=cam@v.intrinsic.T;pix=pix[:,:2]/pix[:,2:]
   # The previous builder would scale original K to the released raster.
   k=old.intrinsic.clone();k[0]*=v.width/old.width;k[1]*=v.height/old.height
   wrong=cam@k.T;wrong=wrong[:,:2]/wrong[:,2:]
   errors.append((pix-xy).norm(dim=-1));shifts.append((wrong-pix).norm(dim=-1))
  if not errors:raise ValueError('no strict-source calibration correspondences')
  error=torch.cat(errors);shift=torch.cat(shifts)
  results.append({'scene':row['scene'],'source_only_track_filter':counts,'sampled_correspondences':len(error),'correct_camera_reprojection_px_median':float(error.median()),'incorrect_scaled_original_projection_shift_px_median':float(shift.median()),'incorrect_scaled_original_projection_shift_px_p95':float(torch.quantile(shift,.95))})
 a.output.write_text(json.dumps({'sources_sha256':sha256_file(a.sources),'scenes':results,'target_rgb_or_gt_opened':False,'metric_domain':'source sparse feature reprojection; not geometry accuracy ground truth'},indent=2)+'\n');print(results)
if __name__=='__main__':main()
