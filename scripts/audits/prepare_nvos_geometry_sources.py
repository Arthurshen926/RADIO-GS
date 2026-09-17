"""Seal actual NVOS source RGBs and matching undistorted COLMAP assets."""
import argparse,json,math,re
from pathlib import Path
from PIL import Image
from radio_gs.geometry.source_camera_assets import source_colmap_assets
from radio_gs.v4.geometry.fuse_lerf_moge3 import _read_cameras_binary,_read_images
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,default=Path('/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/manifests/nvos_strict_unseen_v1.json'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 d=json.loads(a.manifest.read_text());results=[]
 for s in d['scenes']:
  root=Path(s['rgb_directory']).parent;sparse,sidecar=source_colmap_assets(root);c=_read_cameras_binary(sparse/'cameras.bin');names=sorted(x.name for x in (root/'images').iterdir() if x.suffix.lower() in ('.jpg','.jpeg','.png'));name_map={name:i for i,name in enumerate(names)};views=_read_images(sparse/'images.bin',c,name_map)
  forbidden=set(s['evaluation_frame_ids']);rows=[]
  for f in s['training_frames']:
   if f['frame_id'] in forbidden:raise ValueError('target included in source geometry')
   rgb=Path(f['rgb_path']);index=name_map[rgb.name];view=views[index]
   with Image.open(rgb) as image:
    if image.size!=(view.width,view.height):raise ValueError('source RGB/camera raster mismatch')
   rows.append({'frame_index':index,'source_image':str(rgb),'source_image_sha256':sha256_file(rgb)})
  if len({r['frame_index'] for r in rows})!=len(rows):raise ValueError('ambiguous source frame identity')
  fs=[math.degrees(2*math.atan(v.width/(2*float(v.intrinsic[0,0])))) for v in [views[r['frame_index']] for r in rows]]
  if max(fs)-min(fs)>1e-4:raise ValueError('per-view FOV needed; cannot use shared batch FOV')
  out=a.output/s['scene_id'];out.mkdir(parents=True)
  authority={'schema':'nvos_geometry_source_authority.v1','camera_name_to_frame_index':name_map,'frame_indices':[r['frame_index'] for r in rows],'metadata':{'benchmark_images_opened':False,'benchmark_masks_opened':False,'text_queries_opened':False,'target_rgb_opened':False},'records':rows,'manifest':str(a.manifest),'manifest_sha256':sha256_file(a.manifest),'target_frame_ids_excluded':sorted(forbidden),'camera_assets':[{'path':str(x),'sha256':sha256_file(x)} for x in [sparse/'cameras.bin',sparse/'images.bin',sparse/'points3D.bin',sidecar]]}
  path=out/'source_authority.json';path.write_text(json.dumps(authority,indent=2)+'\n')
  original=_read_cameras_binary(root/'sparse/0/cameras.bin')
  results.append({'scene':s['scene_id'],'scene_root':str(root),'source_count':len(rows),'source_authority':str(path.resolve()),'source_authority_sha256':sha256_file(path),'fov_x_degrees':fs[0],'matching_sparse':str(sparse),'original_camera_models':[v['model_id'] for v in original.values()],'released_camera_models':[v['model_id'] for v in c.values()],'released_wh':[views[rows[0]['frame_index']].width,views[rows[0]['frame_index']].height]})
 (a.output/'report.json').write_text(json.dumps({'scenes':results,'target_rgb_or_gt_opened':False,'geometry_built':False},indent=2)+'\n')
 print([(r['scene'],r['source_count'],r['released_wh']) for r in results])
if __name__=='__main__':main()
