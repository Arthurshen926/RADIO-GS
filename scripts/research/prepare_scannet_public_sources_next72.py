"""Uncapped public ScanNet source cameras and unlabelled provided-mesh carriers."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from plyfile import PlyData
from radio_gs.geometry.registered_pinhole import blender_camera_to_opencv
from radio_gs.geometry.registered_mesh_carrier import build_registered_mesh_carrier
from radio_gs.utils.immutable_artifacts import sha256_file

def main():
 p=argparse.ArgumentParser();p.add_argument('--scenes',default='scene0200_00,scene0645_00');a=p.parse_args();torch.set_num_threads(4)
 for scene in a.scenes.split(','):
  root=Path('/mnt/pool/sqy/3d_understanding/scannet_og')/scene;out=Path('output/scannet_public_next72')/scene;out.mkdir(parents=True,exist_ok=True);camera_file=root/'transforms_train.json';data=json.loads(camera_file.read_text());records=[];images=[]
  for frame in data['frames']:
   image=(root/(frame['file_path']+'.jpg')).resolve();camera=blender_camera_to_opencv(frame,data);digest=sha256_file(image);fid=int(image.stem);records.append(dict(frame_index=fid,source_image=str(image),source_sha256=digest,**{k:v.tolist() if torch.is_tensor(v) else v for k,v in camera.items()}));images.append(dict(image_id=f'frame_{fid:05d}',path=str(image),sha256=digest,rgb_role='registered_source_or_mapping_view'))
  records.sort(key=lambda r:r['frame_index']);authority=out/'source_cameras.json';authority.write_text(json.dumps(dict(schema='registered_pinhole_source_authority.v1',scene=scene,records=records,source_camera_file=str(camera_file),source_camera_sha256=sha256_file(camera_file),selection='all provided transforms_train.json frames, no cache intersection',source_frame_indices=[r['frame_index'] for r in records],independent_view_test=False,semantic_labels_opened=False,benchmark_queries_used=False),indent=2)+'\n')
  (out/'source_rgb.json').write_text(json.dumps(dict(schema_version=1,contract='sam3-query-free-source-rgb-authority-v1',information_policy=dict(registered_source_rgb_only=True,query_text_used=False,benchmark_ground_truth_used=False,target_or_evaluation_rgb_used=False),images=images,maximum_images=None),indent=2)+'\n')
  mesh=root/'points3d.ply';ply=PlyData.read(str(mesh));xyz=np.stack([ply['vertex'][k] for k in ['x','y','z']],1);faces=np.stack(ply['face']['vertex_indices']);model,ids=build_registered_mesh_carrier(xyz,faces);geometry=out/'geometry.pt';torch.save(dict(schema='radio_gs.frozen_surface_gaussian_geometry.v1',geometry_trainable_during_semantics=False,state_dict=model.state_dict(),input_track='provided_unlabelled_mesh',source_vertex_ids=torch.from_numpy(ids)),geometry)
  report=dict(scene=scene,input_track='provided_unlabelled_mesh',geometry=str(geometry.resolve()),geometry_sha256=sha256_file(geometry),source_cameras=str(authority.resolve()),source_cameras_sha256=sha256_file(authority),source_frames=len(records),elements=len(ids),source_vertex_ids_preserved=True,semantic_labels_used=False,semantic_field_built=False,official_benchmark_result=False,independent_view_test=False);(out/'geometry_report.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)
if __name__=='__main__':main()
