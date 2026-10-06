"""Seal source RGB/cameras for the complete-method ScanNet build, without GT.

Preparation only: no geometry-quality or benchmark-completion claim. The
provided train/test JSONs overlap; they are not an independent view split.
"""
import argparse,json,sys
from pathlib import Path
import torch
from PIL import Image
from radio_gs.geometry.registered_pinhole import blender_camera_to_opencv
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,default=Path('/mnt/pool/sqy/3d_understanding/scannet_og'));p.add_argument('--output-root',type=Path,required=True);p.add_argument('--views',type=int,default=32);a=p.parse_args();torch.set_num_threads(4);assert a.views>=8;a.output_root.mkdir(parents=True,exist_ok=False)
 def guard(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)) and any(word in str(args[0]) for word in ['.labels.ply','language_features','label-filt','instance-filt']):raise RuntimeError('Semantic labels/features forbidden in source-camera preparation')
 sys.addaudithook(guard)
 contract=Path('paper/artifacts/joint_retrieval_contracts_20260915.json');scenes=json.loads(contract.read_text())['scannet']['scenes'];reports=[]
 for scene in scenes:
  root=a.data_root/scene['scene'];path=root/'transforms_train.json';assert sha256_file(path)==scene['train_camera_sha256'];data=json.loads(path.read_text());records=[]
  for frame in data['frames']:
   image=(root/(frame['file_path']+'.jpg')).resolve(strict=True)
   if image.parent!=(root/'color').resolve():raise ValueError('Source image outside registered color directory')
   c=blender_camera_to_opencv(frame,data);records.append(dict(frame_index=int(image.stem),image=image,camera=c))
  records.sort(key=lambda row:row['frame_index']);assert len({r['frame_index'] for r in records})==len(records) and len(records)>=a.views
  poses=torch.stack([r['camera']['camera_to_world'] for r in records]);centers=poses[:,:3,3];distance=torch.cdist(centers,centers).square();distance/=distance.max().clamp_min(1e-8);distance+=torch.cdist(poses[:,:3,2],poses[:,:3,2]).square();indices=[0]
  while len(indices)<a.views:
   closest=distance[:,indices].amin(1);closest[indices]=-1;indices.append(int(closest.argmax()))
  selected=[]
  for i in sorted(indices):
   r=records[i];c=r['camera']
   with Image.open(r['image']) as im:assert im.size==(c['width'],c['height']),r['image']
   selected.append(dict(frame_index=r['frame_index'],source_image=str(r['image']),source_sha256=sha256_file(r['image']),**{k:v.tolist() if torch.is_tensor(v) else v for k,v in c.items()}))
  out=a.output_root/scene['scene'];out.mkdir();report=dict(schema='registered_pinhole_source_authority.v1',scene=scene['scene'],records=selected,source_frame_indices=[r['frame_index'] for r in selected],camera_convention='OpenCV: +x right, +y down, +z forward; Blender c2w columns1,2 negated',source_camera_file=str(path),source_camera_sha256=sha256_file(path),selection='Greedy normalized center-distance plus viewing-axis distance, seed smallest numeric source frame; no semantic inputs',source_training_views= len(records),declared_train_test_overlap=scene['overlapping_frame_paths'],independent_view_test=False,semantic_labels_opened=False,benchmark_queries_used=False,geometry_built=False,implementation_sha256=sha256_file(__file__),camera_implementation_sha256=sha256_file('radio_gs/geometry/registered_pinhole.py'))
  target=out/'source_authority.json';target.write_text(json.dumps(report,indent=2)+'\n');reports.append(dict(scene=scene['scene'],path=str(target),sha256=sha256_file(target),views=len(selected)));print(scene['scene'],len(selected),flush=True)
 (a.output_root/'report.json').write_text(json.dumps(dict(schema='scannet_registered_sources.v1',scenes=reports,semantic_labels_opened=False,geometry_reconstruction_pending=True),indent=2)+'\n')
if __name__=='__main__':main()
