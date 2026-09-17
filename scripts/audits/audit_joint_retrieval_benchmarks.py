"""Inspect real benchmark assets before extending a field across output contracts."""
import json
from pathlib import Path
from PIL import Image
from radio_gs.scannet_constants import OPENGAUSSIAN_NYU40_CLASS_SPLITS
from radio_gs.utils.immutable_artifacts import sha256_file

def main():
 out=Path('paper/artifacts/joint_retrieval_contracts_20260915.json')
 if out.exists():raise FileExistsError(out)
 scannet=[];root=Path('/mnt/pool/sqy/3d_understanding/scannet_og')
 for scene in sorted(root.glob('scene*')):
  if not scene.is_dir():continue
  train=scene/'transforms_train.json';test=scene/'transforms_test.json';x=json.loads(train.read_text());y=json.loads(test.read_text());tx={v['file_path'] for v in x['frames']};ty={v['file_path'] for v in y['frames']};gt=scene/(scene.name+'_vh_clean_2.labels.ply')
  if not gt.exists():raise ValueError('missing semantic 3D truth')
  scannet.append({'scene':scene.name,'train_camera_sha256':sha256_file(train),'test_camera_sha256':sha256_file(test),'train_frames':len(tx),'test_frames':len(ty),'overlapping_frame_paths':len(tx&ty),'is_disjoint_view_test':not bool(tx&ty),'labels':str(gt),'labels_sha256':sha256_file(gt),'new_frozen_moge_retrieval_bundle_ready':False})
 manifest=Path('/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/manifests/nvos_strict_unseen_v1.json');d=json.loads(manifest.read_text());nvos=[]
 for s in d['scenes']:
  train={f['frame_id'] for f in s['training_frames']};target=set(s['evaluation_frame_ids']);assert not train&target;prompt=s['prompt'];records={f['frame_id']:f for f in s['frames']};assert prompt['frame_id'] not in target;assert records[prompt['frame_id']]['rgb_path']
  shapes={}
  for sign in ['positive','negative']:
   p=Path(prompt[sign+'_path']);assert sha256_file(p)==d['protocol']['prompt_asset_sha256'][s['scene_id']][sign];shapes[sign]=list(Image.open(p).size)
  assert shapes['positive']==shapes['negative'];nvos.append({'scene':s['scene_id'],'source_frames':len(train),'reference_frame':prompt['frame_id'],'target_frames':sorted(target),'target_rgb_excluded':True,'official_signed_scribble_size_wh':shapes['positive'],'new_frozen_moge_retrieval_bundle_ready':False})
 result={'schema':'joint_retrieval_contracts.v1','scannet':{'task':'categorical 3D semantic point labeling','raw_label_space':'NYU40','class_splits':OPENGAUSSIAN_NYU40_CLASS_SPLITS,'readout':'all prescribed class queries -> per-element argmax -> geometry-bound point transfer','unobserved_prediction':0,'gt_usage':'scoring only; never select per-scene query vocabulary using GT presence','camera_split_warning':'local train/test transforms overlap; do not claim heldout-view evaluation. A transductive 3D point task is distinct.','scenes':scannet},'nvos':{'manifest':str(manifest),'manifest_sha256':sha256_file(manifest),'task':'official reference positive/negative scribble -> independent target binary mask','prompt_input':'2D signed scribbles, not fabricated 3D clicks','strict_query_rgb':'reference only; target forbidden','sam_reference_parser':'may infer a transient reference mask and native image query from authorized reference RGB and prompts','rgb_assisted_alternative':'target-RGB SAM/video is a distinct protocol, never combined with strict results','scenes':nvos},'lerf':{'current_task':'2D category-union development text/reference queries','official_point_image_protocol':False},'promotion':'new matched field must pass all three task contracts; LERF gains alone cannot promote','new_joint_benchmark_metrics_available':False}
 out.write_text(json.dumps(result,indent=2)+'\n');print('ScanNet scenes',len(scannet),'overlap counts',[r['overlapping_frame_paths'] for r in scannet],'NVOS scenes',len(nvos))
if __name__=='__main__':main()
