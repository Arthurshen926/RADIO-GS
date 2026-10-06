"""Uncapped legal RGB cohorts, independent of any teacher cache intersection."""
import argparse,json
from pathlib import Path
import cv2,numpy as np,torch
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.utils.immutable_artifacts import sha256_file

def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();torch.set_num_threads(4)
 data=torch.load(Path('output/taskwise_scope_composition_next38')/('lerf_'+a.scene)/'field.pt',map_location='cpu',weights_only=False)
 root=Path(data['root']);info=json.loads((root/'teachers.json').read_text());src=json.loads((root/'source_frames.json').read_text());gr=json.loads(Path(data['geometry_report']).read_text());gp=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256'])
 targets={int(f.stem.split('_')[-1]) for f in (Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label')/a.scene).glob('frame_*.json')};excluded=targets|set(info['validation_frames'])
 paths={int(f.stem.split('_')[-1]):f for f in Path(src['image_dir']).glob('frame_*.jpg')};frames=sorted(set(paths)&set(gp.cameras)-excluded)
 positions=[];directions=[];sharpness=[];records=[]
 for f in frames:
  pose=torch.linalg.inv(gp.cameras[f].world_to_camera).numpy();positions.append(pose[:3,3]);directions.append(pose[:3,2]);im=cv2.imread(str(paths[f]),cv2.IMREAD_GRAYSCALE);im=cv2.resize(im,(256,192));sharpness.append(float(cv2.Laplacian(im,cv2.CV_32F).var()));records.append(dict(image_id=f'frame_{f:05d}',path=str(paths[f]),sha256=sha256_file(paths[f]),rgb_role='registered_source_or_mapping_view'))
 pos=np.array(positions);pos=(pos-np.median(pos,axis=0))/max(float(np.linalg.norm(pos-np.median(pos,axis=0),axis=1).mean()),1e-8);embedding=np.concatenate([pos,np.array(directions)],axis=1)
 # Retain every original view, then fill pose-space gaps, mildly favoring sharp RGB.
 selected=[frames.index(f) for f in info['train_frames']];quality=np.clip(np.array(sharpness)/np.median(sharpness),.5,2.)**.25
 while len(selected)<min(64,len(frames)):
  distance=((embedding[:,None]-embedding[selected][None])**2).sum(-1).min(1);distance[selected]=-1;selected.append(int(np.argmax(distance*quality)))
 groups=dict(old=info['train_frames'],coverage64=sorted(frames[i] for i in selected),all=frames);out=Path('output/source_coverage_next72')/a.scene;out.mkdir(parents=True,exist_ok=True)
 for name,ids in groups.items():
  authority=dict(schema_version=1,contract='sam3-query-free-source-rgb-authority-v1',information_policy=dict(registered_source_rgb_only=True,query_text_used=False,benchmark_ground_truth_used=False,target_or_evaluation_rgb_used=False),images=[records[frames.index(f)] for f in ids],construction=dict(selection=name,maximum_images=None if name=='all' else len(ids),selection_uses='camera pose and RGB sharpness, seeded by original cohort',validation_frames=info['validation_frames'],excluded_target_frame_ids=sorted(targets),cache_intersection_used=False))
  (out/(name+'.json')).write_text(json.dumps(authority,indent=2)+'\n')
 (out/'cohorts.json').write_text(json.dumps(dict(scene=a.scene,frames=groups,validation_frames=info['validation_frames'],missing_camera_frames=sorted(set(paths)-set(gp.cameras)-excluded),sharpness=dict(zip(frames,sharpness)),coverage_type='camera pose diversity; not measured occlusion-aware surface coverage'),indent=2)+'\n');print(a.scene,{k:len(v) for k,v in groups.items()},flush=True)
if __name__=='__main__':main()
