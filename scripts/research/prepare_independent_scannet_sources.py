"""Frozen-camera scenes outside the ten-scene mechanism-development cohort."""
import argparse,json,urllib.request
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from plyfile import PlyData
from radio_gs.geometry.registered_mesh_carrier import build_registered_mesh_carrier
from radio_gs.utils.immutable_artifacts import sha256_file

ROOT=Path('output/independent_source_reliability')
SCENES=['scene0050_02','scene0231_01','scene0412_00']

def cohort_arguments():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--scenes',nargs='+',default=SCENES)
    return p.parse_args()

def main():
    torch.set_num_threads(4)
    ROOT.mkdir(parents=True,exist_ok=True)
    metadata={
        'scannetv2_train.txt':'https://raw.githubusercontent.com/ScanNet/ScanNet/master/Tasks/Benchmark/scannetv2_train.txt',
        'scannetv2_val.txt':'https://raw.githubusercontent.com/ScanNet/ScanNet/master/Tasks/Benchmark/scannetv2_val.txt',
        'scannetv2-labels.combined.tsv':'https://raw.githubusercontent.com/facebookresearch/votenet/main/scannet/meta_data/scannetv2-labels.combined.tsv'}
    for filename,url in metadata.items():
        if not (ROOT/filename).exists():urllib.request.urlretrieve(url,ROOT/filename)
    source=Path('/mnt/pool/sqy/3d_understanding/ScanNet-PFIR-Small')
    for name in SCENES:
        raw=source/'frames_dense20'/name;out=ROOT/'sources'/name;out.mkdir(parents=True,exist_ok=True)
        k=np.loadtxt(raw/'intrinsics_color.txt')[:3,:3];extrinsic=np.loadtxt(raw/'extrinsics_color.txt')
        records=[];images=[]
        for image in sorted((raw/'color').glob('*.jpg')):
            frame=int(image.stem);pose=np.loadtxt(raw/'pose'/f'{frame:06d}.txt')
            if not np.isfinite(pose).all():continue
            w,h=Image.open(image).size;digest=sha256_file(image)
            records.append(dict(frame_index=frame,source_image=str(image),source_sha256=digest,
                width=w,height=h,intrinsic=k.tolist(),world_to_camera=np.linalg.inv(pose@extrinsic).tolist()))
            images.append(dict(image_id=f'frame_{frame:05d}',path=str(image),sha256=digest,rgb_role='registered_source_or_mapping_view'))
        (out/'source_cameras.json').write_text(json.dumps(dict(records=records,selection='all exported stride20 RGB with finite poses; no additional frame cap'),indent=2)+'\n')
        (out/'source_rgb.json').write_text(json.dumps(dict(schema_version=1,contract='sam3-query-free-source-rgb-authority-v1',
            information_policy=dict(registered_source_rgb_only=True,query_text_used=False,benchmark_ground_truth_used=False,target_or_evaluation_rgb_used=False),images=images,maximum_images=None),indent=2)+'\n')
        mesh=source/'annotations'/name/f'{name}_vh_clean_2.ply';ply=PlyData.read(mesh)
        xyz=np.stack([ply['vertex'][c] for c in ['x','y','z']],1);faces=np.stack(ply['face']['vertex_indices'])
        model,ids=build_registered_mesh_carrier(xyz,faces);geometry=out/'geometry.pt'
        torch.save(dict(schema='radio_gs.frozen_surface_gaussian_geometry.v1',geometry_trainable_during_semantics=False,
            state_dict=model.state_dict(),input_track='provided_unlabelled_mesh',source_vertex_ids=torch.from_numpy(ids)),geometry)
        (out/'geometry_report.json').write_text(json.dumps(dict(geometry=str(geometry),geometry_sha256=sha256_file(geometry),
            source_mesh=str(mesh),source_frames=len(records),elements=len(ids),semantic_labels_used=False),indent=2)+'\n')
        provider=dict(scene='scannet_'+name,geometry=dict(path=str(geometry),sha256=sha256_file(geometry)),
            text_encoder=json.loads(Path('configs/pipeline/scannet_identity_observations/scene0062_00.scene.json').read_text())['text_encoder'],
            source_cameras=str(out/'source_cameras.json'),source_frames=[r['frame_index'] for r in records],source_rgb_authority=str(out/'source_rgb.json'))
        (out/'provider_scene.json').write_text(json.dumps(provider,indent=2)+'\n')
        print(name,len(records),'views',len(ids),'elements',flush=True)
    train=set((ROOT/'scannetv2_train.txt').read_text().split())
    fit=sorted(p.name.removesuffix('.scene.json') for p in Path('configs/pipeline/scannet_identity_observations').glob('*.scene.json') if p.name.removesuffix('.scene.json') in train)
    (ROOT/'split.json').write_text(json.dumps(dict(fit=fit,validation=SCENES,
        fit_policy='official training subset from old development scenes; their scores are no longer held-out for this supervised branch',
        validation_policy='official validation scenes, distinct room prefixes, not used to design this mechanism; previous PFIR use not denied',
        parameters_fixed_before_validation=True,training_labels_used=True,original_zero_label_track_unchanged=True),indent=2)+'\n')

if __name__=='__main__':
    args=cohort_arguments();ROOT=args.root;SCENES=args.scenes
    main()
