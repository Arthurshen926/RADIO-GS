"""Declared rendered-RGB training condition for real 3D instance supervision.

Rendering and teacher compilation use xyz/RGB only. Instance IDs are opened by
the separate learner. All retained masks and all prescribed cameras participate;
this pilot does not change the real source-view budget of any benchmark.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy import ndimage, sparse
from scipy.spatial import cKDTree
import torch


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare(root):
    source = json.loads(Path('output/independent_instance_membership/split.json').read_text())
    groups = {}
    for partition, count in [('fit', 24), ('mechanism_validation', 8)]:
        rooms = {}
        for scene in source['partitions'][partition]:
            rooms.setdefault(scene.split('_')[0], scene)
        ordered = sorted(rooms, key=lambda s: hashlib.sha256(('rendered-membership-pilot/' + s).encode()).hexdigest())
        groups[partition] = [rooms[s] for s in ordered[:count]]
    design = dict(fit=groups['fit'], calibration=groups['mechanism_validation'][:4],
        validation=groups['mechanism_validation'][4:], data_root=source['data_root'],
        parent_split='output/independent_instance_membership/split.json',
        condition='synthetic RGB views rendered from official-training xyz/RGB; real instance labels only in learner',
        pilot_budget='24 fit,4 calibration,4 validation rooms; one scan per room;12 cameras per scan',
        source_cameras=12, image_wh=[640,480], mask_grid=12,
        no_benchmark_views_queries_or_labels_used=True, default_changed=False)
    write(root/'design.json', design)
    return design


def camera_poses(xyz):
    lo, hi = np.quantile(xyz, [.02, .98], axis=0)
    z = lo[2] + min(1.5, .65*(hi[2]-lo[2]))
    occupied = xyz[np.abs(xyz[:,2]-z)<.25,:2]
    x,y = np.meshgrid(np.arange(lo[0]+.25,hi[0]-.25,.25), np.arange(lo[1]+.25,hi[1]-.25,.25))
    grid = np.column_stack([x.ravel(),y.ravel()])
    if len(grid) and len(occupied):
        grid = grid[cKDTree(occupied).query(grid)[0]>.35]
    floor=xyz[xyz[:,2]<=lo[2]+.15,:2]
    if len(grid) and len(floor):
        supported=grid[cKDTree(floor).query(grid)[0]<.3]
        if len(supported):grid=supported
    centres = []
    for fraction in [.35,.5,.65]:
        desired = lo[:2]*(1-fraction)+hi[:2]*fraction
        xy = grid[np.argmin(np.linalg.norm(grid-desired,axis=1))] if len(grid) else desired
        centres.append(np.r_[xy,z])
    for centre in centres:
        for yaw in np.arange(4)*np.pi/2:
            forward = np.array([np.sin(yaw),np.cos(yaw),-.12]);forward/=np.linalg.norm(forward)
            right = np.cross(forward,[0.,0.,1.]);right/=np.linalg.norm(right)
            down = np.cross(forward,right)
            yield centre, np.column_stack([right,down,forward])


def render(xyz, rgb, centre, rotation):
    w,h = 640,480; f=420.
    camera = (xyz-centre)@rotation
    ids = np.flatnonzero((camera[:,2]>.2)&(camera[:,2]<8.))
    u = np.rint(f*camera[ids,0]/camera[ids,2]+w/2).astype(int)
    v = np.rint(f*camera[ids,1]/camera[ids,2]+h/2).astype(int)
    inside=(u>=0)&(u<w)&(v>=0)&(v<h);ids,u,v=ids[inside],u[inside],v[inside]
    pixel=v*w+u;depth=camera[ids,2]
    if not len(ids):
        return Image.fromarray(np.full((h,w,3),127,np.uint8)),dict(
            vertices=np.empty(0,np.int32),u=np.empty(0,np.int32),v=np.empty(0,np.int32)),0.
    order=np.lexsort((depth,pixel));unique=np.r_[True,np.diff(pixel[order])!=0]
    owner=np.full(h*w,-1,np.int32);owner[pixel[order[unique]]]=ids[order[unique]];owner=owner.reshape(h,w)
    distance,near=ndimage.distance_transform_edt(owner<0,return_indices=True)
    fill=(owner<0)&(distance<=3)
    owner[fill]=owner[near[0][fill],near[1][fill]]
    image=np.full((h,w,3),127,np.uint8);visible=owner>=0;image[visible]=rgb[owner[visible]]
    zbuffer=np.zeros((h,w),np.float32);zbuffer[visible]=camera[owner[visible],2]
    accepted=np.abs(depth-zbuffer[v,u])<=.08
    return Image.fromarray(image),dict(vertices=ids[accepted].astype(np.int32),
        u=u[accepted].astype(np.int32),v=v[accepted].astype(np.int32)),float(visible.mean())


def render_scene(scene, design, root):
    out=root/'maps'/scene;out.mkdir(parents=True,exist_ok=True)
    vertex=PlyData.read(Path(design['data_root'])/'scans'/f'{scene}.ply')['vertex'].data
    xyz=np.column_stack([vertex[k] for k in ['x','y','z']]);rgb=np.column_stack([vertex[k] for k in ['R','G','B']])
    np.save(out/'mesh_xyz.npy',xyz)
    rows=[]
    for frame,(centre,rotation) in enumerate(camera_poses(xyz)):
        image,projection,fraction=render(xyz,rgb,centre,rotation)
        image.save(out/f'{frame:02d}.png');np.savez_compressed(out/f'{frame:02d}.projection.npz',**projection)
        rows.append(dict(frame=frame,colour_coverage=fraction,visible_vertices=len(projection['vertices']),
                         centre=centre.tolist(),camera_to_world_rotation=rotation.tolist()))
    write(out/'render.json',dict(scene=scene,points=len(xyz),views=rows,instance_labels_used=False,
        source='rendered xyz/RGB, not a real photograph',filled_pixel_radius=3,depth_tolerance_m=.08))
    print('rendered',scene,'coverage',round(float(np.mean([r['colour_coverage'] for r in rows])),3),flush=True)


@torch.inference_mode()
def compile_scenes(scenes, root, stage):
    torch.set_num_threads(3)
    from radio_gs.scripts.build_sam3_automatic_mask_cache import automatic_masks,unpack_masks
    if stage=='masks':
        from radio_gs.scripts.build_sam3_foundation_cache import _load_sam3_model
        teacher=_load_sam3_model(checkpoint_path='checkpoints/sam3_modelscope/sam3.pt',
            device='cuda',confidence_threshold=0.,dtype='bfloat16',resolution=1008,point_only=True)
        args=SimpleNamespace(grid_size=12,minimum_quality=.7,minimum_area_fraction=.0001,
            maximum_area_fraction=.9,minimum_stability=0.,stability_offset=1.,nms_iou=.85,
            duplicate_minimum_area_ratio=.9,maximum_masks=0)
    else:
        from radio_gs.models.native_region_appearance import NativeRegionAppearance
        teacher=NativeRegionAppearance()
    for scene in scenes:
        folder=root/'maps'/scene;n=len(np.load(folder/'mesh_xyz.npy'));matrices=[];seen=[];descriptors=[];offsets=[0]
        start=time.monotonic()
        for view in range(12):
            image=Image.open(folder/f'{view:02d}.png').convert('RGB');file=folder/f'{view:02d}.masks.pt'
            if stage=='masks':
                if not file.exists():torch.save(automatic_masks(teacher,image,args),file)
                print(scene,'masks',view+1,'/12',flush=True);continue
            row=torch.load(file,map_location='cpu');masks=unpack_masks(row['packed_masks'],640)
            projection=np.load(folder/f'{view:02d}.projection.npz');ids,u,v=[projection[k] for k in ['vertices','u','v']]
            region,point=np.nonzero(masks[:,v,u])
            matrices.append(sparse.csr_matrix((np.ones(len(region),np.float32),(ids[point],region)),shape=(n,len(masks))))
            seen.append(sparse.csr_matrix((np.ones(len(ids),np.float32),(ids,np.zeros(len(ids),int))),shape=(n,1)))
            descriptors.append(teacher.encode(image,masks).half().cpu().numpy() if len(masks) else np.zeros((0,768),np.float16))
            offsets.append(offsets[-1]+len(masks))
        if stage=='compile':
            sparse.save_npz(folder/'membership.npz',sparse.hstack(matrices,format='csr'))
            sparse.save_npz(folder/'visibility.npz',sparse.hstack(seen,format='csr'))
            np.save(folder/'appearance_descriptors.npy',np.concatenate(descriptors));np.save(folder/'region_offsets.npy',np.array(offsets))
            write(folder/'map.json',dict(scene=scene,points=n,regions=offsets[-1],views=12,
                instance_labels_used=False,source='rendered training RGB',seconds=time.monotonic()-start))
        print('finished',stage,scene,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('output/learned_instance_additions'))
    p.add_argument('--stage',choices=['render','masks','compile'],required=True)
    p.add_argument('--scenes',nargs='+');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    design=json.loads((a.root/'design.json').read_text()) if (a.root/'design.json').exists() else prepare(a.root)
    scenes=a.scenes or design['fit']+design['calibration']+design['validation'];scenes=scenes[a.shard::a.shards]
    if a.stage=='render':
        with ThreadPoolExecutor(4) as pool:list(pool.map(lambda s:render_scene(s,design,a.root),scenes))
    else:compile_scenes(scenes,a.root,a.stage)


if __name__=='__main__':main()
