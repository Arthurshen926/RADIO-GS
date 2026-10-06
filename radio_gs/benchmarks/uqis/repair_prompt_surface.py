"""Version a same-camera 2D click repair using RGB-D and mesh alone.

No method maps, predictions or evaluation scores are inputs. Original 3D clicks,
reference crops, scene geometry, identities and mapping exclusions are preserved.
"""
import argparse
import hashlib
import hmac
import json
from pathlib import Path
import shutil

import numpy as np
from plyfile import PlyData
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from .construct import ConstructionConfig, projected_observation, write_json
from .review_prompt_surface import first_surface
from .sensor import SensorIndex


def choose_surface_prompt(sensor, target, xyz, ids, faces, config):
    instance=target['instance_id'];frame=target['query_construction']['frame_index']
    pose,depth,_,world,distance,_,labels=projected_observation(sensor,frame,cKDTree(xyz),ids)
    yy,xx=np.mgrid[:sensor.height,:sensor.width];kd,kc=sensor.k_depth,sensor.k_color
    uc=kc[0,0]*(xx-kd[0,2])/kd[0,0]+kc[0,2]
    vc=kc[1,1]*(yy-kd[1,2])/kd[1,1]+kc[1,2]
    rgb_visible=(uc>=0)&(vc>=0)&(uc<sensor.color_width)&(vc<sensor.color_height)
    extent=(labels==instance)&(distance<=config.extent_correspondence_m)&rgb_visible
    interior=distance_transform_edt(np.pad(extent,1))[1:-1,1:-1]
    rgb_margin=config.minimum_rgb_border_px;margin=config.minimum_image_border_px
    legal=extent&(distance<=config.correspondence_m)&(interior>=config.minimum_interior_px)
    legal&=(uc>=rgb_margin)&(vc>=rgb_margin)&(uc<sensor.color_width-rgb_margin)&(vc<sensor.color_height-rgb_margin)
    legal[:margin]=legal[-margin:]=False;legal[:,:margin]=legal[:,-margin:]=False
    pixels=np.flatnonzero(legal)
    pixels=pixels[np.argsort(-interior.reshape(-1)[pixels],kind='stable')]
    target_faces=faces[np.all(ids[faces]==instance,axis=1)]
    for pixel in pixels:
        v,u=np.unravel_index(pixel,depth.shape)
        # A cheap target-only intersection precedes full-scene occlusion checks.
        if first_surface(xyz,target_faces,pose,kd[:3,:3],[u,v]) is None:continue
        hits=[]
        for du,dv in [(0,0),(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            hit=first_surface(xyz,faces,pose,kd[:3,:3],[u+du,v+dv])
            if hit is None or not np.all(ids[faces[hit[0]]]==instance):break
            if np.linalg.norm(hit[2]-world[v+dv,u+du])>config.correspondence_m:break
            hits.append(hit)
        if len(hits)!=9:continue
        return dict(positive_pixel_uv=[int(u),int(v)],point_world_xyz=world[v,u].tolist(),
            frame_index=frame,interior_distance_px=float(interior[v,u]),
            image_border_px=int(min(u,v,sensor.width-1-u,sensor.height-1-v)),
            depth_m=float(depth[v,u]),mesh_surface_world_xyz=hits[0][2].tolist(),
            mesh_depth_distance_m=float(np.linalg.norm(hits[0][2]-world[v,u])),
            verified_front_target_rays=9,selection='largest reliable interior with 3x3 front-target mesh/depth consistency')
    raise ValueError(f"No same-camera surface-compatible prompt for {target['scene_id']}/{instance}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True);p.add_argument('--review',type=Path,required=True)
    p.add_argument('--mesh-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();read=lambda f:json.loads(f.read_text())
    review=read(a.review);evaluation=read(a.package/'evaluator/targets.json')
    mapping=read(a.package/'mapping/scenes.json');release=read(a.package/'release.json')
    queries=read(a.package/'method/queries/point_2d.json')
    key=bytes.fromhex(read(a.package/'evaluator/query_id_key.json')['hmac_key_hex'])
    config=ConstructionConfig(**release['construction_config']);version='scannet-uqis-9-v0.3.1-candidate'
    edits=[]
    for row in review['targets']:
        if row['status']=='front_target':continue
        scene=row['scene_id'];instance=row['instance_id']
        target=next(t for t in evaluation['targets'] if t['scene_id']==scene and t['instance_id']==instance)
        domain=next(d for d in evaluation['scene_domains'] if d['scene_id']==scene)
        source=next(s for s in mapping['scenes'] if s['scene_id']==scene)
        xyz=np.load(a.package/'method'/domain['mesh_xyz_path'])
        ids=np.load(a.package/'evaluator'/domain['mesh_instance_ids_path'])
        faces=np.stack(PlyData.read(a.mesh_root/scene/f'{scene}_vh_clean_2.ply')['face']['vertex_indices'])
        construction=choose_surface_prompt(SensorIndex(source['sens_path']),target,xyz,ids,faces,config)
        query=next(q for q in queries['queries'] if q['query_id']==target['queries']['point_2d'])
        before=dict(query)
        query.update(benchmark_version=version,positive_pixel_uv=construction['positive_pixel_uv'])
        body={k:v for k,v in query.items() if k!='query_id'}
        query['query_id']='uq_'+hmac.new(key,json.dumps(body,sort_keys=True,separators=(',',':')).encode(),hashlib.sha256).hexdigest()[:32]
        target['queries']['point_2d']=query['query_id'];target['point_2d_construction']=construction
        edits.append(dict(scene_id=scene,instance_id=instance,reason=row['status'],before=before,
                          after=query,construction=construction))
    # copytree refuses to modify an existing candidate; all selection precedes export.
    shutil.copytree(a.package,a.output)
    evaluation['benchmark_version']=version;queries['benchmark_version']=version
    release.update(benchmark_version=version,parent_package=str(a.package),
        revision='2D click surface consistency; all other query payloads and mapping inputs identical',
        changed_point_2d_queries=len(edits),unchanged_query_ids_preserved=True,
        compatible_map_inputs_from=str(a.package),old_prediction_reuse_allowed=False)
    write_json(a.output/'method/queries/point_2d.json',queries)
    write_json(a.output/'evaluator/targets.json',evaluation)
    write_json(a.output/'release.json',release)
    write_json(a.output/'evaluator/prompt_surface_changes.json',dict(
        selection_uses_method_scores=False,query_frames_changed=False,mapping_inputs_changed=False,
        other_modalities_changed=False,targets_removed=0,changes=edits))
    print(json.dumps(edits,indent=2))


if __name__=='__main__':main()
