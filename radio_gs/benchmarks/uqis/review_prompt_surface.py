"""Review 2D prompt visibility against the declared official mesh geometry.

Constructor/evaluator-side diagnosis only. A missing triangle is distinguished
from an occluding surface; neither is silently removed from the main cohort.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
from plyfile import PlyData


def first_surface(xyz, faces, pose, intrinsics, pixel):
    """Exact first triangle hit, used only for benchmark construction/review."""
    a=xyz[faces[:,0]];e1=xyz[faces[:,1]]-a;e2=xyz[faces[:,2]]-a
    direction=pose[:3,:3]@np.linalg.solve(intrinsics,np.r_[pixel,1.])
    direction/=np.linalg.norm(direction)
    pv=np.cross(direction,e2);det=np.einsum('ij,ij->i',e1,pv)
    inverse=np.divide(1.,det,out=np.zeros_like(det),where=np.abs(det)>1e-9)
    tv=pose[:3,3]-a;u=np.einsum('ij,ij->i',tv,pv)*inverse
    qv=np.cross(tv,e1);v=(qv@direction)*inverse
    distance=np.einsum('ij,ij->i',e2,qv)*inverse
    hits=np.flatnonzero((np.abs(det)>1e-9)&(u>=0)&(v>=0)&(u+v<=1)&(distance>0))
    if not len(hits):return None
    face=int(hits[np.argmin(distance[hits])])
    return face,float(distance[face]),pose[:3,3]+distance[face]*direction


def review_scene(job):
    package,mesh_root,domain,targets,queries=job
    package=Path(package);scene=domain['scene_id']
    xyz=np.load(package/'method'/domain['mesh_xyz_path'])
    labels=np.load(package/'evaluator'/domain['mesh_instance_ids_path'])
    ply=PlyData.read(Path(mesh_root)/scene/f'{scene}_vh_clean_2.ply')
    faces=np.stack(ply['face']['vertex_indices'])
    rows=[]
    for target in targets:
        query=queries[target['queries']['point_2d']]
        pose=np.asarray(query['camera_to_world']);origin=pose[:3,3]
        hit=first_surface(xyz,faces,pose,query['camera_intrinsics'],query['positive_pixel_uv'])
        construction=target.get('point_2d_construction',target['query_construction'])
        query_distance=float(np.linalg.norm(np.asarray(construction['point_world_xyz'])-origin))
        row=dict(scene_id=scene,instance_id=target['instance_id'],query_id=query['query_id'],
                 query_point_distance_m=query_distance)
        if hit is None:row['status']='no_triangle_on_ray'
        else:
            face,distance,_=hit;ids=labels[faces[face]]
            gap=float(distance-query_distance)
            status=('front_target' if np.all(ids==target['instance_id']) else
                    'mixed_boundary_triangle' if np.any(ids==target['instance_id']) else
                    'target_surface_missing_on_ray' if gap>.05 else
                    'other_surface_in_front' if gap<-.05 else 'nearby_other_surface')
            row.update(status=status,first_triangle_instance_ids=ids.tolist(),
                first_surface_distance_m=distance,surface_minus_query_distance_m=gap)
        rows.append(row)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True)
    p.add_argument('--mesh-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();ev=json.loads((a.package/'evaluator/targets.json').read_text())
    queries={q['query_id']:q for q in json.loads((a.package/'method/queries/point_2d.json').read_text())['queries']}
    jobs=[(str(a.package),str(a.mesh_root),d,[t for t in ev['targets'] if t['scene_id']==d['scene_id']],queries) for d in ev['scene_domains']]
    with ProcessPoolExecutor(3) as pool:rows=[r for chunk in pool.map(review_scene,jobs) for r in chunk]
    result=dict(scope='mesh-projection quality diagnosis, not a method score or independent annotation',
        target_count=len(rows),counts={s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})},
        primary_targets_removed=0,targets=rows)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['counts'],indent=2))


if __name__=='__main__':main()
