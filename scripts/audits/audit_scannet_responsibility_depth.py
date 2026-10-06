"""Source-only consistency of semantic responsibilities with provided mesh depth."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from plyfile import PlyData
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.geometry.registered_pinhole import load_registered_camera_table
from radio_gs.geometry.mesh_visibility import rasterize_registered_mesh_depth
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4);a.output.mkdir(parents=True,exist_ok=False)
    gr=json.loads((Path('output/scannet_provided_mesh_track_20260918')/a.scene/'geometry_report.json').read_text())
    geo=FrozenGaussianGeometry.load(gr['geometry'],expected_sha256=gr['geometry_sha256'],device=a.device)
    assert sha256_file(gr['input_mesh'])==gr['input_mesh_sha256'];mesh=PlyData.read(gr['input_mesh'])
    verts=torch.tensor(np.stack([mesh['vertex'][k] for k in ['x','y','z']],1),device=a.device).float();faces=np.stack(mesh['face']['vertex_indices'])
    cameras=load_registered_camera_table(gr['source_cameras'])
    root=Path('output/scannet_frozen_field_20260918')/a.scene/'reconstruction/field';auth=json.loads((root/'responsibility.json').read_text());info=json.loads((root/'teachers.json').read_text());assert sha256_file(root/'responsibility.json')==info['authority_sha256']
    h=auth['metadata']['feature_height'];w=auth['metadata']['feature_width'];rows=[]
    import nvdiffrast.torch as dr
    context=dr.RasterizeCudaContext(device=a.device)
    for frame in info['train_frames']:
        c=cameras[frame];k=c.intrinsic.to(a.device).clone();k[0]*=w/c.width;k[1]*=h/c.height;rt=c.world_to_camera.to(a.device)
        depth=rasterize_registered_mesh_depth(verts@rt[:3,:3].T+rt[:3,3],faces,k,h,w,context=context).flatten()
        r=auth['views'][info['frames'].index(frame)];assert sha256_file(root/r['relative_path'])==r['sha256'];hit=torch.load(root/r['relative_path'],map_location=a.device,weights_only=False)
        pid=hit['pixel_ids'].long();gid=hit['gaussian_ids'].long();weights=hit['base_weights'].float();reference=depth[pid]
        center=geo.xyz[gid]@rt[:3,:3].T+rt[:3,3];normal=geo.normals[gid]@rt[:3,:3].T
        ray=torch.stack(((pid%w+.5-k[0,2])/k[0,0],(pid//w+.5-k[1,2])/k[1,1],torch.ones_like(weights)),1)
        denom=(normal*ray).sum(1);plane=(normal*center).sum(1)/denom
        valid=torch.isfinite(reference)&torch.isfinite(plane)&(plane>.01)&(denom.abs()>.05)
        tol=torch.maximum(torch.full_like(reference,.03),reference*.02)
        behind=valid&(center[:,2]>reference+tol)&(plane>reference+tol)
        front=valid&(center[:,2]<reference-tol)&(plane<reference-tol)
        rows.append(dict(frame=frame,total_weight=float(weights.sum()),tested_weight=float(weights[valid].sum()),
            behind_weight=float(weights[behind].sum()),front_weight=float(weights[front].sum()),
            behind_elements=int(gid[behind].unique().numel()),
            behind_fraction=float(weights[behind].sum()/weights.sum().clamp_min(1e-8))))
    total=sum(r['total_weight'] for r in rows)
    result=dict(scene=a.scene,source_only=True,benchmark_labels_used=False,geometry_sha256=gr['geometry_sha256'],
        reference='Provided mesh first-surface depth at identical source responsibility resolution; both Gaussian center and tangent-plane depths exceed max(3cm,2%) tolerance. Shared-input consistency only.',
        aggregate_behind_fraction=sum(r['behind_weight'] for r in rows)/total,
        aggregate_front_fraction=sum(r['front_weight'] for r in rows)/total,
        tested_weight_fraction=sum(r['tested_weight'] for r in rows)/total,views=rows,implementation_sha256=sha256_file(__file__))
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    with torch.no_grad():main()
