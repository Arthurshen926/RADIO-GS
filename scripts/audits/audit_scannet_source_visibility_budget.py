"""Source-only: distinguish camera sampling omissions from occluded surfaces."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from plyfile import PlyData
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.geometry.registered_pinhole import blender_camera_to_opencv
from radio_gs.geometry.mesh_visibility import rasterize_registered_mesh_depth
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);a.output.mkdir(parents=True,exist_ok=False)
    root=Path('output/scannet_frozen_field_20260918')/a.scene
    report=json.loads((Path('output/scannet_provided_mesh_track_20260918')/a.scene/'geometry_report.json').read_text())
    assert sha256_file(report['input_mesh'])==report['input_mesh_sha256']
    authority=json.loads(Path(report['source_cameras']).read_text());assert sha256_file(authority['source_camera_file'])==authority['source_camera_sha256']
    source=json.loads(Path(authority['source_camera_file']).read_text())
    mesh=PlyData.read(report['input_mesh']);verts=torch.tensor(np.stack([mesh['vertex'][k] for k in ['x','y','z']],1),device=a.device).float();faces=np.stack(mesh['face']['vertex_indices'])
    geo=FrozenGaussianGeometry.load(report['geometry'],expected_sha256=report['geometry_sha256'],device=a.device)
    fieldroot=root/'reconstruction/field';info=json.loads((fieldroot/'teachers.json').read_text())
    observed=torch.load(fieldroot/'pixel_evidence96/field.pt',map_location='cpu',weights_only=False)['state_dict']['observed'].bool()
    import nvdiffrast.torch as dr
    context=dr.RasterizeCudaContext(device=a.device)
    all_rows=[];ids=[]
    for frame in source['frames']:
        fid=int(Path(frame['file_path']).stem);c=blender_camera_to_opencv(frame,source)
        k=c['intrinsic'].to(device=a.device,dtype=torch.float32);width=640;height=round(c['height']*width/c['width']);k[0]*=width/c['width'];k[1]*=height/c['height']
        w2c=c['world_to_camera'].to(device=a.device,dtype=torch.float32)
        cv=verts@w2c[:3,:3].T+w2c[:3,3]
        depth=rasterize_registered_mesh_depth(cv,faces,k,height,width,context=context)
        xyz=geo.xyz@w2c[:3,:3].T+w2c[:3,3];uv=xyz@k.T;uv=uv[:,:2]/xyz[:,2:].clamp_min(1e-8);ij=uv.floor().long()
        inside=(xyz[:,2]>.01)&(ij[:,0]>=0)&(ij[:,0]<width)&(ij[:,1]>=0)&(ij[:,1]<height)
        reference=depth[ij[:,1].clamp(0,height-1),ij[:,0].clamp(0,width-1)]
        visible=inside&torch.isfinite(reference)&((xyz[:,2]-reference).abs()<=torch.maximum(torch.full_like(reference,.03),reference*.02))
        ids.append(fid);all_rows.append(visible.cpu())
    visibility=torch.stack(all_rows);weight=(geo.scales.prod(1)*geo.opacities.flatten()).cpu();covered=torch.zeros_like(observed);chosen=[]
    eligible=torch.tensor([f not in info['validation_frames'] for f in ids])
    for _ in range(len(info['train_frames'])):
        gain=(visibility.float()*(weight*(~covered))[None]).sum(1);gain[~eligible]=-1
        best=int(gain.argmax());eligible[best]=False;chosen.append(ids[best]);covered|=visibility[best]
    train=visibility[[ids.index(f) for f in info['train_frames']]].any(0)
    original32=visibility[[ids.index(f) for f in authority['source_frame_indices']]].any(0)
    all_visible=visibility.any(0);missing=~observed
    def stats(mask):return dict(elements=int(mask.sum()),volume_fraction=float(weight[mask].sum()/weight.sum()),missing_semantic_elements_reachable=int((mask&missing).sum()))
    result=dict(scene=a.scene,source_only=True,benchmark_labels_used=False,geometry_sha256=report['geometry_sha256'],source_camera_sha256=authority['source_camera_sha256'],
        visibility_definition='Mesh depth width640, center depth tolerance max(3cm,2%); shared geometry consistency, not physical visibility GT',
        available_views=len(ids),training_views=info['train_frames'],validation_views=info['validation_frames'],
        equal_budget_geometry_selected_frames=chosen,selected_using_semantics=False,
        current_train=stats(train),registered32=stats(original32),all_available=stats(all_visible),equal_budget_geometry_selection=stats(covered),
        missing_semantic_elements=int(missing.sum()),missing_outside_all_visible=int((missing&~all_visible).sum()),
        implementation_sha256=sha256_file(__file__))
    torch.save(dict(frame_ids=ids,visibility=visibility),a.output/'visibility.pt')
    result['visibility_sha256']=sha256_file(a.output/'visibility.pt');(a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    with torch.no_grad():main()
