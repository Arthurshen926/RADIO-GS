"""Source-only stage diagnostic: sampling, voxel averaging, and Gaussian footprint."""
import argparse,json
from pathlib import Path
import torch
from gsplat import rasterization
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.scripts.repair_moge_geometry import load_views
from radio_gs.utils.immutable_artifacts import sha256_file


def depth_stats(depth,alpha,view,unit):
    ref=view['depth']/unit;depth=depth/unit;valid=view['valid']
    covered=valid&(alpha>.1)&torch.isfinite(depth)&(depth>0)
    result={'coverage':float(covered.sum()/valid.sum()),'absrel':float(((depth-ref).abs()/ref.clamp_min(1e-8))[covered].mean()) if covered.any() else None}
    for axis in ('horizontal','vertical'):
        if axis=='horizontal':
            delta=depth[:,1:]-depth[:,:-1];rd=ref[:,1:]-ref[:,:-1];known=valid[:,1:]&valid[:,:-1];both=covered[:,1:]&covered[:,:-1]
        else:
            delta=depth[1:]-depth[:-1];rd=ref[1:]-ref[:-1];known=valid[1:]&valid[:-1];both=covered[1:]&covered[:-1]
        edge=known&(rd.abs()>.03);keep=edge&both
        result[axis]={'edge_pairs':int(edge.sum()),'edge_pair_coverage':float(keep.sum()/edge.sum().clamp_min(1)),
                      'signed_edge_mae_metres':float((delta-rd).abs()[keep].mean()) if keep.any() else None,
                      'median_edge_magnitude_ratio':float((delta.abs()/rd.abs().clamp_min(1e-8))[keep].median()) if keep.any() else None}
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--geometry-report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:2')
    a=p.parse_args();torch.set_num_threads(4)
    if a.output.exists():raise FileExistsError(a.output)
    report=json.loads(a.geometry_report.read_text());prior=MogeGeometryPrior(report['source_report'],report['source_report_sha256'])
    ids=[x for x in report['mapping_frames'] if prior.calibration[x]['accepted']][:3]
    # Subset before reading depth: never open heldout prediction arrays.
    prior.records={x:prior.records[x] for x in ids}
    views=load_views(prior,480,a.device)
    g=report['results']['independent_scale'];model=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device=a.device)
    binding=model.fingerprint();results=[]
    def render(m,v):
        h,w=v['depth'].shape
        image,alpha,_=rasterization(means=m.get_xyz(),quats=m.get_rotation(),scales=m.get_scaling(),opacities=m.get_opacity().reshape(-1),colors=torch.zeros(len(m.xyz),1,device=a.device),
           viewmats=torch.linalg.inv(v['pose'])[None],Ks=v['k'][None],width=w,height=h,near_plane=.001*prior.scale,far_plane=100*prior.scale,render_mode='RGB+ED',packed=False)
        return depth_stats(image[0,...,-1],alpha[0,...,0],v,prior.scale)
    def pixel_footprint(m,v):
        z=((m.xyz-v['pose'][:3,3])@v['pose'][:3,:3])[:,2]
        sigma=.35*z.clamp_min(.001*prior.scale)/torch.sqrt(v['k'][0,0]*v['k'][1,1])
        m.scales[:,:2]=sigma[:,None];m.scales[:,2]=sigma*.1
    for v in views:
        xyz=(v['rays']*v['depth'][...,None])@v['pose'][:3,:3].T+v['pose'][:3,3]
        normal=v['normals']@v['pose'][:3,:3].T;good=v['valid']&(normal.norm(dim=-1)>.5)
        xyz=xyz[good];normal=normal[good]
        raw=FrozenGaussianGeometry(xyz,normal,tangent_sigma=.0075*prior.scale,normal_sigma=.001*prior.scale)
        variants={'depth_identity_reference_dependent':depth_stats(v['depth'],v['valid'].float(),v,prior.scale),'fused_fixed':render(model,v),'same_view_fixed_reference_dependent':render(raw,v)}
        pixel_footprint(raw,v);variants['same_view_pixel_reference_dependent']=render(raw,v)
        keys,inverse=torch.unique(torch.floor(xyz/(.01*prior.scale)).long(),dim=0,return_inverse=True)
        counts=torch.bincount(inverse).float();centres=torch.zeros(len(keys),3,device=a.device).index_add_(0,inverse,xyz)/counts[:,None]
        normals=torch.zeros_like(centres).index_add_(0,inverse,normal);keep=normals.norm(dim=-1)>1e-6
        voxel=FrozenGaussianGeometry(centres[keep],normals[keep],tangent_sigma=.0075*prior.scale,normal_sigma=.001*prior.scale)
        pixel_footprint(voxel,v);variants['same_view_voxel_pixel_reference_dependent']=render(voxel,v)
        results.append({'frame':v['frame'],'raw_points':len(raw.xyz),'voxel_points':len(voxel.xyz),'variants':variants})
        print(v['frame'],{k:(x['coverage'],x['horizontal']['median_edge_magnitude_ratio']) for k,x in variants.items()},flush=True)
    model.assert_binding(binding)
    a.output.write_text(json.dumps({'geometry_report_sha256':sha256_file(a.geometry_report),'geometry_sha256':g['geometry_sha256'],'mapping_frames':ids,'geometry_unchanged':True,'heldout_prediction_arrays_opened':False,
        'limitations':'reference-dependent stage controls, not deployable candidates; single-view voxel diagnostic excludes multiview filtering; conditional errors must be read with coverage; no geometric ground truth',
        'results':results},indent=2)+'\n')

if __name__=='__main__':main()
