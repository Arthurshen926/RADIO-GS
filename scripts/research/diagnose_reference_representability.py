"""Single-view vs shared-field source fit, without target RGB or target labels.

Fits diagnose representability under the actual renderer, not retrieval accuracy.
"""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
from radio_gs.querying.typed_evidence import solve_rendered_evidence


def score(pred,target,mask):
    valid=mask[:,0]>0;p=pred[:,0][valid];t=target[:,0][valid];binary=p>=.5;truth=t>=.5;tp=(binary&truth).sum();fp=(binary&~truth).sum();fn=(~binary&truth).sum();error=(p-t).abs()
    return dict(iou=float(tp/(tp+fp+fn).clamp_min(1)),precision=float(tp/(tp+fp).clamp_min(1)),recall=float(tp/(tp+fn).clamp_min(1)),mae=float(error.mean()),huber=float(torch.where(error<=.25,.5*error.square(),.25*(error-.125)).mean()),pixels=int(valid.sum()))

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scene',required=True);parser.add_argument('--output',required=True);a=parser.parse_args();torch.set_num_threads(4);start=time.monotonic()
    scene=json.loads(Path(f'configs/consolidated/nvos/{a.scene}.scene.json').read_text());tracks=torch.load(f'output/reference_integration/fresh_tracks/{a.scene}/tracks.pt',map_location='cpu',weights_only=False);b=scene['geometry'];geo=FrozenGaussianGeometry.load(b['path'],expected_sha256=b['sha256'],device='cuda');gp=MogeGeometryPrior(scene['camera_report'],scene['camera_report_sha256']);n=len(geo.xyz);h,w=scene['source_fit_hw'];radius=scene['boundary_radius'];views=[]
    for frame in scene['source_frames']:
        rec=tracks['frames'].get(frame)
        if rec is None or rec['object_logit']<=0:continue
        cam=gp.cameras[frame];k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height;renderer=SimpleNamespace(scaled_intrinsics=lambda width,height:k,near_plane=.001*gp.scale,far_plane=100*gp.scale)
        hit=rasterize_single_view_contributions(geo,renderer,cam.world_to_camera.cuda(),height=h,width=w,include_depths=False);pid=hit['pixel_ids'].long();gid=hit['gaussian_ids'].long();weight=hit['weights'].float();total=torch.zeros(h*w,device='cuda').scatter_add_(0,pid,weight);op=torch.sparse_coo_tensor(torch.stack([pid,gid]),weight/total[pid].clamp_min(1e-8),(h*w,n)).coalesce()
        y=F.interpolate(rec['logits'].float()[None,None].cuda(),size=(h,w),mode='bilinear',align_corners=False).sigmoid();binary=(y>.5).float();pos=(-F.max_pool2d(-binary,2*radius+1,1,radius)>.5).flatten()[:,None];neg=(F.max_pool2d(binary,2*radius+1,1,radius)<.5).flatten()[:,None];full=(total[:,None]>1e-8).float();known=(pos.float()+neg.float()*float(rec.get('negative_reliability',1.)))*full
        views.append(dict(frame=frame,op=op,y=y.flatten()[:,None],full=full,known=known))
    initial=torch.full((n,1),.5,device='cuda');rows=[];joints={};out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    for domain in ['known','full']:
        observations=[(v['op'],v['y'],v[domain],1.) for v in views]
        joint120,mass=solve_rendered_evidence(initial,observations,steps=120);joint480,_=solve_rendered_evidence(joint120,observations,steps=360);joints[domain]=(joint120,joint480)
        torch.save(dict(elements=joint480.cpu(),geometry=b,diagnostic_source_only=True,supervision_domain=domain),out/f'joint_{domain}.pt')
        for v in views:
            obs=[(v['op'],v['y'],v[domain],1.)];single120,_=solve_rendered_evidence(initial,obs,steps=120);single480,_=solve_rendered_evidence(single120,obs,steps=360)
            results={}
            for name,field in [('joint120',joint120),('joint480',joint480),('single120',single120),('single480',single480)]:
                prediction=torch.sparse.mm(v['op'],field);results[name]={d:score(prediction,v['y'],v[d]) for d in ['known','full']}
            rows.append(dict(frame=v['frame'],training_domain=domain,metrics=results))
        print(a.scene,domain,'complete',flush=True)
    summary={domain:{arm:{metric:sum(r['metrics'][arm][domain][metric] for r in rows if r['training_domain']==domain)/len(views) for metric in ['iou','precision','recall','mae','huber']} for arm in ['joint120','joint480','single120','single480']} for domain in ['known','full']}
    result=dict(scene=a.scene,source_views=len(views),rows=rows,summary=summary,seconds=time.monotonic()-start,target_rgb_used=False,target_labels_used=False,single_view_fields_are_diagnostic_not_deployable=True,proves_global_optimum=False)
    (out/'diagnosis.json').write_text(json.dumps(result,indent=2)+'\n');print(a.scene,summary,flush=True)
if __name__=='__main__':main()
