"""GT-assisted carrier fitting diagnostic, NEVER a benchmark prediction method."""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from plyfile import PlyData
from gsplat import rasterization
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.querying.typed_evidence import visible_operator,view_evidence,solve_set,SetType
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
from radio_gs.scripts.eval_prerendered_lerf_features import load_lerf_objects,_mask_iou
from radio_gs.scripts.eval_opengaussian_lerf_baseline import SCENE_GT_FRAMES,evaluate_scene


def load(p):return torch.load(p,map_location='cpu',weights_only=False)

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--geometry',choices=['rgb','moge','refined'],required=True);p.add_argument('--subset-steps',type=int,default=0);a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(0);start=time.monotonic()
    field=load(Path('output/taskwise_scope_composition_next38')/('lerf_'+a.scene)/'field.pt');report=json.loads(Path(field['geometry_report']).read_text());prior=MogeGeometryPrior(report['source_report'],report['source_report_sha256'])
    if a.geometry in ['moge','refined']:s=load(report['results']['independent_scale']['geometry'] if a.geometry=='moge' else Path('output/carrier_refine_next51')/a.scene/'geometry.pt')['state_dict'];xyz=s['xyz'].cuda();quats=s['quaternions'].cuda();scales=s['scales'].cuda();opacity=s['opacities'].flatten().cuda()
    else:
        d=PlyData.read(f'output/3dgs_models/{a.scene}/point_cloud/iteration_30000/point_cloud.ply')['vertex']
        def array(keys):return torch.from_numpy(np.stack([d[k] for k in keys],1).copy()).float().cuda()
        xyz=array(['x','y','z']);quats=F.normalize(array([f'rot_{i}' for i in range(4)]),dim=1);scales=array([f'scale_{i}' for i in range(3)]).exp();opacity=array(['opacity']).sigmoid().flatten()
    geo=SimpleNamespace(get_xyz=lambda:xyz,get_rotation=lambda:quats,get_scaling=lambda:scales,get_opacity=lambda:opacity)
    frames=load_lerf_objects(Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label')/a.scene,frames=SCENE_GT_FRAMES[a.scene]);texts=sorted({o.query for objs in frames.values() for o in objs});observed=torch.ones(len(xyz),device='cuda',dtype=torch.bool)
    fit_frames={f:o for i,(f,o) in enumerate(sorted(frames.items())) if i%2==0}
    held_frames=[f for f in frames if f not in fit_frames]
    views=[];numerator=torch.zeros(len(xyz),len(texts),device='cuda');den=torch.zeros_like(numerator);h,w=144,192
    for frame,objects in fit_frames.items():
        cam=prior.cameras[int(frame.split('_')[-1])];k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height
        renderer=SimpleNamespace(scaled_intrinsics=lambda width,height:k,near_plane=.001*prior.scale,far_plane=100*prior.scale)
        hit=rasterize_single_view_contributions(geo,renderer,cam.world_to_camera.cuda(),height=h,width=w,include_depths=False);hit['base_weights']=hit['weights']
        op,eop,visible=visible_operator(hit,observed,h*w,False);target=torch.zeros(h*w,len(texts),device='cuda');detected=torch.zeros(len(texts),device='cuda',dtype=torch.bool)
        for obj in objects:
            j=texts.index(obj.query);target[:,j]=F.interpolate(torch.from_numpy(obj.mask).float().cuda()[None,None],size=(h,w),mode='area').flatten();detected[j]=True
        e=view_evidence(op,eop,visible,target,detected,frame,h,w);views.append(e);known=(e.positive|e.negative).float();numerator+=torch.sparse.mm(e.adjoint,target*known);den+=torch.sparse.mm(e.adjoint,known)
    initial=numerator/den.clamp_min(1e-8);value,stats=solve_set(initial,views,SetType.TEXT_UNION,corroborate=False,consistency=False,steps=300,prior_strength=0)
    if a.subset_steps:
        # Oracle intervention on the decoder objective: optimize the actual
        # selected-alpha renderer, not the visibility-normalized feature image.
        with torch.inference_mode(False),torch.enable_grad():
            means=xyz.clone();rot=quats.clone();size=scales.clone();opa=opacity.clone()
            logits=torch.nn.Parameter(torch.logit(value.clone().clamp(.001,.999)))
            opt=torch.optim.Adam([logits],lr=.05);samples=[]
            for frame,objects in fit_frames.items():
                cam=prior.cameras[int(frame.split('_')[-1])];k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height
                for obj in objects:
                    target=F.interpolate(torch.from_numpy(obj.mask).float().cuda()[None,None],size=(h,w),mode='area')[0,...,0:h,0:w].squeeze(0)
                    samples.append((texts.index(obj.query),target,cam.world_to_camera.cuda()[None].clone(),k[None].clone()))
            for step in range(a.subset_steps):
                j,target,pose,k=samples[step%len(samples)];member=logits[:,j].sigmoid()
                _,alpha,_=rasterization(means=means,quats=rot,scales=size,opacities=opa*member,colors=torch.ones(len(means),1,device='cuda'),viewmats=pose,Ks=k,width=w,height=h,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
                pred=alpha[0,...,0];loss=F.binary_cross_entropy(pred.clamp(1e-5,1-1e-5),target)+1-(2*(pred*target).sum()+1)/(pred.sum()+target.sum()+1)
                opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                if (step+1)%500==0:print('subset oracle',step+1,float(loss),flush=True)
            value=logits.sigmoid().detach()
    out=Path('output/carrier_transfer_next51')/(a.geometry+('_subset' if a.subset_steps else ''))/a.scene;out.mkdir(parents=True,exist_ok=True);rows=[]
    for frame,objects in frames.items():
        cam=prior.cameras[int(frame.split('_')[-1])];h,w=objects[0].mask.shape;k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height
        args=dict(viewmats=cam.world_to_camera.cuda()[None],Ks=k[None],width=w,height=h,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
        for obj in objects:
            v=value[:,texts.index(obj.query)];rgb,alpha,_=rasterization(means=xyz,quats=quats,scales=scales,opacities=opacity,colors=v[:,None],**args);prob=rgb[0,...,0]/alpha[0,...,0].clamp_min(1e-8);sel=v>.5
            if sel.any():
                _,sa,_=rasterization(means=xyz[sel],quats=quats[sel],scales=scales[sel],opacities=opacity[sel],colors=torch.ones(int(sel.sum()),1,device='cuda'),**args);mask=(sa[0,...,0]*255+.5).floor().clamp(0,255).byte().cpu().numpy()
            else:mask=np.zeros((h,w),dtype=np.uint8)
            Image.fromarray(mask).save(out/(frame+'_'+obj.query+'.png'));rows.append(dict(frame=frame,query=obj.query,continuous_iou=_mask_iou(torch.from_numpy(obj.mask),prob.cpu()>.5)))
    gt=Path('output/protocol_audit_20260801/vala/lerf3d_occam_geometry_v1/eval_gt')/a.scene/'gt';selected=asdict(evaluate_scene(gt,out,a.scene,threshold=10))
    result=dict(scene=a.scene,geometry=a.geometry,subset_steps=a.subset_steps,diagnostic_only=True,aggregate_includes_fit_views=True,fit_frames=list(fit_frames),held_frames=held_frames,annotation_assisted_transfer=True,theoretical_upper_bound=False,production_use=False,continuous_iou=float(np.mean([r['continuous_iou'] for r in rows])),selected=selected,rows=rows,seconds=time.monotonic()-start,solver=stats)
    (out/'diagnostic.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k not in ['rows','selected']},selected['miou'],flush=True)

if __name__=='__main__':main()
