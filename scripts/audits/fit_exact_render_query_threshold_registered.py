"""Source-only query calibration through the deployed sigmoid-before-raster path.

Same source queries/masks as the retained mean-cosine calibration. Three
coarse-to-fine 17-point grids, always including each query's previous optimum.
No development labels or target RGB are read.
"""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from gsplat import rasterization
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.models.native_query_threshold import fit_threshold_ridge
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.geometry.registered_source_prior import RegisteredSourcePrior as MogeGeometryPrior
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.geometry.source_report_binding import load_source_geometry_report


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);p.add_argument('--root',type=Path);p.add_argument('--geometry-report',type=Path);p.add_argument('--parent',default='native_within_extent96');p.add_argument('--output-name',default='exact_render_query_threshold96');a=p.parse_args();torch.set_num_threads(4)
    base=Path('/mnt/pool/sqy/results/RADIO-GS/output');root=a.root if a.root is not None else base/'preserved_source_inventory_20260916'/a.scene
    parent=root/a.parent;receipt=json.loads((parent/'report.json').read_text());assert sha256_file(parent/'field.pt')==receipt['field_sha256'];payload=torch.load(parent/'field.pt',weights_only=False)
    old=json.loads((root/'full_source_query_threshold_control/report.json').read_text())
    ep=root/'episodes.pt';assert sha256_file(ep)==old['episode_sha256'];data=torch.load(ep,weights_only=False)
    info=json.loads((root/'teachers.json').read_text());assert sha256_file(root/'teachers.json')==data['teacher_manifest_sha256']
    rawpath=root/'source_grounded_text.pt';assert sha256_file(rawpath)==old['raw_source_grounding_sha256'];raw=torch.load(rawpath,weights_only=False)
    assert raw['benchmark_labels_used'] is False and raw['frames']==info['train_frames']
    assert data['geometry_binding']==payload['geometry_binding']
    field=ElementRetrievalField(**payload['architecture']).to(a.device);field.load_state_dict(payload['state_dict']);field.requires_grad_(False)
    grpath,gr=load_source_geometry_report(root,a.scene,a.geometry_report);g=gr['results']['independent_scale'];geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device=a.device);geo.assert_binding(payload['geometry_binding']);prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256'])
    ids=[r['query_index'] for r in old['source_query_fits']];q=data['queries']['text'][ids].float().to(a.device);Q=len(ids);h,w=raw['probability'].shape[-2:]
    targets=[];eligible=[];cameras=[]
    for vi,frame in enumerate(raw['frames']):
        target=raw['probability'][vi,ids].to(a.device)>.5;use=(raw['scores'][vi,ids].to(a.device)>=.7)&(target.flatten(1).sum(1)>=20)
        targets.append(target.permute(1,2,0));eligible.append(use)
        camera=prior.cameras[frame];k=camera.intrinsic.to(a.device).clone();k[0]*=w/camera.width;k[1]*=h/camera.height;cameras.append((camera.world_to_camera.to(a.device)[None],k[None]))
    history=[]
    with torch.no_grad():
        cosine=field.native_cosine(q);scale=field.readout_log_scale['text'].clamp(-4.,6.).exp()
        best=torch.tensor([r['threshold'] for r in old['source_query_fits']],device=a.device)
        lower=cosine[field.observed].amin(0);upper=cosine[field.observed].amax(0)
        baseline=None;initial=None
        for stage in range(3):
            grid=lower[:,None]+(upper-lower)[:,None]*torch.linspace(0,1,17,device=a.device)[None]
            grid=torch.cat([grid,best[:,None]],1);T=grid.shape[1]
            response=torch.sigmoid(scale*(cosine[:,:,None]-grid[None]))*field.observed[:,None,None]
            colors=torch.cat([response.reshape(len(field.latent),-1),field.observed.float()[:,None]],1)
            hits=torch.zeros(Q,T,device=a.device);unions=torch.zeros_like(hits)
            for vi,(view,k) in enumerate(cameras):
                image,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten(),colors=colors,viewmats=view,Ks=k,width=w,height=h,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
                mass=image[0,...,-1];valid=(mass/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8)
                pred=(image[0,...,:-1].reshape(h,w,Q,T)/mass[...,None,None].clamp_min(1e-8)>.5)&valid[...,None,None]
                target=targets[vi][...,None];use=eligible[vi][:,None]
                hits+=(pred&target).sum((0,1))*use;unions+=(pred|target).sum((0,1))*use
            iou=hits/unions.clamp_min(1)
            if stage==0:baseline=iou[:,-1].cpu().tolist();initial=best.cpu().tolist()
            which=iou.argmax(1);best=grid[torch.arange(Q,device=a.device),which]
            width=(upper-lower)/16;lower=(best-width).clamp_min(-1);upper=(best+width).clamp_max(1)
            history.append(dict(stage=stage,thresholds=best.cpu().tolist(),source_teacher_iou=iou.max(1).values.cpu().tolist()))
            print(a.scene,'stage',stage,'source IoU',float(iou.max(1).values.mean()),flush=True)
            del colors,response,pred,image
        penalties=[.01,.1,1.,10.];errors=[]
        for penalty in penalties:
            values=[]
            for i in range(Q):
                keep=torch.arange(Q,device=a.device)!=i;weight,bias=fit_threshold_ridge(q[keep],best[keep],penalty)
                values.append(float((F.normalize(q[i],dim=-1)@weight+bias-best[i]).square()))
            errors.append(sum(values)/len(values))
        selected=penalties[min(range(len(errors)),key=errors.__getitem__)];weight,bias=fit_threshold_ridge(q,best,selected)
    arch=dict(payload['architecture'],native_query_thresholds=dict(payload['architecture']['native_query_thresholds'],text={'lower':float(best.min()),'upper':float(best.max())}))
    state=dict(payload['state_dict']);state.update({'native_query_thresholds.text.linear.weight':weight.cpu()[None],'native_query_thresholds.text.linear.bias':bias.cpu()[None],'native_query_thresholds.text.bounds':torch.tensor([float(best.min()),float(best.max())])})
    cold=ElementRetrievalField(**arch);cold.load_state_dict(state,strict=True)
    for key,value in payload['state_dict'].items():
        if not key.startswith('native_query_thresholds.text.'):assert torch.equal(value,state[key]),key
    out=root/a.output_name;out.mkdir(exist_ok=False);torch.save(dict(payload,architecture=arch,state_dict=state),out/'field.pt')
    report=dict(schema='exact_render_query_threshold.v1',field_sha256=sha256_file(out/'field.pt'),parent_field_sha256=receipt['field_sha256'],source_only=True,benchmark_data_used=False,
        query_indices=ids,source_initial_thresholds=initial,source_initial_deployed_iou=baseline,history=history,ridge_candidates=penalties,leave_one_concept_out_mse=errors,selected_ridge=selected,
        native_field_and_extent_tensors_exact=True,image_prompt_unchanged=True,raw_source_grounding_sha256=sha256_file(rawpath),episode_sha256=sha256_file(ep),geometry_sha256=g['geometry_sha256'],
        source_calibration_shape=[h,w],calibration_rendering='mean of per-element sigmoid, observed-mass normalized; same order as query execution',
        search='3 rounds of per-query 17-point grid plus previous optimum; approximate source IoU optimum, no claim of exhaustive optimum',
        implementation_sha256=sha256_file(__file__),readout_implementation_sha256=sha256_file('radio_gs/models/native_query_threshold.py'))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(a.scene,'done',selected,flush=True)


if __name__=='__main__':main()
