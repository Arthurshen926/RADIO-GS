"""Fixed-contract full-cohort text evaluation of source-only uplift controls."""
import argparse,json
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from sklearn.metrics import average_precision_score
from gsplat import rasterization
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.scripts.eval_prerendered_lerf_features import load_lerf_objects,_mask_iou
from radio_gs.scripts.eval_opengaussian_lerf_baseline import SCENE_GT_FRAMES,evaluate_scene
from radio_gs.utils.immutable_artifacts import sha256_file
from scripts.audits.source_uplift_scope_next20 import local_correction


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();torch.set_num_threads(4)
    base=Path('output/full_field_rebuild_next19')/a.scene/'evaluation';root=Path('output/caption_view_consistency_next22')/a.scene;out=root/'evaluation';out.mkdir(exist_ok=False)
    report=json.loads((root/'report.json').read_text());assert report['complete'];assert sha256_file(root/'uplift.pt')==report['uplift_sha256']
    old=json.loads((base/'report.json').read_text());assert old['complete'];assert sha256_file(base/'scores.pt')==old['scores_sha256'];assert sha256_file(base/'text_online.encodings.pt')==old['encoding_bank_sha256']
    data=torch.load(base/'scores.pt',map_location='cpu',weights_only=False);bank=torch.load(base/'text_online.encodings.pt',map_location='cpu',weights_only=False)[0];up=torch.load(root/'uplift.pt',map_location='cpu',weights_only=False)
    assert data['geometry_binding']==up['geometry_binding'];assert old['field_sha256']==up['source_field_sha256']
    # Preserve the deployed caption route and queries that actually use a scope.
    from radio_gs.models.element_retrieval_field import ElementRetrievalField
    fieldpath=Path('output/full_field_rebuild_next19')/a.scene/'reconstruction/field/bounded_fallback_admission/field.pt'
    assert sha256_file(fieldpath)==old['field_sha256']
    payload=torch.load(fieldpath,map_location='cpu',weights_only=False);field=ElementRetrievalField(**payload['architecture']);field.load_state_dict(payload['state_dict']);field.requires_grad_(False)
    assert torch.equal(field.grounded_transfer.prototypes,up['queries'])
    indices,concept_accepted,best=field.grounded_transfer.route(bank['encoding'].half().float())
    scope_used=(data['scores']['final']-data['scores']['native_compact_calibrated']).abs().amax(0)>2e-5
    accepted=concept_accepted&scope_used
    scores={n:data['scores']['final'].clone() for n in ['final','lifted_calibrated','local_calibrated']};changes=[]
    for qi in torch.where(accepted)[0].tolist():
        j=int(indices[qi]);mean=up['mean'][:,j];var=up['variance'][:,j];count=up['count'][:,j];baseline=scores['final'][:,qi]
        scores['lifted_calibrated'][:,qi]=torch.where(count>0,mean,baseline)*data['observed']
        scores['local_calibrated'][:,qi],reliable=local_correction(baseline,mean,var,count)
        scores['local_calibrated'][:,qi]*=data['observed']
        assert torch.equal(scores['local_calibrated'][~reliable,qi],baseline[~reliable])
        changes.append(dict(query=data['texts'][qi],source_query_id=up['query_ids'][j],cosine=float(best[qi]),local_changed=int((scores['local_calibrated'][:,qi]!=baseline).sum())))
    torch.save(dict(data,scores=scores),out/'scores.pt')
    (out/'preparation.json').write_text(json.dumps(dict(route_policy="unchanged deployed concept route and non-native final response",changes=changes,source_admission=report['source_accepted'],diagnostic_even_if_source_rejected=True,default_promoted=False,parent_field_unchanged=True,image_prompt_unchanged=True,input_scores_sha256=old['scores_sha256'],uplift_sha256=report['uplift_sha256'],scores_sha256=sha256_file(out/'scores.pt'),implementation_sha256=sha256_file(__file__)),indent=2))
    gp=Path('output/depth_opacity_repair_20260918')/a.scene/'geometry_report.json';gr=json.loads(gp.read_text());g=gr['results']['independent_scale'];geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device='cuda:0');geo.assert_binding(data['geometry_binding']);prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256'])
    frames=load_lerf_objects(Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label')/a.scene,frames=SCENE_GT_FRAMES[a.scene]);assert {int(f.split('_')[-1]) for f in frames}.isdisjoint(up['train_frames']+up['validation_frames'])
    obs=data['observed'].cuda();rows={n:[] for n in scores};selected={};baseline_disagreement=0
    for name,values in scores.items():
        dst=out/name;dst.mkdir();values=values.cuda()
        for frame,objects in frames.items():
            cam=prior.cameras[int(frame.split('_')[-1])];h,w=objects[0].mask.shape;k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height
            kw=dict(viewmats=cam.world_to_camera.cuda()[None],Ks=k[None],width=w,height=h,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
            for obj in objects:
                qi=data['texts'].index(obj.query);colors=torch.stack([values[:,qi],obs.float()],1)
                im,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten(),colors=colors,**kw)
                mass=im[0,...,1];valid=(mass/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8);prob=torch.where(valid,im[0,...,0]/mass.clamp_min(1e-8),0).cpu();binary=prob>.5
                if name=='final':
                    reference=np.asarray(Image.open(base/'final/lerf2d'/(frame+'_'+obj.query+'.png')))>0
                    baseline_disagreement+=int((reference!=binary.numpy()).sum())
                rows[name].append(dict(frame=frame,query=obj.query,iou=_mask_iou(torch.from_numpy(obj.mask),binary),ap=float(average_precision_score(obj.mask.flatten(),prob.numpy().flatten()))))
                ids=(values[:,qi]>.5)&obs
                if ids.any():
                    _,aa,_=rasterization(means=geo.xyz[ids],quats=geo.quaternions[ids],scales=geo.scales[ids],opacities=geo.opacities.flatten()[ids],colors=torch.ones(int(ids.sum()),1,device='cuda'),**kw);byte=(aa[0,...,0]*255+.5).floor().clamp(0,255).byte().cpu().numpy()
                else:byte=np.zeros((h,w),dtype=np.uint8)
                Image.fromarray(byte).save(dst/(frame+'_'+obj.query+'.png'))
            print(a.scene,name,frame,flush=True)
        selected[name]=asdict(evaluate_scene(Path('output/protocol_audit_20260801/vala/lerf3d_occam_geometry_v1/eval_gt')/a.scene/'gt',dst,a.scene,threshold=10));assert selected[name]['missing']==0
    assert baseline_disagreement==0
    def comparable(result):
        return dict(result,objects=[{k:v for k,v in row.items() if k!='pred_path'} for row in result['objects']])
    assert comparable(selected['final'])==comparable(old['selected']['final'])
    result=dict(complete=True,source_accepted=report['source_accepted'],baseline_disagreement_pixels=baseline_disagreement,lerf2d={n:dict(miou=float(np.mean([r['iou'] for r in rr])),ap=float(np.mean([r['ap'] for r in rr])),observations=rr) for n,rr in rows.items()},selected=selected,routes=changes,source_report_sha256=sha256_file(root/'report.json'),scores_sha256=sha256_file(out/'scores.pt'),default_promoted=False,independent_test=False)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(a.scene,{n:r['miou'] for n,r in result['lerf2d'].items()},{n:r['miou'] for n,r in selected.items()},flush=True)


if __name__=='__main__':main()
