"""Fixed-carrier native/full versus compact fusion and final-query ranking.

Raw cosine branches are compared by AP, not a fabricated .5 membership IoU.
No target RGB is used. Preparation reads source teachers on CPU; rendering
consumes only persisted scores and frozen geometry before GT scoring.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.training.source_pixel_evidence import source_pixel_evidence_weights
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--render-only',action='store_true');p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(4)
    root=Path('output/depth_opacity_pipeline_20260918')/a.scene/'reconstruction/field'
    gp=Path('output/depth_opacity_repair_20260918')/a.scene/'geometry_report.json'
    def load(path,digest=None):
        if digest:assert sha256_file(path)==digest
        return torch.load(path,map_location='cpu',weights_only=False)
    if not a.render_only:
        a.output.mkdir(parents=True,exist_ok=False)
        info=json.loads((root/'teachers.json').read_text());rec=info['teacher_files']['siglip']
        desc=F.normalize(load(root/rec['path'],rec['sha256'])['descriptors'].float(),dim=1)
        ep=load(root/'episodes.pt',json.loads((root/'episodes.json').read_text())['sha256'])
        auth=json.loads((root/'responsibility.json').read_text());assert sha256_file(root/'responsibility.json')==info['authority_sha256']
        report=json.loads((root/'pixel_evidence96/report.json').read_text());payload=load(root/'pixel_evidence96/field.pt',report['field_sha256'])
        sd=payload['state_dict'];basis=sd['native_decoder.weight'];observed=sd['observed'];n=len(observed)
        qp=Path('output/bounded_fallback_repair_20260918')/a.scene/'text_online.encodings.pt';bank=load(qp)[0];q=F.normalize(bank['encoding'].half().float(),dim=1)
        numerator=torch.zeros(n,desc.shape[1]);mass=torch.zeros(n)
        for frame in info['train_frames']:
            vi=info['frames'].index(frame);ids=torch.where(ep['proposal_views']==vi)[0];rec=auth['views'][vi];hit=load(root/rec['relative_path'],rec['sha256'])
            weights,_=source_pixel_evidence_weights(ep['masks'][ids].flatten(1),hit['pixel_ids'].long(),hit['gaussian_ids'].long(),hit['base_weights'].float(),n,partition_overlap=True)
            numerator+=weights.T@desc[ids];mass+=weights.sum(0)
        full=numerator/mass[:,None].clamp_min(1e-8);del numerator
        error=float(((full@basis)[observed]-sd['latent'][observed,:basis.shape[1]]).abs().max());assert error<2e-4,error
        compressed=sd['latent'][:,:basis.shape[1]]@basis.T
        report=json.loads((root/'bounded_fallback_admission/report.json').read_text());final=load(root/'bounded_fallback_admission/field.pt',report['field_sha256'])
        model=ElementRetrievalField(**final['architecture']);model.load_state_dict(final['state_dict']);model.requires_grad_(False)
        assert torch.equal(model.observed,observed)
        scores=dict(native_full=F.normalize(full,dim=1)@q.T,native_compact=F.normalize(compressed,dim=1)@q.T,final=model.retrieve(q,'text')['scores'])
        for k in ['native_full','native_compact']:scores[k]*=observed[:,None]
        native_region=desc@q.T;compact_region=F.normalize((desc@basis)@basis.T,dim=1)@q.T
        train=torch.tensor([info['frames'][int(v)] in info['train_frames'] for v in ep['proposal_views']])
        packed=dict(texts=bank['texts'],scores=scores,observed=observed,geometry_binding=payload['geometry_binding'])
        torch.save(packed,a.output/'scores.pt')
        result=dict(scene=a.scene,diagnostic_only=True,source_only_construction=True,target_rgb_used=False,
            region_top1_retention=float((native_region[train].argmax(1)==compact_region[train].argmax(1)).float().mean()),
            fusion_replay_max_error=error,geometry_report_sha256=sha256_file(gp),field_sha256=report['field_sha256'],
            scores_sha256=sha256_file(a.output/'scores.pt'),query_sha256=sha256_file(qp),implementation_sha256=sha256_file(__file__))
        (a.output/'preparation.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True);return
    if (a.output/'report.json').exists():raise FileExistsError(a.output/'report.json')
    prep=json.loads((a.output/'preparation.json').read_text());assert sha256_file(gp)==prep['geometry_report_sha256']
    data=load(a.output/'scores.pt',prep['scores_sha256']);gr=json.loads(gp.read_text());g=gr['results']['independent_scale']
    from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
    from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
    from radio_gs.scripts.eval_prerendered_lerf_features import load_lerf_objects
    from radio_gs.scripts.eval_opengaussian_lerf_baseline import SCENE_GT_FRAMES
    from gsplat import rasterization
    geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device=a.device);geo.assert_binding(data['geometry_binding'])
    prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256'])
    frames=load_lerf_objects(Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label')/a.scene,frames=SCENE_GT_FRAMES[a.scene])
    observed=data['observed'].to(a.device).float();rows=[]
    for frame,objects in frames.items():
        camera=prior.cameras[int(frame.split('_')[-1])];h,w=objects[0].mask.shape;k=camera.intrinsic.to(a.device).clone();k[0]*=w/camera.width;k[1]*=h/camera.height
        for obj in objects:
            qi=data['texts'].index(obj.query);names=list(data['scores']);colors=torch.stack([data['scores'][name][:,qi] for name in names],dim=1).to(a.device);colors=torch.cat([colors,observed[:,None]],dim=1)
            rendered,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten(),colors=colors,viewmats=camera.world_to_camera.to(a.device)[None],Ks=k[None],width=w,height=h,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
            mass=rendered[0,...,-1];valid=(mass/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8)
            response=torch.where(valid[...,None],rendered[0,:,:,:len(names)]/mass[...,None].clamp_min(1e-8),0).cpu().numpy()
            ap={name:float(average_precision_score(obj.mask.flatten(),response[...,j].flatten())) for j,name in enumerate(names)}
            from PIL import Image
            final_binary=response[...,names.index('final')]>.5
            reference=np.asarray(Image.open(Path('output/frozen_method_lerf2d_20260918')/a.scene/(frame+'_'+obj.query+'.png')))>0
            mismatches=int((final_binary!=reference).sum())
            rows.append(dict(frame=frame,query=obj.query,ap=ap,positive_support_recall=float(valid.cpu().numpy()[obj.mask].mean()),final_prediction_disagreement_pixels=mismatches,final_iou=float((final_binary&obj.mask).sum()/max((final_binary|obj.mask).sum(),1))))
        print(a.scene,frame,flush=True)
    result=dict(prep,observations=rows,observation_count=len(rows),macro_ap={k:float(np.mean([r['ap'][k] for r in rows])) for k in rows[0]['ap']},
        render_implementation_sha256=sha256_file(__file__),final_prediction_disagreement_pixels=sum(r['final_prediction_disagreement_pixels'] for r in rows),final_miou=float(np.mean([r['final_iou'] for r in rows])),geometry_unchanged=True,raw_cosine_not_membership=True,threshold_search=False,grid='native annotation grid',independent_test=False)
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(result['macro_ap'],flush=True)


if __name__=='__main__':
    with torch.no_grad():main()
