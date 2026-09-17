"""Transfer source 2D complete teacher-mask decision boundaries, not partial 3D BCE.

Teacher masks are pseudo labels, not GT; only confidently detected source views
are used. Benchmark queries/labels and target RGB are never opened.
"""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from gsplat import rasterization
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.models.element_retrieval_field import load_element_retrieval_field, ElementRetrievalField
from radio_gs.models.native_query_threshold import fit_threshold_ridge
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True)
    p.add_argument('--root',type=Path);a=p.parse_args();torch.set_num_threads(4)
    base=Path('/mnt/pool/sqy/results/RADIO-GS/output');root=a.root if a.root is not None else base/'preserved_source_inventory_20260916'/a.scene
    parent=root/'scale_affinity_v1_direct_control';out=root/'full_source_query_threshold_control';out.mkdir(exist_ok=False)
    report=json.loads((parent/'report.json').read_text());info=json.loads((root/'teachers.json').read_text())
    receipt=json.loads((root/'episodes.json').read_text());assert sha256_file(root/'episodes.pt')==receipt['sha256']
    data=torch.load(root/'episodes.pt',weights_only=False)
    assert sha256_file(root/'teachers.json')==data['teacher_manifest_sha256']
    compiled=torch.load(root/'grounded_text_supervision.pt',weights_only=False)
    assert sha256_file(root/'grounded_text_supervision.pt')==report['grounded_text_supervision_sha256']
    rawpath=root/'source_grounded_text.pt';raw=torch.load(rawpath,weights_only=False)
    assert sha256_file(rawpath)==compiled['grounding_sha256']
    assert raw['teacher_manifest_sha256']==data['teacher_manifest_sha256']
    assert raw['frames']==info['train_frames'] and raw['queries']==compiled['queries'] and raw['benchmark_labels_used'] is False
    gr=json.loads((base/'geometry_only_repair_final_20260914'/a.scene/'report.json').read_text());g=gr['results']['independent_scale']
    geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device=a.device);geo.assert_binding(data['geometry_binding'])
    prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256'])
    field,payload=load_element_retrieval_field(parent/'field.pt',expected_sha256=report['field_sha256'],geometry=geo,expected_query_encoder=json.loads((root/'native_query_encoder.json').read_text()),device=a.device)
    q=data['queries']['text'].float().to(a.device);h,w=raw['probability'].shape[-2:]
    byquery={};rows=[]
    with torch.no_grad():
        response=(field.native_cosine(q)+1)/2
        colors=torch.cat([response*field.observed[:,None],field.observed[:,None].float()],1)
        for vi,frame in enumerate(raw['frames']):
            camera=prior.cameras[frame];k=camera.intrinsic.to(a.device).clone();k[0]*=w/camera.width;k[1]*=h/camera.height
            image,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities[:,0],
                colors=colors,viewmats=camera.world_to_camera.to(a.device)[None],Ks=k[None],width=w,height=h,
                near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
            mass=image[0,...,-1];valid=(mass/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8)
            maps=2*image[0,...,:-1]/mass[...,None].clamp_min(1e-8)-1
            for qi in range(len(q)):
                target=raw['probability'][vi,qi].to(a.device)>.5
                if raw['scores'][vi,qi]<.7 or target.sum()<20:continue
                byquery.setdefault(qi,[]).append((torch.where(valid,maps[...,qi],-2.).flatten(),target.flatten()))
        admitted=[];thresholds=[]
        for qi,values in sorted(byquery.items()):
            score=torch.cat([s for s,t in values]);target=torch.cat([t for s,t in values]);sortedscore,indices=score.sort(descending=True)
            tp=target[indices].float().cumsum(0);count=torch.arange(1,len(score)+1,device=a.device)
            end=torch.cat([sortedscore[:-1]!=sortedscore[1:],torch.ones(1,device=a.device,dtype=torch.bool)])&(sortedscore>=-1)
            iou=tp/(target.sum()+count-tp).clamp_min(1);iou[~end]=-1
            best=int(iou.argmax());threshold=float(sortedscore[best])-1e-7
            admitted.append(qi);thresholds.append(threshold)
            rows.append(dict(query_index=qi,source_views=len(values),threshold=threshold,source_teacher_iou=float(iou[best]),positive_pixels=int(target.sum())))
        features=q[admitted];targets=torch.tensor(thresholds,device=a.device);assert len(targets)>=4
        candidates=[.01,.1,1.,10.];validation=[]
        for penalty in candidates:
            errors=[]
            for i in range(len(targets)):
                keep=torch.arange(len(targets),device=a.device)!=i
                weight,bias=fit_threshold_ridge(features[keep],targets[keep],penalty)
                errors.append(float((F.normalize(features[i],dim=-1)@weight+bias-targets[i]).square()))
            validation.append(sum(errors)/len(errors))
        chosen=min(range(len(candidates)),key=validation.__getitem__)
        weight,bias=fit_threshold_ridge(features,targets,candidates[chosen])
    architecture=dict(payload['architecture'],native_query_thresholds={'text':{'lower':float(targets.min()),'upper':float(targets.max())}})
    state=dict(payload['state_dict']);state.update({'native_query_thresholds.text.linear.weight':weight.cpu()[None],
        'native_query_thresholds.text.linear.bias':bias.cpu()[None],
        'native_query_thresholds.text.bounds':torch.tensor([float(targets.min()),float(targets.max())])})
    cold=ElementRetrievalField(**architecture);cold.load_state_dict(state,strict=True)
    torch.save(dict(payload,architecture=architecture,state_dict=state),out/'field.pt')
    result=dict(report,schema='full_source_query_threshold_control.v1',field_sha256=sha256_file(out/'field.pt'),
        parent_field_sha256=report['field_sha256'],raw_source_grounding_sha256=sha256_file(rawpath),
        source_only=True,benchmark_data_used=False,native_text_image_max_error=None,
        native_cosine_exactly_preserved=True,image_prompt_unchanged=True,all_parent_tensors_unchanged=True,
        source_query_fits=rows,ridge_candidates=candidates,leave_one_concept_out_mse=validation,selected_ridge=candidates[chosen],
        implementation_sha256=sha256_file(__file__),readout_implementation_sha256=sha256_file('radio_gs/models/native_query_threshold.py'),
        label_contract='full source SAM3 masks on confidently detected source views; teacher pseudo negatives outside mask, not verified GT',
        calibration_rendering='mean native cosine at source raster; deployed sigmoid is applied before raster, not identical nonlinear mixing')
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(scene=a.scene,queries=len(rows),ridge=candidates[chosen],thresholds=thresholds)),flush=True)


if __name__=='__main__':main()
