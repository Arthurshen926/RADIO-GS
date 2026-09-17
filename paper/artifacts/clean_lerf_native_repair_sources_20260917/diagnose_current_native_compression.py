"""Matched source fusion before/after the deployed 64D projection.

No target RGB or masks enter features. GT is opened only to score diagnostics.
Both native branches inherit the same source query threshold; AP is primary.
"""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from gsplat import rasterization
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.models.native_query_encoders import NativeThreeQueryEncoder
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.v4.evaluation.lerf_common import load_labels,ground_truth_masks,binary_iou
from radio_gs.training.retrieval_metrics import binary_ranking_metrics
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.training.source_evidence_balancing import balance_source_views_and_origins


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);p.add_argument('--root',type=Path);p.add_argument('--output',type=Path);p.add_argument('--run',default='hard_source_scope_gate96');a=p.parse_args();torch.set_num_threads(4)
    root=a.root or Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/a.scene
    output=a.output or Path(f'paper/artifacts/{a.scene}_current_native_compression_20260916.json')
    if output.exists():raise FileExistsError(output)
    info=json.loads((root/'teachers.json').read_text());ep=root/'episodes.pt';assert sha256_file(ep)==json.loads(ep.with_suffix('.json').read_text())['sha256']
    data=torch.load(ep,weights_only=False);assert sha256_file(root/'teachers.json')==data['teacher_manifest_sha256']
    sr=info['teacher_files']['siglip'];tp=root/sr['path'];assert sha256_file(tp)==sr['sha256'];teacher=torch.load(tp,weights_only=False)
    fp=root/a.run/'field.pt';receipt=json.loads((fp.parent/'report.json').read_text());assert sha256_file(fp)==receipt['field_sha256']
    payload=torch.load(fp,weights_only=False);assert payload['geometry_binding']==data['geometry_binding']
    field=ElementRetrievalField(**payload['architecture']).to(a.device);field.load_state_dict(payload['state_dict']);field.requires_grad_(False)
    train=torch.tensor([info['frames'][int(v)] in info['train_frames'] for v in data['proposal_views']]);weights=data['region_probability'][train].float().to(a.device)
    descriptors=F.normalize(teacher['descriptors'][train].float().to(a.device),dim=-1)
    initialization=root/'source_fusion_initialization.pt'
    init_receipt=json.loads(initialization.with_suffix('.json').read_text())
    assert sha256_file(initialization)==init_receipt['sha256']
    init=torch.load(initialization,map_location='cpu',weights_only=False)
    policy=init.get('fusion_policy','region_mass_weighted')
    assert init['episode_sha256']==sha256_file(ep)
    assert init['siglip_teacher_sha256']==sr['sha256']
    assert torch.equal(torch.tensor(init['source_regions']),torch.where(train)[0])
    with torch.no_grad():
        if policy=='equal_available_source_views':
            views=data['proposal_views'][train].to(a.device)
            native,mass=balance_source_views_and_origins(descriptors,weights,views,torch.zeros_like(views))
        elif policy=='region_mass_weighted':
            mass=weights.sum(0);native=weights.T@descriptors/mass[:,None].clamp_min(1e-8)
        else:raise ValueError('Unsupported source fusion policy: '+str(policy))
        projected=native@field.native_decoder.weight
        error=float((projected[field.observed]-field.latent[field.observed,:64]).abs().max())
        torch.testing.assert_close(projected[field.observed],field.latent[field.observed,:64],atol=2e-5,rtol=2e-4)
        assert not field.native_decoder.bias.any()
        native=F.normalize(native,dim=-1);del weights,descriptors,projected,data,teacher
    grpath=Path('/mnt/pool/sqy/results/RADIO-GS/output/geometry_only_repair_final_20260914')/a.scene/'report.json';gr=json.loads(grpath.read_text())
    g=gr['results']['independent_scale'];geometry=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device=a.device);geometry.assert_binding(payload['geometry_binding'])
    prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256'])
    labels=Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label');annotations,categories,sh,sw=load_labels(labels,a.scene);assert not set(annotations)&set(info['train_frames'])
    encoder=NativeThreeQueryEncoder(payload['query_encoder'],a.device)
    rows={name:[] for name in ['full_native_fusion','native64','current_readout']}
    with torch.no_grad():
        q=encoder.encode_text(categories).half().float();threshold=field.native_query_thresholds['text'](q);scale=field.readout_log_scale['text'].clamp(-4,6).exp()
        response=[torch.sigmoid(scale*(native@F.normalize(q,dim=-1).T-threshold)),torch.sigmoid(scale*(field.native_cosine(q)-threshold)),field.retrieve(q,'text')['scores']]
        colors=torch.cat([r*field.observed[:,None] for r in response]+[field.observed.float()[:,None]],1);height=round(sh*480/sw)
        for frame,objects in sorted(annotations.items()):
            camera=prior.cameras[frame];k=camera.intrinsic.to(a.device).clone();k[0]*=480/camera.width;k[1]*=height/camera.height
            image,alpha,_=rasterization(means=geometry.get_xyz(),quats=geometry.get_rotation(),scales=geometry.get_scaling(),opacities=geometry.get_opacity().flatten(),colors=colors,viewmats=camera.world_to_camera.to(a.device)[None],Ks=k[None],width=480,height=height,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
            support=image[0,...,-1];valid=(support/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8)
            targets=ground_truth_masks(objects,categories,height,480,sh,sw)
            for si,name in enumerate(rows):
                score=image[0,...,si*len(q):(si+1)*len(q)]/support[...,None].clamp_min(1e-8)
                for ci,c in enumerate(categories):
                    truth=targets[c].to(a.device)
                    if not truth.any():continue
                    values=torch.where(valid,score[...,ci],torch.zeros_like(support));iou,_,_=binary_iou((values>.5)&valid,truth)
                    rows[name].append(dict(frame=frame,category=c,iou=iou,**binary_ranking_metrics(values,truth)))
        macro={name:{k:sum(r[k] for r in records)/len(records) for k in ['iou','average_precision']} for name,records in rows.items()}
    result=dict(schema='matched_native_compression.v1',scene=a.scene,diagnostic_only=True,target_rgb_used=False,benchmark_masks_for_metrics_only=True,
        width=480,threshold=.5,shared_support=True,unchanged_geometry=True,projection_max_error=error,field_sha256=sha256_file(fp),episode_sha256=sha256_file(ep),teacher_sha256=sha256_file(tp),geometry_sha256=g['geometry_sha256'],implementation_sha256=sha256_file(__file__),
        fusion_policy=policy,initialization_sha256=sha256_file(initialization),fusion_implementation_sha256=sha256_file('radio_gs/training/source_evidence_balancing.py'),
        calibration='same inherited source query threshold and scale on both native branches; full-native calibration not refitted; AP primary',macro=macro,observations=rows)
    output.write_text(json.dumps(result,indent=2)+'\n');print(a.scene,error,macro,flush=True)


if __name__=='__main__':main()
