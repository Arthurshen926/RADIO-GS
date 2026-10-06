"""CPU, fixed-carrier diagnostic; source-view pseudo-GT is NOT a benchmark score.

Four source-raster stages share cameras, queries and a GT-derived denominator.
Region native/compact ranking agreement is also measured before mask blending.
No fitted parameters or benchmark-driven choices are produced.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.training.source_pixel_evidence import source_pixel_evidence_weights
from radio_gs.scripts.eval_scannet_vala_gaussian_protocol import assign_vala_pseudo_labels, volume_weighted_split_metrics
from radio_gs.scripts.eval_scannet_pointcloud_radio_gs import _read_label_ply
from radio_gs.scannet_constants import OPENGAUSSIAN_NYU40_CLASS_SPLITS, NYU40_ID_TO_NAME
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scene',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); torch.set_num_threads(4)
    a.output.mkdir(parents=True,exist_ok=False)
    root=Path('output/scannet_frozen_field_20260918')/a.scene
    r=root/'reconstruction/field'
    info=json.loads((r/'teachers.json').read_text())
    def asset(path,digest=None):
        if digest: assert sha256_file(path)==digest
        return torch.load(path,map_location='cpu',weights_only=False)
    def model(run):
        report=json.loads((r/run/'report.json').read_text())
        payload=asset(r/run/'field.pt',report['field_sha256'])
        m=ElementRetrievalField(**payload['architecture']);m.load_state_dict(payload['state_dict'])
        m.requires_grad_(False);return m,payload
    base,bp=model('pixel_evidence96');final,fp=model('bounded_fallback_admission')
    assert bp['geometry_binding']==fp['geometry_binding']
    query=asset(root/'official_protocol_evaluation/queries.pt')
    classes=OPENGAUSSIAN_NYU40_CLASS_SPLITS['19']
    q=F.normalize(query['encoding'][[query['texts'].index(NYU40_ID_TO_NAME[c]) for c in classes]].float(),dim=1)
    rec=info['teacher_files']['siglip'];teacher=asset(r/rec['path'],rec['sha256'])
    desc=F.normalize(teacher['descriptors'].float(),dim=1)
    ep=asset(r/'episodes.pt',json.loads((r/'episodes.json').read_text())['sha256'])
    auth=json.loads((r/'responsibility.json').read_text());assert sha256_file(r/'responsibility.json')==info['authority_sha256']
    basis=base.native_decoder.weight.detach();compact=desc@basis
    decoded=F.normalize(compact@basis.T,dim=1)
    native_region=desc@q.T;compact_region=decoded@q.T
    train_region=torch.tensor([int(v) in [info['frames'].index(f) for f in info['train_frames']] for v in ep['proposal_views']])
    agreement=float((native_region[train_region].argmax(1)==compact_region[train_region].argmax(1)).float().mean())
    n=len(base.latent);h=auth['metadata']['feature_height'];w=auth['metadata']['feature_width']
    numerator=torch.zeros(n,desc.shape[1]);mass=torch.zeros(n);operators={};frame_ids={}
    for frame in info['train_frames']:
        vi=info['frames'].index(frame);ids=torch.where(ep['proposal_views']==vi)[0];rec=auth['views'][vi]
        hit=asset(r/rec['relative_path'],rec['sha256']);pid=hit['pixel_ids'].long();gid=hit['gaussian_ids'].long();val=hit['base_weights'].float()
        weights,_=source_pixel_evidence_weights(ep['masks'][ids].flatten(1),pid,gid,val,n,partition_overlap=True)
        numerator+=weights.T@desc[ids];mass+=weights.sum(0)
        pm=torch.zeros(h*w).scatter_add_(0,pid,val)
        operators[frame]=torch.sparse_coo_tensor(torch.stack([pid,gid]),val/pm[pid].clamp_min(1e-8),(h*w,n)).coalesce()
        frame_ids[frame]=ids
    full=numerator/mass[:,None].clamp_min(1e-8);del numerator
    supported=base.observed & (mass>0)
    compact_elements=base.latent[:,:basis.shape[1]].detach()
    error=float(((full@basis)[supported]-compact_elements[supported]).abs().max())
    if error>2e-4:raise ValueError(f'Fusion replay disagrees: {error}')
    with torch.no_grad():final_scores=final.retrieve(q,'text')['scores'].cpu()
    # Persist predictions BEFORE reading semantic ground truth.
    elements={'native_uncompressed_fusion':F.normalize(full,dim=1)@q.T,
              'compact_fusion':base.native_cosine(q).detach(), 'final_scope':final_scores}
    predictions={k:torch.where(supported,torch.tensor(classes)[v.argmax(1)],0).numpy() for k,v in elements.items()}
    np.savez_compressed(a.output/'element_predictions.npz',**predictions)
    gr=json.loads((root/'geometry_report.json').read_text())['results']['independent_scale']
    geo=FrozenGaussianGeometry.load(gr['geometry'],expected_sha256=gr['geometry_sha256'],device='cpu')
    gtpath=Path('/mnt/pool/sqy/3d_understanding/scannet_og')/a.scene/(a.scene+'_vh_clean_2.labels.ply')
    xyz,labels=_read_label_ply(gtpath)
    gt,gt_audit=assign_vala_pseudo_labels(geo.xyz.numpy(),geo.scales.numpy(),geo.quaternions.numpy(),xyz,labels,radius_factor=5.,candidate_k=1000,fallback_k=1,class_balance=True,chunk_size=512)
    weights=(geo.scales.prod(1)*geo.opacities.flatten()).numpy()
    masks=torch.stack([torch.tensor(gt)==c for c in classes],dim=1).float()
    element_metrics={k:volume_weighted_split_metrics(gt,v,weights,classes) for k,v in predictions.items()}
    missing={str(c):float(weights[(gt==c)&~supported.numpy()].sum()/max(weights[gt==c].sum(),1e-20)) for c in classes if (gt==c).any()}
    rows=[]
    for frame,op in operators.items():
        ids=frame_ids[frame];sm=ep['masks'][ids].reshape(len(ids),-1).float();den=sm.sum(0)
        region_weights=sm.T/den[:,None].clamp_min(1)
        pixels1=F.normalize(region_weights@desc[ids],dim=1)@q.T
        pixels2=F.normalize((region_weights@compact[ids])@basis.T,dim=1)@q.T
        pixels3=F.normalize(torch.sparse.mm(op,compact_elements)@basis.T,dim=1)@q.T
        pixels4=torch.sparse.mm(op,final_scores*supported[:,None])
        truth=torch.sparse.mm(op,masks);valid=truth.sum(1)>1e-8
        target=truth.argmax(1);source_missing=float(((den==0)&valid).sum()/valid.sum().clamp_min(1))
        score_sets=dict(region_native=pixels1,region_compact=pixels2,fused_compact=pixels3,final_scope=pixels4)
        result={}
        for name,scores in score_sets.items():
            pred=scores.argmax(1);covered=den>0 if name.startswith('region_') else torch.sparse.mm(op,supported[:,None].float()).flatten()>0
            pred=pred.clone();pred[~covered]=-1
            ious=[];aps=[]
            for c in range(len(classes)):
                positive=(target==c)&valid
                if not positive.any():continue
                pp=(pred==c)&valid
                ious.append(float((pp&positive).sum()/(pp|positive).sum().clamp_min(1)))
                aps.append(float(average_precision_score(positive[valid].numpy(),scores[valid,c].numpy())))
            pred=pred.reshape(h,w);tg=target.reshape(h,w);vv=valid.reshape(h,w)
            tp=fpn=fn=0
            for dim in [0,1]:
                sl1=[slice(None),slice(None)];sl2=sl1.copy();sl1[dim]=slice(1,None);sl2[dim]=slice(None,-1);s1=tuple(sl1);s2=tuple(sl2)
                use=vv[s1]&vv[s2];pb=(pred[s1]!=pred[s2])&use;tb=(tg[s1]!=tg[s2])&use
                tp+=int((pb&tb).sum());fpn+=int((pb&~tb).sum());fn+=int((~pb&tb).sum())
            result[name]=dict(miou=float(np.mean(ious)),macro_ap=float(np.mean(aps)),boundary_edge_f1=2*tp/max(2*tp+fpn+fn,1))
        rows.append(dict(frame=frame,valid_pixels=int(valid.sum()),source_missing_fraction=source_missing,metrics=result))
        print(a.scene,frame,{k:round(v['miou'],4) for k,v in result.items()},flush=True)
    report=dict(scene=a.scene,diagnostic_only=True,independent_test=False,
        reference='Fixed current-carrier VALA pseudo labels projected through identical source responsibility operators; source views, not official 2D GT.',
        geometry_sha256=gr['geometry_sha256'],gt_sha256=sha256_file(gtpath),gt_audit=gt_audit,
        region_top1_retention=agreement,fusion_replay_max_error=error,
        semantic_missing_weight_by_class=missing,element_metrics=element_metrics,source_views=rows,
        source_macro={k:{m:float(np.mean([row['metrics'][k][m] for row in rows])) for m in ['miou','macro_ap','boundary_edge_f1']} for k in rows[0]['metrics']},
        implementation_sha256=sha256_file(__file__),teacher_sha256=info['teacher_files']['siglip']['sha256'],
        final_field_sha256=sha256_file(r/'bounded_fallback_admission/field.pt'))
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with torch.no_grad():main()
