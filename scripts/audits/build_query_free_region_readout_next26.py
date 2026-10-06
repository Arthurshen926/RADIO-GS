"""Test query-first native responses on query-free SAM proposals.

All evidence is source-side, fixed before benchmark labels are opened. The
same source-trained native threshold and scale calibrate every query.
"""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.utils.immutable_artifacts import sha256_file


def load(path, expected=None):
    if expected is not None:
        assert sha256_file(path) == expected
    return torch.load(path, map_location='cpu', weights_only=False)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args()
    torch.set_num_threads(4)
    root=Path('output/full_field_rebuild_next19')/a.scene
    fieldroot=root/'reconstruction/field';base=root/'evaluation'
    out=Path('output/query_free_region_readout_next26')/a.scene
    if out.exists():
        # A failed preparation may leave its newly created directory empty.
        assert not any(out.iterdir()), 'refuse to overwrite an existing preparation'
    else:
        out.mkdir(parents=True)
    old=json.loads((base/'report.json').read_text());assert old['complete']
    data=load(base/'scores.pt',old['scores_sha256'])
    bank=load(base/'text_online.encodings.pt',old['encoding_bank_sha256'])[0]
    fieldpath=fieldroot/'bounded_fallback_admission/field.pt'
    payload=load(fieldpath,old['field_sha256'])
    assert data['geometry_binding']==payload['geometry_binding']
    model=ElementRetrievalField(**payload['architecture']).cuda()
    model.load_state_dict(payload['state_dict']);model.eval();model.requires_grad_(False)
    info=json.loads((fieldroot/'teachers.json').read_text())
    epfile=fieldroot/'episodes.pt';epreceipt=json.loads((fieldroot/'episodes.json').read_text())
    ep=load(epfile,epreceipt['sha256'])
    teacherrec=info['teacher_files']['siglip']
    teacher=load(fieldroot/teacherrec['path'],teacherrec['sha256'])
    assert ep['geometry_binding']==payload['geometry_binding']
    assert ep['teacher_manifest_sha256']==sha256_file(fieldroot/'teachers.json')
    assert ep['region_probability'].shape[1]==len(model.latent)
    assert len(ep['region_probability'])==len(teacher['descriptors'])
    train=torch.tensor([info['frames'][int(vi)] in info['train_frames'] for vi in ep['proposal_views']])
    assert train.any()
    q=F.normalize(bank['encoding'].half().float().cuda(),dim=-1)
    descriptors=F.normalize(teacher['descriptors'][train].float().cuda(),dim=-1)
    assert descriptors.shape[1]==q.shape[1]
    region_cos=descriptors@q.T
    threshold=model.native_query_thresholds['text'](q)
    scale=model.readout_log_scale['text'].clamp(-4.,6.).exp()
    region_logits=scale*(region_cos-threshold[None])
    region_prob=region_logits.sigmoid()
    # Query-free source proposals are kept separate until the query is read.
    # No held-out image or target GT contributes to the element response.
    weights=ep['region_probability'][train].cuda()
    support=torch.zeros(len(model.latent),device='cuda')
    numerator=torch.zeros(len(model.latent),len(q),device='cuda')
    hard=torch.zeros_like(numerator)
    for start in range(0,len(weights),256):
        w=weights[start:start+256].float()
        support+=w.sum(0)
        numerator+=w.T@region_prob[start:start+256]
        hard+=w.T@(region_logits[start:start+256]>=0).float()
    covered=(support>1e-6)&model.observed
    avg=torch.where(covered[:,None],numerator/support[:,None].clamp_min(1e-8),data['scores']['final'].cuda())
    gate=torch.where(covered[:,None],hard/support[:,None].clamp_min(1e-8),data['scores']['final'].cuda())
    avg=avg.clamp(0,1);gate=gate.clamp(0,1)
    # A source-only, query-independent routing diagnostic. The mixed arm
    # changes only queries the deployed caption route cannot serve.
    route,accepted,_=model.grounded_transfer.route(q)
    scope_used=(data['scores']['final'].cuda()-data['scores']['native_compact_calibrated'].cuda()).abs().amax(0)>2e-5
    admitted=accepted&scope_used
    mixed=torch.where(admitted[None],data['scores']['final'].cuda(),avg)
    scores={n:data['scores']['final'].clone() for n in ['final','region_mean','region_gate','region_unrouted']}
    scores['region_mean']=avg.cpu();scores['region_gate']=gate.cpu();scores['region_unrouted']=mixed.cpu()
    torch.save(dict(data,scores=scores),out/'scores.pt')
    report=dict(complete=True,map_source_only=True,source_only=False,
                benchmark_query_text_used=True,score_query_conditioned=True,
                target_gt_used=False,target_rgb_used=False,
                region_evidence_sha256=epreceipt['sha256'],teacher_sha256=teacherrec['sha256'],
                evidence_kind='query_free_sam_source_proposals',field_sha256=old['field_sha256'],
                source_views=list(info['train_frames']),geometry_binding=payload['geometry_binding'],
                source_scale=float(scale),source_threshold_range=[float(threshold.min()),float(threshold.max())],
                region_cosine_range=[float(region_cos.min()),float(region_cos.max())],
                support_fraction=float(covered.float().mean()),unrouted_queries=int((~admitted).sum()),
                routed_queries=int(admitted.sum()),observed_regions=len(weights),
                arms=list(scores),source_metric_selected=False,parameter_search=False,
                baseline_score_sha256=old['scores_sha256'],scores_sha256=sha256_file(out/'scores.pt'),
                implementation_sha256=sha256_file(__file__),default_promoted=False)
    (out/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(a.scene,{k:report[k] for k in ['source_scale','source_threshold_range','region_cosine_range','support_fraction','unrouted_queries','routed_queries']},flush=True)


if __name__=='__main__':main()
