"""Matched native-map readout experiment; no target labels enter either head."""
import argparse, json, os, time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--gpu',default='0');p.add_argument('--output',required=True);a=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
    import torch
    from torch.nn import functional as F
    from radio_gs.models.native_query_encoders import load_siglip_text
    from radio_gs.querying.scannet_runtime import load, declared_source_views
    from radio_gs.querying.typed_evidence import solve_set, SetType
    torch.set_num_threads(4);start=time.monotonic();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    old=json.loads(Path(f'configs/consolidated/scannet/{a.scene}.scene.json').read_text())
    scene=json.loads(Path(f'output/scannet_native_completion/{a.scene}/scene.json').read_text())
    queries=json.loads(Path(f'configs/consolidated/scannet/{a.scene}.queries.json').read_text())['queries'];texts=[q['text'] for q in queries]
    encoder,tokenizer=load_siglip_text(scene['text_encoder'],'cuda')
    with torch.no_grad():
        tokens=tokenizer(texts,padding='max_length',max_length=64,truncation=True,return_tensors='pt').to('cuda')
        q=F.normalize(F.normalize(encoder(**tokens).pooler_output.float(),dim=-1).half().float(),dim=-1)
    del encoder,tokens
    field=load(scene['native_field']);observed=field['supported'].reshape(-1).cuda()
    cosine=torch.cat([x.float().cuda()@q.T for x in field['features'].split(8192)]);del field
    state=load(old['base_field'])['state_dict'];cal=load(old['calibration']);memory=load(old['memory'])
    bounds=state['native_query_thresholds.text.bounds'].cuda()
    threshold=(q@state['native_query_thresholds.text.linear.weight'].cuda().T+state['native_query_thresholds.text.linear.bias'].cuda()).flatten().clamp(*bounds)
    threshold+=(q@cal['weight'].cuda()+cal['bias'].cuda()).clamp(-cal['max_delta'],cal['max_delta'])
    logscale=state['readout_log_scale.text'].cuda().clamp(-4,6)
    competition=(q@memory['category_competition']['direction'].float().cuda()).tanh();del memory,state
    views=declared_source_views(observed,old,texts)
    positive=torch.zeros_like(cosine);negative=torch.zeros_like(cosine)
    for e in views:
        positive+=torch.sparse.mm(e.adjoint,e.positive.float()*e.confidence)
        negative+=torch.sparse.mm(e.adjoint,e.negative.float()*e.confidence)
    # Equal class and sign weight: all unknown elements contribute zero loss.
    wp=positive/positive.sum(0).clamp_min(1e-8);wn=negative/negative.sum(0).clamp_min(1e-8)
    delta=torch.zeros(len(q),device='cuda',requires_grad=True)
    ls=logscale.detach().clone().requires_grad_(True)
    opt=torch.optim.Adam([delta,ls],lr=.02)
    for step in range(200):
        opt.zero_grad();z=ls.clamp(-4,6).exp()*(cosine-threshold-.05*delta.tanh())+competition
        loss=(wp*F.softplus(-z)+wn*F.softplus(z)).sum()/len(q)+.01*delta.square().mean()+.01*(ls-logscale).square()
        loss.backward();opt.step()
    torch.save(dict(query_encodings=q.cpu(),threshold=threshold.cpu(),threshold_delta=(.05*delta.detach().tanh()).cpu(),logscale=ls.detach().cpu(),competition=competition.cpu(),source_frames=[v.frame for v in views],steps=200,ground_truth_used=False,contract='declared-category calibration, not an open-vocabulary query adapter'),out/'source_head.pt')
    results={}
    for arm,t,s in [('transferred',threshold,logscale),('source_refit',threshold+.05*delta.detach().tanh(),ls.detach())]:
        with torch.inference_mode():
            prior_logits=s.clamp(-4,6).exp()*(cosine-t)+competition
            prior=prior_logits.sigmoid();value,stats=solve_set(prior,views,SetType.TEXT_UNION,corroborate=False,consistency=True)
            logits=torch.where(value!=prior,torch.logit(value.clamp(1e-6,1-1e-6)),prior_logits)
        dest=out/arm;dest.mkdir(exist_ok=True)
        torch.save(dict(source_evidence_used=True,texts=texts,queries=queries,scores=logits.cpu(),category_logits=logits.cpu(),native_logits=prior_logits.cpu(),encodings=q.cpu(),semantic_geometry=scene['geometry'],semantic_observed=observed.cpu(),output_contract='categorical_logits_all_instances_no_independent_subset_head'),dest/'typed_readout.pt')
        results[arm]=stats
    report=dict(scene=a.scene,seconds=time.monotonic()-start,source_views=len(views),source_frames=[v.frame for v in views],geometry=scene['geometry'],native_field=scene['native_field'],old_predictions_used=False,old_scope_graph_used=False,ground_truth_used=False,calibration_loss=float(loss.detach()),arms=results)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
