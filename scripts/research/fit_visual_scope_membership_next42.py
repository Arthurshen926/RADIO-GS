"""Fixed shared visual membership fit; same source episodes as Stage1."""
import argparse
import gc
import json
import time
from pathlib import Path
import torch
from radio_gs.querying.query_region_evidence import load,digest
from radio_gs.querying.visual_scope_membership import (VisualMembershipHead,VisualMembershipRuntime,region_supervision,membership_loss,known_loss,balanced_query_risks,render_probabilities,load_runtime)
from scripts.research.fit_query_region_evidence_next42 import prepare,KEYS,clean_json
from radio_gs.training.partial_scope_metrics import episode_rows,aggregate,extent_measures


def attach_supervision(scene):
    runtime=VisualMembershipRuntime(scene['bank'],VisualMembershipHead(),device='cuda')
    # Project once per source view, and keep only small region/query targets.
    for view in sorted({e['view'] for e in scene['episodes']}):
        episodes=[e for e in scene['episodes'] if e['view']==view]
        projected=torch.sparse.mm(episodes[0]['op'],runtime.m).to_dense()
        allowed=runtime.allowed & (runtime.views!=view)
        for e in episodes:
            e['region']=region_supervision(projected,e,allowed)
        del projected
    del runtime;gc.collect();torch.cuda.empty_cache()


def coverage_rows(e):
    s=e['region'];out=[]
    for j,qi in enumerate(e['query_ids']):
        valid=s['valid'][:,j];pos=valid&(s['target'][:,j]>=.5);neg=valid&~pos
        out.append(dict(view=e['view'],query_id=qi,group=e['group'],valid=int(valid.sum()),positive=int(pos.sum()),negative=int(neg.sum()),classes=int(pos.any())+int(neg.any()),weak_area=int((s['effective_pixels'][:,j]<20).sum()),weak_known=int(((s['effective_pixels'][:,j]>=20)&(s['known_fraction'][:,j]<.5)).sum()),valid_mean_known=float(s['known_fraction'][:,j][valid].mean()) if valid.any() else None))
    return out


def standardize(head,runtimes,scenes):
    count=0;total=torch.zeros(8,device='cuda');square=total.clone()
    for key,s in scenes.items():
        for e in s['episodes']:
            if e['group']!='train':continue
            x=runtimes[key].features(s['q'][e['columns']])[e['region']['valid']]
            count+=len(x);total+=x.sum(0);square+=x.square().sum(0)
    if not count:raise RuntimeError('No adequately covered cross-view source region pairs')
    head.mean.copy_(total/count);head.scale.copy_((square/count-head.mean.square()).clamp_min(1e-6).sqrt())
    return count


def main(out):
    torch.set_num_threads(4);torch.manual_seed(0);started=time.monotonic()
    scenes={}
    for key in KEYS:
        print('PREPARE',key,flush=True);s=prepare(key,out);attach_supervision(s);scenes[key]=s
        rows=[r for e in s['episodes'] for r in coverage_rows(e)]
        (out/key/'coverage.json').write_text(json.dumps(rows,indent=2))
        print('COVERAGE',key,{g:sum(r['valid'] for r in rows if r['group']==g) for g in ['train','held_view','held_phrase']},flush=True)
    prep_seconds=time.monotonic()-started;results={}
    for arm in ['B1','B2']:
        start=time.monotonic();torch.manual_seed(0);head=VisualMembershipHead().cuda()
        runtimes={k:VisualMembershipRuntime(s['bank'],head,arm,'cuda') for k,s in scenes.items()}
        with torch.no_grad():fit_pairs=standardize(head,runtimes,scenes)
        opt=torch.optim.Adam(head.parameters(),lr=.01);generator=torch.Generator().manual_seed(0);trace=[];grad_max=0
        for step in range(200):
            key=KEYS[step%3];s=scenes[key];runtime=runtimes[key]
            pool=[e for e in s['episodes'] if e['group']=='train'];e=pool[int(torch.randint(len(pool),(1,),generator=generator))]
            cols=e['columns'];base=s['base'][:,cols];logits=runtime.membership_logits(s['q'][cols]);delta,_=runtime.transport(logits,base,e['view'])
            pred=render_probabilities(e['op'],base+delta)
            region=membership_loss(logits,e['region']);known=known_loss(pred,e)
            trust=((pred-e['prior']).square()*e['visible'][:,None]).sum()/e['visible'].sum().clamp_min(1)/len(cols)
            terms=[.1*trust]
            if region is not None:terms.append(region)
            if known is not None:terms.append(.25*known)
            loss=sum(terms);opt.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
            assert torch.isfinite(norm);grad_max=max(grad_max,float(norm));opt.step()
            if step%20==0 or step==199:
                trace.append(dict(step=step,scene=key,view=e['view'],queries=e['query_ids'],loss=float(loss),region_bce=None if region is None else float(region),known_brier=None if known is None else float(known),gradient_norm=float(norm)));print(arm,trace[-1],flush=True)
        directory=out/arm;directory.mkdir(exist_ok=True);checkpoint=directory/'head.pt'
        torch.save(dict(schema='visual_membership.next42.v2',state_dict={k:v.detach().cpu() for k,v in head.state_dict().items()},arm=arm,seed=0,steps=200,recipe_sha256=digest(out/'recipe.json')),checkpoint)
        rows=[];region_rows=[];activation={};query_times=[];cold={};visual={}
        with torch.no_grad():
            for key,s in scenes.items():
                runtime=runtimes[key];counts={g:dict(positive=0,negative=0,positive_active=0,negative_active=0,allowed_pairs=0,unlabelled_positive=0,unlabelled_pairs=0,unknown_supervision_positive=0,unknown_supervision_pairs=0,unlabelled_known_positive=0,unlabelled_known_positive_recalled=0,saturated=0,changed_elements=0,total_elements=0) for g in ['train','held_view','held_phrase']}
                unlabelled=torch.tensor([not x for x in s['bank']['associations']],device='cuda')
                dependence=[]
                for e in s['episodes']:
                    cols=e['columns'];q=s['q'][cols];base=s['base'][:,cols];torch.cuda.synchronize();t=time.monotonic()
                    logits=runtime.membership_logits(q);delta,state=runtime.transport(logits,base,e['view']);torch.cuda.synchronize();query_times.append((time.monotonic()-t)/len(cols))
                    pred=render_probabilities(e['op'],base+delta)
                    measures=episode_rows(extent_measures(pred,e,parent=e['prior']),e['view'],e['query_ids'])
                    for row in measures:row.update(scene=key,group=e['group']);rows.append(row)
                    sup=e['region'];risks=balanced_query_risks((logits.sigmoid()-sup['target']).square(),sup['target'],sup['valid'],sup['known_fraction'])
                    ablated=runtime.features(q).clone();ablated[...,:2]=head.mean[:2]
                    if arm=='B2':ablated[...,4:]=head.mean[4:]
                    dependence.append(float((head(ablated).sigmoid()-logits.sigmoid()).abs().mean()))
                    for j,qi in enumerate(e['query_ids']):
                        valid=sup['valid'][:,j];pos=valid&(sup['target'][:,j]>=.5);neg=valid&~pos
                        region_rows.append(dict(scene=key,view=e['view'],query_id=qi,group=e['group'],brier=float(risks[j]),positive=int(pos.sum()),negative=int(neg.sum()),positive_mean=float(logits[:,j].sigmoid()[pos].mean()) if pos.any() else None,negative_mean=float(logits[:,j].sigmoid()[neg].mean()) if neg.any() else None))
                    c=counts[e['group']];allowed=state['allowed'][:,None].expand_as(logits);untag=unlabelled[:,None]&allowed;unknown=~sup['valid']&allowed
                    c['positive']+=int(state['positive'].sum());c['negative']+=int(state['negative'].sum());c['positive_active']+=int((state['coefficient']>0).sum());c['negative_active']+=int((state['coefficient']<0).sum());c['allowed_pairs']+=int(allowed.sum());c['unlabelled_positive']+=int((state['positive']&untag).sum());c['unlabelled_pairs']+=int(untag.sum());c['unknown_supervision_positive']+=int((state['positive']&unknown).sum());c['unknown_supervision_pairs']+=int(unknown.sum());truth=untag&sup['valid']&(sup['target']>=.5);c['unlabelled_known_positive']+=int(truth.sum());c['unlabelled_known_positive_recalled']+=int((truth&state['positive']).sum());c['saturated']+=int((((state['probability']<.01)|(state['probability']>.99))&allowed).sum());c['changed_elements']+=int((delta!=0).sum());c['total_elements']+=delta.numel()
                activation[key]=counts;visual[key]=dict(mean_probability_change_when_image_features_replaced_by_fit_mean=sum(dependence)/len(dependence))
                # Generic cold runtime uses source query encodings + freshly recomputed
                # parent from prepare(), with no evaluator or benchmark score assets.
                addon=dict(kind='visual_membership',bank=str(out/key/'bank.pt'),checkpoint=str(checkpoint),arm=arm)
                other=load_runtime(addon,'cuda');q=s['q'][:2];base=s['base'][:,:2]
                warm=runtime.logits(q,base);fresh=other.logits(q,base)
                clean={k:v for k,v in s['bank'].items() if k not in ['text','associations','records']}
                erased=VisualMembershipRuntime(clean,head,arm,'cuda').logits(q,base)
                single=torch.cat([other.logits(q[i:i+1],base[:,i:i+1]) for i in range(len(q))],1)
                error=float((fresh-single).abs().max());assert torch.equal(warm,fresh) and torch.equal(warm,erased) and torch.isfinite(fresh).all() and error<=2e-6
                cold[key]=dict(no_benchmark_assets=True,parent='fresh source parent_model() logits',serialization_exact=True,labels_erased_exact=True,batch_max_abs_error=error,finite=True,queries=len(q),score_min=float(fresh.min()),score_max=float(fresh.max()))
                del other
        metrics={};regsummary={}
        import math
        for group in ['train','held_view','held_phrase']:
            selected=[r for r in rows if r['group']==group];metrics[group]=aggregate([{k:v for k,v in r.items() if k not in ['scene','group']} for r in selected])
            rr=[r for r in region_rows if r['group']==group];values=[r['brier'] for r in rr if math.isfinite(r['brier'])]
            regsummary[group]=dict(brier=sum(values)/len(values) if values else None,supervised_query_views=len(values),total_query_views=len(rr),positive=sum(r['positive'] for r in rr),negative=sum(r['negative'] for r in rr),single_class=sum((r['positive']>0)+(r['negative']>0)==1 for r in rr),unsupported=sum(r['positive']+r['negative']==0 for r in rr))
        result=dict(arm=arm,seconds=time.monotonic()-start,preparation_seconds=prep_seconds,fit_standardization_pairs=fit_pairs,gradient_max=grad_max,trace=trace,metrics=metrics,region=regsummary,activation=activation,visual_dependence=visual,cold=cold,mean_query_seconds=sum(query_times)/len(query_times),checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint))
        for name,value in [('source_diagnostics',result),('source_episodes',rows),('region_episodes',region_rows)]: (directory/(name+'.json')).write_text(json.dumps(clean_json(value),indent=2,allow_nan=False))
        results[arm]=result;del runtimes;gc.collect();torch.cuda.empty_cache()
    gate={}
    for group in ['held_view','held_phrase']:
        b1,b2=results['B1'],results['B2'];known=b1['metrics'][group]['known_brier']-b2['metrics'][group]['known_brier'];x,y=b1['region'][group]['brier'],b2['region'][group]['brier'];region=None if x is None or y is None else x-y
        gate[group]=dict(known_brier_gain=known,region_brier_gain=region,known_pass=known>=1e-4,region_pass=region is not None and region>=1e-4)
    gate['passed']=all(v['known_pass'] and v['region_pass'] for v in gate.values());gate['recorded_before_target_evaluation']=True;gate['checkpoints']={a:r['checkpoint_sha256'] for a,r in results.items()}
    (out/'source_gate.json').write_text(json.dumps(gate,indent=2));print('SOURCE GATE',gate,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(Path(a.output))
