"""Shared 200-step source-rendered relation head, matched raw/intervention arms."""
import argparse
import gc
import math
import json
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.querying.query_region_evidence import load,digest,held_phrase,RelationHead,RegionEvidence,parent_model
from radio_gs.training.partial_scope_metrics import episode_rows,aggregate,extent_measures

KEYS=['lerf_figurines','lerf_ramen','scannet_scene0000_00']

def clean_json(x):
    if isinstance(x,dict):return {k:clean_json(v) for k,v in x.items()}
    if isinstance(x,list):return [clean_json(v) for v in x]
    return None if isinstance(x,float) and not math.isfinite(x) else x


def prepare(key,out):
    bank=load(out/key/'bank.pt');model,predict,payload=parent_model(key);root=Path(payload['root'])
    cap=load(root/'caption_region_evidence.pt');info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text())
    views=[info['frames'].index(f) for f in info['train_frames']];held=set(bank['held_views']);mask=cap['masks'].flatten(1).bool();pv=cap['proposal_views'];records=cap['records']
    usable=[i for i,r in enumerate(records) if r['confidence']>=.7 and mask[i].sum()>=20]
    occurrences={}
    for i in usable:
        for qi in records[i]['phrase_indices']:occurrences.setdefault(qi,set()).add(int(pv[i]))
    eligible=[qi for qi,v in occurrences.items() if len(v-held)>=2]
    import hashlib
    ordered=sorted(eligible,key=lambda i:hashlib.sha256(cap['phrases'][i].encode()).hexdigest())
    train=[i for i in ordered if not held_phrase(cap['phrases'][i])][:48]
    heldq=[i for i in ordered if held_phrase(cap['phrases'][i])][:16]
    selected=train+heldq;qi_to_column={qi:j for j,qi in enumerate(selected)}
    q=F.normalize(cap['text_embeddings'][selected].float().cuda(),dim=1)
    with torch.no_grad():base=torch.cat([predict(part) for part in q.split(8)],dim=1).detach()
    episodes=[];n=len(model.observed);pixels=mask.shape[1]
    for vi in views:
        ids=[i for i in usable if int(pv[i])==vi]
        if not ids:continue
        hit=load(root/auth['views'][vi]['relative_path']);pid=hit['pixel_ids'].long().cuda();gid=hit['gaussian_ids'].long().cuda();w=hit['base_weights'].float().cuda();total=torch.zeros(pixels,device='cuda').scatter_add_(0,pid,w)
        w=w*model.observed[gid];mass=torch.zeros_like(total).scatter_add_(0,pid,w);visible=(mass/total.clamp_min(1e-8)>.5)&(total>1e-8)
        op=torch.sparse_coo_tensor(torch.stack([pid,gid]),w/mass[pid].clamp_min(1e-8),(pixels,n)).coalesce()
        local=mask[ids].cuda();desc=F.normalize(cap['native_descriptors'][ids].float().cuda(),dim=1)
        for group,queries in [('held_phrase',heldq),('held_view' if vi in held else 'train',train)]:
            present=[qi for qi in queries if vi in occurrences[qi]][:8]
            if group=='held_phrase' and vi in held:continue # separate phrase/view tests
            for start in range(0,len(present),4):
                chosen=present[start:start+4];cols=torch.tensor([qi_to_column[qi] for qi in chosen],device='cuda')
                affinity=desc@q[cols].T;targets=[];known=[]
                for j,qi in enumerate(chosen):
                    posids=[k for k,i in enumerate(ids) if qi in records[i]['phrase_indices']];positive=local[posids].any(0)
                    cutoff=affinity[posids,j].quantile(.25)-.03
                    eligible=(affinity[:,j]<cutoff)&~(local&positive[None]).any(1)
                    negative=local[eligible].any(0)&~positive
                    targets.append(positive);known.append(positive|negative)
                target=torch.stack(targets,1).float();known=torch.stack(known,1)&visible[:,None]
                b=base[:,cols];prior=torch.sparse.mm(op,b.sigmoid())
                episodes.append(dict(key=key,view=vi,group=group,columns=cols,op=op,target=target,known=known,visible=visible,prior=prior,query_ids=chosen))
    manifest=dict(phrases=[dict(id=qi,text=cap['phrases'][qi],held=qi in heldq) for qi in selected],held_views=sorted(held),episodes=[dict(view=e['view'],group=e['group'],query_ids=e['query_ids'],positive=int((e['known']&e['target'].bool()).sum()),negative=int((e['known']&~e['target'].bool()).sum())) for e in episodes],split_before_fit=True,holdout_scope='additional head only; base map saw all source views')
    (out/key/'source_split.json').write_text(json.dumps(manifest,indent=2))
    del cap,model;gc.collect();torch.cuda.empty_cache()
    return dict(bank=bank,q=q,base=base,episodes=episodes)


def known_loss(pred,e):
    risks=[]
    for label in [True,False]:
        domain=e['known']&(e['target'].bool()==label)
        count=domain.sum(0);valid=count>0
        if valid.any():risks.append((((pred-e['target']).square()*domain).sum(0)/count.clamp_min(1))[valid].mean())
    return torch.stack(risks).mean() if risks else pred.sum()*0


def margin(pred,e):
    p=e['known']&e['target'].bool();n=e['known']&~e['target'].bool();valid=(p.sum(0)>0)&(n.sum(0)>0)
    value=(pred*p).sum(0)/p.sum(0).clamp_min(1)-(pred*n).sum(0)/n.sum(0).clamp_min(1)
    return value[valid]


def main(out):
    torch.set_num_threads(4);torch.manual_seed(0)
    scenes={key:prepare(key,out) for key in KEYS}
    alltrain=[(key,e) for key,s in scenes.items() for e in s['episodes'] if e['group']=='train']
    assert alltrain
    results={}
    for arm in ['B1','B2']:
        start=time.monotonic();torch.manual_seed(0);head=RelationHead().cuda();opt=torch.optim.Adam(head.parameters(),lr=.01)
        runtimes={k:RegionEvidence(s['bank'],head,arm,'cuda') for k,s in scenes.items()}
        # Round-robin scene budget, deterministic random episode sampling seed0.
        generator=torch.Generator().manual_seed(0);trace=[];grad_max=0
        for step in range(200):
            key=KEYS[step%3];s=scenes[key];pool=[e for e in s['episodes'] if e['group']=='train'];e=pool[int(torch.randint(len(pool),(1,),generator=generator))]
            cols=e['columns'];base=s['base'][:,cols];delta=runtimes[key].delta(s['q'][cols],base,e['view']);value=(base+delta).sigmoid();pred=torch.sparse.mm(e['op'],value)
            risk=known_loss(pred,e);m=margin(pred,e);discrimination=F.relu(.25-m).mean() if len(m) else risk*0
            # Visible-mass trust region, including U, not an unknown-negative loss.
            trust=((pred-e['prior']).square()*e['visible'][:,None]).sum()/e['visible'].sum().clamp_min(1)/len(cols)
            loss=risk+.02*discrimination+.1*trust
            opt.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
            assert torch.isfinite(norm) and all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters())
            grad_max=max(grad_max,float(norm));opt.step()
            if step%20==0:trace.append(dict(step=step,loss=float(loss),known_brier=float(risk),gradient_norm=float(norm)));print(arm,trace[-1],flush=True)
        assert grad_max>0 and any(bool((p!=0).any()) for p in head.net[-1].parameters())
        checkpoint=out/arm/'head.pt';checkpoint.parent.mkdir(exist_ok=True)
        torch.save(dict(state_dict={k:v.detach().cpu() for k,v in head.state_dict().items()},arm=arm,seed=0,steps=200),checkpoint)
        rows=[];margins={};coverage={};query_times=[]
        with torch.no_grad():
            for key,s in scenes.items():
                runtime=runtimes[key];changes=[]
                for e in s['episodes']:
                    cols=e['columns'];base=s['base'][:,cols];torch.cuda.synchronize();t=time.monotonic();delta=runtime.delta(s['q'][cols],base,e['view']);torch.cuda.synchronize();query_times.append((time.monotonic()-t)/len(cols))
                    pred=torch.sparse.mm(e['op'],(base+delta).sigmoid());changes.append(float(delta.abs().max()))
                    # Task A's corrected P/N/U metric implementation is read-only.
                    measures=episode_rows(extent_measures(pred,e,parent=e['prior']),e['view'],e['query_ids'])
                    for j,row in enumerate(measures):row.update(scene=key,view=e['view'],group=e['group'],query_id=e['query_ids'][j]);rows.append(row)
                    mm=margin(pred,e);margins.setdefault(e['group'],[]).extend(mm.cpu().tolist())
                coverage[key]=dict(max_abs_logit_change=max(changes),nonzero_information_transfer=max(changes)>0)
        summary={}
        for group in ['train','held_view','held_phrase']:
            selected=[r for r in rows if r['group']==group];summary[group]=aggregate([{k:v for k,v in r.items() if k not in ['scene','group']} for r in selected]);summary[group]['independent_identity_margin']=sum(margins[group])/len(margins[group]) if margins.get(group) else None;summary[group]['margin_episodes']=len(margins.get(group,[]))
        result=dict(arm=arm,seconds=time.monotonic()-start,gradient_max=grad_max,trace=trace,metrics=summary,coverage=coverage,mean_query_seconds=sum(query_times)/len(query_times),checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),identity_margin_semantics='source-rendered positive-minus-qualified-disjoint-negative mean probability; query source view excluded from all bank prototypes, no verified physical identity labels')
        (out/arm/'source_diagnostics.json').write_text(json.dumps(result,indent=2));(out/arm/'source_episodes.json').write_text(json.dumps(clean_json(rows),indent=2));results[arm]=result
    gate={}
    for group in ['held_view','held_phrase']:
        b1=results['B1']['metrics'][group];b2=results['B2']['metrics'][group]
        gate[group]=dict(brier_improved=b2['known_brier'] is not None and b1['known_brier'] is not None and b2['known_brier']<b1['known_brier'],margin_improved=b2['independent_identity_margin'] is not None and b1['independent_identity_margin'] is not None and b2['independent_identity_margin']>b1['independent_identity_margin'])
    gate['passed']=all(v['brier_improved'] and v['margin_improved'] for v in gate.values());gate['recorded_before_target_evaluation']=True;gate['checkpoints']={a:r['checkpoint_sha256'] for a,r in results.items()}
    (out/'source_gate.json').write_text(json.dumps(gate,indent=2));print('SOURCE GATE',gate,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(Path(a.output))
