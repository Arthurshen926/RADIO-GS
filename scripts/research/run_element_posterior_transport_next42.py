"""Fixed Stage3: reuse banks/heads, source screen, then six real evaluations."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import time
import torch
from radio_gs.querying.query_region_evidence import load,digest
from radio_gs.querying.element_posterior_transport import camera_groups,load_runtime
from radio_gs.querying.visual_scope_membership import load_runtime as old_runtime
from scripts.research.fit_query_region_evidence_next42 import prepare,KEYS,clean_json
from radio_gs.training.partial_scope_metrics import extent_measures,episode_rows,aggregate

STAGE2=Path('output/query_region_evidence_next42/run_20260928_stage2')
STAGE1=Path('output/query_region_evidence_next42/run_20260928_stage1')

def write(path,value):path.write_text(json.dumps(clean_json(value),indent=2,allow_nan=False))


def metadata(key,out):
    bank=load(out/key/'bank.pt');p=load(Path('output/taskwise_scope_composition_next38')/key/'field.pt');root=Path(p['root'])
    info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text())
    views=[info['frames'].index(f) for f in info['train_frames']]
    n=bank['shape'][0];pixels=bank['masks'].shape[-2]*bank['masks'].shape[-1]
    visibility=torch.zeros(n);bindings=[root/'teachers.json',root/'responsibility.json']
    # Reproduce builder spatial_support denominator exactly. It is a vector of
    # per-element counts, not a guessed view-by-element visibility matrix.
    for vi in views:
        path=root/auth['views'][vi]['relative_path'];bindings.append(path);hit=load(path)
        pid=hit['pixel_ids'].long().cuda();gid=hit['gaussian_ids'].long().cuda()
        w=hit['base_weights'].float().cuda()*p['observed'][hit['gaussian_ids'].long()].cuda()
        mass=torch.zeros(pixels,device='cuda').scatter_add_(0,pid,w)
        op=torch.sparse_coo_tensor(torch.stack([pid,gid]),w/mass[pid].clamp_min(1e-8),(pixels,n)).coalesce()
        visibility+=(torch.sparse.sum(op,dim=0).to_dense()>1e-8).cpu()
    poses=[auth['metadata']['cameras'][str(info['frames'][vi])]['world_to_camera'] for vi in views]
    groups,camera=camera_groups(poses);mapping=dict(zip(views,groups.tolist()))
    keep=torch.ones(bank['shape'][1],dtype=torch.bool);seen_masks=set();seen_ids=set()
    import hashlib
    for k,v in enumerate(bank['views'].tolist()):
        mh=(v,hashlib.sha256(bank['masks'][k].numpy().tobytes()).hexdigest())
        record=bank['records'][k];identity=(v,record['origin'],record['origin_index'])
        if mh in seen_masks or identity in seen_ids:keep[k]=False
        seen_masks.add(mh);seen_ids.add(identity)
    ii,kk=bank['indices'].long();w=bank['weights'].double()*visibility[ii].clamp_min(1)
    # Existing normalized shares sum to the original strongest footprint. After
    # removing duplicate masks, preserve that total with renormalized shares.
    pair=ii*(max(views)+1)+bank['views'][kk];_,inv=torch.unique(pair,return_inverse=True)
    total=torch.zeros(int(inv.max())+1,dtype=torch.float64).scatter_add_(0,inv,w)
    retained=torch.zeros_like(total).scatter_add_(0,inv,w*keep[kk])
    weights=w*keep[kk]*total[inv]/retained[inv].clamp_min(1e-30)
    meta=dict(schema='element_posterior.next42.v3',keep=keep,footprint_weights=weights,visibility=visibility,region_groups=torch.tensor([mapping[v] for v in bank['views'].tolist()]),bank_sha256=digest(out/key/'bank.pt'))
    torch.save(meta,out/key/'transport.pt')
    write(out/key/'camera_evidence_bind.json',dict(source_views=views,frames=[info['frames'][v] for v in views],groups=groups.tolist(),world_to_camera=poses,camera=camera,grouping='source-only deterministic FPS, center / source bbox diagonal plus unit forward, first camera seed, nearest assignment',fallback=False,duplicates_removed=int((~keep).sum()),visibility_shape=list(visibility.shape),visibility_definition='count across source train views of sum normalized observed responsibility per element >1e-8; identical builder formula',bank_sha256=meta['bank_sha256'],transport_sha256=digest(out/key/'transport.pt'),inputs=[dict(path=str(p),sha256=digest(p)) for p in bindings]))
    del op;gc.collect();torch.cuda.empty_cache()


def summarize(rows):
    return aggregate([{k:v for k,v in r.items() if k not in ['scene','group','method']} for r in rows])


def source(out):
    allrows=[];diagnostics={};started=time.monotonic()
    with torch.no_grad():
        for key in KEYS:
            print('PREPARE',key,flush=True);metadata(key,out);s=prepare(key,out)
            assert (out/key/'source_split.json').read_bytes()==(STAGE2/key/'source_split.json').read_bytes()
            for arm in ['B1','B2']:
                addon=dict(kind='element_posterior',arm=arm,bank=str(out/key/'bank.pt'),checkpoint=str(out/arm/'head.pt'),transport=str(out/key/'transport.pt'))
                runtime=load_runtime(addon,'cuda');old=old_runtime(addon,'cuda');stats=[];times=[]
                for e in s['episodes']:
                    cols=e['columns'];q=s['q'][cols];base=s['base'][:,cols]
                    torch.cuda.synchronize();t=time.monotonic();member=runtime.membership_logits(q);new,state=runtime.transport(member,base,e['view']);torch.cuda.synchronize();times.append((time.monotonic()-t)/len(cols))
                    delta,oldstate=old.transport(member,base,e['view'])
                    for name,value in [('PARENT',base),('old_'+arm,base+delta),(arm,new)]:
                        if name=='PARENT' and arm=='B2':continue
                        pred=torch.sparse.mm(e['op'],value.sigmoid());rows=episode_rows(extent_measures(pred,e,parent=e['prior']),e['view'],e['query_ids'])
                        for row in rows:row.update(scene=key,group=e['group'],method=name);allrows.append(row)
                    gm=state['group_mass'];nz=gm[gm>0]
                    # Same source model diagnostic as outer readback, not GT error.
                    ii,kk=runtime.m.indices();r=state['probability'];p=base.sigmoid();eligible=oldstate['positive'][kk]&(r[kk]-p[ii]>.1)
                    blocked=eligible&(oldstate['coefficient'][kk]<=0);edgeweight=runtime.m.values()[:,None].double()
                    stats.append(dict(view=e['view'],group=e['group'],queries=e['query_ids'],positive_regions=int(state['positive'].sum()),negative_regions=int(state['negative'].sum()),positive_elements=int((state['probability_change']>0).sum()),negative_elements=int((state['probability_change']<0).sum()),total_elements=new.numel(),group_nonzero=int((gm>0).sum()),group_total=gm.numel(),group_mass_sum=float(gm.sum()),group_mass_quantiles=torch.quantile(nz,torch.tensor([0.,.5,.9,.99,1.],device='cuda',dtype=torch.float64)).tolist() if len(nz) else [],potential_low_member_weight=float((eligible*edgeweight).sum()),old_suppressed_weight=float((blocked*edgeweight).sum())))
                # Generic cold replay: source parent recomputed by prepare, no target
                # files in addon. Query batch and A-B-A test actual frozen heads.
                q=s['q'][:2];base=s['base'][:,:2];warm=runtime.logits(q,base);fresh=load_runtime(addon,'cuda');cold=fresh.logits(q,base)
                single=torch.cat([fresh.logits(q[j:j+1],base[:,j:j+1]) for j in range(2)],1)
                a=fresh.logits(q[:1],base[:,:1]);fresh.logits(q[1:],base[:,1:]);aa=fresh.logits(q[:1],base[:,:1])
                stripped={k:v for k,v in s['bank'].items() if k not in ['text','associations','records']}
                from radio_gs.querying.element_posterior_transport import ElementPosteriorRuntime
                erased=ElementPosteriorRuntime(stripped,runtime.head,load(out/key/'transport.pt'),arm,'cuda').logits(q,base)
                error=float((single-cold).abs().max());assert torch.equal(warm,cold) and torch.equal(a,aa) and torch.equal(warm,erased) and error<=2e-6, dict(scene=key,arm=arm,warm_cold=float((warm-cold).abs().max()),aba=float((a-aa).abs().max()),labels=float((warm-erased).abs().max()),batch=error)
                diagnostics.setdefault(arm,{})[key]=dict(activation=stats,mean_query_seconds=sum(times)/len(times),cold=dict(serialization_exact=True,labels_erased_exact=True,aba_exact=True,batch_max_abs_error=error,no_benchmark_assets=True,finite=bool(torch.isfinite(cold).all())))
                write(out/arm/'source_runtime.json',diagnostics[arm]);write(out/'source_episodes.json',allrows)
                del runtime,old,fresh;gc.collect();torch.cuda.empty_cache()
            del s;gc.collect();torch.cuda.empty_cache()
    summaries={name:{group:summarize([r for r in allrows if r['method']==name and r['group'] in groups]) for group,groups in [('train',['train']),('held_view',['held_view']),('held_phrase',['held_phrase']),('combined',['held_view','held_phrase'])]} for name in ['PARENT','old_B1','old_B2','B1','B2']}
    parent=summaries['PARENT']['combined'];screen={a:dict(known_brier=summaries[a]['combined']['known_brier'],negative_fpr=summaries[a]['combined']['negative_selection'],negative_fpr_delta=summaries[a]['combined']['negative_selection']-parent['negative_selection']) for a in ['B1','B2']}
    eligible=[a for a in screen if screen[a]['negative_fpr_delta']<=.005]
    preferred=min(eligible,key=lambda a:screen[a]['known_brier']) if eligible else None
    gate=dict(preferred_arm=preferred,passed=preferred is not None,arms=screen,parent=parent,criterion='minimum combined held-view/held-phrase query-view known Brier among arms with aggregate negative FPR <= parent + .005',recorded_before_target_evaluation=True,global_across_scenes=True)
    write(out/'source_diagnostics.json',dict(metrics=summaries,seconds=time.monotonic()-started,prototype_exclusion='all held source views and current episode view; phrase heldout identical Stage2; base map saw these views',no_benchmark_training=True,no_fit=True));write(out/'source_gate.json',gate)
    print('SOURCE GATE',gate,flush=True)


def main(out):
    torch.set_num_threads(4);torch.manual_seed(0)
    assert (out/'source_before.json').exists() and not (out/'recipe.json').exists()
    recipe=dict(schema='element_posterior.next42.v3',stage1=str(STAGE1),stage2=str(STAGE2),scenes=KEYS,arms=['B1','B2'],parent_opinion=1,maximum_camera_groups=6,confidence_positive='relu(2r-1)',confidence_negative='relu(1-2r) * max same-view disjoint positive confidence',group_mass='max actual per-view query evidence mass capped 1',group_displacement='footprint * directional confidence weighted mean; missing mass zero',no_training=True,no_encoding=True,no_tuning=True,source_gate_negative_fpr_absolute=.005,target_screen=dict(lerf_iou_pp=.5,lerf_selected_pp=.5,lerf_ap_pp=-.25,scannet_each_pp=-.5),not_independent_bayes=True)
    for key in KEYS:
        (out/key).mkdir();(out/key/'bank.pt').symlink_to((STAGE2/key/'bank.pt').resolve())
    for arm in ['B1','B2']:
        (out/arm).mkdir();(out/arm/'head.pt').symlink_to((STAGE2/arm/'head.pt').resolve())
    recipe['bindings']={str(p):digest(p) for p in [*[out/k/'bank.pt' for k in KEYS],*[out/a/'head.pt' for a in ['B1','B2']]]}
    write(out/'recipe.json',recipe);source(out)
    from scripts.research.evaluate_element_posterior_transport_next42 import main as evaluate
    evaluate(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(Path(a.output))
