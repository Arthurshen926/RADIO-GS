"""Independent scope definitions, physical objectives and actual-mask evaluation."""
import gc
from pathlib import Path
import numpy as np
import torch
from radio_gs.query_field.storage import load_field
from radio_gs.query_field.extent_decision import render_binary, CUTOFF
from radio_gs.scripts.eval_opengaussian_lerf_baseline import load_binary_mask
from scripts.research.run_query_field_signed_scope import ROOT,PARENT,CANONICAL,ARMS,OLD_ARMS,NEW_ARMS,CANDIDATE,HYBRIDS,design
from scripts.research.run_query_field_relations import read,write,receipt,check
from scripts.research.summarize_query_field_relations import paired


def objective(field,selected,scopes,prepared):
    confidence=prepared['confidence'].double().numpy();active=confidence>0
    cost=.1*float(prepared['cost'].double().numpy()@selected);data=0.
    for vi,(view,scope) in enumerate(zip(field.extent_factors.views,scopes)):
        if not active[vi]:continue
        alpha=render_binary(view.pointer.numpy(),view.index.numpy(),view.alpha.numpy(),selected).astype(np.float64).clip(1e-6,1-1e-6)
        p=alpha*(1-CUTOFF)/(alpha*(1-CUTOFF)+CUTOFF*(1-alpha));q=p.clip(1e-5,1-1e-5)
        y=scope['target'].double().numpy();pos=scope['positive'].numpy();neg=scope['negative'].numpy();known=pos|neg
        bce=-y*np.log(q)-(1-y)*np.log1p(-q)
        loss=.5*((bce*pos).sum()/max(pos.sum(),1)+(bce*neg).sum()/max(neg.sum(),1))
        loss+=1-(2*(p*y*known).sum()+1)/((p*known).sum()+(y*known).sum()+1)
        data+=confidence[vi]/int(active.sum())*loss
    return dict(data=float(data),cost=cost,total=float(data+cost))


def mix_numpy(pos,negative,confidence):
    y=pos['target'].numpy();n=negative.numpy()
    conflict=n&(y>.5)
    return dict(target=pos['target'],positive=pos['positive'],negative=torch.from_numpy(n&~conflict),confidence=confidence),int(conflict.sum())


def scopes_for(qi,atom,previous,fixed):
    result={CANDIDATE:atom[qi],**{a:[] for a in HYBRIDS}};conflicts={a:[] for a in HYBRIDS}
    for vi,(native,matched) in enumerate(zip(previous['native_complete_scope'][qi],previous['matched_positive_scope'][qi])):
        n=fixed['views'][vi]['negative'][:,qi];c=native['confidence']
        for arm,p,negative in [(HYBRIDS[0],matched,native['negative']),(HYBRIDS[1],native,n),(HYBRIDS[2],matched,n)]:
            s,k=mix_numpy(p,negative,c);result[arm].append(s);conflicts[arm].append(k)
    return result,conflicts


def replay_preparation(field,prior,scopes,prepared):
    count=len(prior);positive=np.zeros(count);known=np.zeros(count);cost=np.zeros(count)
    for view,scope in zip(field.extent_preparation.views,scopes):
        c=float(scope['confidence']);p=scope['positive'].numpy();n=scope['negative'].numpy()
        if c<=0 or not p.any():continue
        row,col=view.index.numpy();weight=view.weight.double().numpy();labels=(p|n)*c
        known+=np.bincount(col,weights=weight*labels[row],minlength=count)
        positive+=np.bincount(col,weights=weight*labels[row]*scope['target'].numpy()[row],minlength=count)
        cost+=c*view.footprint.double().numpy()
    initial=np.where(known>1e-8,positive/np.maximum(known,1e-8),prior.double().numpy())
    cost/=max(float(cost@initial),1e-8)
    e=float(np.max(np.abs(initial-prepared['initial'].numpy())))
    ce=float(np.max(np.abs(cost-prepared['cost'].numpy())/np.maximum(cost,1e-8)))
    assert e<=2e-6 and ce<=2e-5,(e,ce)
    return dict(initial_max_absolute=e,cost_max_relative=ce)


def main():
    design();cfg=read(ROOT/'prediction_config.json');summary=read(ROOT/'summary.json')
    for r in read(ROOT/'predictions_complete.json')['records']:check(r)
    scenes={}
    for job in cfg['jobs']:
        scene=job['scene'];out=ROOT/'predictions'/job['name'];record=read(out/'prediction.json')
        for key in ['prepared','atomic_scopes','parent_predictions','parent_scopes','reference_scope','field_manifest']:check(record[key])
        check(dict(path=str(out/'predictions.pt'),sha256=record['predictions_sha256']))
        assert not record['target_labels_opened'] and record['candidate_source_free']
        data=torch.load(out/'predictions.pt',weights_only=True);old=torch.load(record['parent_predictions']['path'],weights_only=True)
        previous=torch.load(record['parent_scopes']['path'],weights_only=True);fixed=torch.load(record['reference_scope']['path'],weights_only=True)
        atomic=torch.load(record['atomic_scopes']['path'],weights_only=True);prepared=torch.load(record['prepared']['path'],weights_only=True)
        prior=torch.load(CANONICAL/'inputs'/scene/'response.pt',weights_only=True)['semantic_probability']
        for arm in OLD_ARMS:
            assert torch.equal(data['membership_by_control'][arm],old['membership_by_control'][arm])
            assert torch.equal(data['selected_by_control'][arm],old['selected_by_control'][arm])
        del old;gc.collect();field,_=load_field(job['field'])
        for k,value in data['geometry'].items():assert torch.equal(value,getattr(field,k))
        # Independent NumPy semantic intervals and atom/neighbourhood signs.
        queries=data['native_query_vectors'].double().numpy();queries/=np.linalg.norm(queries,axis=1,keepdims=True)
        negatives=data['generic_negative_vectors'].double().numpy();negatives/=np.linalg.norm(negatives,axis=1,keepdims=True)
        source_agreement={a:[] for a in ['native_complete_scope',CANDIDATE]}
        for vi,(view,pv) in enumerate(zip(field.scope_topology.views,field.extent_preparation.views)):
            semantic=view.semantic.double().numpy();semantic/=np.linalg.norm(semantic,axis=1,keepdims=True)
            probability=1/(1+np.exp(-10*(semantic@queries.T-(semantic@negatives.T).max(1)[:,None])))
            projected=(pv.operator()@prior).numpy();valid=pv.valid.numpy()
            incidence=view.incidence.numpy();neighbors=view.neighborhood.numpy();covered=incidence.any(1)
            for qi in range(len(queries)):
                prob=probability[:,qi];lower=np.where(incidence,prob[None],1.).min(1);upper=np.where(incidence,prob[None],0.).max(1)
                ap=covered&(lower>.5);an=covered&(upper<.5)
                target=ap[neighbors[:,4]].astype(np.float32);pos=ap[neighbors].all(1)&valid
                neg=an[neighbors].all(1)&valid&~(projected[:,qi]>.5)
                actual=atomic[qi][vi]
                for key,expected in [('target',target),('positive',pos),('negative',neg)]:
                    assert np.array_equal(actual[key].numpy(),expected),(scene,vi,qi,key)
                native=previous['native_complete_scope'][qi][vi]
                assert torch.equal(actual['confidence'],native['confidence'])
                c=max(float((2*prob[prob>.5]-1).max()),0.) if (prob>.5).any() else 0.
                assert abs(c-float(actual['confidence']))<=2e-6
                diagnostic=record['diagnostics'][qi]['atomic'][vi]
                assert diagnostic['positive_atoms']==int(ap.sum()) and diagnostic['negative_atoms']==int(an.sum())
                assert diagnostic['positive_rays']==int(pos.sum()) and diagnostic['negative_rays']==int(neg.sum())
                truth=fixed['views'][vi]['target'][:,qi].numpy()>.5
                if (truth&valid).any():
                    for arm,s in [('native_complete_scope',native),(CANDIDATE,actual)]:
                        p=s['positive'].numpy();n=s['negative'].numpy();active=bool(s['confidence']>0 and p.any())
                        t=s['target'].numpy()>.5
                        source_agreement[arm].append(dict(query=qi,view=vi,active=active,
                            iou=int((t&truth&valid).sum())/max(int(((t|truth)&valid).sum()),1),
                            positive_rays=int(p.sum()),positive_true=int((p&truth).sum()),
                            negative_rays=int(n.sum()),negative_conflicting=int((n&truth).sum())))
        probes={};counts={a:dict(active_queries=0,active_views=0,conflicts=0) for a in NEW_ARMS}
        raw_permission=dict(inactive_reference_views=0,newly_active_views=0,newly_active_negative_rays=0)
        for qi,text in enumerate(data['texts']):
            scopes,conflicts=scopes_for(qi,atomic,previous,fixed);row=record['diagnostics'][qi]
            for mixed,matched in zip(scopes[HYBRIDS[2]],previous['matched_positive_scope'][qi]):
                for key in ['target','positive','negative']:assert torch.equal(mixed[key],matched[key])
            assert row['native_scope_exact'] and row['native_prior_max_error']<=2e-6
            assert conflicts==row['conflicts']
            for vi,ss in enumerate(scopes[HYBRIDS[1]]):
                ref=fixed['views'][vi]
                if not bool(ref['confidence'][qi]>0 and ref['positive'][:,qi].any()):
                    raw_permission['inactive_reference_views']+=1
                    if bool(ss['confidence']>0 and ss['positive'].any()):
                        raw_permission['newly_active_views']+=1
                        raw_permission['newly_active_negative_rays']+=int(ss['negative'].sum())
            for arm in NEW_ARMS:
                ss=scopes[arm];pr=prepared[qi][arm];report=row['arms'][arm]
                active=[vi for vi,s in enumerate(ss) if s['confidence']>0 and s['positive'].any()]
                assert active==report['active_views']
                expected=torch.zeros(len(ss),dtype=torch.float64)
                for vi in active:expected[vi]=ss[vi]['confidence'].double()
                assert torch.equal(expected,pr['confidence'])
                for view,s in zip(field.extent_preparation.views,ss):
                    assert not (s['positive']&s['negative']).any()
                    assert not ((s['positive']|s['negative'])&~view.valid).any()
                assert not pr['cost'][~data['observed']].any()
                counts[arm]['active_queries']+=bool(active);counts[arm]['active_views']+=len(active)
                if arm in HYBRIDS:counts[arm]['conflicts']+=sum(conflicts[arm])
                if not active:assert torch.equal(data['membership_by_control'][arm][:,qi],pr['initial'])
                else:
                    assert np.all(np.diff(report['energy_trace'])<=1e-8)
                    assert abs(float(pr['cost'].double()@pr['initial'].double())-1)<2e-5
                    if arm not in probes:
                        init=objective(field,(pr['initial']>.5).numpy(),ss,pr)
                        final=objective(field,data['selected_by_control'][arm][:,qi].numpy(),ss,pr)
                        assert abs(init['total']-report['energy_trace'][0])<1e-7,(scene,arm,'initial',init,report['energy_trace'][0])
                        assert abs(final['total']-report['energy_trace'][-1])<1e-7,(scene,arm,'final',final,report['energy_trace'][-1])
                        probes[arm]=dict(query=text,initial=init,final=final,preparation=replay_preparation(field,prior[:,qi],ss,pr))
        for arm in ARMS:
            assert torch.equal(data['selected_by_control'][arm],data['membership_by_control'][arm]>.5)
            assert not data['selected_by_control'][arm][~data['observed']].any()
        del field,previous,fixed,atomic,prepared,prior;gc.collect()
        metrics={};evaluation=read(out/'evaluation/evaluation.json')
        baselines=['seed_addition','native_complete_scope'];baseline_masks={b:{} for b in baselines}
        changes={a:{b:{k:0 for k in ['added_tp','added_fp','removed_tp','removed_fp','beneficial_cases','harmful_cases']}
                    for b in baselines} for a in NEW_ARMS}
        for arm in ARMS:
            ious=[];ps=[];rs=[]
            for item in evaluation['metrics'][arm]['selected']['objects']:
                assert not item['missing'];gt=load_binary_mask(Path(item['gt_path']),threshold=10)
                pred=load_binary_mask(Path(item['pred_path']),threshold=10,grayscale=True)
                tp=int((gt&pred).sum());iou=tp/max(int((gt|pred).sum()),1)
                assert abs(iou-item['iou'])<1e-12
                ious.append(iou);ps.append(tp/max(int(pred.sum()),1));rs.append(tp/max(int(gt.sum()),1))
                key=(item['frame'],item['query'])
                if arm in baselines:baseline_masks[arm][key]=(pred,iou,item['gt_path'])
                if arm in NEW_ARMS:
                    for base in baselines:
                        before,old_iou,gt_path=baseline_masks[base][key];assert gt_path==item['gt_path']
                        added=pred&~before;removed=before&~pred;change=changes[arm][base]
                        change['added_tp']+=int((added & gt).sum());change['added_fp']+=int((added & ~gt).sum())
                        change['removed_tp']+=int((removed & gt).sum());change['removed_fp']+=int((removed & ~gt).sum())
                        change['beneficial_cases']+=int(iou>old_iou);change['harmful_cases']+=int(iou<old_iou)
            metrics[arm]=dict(iou=float(np.mean(ious)),precision=float(np.mean(ps)),recall=float(np.mean(rs)))
            assert abs(metrics[arm]['iou']-summary['per_scene'][arm][scene])<1e-12
        scenes[scene]=dict(metrics=metrics,probes=probes,counts=counts,source_pseudo_agreement=source_agreement,
                           projected_pixel_changes=changes,
                           raw_reference_negative_permission_coupling=raw_permission,
                           inherited_memberships_exact=True,atomic_scope_queries_replayed=len(data['texts']))
        print('audited',scene,{a:metrics[a] for a in NEW_ARMS},flush=True);del data;gc.collect()
    macro={a:{k:sum(s['metrics'][a][k] for s in scenes.values())/4 for k in ['iou','precision','recall']} for a in ARMS}
    comparisons={}
    pairs=[('standard_to_atomic','seed_addition',CANDIDATE),('native_to_atomic','native_complete_scope',CANDIDATE),
        ('positive_at_native_negative','native_complete_scope',HYBRIDS[0]),('negative_at_native_positive','native_complete_scope',HYBRIDS[1]),
        ('positive_at_reference_negative',HYBRIDS[1],HYBRIDS[2]),('negative_at_matched_positive',HYBRIDS[0],HYBRIDS[2]),
        ('confidence_at_matched_scope',HYBRIDS[2],'matched_positive_scope')]
    for name,before,after in pairs:
        comparisons[name]=paired({s:v['metrics'][before]['iou'] for s,v in scenes.items()},
                                  {s:v['metrics'][after]['iou'] for s,v in scenes.items()},'actual selected-set IoU')
    result=dict(passed=True,macro=macro,comparisons=comparisons,scenes=scenes,design=receipt(ROOT/'design.json'),code=receipt(__file__),
        all_atomic_scopes_independently_replayed=True,all_actual_mask_ious_replayed=True,
        candidate_source_free=True,hybrids_privileged=HYBRIDS,default_promoted=False,whole_five_task=False)
    result['raw_negative_intervention_is_not_a_validated_permission_transfer']=True
    write(ROOT/'audit.json',result);write('docs/experiments/2026-10-12-signed-scope-audit.json',result)
    print('PASS',macro,flush=True)


if __name__=='__main__':torch.set_num_threads(4);main()
