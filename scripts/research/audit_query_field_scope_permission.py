"""Audit original permission, corrected counterfactual and actual masks."""
import gc
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from radio_gs.query_field.storage import load_field
from radio_gs.scripts.eval_opengaussian_lerf_baseline import load_binary_mask
from scripts.research.run_query_field_scope_permission import ROOT,RAW,PARENT,CANONICAL,ARM,BASE,ARMS,design
from scripts.research.audit_query_field_signed_scope import objective
from scripts.research.run_query_field_relations import read,write,receipt,check
from scripts.research.summarize_query_field_relations import paired


def reconstruct(qi,previous,fixed):
    scopes=[];permission=[]
    for vi,native in enumerate(previous[BASE][qi]):
        ref=fixed['views'][vi];active=bool(ref['confidence'][qi]>0 and ref['positive'][:,qi].any())
        n=ref['negative'][:,qi].numpy();y=native['target'].numpy()
        raw=n&~(y>.5);admitted=raw if active else np.zeros_like(raw)
        scopes.append(dict(native,negative=torch.from_numpy(admitted)))
        used=bool(native['confidence']>0 and native['positive'].any())
        permission.append(dict(reference_active=active,native_active=used,
            conflicts=int((n&(y>.5)).sum()) if active else 0,
            removed_unpermitted_rays=int((raw&~admitted).sum()),
            effective_removed_rays=int((raw&~admitted).sum()) if used else 0))
        matched=previous['matched_positive_scope'][qi][vi]
        if bool(native['confidence']>0 and matched['positive'].any()):
            for origin_p,origin_n,origin_c in [(native['positive'],native['negative'],native['confidence']),
                                              (ref['positive'][:,qi],ref['negative'][:,qi],ref['confidence'][qi])]:
                if not bool(origin_c>0 and origin_p.any()):
                    assert not (origin_n & ~(matched['target']>.5)).any()
    return scopes,permission


def preparation_batch(field,prior,all_scopes,prepared):
    count,queries=prior.shape;known=np.zeros((count,queries));positive=np.zeros_like(known);cost=np.zeros_like(known)
    for vi,view in enumerate(field.extent_preparation.views):
        ss=[s[vi] for s in all_scopes];c=np.array([float(s['confidence']) if s['positive'].any() else 0. for s in ss])
        labels=np.stack([(s['positive']|s['negative']).numpy() for s in ss],1)*c[None]
        y=np.stack([s['target'].numpy() for s in ss],1)
        row,col=view.index.numpy();op=sparse.coo_matrix((view.weight.double().numpy(),(row,col)),shape=(len(view.valid),count)).tocsr()
        known+=op.T@labels;positive+=op.T@(labels*y)
        cost+=view.footprint.double().numpy()[:,None]*c[None]
    initial=np.where(known>1e-8,positive/np.maximum(known,1e-8),prior.double().numpy())
    cost/=np.maximum((cost*initial).sum(0),1e-8)[None]
    ei=max(float(np.max(np.abs(initial[:,i]-p['initial'].numpy()))) for i,p in enumerate(prepared))
    ec=max(float(np.max(np.abs(cost[:,i]-p['cost'].numpy())/np.maximum(cost[:,i],1e-8))) for i,p in enumerate(prepared))
    assert ei<=2e-6 and ec<=2e-5,(ei,ec)
    return dict(queries=queries,initial_max_absolute=ei,cost_max_relative=ec)


def main():
    d=design();assert not d['raw_scores_exist_before_design']
    cfg=read(ROOT/'prediction_config.json');summary=read(ROOT/'summary.json');raw_audit=read(RAW/'audit.json');assert raw_audit['passed']
    for r in read(ROOT/'predictions_complete.json')['records']:check(r)
    scenes={}
    for j in cfg['jobs']:
        scene=j['scene'];out=ROOT/'predictions'/j['name'];record=read(out/'prediction.json')
        for k in ['prepared','parent_predictions','parent_scopes','reference_scope','field_manifest']:check(record[k])
        check(dict(path=str(out/'predictions.pt'),sha256=record['predictions_sha256']))
        data=torch.load(out/'predictions.pt',weights_only=True);old=torch.load(record['parent_predictions']['path'],weights_only=True)
        assert torch.equal(data['membership_by_control'][BASE],old['membership_by_control'][BASE]);del old
        previous=torch.load(record['parent_scopes']['path'],weights_only=True);fixed=torch.load(record['reference_scope']['path'],weights_only=True)
        prepared=torch.load(record['prepared']['path'],weights_only=True);prior=torch.load(CANONICAL/'inputs'/scene/'response.pt',weights_only=True)['semantic_probability']
        all_scopes=[];removed=0
        for qi,row in enumerate(record['diagnostics']):
            ss,permission=reconstruct(qi,previous,fixed);all_scopes.append(ss)
            assert permission==row['permission'];removed+=sum(v['effective_removed_rays'] for v in permission)
            assert row['other_factorial_arms_effective_scope_exact'] and row['native_prior_max_error']<=2e-6
            active=[vi for vi,s in enumerate(ss) if s['confidence']>0 and s['positive'].any()]
            assert active==row['active_views'];expected=torch.zeros(len(ss),dtype=torch.float64)
            for vi in active:expected[vi]=ss[vi]['confidence'].double()
            assert torch.equal(expected,prepared[qi]['confidence'])
            if active:assert np.all(np.diff(row['energy_trace'])<=1e-8)
            else:assert torch.equal(data['membership_by_control'][ARM][:,qi],prepared[qi]['initial'])
        assert removed==raw_audit['scenes'][scene]['raw_reference_negative_permission_coupling']['newly_active_negative_rays']
        field,_=load_field(j['field'])
        for k,v in data['geometry'].items():assert torch.equal(v,getattr(field,k))
        prep_report=preparation_batch(field,prior,all_scopes,prepared)
        qi=next(i for i,r in enumerate(record['diagnostics']) if r['active_views']);ss=all_scopes[qi];p=prepared[qi]
        init=objective(field,(p['initial']>.5).numpy(),ss,p);final=objective(field,data['selected_by_control'][ARM][:,qi].numpy(),ss,p)
        trace=record['diagnostics'][qi]['energy_trace'];assert abs(init['total']-trace[0])<1e-7 and abs(final['total']-trace[-1])<1e-7
        del field,previous,fixed,prepared,prior,all_scopes;gc.collect()
        metrics={};evaluation=read(out/'evaluation/evaluation.json')
        for a in ARMS:
            assert torch.equal(data['selected_by_control'][a],data['membership_by_control'][a]>.5)
            assert not data['selected_by_control'][a][~data['observed']].any()
            ious=[];ps=[];rs=[]
            for item in evaluation['metrics'][a]['selected']['objects']:
                assert not item['missing'];gt=load_binary_mask(Path(item['gt_path']),threshold=10);pred=load_binary_mask(Path(item['pred_path']),threshold=10,grayscale=True)
                tp=int((gt&pred).sum());iou=tp/max(int((gt|pred).sum()),1);assert abs(iou-item['iou'])<1e-12
                ious.append(iou);ps.append(tp/max(int(pred.sum()),1));rs.append(tp/max(int(gt.sum()),1))
            metrics[a]=dict(iou=float(np.mean(ious)),precision=float(np.mean(ps)),recall=float(np.mean(rs)))
            assert abs(metrics[a]['iou']-summary['per_scene'][a][scene])<1e-12
        assert metrics[BASE]==raw_audit['scenes'][scene]['metrics'][BASE]
        scenes[scene]=dict(metrics=metrics,permission_rays_removed=removed,all_queries_preparation=prep_report,
            probe_query=data['texts'][qi],independent_objective=dict(initial=init,final=final),other_factorial_arms_effective_scope_exact=True)
        print('audited',scene,metrics[ARM],flush=True);del data;gc.collect()
    macro={a:{k:sum(s['metrics'][a][k] for s in scenes.values())/4 for k in ['iou','precision','recall']} for a in ARMS}
    values={a:{s:v['metrics'][a]['iou'] for s,v in scenes.items()} for a in ARMS}
    raw=lambda a:{s:v['metrics'][a]['iou'] for s,v in raw_audit['scenes'].items()}
    comparisons=dict(negative_at_native_positive=paired(values[BASE],values[ARM],'actual selected-set IoU'),
        positive_at_admitted_reference_negative=paired(values[ARM],raw('matched_positive_reference_negative'),'actual selected-set IoU'),
        permission_correction=paired(raw('native_positive_reference_negative'),values[ARM],'actual selected-set IoU'))
    interaction={s:raw('matched_positive_reference_negative')[s]-raw('matched_positive_native_negative')[s]-values[ARM][s]+values[BASE][s] for s in scenes}
    result=dict(passed=True,macro=macro,scenes=scenes,comparisons=comparisons,factorial_interaction=interaction,
        mean_factorial_interaction=float(np.mean(list(interaction.values()))),design=receipt(ROOT/'design.json'),code=receipt(__file__),
        raw_audit=receipt(RAW/'audit.json'),helper=receipt('scripts/research/audit_query_field_signed_scope.py'),
        all_actual_mask_ious_replayed=True,all_preparations_independently_replayed=True,new_arm_privileged=True,default_promoted=False,whole_five_task=False)
    write(ROOT/'audit.json',result);write('docs/experiments/2026-10-12-scope-permission-audit.json',result);print('PASS',macro,comparisons,flush=True)


if __name__=='__main__':torch.set_num_threads(4);main()
