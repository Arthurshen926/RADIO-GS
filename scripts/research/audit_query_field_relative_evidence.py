"""Audit fixed scopes, inherited controls, normalized objective and real masks."""
from pathlib import Path
import gc
import numpy as np
import torch
from radio_gs.query_field.storage import load_field
from radio_gs.query_field.extent_decision import render_binary,CUTOFF
from radio_gs.scripts.eval_opengaussian_lerf_baseline import load_binary_mask
from scripts.research.run_query_field_relative_evidence import ROOT,PARENT,ARMS,ARM,design
from scripts.research.run_query_field_relations import read,write,receipt,check
from scripts.research.summarize_query_field_relations import paired


def objective(field,selected,scopes,prepared):
    c=prepared['confidence'].double().numpy();result=.1*float(prepared['cost'].double().numpy()@selected)
    data=0.
    for vi,(v,s) in enumerate(zip(field.extent_factors.views,scopes)):
        if c[vi]==0:continue
        alpha=render_binary(v.pointer.numpy(),v.index.numpy(),v.alpha.numpy(),selected).astype(np.float64).clip(1e-6,1-1e-6)
        p=alpha*(1-CUTOFF)/(alpha*(1-CUTOFF)+CUTOFF*(1-alpha));q=p.clip(1e-5,1-1e-5)
        y=s['target'].double().numpy();pos=s['positive'].numpy();neg=s['negative'].numpy();known=pos|neg
        bce=-y*np.log(q)-(1-y)*np.log1p(-q)
        loss=.5*((bce*pos).sum()/max(pos.sum(),1)+(bce*neg).sum()/max(neg.sum(),1))
        loss+=1-(2*(p*y*known).sum()+1)/((p*known).sum()+(y*known).sum()+1)
        data+=c[vi]/c.sum()*loss
    return dict(data=float(data),cost=float(result),total=float(data+result))


def main():
    d=design();cfg=read(ROOT/'prediction_config.json');summary=read(ROOT/'summary.json')
    assert not d['parent_scored_before_design']
    for r in read(ROOT/'predictions_complete.json')['records']:check(r)
    scenes={}
    for j in cfg['jobs']:
        scene=j['scene'];out=ROOT/'predictions'/j['name'];record=read(out/'prediction.json')
        for r in [record['prepared'],record['parent_predictions'],record['parent_scopes'],record['field_manifest']]:check(r)
        check(dict(path=str(out/'predictions.pt'),sha256=record['predictions_sha256']))
        data=torch.load(out/'predictions.pt',weights_only=True);old=torch.load(record['parent_predictions']['path'],weights_only=True)
        prepared=torch.load(record['prepared']['path'],weights_only=True)
        scopes=torch.load(record['parent_scopes']['path'],weights_only=True)['native_complete_scope']
        for a in ARMS[:-1]:
            assert torch.equal(data['membership_by_control'][a],old['membership_by_control'][a])
            assert torch.equal(data['selected_by_control'][a],old['selected_by_control'][a])
        assert not record['target_labels_opened'] and record['new_scope_recomputed_from_field'] and record['scope_exact']
        for qi,r in enumerate(record['diagnostics']):
            p=prepared[qi];active=[vi for vi,s in enumerate(scopes[qi]) if s['confidence']>0 and s['positive'].any()]
            assert active==r['active_views'] and r['scope_exact'] and r['native_prior_max_error']<=2e-6
            expected=torch.zeros(18,dtype=torch.float64)
            for vi in active:expected[vi]=scopes[qi][vi]['confidence'].double()
            assert torch.equal(expected,p['confidence'])
            assert not p['cost'][~data['observed']].any()
            if active:
                maximum=float(expected.max());factor=float((expected[expected>0]/maximum).mean())
                assert maximum==r['confidence_maximum'] and factor==r['cost_factor']
                assert np.max(np.abs(np.array(r['energy_trace'])/factor-r['weighted_mean_energy_trace']))<1e-12
                assert np.all(np.diff(r['weighted_mean_energy_trace'])<=1e-7)
                assert abs(float(p['cost'].double()@p['initial'].double())-1)<2e-5
            else:assert torch.equal(data['membership_by_control'][ARM][:,qi],p['initial'])
        for a in ARMS:
            assert torch.equal(data['selected_by_control'][a],data['membership_by_control'][a]>.5)
            assert not data['selected_by_control'][a][~data['observed']].any()
        field,_=load_field(PARENT/'fields'/scene/'field')
        for k,v in data['geometry'].items():assert torch.equal(v,getattr(field,k))
        probe=next(qi for qi,r in enumerate(record['diagnostics']) if r['active_views'])
        init=objective(field,(prepared[probe]['initial']>.5).numpy(),scopes[probe],prepared[probe])
        final=objective(field,data['selected_by_control'][ARM][:,probe].numpy(),scopes[probe],prepared[probe])
        trace=record['diagnostics'][probe]['weighted_mean_energy_trace']
        assert abs(init['total']-trace[0])<1e-7 and abs(final['total']-trace[-1])<1e-7,(scene,init,final,trace)
        del field,prepared,old,scopes;gc.collect()
        metrics={};evaluation=read(out/'evaluation/evaluation.json')
        for a in ARMS:
            ious=[];ps=[];rs=[]
            for r in evaluation['metrics'][a]['selected']['objects']:
                assert not r['missing'];gt=load_binary_mask(Path(r['gt_path']),threshold=10)
                pred=load_binary_mask(Path(r['pred_path']),threshold=10,grayscale=True)
                tp=int((gt&pred).sum());iou=tp/max(int((gt|pred).sum()),1)
                assert abs(iou-r['iou'])<1e-12
                ious.append(iou);ps.append(tp/max(int(pred.sum()),1));rs.append(tp/max(int(gt.sum()),1))
            metrics[a]=dict(iou=float(np.mean(ious)),precision=float(np.mean(ps)),recall=float(np.mean(rs)))
            assert abs(metrics[a]['iou']-summary['per_scene'][a][scene])<1e-12
        scenes[scene]=dict(metrics=metrics,probe_query=data['texts'][probe],independent_objective=dict(initial=init,final=final),
            inherited_memberships_exact=True,scope_parity_queries=len(data['texts']),new_field_state_bytes=0)
        print('audited',scene,metrics[ARM],flush=True)
        del data;gc.collect()
    macro={a:{k:sum(s['metrics'][a][k] for s in scenes.values())/4 for k in ['iou','precision','recall']} for a in ARMS}
    comparisons={a:paired({s:r['metrics'][a]['iou'] for s,r in scenes.items()},
                          {s:r['metrics'][ARM]['iou'] for s,r in scenes.items()},'actual selected-set IoU') for a in ARMS[:-1]}
    result=dict(passed=True,macro=macro,comparisons=comparisons,scenes=scenes,design=receipt(ROOT/'design.json'),code=receipt(__file__),
        all_mask_ious_replayed=True,scope_exact_all_queries=True,objective_replayed_one_query_per_scene=True,
        new_arm_source_free=True,default_promoted=False,whole_five_task=False)
    write(ROOT/'audit.json',result);write('docs/experiments/2026-10-11-relative-evidence-audit.json',result)
    print(macro,flush=True)


if __name__=='__main__':torch.set_num_threads(4);main()
