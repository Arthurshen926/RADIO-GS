"""Independent float64 spatial witnesses, scope permissions and actual outputs."""
import gc
from pathlib import Path
import numpy as np
from scipy import sparse
import torch
from radio_gs.query_field.storage import load_field
from radio_gs.scripts.eval_opengaussian_lerf_baseline import load_binary_mask
from scripts.research.run_query_field_scope_completion import ROOT,PARENT,CANONICAL,BASE,CANDIDATE,ARMS,OLD_ARMS,NEW_ARMS,design
from scripts.research.audit_query_field_signed_scope import objective,replay_preparation
from scripts.research.run_query_field_relations import read,write,receipt,check
from scripts.research.summarize_query_field_relations import paired


def enabled(s):
    return bool(float(s['confidence'])>0 and s['positive'].numpy().any())


def qualified_negative(s):
    n=s['negative'].numpy()
    return n.copy() if enabled(s) else np.zeros_like(n)


def reconstruct(ops,valid,atomic,native):
    masses=[];active=[enabled(s) for s in atomic]
    for op,s,on in zip(ops,atomic,active):
        labels=np.column_stack([s['positive'].numpy(),s['negative'].numpy()]).astype(np.float64)
        masses.append(op.T@(labels*(float(s['confidence']) if on else 0.)))
    scopes={a:[] for a in NEW_ARMS};diagnostics=[];conflicts={a:[] for a in NEW_ARMS[:2]}
    for vi,(op,base,ns) in enumerate(zip(ops,atomic,native)):
        for arm,p,n in [(NEW_ARMS[0],base,ns),(NEW_ARMS[1],ns,base)]:
            neg=qualified_negative(n);conflict=neg&(p['target'].numpy()>.5)
            scopes[arm].append(dict(p,negative=torch.from_numpy(neg&~conflict),confidence=ns['confidence']))
            conflicts[arm].append(int(conflict.sum()))
        donor=np.zeros_like(masses[vi])
        for other in range(len(ops)):
            if other!=vi:donor+=masses[other]
        witness=(donor[:,0]-donor[:,1])>1e-8
        projected=op@np.column_stack([witness,np.ones(len(witness))])
        supported=projected[:,0]>.5*projected[:,1]+1e-8
        original_n=qualified_negative(base);p=base['positive'].numpy()
        added=~(p|original_n)&ns['positive'].numpy()&supported&valid[vi]
        y=base['target'].numpy().copy();y[added]=1
        s=dict(target=torch.from_numpy(y),positive=torch.from_numpy(p|added),
               negative=torch.from_numpy(original_n&~(y>.5)),confidence=base['confidence'])
        scopes[CANDIDATE].append(s)
        diagnostics.append(dict(added_positive_rays=int(added.sum()),donor_views=sum(active)-int(active[vi]),
            donor_positive_elements=int(witness.sum()),witness_rays=int((supported&valid[vi]).sum()),
            originally_active=active[vi],newly_active=bool(not active[vi] and enabled(s))))
    return scopes,diagnostics,conflicts


def main():
    design();cfg=read(ROOT/'prediction_config.json');summary=read(ROOT/'summary.json')
    parent_audit=read(PARENT/'audit.json');assert parent_audit['passed']
    for r in read(ROOT/'predictions_complete.json')['records']:check(r)
    scenes={}
    for j in cfg['jobs']:
        scene=j['scene'];out=ROOT/'predictions'/j['name'];record=read(out/'prediction.json')
        assert record['all_new_arms_source_free'] and not record['target_labels_opened']
        for k in ['prepared','scopes','parent_predictions','parent_scopes','parent_atomic','field_manifest']:check(record[k])
        check(dict(path=str(out/'predictions.pt'),sha256=record['predictions_sha256']))
        data=torch.load(out/'predictions.pt',weights_only=True);old=torch.load(record['parent_predictions']['path'],weights_only=True)
        for arm in OLD_ARMS:
            assert torch.equal(data['membership_by_control'][arm],old['membership_by_control'][arm])
            assert torch.equal(data['selected_by_control'][arm],old['selected_by_control'][arm])
        del old;gc.collect()
        previous=torch.load(record['parent_scopes']['path'],weights_only=True)['native_complete_scope']
        atomic=torch.load(record['parent_atomic']['path'],weights_only=True)
        actual_scopes=torch.load(record['scopes']['path'],weights_only=True)
        prepared=torch.load(record['prepared']['path'],weights_only=True)
        prior=torch.load(CANONICAL/'inputs'/scene/'response.pt',weights_only=True)['semantic_probability']
        field,_=load_field(j['field']);ops=[];valid=[]
        for k,v in data['geometry'].items():assert torch.equal(v,getattr(field,k))
        for v in field.extent_preparation.views:
            ops.append(sparse.coo_matrix((v.weight.double().numpy(),v.index.numpy()),shape=(len(v.valid),len(field.xyz))).tocsr())
            valid.append(v.valid.numpy())
        counts={a:dict(active_queries=0,active_views=0,conflicts=0) for a in NEW_ARMS};probes={}
        completion=dict(added_positive_rays=0,newly_active_views=0,changed_queries=0)
        for qi,row in enumerate(record['diagnostics']):
            ss,info,conflicts=reconstruct(ops,valid,atomic[qi],previous[qi])
            assert info==row['completion'],(scene,qi,'completion diagnostics')
            assert conflicts==row['conflicts']
            assert row['parent_scopes_exact'] and row['native_prior_max_error']<=2e-6
            added=sum(x['added_positive_rays'] for x in info);completion['added_positive_rays']+=added
            completion['newly_active_views']+=sum(x['newly_active'] for x in info)
            completion['changed_queries']+=added>0
            for arm in NEW_ARMS:
                for vi,(s,actual) in enumerate(zip(ss[arm],actual_scopes[qi][arm])):
                    for key in s:assert torch.equal(s[key],actual[key]),(scene,qi,arm,vi,key)
                    assert not (s['positive']&s['negative']).any()
                    assert not ((s['positive']|s['negative'])&~torch.from_numpy(valid[vi])).any()
                pr=prepared[qi][arm];report=row['arms'][arm]
                active=[vi for vi,s in enumerate(ss[arm]) if enabled(s)]
                assert active==report['active_views']
                c=torch.tensor([float(s['confidence']) if enabled(s) else 0. for s in ss[arm]],dtype=torch.float64)
                assert torch.equal(c,pr['confidence'])
                counts[arm]['active_queries']+=bool(active);counts[arm]['active_views']+=len(active)
                counts[arm]['conflicts']+=sum(conflicts.get(arm,[]))
                if not active:assert torch.equal(data['membership_by_control'][arm][:,qi],pr['initial'])
                else:
                    assert np.all(np.diff(report['energy_trace'])<=1e-8)
                    if arm not in probes:
                        initial=objective(field,(pr['initial']>.5).numpy(),ss[arm],pr)
                        final=objective(field,data['selected_by_control'][arm][:,qi].numpy(),ss[arm],pr)
                        assert abs(initial['total']-report['energy_trace'][0])<1e-7
                        assert abs(final['total']-report['energy_trace'][-1])<1e-7
                        probes[arm]=dict(query=data['texts'][qi],initial=initial,final=final,
                            preparation=replay_preparation(field,prior[:,qi],ss[arm],pr))
        del field,ops,previous,atomic,actual_scopes,prepared,prior;gc.collect()
        metrics={};evaluation=read(out/'evaluation/evaluation.json');baseline_masks={b:{} for b in OLD_ARMS}
        changes={a:{b:{k:0 for k in ['added_tp','added_fp','removed_tp','removed_fp','beneficial_cases','harmful_cases']}
                    for b in OLD_ARMS} for a in NEW_ARMS}
        for arm in ARMS:
            assert torch.equal(data['selected_by_control'][arm],data['membership_by_control'][arm]>.5)
            assert not data['selected_by_control'][arm][~data['observed']].any()
            ious=[];ps=[];rs=[]
            for item in evaluation['metrics'][arm]['selected']['objects']:
                assert not item['missing'];gt=load_binary_mask(Path(item['gt_path']),threshold=10)
                pred=load_binary_mask(Path(item['pred_path']),threshold=10,grayscale=True)
                tp=int((gt&pred).sum());iou=tp/max(int((gt|pred).sum()),1);assert abs(iou-item['iou'])<1e-12
                ious.append(iou);ps.append(tp/max(int(pred.sum()),1));rs.append(tp/max(int(gt.sum()),1))
                key=(item['frame'],item['query'])
                if arm in OLD_ARMS:baseline_masks[arm][key]=(pred,iou,item['gt_path'])
                else:
                    for b in OLD_ARMS:
                        before,old_iou,gt_path=baseline_masks[b][key];assert gt_path==item['gt_path']
                        added=pred&~before;removed=before&~pred;change=changes[arm][b]
                        for k,v in dict(added_tp=(added&gt).sum(),added_fp=(added&~gt).sum(),
                            removed_tp=(removed&gt).sum(),removed_fp=(removed&~gt).sum(),
                            beneficial_cases=iou>old_iou,harmful_cases=iou<old_iou).items():change[k]+=int(v)
            metrics[arm]=dict(iou=float(np.mean(ious)),precision=float(np.mean(ps)),recall=float(np.mean(rs)))
            assert abs(metrics[arm]['iou']-summary['per_scene'][arm][scene])<1e-12
            if arm in OLD_ARMS:assert metrics[arm]==parent_audit['scenes'][scene]['metrics'][arm]
        scenes[scene]=dict(metrics=metrics,counts=counts,completion=completion,probes=probes,
            projected_pixel_changes=changes,all_scopes_independently_replayed=True,inherited_memberships_exact=True)
        print('audited',scene,{a:metrics[a] for a in NEW_ARMS},flush=True);del data;gc.collect()
    macro={a:{k:sum(s['metrics'][a][k] for s in scenes.values())/4 for k in ['iou','precision','recall']} for a in ARMS}
    values=lambda a:{s:v['metrics'][a]['iou'] for s,v in scenes.items()}
    pairs=[('completion_vs_atomic',BASE,CANDIDATE),('completion_vs_seed','seed_addition',CANDIDATE),
        ('positive_at_native_negative','native_complete_scope',NEW_ARMS[0]),
        ('negative_at_native_positive','native_complete_scope',NEW_ARMS[1]),
        ('negative_at_atomic_positive',NEW_ARMS[0],BASE),('positive_at_atomic_negative',NEW_ARMS[1],BASE)]
    comparisons={name:paired(values(before),values(after),'actual selected-set IoU') for name,before,after in pairs}
    interaction={s:values(BASE)[s]-values(NEW_ARMS[0])[s]-values(NEW_ARMS[1])[s]+values('native_complete_scope')[s] for s in scenes}
    result=dict(passed=True,macro=macro,comparisons=comparisons,scenes=scenes,factorial_interaction=interaction,
        mean_factorial_interaction=float(np.mean(list(interaction.values()))),design=receipt(ROOT/'design.json'),
        code=receipt(__file__),objective_helper=receipt('scripts/research/audit_query_field_signed_scope.py'),
        all_scopes_independently_replayed=True,all_actual_mask_ious_replayed=True,
        all_new_arms_source_free=True,default_promoted=False,whole_five_task=False)
    write(ROOT/'audit.json',result);write('docs/experiments/2026-10-12-scope-completion-audit.json',result)
    print('PASS',macro,comparisons,flush=True)


if __name__=='__main__':torch.set_num_threads(4);main()
