"""Post-score all-scene comparisons, projection precision/recall and bindings."""
from itertools import combinations
from pathlib import Path
import numpy as np
import torch
from radio_gs.scripts.eval_opengaussian_lerf_baseline import load_binary_mask
from scripts.research.run_query_field_native_region import ROOT,SOURCE,BASE,ARMS,read,write,receipt,check,design
from scripts.research.summarize_query_field_relations import paired


def main():
    d=design(verify_inputs=False);barrier=read(ROOT/'predictions_complete.json');scored=read(ROOT/'scoring_complete.json')
    for r in barrier['records']+scored['records']:check(r)
    cfg=read(d['configuration']['path']);result=read(ROOT/'summary.json');per=result['per_scene'];details={};labels={};monotonicity={}
    for j in cfg['jobs']:
        p=ROOT/'predictions'/j['name'];record=read(p/'prediction.json')
        check(dict(path=str(p/'predictions.pt'),sha256=record['predictions_sha256']))
        source=read(SOURCE/'predictions'/j['name']/'prediction.json');manifest=read(Path(j['field'])/'manifest.json')
        assert source['field']['field.pt']['sha256']==manifest['provenance']['parent_field_sha256']
        data=torch.load(p/'predictions.pt',map_location='cpu',weights_only=True)
        old=torch.load(BASE/'predictions'/j['name']/'predictions.pt',map_location='cpu',weights_only=True)
        assert torch.equal(data['scores']['admission'],old['scores']['admission_preserving'])
        assert torch.equal(data['selected_by_control']['admission'],old['selected'])
        assert not record['target_labels_opened'] and not record['runtime_source_access']
        for acceptor in ['pooled','native']:
            plain=data['scores'][acceptor+'_parts'];protected=data['scores'][acceptor+'_protected']
            delta=protected-plain
            monotonicity[j['scene']+'_'+acceptor]=dict(maximum_decrease=float((-delta).clamp_min(0).max()),
                decreased_entries=int((delta<0).sum()),exact_gate_passed=bool((delta>=0).all()),
                protection_lost_selections=int(((plain>.5)&(protected<=.5)).sum()))
            assert bool((delta>=0).all())
            for query in record['diagnostics']:
                left=query['controls'][acceptor+'_parts'];right=query['controls'][acceptor+'_protected']
                for key in ['positive_mass','negative_mass_before','admitted_regions','confidence_sum']:
                    assert left[key]==right[key]
        expected=read(j['queries'])['queries'];assert data['texts']==[x['text'] for x in expected]
        e=read(p/'evaluation/evaluation.json');baseline=read(BASE/'predictions'/j['name']/'evaluation/evaluation.json')
        assert e['evaluation_scene_sha256']==baseline['evaluation_scene_sha256']
        changes={}
        for arm in ARMS:
            metrics=e['metrics'][arm];precision=[];recall=[]
            for item in metrics['selected']['objects']:
                assert not item['missing']
                target=load_binary_mask(Path(item['gt_path']),threshold=10)
                pred=load_binary_mask(Path(item['pred_path']),threshold=10,grayscale=True)
                tp=int((target & pred).sum());union=int((target | pred).sum())
                assert abs(tp/max(union,1)-item['iou'])<1e-12
                precision.append(tp/max(int(pred.sum()),1));recall.append(tp/max(int(target.sum()),1))
                if item['gt_path'] not in labels:labels[item['gt_path']]=receipt(item['gt_path'])
            per[arm][j['scene']].update(precision=float(np.mean(precision)),recall=float(np.mean(recall)))
            before=data['selected_by_control']['admission'];after=data['selected_by_control'][arm]
            changes[arm]=dict(new_member_query_pairs=int((after & ~before).sum()),removed_member_query_pairs=int((before & ~after).sum()))
        details[j['scene']]=dict(queries=len(expected),changes=changes,diagnostics=record['diagnostics'])
    macro={a:{k:sum(x[k] for x in per[a].values())/4 for k in next(iter(per[a].values()))} for a in ARMS}
    comparisons={a+'__'+b:{k:paired({s:v[k] for s,v in per[a].items()},{s:v[k] for s,v in per[b].items()},'LERF '+k) for k in macro[a]} for a,b in combinations(ARMS,2)}
    report=dict(passed=all(x['exact_gate_passed'] for x in monotonicity.values()),other_checks_passed=True,
        monotonicity=monotonicity,previous_study_failure_retained=True,
        interpretation='Shared P/N once per acceptor; new exact monotonicity gate. Previous study failure is not overwritten.',
        scenes=4,queries=sum(x['queries'] for x in details.values()),macro=macro,per_scene=per,comparisons=comparisons,
        details=details,gt_files=list(labels.values()),baseline_payload_exact=True,source_field_binding=True,
        design=receipt(ROOT/'design.json'),code=receipt(__file__),post_scoring_only=True,
        inputs_verified_by_driver_and_each_scene_loader=True,source_arrays_rehashed_in_audit=False,
        gt_hashes_current_audit_snapshot=True,whole_five_task=False)
    write(ROOT/'validation/audit.json',report);write('docs/experiments/2026-10-10-native-region-audit.json',report)
    print(macro,flush=True)

if __name__=='__main__':main()
