"""Post-failure diagnostic; preserve the full replay's strict equality failure.

Report exact sets and continuous values separately without changing predictions,
relaxing the frozen gate, reading new target labels or accepting a tolerance.
"""
from pathlib import Path
import torch
from scripts.research.validate_query_field_extent_runtime_full import ROOT,design
from scripts.research.run_query_field_relations import read,write,receipt,check


def main():
    d=design();seal=read(ROOT/'predictions_complete.json')
    for r in seal['records']:check(r)
    rows=[]
    for j in d['jobs']:
        record=read(ROOT/'predictions'/j['scene']/'result.json')
        check(record['output']);check(j['reference'])
        actual=torch.load(record['output']['path'],weights_only=True)
        old=torch.load(j['reference']['path'],weights_only=True)
        target=old['membership_by_control']['native_atomic_abstention']
        assert torch.equal(actual['selected'],actual['membership']>.5)
        assert torch.equal(old['selected_by_control']['native_atomic_abstention'],target>.5)
        delta=(actual['membership']-target).abs();different=delta!=0
        queries=[]
        for qi in different.any(0).nonzero().flatten().tolist():
            queries.append(dict(index=qi,text=j['texts'][qi],
                fallback=record['diagnostics'][qi]['fallback'],
                differing_values=int(different[:,qi].sum()),
                max_absolute_difference=float(delta[:,qi].max())))
        rows.append(dict(scene=j['scene'],queries=record['queries'],
            member_values_exact=not bool(different.any()),
            selected_bit_differences=int((actual['selected']!=(target>.5)).sum()),
            differing_values=int(different.sum()),max_absolute_difference=float(delta.max()),
            differing_queries=queries,record=receipt(ROOT/'predictions'/j['scene']/'result.json')))
        del actual,old,target,delta,different
    assert sum(r['queries'] for r in rows)==68
    result=dict(diagnostic_complete=True,
        frozen_strict_gate_passed=all(r['member_values_exact'] and not r['selected_bit_differences'] for r in rows),
        all_actual_sets_exact=all(r['selected_bit_differences']==0 for r in rows),
        queries=68,scenes=rows,design=receipt(ROOT/'design.json'),
        prediction_seal=receipt(ROOT/'predictions_complete.json'),code=receipt(__file__),
        failed_run_log=receipt('/tmp/radio_gs_text_extent_public_full_run.log'),
        posthoc_diagnostic=True,gate_changed=False,tolerance_accepted=False,
        predictions_changed=False,new_target_scores=False,
        numerical_origin='continuous fallback native prior; exact floating-point cause not established',
        raw_encoder_audit=receipt('docs/experiments/2026-10-12-text-extent-encoder-audit.json'))
    write(ROOT/'post_failure_diagnostic.json',result)
    write('docs/experiments/2026-10-12-text-extent-public-full-diagnostic.json',result)
    print(result,flush=True)


if __name__=='__main__':torch.set_num_threads(4);main()
