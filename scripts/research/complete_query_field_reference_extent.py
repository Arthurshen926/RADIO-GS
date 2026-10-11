"""Finish sealed predictions after an exclusive-write orchestration failure.

The frozen original runner tried to create baseline_checks.json inside a loop.
No prediction or scoring changed; keep that partial record and validate all
eight controls before the original planned scoring. Do not rerun predictions.
"""
import torch
from scripts.research.run_query_field_reference_extent import ROOT,PREVIOUS,design
from scripts.research.run_query_field_relations import read,write,receipt,check


def main():
    d=design();records=[];baseline=[]
    write(ROOT/'completion_design.json',dict(code=receipt(__file__),frozen_design=receipt(ROOT/'design.json'),
        reason='exclusive baseline_checks.json creation inside original loop; no prediction change',
        original_failure_log=receipt(ROOT/'orchestration_failure.log'),
        partial_checks=receipt(ROOT/'baseline_checks.json'),new_target_scores_exist=False))
    assert not (ROOT/'scoring_complete.json').exists()
    for j in d['jobs']:
        cfg=read(j['configuration']['path'])
        assert not (ROOT/'predictions'/j['name']/'evaluation.json').exists()
        for job in cfg['jobs']:
            p=ROOT/'predictions'/j['name']/job['name'];r=read(p/'prediction.json')
            assert r['configuration']==cfg and r['job']==job and not r['target_labels_opened']
            check(dict(path=str(p/'predictions.pt'),sha256=r['predictions_sha256']))
            if job['query_program']!='reference_scope_admitted':check(r['extent_state'])
            records.append(receipt(p/'prediction.json'))
        old=torch.load(PREVIOUS/'predictions'/j['name']/(j['name']+'__admission')/'predictions.pt',weights_only=True)
        new=torch.load(ROOT/'predictions'/j['name']/(j['name']+'__composed')/'predictions.pt',weights_only=True)
        errors={k:float((old[k]-new[k]).abs().max()) for k in ['membership','response','retained_response']}
        exact=torch.equal(old['selected'],new['selected'])
        row=dict(case=j['name'],selected_exact=exact,max_errors=errors);baseline.append(row)
        assert exact and max(errors.values())<=1e-6,(j['name'],row)
    assert read(ROOT/'baseline_checks.json')['records']==baseline[:1]
    assert len(records)==32
    write(ROOT/'baseline_checks_complete.json',dict(records=baseline))
    write(ROOT/'predictions_complete.json',dict(records=records,baseline=baseline,completion=receipt(ROOT/'completion_design.json')))
    from radio_gs.query_field.public_nvos import score
    for j in d['jobs']:
        check(j['scoring']);score(read(j['scoring']['path']),ROOT/'predictions'/j['name']);print('scored',j['name'],flush=True)
    write(ROOT/'scoring_complete.json',dict(records=[receipt(ROOT/'predictions'/j['name']/'evaluation.json') for j in d['jobs']]))
    from scripts.research.summarize_query_field_signed_regions import main as summarize
    summarize(ROOT,'reference-extent',document_date='2026-10-12')


if __name__=='__main__':torch.set_num_threads(4);main()
