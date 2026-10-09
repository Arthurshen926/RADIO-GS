"""Verify scope-study inputs, complete barrier, baseline and score provenance."""
from scripts.research.run_query_field_scope_conditioning import ROOT,PREVIOUS,design,read,write,receipt,check

def main():
    d=design();barrier=read(ROOT/'predictions_complete.json');scored=read(ROOT/'scoring_complete.json')
    assert len(barrier['records'])==24 and len(barrier['baseline'])==8 and len(scored['records'])==8
    for r in barrier['records']+scored['records']:check(r)
    for j in d['jobs']:
        for r in [j['configuration'],j['scoring'],j['field'],j['manifest']]+j['inputs']:check(r)
        result=read(ROOT/'predictions'/j['name']/'evaluation.json')
        assert result['gt_sha256']==read(PREVIOUS/'predictions'/j['name']/'evaluation.json')['gt_sha256']
        check(dict(path=read(j['scoring']['path'])['gt_mask'],sha256=result['gt_sha256']))
        for job in read(j['configuration']['path'])['jobs']:
            path=ROOT/'predictions'/j['name']/job['name'];r=read(path/'prediction.json')
            check(dict(path=str(path/'predictions.pt'),sha256=r['predictions_sha256']))
            assert not r['target_labels_opened'] and not r['reference_rgb_opened'] and not r['runtime_source_access']
            assert r['negative_used_for_membership'] and r['diagnostics']['local_kernel_scale_unchanged']
    report=dict(passed=True,cases=8,predictions=24,frozen_files=len(d['code'])+1,
        baseline=barrier['baseline'],unchanged_gt=True,barrier=receipt(ROOT/'predictions_complete.json'),
        scoring=receipt(ROOT/'scoring_complete.json'),summary=receipt(ROOT/'summary.json'),
        scope='Provenance/numerical audit; independent runtime denial recorded separately.')
    write(ROOT/'validation/audit.json',report);write('docs/experiments/2026-10-09-scope-conditioning-audit.json',report)
    print(report,flush=True)

if __name__=='__main__':main()
