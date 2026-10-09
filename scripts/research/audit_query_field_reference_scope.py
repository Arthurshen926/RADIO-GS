"""Audit full scope input/decoder controls and retained query-mask provenance."""
from scripts.research.run_query_field_reference_scope import ROOT,PREVIOUS,design,read,write,receipt,check

def main():
    d=design();barrier=read(ROOT/'predictions_complete.json');scored=read(ROOT/'scoring_complete.json')
    assert len(barrier['records'])==32 and len(barrier['baseline'])==8 and len(scored['records'])==8
    for r in barrier['records']+scored['records']:check(r)
    cases={}
    for j in d['jobs']:
        for r in [j['configuration'],j['scoring'],j['field'],j['manifest']]+j['inputs']:check(r)
        result=read(ROOT/'predictions'/j['name']/'evaluation.json')
        assert result['gt_sha256']==read(PREVIOUS/'predictions'/j['name']/'evaluation.json')['gt_sha256']
        check(dict(path=read(j['scoring']['path'])['gt_mask'],sha256=result['gt_sha256']))
        inputs={};timing={};details={}
        for job in read(j['configuration']['path'])['jobs']:
            arm=job['name'].split('__')[-1];path=ROOT/'predictions'/j['name']/job['name'];r=read(path/'prediction.json')
            check(dict(path=str(path/'predictions.pt'),sha256=r['predictions_sha256']))
            assert not r['target_labels_opened'] and not r['reference_rgb_opened'] and not r['runtime_source_access']
            assert r['negative_used_for_membership']==(arm=='raw_signed')
            inputs[arm]=r['reference_mask_receipt'];timing[arm]=r['timing_seconds']
            if arm=='mask_composition':
                x=r['diagnostics'];trace=x['reference_soft_iou_trace']
                assert all(b>a for a,b in zip(trace,trace[1:]))
                assert len(set(x['selected_regions']))==len(trace)
                assert not x['target_camera_used_by_decoder'] and not x['learned_local_kernel_used']
                details=x
        assert inputs['raw_signed'] is None and inputs['raw_positive'] is None
        assert inputs['mask_rays']==inputs['mask_composition']
        cases[j['name']]=dict(same_expanded_input=True,composition=details,timing_seconds=timing)
    report=dict(passed=True,cases=cases,predictions=32,frozen_files=len(d['code'])+1,baseline=barrier['baseline'],
        input_conditions=d['input_conditions'],unchanged_gt=True,barrier=receipt(ROOT/'predictions_complete.json'),
        scoring=receipt(ROOT/'scoring_complete.json'),summary=receipt(ROOT/'summary.json'),
        scope='Provenance and numerical checks; explicit cached query-mask input, no generation parity claim.')
    write(ROOT/'validation/audit.json',report);write('docs/experiments/2026-10-09-reference-scope-audit.json',report)
    print(dict(passed=True,cases=len(cases),predictions=32,frozen_files=report['frozen_files']),flush=True)

if __name__=='__main__':main()
