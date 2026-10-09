from scripts.research.run_query_field_reference_visibility import ROOT,PREVIOUS,design,read,write,receipt,check

def main():
    d=design();barrier=read(ROOT/'predictions_complete.json');scored=read(ROOT/'scoring_complete.json')
    assert len(barrier['records'])==16 and len(barrier['baseline'])==8 and len(scored['records'])==8
    for r in barrier['records']+scored['records']:check(r)
    per={}
    for j in d['jobs']:
        for r in [j['configuration'],j['scoring'],j['field'],j['manifest']]+j['inputs']:check(r)
        e=read(ROOT/'predictions'/j['name']/'evaluation.json')
        assert e['gt_sha256']==read(PREVIOUS/'predictions'/j['name']/'evaluation.json')['gt_sha256']
        check(dict(path=read(j['scoring']['path'])['gt_mask'],sha256=e['gt_sha256']))
        inputs=[];details={}
        for job in read(j['configuration']['path'])['jobs']:
            arm=job['name'].split('__')[-1];p=ROOT/'predictions'/j['name']/job['name'];r=read(p/'prediction.json')
            check(dict(path=str(p/'predictions.pt'),sha256=r['predictions_sha256']))
            assert not r['target_labels_opened'] and not r['reference_rgb_opened'] and not r['runtime_source_access']
            assert not r['negative_used_for_membership']
            x=r['diagnostics'];assert x['reference_observation']==('scene' if arm=='visible_reference' else 'subset')
            assert not x['target_camera_used_by_decoder']
            inputs.append(r['reference_mask_receipt']);details[arm]=x
        assert inputs[0]==inputs[1]
        per[j['name']]=details
    report=dict(passed=True,cases=8,predictions=16,frozen_files=len(d['code'])+1,baseline=barrier['baseline'],
        same_query_inputs=True,unchanged_gt=True,diagnostics=per,summary=receipt(ROOT/'summary.json'))
    write(ROOT/'validation/audit.json',report);write('docs/experiments/2026-10-09-reference-visibility-audit.json',report);print(dict(passed=True,cases=8,predictions=16),flush=True)

if __name__=='__main__':main()
