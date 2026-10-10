"""Validate that the readout intervention changes no selected Gaussian set."""
import torch
from scripts.research.run_query_field_reference_member_readout import ROOT,PREVIOUS,design,read,write,receipt,check


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
        inputs=[];details={};payloads={}
        for job in read(j['configuration']['path'])['jobs']:
            arm=job['name'].split('__')[-1];p=ROOT/'predictions'/j['name']/job['name'];r=read(p/'prediction.json')
            check(dict(path=str(p/'predictions.pt'),sha256=r['predictions_sha256']))
            assert not r['target_labels_opened'] and not r['reference_rgb_opened'] and not r['runtime_source_access']
            assert not r['negative_used_for_membership']
            x=r['diagnostics'];assert x['reference_observation']=='scene' and x['reference_member_conditioning']
            assert x['reference_member_readout']==arm and not x['target_camera_used_by_decoder']
            inputs.append(r['reference_mask_receipt']);details[arm]=x
            payloads[arm]=torch.load(p/'predictions.pt',map_location='cpu',weights_only=True)
        a,b=details['fraction'],details['admission'];assert inputs[0]==inputs[1]
        for key in ['selected_regions','reference_soft_iou_trace','reference_proxy_soft_iou','reference_visible_members','reference_unseen_members','reference_rejected_members','reference_tied_members']:
            assert a[key]==b[key]
        before,after=payloads['fraction'],payloads['admission']
        assert torch.equal(before['selected'],after['selected']) and torch.equal(before['response'],after['response'])
        assert torch.equal(after['membership'],after['selected'].float())
        assert b['reference_unseen_changed']==0
        per[j['name']]=dict(same_actual_set_and_projection=True,binary_membership=True,diagnostics=b)
    report=dict(passed=True,cases=8,predictions=16,frozen_files=len(d['code'])+1,baseline=barrier['baseline'],
        same_query_inputs=True,unchanged_gt=True,details=per,summary=receipt(ROOT/'summary.json'),audit_code=receipt(__file__))
    write(ROOT/'validation/audit.json',report);write('docs/experiments/2026-10-10-reference-member-readout-audit.json',report)
    print(dict(passed=True,cases=8,predictions=16),flush=True)

if __name__=='__main__':main()
