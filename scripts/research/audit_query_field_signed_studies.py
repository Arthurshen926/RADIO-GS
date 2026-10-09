"""Audit all five fixed-cohort studies without modifying scored predictions."""
from pathlib import Path
import torch
from scripts.research.run_query_field_relations import read,write,receipt,check

BASE=Path('output/query_field_20261009');PREVIOUS=BASE/'positive_rays_v1';records=[]
STUDIES=[('extent_stages_v1',56,'local'),('positive_union_v1',24,'positive_ray_local'),('positive_expected_v1',32,'positive_ray_local'),('signed_expected_v1',24,'positive_ray_local'),('signed_competition_v1',32,'positive_ray_local')]
for name,count,arm in STUDIES:
    root=BASE/name;design=read(root/'design.json');barrier=read(root/'predictions_complete.json');scored=read(root/'scoring_complete.json')
    assert len(barrier['records'])==count and len(scored['records'])==8
    for r in barrier['records']:
        check(r);p=read(r['path']);assert p['target_labels_opened'] is False
        check(dict(path=str(Path(r['path']).with_name('predictions.pt')),sha256=p['predictions_sha256']))
    for r in scored['records']:check(r)
    cases=[]
    for j in design['jobs']:
        scene=j['name'];p=root/'predictions'/scene;old=PREVIOUS/'predictions'/scene
        assert read(p/'evaluation.json')['gt_sha256']==read(old/'evaluation.json')['gt_sha256']
        actual=torch.load(p/(scene+'__'+arm)/'predictions.pt',map_location='cpu',weights_only=True)
        expected=torch.load(old/(scene+'__positive_ray_local')/'predictions.pt',map_location='cpu',weights_only=True)
        errors={k:float((actual[k].float()-expected[k].float()).abs().max()) for k in actual}
        exact=torch.equal(actual['selected'],expected['selected']);assert exact and max(errors.values())<=1e-6
        cases.append(dict(case=scene,selected_exact=exact,maximum_errors=errors,gt_sha256_unchanged=True))
    records.append(dict(study=name,predictions=count,scored_cases=8,controls=cases,design=receipt(root/'design.json')))
root=BASE/'signed_competition_v1';inclusions=[]
for j in read(root/'design.json')['jobs']:
    s=j['name'];p=root/'predictions'/s
    product=torch.load(p/(s+'__positive_ray_expected_signed')/'predictions.pt',weights_only=True)
    competition=torch.load(p/(s+'__positive_ray_expected_competitive')/'predictions.pt',weights_only=True)
    violations=int((product['selected']&~competition['selected']).sum())
    difference=float((competition['membership']-product['membership']).min())
    assert violations==0 and difference>=-1e-6
    inclusions.append(dict(case=s,product_selection_outside_competition=violations,minimum_membership_delta=difference))
report=dict(complete=True,studies=5,predictions=sum(r['predictions'] for r in records),unique_requests=8,physical_scenes=7,
    records=records,competitive_superset_checks=inclusions,predictions_modified=False,scoring_modified=False,code=receipt(__file__))
write(root/'validation/series_audit.json',report);write('docs/experiments/2026-10-09-signed-series-audit.json',report)
print('Audited',report['predictions'],'predictions;40 baseline selections exact; signed inclusion8/8',flush=True)
