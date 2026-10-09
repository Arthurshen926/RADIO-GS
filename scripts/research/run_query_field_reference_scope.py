"""Separate input expansion from reference-conditioned field composition, all8."""
import argparse
import copy
import subprocess
from pathlib import Path
import torch
from scripts.research.run_query_field_relations import read,write,receipt,check,execute,lanes
ROOT=Path('output/query_field_20261009/reference_scope_v1')
PREVIOUS=Path('output/query_field_20261009/signed_competition_v1')
ARMS=['raw_signed','raw_positive','mask_rays','mask_composition']
PROTOCOL=Path('docs/experiments/2026-10-09-reference-scope-protocol.md')


def prepare():
    ROOT.mkdir(parents=True,exist_ok=False);jobs=[]
    for old in read(PREVIOUS/'design.json')['jobs']:
        check(old['configuration']);cfg=copy.deepcopy(read(old['configuration']['path']))
        base=next(j for j in cfg['jobs'] if j['query_program']=='positive_ray_expected_competitive')
        qp=Path('configs/consolidated/nvos')/(old['name']+'.query.json');q=read(qp)
        rp=Path(q['reference_mask']).parent/'receipt.json';r=read(rp)
        assert q['reference_mask_is_gt'] is False and r['reference_mask_is_gt'] is False
        assert r['negative_scribble_opened'] is False and r['target_reads_forbidden'] is True
        assert receipt(base['positive'])['sha256']==r['official_scribbles']['positive']['sha256']
        assert receipt(q['reference_mask'])['sha256']==r['mask_sha256']
        assert q['prompt_raster_wh']==r['prompt_raster_wh']
        assert q['reference_rgb']==r['reference_rgb']['path']
        fixture=torch.load(base['camera_fixture'],weights_only=True)
        assert fixture['reference']['frame']==q['reference_frame']
        assert [fixture['reference']['width'],fixture['reference']['height']]==q['prompt_raster_wh']
        cfg['jobs']=[]
        for arm in ARMS:
            job=dict(base,name=old['name']+'__'+arm,mode='signed' if arm=='raw_signed' else 'positive')
            if arm.startswith('mask_'):
                job.update(prompt_input='reference_mask',reference_mask=q['reference_mask'],reference_frame=q['reference_frame'],reference_raster_wh=q['prompt_raster_wh'])
            if arm=='mask_composition':job['query_program']='reference_scope'
            cfg['jobs'].append(job)
        cfg['prediction_policy']='separate positive-only retained SAM query-mask input from field region composition; no source tracks or solver'
        path=ROOT/'configs'/(old['name']+'.json');write(path,cfg)
        check(old['scoring']);scoring=read(old['scoring']['path']);scoring['prediction_config']=str(path)
        sp=ROOT/'configs'/(old['name']+'_score.json');write(sp,scoring)
        jobs.append(dict(old,configuration=receipt(path),scoring=receipt(sp),inputs=old['inputs']+[receipt(qp),receipt(rp),receipt(q['reference_mask'])]))
    assert len(jobs)==8
    helpers=['run_query_field_relations.py','summarize_query_field_signed_regions.py','summarize_query_field_relations.py','run_query_field_signed_regions.py']
    write(ROOT/'design.json',dict(jobs=jobs,arms=ARMS,primary='mask_composition',protocol=receipt(PROTOCOL),
        code=[receipt(p) for p in sorted(Path('radio_gs/query_field').glob('*.py'))]+[receipt(__file__)]+[receipt(Path('scripts/research')/p) for p in helpers],
        implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),training_steps=0,new_state_bytes=0,old_source_frames=120,whole_five_task=False,
        input_conditions=dict(raw_signed='official raw signed strokes',raw_positive='official positive stroke only',mask_rays='retained v10 positive-derived SAM reference mask; inherited generating RGB/SAM cost',mask_composition='same retained query mask; field-owned region composition instead of ray kernels'),reference_mask_generation_repeated=False))


def design():
    d=read(ROOT/'design.json')
    for r in d['code']+[d['protocol']]:check(r)
    return d


def run(gpus):
    d=design()
    def predict(j,g):
        for r in [j['configuration'],j['field'],j['manifest']]+j['inputs']:check(r)
        for job in read(j['configuration']['path'])['jobs']:
            execute(['-m','radio_gs.query_field.public_nvos','predict','--config',j['configuration']['path'],'--job',job['name'],'--output',str(ROOT/'predictions'/j['name'])],ROOT/'logs'/(job['name']+'.log'),g)
        print('predicted',j['name'],flush=True)
    lanes(d['jobs'],gpus,predict)
    records=[];checks=[]
    for j in d['jobs']:
        cfg=read(j['configuration']['path'])
        for job in cfg['jobs']:
            root=ROOT/'predictions'/j['name']/job['name'];r=read(root/'prediction.json')
            assert r['configuration']==cfg and r['job']==job and not r['target_labels_opened']
            check(dict(path=str(root/'predictions.pt'),sha256=r['predictions_sha256']));records.append(receipt(root/'prediction.json'))
        old=torch.load(PREVIOUS/'predictions'/j['name']/(j['name']+'__positive_ray_expected_competitive')/'predictions.pt',map_location='cpu',weights_only=True)
        new=torch.load(ROOT/'predictions'/j['name']/(j['name']+'__raw_signed')/'predictions.pt',map_location='cpu',weights_only=True)
        assert torch.equal(old['selected'],new['selected'])
        errors={k:float((old[k]-new[k]).abs().max()) for k in ['membership','response','retained_response']}
        assert max(errors.values())<=1e-6
        checks.append(dict(case=j['name'],selected_exact=True,max_errors=errors))
    assert len(records)==32;write(ROOT/'predictions_complete.json',dict(records=records,baseline=checks))
    from radio_gs.query_field.public_nvos import score
    for j in d['jobs']:
        check(j['scoring']);score(read(j['scoring']['path']),ROOT/'predictions'/j['name']);print('scored',j['name'],flush=True)
    write(ROOT/'scoring_complete.json',dict(records=[receipt(ROOT/'predictions'/j['name']/'evaluation.json') for j in d['jobs']]))
    from scripts.research.summarize_query_field_signed_regions import main
    main(ROOT,'reference-scope')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);p.add_argument('--gpus',default='2,3');a=p.parse_args();torch.set_num_threads(4)
    prepare() if a.stage=='prepare' else run(list(map(int,a.gpus.split(','))))
