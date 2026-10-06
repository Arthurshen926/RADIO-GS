"""Pre-register, execute both frozen-map query arms, then score all eight."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from radio_gs.utils.immutable_artifacts import sha256_file

OUT=Path('output/nvos_positive_ensemble_next32')
TASKS=['flower','fortress','fern','horns_center','horns_left','leaves','orchids','trex']
MANIFEST=Path('/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/manifests/nvos_strict_unseen_v1.json')
SOURCES=[__file__,'scripts/audits/build_nvos_positive_ensemble_next32.py',
 'scripts/audits/predict_nvos_positive_ensemble_next32.py',
 'radio_gs/querying/positive_reference_ensemble.py',
 'radio_gs/querying/sam3_reference_completion.py','radio_gs/querying/interactive_2d.py',
 'scripts/audits/nvos_reference_completion_next19_runtime.py',
 'scripts/audits/prompt_transfer_20260919_evaluate.py',
 'radio_gs/scripts/build_sam3_foundation_cache.py',
 'radio_gs/models/element_retrieval_field.py','radio_gs/models/native_query_encoders.py',
 'tests/test_positive_reference_ensemble.py']


def register():
    OUT.mkdir(exist_ok=False)
    parent=Path('output/nvos_positive_queries_20260918')
    checkpoint='checkpoints/sam3_modelscope/sam3.pt'
    r=dict(schema='nvos_positive_ensemble_next32.registration.v1',tasks=TASKS,
        variants=['baseline','ensemble'],baseline='reference_core_component_next19',
        hypothesis='Reference-only positive ensemble may preserve object scope more reliably than one SAM call before query encoding',
        trials=10,points_per_trial=3,multimask=False,
        sampler='existing deterministic_positive_points: splitmix64 salted pixel index,30 distinct pixels,reshape10x3',
        admission='full raw positive integer coverage >= frozen single8point parent mask coverage',
        aggregation='strict majority >0.5 of admitted binary masks; overwrite raw positives True',
        fallback='exact parent mask if no admitted trial; unchanged downstream positive repair',
        no_coverage_guarantee_beyond_raw_positives=True,
        query='fresh same native image encoder on new masked crop; complete existing frozen downstream pipeline',
        threshold=.5,score_resize='cv2.INTER_LINEAR continuous score to GT grid before >=0.5',
        new_field_training=False,geometry_changed=False,negative_scribbles_allowed=False,
        target_rgb_allowed=False,target_gt_read_only_after_all_eight_paired_predictions_persist=True,
        training_feature_reads=False,target_parameter_search=False,
        increased_sam_decoder_budget='ten3positive calls vs parent one8positive multimask call; not equal compute',
        related_prior='old signed/canonical-hash10x3 SAM3 completion existed; this tests positive-only current64+32 field, no first-use claim',
        admission_to_default='requires complete paired macro IoU gain with AP and raw-positive-agreement nondegradation; any regression disclosed; no per-task winner',
        full_method=False,default_promoted=False,parent_query_root=str(parent),
        manifest_sha256=sha256_file(MANIFEST),checkpoint=checkpoint,checkpoint_sha256=sha256_file(checkpoint),
        parent_receipts={task:sha256_file(parent/task/'receipt.json') for task in TASKS},
        implementations={path:sha256_file(path) for path in SOURCES})
    (OUT/'registration.json').write_text(json.dumps(r,indent=2)+'\n')


def score(reg):
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    from radio_gs.training.retrieval_metrics import binary_ranking_metrics
    # Whole cohort must exist and validate before any GT file can be opened.
    reports={}
    for task in TASKS:
        folder=OUT/task;r=json.loads((folder/'prediction_report.json').read_text())
        assert r['registration_sha256']==sha256_file(OUT/'registration.json')
        assert not r['target_gt_opened'] and not r['target_rgb_opened'] and not r['negative_scribble_opened']
        for arm in reg['variants']:
            assert sha256_file(folder/(arm+'.pt'))==r['predictions'][arm]
            assert sha256_file(folder/(arm+'_elements.pt'))==r['element_predictions'][arm]
        reports[task]=r
    manifest=json.loads(MANIFEST.read_text());assert sha256_file(MANIFEST)==reg['manifest_sha256']
    results={};changes={}
    for task in TASKS:
        scene=next(s for s in manifest['scenes'] if s['scene_id']==task)
        target=next(f for f in scene['frames'] if f['frame_id'] in scene['evaluation_frame_ids'])
        assert sha256_file(target['gt_mask_path'])==target['ground_truth_sha256']
        gt=np.asarray(Image.open(target['gt_mask_path']));gt=np.any(gt>127,axis=2) if gt.ndim==3 else gt>127
        truth=torch.from_numpy(gt);results[task]={};rendered={};elements={}
        old=torch.load(Path('output/nvos_reference_completion_next19')/task/'reference_core_component.pt',weights_only=False,map_location='cpu')['response']
        old=torch.from_numpy(cv2.resize(old.numpy(),(gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_LINEAR))
        for arm in reg['variants']:
            payload=torch.load(OUT/task/(arm+'.pt'),weights_only=False,map_location='cpu')
            response=torch.from_numpy(cv2.resize(payload['response'].numpy(),(gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_LINEAR))
            selected=response>=.5;rendered[arm]=selected
            if arm=='baseline':assert torch.equal(selected,old>=.5)
            results[task][arm]=dict(iou=float((selected&truth).sum()/(selected|truth).sum()),
                pixel_accuracy=float((selected==truth).float().mean()),
                **binary_ranking_metrics(response.flatten(),truth.flatten()))
            elements[arm]=torch.load(OUT/task/(arm+'_elements.pt'),weights_only=False,map_location='cpu')['scores']
        changes[task]=dict(selected_element_changes=int(((elements['baseline']>=.5)!=(elements['ensemble']>=.5)).sum()),
            selected_target_pixel_changes=int((rendered['baseline']!=rendered['ensemble']).sum()))
    summary=dict(registration_sha256=sha256_file(OUT/'registration.json'),complete=True,results=results,
        macro={arm:{metric:sum(results[t][arm][metric] for t in TASKS)/len(TASKS)
            for metric in ('iou','pixel_accuracy','average_precision')} for arm in reg['variants']},
        changes=changes,baseline_prediction_and_scoring_masks_exact=True,
        target_gt_opened_only_after_all_paired_predictions=True,
        source_ensemble={t:json.loads((OUT/'queries'/t/'receipt.json').read_text())['ensemble'] for t in TASKS},
        reference_positive_agreement={t:reports[t]['reference_constraints'] for t in TASKS},
        prediction_reports={t:sha256_file(OUT/t/'prediction_report.json') for t in TASKS})
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary['macro'],indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--register',action='store_true')
    p.add_argument('--tasks',nargs='+',choices=TASKS);p.add_argument('--score',action='store_true');a=p.parse_args()
    if a.register:register()
    reg=json.loads((OUT/'registration.json').read_text())
    for path,digest in reg['implementations'].items():assert sha256_file(path)==digest,path
    for task in a.tasks or []:
        parent=Path('output/nvos_ropefix_pipeline_20260917')/('horns_center' if task=='horns_left' else task)
        command=[sys.executable,'scripts/audits/predict_nvos_positive_ensemble_next32.py',
            '--scene',task,'--root',str(parent/'field'),'--geometry-report',str(parent/'frozen_geometry/report.json'),
            '--runs',*reg['variants'],'--query-root',reg['parent_query_root'],
            '--device','cuda:0','--output',str(OUT/task)]
        start=time.time()
        with (OUT/(task+'.log')).open('x') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONPATH='.'))
        (OUT/(task+'.execution.json')).write_text(json.dumps(dict(command=command,
            returncode=result.returncode,seconds=time.time()-start),indent=2)+'\n')
        if result.returncode:raise RuntimeError(task+' failed; preserve attempt')
        print(task,'paired predictions complete',flush=True)
    if a.score:score(reg)


if __name__=='__main__':main()
