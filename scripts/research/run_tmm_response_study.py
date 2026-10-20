"""Bounded matched refinement of retained v10 responses, full-five feedback.

Frozen v10 frontend outputs are explicitly the experimental inputs, not new
end-to-end predictions. Two fixed permission arms change response values only.
Independent selected-set heads are retained and reported as unchanged controls.
All 31 response jobs finish before any new target-GT scoring begins.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from radio_gs.query_field.bounded_experiment import BoundedExperiment, tree_bytes
from radio_gs.querying.response_refinement import ResponseRefiner
from radio_gs.querying.response_study_adapters import (
    SYSTEM, load, read, readout_path, source_observations, uqis_inputs, uqis_observations,
)
from scripts.research.replay_tmm_response import POLICY, STUDY, sha

ARMS = ('partial', 'complement')
CONTROL = 'permission_pair'
SOURCE_PATHS = ('radio_gs/querying/response_refinement.py', 'radio_gs/querying/response_permissions.py',
                'radio_gs/querying/response_study_adapters.py', 'scripts/research/run_tmm_response_study.py',
                'radio_gs/querying/visibility_refinement.py')


def run_name(prefix, job):
    return f"{prefix}_{job['benchmark']}_{job['name']}"


def save_columns(run, arm, values):
    for i in range(values.shape[1]):
        run.public_artifact(run.root/f'{arm}_{i:03d}.npy',values[:,i].cpu().numpy(),'numpy')


def read_columns(run, arm, count):
    return torch.from_numpy(np.stack([np.load(run.root/f'{arm}_{i:03d}.npy') for i in range(count)],1))


def apply_pair(prior, observations):
    if CONTROL == 'observable':
        from radio_gs.querying.visibility_refinement import VisibilityNormalizedRefiner
        refiner=VisibilityNormalizedRefiner(prior)
        for partial,_ in observations:refiner.add(partial)
        return [refiner.finish()]
    refiners = [ResponseRefiner(prior) for _ in ARMS]
    for pair in observations:
        for refiner, item in zip(refiners,pair):
            refiner.add(item)
    return [r.finish() for r in refiners]


@torch.no_grad()
def predict(job, run):
    started = time.monotonic()
    report = dict(job=job, frozen_backend_response_input=True, independent_set_head_unchanged=True,
                  no_new_teacher_calls=True, ground_truth_read=False)
    if job['benchmark'] == 'uqis':
        membership, visibility, frames, details = uqis_inputs(job)
        from radio_gs.querying.registered_observation_query import public_queries
        queries,_ = public_queries(Path(job['queries']),job['name'])
        if {q['query_id'] for q in queries} != {q['query_id'] for q in details}:
            raise ValueError('cached frozen-provider query set differs from declared queries')
        root = SYSTEM/'uqis'/job['name']/'query/registered/identity_scope'
        rows = []
        for qi, detail in enumerate(details):
            qid = detail['query_id']; path = root/'predictions'/f'{qid}.npy'
            prior = torch.from_numpy(np.load(path)[:,None]).cuda()
            pair = apply_pair(prior,uqis_observations(membership,visibility,frames,detail))
            row = dict(query_id=qid,modality=detail['modality'],source_response_sha256=sha(path),
                       source_selection_sha256=sha(root/'selections'/f'{qid}.npy'),arms={})
            for arm,(value,stats) in zip(ARMS,pair):
                run.public_artifact(run.root/f'{arm}_{qi:03d}.npy',value[:,0].cpu().numpy(),'numpy')
                row['arms'][arm] = stats
            rows.append(row)
        report.update(columns=[r['query_id'] for r in rows],queries=rows,
            map_manifest_sha256=sha(job['scene']),source_rgb_read=False,source_depth_read=False,
            source_masks_read=False,query_identity='retained frozen provider, not ground-truth identity')
    else:
        path = readout_path(job); data = load(path)
        prior = (data['category_logits'].sigmoid() if job['benchmark']=='scannet' else data['scores']).cuda()
        pair = apply_pair(prior,source_observations(job,data,prior))
        report.update(columns=data['texts'],source_response=str(path),source_response_sha256=sha(path),arms={})
        observed = data['semantic_observed'].cuda()
        for arm,(value,stats) in zip(ARMS,pair):
            # Frozen validity policy: new evidence never expands scene support.
            value[~observed] = prior[~observed]
            save_columns(run,arm,value)
            report['arms'][arm] = stats
    report['seconds'] = time.monotonic()-started
    report['prediction_files'] = {p.name:sha(p) for p in run.root.glob('*.npy')}
    run.json('prediction.json',report)
    print('predicted',job['benchmark'],job['name'],flush=True)
    return report


def score_lerf(job, run, report, *, response_loader=read_columns, arms=None):
    arms = ARMS if arms is None else arms
    from radio_gs.evaluation.lerf_readouts import evaluate_lerf_readouts
    from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
    data = load(readout_path(job)); scene = read(job['scene']); b = data['semantic_geometry']
    geo = FrozenGaussianGeometry.load(b['path'],expected_sha256=b['sha256'],device='cuda')
    observed = data['semantic_observed'].cuda()
    values = {'baseline':data['scores'].cuda()}
    values.update({arm:response_loader(run,arm,len(report['columns'])).cuda() for arm in arms})
    metrics = evaluate_lerf_readouts(scene=job['name'],out=run.root,texts=data['texts'],scores=values,
        selection_scores=None,geo=geo,default_observed=observed,semantic_geo=geo,semantic_observed=observed,
        selection_geo=None,selection_observed=None,camera_report=scene['camera_report'],
        camera_report_sha256=scene['camera_report_sha256'],score_selected=False,report_writer=run.json,
        comparison_reference='baseline')
    retained = read(SYSTEM/'lerf'/job['name']/'result.json')['metrics']
    if abs(metrics['baseline']['iou']-retained['semantic_iou']) > 1e-6:
        raise AssertionError('LERF baseline did not reproduce retained IoU')
    result = {a:dict(semantic_iou=m['iou'],ap=m['ap'],selected_projection_iou=retained['selected_projection_iou'])
              for a,m in metrics.items()}
    return dict(metrics=result,selected_projection_status='frozen identical selected-set input; retained score')


@torch.no_grad()
def score_nvos(job, run, report, *, response_loader=read_columns, arms=None):
    arms = ARMS if arms is None else arms
    import cv2
    from PIL import Image
    from gsplat import rasterization
    from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
    from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
    from radio_gs.training.retrieval_metrics import binary_ranking_metrics
    data = load(readout_path(job)); scene = read(job['scene']); spec = read(job['evaluation'])
    b = data['semantic_geometry']; geo = FrozenGaussianGeometry.load(b['path'],expected_sha256=b['sha256'],device='cuda')
    gp = MogeGeometryPrior(scene['camera_report'],scene['camera_report_sha256'])
    cam = gp.cameras[spec['target_frame']]; w,h = spec['raster_wh']
    k = cam.intrinsic.cuda().clone(); k[0]*=w/cam.width; k[1]*=h/cam.height
    observed = data['semantic_observed'].cuda()
    values = {'baseline':data['scores'].cuda()}
    values.update({arm:response_loader(run,arm,1).cuda() for arm in arms})
    predicted = {}
    for arm,value in values.items():
        image,alpha,_ = rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,
            opacities=geo.opacities.flatten(),colors=torch.cat([value,observed.float()[:,None]],1),
            viewmats=cam.world_to_camera.cuda()[None],Ks=k[None],width=w,height=h,
            near_plane=.001*gp.scale,far_plane=100*gp.scale,packed=False)
        mass = image[0,...,1]; valid = (mass/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8)
        predicted[arm] = torch.where(valid,image[0,...,0]/mass.clamp_min(1e-8),0).cpu()
    gt = np.asarray(Image.open(spec['gt_mask']))
    truth = torch.from_numpy(np.any(gt>127,axis=2) if gt.ndim==3 else gt>127)
    metrics = {}; binary = {}
    for arm,value in predicted.items():
        response = torch.from_numpy(cv2.resize(value.numpy(),(truth.shape[1],truth.shape[0]),interpolation=cv2.INTER_LINEAR))
        binary[arm] = response >= spec['threshold']; union = (binary[arm]|truth).sum()
        metrics[arm] = dict(iou=float((binary[arm]&truth).sum()/union) if union else 1.,
            ap=binary_ranking_metrics(response.flatten(),truth.flatten())['average_precision'])
    old = read(SYSTEM/'nvos'/job['name']/'result.json')['metrics']
    if abs(metrics['baseline']['iou']-old['iou']) > 1e-6:
        raise AssertionError('NVOS baseline did not reproduce retained IoU')
    changes = {arm:dict(corrected=int(((binary[arm]==truth)&(binary['baseline']!=truth)).sum()),
                       harmed=int(((binary[arm]!=truth)&(binary['baseline']==truth)).sum())) for arm in arms}
    return dict(metrics=metrics,threshold=spec['threshold'],pixel_corrections=changes)


def score_scannet(job, run, report, *, response_loader=read_columns, arms=None):
    arms = ARMS if arms is None else arms
    from scipy.spatial import cKDTree
    from plyfile import PlyData
    from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
    from radio_gs.scannet_constants import compute_split_metrics
    data = load(readout_path(job)); spec = read(job['evaluation']); b = data['semantic_geometry']
    geo = FrozenGaussianGeometry.load(b['path'],expected_sha256=b['sha256'],device='cpu')
    values = {'baseline':data['category_logits'].numpy()}
    # Sigmoid is monotonic; the same argmax/void rule applies after refinement.
    values.update({arm:response_loader(run,arm,len(report['columns'])).numpy() for arm in arms})
    observed = data['semantic_observed'].numpy(); predicted = {}
    for arm,value in values.items():
        predicted[arm] = {}
        for split,ids in spec['splits'].items():
            cols = [next(i for i,q in enumerate(data['queries']) if q['class_id']==c) for c in ids]
            pred = np.asarray(ids)[value[:,cols].argmax(1)]; pred[~observed]=0
            predicted[arm][split] = pred
    vertices = PlyData.read(spec['ground_truth'])['vertex'].data
    xyz = np.stack([vertices[k] for k in ['x','y','z']],1)
    _,index = cKDTree(geo.xyz.numpy()).query(xyz,workers=4); truth = np.asarray(vertices['label'])
    results = {arm:{split:compute_split_metrics(pred[index],truth,spec['splits'][split])
        for split,pred in preds.items()} for arm,preds in predicted.items()}
    metrics = {arm:{'miou_'+split:d['miou'] for split,d in splits.items()} for arm,splits in results.items()}
    old = read(SYSTEM/'scannet'/job['name']/'result.json')['metrics']
    if max(abs(metrics['baseline'][k]-old[k]) for k in old) > 1e-12:
        raise AssertionError('ScanNet baseline did not reproduce retained metrics')
    changes = {arm:{split:dict(corrected=int(((pred[index]==truth)&(predicted['baseline'][split][index]!=truth)&np.isin(truth,spec['splits'][split])).sum()),
        harmed=int(((pred[index]!=truth)&(predicted['baseline'][split][index]==truth)&np.isin(truth,spec['splits'][split])).sum()))
        for split,pred in predicted[arm].items()} for arm in arms}
    return dict(metrics=metrics,point_corrections=changes,per_class=results)


def score_uqis(jobs, runs, reports, summary_run, *, response_loader=None, arms=None):
    arms = ARMS if arms is None else arms
    from radio_gs.benchmarks.uqis.evaluate import score_predictions, score_selections
    from radio_gs.benchmarks.uqis import MODALITIES
    package = Path(jobs[0]['package']); targets = read(package/'evaluator/targets.json')
    xyz = {d['scene_id']:np.load(package/'method'/d['mesh_xyz_path']) for d in targets['scene_domains']}
    ids = {d['scene_id']:np.load(package/'evaluator'/d['mesh_instance_ids_path']) for d in targets['scene_domains']}
    scores = {}; selections = {}
    for arm in ('baseline',)+arms:
        predicted = {}
        for job,run,report in zip(jobs,runs,reports):
            source = SYSTEM/'uqis'/job['name']/'query/registered/identity_scope'
            columns = response_loader(run,arm,len(report['columns'])).numpy() if arm!='baseline' and response_loader else None
            for qi,qid in enumerate(report['columns']):
                predicted[qid] = (columns[:,qi] if columns is not None else
                    np.load(source/'predictions'/f'{qid}.npy' if arm=='baseline' else run.root/f'{arm}_{qi:03d}.npy'))
                if arm == 'baseline': selections[qid] = np.load(source/'selections'/f'{qid}.npy')
        result = score_predictions(targets['targets'],predicted,xyz,ids)
        summary_run.json('uqis_'+arm+'.json',result)
        scores[arm] = {m+'_ap':v['scene_macro']['average_precision'] for m,v in result['modalities'].items()}
    selected = score_selections(targets['targets'],selections,xyz,ids,MODALITIES)
    summary_run.json('uqis_fixed_selections.json',selected)
    for arm in scores:
        scores[arm].update({m+'_selected_iou':v['scene_macro']['iou'] for m,v in selected['modalities'].items()})
    old = read(SYSTEM/'summary.json')['macro']['uqis']
    if max(abs(scores['baseline'][k]-old[k]) for k in old)>1e-6:
        raise AssertionError('UQIS baseline did not reproduce retained metrics')
    return dict(metrics=scores,selected_set_status='same frozen boolean output, freshly rescored',queries=268)


def main():
    global ARMS,CONTROL
    parser = argparse.ArgumentParser(); parser.add_argument('--run',default='post_response_v1')
    parser.add_argument('--control',choices=['permission_pair','observable'],default='permission_pair')
    parser.add_argument('--predict-only',action='store_true'); parser.add_argument('--score-only',action='store_true')
    parser.add_argument('--resume-predictions',action='store_true')
    args = parser.parse_args(); torch.set_num_threads(4)
    CONTROL=args.control
    if CONTROL=='observable':ARMS=('observable',)
    if args.predict_only and args.score_only: raise ValueError('choose one phase')
    jobs = read('configs/pipeline/system_v10.json')['jobs']
    # All protocols are declared up front; no per-scene metric-driven branching.
    jobs = sorted(jobs,key=lambda j:({'lerf':0,'nvos':1,'scannet':2,'uqis':3}[j['benchmark']],j['name']))
    summary_run = BoundedExperiment(STUDY,args.run,POLICY,resume=args.score_only or args.resume_predictions)
    if not args.score_only:
        manifest=dict(jobs=jobs,arms=ARMS,source_hashes={p:sha(p) for p in SOURCE_PATHS},
            baseline='retained v10 response as frozen experimental input',new_end_to_end_system=False,
            predictions_before_scoring=True,selection_heads_frozen=True,step_size=1.,
            control=CONTROL,changed_variable=('partial-evidence exclusion permission; same update/target/weight' if CONTROL=='permission_pair'
                else 'observable assignment normalization; same partial permissions and targets as post_response_v1'),
            input_contract='source assisted LERF/NVOS/ScanNet; map-only UQIS',
            additional_supervision='none; inherited v10 member training remains',max_new_predictions_bytes=660000000*len(ARMS))
        if args.resume_predictions:
            import ast
            old=read(summary_run.root/'manifest.json');failure=read(summary_run.root/'uqis_roundoff_failure.json')
            if old['jobs']!=jobs or old['arms']!=list(ARMS):raise ValueError('resume protocol changed')
            for p in SOURCE_PATHS[:2]:
                if sha(p)!=old['source_hashes'][p]:raise ValueError('common inference rule changed during resume')
            code=Path(SOURCE_PATHS[2]).read_text();tree=ast.parse(code)
            fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='source_observations')
            if hashlib.sha256(ast.get_source_segment(code,fn).encode()).hexdigest()!=failure['source_observations_source_sha256']:
                raise ValueError('completed source-assisted jobs changed during UQIS repair')
            summary_run.json('prediction_resume.json',dict(**manifest,
                repair='float64 accumulation for mean of nested binary map memberships; formula unchanged',
                completed_source_adapter_unchanged=True,failed_unsealed_UQIS_run_preserved=True))
        else:
            summary_run.json('manifest.json',manifest)
        runs=[]; reports=[]
        for job in jobs:
            completed=STUDY/run_name(args.run,job)/'prediction.json'
            if args.resume_predictions and completed.exists():
                run=BoundedExperiment(STUDY,run_name(args.run,job),POLICY,resume=True);report=read(completed)
                if report['job']!=job or any(sha(run.root/f)!=h for f,h in report['prediction_files'].items()):
                    raise ValueError('completed prediction changed during resume')
                runs.append(run);reports.append(report);continue
            run = BoundedExperiment(STUDY,run_name(args.run,job),POLICY)
            run.preflight(1 if job['benchmark']=='uqis' else 2*load(readout_path(job))['scores'].numel()*4+2**20)
            report = predict(job,run); runs.append(run); reports.append(report)
            gc.collect(); torch.cuda.empty_cache()
        summary_run.json('predictions_complete.json',dict(jobs=31,all_five_present=True,
            reports={str(r.root/'prediction.json'):sha(r.root/'prediction.json') for r in runs},
            storage_bytes=tree_bytes(STUDY),no_target_gt_read=True))
    else:
        seal = read(summary_run.root/'predictions_complete.json')
        if any(sha(p)!=h for p,h in seal['reports'].items()):raise ValueError('prediction report changed after seal')
        runs=[BoundedExperiment(STUDY,run_name(args.run,j),POLICY,resume=True) for j in jobs]
        reports=[read(r.root/'prediction.json') for r in runs]
    if args.predict_only:return
    for run,report in zip(runs,reports):
        if any(sha(run.root/f)!=h for f,h in report['prediction_files'].items()):raise ValueError('prediction changed after seal')
        if 'source_response' in report and sha(report['source_response'])!=report['source_response_sha256']:
            raise ValueError('frozen baseline response changed')
        if report['job']['benchmark']=='uqis':
            root=SYSTEM/'uqis'/report['job']['name']/'query/registered/identity_scope'
            for row in report['queries']:
                for folder,key in [('predictions','source_response_sha256'),('selections','source_selection_sha256')]:
                    if sha(root/folder/(row['query_id']+'.npy'))!=row[key]:raise ValueError('UQIS frozen input changed')
    summary_run.json('evaluation_manifest.json',dict(prediction_seal_sha256=sha(summary_run.root/'predictions_complete.json'),
        scoring_source_hashes={p:sha(p) for p in ('scripts/research/run_tmm_response_study.py','radio_gs/evaluation/lerf_readouts.py',
            'radio_gs/benchmarks/uqis/evaluate.py','radio_gs/scannet_constants.py')},all_predictions_checked=True))
    functions={'lerf':score_lerf,'nvos':score_nvos,'scannet':score_scannet};results=[]
    for job,run,report in zip(jobs,runs,reports):
        if job['benchmark']=='uqis':continue
        result=functions[job['benchmark']](job,run,report)
        run.json('evaluation.json',result); results.append(dict(job=job,**result))
        print('scored',job['benchmark'],job['name'],result['metrics'],flush=True)
        gc.collect();torch.cuda.empty_cache()
    indices=[i for i,j in enumerate(jobs) if j['benchmark']=='uqis']
    uqis=score_uqis([jobs[i] for i in indices],[runs[i] for i in indices],[reports[i] for i in indices],summary_run)
    macro={}
    for family in functions:
        group=[r for r in results if r['job']['benchmark']==family]
        macro[family]={arm:{k:sum(r['metrics'][arm][k] for r in group)/len(group) for k in group[0]['metrics'][arm]}
            for arm in ('baseline',)+ARMS}
    macro['uqis']=uqis['metrics']
    summary_run.json('summary.json',dict(complete=True,macro=macro,rows=results,uqis=uqis,
        all_prediction_jobs=31,independent_test=False,default_changed=False,selection_heads_frozen=True,
        actual_new_storage_bytes=tree_bytes(STUDY),new_end_to_end_system=False))
    print(json.dumps(macro,indent=2),flush=True)


if __name__=='__main__':
    main()
