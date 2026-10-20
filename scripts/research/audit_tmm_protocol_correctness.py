"""CPU audit of retained predictions; never fits or promotes benchmark choices.

Independent pixel/confusion scoring and a ragged-neighbour Gaussian evaluator
check the retained implementations. NVOS threshold-oracle values are explicitly
diagnostic upper bounds, not selectable results. No dense outputs are saved.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F

from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.query_field.bounded_experiment import BoundedExperiment
from radio_gs.scripts.eval_opengaussian_lerf_baseline import SCENE_GT_FRAMES
from scripts.research.replay_tmm_response import POLICY, STUDY, sha

REPO = Path(__file__).resolve().parents[2]
SYSTEM = REPO / 'output/query_conditioned_membership_20261006/image_convergence/system'
BRIDGE = STUDY / 'scannet_paper8_metric_bridge_v1/summary.json'
LUDVIG = REPO / 'output/protocol_audit_20260731/ludvig/nvos/released_all_view_full8_3seed_summary.json'
VALA_CODE = Path('/root/baselines/VALA/eval/evaluator3D.py')


def read(path):
    return json.loads(Path(path).read_text())


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def digest(array):
    return hashlib.sha256(np.asarray(array, dtype='<i4').tobytes()).hexdigest()


def binary_counts(pred, truth):
    assert pred.shape == truth.shape
    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    return dict(tp=tp, fp=fp, fn=fn, iou=tp/(tp+fp+fn) if tp+fp+fn else 1.,
                precision=tp/(tp+fp) if tp+fp else 0., recall=tp/(tp+fn) if tp+fn else 0.)


def confusion_metrics(pred, truth, ids, weights=None):
    """Independent raw-NYU40 confusion matrix, including a predicted void column."""
    size = max(int(np.max(truth)), int(np.max(pred)), max(ids)) + 1
    keep = np.isin(truth, ids)
    matrix = np.bincount(truth[keep]*size+pred[keep],
                        weights=None if weights is None else weights[keep].astype(np.float64),
                        minlength=size*size).reshape(size, size)
    gt_mass, pred_mass = matrix.sum(1), matrix.sum(0)
    present = np.asarray([c for c in ids if gt_mass[c] > 0])
    tp = matrix.diagonal()[present]
    return dict(miou=float(np.mean(tp/(gt_mass[present]+pred_mass[present]-tp))),
                macc=float(np.mean(tp/gt_mass[present])), classes=present.tolist(),
                valid_count=int(keep.sum()))


def radius_pseudo_labels(xyz, scales, quats, points, labels):
    """Independent float64 local-coordinate density vote, no helper reuse.

    All current carriers have <1000 radius neighbours, hence top-k truncation
    is inactive. Fail if that precondition or nonempty neighbourhood fails.
    """
    tree = cKDTree(points)
    rotation = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
    nclasses = int(labels.max())+1
    result = np.zeros(len(xyz), np.int32)
    largest = 0
    for start in range(0, len(xyz), 4096):
        end = min(start+4096, len(xyz))
        neighbours = tree.query_ball_point(xyz[start:end], 5*scales[start:end].max(1), workers=4)
        counts = np.asarray([len(v) for v in neighbours])
        assert counts.min() > 0 and counts.max() < 1000
        largest = max(largest, int(counts.max()))
        row = np.repeat(np.arange(end-start), counts)
        index = np.concatenate(neighbours).astype(np.int64)
        local = np.einsum('ni,nij->nj', points[index]-xyz[start:end][row], rotation[start:end][row])
        d2 = np.square(local / scales[start:end][row]).sum(1)
        key = row*nclasses + labels[index]
        mass = np.bincount(key, weights=np.exp(-.5*d2), minlength=(end-start)*nclasses)
        number = np.bincount(key, minlength=(end-start)*nclasses)
        result[start:end] = (mass/np.maximum(number, 1)).reshape(end-start, nclasses).argmax(1)
    return result, dict(max_radius_neighbours=largest, empty_neighbourhoods=0, topk_truncation_active=False)


def audit_lerf(jobs):
    rows = {}
    for job in jobs:
        name = job['name']
        root = SYSTEM/'lerf'/name/'evaluation'
        retained = read(root/'evaluation.json')['metrics']['hierarchical_pipeline']
        objects = retained['selected']['objects']
        independent = []
        for obj in objects:
            truth = np.asarray(Image.open(obj['gt_path'])) > 10
            pred = np.asarray(Image.open(obj['pred_path']).convert('L')) > 10
            result = binary_counts(pred, truth)
            independent.append(dict(frame=obj['frame'], query=obj['query'], **result,
                                    retained_delta=result['iou']-obj['iou']))
        semantic = read(root/'query_results.json')['hierarchical_pipeline']
        expected = set()
        for frame in SCENE_GT_FRAMES[name]:
            ann = read(Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label')/name/(frame+'.json'))
            expected.update((frame, x['category'].strip()) for x in ann['objects']
                            if x.get('segmentation') and x.get('category', '').strip())
        actual = [(o['frame'], o['query']) for o in objects]
        semantic_keys = [(o['frame'], o['query']) for o in semantic]
        assert len(actual) == len(set(actual)) and set(actual) == expected
        assert len(semantic_keys) == len(set(semantic_keys)) and set(semantic_keys) == expected
        assert max(abs(r['retained_delta']) for r in independent) < 1e-12
        scene = read(job['scene'])
        fit = read(Path(scene['geometry']['path']).parent/'fit.json')
        authority = read(scene['source_rgb_authority'])
        held = {int(f.split('_')[-1]) for f in SCENE_GT_FRAMES[name]}
        sources = dict(semantic=scene['source_frames'], calibration=scene['calibration']['source_frames'],
                       geometry_rgb=fit['source_frames'],
                       source_authority=[int(x['image_id'].split('_')[-1]) for x in authority['images']])
        intersections = {k: sorted(held.intersection(v)) for k,v in sources.items()}
        assert not any(intersections.values())
        rows[name] = dict(selected_iou=float(np.mean([r['iou'] for r in independent])),
            selected_acc025=float(np.mean([r['iou'] > .25 for r in independent])),
            semantic_iou_reaggregated=float(np.mean([r['iou'] for r in semantic])),
            localization_reaggregated=float(np.mean([r['localization_hit'] for r in semantic])),
            count=len(objects), missing=0, duplicate=0, heldout_intersections=intersections,
            source_counts={k:len(v) for k,v in sources.items()}, objects=independent,
            semantic_limit='Per-object aggregates checked; semantic rasters not freshly rendered in this CPU audit.')
        print('LERF', name, rows[name]['selected_iou'], flush=True)
    return dict(rows=rows, macro={k:float(np.mean([r[k] for r in rows.values()])) for k in
        ('selected_iou','selected_acc025','semantic_iou_reaggregated','localization_reaggregated')})


def audit_scannet(jobs, prepared):
    bridge = read(BRIDGE)
    module_spec = importlib.util.spec_from_file_location('vala_evaluation_audit', VALA_CODE)
    upstream = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(upstream)
    rows = {}
    for job in jobs:
        name = job['name']; data, geo, predictions, spec = prepared.pop(name)
        vertices = PlyData.read(spec['ground_truth'])['vertex'].data
        points = np.stack([vertices[k] for k in ('x','y','z')], 1)
        truth = np.asarray(vertices['label'], dtype=np.int64)
        xyz = geo.xyz.numpy()
        distance, index = cKDTree(xyz).query(points, workers=4)
        retained = read(SYSTEM/'scannet'/name/'result.json')['metrics']
        mesh = {s:confusion_metrics(pred[index], truth, spec['splits'][s]) for s,pred in predictions.items()}
        assert max(abs(mesh[s]['miou']-retained['miou_'+s]) for s in mesh) < 1e-12
        geometry_bundle = load(data['semantic_geometry']['path'])
        ids = geometry_bundle['source_vertex_ids'].numpy()
        assert set(geometry_bundle['state_dict']) == {'xyz','normals','quaternions','scales','opacities'}
        row = dict(mesh=mesh, gaussian_count=len(xyz), mesh_count=len(points),
                   max_mesh_to_gaussian_distance=float(distance.max()),
                   source_vertex_coordinate_error=float(np.abs(xyz-points[ids]).max()),
                   observed_fraction=float(data['semantic_observed'].float().mean()),
                   geometry_contains_semantic_labels=False)
        if name in bridge['scenes']:
            pseudo, stats = radius_pseudo_labels(xyz, geo.scales.numpy(), geo.quaternions.numpy(), points, truth)
            same_hash = digest(pseudo) == bridge['rows'][name]['pseudo_gt_sha256']
            assert same_hash, ('independent pseudo-GT differs', name)
            sample = np.unique(np.linspace(0,len(xyz)-1,32,dtype=np.int64))
            # Exact released function, CPU device override only; dense cdist/topk.
            with patch('torch.cuda.is_available', return_value=False):
                official = upstream.GaussianEvaluationProtocol().assign_semantic_labels_mahalanobis(
                    dict(mu=xyz[sample],scale=geo.scales.numpy()[sample],rotation=geo.quaternions.numpy()[sample]),
                    points, truth, list(range(int(truth.max())+1)), chunk_size=8)
            assert np.array_equal(pseudo[sample], official)
            weights = (geo.opacities.flatten()*geo.scales.prod(1)).numpy()
            gaussian = {s:confusion_metrics(p, pseudo, spec['splits'][s], weights) for s,p in predictions.items()}
            delta = max(abs(gaussian[s][m]-bridge['rows'][name]['splits'][s][m])
                        for s in gaussian for m in ('miou','macc'))
            assert delta < 1e-12
            row.update(gaussian=gaussian, pseudo_gt_hash_match=same_hash,
                       official_dense_samples=len(sample), official_sample_mismatches=0,
                       pseudo_gt_stats=stats, metric_max_delta=delta)
        rows[name] = row
        print('SCANNET',name,{s:v['miou'] for s,v in mesh.items()},flush=True)
    macro = {}
    for track, names, key in [('mesh10',list(rows),'mesh'),('mesh8',bridge['scenes'],'mesh'),
                             ('gaussian8',bridge['scenes'],'gaussian')]:
        macro[track] = {s:float(np.mean([rows[n][key][s]['miou'] for n in names])) for s in ('19','15','10')}
    return dict(rows=rows, macro=macro, evaluation_domain_changes_are_not_method_gains=True)


def threshold_grid_upper_bound(response, truth):
    """Post-hoc GT oracle over 99 thresholds. NEVER a reported method result."""
    positive = np.sort(response[truth]); negative = np.sort(response[~truth])
    thresholds = np.arange(1,100)/100.
    tp = len(positive)-np.searchsorted(positive, thresholds, side='left')
    fp = len(negative)-np.searchsorted(negative, thresholds, side='left')
    iou = tp/(len(positive)+fp)
    i = int(iou.argmax())
    return dict(iou=float(iou[i]), threshold=float(thresholds[i]), oracle=True, deployable=False)


def audit_nvos(jobs):
    reference = read(LUDVIG)['per_scene']; rows = {}
    for job in jobs:
        name=job['name']; root=SYSTEM/'nvos'/name
        spec, scene = read(job['evaluation']), read(job['scene'])
        response = load(root/'evaluation/response.pt')['response'].numpy()
        assert np.isfinite(response).all()
        d = load(root/'query/typed_readout.pt')
        outside = d['scores'][~d['semantic_observed']]
        outside_nonzero = int(torch.count_nonzero(outside))
        assert outside_nonzero == 0, 'unobserved scores contaminate normalized renderer'
        gt = np.asarray(Image.open(spec['gt_mask']))
        truth = np.any(gt>127,axis=2) if gt.ndim==3 else gt>127
        assert gt.ndim == 2, 'Inspect upstream channel semantics before comparing RGB masks'
        upstream_truth = (gt//gt.max()).astype(bool)
        resized = cv2.resize(response, (truth.shape[1],truth.shape[0]), interpolation=cv2.INTER_LINEAR)
        fixed = binary_counts(resized >= spec['threshold'], truth)
        retained = read(root/'result.json')['metrics']
        assert abs(fixed['iou']-retained['iou']) < 1e-7
        torch_resized = F.interpolate(torch.from_numpy(response)[None,None],size=truth.shape,mode='bilinear',align_corners=False)[0,0].numpy()
        normalized = (response-response.min())/(response.max()-response.min())
        published_readout = F.interpolate(torch.from_numpy(normalized)[None,None],size=truth.shape,mode='bilinear',align_corners=False)[0,0].numpy()
        alternate = binary_counts(published_readout > .25, truth)
        fit = read(Path(scene['geometry']['path']).parent/'fit.json')
        target = spec['target_frame']
        source_lists = dict(semantic=scene['source_frames'],tracking=scene['tracking_frames'],geometry_rgb=fit['source_frames'],
                            source_rgb_keys=[int(k) for k in scene['source_rgb']])
        intersections = {k: [target] if target in v else [] for k,v in source_lists.items()}
        assert not any(intersections.values())
        row = dict(fixed=fixed, retained_iou_delta=fixed['iou']-retained['iou'],
            ap_retained=retained['ap'], source_shape=list(response.shape),gt_shape=list(truth.shape),
            min_response=float(response.min()),max_response=float(response.max()),
            torch_bilinear_same_threshold=binary_counts(torch_resized>=.5,truth),
            published_readout_only=alternate, published_readout_is_full_ludvig_reproduction=False,
            upstream_gt_same_prediction=binary_counts(resized>=.5,upstream_truth),
            upstream_gt_and_readout=binary_counts(published_readout>.25,upstream_truth),
            gt_binarization_changed_pixels=int(np.count_nonzero(truth!=upstream_truth)),
            diagnostic_grid_oracle=threshold_grid_upper_bound(resized, truth),
            ludvig_local_iou=reference[name]['local_mean_iou_percent']/100,
            gap_pp=reference[name]['local_mean_iou_percent']-100*fixed['iou'],
            unobserved_nonzero_scores=outside_nonzero, heldout_intersections=intersections,
            source_counts={k:len(v) for k,v in source_lists.items()}, geometry_rgb_steps=fit['steps'])
        rows[name]=row
        print('NVOS',name,dict(iou=fixed['iou'],precision=fixed['precision'],recall=fixed['recall'],
                             readout=alternate['iou'],oracle=row['diagnostic_grid_oracle']['iou']),flush=True)
    macro={k:float(np.mean([r[k]['iou'] for r in rows.values()])) for k in
           ('fixed','published_readout_only','torch_bilinear_same_threshold','diagnostic_grid_oracle',
            'upstream_gt_same_prediction','upstream_gt_and_readout')}
    macro['ludvig_local']=float(np.mean([r['ludvig_local_iou'] for r in rows.values()]))
    gap=sum(r['gap_pp'] for r in rows.values())
    for row in rows.values(): row['fraction_of_net_gap']=row['gap_pp']/gap
    return dict(rows=rows,macro=macro,oracle_not_a_method_result=True,
                target_rgb_effect_not_causally_measured=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='protocol_correctness_audit_v1')
    args=parser.parse_args(); os.chdir(REPO); torch.set_num_threads(4)
    if hasattr(os,'sched_getaffinity'): os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
    started=time.monotonic(); run=BoundedExperiment(STUDY,args.run,POLICY); run.preflight(2**20)
    jobs=read('configs/pipeline/system_v10.json')['jobs']
    jobs=[j for j in jobs if j['benchmark'] in ('lerf','scannet','nvos')]
    inputs={str(p):sha(p) for p in (Path(__file__),BRIDGE,LUDVIG,VALA_CODE)}
    prepared={}; predictions={}
    for job in jobs:
        name=job['name']; kind=job['benchmark']; root=SYSTEM/kind/name
        inputs[job['scene']]=sha(job['scene']); inputs[job['queries']]=sha(job['queries'])
        if kind=='scannet':
            path=root/'query/category/typed_readout.pt'; data=load(path); inputs[str(path)]=sha(path)
            spec=read(job['evaluation']); binding=data['semantic_geometry']
            geo=FrozenGaussianGeometry.load(binding['path'],expected_sha256=binding['sha256'])
            inputs[binding['path']]=binding['sha256']; classes=[q['class_id'] for q in data['queries']]
            preds={}
            for split,ids in spec['splits'].items():
                columns=[classes.index(c) for c in ids]
                pred=np.asarray(ids,np.int32)[data['category_logits'][:,columns].argmax(1).numpy()]
                pred[~data['semantic_observed'].numpy()]=0; preds[split]=pred
            predictions[name]={s:digest(v) for s,v in preds.items()}
            prepared[name]=(data,geo,preds,spec)
        elif kind=='nvos':
            path=root/'evaluation/response.pt';inputs[str(path)]=sha(path)
            path=root/'query/typed_readout.pt';inputs[str(path)]=sha(path)
        else:
            path=root/'evaluation/evaluation.json';inputs[str(path)]=sha(path)
            path=root/'evaluation/query_results.json';inputs[str(path)]=sha(path)
            for obj in read(root/'evaluation/evaluation.json')['metrics']['hierarchical_pipeline']['selected']['objects']:
                inputs[obj['pred_path']]=sha(obj['pred_path'])
    run.json('prediction_seal.json',dict(inputs=inputs,category_predictions=predictions,
        scope='retained-prediction replay audit; original benchmark already used in development',
        audit_predictions_bound_before_gt=True,new_inference=False))
    result={}
    for kind,fn in [('lerf',audit_lerf),('nvos',audit_nvos),('scannet',lambda j:audit_scannet(j,prepared))]:
        result[kind]=fn([j for j in jobs if j['benchmark']==kind]);run.json(kind+'.json',result[kind])
    result.update(complete=True,durable_dense_bytes=0,elapsed_seconds=time.monotonic()-started,
                  new_model_gain_claim=False,global_no_leakage_proof=False)
    run.json('summary.json',result)
    print(json.dumps({k:v['macro'] for k,v in result.items() if isinstance(v,dict)},indent=2))


if __name__=='__main__': main()
