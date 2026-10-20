"""A06: isolate forced positivity from the full estimated-identity condition.

Predict all 67 image patches before loading evaluator labels. Freshly score all
268 UQIS queries. Other benchmark outputs remain explicit retained compatibility
references, not new inference. No dense predictions or model weights are saved.
"""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse
import torch

from radio_gs.query_field.bounded_experiment import BoundedExperiment, tree_bytes
from radio_gs.querying.direct_anchor_membership import DirectAnchorMembership
from radio_gs.querying.estimated_anchor_authority import estimated_anchor_patch, apply_patch
from radio_gs.querying.registered_observation_query import public_queries
from radio_gs.querying.response_study_adapters import SYSTEM, read
from scripts.research.replay_tmm_response import STUDY, POLICY, sha
from scripts.research.tmm_ram_predictions import public_query_files


def predict(job):
    started = time.monotonic()
    folder = Path(job['scene']).parent
    spec = read(job['scene'])
    path = SYSTEM/'uqis'/job['name']/'query/registered/runs'/f"{job['name']}.json"
    details = read(path)['query_details']
    queries, _ = public_queries(Path(job['queries']), job['name'])
    if {d['query_id']: d['modality'] for d in details} != {q['query_id']: q['modality'] for q in queries}:
        raise ValueError('public query identities differ from retained provider')
    if spec.get('query_conditioned_membership') or spec['inferred_anchor_membership'] != ['image']:
        raise ValueError('unexpected effective provider')
    membership = sparse.load_npz(folder/'membership.npz').tocsr()
    visibility = sparse.load_npz(folder/'visibility.npz')
    offsets = np.load(folder/'region_offsets.npy')
    frames = np.repeat(np.arange(visibility.shape[1]), np.diff(offsets))
    descriptors = np.load(folder/'descriptors.npy').astype(np.float32)
    descriptors /= np.maximum(np.linalg.norm(descriptors, axis=1, keepdims=True), 1e-8)
    checkpoint = spec['direct_anchor_membership']['checkpoint']
    reader = DirectAnchorMembership(membership, visibility, frames, np.load(folder/'mesh_xyz.npy'),
        descriptors, sparse.load_npz(folder/'transfer.npz'), checkpoint)
    paths = [Path(job['scene']), path, Path(checkpoint), *public_query_files(job['queries'])]
    paths += [folder/name for name in ['membership.npz', 'visibility.npz', 'region_offsets.npy',
                                      'descriptors.npy', 'mesh_xyz.npy', 'transfer.npz']]
    source = SYSTEM/'uqis'/job['name']/'query/registered/identity_scope'
    patches, unchanged = [], []
    for detail in details:
        qid = detail['query_id']
        p, s = source/'predictions'/f'{qid}.npy', source/'selections'/f'{qid}.npy'
        paths.extend([p, s])
        if detail['modality'] == 'image':
            patch = estimated_anchor_patch(reader, detail, np.load(p), np.load(s))
            patch['seed_region'] = detail['inferred_anchor_membership']['seed_region']
            patches.append(patch)
        else:
            unchanged.append(dict(query_id=qid, modality=detail['modality']))
    return dict(job=job, ground_truth_read=False, patches=patches, unchanged_queries=unchanged,
        all_non_anchor_vertices_unchanged=True, classifier_and_identity_condition_unchanged=True,
        actual_point_contract_preserved=True, source_rgb_read=False, source_depth_read=False,
        input_hashes={str(p): sha(p) for p in sorted(set(paths))}, seconds=time.monotonic()-started)


def score(jobs, reports, run):
    from radio_gs.benchmarks.uqis import MODALITIES
    from radio_gs.benchmarks.uqis.evaluate import score_predictions, score_selections
    from scripts.research.summarize_tmm_response_study import paired
    package = Path(jobs[0]['package'])
    target_path = package/'evaluator/targets.json'
    targets = read(target_path)
    xyz = {d['scene_id']: np.load(package/'method'/d['mesh_xyz_path']) for d in targets['scene_domains']}
    ids = {d['scene_id']: np.load(package/'evaluator'/d['mesh_instance_ids_path']) for d in targets['scene_domains']}
    references, selections, patches = {}, {}, {}
    for job, report in zip(jobs, reports):
        source = SYSTEM/'uqis'/job['name']/'query/registered/identity_scope'
        records = report['patches'] + report['unchanged_queries']
        for row in records:
            qid = row['query_id']
            references[qid] = np.load(source/'predictions'/f'{qid}.npy')
            selections[qid] = np.load(source/'selections'/f'{qid}.npy')
        patches.update({p['query_id']: p for p in report['patches']})
    scores, sets = {}, {}
    for arm in ['baseline', 'unforced']:
        prediction, selection = dict(references), dict(selections)
        if arm == 'unforced':
            for qid, patch in patches.items():
                prediction[qid], selection[qid] = apply_patch(references[qid], selections[qid], patch)
        scores[arm] = score_predictions(targets['targets'], prediction, xyz, ids)
        sets[arm] = score_selections(targets['targets'], selection, xyz, ids, MODALITIES)
        run.json('uqis_' + arm + '.json', scores[arm])
        run.json('uqis_' + arm + '_selections.json', sets[arm])
        print('scored all 268 queries:', arm, flush=True)
    reference = read(SYSTEM/'summary.json')
    macro = {}
    for arm in scores:
        macro[arm] = {m+'_ap': scores[arm]['modalities'][m]['scene_macro']['average_precision'] for m in MODALITIES}
        macro[arm].update({m+'_selected_iou': sets[arm]['modalities'][m]['scene_macro']['iou'] for m in MODALITIES})
    for metric, value in macro['baseline'].items():
        if abs(value-reference['macro']['uqis'][metric]) > 1e-8:
            raise AssertionError('baseline UQIS scoring differs: ' + metric)
    effects = {}
    for m in MODALITIES:
        for name, records, key in [('ap', scores, 'average_precision'), ('selected_iou', sets, 'iou')]:
            base = records['baseline']['modalities'][m]['per_scene']
            new = records['unforced']['modalities'][m]['per_scene']
            effects[m+'_'+name] = paired([new[s][key]-v[key] for s,v in base.items()])
    # Labels enter only here, after all intervention values and their inputs seal.
    by_query = {r['query_id']: r for r in sets['baseline']['per_query']}
    diagnostic = []
    for job in jobs:
        members = sparse.load_npz(Path(job['scene']).parent/'membership.npz').tocsc()
        for target in targets['targets']:
            if target['scene_id'] != job['name']:
                continue
            qid = target['queries']['image']; patch = patches[qid]
            anchor, seed = patch['anchor_vertex'], patch['seed_region']
            truth = ids[job['name']] == target['instance_id']
            vertices = members[:, seed].indices
            intersection = int(truth[vertices].sum())
            item = dict(patch, scene_id=job['name'], instance_id=target['instance_id'],
                anchor_correct=bool(truth[anchor]), anchor_instance_id=int(ids[job['name']][anchor]),
                seed_target_overlap=intersection, seed_elements=len(vertices),
                seed_target_purity=intersection/max(len(vertices),1),
                seed_target_coverage=intersection/max(int(truth.sum()),1),
                original_image_iou=by_query[qid]['iou'],
                same_target_actual_point_iou={m: by_query[target['queries'][m]]['iou'] for m in ['point_2d','point_3d']})
            # If the bit changes, selected count and intersection each change by
            # at most one. This is an evaluation diagnosis, never an oracle repair.
            diagnostic.append(item)
    strata = {}
    for correct in [True, False]:
        rows = [r for r in diagnostic if r['anchor_correct'] == correct]
        values = {s: np.mean([r['original_image_iou'] for r in rows if r['scene_id']==s])
                  for s in sorted({r['scene_id'] for r in rows})}
        strata['correct' if correct else 'incorrect'] = dict(queries=len(rows), scenes=len(values),
            image_iou_query_mean=float(np.mean([r['original_image_iou'] for r in rows])) if rows else None,
            image_iou_conditional_scene_macro=float(np.mean(list(values.values()))) if values else None)
    run.json('anchor_diagnosis.json', dict(rows=diagnostic, strata=strata,
        incorrect_anchor_but_seed_contains_target=sum(not r['anchor_correct'] and r['seed_target_overlap']>0 for r in diagnostic),
        seed_has_no_target=sum(r['seed_target_overlap']==0 for r in diagnostic),
        conditional_statistics_are_not_causal_effects=True, diagnosis_uses_gt_after_prediction_seal=True,
        no_oracle_anchors_or_predictions_created=True))
    return dict(complete=True, scientific_intervention='remove only estimated-image-anchor positive override',
        full_query_cohort=268, image_queries=67, image_patch_count=len(patches),
        changed_rank_entries=sum(p['rank_entries_changed'] for p in patches.values()),
        changed_selected_entries=sum(p['selection_entries_changed'] for p in patches.values()),
        classifier_raw_anchor_probability_range=[min(p['unforced_probability'] for p in patches.values()),
                                                max(p['unforced_probability'] for p in patches.values())],
        uqis_macro=macro, effects=effects, anchor_strata=strata,
        source_benchmark_compatibility={f: dict(metrics=v, changed=False, fresh_inference=False,
            provenance=str(SYSTEM/'summary.json'), sha256=sha(SYSTEM/'summary.json'))
            for f,v in reference['macro'].items() if f!='uqis'},
        all_five_feedback_present=True, all31_fresh_inference=False,
        selection_rule_frozen=True, image_selected_sets_allowed_to_change_at_anchor_only=True,
        actual_point_and_text_outputs_exact=True, independent_test=False, default_changed=False,
        new_common_admission_algorithm=False, no_new_dense_predictions=True,
        evaluator_source_sha256={p:sha(p) for p in ['radio_gs/benchmarks/uqis/evaluate.py',
            'radio_gs/benchmarks/uqis/metrics.py','scripts/research/summarize_tmm_response_study.py']},
        target_manifest_sha256=sha(target_path), study_logical_bytes=tree_bytes(STUDY))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', default='estimated_anchor_authority_v1')
    args = parser.parse_args(); torch.set_num_threads(4)
    run = BoundedExperiment(STUDY, args.run, POLICY); run.preflight(8*2**20)
    recipe = read('configs/pipeline/system_v10.json')
    jobs = [j for j in recipe['jobs'] if j['benchmark']=='uqis']
    sources = ['scripts/research/audit_tmm_estimated_anchor.py',
               'radio_gs/querying/estimated_anchor_authority.py',
               'radio_gs/querying/direct_anchor_membership.py',
               'radio_gs/querying/anchor_observation_evidence.py', 'configs/pipeline/system_v10.json',
               'radio_gs/benchmarks/uqis/evaluate.py', 'radio_gs/benchmarks/uqis/metrics.py',
               'scripts/research/summarize_tmm_response_study.py']
    run.json('manifest.json', dict(jobs=recipe['jobs'], intervention_jobs=jobs,
        kind='targeted causal audit; all-five reference feedback, not a shared-method trial',
        changed_variable='explicit image-estimated anchor override only',
        all_image_patches_before_any_target_scoring=True, frozen_classifier_and_conditioning_anchor=True,
        threshold_fitted=False, actual_point_contract_changed=False, dense_prediction_storage=False,
        source_hashes={p:sha(p) for p in sources}))
    run.json('source_snapshot.json', {p:dict(text=Path(p).read_text(),sha256=sha(p)) for p in sources})
    reports = []
    for job in jobs:
        report = predict(job); run.json(job['name']+'_patches.json', report); reports.append(report)
        print('sealed', job['name'], len(report['patches']), 'image patches', flush=True)
        gc.collect()
    if len(jobs)!=9 or sum(len(r['patches']) for r in reports)!=67:
        raise ValueError('complete intervention cohort changed')
    run.json('patches_complete.json', dict(image_queries=67, all_uqis_queries=268, ground_truth_read=False,
        report_hashes={str(run.root/(j['name']+'_patches.json')):sha(run.root/(j['name']+'_patches.json')) for j in jobs}))
    for report in reports:
        for path, digest in report['input_hashes'].items():
            if sha(path)!=digest:raise ValueError('input changed after prediction seal')
    run.json('evaluation_manifest.json', dict(patch_seal_sha256=sha(run.root/'patches_complete.json'),
        inputs_rechecked=True, all_patches_before_scoring=True))
    summary = score(jobs, reports, run); run.json('summary.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['effects','source_benchmark_compatibility']},indent=2))


if __name__=='__main__':main()
