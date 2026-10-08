"""Post-score, full-UQIS diagnosis of member recall versus source visibility.

Full-scene projection contribution is an exposure proxy, not a measured training
gradient. This audit cannot establish a causal mechanism or choose a model.
"""
import gc
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch

from radio_gs.query_field.storage import load_field, sha256
from scripts.research.run_query_field_protected import ROOT, PREVIOUS, read, write, receipt, check


def main(root=ROOT, cohort_arm='protected_training', extra_roots=None):
    read(root / 'scoring_complete.json')
    out = root / 'diagnostics/point_observation_mass'
    out.mkdir(parents=True, exist_ok=False)
    cohort_path = root / cohort_arm / 'cohort.json'
    cohort = read(cohort_path)
    old = read('configs/query_field/five_benchmark_v1/cohort.json')
    roots = dict(parent=Path(old['root']) / 'predictions/uqis',
                 previous_common=PREVIOUS / 'common_program/uqis',
                 **{cohort_arm: Path(cohort['root']) / 'predictions/uqis'})
    roots.update(extra_roots or {})
    design = read(root / 'design.json')
    write(out / 'design.json', dict(code=receipt(Path(__file__)), inputs=receipt(root / 'predictions_complete.json'),
        cohort=receipt(cohort_path), prediction_roots={k: str(v) for k, v in roots.items()}, scenes=[j['name'] for j in cohort['uqis']],
        modality='point_3d', bins='five equal-frequency bins over observed Gaussian source contribution',
        contribution='sum of raw full-scene Gaussian projection contributions over legal training frames',
        no_fitting=True, no_model_selection=True, causal_claim=False))
    package = Path(cohort['uqis_package'])
    targets = read(package / 'evaluator/targets.json')
    records, sources = [], []
    for job in cohort['uqis']:
        scene = job['name']; specification = next(j for j in design['fields'] if j['key'] == 'uqis_' + scene)
        field, manifest = load_field(job['field'])
        xyz, observed = field.xyz.numpy().copy(), field.observed.numpy().copy()
        del field; gc.collect()
        root = Path(specification['observations']); source = read(root / 'observations.json')
        if sha256(root / 'observations.json') != manifest['provenance']['observations_sha256']:
            raise ValueError('observation mass audit source differs')
        mass = np.zeros(len(xyz), np.float64)
        for frame in source['splits']['train_frames']:
            entry = next(e for e in source['frames'] if e['frame'] == frame)
            path = root / f'view_{frame:05d}.pt'
            if sha256(path) != entry['compiled']['sha256']:
                raise ValueError('observation mass source payload changed')
            raw = torch.load(path, weights_only=True)
            op = raw['operator'].coalesce(); pixels, rows = op.indices().numpy()
            contribution = op.values().numpy() * raw['raw_mass'].numpy()[pixels]
            mass += np.bincount(rows, weights=contribution, minlength=len(xyz))
            del raw, op
        boundaries = np.quantile(mass[observed], [.2, .4, .6, .8])
        bins = np.searchsorted(boundaries, mass, side='right'); bins[~observed] = -1
        domain = next(d for d in targets['scene_domains'] if d['scene_id'] == scene)
        mesh_path = package / 'method' / domain['mesh_xyz_path']
        label_path = package / 'evaluator' / domain['mesh_instance_ids_path']
        mesh, ids = np.load(mesh_path), np.load(label_path)
        _, index = cKDTree(xyz).query(mesh, workers=4)
        distance, back = cKDTree(mesh).query(xyz, workers=4)
        row_labels = ids[back]; registered = distance <= 1e-6
        observed_mesh, mesh_bins = observed[index], bins[index]
        receipts = {a: {q['query_id']: q for q in read(p / 'receipts' / (scene + '.json'))['queries']}
                    for a, p in roots.items()}
        cases = []
        for target in [t for t in targets['targets'] if t['scene_id'] == scene]:
            qid = target['queries']['point_3d']; truth = ids == target['instance_id']
            values = {}
            for arm, directory in roots.items():
                path = directory / 'gaussians' / (qid + '.pt')
                check(dict(path=str(path), sha256=receipts[arm][qid]['gaussian_sha256']))
                prediction = torch.load(path, weights_only=True)
                selected = prediction['selected'].numpy()[index]
                candidates = prediction['candidates'].numpy()
                correct = bool(len(candidates) and registered[candidates].all()
                               and (row_labels[candidates] == target['instance_id']).all())
                strata = []
                for b in range(5):
                    relevant = truth & (mesh_bins == b)
                    count = int(relevant.sum()); retained = int((relevant & selected).sum())
                    strata.append(dict(bin=b, target_vertices=count, retained=retained,
                                       recall=None if not count else retained / count))
                seen = truth & observed_mesh
                values[arm] = dict(status=receipts[arm][qid]['status'], correct_registered_support=correct,
                    observed_target_recall=int((seen & selected).sum()) / max(int(seen.sum()), 1),
                    actual_iou=int((truth & selected).sum()) / max(int((truth | selected).sum()), 1),
                    actual_precision=int((truth & selected).sum()) / max(int(selected.sum()), 1),
                    all_target_recall=int((truth & selected).sum()) / max(int(truth.sum()), 1), strata=strata)
            cases.append(dict(query_id=qid, target_vertices=int(truth.sum()),
                              observed_target_vertices=int((truth & observed_mesh).sum()), arms=values))
        records.append(dict(scene=scene, boundaries=boundaries.tolist(),
            observed_gaussians=int(observed.sum()), gaussian_bin_counts=np.bincount(bins[observed], minlength=5).tolist(),
            queries=cases))
        sources.extend([receipt(root / 'observations.json'), receipt(mesh_path), receipt(label_path)])
        print('observation mass audit', scene, flush=True)
    if sum(len(r['queries']) for r in records) != 67:
        raise ValueError('point diagnosis denominator differs')
    summary = {}
    for arm in roots:
        cases = [q['arms'][arm] for scene in records for q in scene['queries']]
        known = [q for q in cases if q['correct_registered_support']]
        summary[arm] = dict(all_queries=len(cases), known_correct=len(known),
            known_correct_observed_target_recall=float(np.mean([q['observed_target_recall'] for q in known])),
            known_correct_strata=[dict(bin=b,
                contributing_queries=sum(q['strata'][b]['recall'] is not None for q in known),
                query_mean_recall=float(np.mean([q['strata'][b]['recall'] for q in known if q['strata'][b]['recall'] is not None])),
                pooled_recall=sum(q['strata'][b]['retained'] for q in known) / max(sum(q['strata'][b]['target_vertices'] for q in known), 1))
                for b in range(5)])
    write(out / 'report.json', dict(scenes=records, summary=summary, sources=sources,
        design=receipt(out / 'design.json'), no_model_change=True,
        interpretation='Association only; contribution is not the subset-alpha training gradient. Object geometry and viewpoint may confound it.',
        aggregation='query means on known-correct supports for diagnosis, not the scene-macro benchmark headline'))
    print(summary)


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
