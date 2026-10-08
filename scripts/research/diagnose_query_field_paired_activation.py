"""Post-score complete point-query audit of relation availability and decisions.

Evaluator labels enter this report only. No fitting, model or threshold changes.
Replay the actual finite-evidence rule against saved protected predictions to
separate support admission, target evidence, conflict and selected-set changes.
"""
import gc
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch

from radio_gs.query_field.contracts import Support
from radio_gs.query_field.paired_evidence import PairedEvidence
from radio_gs.query_field.soft_paired_evidence import SoftPairedEvidence
from scripts.research.run_query_field_soft_functions import ROOT, HARD, ARMS, read, write, receipt, check
from scripts.research.run_query_field_protected import ROOT as PROTECTED


def load_relation(directory, cls):
    manifest = read(directory / 'manifest.json')
    check(dict(path=str(directory / 'field.pt'), sha256=manifest['files']['field.pt']['sha256']))
    state = torch.load(directory / 'field.pt', map_location='cpu', weights_only=True)
    relation = cls({k.removeprefix('evidence.'): v for k, v in state.items() if k.startswith('evidence.')}, len(state['xyz']))
    return relation, state['xyz'], state['observed'], manifest['spec']


@torch.no_grad()
def main():
    read(ROOT / 'scoring_complete.json'); read(HARD / 'summary.json')
    out = ROOT / 'diagnostics/paired_activation'; out.mkdir(parents=True, exist_ok=False)
    cohort = read(ROOT / ARMS[0] / 'cohort.json'); package = Path(cohort['uqis_package'])
    targets = read(package / 'evaluator/targets.json')
    roots = dict(protected_training=PROTECTED / 'protected_training/predictions/uqis',
        paired_function=HARD / 'paired_function/predictions/uqis',
        soft_paired_function=ROOT / ARMS[0] / 'predictions/uqis')
    write(out / 'design.json', dict(code=receipt(Path(__file__)), inputs=receipt(ROOT / 'predictions_complete.json'),
        targets=receipt(package / 'evaluator/targets.json'), all_point_queries=True,
        post_score=True, no_fitting=True, no_model_selection=True, causal_claim=False))
    records, evidence = [], []
    for job in cohort['uqis']:
        scene = job['name']; directory = Path(job['field'])
        soft, xyz, observed, spec = load_relation(directory, SoftPairedEvidence)
        hard, hxyz, hobs, hspec = load_relation(HARD / 'fits/paired_function' / ('uqis_' + scene) / 'field', PairedEvidence)
        assert torch.equal(xyz, hxyz) and torch.equal(observed, hobs)
        assert spec['default_scale'] == hspec['default_scale']
        domain = next(d for d in targets['scene_domains'] if d['scene_id'] == scene)
        mesh_path = package / 'method' / domain['mesh_xyz_path']
        label_path = package / 'evaluator' / domain['mesh_instance_ids_path']
        mesh, labels = np.load(mesh_path), np.load(label_path)
        _, index = cKDTree(xyz.numpy()).query(mesh, workers=4)
        distance, back = cKDTree(mesh).query(xyz.numpy(), workers=4)
        row_labels = labels[back]; registered = distance <= 1e-6
        observed_mesh = observed.numpy()[index]
        reports = {a: {q['query_id']: q for q in read(p / 'receipts' / (scene + '.json'))['queries']} for a, p in roots.items()}
        cases = []
        for target in [t for t in targets['targets'] if t['scene_id'] == scene]:
            qid = target['queries']['point_3d']; truth = labels == target['instance_id']; seen = truth & observed_mesh
            predictions = {}
            for a, root in roots.items():
                path = root / 'gaussians' / (qid + '.pt')
                check(dict(path=str(path), sha256=reports[a][qid]['gaussian_sha256']))
                predictions[a] = torch.load(path, map_location='cpu', weights_only=True)
            base = predictions['protected_training']; ids = base['candidates']
            correct = bool(len(ids) and registered[ids].all() and (row_labels[ids] == target['instance_id']).all())
            for pred in predictions.values():
                assert torch.equal(ids, pred['candidates']) and torch.equal(base['candidate_weights'], pred['candidate_weights'])
            case = dict(query_id=qid, correct_registered_support=correct, target_vertices=int(truth.sum()),
                observed_target_vertices=int(seen.sum()), unsupported=not len(ids), arms={})
            coefficients = None
            for arm, relation in [('paired_function', hard), ('soft_paired_function', soft)]:
                if len(ids):
                    support = Support(ids, base['candidate_weights']); coefficient = relation.coefficients(support, spec['default_scale'])
                    if coefficients is not None and not torch.equal(coefficients, coefficient):
                        raise ValueError('target-only extension changed actual query association')
                    coefficients = coefficient
                    pos, neg = relation.masses(support, spec['default_scale'])
                    views = int(torch.unique(relation.views[coefficient > 0]).numel())
                else:
                    pos = torch.zeros(len(xyz)); neg = pos.clone(); views = 0
                pred = predictions[arm]; expected = (base['membership'] + pos) / (1 + pos + neg)
                expected[~observed] = 0; expected[ids] = 1
                error = float((expected - pred['membership']).abs().max())
                if error > 1e-5: raise ValueError('saved point prediction does not match functional update')
                p, n = pos.numpy()[index], neg.numpy()[index]
                known = (p + n) > 0; selected = pred['selected'].numpy()[index]
                potential = ((relation.positive_pointer.diff() > 0) | (relation.negative_pointer.diff() > 0)).numpy()[index]
                masks = dict(potential=potential, activated=known, conflicting=(p > 0) & (n > 0),
                    positive_dominant=p > n, negative_dominant=n > p, equal_nonzero=known & (p == n),
                    missed_unknown=(~selected) & ~known, missed_positive=(~selected) & (p > n),
                    missed_nonpositive=(~selected) & known & (p <= n))
                fractions = {k: int((seen & v).sum()) / max(int(seen.sum()), 1) for k, v in masks.items()}
                changes = {}
                for reference in ['protected_training', 'paired_function']:
                    before = predictions[reference]['selected'].numpy()[index]
                    changes[reference] = dict(
                        gained_target_vertices=int((selected & ~before & truth).sum()),
                        lost_target_vertices=int((before & ~selected & truth).sum()),
                        gained_other_vertices=int((selected & ~before & ~truth).sum()),
                        removed_other_vertices=int((before & ~selected & ~truth).sum()))
                case['arms'][arm] = dict(admitted_views=views, prediction_replay_max_error=error,
                    observed_target_fractions=fractions, changes=changes)
            cases.append(case)
        records.append(dict(scene=scene, queries=cases))
        evidence.extend([receipt(directory / 'manifest.json'), receipt(mesh_path), receipt(label_path)])
        del soft, hard, xyz, hxyz, observed, hobs; gc.collect()
        print('paired activation audit', scene, flush=True)
    all_cases = [q for r in records for q in r['queries']]
    known = [q for q in all_cases if q['correct_registered_support']]
    if len(all_cases) != 67 or len(known) != 59: raise ValueError('complete point cohort differs')
    summary = {}
    for arm in ['paired_function', 'soft_paired_function']:
        values = [q['arms'][arm] for q in known]
        summary[arm] = dict(known_correct_queries=len(known),
            queries_with_admitted_views=sum(q['admitted_views'] > 0 for q in values),
            mean_admitted_views=float(np.mean([q['admitted_views'] for q in values])),
            observed_target_fractions_query_mean={k: float(np.mean([q['observed_target_fractions'][k] for q in values]))
                for k in values[0]['observed_target_fractions']},
            pooled_changes={a: {k: sum(q['changes'][a][k] for q in values) for k in values[0]['changes'][a]}
                for a in values[0]['changes']})
    write(out / 'report.json', dict(queries=67, known_correct=59, scenes=records, summary=summary,
        exact_query_associations=True, no_model_change=True,
        prediction_replay_max_error=max(q['arms'][a]['prediction_replay_max_error'] for q in all_cases for a in q['arms']),
        aggregation='Known-correct query means; transition counts pooled separately, not benchmark macro scores',
        interpretation='Post-score decomposition, not proof evidence polarity is ground truth or a unique causal attribution',
        design=receipt(out / 'design.json'), evidence=evidence))
    print(summary, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4); main()
