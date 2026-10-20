"""Protocol regressions for the study ledger, independent of private datasets."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from radio_gs.response_study.registry import (
    REGISTRY, check_aggregates, check_cohort, check_seal, provider_audit,
    replay_plan, result_rows, sha, source_availability, source_index,
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_cohort_rejects_duplicate_missing_and_changed_access():
    jobs = [dict(benchmark='uqis', name='a', access='map-only'),
            dict(benchmark='uqis', name='b', access='map-only')]
    check_cohort(jobs[::-1], jobs, {'uqis': 2})
    for changed in [jobs[:1], [jobs[0], jobs[0]],
                    [jobs[0], dict(jobs[1], access='source-assisted')]]:
        with pytest.raises(ValueError):
            check_cohort(changed, jobs, {'uqis': 2})


def test_seal_detects_tampered_report_even_when_aggregate_is_unchanged(tmp_path):
    study = tmp_path / 'study'
    job = dict(benchmark='lerf', name='fixture')
    report = study / 'run_lerf_fixture/prediction.json'
    write(report, dict(job=job, ground_truth_read=False))
    seal = study / 'run/predictions_complete.json'
    write(seal, dict(jobs=1, all_five_present=True, no_target_gt_read=True,
                     reports={str(report): sha(report)}))
    write(study / 'run/evaluation_manifest.json', dict(prediction_seal_sha256=sha(seal),
                                                     all_predictions_checked=True))
    check_seal(tmp_path, study, 'run', [job])
    write(report, dict(job=job, ground_truth_read=True))
    with pytest.raises(ValueError, match='altered prediction report'):
        check_seal(tmp_path, study, 'run', [job])


def test_source_receipts_do_not_imply_source_recovery(tmp_path):
    path = tmp_path / 'kernel.py'
    path.write_text('new version')
    old_text = 'old version'
    digest = hashlib.sha256(old_text.encode()).hexdigest()
    stages = {'prediction': {'kernel.py': digest}}
    assert source_availability(tmp_path, stages, {})[0]['availability'] == 'missing_version'
    snapshot = tmp_path / 'study/old/source_snapshot.json'
    write(snapshot, {'files': {'kernel.py': {'text': old_text, 'sha256': digest}}})
    index = source_index(tmp_path / 'study')
    assert source_availability(tmp_path, stages, index)[0]['availability'] == 'archived'
    write(snapshot, {'files': {'kernel.py': {'text': 'tampered', 'sha256': digest}}})
    with pytest.raises(ValueError, match='corrupt source snapshot'):
        source_index(tmp_path / 'study')


def test_result_export_marks_compatibility_and_projection_honestly():
    experiment = dict(id='E04', phase='response_generation', arms=['diagonal'],
                      uqis_intervention=False, status='completed_not_promoted')
    rows = []
    for family, metric, frozen in [('uqis', 'image_ap', False),
                                    ('lerf', 'selected_projection_iou', True),
                                    ('lerf', 'semantic_iou', False)]:
        rows.append(dict(family=family, metric=metric, independent_set_head_frozen=frozen,
                         values_percent={'baseline': 50., 'diagonal': 50.},
                         effects={'diagonal': {'delta_pp': 0.}}))
    exported = result_rows(experiment, {'rows': rows}, 'receipt')
    assert [r['intervention_applied'] for r in exported] == [False, False, True]
    assert exported[1]['benchmark'] == 'lerf_selected_projection_2d'
    assert exported[0]['input_access'] == 'map-only'
    assert all(not r['independent_test'] for r in exported)


@pytest.fixture
def aggregates(tmp_path):
    counts = {'lerf': 4, 'nvos': 8, 'scannet': 10, 'uqis': 9}
    keys = {'lerf': ['semantic_iou', 'ap', 'selected_projection_iou'],
            'nvos': ['iou', 'ap'], 'scannet': ['miou_19', 'miou_15', 'miou_10'],
            'uqis': [m + s for m in ['text', 'image', 'point_2d', 'point_3d']
                     for s in ['_ap', '_selected_iou']]}
    reference = {f: {k: .5 for k in metrics} for f, metrics in keys.items()}
    macro = {f: {a: dict(values) for a in ['baseline', 'candidate']}
             for f, values in reference.items()}
    rows = [dict(job=dict(benchmark=f, name=str(i)), metrics=copy.deepcopy(macro[f]))
            for f in ['lerf', 'nvos', 'scannet'] for i in range(counts[f])]
    feedback = {'rows': [dict(family=f, metric=k,
        independent_set_head_frozen=k == 'selected_projection_iou' or k.endswith('_selected_iou'),
        values_percent={'baseline': 50., 'candidate': 50.}, effects={'candidate': {'delta_pp': 0.}})
        for f, metrics in keys.items() for k in metrics]}
    for name, key in [('fixed_selections', 'iou'), ('baseline', 'average_precision'),
                      ('candidate', 'average_precision')]:
        write(tmp_path / f'uqis_{name}.json', dict(query_count=268, scene_count=9,
            modalities={m: dict(scene_macro={key: .5}, per_scene={str(i): {key: .5} for i in range(9)})
                        for m in ['text', 'image', 'point_2d', 'point_3d']}))
    return (tmp_path, dict(run='fixture', arms=['candidate'], uqis_intervention=False),
            dict(macro=macro, rows=rows), feedback, reference, counts)


def test_metrics_are_recomputed_from_scene_reports(aggregates):
    check_aggregates(*aggregates)
    aggregates[2]['rows'][0]['metrics']['candidate']['semantic_iou'] = .9
    with pytest.raises(ValueError, match='source aggregation'):
        check_aggregates(*aggregates)


def test_feedback_cannot_silently_drop_a_metric(aggregates):
    aggregates[3]['rows'].pop()
    with pytest.raises(ValueError, match='feedback metrics missing'):
        check_aggregates(*aggregates)


def test_reference_cannot_be_changed_to_make_a_gain(aggregates):
    aggregates[2]['macro']['lerf']['baseline']['semantic_iou'] = .4
    with pytest.raises(ValueError, match='changed baseline'):
        check_aggregates(*aggregates)


@pytest.fixture
def replay_repo(tmp_path):
    job = dict(benchmark='lerf', name='fixture')
    (tmp_path / 'runner.py').write_text('# frozen runner\n')
    write(tmp_path / 'recipe.json', {'jobs': [job]})
    write(tmp_path / 'study/previous/manifest.json', dict(jobs=[job],
        source_hashes={'runner.py': sha(tmp_path / 'runner.py')}))
    write(tmp_path / REGISTRY, dict(study_root='study', recipe='recipe.json', cohort={'lerf': 1},
        storage={}, experiments=[dict(id='E04', run='previous', launch_enabled=True,
        launch_note='bounded', runner='runner.py', runner_args=[], prediction_storage='volatile_released')]))
    return tmp_path


def test_replay_plan_is_read_only_and_requires_unique_safe_name(replay_repo):
    before = set(replay_repo.rglob('*'))
    plan = replay_plan('E04', 'new_run', replay_repo)
    assert '--run' in plan['command'] and not plan['executed']
    assert set(replay_repo.rglob('*')) == before
    for name in ['../escape', 'previous', 'a/b', 'name;command']:
        with pytest.raises(ValueError):
            replay_plan('E04', name, replay_repo)
    (replay_repo / 'study/new_run_lerf_fixture').mkdir()
    with pytest.raises(ValueError, match='collides'):
        replay_plan('E04', 'new_run', replay_repo)


def test_replay_blocks_changed_code_and_disabled_writer(replay_repo):
    (replay_repo / 'runner.py').write_text('# modified runner\n')
    with pytest.raises(ValueError, match='historical runner code differs'):
        replay_plan('E04', 'new', replay_repo)
    registry = json.loads((replay_repo / REGISTRY).read_text())
    registry['experiments'][0].update(launch_enabled=False, launch_note='old dense writer disabled')
    write(replay_repo / REGISTRY, registry)
    with pytest.raises(ValueError, match='old dense writer disabled'):
        replay_plan('E04', 'new', replay_repo)


def test_targeted_replay_is_not_mislabeled_as_full_inference(replay_repo):
    registry = json.loads((replay_repo / REGISTRY).read_text())
    item = registry['experiments'].pop()
    item.update(id='Q01')
    registry['targeted_studies'] = [item]
    write(replay_repo / REGISTRY, registry)
    manifest = json.loads((replay_repo / 'study/previous/manifest.json').read_text())
    manifest['intervention_jobs'] = manifest['jobs'][:1]
    write(replay_repo / 'study/previous/manifest.json', manifest)
    plan = replay_plan('Q01', 'new_targeted_run', replay_repo)
    assert plan['jobs'] == 1 and not plan['executed']


def test_provider_audit_follows_configuration_instead_of_module_name(tmp_path):
    for name in ['direct_anchor_membership', 'registered_observation_query']:
        path = tmp_path / f'radio_gs/querying/{name}.py'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# fixture')
    jobs = []
    for i in range(9):
        path = f'maps/{i}.json'
        write(tmp_path / path, dict(direct_anchor_membership={'applies_to': ['point_2d', 'point_3d']},
                                   inferred_anchor_membership=['image']))
        jobs.append(dict(benchmark='uqis', name=str(i), scene=path))
    audit = provider_audit(tmp_path, {'jobs': jobs})
    assert not audit['query_conditioned_membership_candidate_active']
    changed = json.loads((tmp_path / 'maps/0.json').read_text())
    changed['query_conditioned_membership'] = {'applies_to': ['image']}
    write(tmp_path / 'maps/0.json', changed)
    with pytest.raises(ValueError, match='optional membership candidate became active'):
        provider_audit(tmp_path, {'jobs': jobs})


def test_inherited_estimated_anchor_is_still_forced_positive():
    # Documents the actual v10 limitation. A deliberate change needs new controls,
    # a new method attribution and this expectation updated together.
    import numpy as np
    from scipy import sparse
    from radio_gs.querying.direct_anchor_membership import DirectAnchorMembership
    model = DirectAnchorMembership.__new__(DirectAnchorMembership)
    model.evidence = SimpleNamespace(
        membership=sparse.csr_matrix(np.ones((3, 1))),
        xyz=np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]),
        encode=lambda anchor, vertices: np.zeros((len(vertices), 1)))
    model.saved = dict(threshold=.35, condition='fixture', estimator=SimpleNamespace(
        predict_proba=lambda features: np.tile([.9, .1], (len(features), 1))))
    probability, selected, detail = model.infer_region(0)
    assert detail['anchor_identity_guaranteed'] is False
    assert probability.tolist() == pytest.approx([.1, 1., .1])
    assert selected.tolist() == [False, True, False]
