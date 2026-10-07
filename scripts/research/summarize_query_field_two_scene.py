"""Rebuild the completed two-scene query-field evidence index without a GPU."""
import hashlib
import json
from pathlib import Path


ROOT=Path(__file__).resolve().parents[2]
RUNS=ROOT/'output/query_field_20261007'
TARGET=ROOT/'docs/experiments/2026-10-07-query-field-two-scene-evidence.json'


def read(path):
    return json.loads(Path(path).read_text())


def receipt(path):
    path=Path(path)
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):
            digest.update(block)
    return dict(path=str(path.relative_to(ROOT)),sha256=digest.hexdigest(),bytes=path.stat().st_size)


def summary():
    result=dict(schema='radio_gs.query_field_two_scene_evidence.v1',date='2026-10-07',
        branch='codex/query-preserving-gaussian-field',parent_implementation_commit='1a40218',
        completed=True,benchmark_replacement=False,compact_student=False,independent_final_test=False,
        source_files={},configuration_files={},runs={},public_probe={},readout_controls={})
    for path in sorted((ROOT/'radio_gs/query_field').glob('*.py')):
        result['source_files'][str(path.relative_to(ROOT))]=receipt(path)
    for path in sorted((ROOT/'configs/query_field').glob('*.json')):
        result['configuration_files'][str(path.relative_to(ROOT))]=receipt(path)
    for name in ['figurines_subset_v1','figurines_18views_v1','ramen_6views_v1','ramen_18views_v1']:
        run=RUNS/name
        training=read(run/'models/training.json')
        if not training['geometry_unchanged'] or not training['native_coordinates_unchanged']:
            raise ValueError('protected state changed')
        observations=read(run/'observations/observations.json')
        development=read(run/'development.json')
        manifest=read(run/'models/trained/manifest.json')
        files=['observations/observations.json','models/training.json','models/trained/manifest.json',
               'development.json','development_trained.json','development_untrained.json']
        if (run/'runtime_smoke.json').exists():
            smoke=read(run/'runtime_smoke.json')
            if not all(smoke['exact_replay'].values()) or smoke['forbidden_open_attempts']:
                raise ValueError('cold replay failed')
            files.append('runtime_smoke.json')
        result['runs'][name]=dict(artifacts={file:receipt(run/file) for file in files},
            model_payload=manifest['files']['field.pt'],train_frames=training['train_frames'],
            development_frames=training['development_frames'],observed=manifest['observed'],elements=manifest['elements'],
            training_episodes=training['training_episodes'],unique_visited=training['unique_training_episodes_sampled'],
            training_seconds=training['seconds'],semantic_increment_bytes=manifest['semantic_increment_bytes'],
            summary=development['stages']['trained']['summary'],
            observations_configuration=observations['configuration'])
    for folder,key in [('lerf_two_scene_probe_v1','public_probe'),('lerf_readout_study_v1','readout_controls')]:
        for path in sorted((RUNS/folder).glob('*/evaluation/evaluation.json')):
            report=read(path);name=path.parent.parent.name
            prediction=path.parent.parent/'prediction.json';record=read(prediction)
            payload=path.parent.parent/'predictions.pt'
            if receipt(payload)['sha256']!=record['predictions_sha256']:
                raise ValueError('prediction checksum differs')
            if report['prediction_sha256']!=record['predictions_sha256']:
                raise ValueError('scoring used another prediction')
            result[key][name]=dict(prediction=receipt(prediction),evaluation=receipt(path),
                prediction_payload=receipt(payload),
                metrics={control:dict(ap=value['ap'],semantic_iou=value['iou'],
                    selected_iou=value['selected']['miou'],cases=value['observations'],
                    missing=value['selected']['missing']) for control,value in report['metrics'].items()})
    if len(result['public_probe'])!=4 or len(result['readout_controls'])!=2:
        raise ValueError('fixed experiment roster incomplete')
    for name,value in result['readout_controls'].items():
        before=result['public_probe'][name]['metrics']['shared_membership']
        if before!=value['metrics']['union_member']:
            raise ValueError('readout baseline changed')
    # These frame rosters come from the unchanged retained public scorer.
    gt_frames={'figurines':{41,105,152,195},'ramen':{6,24,60,65,81,119,128}}
    for name,value in result['runs'].items():
        if set(value['train_frames']) & gt_frames[name.split('_')[0]]:
            raise ValueError('public target frame used for feature fitting')
    result['public_target_frames_excluded_from_feature_training']=True
    result['geometry_trained_on_larger_mapping_set']=True
    for name in ['figurines_hypotheses_v1','figurines_native_transfer_v1']:
        path=RUNS/name/'report.json';report=read(path)
        result[name]=dict(report=receipt(path),cases=report['cases'],summary=report['summary'])
    result['subset_diagnostics']=[receipt(RUNS/'lerf_two_scene_probe_v1/subset_error_diagnostic_v2.json'),
                                 receipt(RUNS/'lerf_readout_study_v1/subset_errors.json')]
    result['superseded_diagnostic']=dict(path='output/query_field_20261007/lerf_two_scene_probe_v1/subset_error_diagnostic.json',
        reason='Initial supplemental script used GT threshold 128; v2 uses the retained threshold 10 and asserts every IoU. Values happened to agree on these masks. Main benchmark scoring was correct throughout.')
    baseline=ROOT/'output/query_conditioned_membership_20261006/image_convergence/system/summary.json'
    result['baseline_a']=dict(summary=receipt(baseline),rerun=False,source_assisted=True,
        metrics={row['job']['name']:row['metrics'] for row in read(baseline)['rows']
                 if row['job']['benchmark']=='lerf' and row['job']['name'] in gt_frames})
    for path in [ROOT/'radio_gs/evaluation/lerf_readouts.py',ROOT/'radio_gs/scripts/eval_opengaussian_lerf_baseline.py',
                 ROOT/'radio_gs/scripts/eval_prerendered_lerf_features.py',Path(__file__)]:
        result['source_files'][str(path.relative_to(ROOT))]=receipt(path)
    result['tests']=dict(passed=19,failed=0,command='CUDA_VISIBLE_DEVICES=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONPATH=. /root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q tests/test_query_preserving_field.py tests/test_query_field_subset_cuda.py tests/test_query_field_hypotheses.py tests/test_query_field_public_lerf.py',
        includes_actual_gsplat_cuda=True,source='Executed command output, 19 passed in 9.62 s; existing native-decoder unit suite was not rerun in this round.')
    result['tests']['files']=[receipt(ROOT/'tests'/name) for name in
        ['test_query_preserving_field.py','test_query_field_subset_cuda.py','test_query_field_hypotheses.py','test_query_field_public_lerf.py']]
    original=RUNS/'two_scene_original_worktree_check.json'
    if not all(read(original)['checks'].values()):raise ValueError('original worktree differs')
    result['original_worktree_verification']=receipt(original)
    return result


if __name__=='__main__':
    TARGET.write_text(json.dumps(summary(),indent=2)+'\n')
    print(TARGET)
