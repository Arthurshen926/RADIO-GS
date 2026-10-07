"""Lock source-budget-matched support fits and both benchmark prediction cohorts."""
import json
from pathlib import Path


def write(path, value):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2)+'\n')


def main():
    root = Path('output/query_field_20261007/support_training_v1')
    root.mkdir(exist_ok=False)
    config_root = Path('configs/query_field')
    scenes = dict(figurines='figurines_spatial_metric', ramen='ramen_spatial_metric', fern='nvos_fern_physical')
    sources = dict(figurines='figurines_18views_v1', ramen='ramen_18views_v1', fern='nvos_fern_v1')
    fits, lerf_jobs, nvos_jobs = [], [], []
    old_nvos = json.loads((config_root/'nvos_fern_prediction.json').read_text())
    for scene, parent_config in scenes.items():
        for policy in ['mixed_unsigned', 'mixed_signed']:
            name = f'{scene}_{policy}'
            directory = root/name
            directory.mkdir()
            (directory/'observations').symlink_to((root.parent/sources[scene]/'observations').resolve(), target_is_directory=True)
            config = json.loads((config_root/f'{parent_config}.json').read_text())
            config['training']['support_policy'] = policy
            config['purpose'] = 'Fixed source support distribution study; unchanged common member energy'
            path = config_root/f'{name}.json'
            write(path, config)
            fits.append(dict(name=name, scene=scene, policy=policy, config=str(path), output=str(directory)))
            if scene == 'fern':
                for mode in ['positive', 'signed']:
                    job = next(j.copy() for j in old_nvos['jobs'] if j['name'] == f'trained_{mode}_default')
                    job.update(name=f'{name}_{mode}', field=str(directory/'models/trained'))
                    nvos_jobs.append(job)
            else:
                lerf_jobs.append(dict(name=name, scene=scene, arm=policy, field=str(directory/'models/trained'),
                    reference_prediction=f'output/query_field_20261007/spatial_metric_lerf_v1/{scene}_fitted_metric',
                    evaluation_scene=f'output/system_map_rebuild/{scene}/complete/scene.json'))
    # A single-policy replay verifies the common-core refactor against the prior fit.
    replay = root/'fern_single_replay'; replay.mkdir()
    (replay/'observations').symlink_to((root.parent/'nvos_fern_v1/observations').resolve(), target_is_directory=True)
    write(config_root/'support_training_study.json', dict(parent_commit='b831d03', fits=fits,
        jobs=lerf_jobs, member_intervention=dict(local_metric='none'),
        protocol='docs/experiments/2026-10-07-support-training-protocol.md',
        source_query_labels_used=False, parameter_search=False, independent_final_test=False))
    old_nvos['jobs'] = nvos_jobs
    old_nvos['prediction_policy'] = 'default physical scale; unchanged inference; fixed mixed-support training controls'
    write(config_root/'support_training_nvos.json', old_nvos)
    old_score = json.loads((config_root/'nvos_fern_scoring.json').read_text())
    old_score['prediction_config'] = str(config_root/'support_training_nvos.json')
    write(config_root/'support_training_nvos_scoring.json', old_score)
    print(json.dumps(fits, indent=2))


if __name__ == '__main__':
    main()
