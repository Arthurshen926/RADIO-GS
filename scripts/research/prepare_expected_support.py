"""Prepare the fixed aggregation/paired-training study without target access."""
import json
from pathlib import Path

from radio_gs.query_field.storage import sha256


ROOT = Path('output/query_field_20261007/expected_support_v1')
CONFIG = Path('configs/query_field')


def write(path, value):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2)+'\n')


def main():
    ROOT.mkdir(exist_ok=False)
    scenes = dict(figurines=('figurines_spatial_metric', 'figurines_18views_v1', 'figurines_fitted_metric_v1'),
                  ramen=('ramen_spatial_metric', 'ramen_18views_v1', 'ramen_fitted_metric_v1'),
                  fern=('nvos_fern_physical', 'nvos_fern_v1', 'nvos_fern_v1'))
    old_nvos = json.loads((CONFIG/'nvos_fern_prediction.json').read_text())
    fits, jobs, prompts, models, files = [], [], [], [], []
    for scene, (config_name, source, parent) in scenes.items():
        for arm in ['frozen_expected', 'paired_legacy', 'paired_expected']:
            name = f'{scene}_{arm}'
            output = ROOT/name
            output.mkdir()
            (output/'observations').symlink_to((ROOT.parent/source/'observations').resolve(), target_is_directory=True)
            config = json.loads((CONFIG/f'{config_name}.json').read_text())
            aggregation = 'normalized_prototype' if arm == 'paired_legacy' else 'expected_energy'
            config['training'].update(support_policy='paired_signed', support_aggregation=aggregation)
            config['purpose'] = 'Fixed expected-energy aggregation and matched paired source training'
            path = CONFIG/f'{name}.json'
            write(path, config)
            files.append(path)
            model = dict(name=name, scene=scene, arm=arm, aggregation=aggregation,
                         config=str(path), output=str(output), field=str(output/'models/trained'),
                         parent_field=str(ROOT.parent/parent/'models/trained'))
            models.append(model)
            if arm != 'frozen_expected':
                fits.append(model)
            if scene == 'fern':
                for mode in ['positive', 'signed']:
                    job = next(j.copy() for j in old_nvos['jobs'] if j['name'] == f'trained_{mode}_default')
                    job.update(name=f'{name}_{mode}', field=model['field'])
                    prompts.append(job)
            else:
                jobs.append(dict(name=name, scene=scene, arm=arm, field=model['field'],
                                 reference_prediction=f'output/query_field_20261007/spatial_metric_lerf_v1/{scene}_fitted_metric',
                                 evaluation_scene=f'output/system_map_rebuild/{scene}/complete/scene.json'))
    protocol = 'docs/experiments/2026-10-07-expected-support-protocol.md'
    study = CONFIG/'expected_support_study.json'
    write(study, dict(parent_commit='b3ea675', models=models, fits=fits, jobs=jobs,
                      protocol=protocol, source_query_labels_used=False, parameter_search=False,
                      independent_final_test=False))
    old_nvos['jobs'] = prompts
    old_nvos['prediction_policy'] = 'default scope; fixed expected-energy and paired source-training study'
    prompt_path = CONFIG/'expected_support_nvos.json'
    write(prompt_path, old_nvos)
    scoring = json.loads((CONFIG/'nvos_fern_scoring.json').read_text())
    scoring['prediction_config'] = str(prompt_path)
    score_path = CONFIG/'expected_support_nvos_scoring.json'
    write(score_path, scoring)
    files += [study, prompt_path, score_path, Path(protocol)]
    files += list(Path('radio_gs/query_field').glob('*.py'))
    files += list(Path('scripts/research').glob('*expected_support*.py'))
    files += [Path('tests/test_query_field_support_energy.py')]
    write(ROOT/'launch.json', dict(parent_commit='b3ea675', code_config_protocol=[
        dict(path=str(p), sha256=sha256(p)) for p in sorted(set(files))]))
    print(json.dumps(models, indent=2))


if __name__ == '__main__':
    main()
