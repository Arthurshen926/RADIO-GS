"""Seal the fixed signed-evidence cohort without opening target annotations."""
import json
from pathlib import Path

from radio_gs.query_field.storage import sha256


ROOT = Path('output/query_field_20261007/competitive_evidence_v1')
CONFIG = Path('configs/query_field')


def write(path, value):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2)+'\n')


def main():
    ROOT.mkdir(exist_ok=False)
    scenes = dict(figurines=('figurines_18views_v1', 'figurines_fitted_metric_v1'),
                  ramen=('ramen_18views_v1', 'ramen_fitted_metric_v1'),
                  fern=('nvos_fern_v1', 'nvos_fern_v1'))
    old_nvos = json.loads((CONFIG/'nvos_fern_prediction.json').read_text())
    models, prompts, diagnostics = [], [], []
    for scene, (source, parent) in scenes.items():
        parent_field = str(ROOT.parent/parent/'models/trained')
        observations = str(ROOT.parent/source/'observations')
        diagnostics.append(dict(scene=scene, parent_field=parent_field, observations=observations))
        for rule in ['competitive_pooled', 'competitive_rays']:
            name = f'{scene}_{rule}'
            output = ROOT/name
            output.mkdir()
            (output/'observations').symlink_to(Path(observations).resolve(), target_is_directory=True)
            models.append(dict(name=name, scene=scene, negative_rule=rule, output=str(output),
                               field=str(output/'models/trained'), parent_field=parent_field))
        if scene == 'fern':
            for rule in ['product', 'competitive_pooled', 'competitive_rays']:
                field = parent_field if rule == 'product' else str(ROOT/f'{scene}_{rule}'/'models/trained')
                for mode in ['positive', 'signed']:
                    job = next(j.copy() for j in old_nvos['jobs'] if j['name'] == f'trained_{mode}_default')
                    job.update(name=f'{rule}_{mode}', field=field)
                    prompts.append(job)
    references = dict(figurines='spatial_metric_lerf_v1/figurines_fitted_metric',
                      ramen='spatial_metric_lerf_v1/ramen_fitted_metric',
                      teatime='spatial_metric_teatime_lerf_v1/teatime_fitted_metric',
                      waldo_kitchen='waldo_kitchen_physical_lerf_v1/waldo_kitchen')
    unsigned = [dict(scene=scene, field=str(ROOT.parent/f'{scene}_fitted_metric_v1/models/trained'),
                     reference_prediction=str(ROOT.parent/reference)) for scene, reference in references.items()]
    protocol = 'docs/experiments/2026-10-07-competitive-evidence-protocol.md'
    study = CONFIG/'competitive_evidence_study.json'
    write(study, dict(parent_commit='a1a3b8e', models=models, diagnostics=diagnostics,
                      unsigned_lerf=unsigned, protocol=protocol, fits=[], parameter_search=False,
                      independent_final_test=False, negative_pixels_per_teacher_region=4))
    old_nvos['jobs'] = prompts
    old_nvos['prediction_policy'] = 'Frozen physical field; fixed pooled and ray-mixture relative signed competition'
    prompt_path = CONFIG/'competitive_evidence_nvos.json'
    write(prompt_path, old_nvos)
    scoring = json.loads((CONFIG/'nvos_fern_scoring.json').read_text())
    scoring['prediction_config'] = str(prompt_path)
    score_path = CONFIG/'competitive_evidence_nvos_scoring.json'
    write(score_path, scoring)
    files = [study, prompt_path, score_path, Path(protocol)]
    files += list(Path('radio_gs/query_field').glob('*.py'))
    files += list(Path('scripts/research').glob('*competitive_evidence*.py'))
    files += [Path('tests/test_query_field_negative_evidence.py')]
    write(ROOT/'launch.json', dict(parent_commit='a1a3b8e', code_config_protocol=[
        dict(path=str(p), sha256=sha256(p)) for p in sorted(set(files))]))
    print('sealed', ROOT, flush=True)


if __name__ == '__main__':
    main()
