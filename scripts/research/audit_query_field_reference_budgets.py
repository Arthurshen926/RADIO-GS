"""Manifest-only source budget audit; no benchmark labels or model changes."""
from pathlib import Path

from scripts.research.run_query_field_mixture_functions import ROOT, ARMS, read, write, receipt


def main():
    config = Path('configs/pipeline/system_v10.json'); legacy = read(config)
    cohort = read(ROOT / ARMS[0] / 'cohort.json'); design = read(ROOT / 'design.json')
    fields = {j['key']: j for j in design['fields']}
    old = {(j['benchmark'], j['name']): j for j in legacy['jobs']}
    records, evidence = [], [receipt(config), receipt(ROOT / 'design.json')]
    for family in ['lerf', 'nvos', 'scannet', 'uqis']:
        for job in cohort[family]:
            name = job.get('name', job.get('scene')); reference = old[family, name]
            key = Path(job['field']).parent.name; field = fields[key]
            source_path = Path(field['observations']) / 'observations.json'; source = read(source_path)
            old_path = Path(reference['scene']); manifest = read(old_path)
            old_frames = set(manifest['source_frames'])
            train = set(source['splits']['train_frames']); development = set(source['splits']['development_frames'])
            construction = read(Path(field['parent']).parent / 'construction.json')
            valid = set(construction['source_frames'])
            if not valid <= train or train & development: raise ValueError('source split/trace invalid')
            records.append(dict(family=family, query_job=name, field_key=key,
                v10_declared_source_views=len(old_frames), current_training_views=len(train),
                current_development_views=len(development), current_relation_views=len(valid),
                current_training_views_omitted_by_relations=sorted(train - valid),
                current_train_in_v10=len(train & old_frames), current_train_not_in_v10=sorted(train - old_frames),
                v10_frames_outside_current_training=len(old_frames - train),
                v10_access=reference['access'], current_access='standalone field + current query',
                v10_manifest=receipt(old_path), current_observations=receipt(source_path)))
            evidence.extend([receipt(old_path), receipt(source_path), receipt(Path(field['parent']).parent / 'construction.json')])
    summary = {}
    for family in ['lerf', 'nvos', 'scannet', 'uqis']:
        rows = [r for r in records if r['family'] == family]
        unique = {r['field_key']: r for r in rows}
        for r in rows:
            other = unique[r['field_key']]
            if any(r[k] != other[k] for k in ['v10_declared_source_views', 'current_training_views', 'current_relation_views']):
                raise ValueError('shared field has inconsistent source comparison')
        summary[family] = dict(query_jobs=len(rows), distinct_fields=len(unique),
            **{k: sum(r[k] for r in unique.values()) for k in
               ['v10_declared_source_views', 'current_training_views', 'current_development_views',
                'current_relation_views', 'current_train_in_v10', 'v10_frames_outside_current_training']},
            training_views_omitted_by_relations=sum(len(r['current_training_views_omitted_by_relations']) for r in unique.values()))
    write(ROOT / 'diagnostics/reference_budgets.json', dict(records=records, summary=summary,
        source=receipt(Path(__file__)), evidence=evidence, benchmark_labels_accessed=False,
        model_change=False, causal_estimate=False, same_resource_comparison=False,
        count_basis='distinct current fields; horns signed query jobs share one field',
        limitation='manifest-declared source views; teacher, raster, query assistance and extra supervision also differ'))
    print(summary, flush=True)


if __name__ == '__main__': main()
