"""Complete-cohort role/landmark results, actual sets, and native controls."""
import argparse
import json
from pathlib import Path
import numpy as np
from radio_gs.benchmarks.uqis.metrics import scene_clustered_bootstrap


def read(path): return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('output/referent_identity_study'))
    p.add_argument('--candidates', nargs='+', default=['target', 'relations', 'native_relations', 'relation_support'])
    a = p.parse_args()
    baseline = read(Path('output/observation_hypotheses_study/identity_diagnostics/v7.json'))
    baseline_root = Path('output/system_v7')
    results = {'v7': baseline}
    for name in a.candidates:
        results[name] = read(a.root / name / 'system/uqis/identity_scope/result.json')
    summary = {}
    for name, result in results.items():
        row = dict(modalities={}, paired=result['explicit_selection']['paired_query_robustness'])
        for modality, value in result['modalities'].items():
            selected = result['explicit_selection']['modalities'][modality]
            changes = {scene: 100*(v['iou']-baseline['explicit_selection']['modalities'][modality]['per_scene'][scene]['iou'])
                for scene, v in selected['per_scene'].items()}
            row['modalities'][modality] = dict(ap=100*value['scene_macro']['average_precision'],
                **{k: 100*v for k, v in selected['scene_macro'].items()},
                per_scene_iou_gain_pp=changes, iou_gain_pp=scene_clustered_bootstrap(changes),
                ap_gain_pp=scene_clustered_bootstrap({scene:100*(v['average_precision']-baseline['modalities'][modality]['per_scene'][scene]['average_precision'])
                    for scene, v in value['per_scene'].items()}),
                identity=result['explicit_selection']['identity_diagnosis'][modality])
        if name != 'v7':
            root = a.root / name / 'system'
            row['control_parity'] = {}
            for control in ['native_semantic', 'anchor_comembership']:
                differences = [float(np.abs(np.load(f)-np.load(baseline_root/'uqis'/control/'predictions'/f.name)).max())
                    for f in (root/'uqis'/control/'predictions').glob('*.npy')]
                row['control_parity'][control] = dict(queries=len(differences), max_absolute_difference=max(differences))
            reports = [read(f) for f in (root/'uqis').glob('*/query/registered/runs/*.json')]
            row['cost'] = dict(query_seconds_sum=sum(r['seconds'] for r in reports),
                map_mib=sum(r['map_bytes'] for r in reports)/2**20,
                source_images_read_at_query=sum(int(r['source_rgb_read']) for r in reports))
        summary[name] = row
    prepared = read(a.root/'target/queries.json')
    plans = [q['referent_plan'] for q in prepared['queries'] if q['modality']=='text']
    selected_rows={name:{q['query_id']:q for q in result['explicit_selection']['per_query']}
        for name,result in results.items()}
    cases=[dict(query_id=q['query_id'],scene_id=q['scene_id'],expression=q['expression'],
        role_plan=q['referent_plan'],methods={name:values[q['query_id']] for name,values in selected_rows.items()})
        for q in prepared['queries'] if q['modality']=='text']
    report = dict(scope='9 opened-development scenes; all 67 targets and 268 independent queries',
        completed_scene_jobs=9*len(a.candidates), completed_queries=268*len(a.candidates),
        additional_fresh_unparsed_text_query='output/referent_identity_study/fresh_query',
        results=summary, query_language_cost=prepared['language_preprocessing'],
        parsed_queries_with_relations=sum(bool(v['relations']) for v in plans),
        parsed_queries_with_unresolved_relations=sum(bool(v['unresolved']) for v in plans),
        relation_count=sum(len(v['relations']) for v in plans),
        complete_text_case_comparison=cases,
        native_coordinates_and_scope_decoder_unchanged=True,
        additional_source_photos=0, independent_test=False,
        public_benchmarks_reexecuted=False, default_changed=False)
    (a.root/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({n:{m:{k:v[k] for k in ['ap','iou','purity','coverage']}
        for m,v in r['modalities'].items()} for n,r in summary.items()}, indent=2))


if __name__ == '__main__': main()
