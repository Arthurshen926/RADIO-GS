"""Summarize complete diagnostic cohorts with explicit conditional denominators."""
from pathlib import Path
import numpy as np

from scripts.research.attribute_query_field_architecture import OUT,COHORT,BASE,read,write,macro
from radio_gs.query_field.storage import sha256


def main():
    cohort=read(COHORT);inventory=read(OUT/'inventory.json')['fields']
    scan=read(OUT/'scannet.json')['scenes'];lerf=read(OUT/'lerf.json')
    uq=[x for p in sorted((OUT/'uqis').glob('*.json')) for x in read(p)['queries']]
    nv=[read(p) for p in sorted((OUT/'nvos').glob('*.json'))]
    fu=[read(p) for p in sorted((OUT/'fusion').glob('*.json'))]
    scope=[x for p in sorted((OUT/'scope_uqis').glob('*.json')) for x in read(p)['queries']]
    nvscope=[read(p) for p in sorted((OUT/'scope_nvos').glob('*.json'))]
    assert (len(inventory),len(scan),len(uq),len(nv),len(fu),len(scope),len(nvscope))==(30,10,268,8,30,268,8)
    assert len(lerf['query_views'])==208
    uqsummary={}
    for modality in ['text','image','point_2d','point_3d']:
        rows=[r for r in uq if r['modality']==modality]
        supported=[r for r in rows if r['status']=='ok'];correct=[r for r in supported if r['top_candidate_target']]
        values=macro(rows,['iou','precision','recall','target_observed_fraction','observed_target_recall'])
        for r in rows:
            r['target_unobserved_fraction']=r['fn_unobserved']/(r['tp']+r['fn'])
            r['target_observed_missed_fraction']=r['fn_observed']/(r['tp']+r['fn'])
        values.update(macro(rows,['target_unobserved_fraction','target_observed_missed_fraction']))
        values.update(queries=len(rows),supported=len(supported),top_candidate_target_count=len(correct),
            any_candidate_target_count=sum(r['any_candidate_target'] for r in supported),
            target_associated_conditional_scene_count=len({r['scene'] for r in correct}),
            target_associated_conditional=macro(correct,['iou','precision','recall']),
            pooled_false_negative_unobserved_share=sum(r['fn_unobserved'] for r in rows)/sum(r['fn'] for r in rows))
        if modality in {'text','image'}:
            values.update(macro(rows,['oracle_candidate_iou']),candidate_masks_evaluated=sum(len(r['alternatives']) for r in rows),
                          maximum_replay_error=max(r['replay_max_abs_error'] for r in rows))
        part=[r for r in scope if r['modality']==modality and r['supported']]
        keys=['target_unobserved','target_observed_outside_envelope','target_observed_inside_envelope_missed','target_selected']
        normalized=[dict(scene=r['scene'],**{k:r[k]/r['target_elements'] for k in keys}) for r in part]
        values['scope_bound_supported_only']=dict(queries=len(part),scene_count=len({r['scene'] for r in part}),
            scene_macro=macro(normalized,keys),
            target_vertex_pooled={k:sum(r[k] for r in part)/sum(r['target_elements'] for r in part) for k in keys},
            effective_support_majority_target_count=sum(r['effective_target_support_mass']>=.5 for r in part),
            effective_support_zero_target_count=sum(r['effective_target_support_mass']==0 for r in part))
        uqsummary[modality]=values
    scansummary={}
    for split in ['19','15','10']:
        rows=[dict(scene=r['scene'],**r['splits'][split]) for r in scan]
        scansummary[split]=macro(rows,['native_miou','member_miou','perfect_observed_label_miou',
            'observed_valid_fraction','all_member_below_half_fraction','member_top_tie_fraction','member_top_margin_median'])
        keys=['both_correct','native_correct_member_wrong','native_wrong_member_correct','both_wrong_observed']
        scansummary[split]['observed_vertex_transitions']={k:sum(r[k] for r in rows) for k in keys}
    fusion_scenes=[]
    for record in fu:
        regions=[r for f in record['frames'] for r in f['regions']]
        fusion_scenes.append(dict(scene=record['scene'],family=record['family'],regions=len(regions),
            **{k:float(np.mean([r[k] for r in regions])) for k in ['teacher_ap','field_ap','observed_positive_fraction']},
            **{k+'_cosine':float(np.mean([f['native_direction'][k]['cosine_mean'] for f in record['frames']])) for k in ['semantic','local']}))
    fusion_family={family:macro([r for r in fusion_scenes if r['family']==family],
        ['teacher_ap','field_ap','observed_positive_fraction','semantic_cosine','local_cosine']) for family in ['lerf','nvos','scannet','uqis']}
    nvmacro={arm:{k:float(np.mean([r['metrics'][arm][k] for r in nv])) for k in ['iou','precision','recall']}
             for arm in ['positive_only','signed','actual_subset']}
    v10=read('configs/pipeline/system_v10.json');budgets={}
    for family in ['scannet','nvos','uqis']:
        records=[]
        for job in cohort[family]:
            if family=='uqis':
                old=next(x for x in v10['jobs'] if x['benchmark']=='uqis' and x['name']==job['name'])
                p=Path(old['scene']);data=read(p);count=data['source_count'];source='used map manifest'
            else:
                p=Path('configs/consolidated/scannet_native' if family=='scannet' else 'configs/pipeline/nvos_all_sources')/(job['name']+'.scene.json')
                data=read(p);count=len(data['source_frames']);source='declared source authority; not claimed actual cache admission count'
            records.append(dict(scene=job['name'],v10_source_frames=count,new_training_frames=12 if job['name']=='fern' else 18,
                                evidence=str(p),sha256=sha256(p),source_count_interpretation=source))
        budgets[family]=records
    evidence=[dict(path=str(p),sha256=sha256(p)) for p in sorted(OUT.rglob('*.json'))]
    evidence += [dict(path=str(p),sha256=sha256(p)) for p in [
        Path('scripts/research/attribute_query_field_architecture.py'),Path('scripts/research/bound_query_field_scope.py'),
        Path('scripts/research/summarize_query_field_attribution.py'),
        Path('docs/experiments/2026-10-08-architecture-attribution-protocol.md'),
        Path('docs/experiments/2026-10-08-scope-bound-protocol.md')]]
    report=dict(schema='radio_gs.architecture_attribution.v1',parent='c6a7730',diagnostic_only=True,
        baseline_predictions_changed=False,new_method_promoted=False,independent_final_test=False,
        denominators=dict(fields=30,lerf_query_views=208,scannet_rooms=10,nvos_cases=8,uqis_queries=268,uqis_targets=67,
                          fusion_development_views=60,source_partial_regions=sum(r['regions'] for r in fusion_scenes)),
        scannet=scansummary,uqis=uqsummary,nvos=dict(macro=nvmacro,per_scene=nv,scope_bounds=nvscope),
        lerf=lerf,fusion=dict(families=fusion_family,scenes=fusion_scenes),training_inventory=inventory,
        source_budget_comparison=budgets,evidence=evidence,
        caveats=['These stage deltas are not additive causal components of the v10 gap.',
                 'Candidate oracles, observed-label ceilings and cosine bounds are privileged diagnostics, not predictions.',
                 'Fusion probe combines source fusion, observation coverage and projection on two held-out source views.',
                 'Conditional results retain their own counts and do not replace complete-cohort accuracy.',
                 'Teacher, source budget, source access and extra 3D supervision differ from v10.'])
    write(OUT/'summary.json',report)
    write('docs/experiments/2026-10-08-architecture-attribution-results.json',report)
    import json
    print(json.dumps(dict(scannet=scansummary,uqis=uqsummary,nvos=nvmacro,fusion=fusion_family,denominators=report['denominators']),indent=2))


if __name__=='__main__':main()
