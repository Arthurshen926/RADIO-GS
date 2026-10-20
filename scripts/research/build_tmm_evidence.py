"""Build paper tables from complete retained results and matched interventions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUTPUT = Path('/mnt/pool/sqy/results/RADIO-GS/output')
SYSTEM = OUTPUT / 'query_conditioned_membership_20261006/image_convergence/system'
DEST = REPO / 'paper/tmm_response'


def build():
    receipts = {}

    def read(path):
        path = Path(path)
        raw = path.read_bytes()
        receipts[str(path)] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        return json.loads(raw)

    recipe = read(REPO / 'configs/pipeline/system_v10.json')
    system = read(SYSTEM / 'summary.json')
    expected = {(j['benchmark'], j['name']): j for j in recipe['jobs']}
    actual = {(r['job']['benchmark'], r['job']['name']): r for r in system['rows']}
    if len(actual) != 31 or set(expected) != set(actual) or system['completed'] != 31:
        raise ValueError('incomplete retained system cohort')
    if not system['all_predictions_before_scoring']:
        raise ValueError('retained system lacks prediction barrier')
    for key, row in actual.items():
        for attr in ('scene', 'queries', 'access', 'task'):
            if row['job'][attr] != expected[key][attr]:
                raise ValueError('v10 job contract differs: ' + str(key))
        saved = read(SYSTEM / key[0] / key[1] / 'result.json')
        if saved['metrics'] != row['metrics']:
            raise ValueError('summary/result metrics differ')
    # Independently recompute all reported macro columns from every retained job.
    for family, metrics in system['macro'].items():
        rows = [r for (f, _), r in actual.items() if f == family]
        for metric, value in metrics.items():
            computed = sum(r['metrics'][metric] for r in rows) / len(rows)
            if abs(value - computed) > 1e-12:
                raise ValueError('macro mismatch: ' + family + '/' + metric)
    uqis = read(SYSTEM / 'uqis/identity_scope/result.json')
    if uqis['query_count'] != 268 or uqis['target_count'] != 67:
        raise ValueError('UQIS denominator differs')

    response = read(OUTPUT / 'consolidated_response_control/summary.json')
    identity = read(OUTPUT / 'conditioned_identity/summary.json')
    appearance = read(OUTPUT / 'local_appearance_study/comparison.json')
    member = read(OUTPUT / 'query_conditioned_membership_20261006/image_convergence/comparison.json')
    point_member = read(OUTPUT / 'real_direct_membership_20261006/comparison.json')
    category = read(OUTPUT / 'local_element_fusion/summary.json')
    reference_control = read(REPO / 'docs/experiments/2026-10-12-reference-proposals-results.json')
    reference_negative = read(REPO / 'docs/experiments/2026-10-15-unified-hypothesis-results.json')

    def effect(name, before, after, metric, family, scope, evidence, **extra):
        return dict(name=name, before=before, after=after, delta_pp=100*(after-before),
                    metric=metric, family=family, scope=scope, evidence=evidence, **extra)

    effects = [
        effect('Query-conditioned identity admission',
            identity['arms']['full_reference']['macro']['selected_iou'],
            identity['arms']['full_conditioned']['macro']['selected_iou'],
            'selected-set projection IoU', 'LERF', 'in-house v10 evidence mechanism',
            'conditioned_identity/summary.json', controlled='geometry/features/set solver unchanged'),
        effect('Admitted source response fusion', response['macro']['baseline']['iou'],
            response['macro']['fusion']['iou'], 'semantic IoU', 'LERF', 'in-house v10 evidence mechanism',
            'consolidated_response_control/summary.json',
            controlled='same evidence/map/selected set; output integration changes',
            cannot_attribute='entire gain to a novel fusion equation'),
        effect('Native local correspondence', appearance['results']['pooled']['modalities']['image']['iou']/100,
            appearance['results']['chamfer']['modalities']['image']['iou']/100,
            'actual instance IoU', 'UQIS image', 'in-house identity component',
            'local_appearance_study/comparison.json', controlled='same teachers/source/geometry/extent head'),
        effect('Complete-carrier membership reuse', member['metrics']['v9']['modalities']['image']['iou'],
            member['metrics']['candidate']['modalities']['image']['iou'],
            'actual instance IoU', 'UQIS image', 'in-house complete extent component',
            'query_conditioned_membership_20261006/image_convergence/comparison.json',
            interval95_pp=[100*x for x in member['effects']['image']['bootstrap95']],
            improved_scenes=member['effects']['image']['improved']),
    ]
    providers = [effect('Dual-native category probabilities',
        category['macro']['siglip_vote']['source']['19'],
        category['macro']['average_prior']['source']['19'],
        '19-class mIoU', 'ScanNet three held-out rooms', 'in-house category fusion component',
        'local_element_fusion/summary.json',
        cannot_attribute='additional native teacher capability to response refinement')]
    for modality in ('point_2d', 'point_3d'):
        comparison = point_member['matched_comparisons']['real:rendered'][modality]
        providers.append(effect('Real-observation membership '+modality,
            point_member['conditions']['rendered']['metrics'][modality]['iou'],
            point_member['conditions']['real']['metrics'][modality]['iou'],
            'actual instance IoU', 'UQIS '+modality, 'in-house complete extent component',
            'real_direct_membership_20261006/comparison.json',
            interval95_pp=[comparison['delta_pp']['lower'], comparison['delta_pp']['upper']],
            improved_scenes=comparison['improved_scenes'],
            controlled='same learner; observation distribution and counts differ',
            additional_supervision=point_member['additional_supervision']))
    m = system['macro']
    main = [
        dict(task='LERF2D', metric='semantic IoU', value=m['lerf']['semantic_iou'], count='4 scenes', mechanism='identity-admitted source response'),
        dict(task='LERF3D projection', metric='selected-set projection IoU', value=m['lerf']['selected_projection_iou'], count='same 4 scenes', mechanism='physical set readout; not true-3D GT'),
        dict(task='NVOS', metric='retained positive-reference IoU', value=m['nvos']['iou'], count='8 cases', mechanism='reference evidence and retained response solver'),
        dict(task='ScanNet OVS', metric='19-class point mIoU', value=m['scannet']['miou_19'], count='10 rooms', mechanism='dual-native probabilities and visibility vote'),
    ]
    for modality in ('text', 'image', 'point_2d', 'point_3d'):
        main.append(dict(task='UQIS '+modality, metric='actual instance IoU', value=m['uqis'][modality+'_selected_iou'],
            ap=m['uqis'][modality+'_ap'], count='67 queries / 9 rooms',
            mechanism='native identity and typed extent; points/image inherit supervised member model'))
    data = dict(schema='radio_gs.tmm_response_evidence.v1', date='2026-10-17',
        paper_focus='query-conditioned evidence fusion for 3D responses',
        method_ownership='v10 is our implementation; teacher models and external baselines retain their attribution',
        historical_control_gains_are_nonadditive=True,
        retained_system='system_v10', complete_jobs=31, all_five_represented=True,
        selection='one retained complete system recipe; no per-scene or per-query winner stitching',
        independent_final_test=False, new_common_core_all_tasks_validated=False,
        system_table=main, all_system_metrics=m, paired_effects=effects,
        supporting_provider_effects=providers,
        response_alternatives=response['macro'],
        negative_evidence=dict(pooled_hypothesis_rule=reference_negative['macro']['pooled'],
            nvos_reference_scope=reference_control['macro'],
            not_promoted=True, report='docs/experiments/2026-10-15-unified-hypothesis-results.json'),
        input_contracts=dict(lerf='source-assisted frozen RGB geometry; selected output uses 2D projection',
            nvos='retained positive-derived reference SAM mask and declared tracks; not raw signed interaction',
            scannet='provided unlabelled mesh and declared source evidence',
            uqis='map-only; official mesh/RGBD construction; member model trained on 22 labelled rooms and calibrated on 3 others'),
        source_files=receipts, benchmark_runs_in_this_builder=0)
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST/'tables').mkdir(exist_ok=True)
    (DEST/'evidence.json').write_text(json.dumps(data, indent=2)+'\n')
    def esc(s):
        return s.replace('_', r'\_')
    lines = [r'\begin{tabular}{llr}', r'\toprule', r'Task & Metric & Result (\%) \\', r'\midrule']
    lines += [f"{esc(r['task'])} & {r['metric']} & {100*r['value']:.2f} \\\\" for r in main]
    lines += [r'\bottomrule', r'\end{tabular}']
    (DEST/'tables/system_reference.tex').write_text('\n'.join(lines)+'\n')
    lines = [r'\begin{tabular}{lrrr}', r'\toprule', r'Controlled intervention & Before & After & $\Delta$ (pp) \\', r'\midrule']
    lines += [f"{r['name']} & {100*r['before']:.2f} & {100*r['after']:.2f} & {r['delta_pp']:+.2f} \\\\" for r in effects]
    lines += [r'\bottomrule', r'\end{tabular}']
    (DEST/'tables/mechanism_evidence.tex').write_text('\n'.join(lines)+'\n')
    lines = [r'\begin{tabular}{lrrr}', r'\toprule', r'In-house component contrast & Before & After & $\Delta$ (pp) \\', r'\midrule']
    lines += [f"{esc(r['name'])} & {100*r['before']:.2f} & {100*r['after']:.2f} & {r['delta_pp']:+.2f} \\\\" for r in providers]
    lines += [r'\bottomrule', r'\end{tabular}']
    (DEST/'tables/provider_evidence.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(system_table=main, paired_effects=effects, verified_files=len(receipts)), indent=2))
    return data


if __name__ == '__main__':
    build()
