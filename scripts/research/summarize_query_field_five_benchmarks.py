"""One complete five-task table; no cross-scene or cross-input model selection."""
import json
from pathlib import Path

import numpy as np

from radio_gs.query_field.storage import sha256
from scripts.research.prepare_query_field_five_benchmarks import ROOT, CONFIG, write
from scripts.research.run_query_field_five_predict import verify_predictions


def main():
    cohort=json.loads((CONFIG/'cohort.json').read_text())
    if not (ROOT/'scoring_complete.json').exists():raise ValueError('full five-task scoring is not complete')
    verify_predictions(cohort)
    baseline_path=Path('output/query_field_20261007/frozen_v10/baseline_a.json')
    baseline=json.loads(baseline_path.read_text())
    evidence=[CONFIG/'cohort.json',ROOT/'implementation.json',ROOT/'integration_correction.json',
              ROOT/'construction_complete.json',ROOT/'predictions_complete.json',ROOT/'scoring_complete.json',
              ROOT/'integration_tests.log',ROOT/'rgb_stroke_regression.log']
    lerf=[];nvos=[];scannet=[];fields=[]
    for job in cohort['lerf']:
        path=Path(job['predictions'])/'evaluation/evaluation.json'
        report=json.loads(path.read_text());values=report['metrics']
        lerf.append(dict(scene=job['scene'],native_iou=values['native_generic_relevance']['iou'],
            native_ap=values['native_generic_relevance']['ap'],member_iou=values['shared_membership']['iou'],
            member_ap=values['shared_membership']['ap'],subset_iou=values['shared_membership']['selected']['miou'],
            query_views=values['shared_membership']['observations'],missing=values['shared_membership']['selected']['missing']))
        evidence.append(path)
    for job in cohort['nvos']:
        path=ROOT/'predictions/nvos'/job['name']/'evaluation.json'
        report=json.loads(path.read_text());row=report['results'][0]
        nvos.append(dict(scene=job['name'],response_iou=row['retained_response']['foreground_iou'],
                         ap=row['retained_response']['average_precision'],subset_iou=row['foreground_iou'],selected=row['selected']))
        evidence.append(path)
    for job in cohort['scannet']:
        path=ROOT/'predictions/scannet'/job['name']/'evaluation.json'
        report=json.loads(path.read_text())
        scannet.append(dict(scene=job['name'],observed_vertex_fraction=report['observed_vertex_fraction'],
                            member={k:v['miou'] for k,v in report['metrics']['member'].items()},
                            native={k:v['miou'] for k,v in report['metrics']['native'].items()}))
        evidence.append(path)
    uqis_path=ROOT/'predictions/uqis/evaluation.json';uqis=json.loads(uqis_path.read_text());evidence.append(uqis_path)
    if len(lerf)!=4 or len(nvos)!=8 or len(scannet)!=10 or uqis['query_count']!=268 or uqis['target_count']!=67:
        raise ValueError('complete benchmark denominators differ')
    macro=dict(lerf={key:float(np.mean([r[key] for r in lerf])) for key in ['native_iou','native_ap','member_iou','member_ap','subset_iou']},
               nvos={key:float(np.mean([r[key] for r in nvos])) for key in ['response_iou','ap','subset_iou']},
               scannet={arm:{k:float(np.mean([r[arm][k] for r in scannet])) for k in ['19','15','10']} for arm in ['native','member']},
               uqis={m:dict(ap=uqis['modalities'][m]['scene_macro']['average_precision'],
                            subset_iou=uqis['explicit_selection']['modalities'][m]['scene_macro']['iou'],
                            empty_selections=uqis['explicit_selection']['modalities'][m]['empty_selections'])
                     for m in ['text','image','point_2d','point_3d']})
    rows=[]
    def comparison(task,metric,candidate,reference,denominator):
        rows.append(dict(task=task,metric=metric,candidate=candidate,v10=reference,
                         delta_pp=100*(candidate-reference),denominator=denominator))
    comparison('lerf2d','member response IoU',macro['lerf']['member_iou'],baseline['macro']['lerf']['semantic_iou'],'4 scenes')
    comparison('lerf2d','member response AP',macro['lerf']['member_ap'],baseline['macro']['lerf']['ap'],'4 scenes')
    comparison('lerf3d_local_projection','actual subset projection IoU',macro['lerf']['subset_iou'],baseline['macro']['lerf']['selected_projection_iou'],'4 scenes')
    for split in ['19','15','10']:
        comparison('scannet_ovs','member mIoU '+split,macro['scannet']['member'][split],baseline['macro']['scannet']['miou_'+split],'10 rooms')
    comparison('nvos','signed response IoU',macro['nvos']['response_iou'],baseline['macro']['nvos']['iou'],'8 cases')
    comparison('nvos','signed response AP',macro['nvos']['ap'],baseline['macro']['nvos']['ap'],'8 cases')
    for modality,value in macro['uqis'].items():
        comparison('uqis/'+modality,'AP',value['ap'],baseline['macro']['uqis'][modality+'_ap'],'9 rooms,67 targets')
        comparison('uqis/'+modality,'actual subset IoU',value['subset_iou'],baseline['macro']['uqis'][modality+'_selected_iou'],'9 rooms,67 targets')
    for job in cohort['construction']:
        root=Path(job['output']);build=json.loads((root/'build_complete.json').read_text())
        fields.append(build);evidence += [root/'build_complete.json',root/'models/trained/manifest.json',root/'observations/observations.json']
    unsupported={m:0 for m in macro['uqis']}
    for path in (ROOT/'predictions/uqis/receipts').glob('*.json'):
        record=json.loads(path.read_text());evidence.append(path)
        for row in record['queries']:
            unsupported[row['modality']] += row['status']!='ok'
    report=dict(schema='radio_gs.query_field_five_benchmark_results.v1',
        architecture_parent='fb3c768',complete=True,primary_method='same fitted physical member field on every task',
        cohort_sha256=sha256(CONFIG/'cohort.json'),baseline_sha256=sha256(baseline_path),
        comparison=rows,macro=macro,per_scene=dict(lerf=lerf,nvos=nvos,scannet=scannet),
        uqis_unsupported=unsupported,uqis_actual_set_joint_success=uqis['explicit_selection']['paired_query_robustness'],
        construction=fields,independent_final_test=False,compact_model=False,accuracy_promotion=False,
        contrasts=dict(source_budget='New fields18 training/2 development source views; retained LERF18 and fern12; v10 generally more observations.',
          teachers='SigLIP/DINO/query-free SAM only; v10 ScanNet includes MaskAdapter and v10 UQIS adds labelled3D member training.',
          deployment='One scene Gaussian field; no query-time source images, masks or tracks.',
          geometry='LERF/NVOS saved geometry; ScanNet saved unlabelled-mesh Gaussians; UQIS same official unlabelled mesh converted to Gaussians.',
          readout='Same member core; typed category competition and fixed mesh transport. LERF3D is retained subset projection, not independent3D truth.',
          nvos='Raw signed scribbles versus v10 positive-derived SAM reference/source tracks.'),
        evidence=[dict(path=str(p),sha256=sha256(p)) for p in evidence])
    write(ROOT/'summary.json',report)
    write(Path('docs/experiments/2026-10-08-five-benchmark-results.json'),report)
    lines=['# Complete five-benchmark architecture baseline','',
           'The fixed common physical architecture has now completed all five local benchmark families. '+
           'This is a high-capacity development baseline, with no per-scene/input checkpoint selection. '+
           'Full coverage precedes further architecture changes or staged error attribution.','',
           '[Protocol](2026-10-08-five-benchmark-architecture-protocol.md) · '+
           '[Machine-readable results](2026-10-08-five-benchmark-results.json)','',
           '## Complete comparison','',
           'Values are percent; delta is candidate minus saved v10 in percentage points. '+
           'The table is a system comparison with the input/teacher differences below, not an isolated architectural causal effect.','',
           '| Task | Metric | Candidate | Saved v10 | Delta (pp) | Cohort |',
           '|---|---|---:|---:|---:|---|']
    for row in rows:
        lines.append(f"| {row['task']} | {row['metric']} | {100*row['candidate']:.4f} | {100*row['v10']:.4f} | {row['delta_pp']:+.4f} | {row['denominator']} |")
    lines += ['', 'All eight NVOS cases and all268 UQIS queries remain in the denominator. '+
              'Unsupported UQIS registration/observation counts: '+str(unsupported)+'. Empty outputs are scored, not removed.', '',
              'Native controls are separate from the common member result: LERF native IoU/AP '+
              f"{100*macro['lerf']['native_iou']:.4f}/{100*macro['lerf']['native_ap']:.4f}%; ScanNet native19/15/10 mIoU "+
              '/'.join(f"{100*macro['scannet']['native'][k]:.4f}" for k in ['19','15','10'])+'%.', '',
              'NVOS actual selected-subset projection IoU is '+f"{100*macro['nvos']['subset_iou']:.4f}%"+
              ', reported separately from its retained normalized-response benchmark score.', '',
              '## Contracts and scope', '']
    lines += ['- '+v for v in report['contrasts'].values()]
    lines += ['', 'The reused four-scene LERF predictions are hash-bound to the accepted fields; fern is replayed against its saved output. '+
              'All new cohort predictions are persisted before new scoring. The UQIS evaluator retains its original67 targets, '+
              'all four modalities and hidden target pairing. Gaussian scores/selected rows are also saved before mesh transport.', '',
              'New construction contains 25 fields, with horns left/centre sharing a field. Existing four LERF fields and fern are reused. '+
              'No compression, latency tuning, support-mixture search or negative-rule experiment was conducted.', '',
              'The only post-freeze integration correction accepts the official trex scalar scribbles stored as three identical RGB channels. '+
              'It collapses exact replicas without changing evidence or the member method. The failed attempt, original implementation hashes '+
              'and correction receipt are retained. The original integration suite passed 22 tests; all 8 signed-raster tests pass after the '+
              'new regression case, and fern reproduces all four saved output tensors exactly.', '',
              '## Next: attribute the measured gaps', '',
              'Use this fixed baseline to isolate source/geometry coverage, native observation fusion, query identity, '+
              'member extent, and output projection/transport in that order. Save the complete cohort while changing one '+
              'stage at a time; do not let local score gates prevent reporting other benchmarks. These are follow-up '+
              'diagnostic questions, not causal conclusions from the current system table.', '',
              'The existing development scenes and queries have been opened historically. No independent final-test, '+
              'v10-equivalent supervision, compactness, or overall accuracy-promotion claim is made.']
    with Path('docs/experiments/2026-10-08-five-benchmark-results.md').open('x') as handle:handle.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(macro=macro,uqis_unsupported=unsupported),indent=2))


if __name__=='__main__':main()
