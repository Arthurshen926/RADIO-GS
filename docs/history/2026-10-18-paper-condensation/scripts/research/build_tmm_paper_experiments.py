"""Build paper comparisons and matched ablations from cited, sealed evidence.

No inference or baseline training. Published values are explicitly attributed
to the reporting paper; local reproductions are reaggregated from frozen files.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import yaml

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / 'paper/tmm_response'
CONFIG = 'configs/research/tmm_paper_experiments.json'
KINDS = {'published': 'P', 'reproduced': 'R', 'provider': 'O', 'policy': 'M', 'pending': '--'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def checked_mean(rows, expected, metric):
    if set(rows) != set(expected) or len(rows) != len(expected):
        raise ValueError('cohort mismatch: refusing partial or substituted aggregate')
    values = [rows[name][metric] for name in expected]
    if not all(math.isfinite(v) for v in values):
        raise ValueError('nonfinite metric')
    return mean(values)


def validate_row(row, config):
    if row['kind'] not in KINDS:
        raise ValueError('unrecognized result provenance')
    if set(row['metrics_percent']) != set(config['metrics'][row['benchmark']]):
        raise ValueError('metric identity mismatch; AP is not accuracy')
    if row['kind'] == 'pending':
        if any(v is not None for v in row['metrics_percent'].values()):
            raise ValueError('pending mechanism cannot inherit provider performance')
    elif any(v is not None and (not math.isfinite(v) or not 0 <= v <= 100)
             for v in row['metrics_percent'].values()):
        raise ValueError('invalid percentage metric')
    if row['kind'] == 'published' and row.get('source') not in config['sources']:
        raise ValueError('published row requires reporting source')


def tex_table(headers, rows):
    lines = [r'\begin{tabular}{ll'+'r'*(len(headers)-2)+'}', r'\toprule',
             ' & '.join(headers)+r' \\', r'\midrule']
    lines += [row if isinstance(row,str) else ' & '.join(str(v) for v in row)+r' \\' for row in rows]
    return '\n'.join(lines+[r'\bottomrule', r'\end{tabular}'])+'\n'


def build():
    config = read(REPO / CONFIG)
    receipts = {}

    def bind(path, expected=None):
        path = Path(path)
        if not path.is_absolute(): path = REPO / path
        digest = sha(path)
        if expected and digest != expected:
            raise ValueError('source hash changed: '+str(path))
        try: key = str(path.relative_to(REPO))
        except ValueError: key = str(path)
        receipts[key] = digest
        return path

    bind(CONFIG)
    freeze = yaml.safe_load(bind(config['freeze']).read_text())['canonical_tasks']
    sources = {}
    for key in config['freeze_keys']:
        sources[key] = [read(bind(a['path'], a['sha256'])) for a in freeze[key]['authoritative_artifacts']]
    rows = [dict(r) for r in config['published_rows']]
    for row in rows:
        row['group'] = 'paper8_gaussian' if row['benchmark']=='scannet' else 'paper_protocol'

    def add(bench, method, kind, values, group='paper_protocol', **kwargs):
        rows.append(dict(benchmark=bench, method=method, kind=kind, group=group,
            metrics_percent=dict(zip(config['metrics'][bench], values)), **kwargs))

    key = 'concept_lerf2d_occamlgs'
    scenes = freeze[key]['cohort']['scenes']
    local = {name: item['macro'] for name,item in zip(scenes,sources[key])}
    add('lerf2d', "Occam's LGS", 'reproduced',
        [100*checked_mean(local,scenes,m) for m in ('miou','loc_acc')],
        receipt_key=key, input_contract='released semantic path; compatible RGB geometry')
    key = 'concept_lerf3d_vala'
    local = sources[key][0]['per_scene']
    add('lerf3d', 'VALA (compatible geometry)', 'reproduced',
        [100*checked_mean(local,scenes,m) for m in ('mIoU','Acc@0.25')],
        receipt_key=key, input_contract='VALA semantic/evaluator path on Occam RGB geometry')
    key = 'concept_scannet_ovs_vala_paper8'
    paper8 = freeze[key]['cohort']['scenes']
    local = sources[key][0]['scenes']
    values = [100*checked_mean({s:r['splits'][split] for s,r in local.items()},paper8,m)
              for split in ('19','15','10') for m in ('miou','macc')]
    add('scannet','VALA','reproduced',values,'paper8_gaussian',receipt_key=key,
        input_contract='official significance, robust lifting and Gaussian pseudo-GT')
    key = 'spatial_nvos_ludvig'
    local = sources[key][0]['per_scene']
    tasks = freeze[key]['cohort']['tasks']
    for record in local.values():
        if record['seeds'] != [0,1,2] or record['num_runs'] != 3:
            raise ValueError('NVOS reproduction requires all three seeds per task')
        if abs(mean(record['run_values_iou_percent'])-record['local_mean_iou_percent'])>1e-10:
            raise ValueError('NVOS seed aggregation changed')
    add('nvos','LUDVIG-SAM','reproduced',
        [checked_mean(local,tasks,'local_mean_iou_percent')],receipt_key=key,
        input_contract='positive scribbles; online target RGB; all-view RGB geometry')

    system_path = bind(config['system_summary'])
    system = read(system_path)
    if system['completed'] != 31: raise ValueError('incomplete frozen provider cohort')
    local = {scene: read(bind(system_path.parent/f'lerf/{scene}/evaluation/evaluation.json'))
             ['metrics']['hierarchical_pipeline'] for scene in scenes}
    if sum(r['observations'] for r in local.values()) != 208:
        raise ValueError('LERF object cohort changed')
    add('lerf2d','Ours (v10)','provider',
        [100*checked_mean(local,scenes,m) for m in ('iou','localization')],
        input_contract='frozen RGB geometry; source-assisted full teachers')
    add('lerf3d','Ours (v10)','provider',
        [100*checked_mean({s:r['selected'] for s,r in local.items()},scenes,m)
         for m in ('miou','acc025')],input_contract='independent selected Gaussians; same projection GT')
    nvos = {r['job']['name']:r['metrics'] for r in system['rows'] if r['job']['benchmark']=='nvos'}
    add('nvos','Ours (v10)','provider',[100*checked_mean(nvos,tasks,'iou')],
        input_contract='positive-derived reference SAM mask; cached tracks; no online target RGB')
    coverage_path=PAPER/'nvos_coverage_feedback.json'
    if coverage_path.exists():
        coverage=read(bind(coverage_path))
        if not coverage['complete'] or coverage['default_changed'] or coverage['independent_test']:
            raise ValueError('NVOS coverage study changed scope')
        for p,h in coverage['receipts'].items(): bind(p,h)
        values={s:r['coverage_high']['retained_gt_gt127'] for s,r in coverage['rows'].items()}
        add('nvos','Ours + coverage (N01)','policy',[100*checked_mean(values,tasks,'iou')],
            source_path=str(coverage_path.relative_to(REPO)),
            input_contract='fixed coverage/high recipe on all eight development tasks; same teachers, tracks, geometry and output threshold; no target RGB; not combined with L11')
    source_readout_path=PAPER/'nvos_source_readout_feedback.json'
    if source_readout_path.exists():
        calibration=read(bind(source_readout_path))
        if not calibration['complete'] or calibration['default_changed'] or not calibration['shared_final_response']:
            raise ValueError('NVOS source-readout experiment changed scope')
        for p,h in calibration['receipts'].items(): bind(p,h)
        # This named candidate is fixed before observing its target scores.
        # The fit-source estimator stays in the paired ablation regardless of rank.
        values={s:r['source_holdout']['retained_gt_gt127'] for s,r in calibration['rows'].items()}
        add('nvos','Ours + coverage + source readout (N02)','policy',[100*checked_mean(values,tasks,'iou')],
            source_path=str(source_readout_path.relative_to(REPO)),
            input_contract='N01 coverage/high field; threshold estimated from held-out reconstruction views using source teacher evidence; all eight tasks; no target RGB/GT calibration; not combined with L11')
    local_credit_path=PAPER/'nvos_local_credit_feedback.json'
    if local_credit_path.exists():
        local=read(bind(local_credit_path))
        if not local['complete'] or local['default_changed'] or local['primary_arm']!='held_local':
            raise ValueError('NVOS local-credit experiment changed scope')
        for p,h in local['receipts'].items():bind(p,h)
        values={s:r['held_local']['retained_gt_gt127'] for s,r in local['rows'].items()}
        add('nvos','Ours + local edit credit (N03)','policy',[100*checked_mean(values,tasks,'iou')],
            source_path=str(local_credit_path.relative_to(REPO)),
            input_contract='predeclared held-local gate on N01 edits to v10; same teachers/geometry and fixed 0.5; source-only credit; all eight; no target RGB; not combined with L11')
    for filename,primary,label in [('nvos_image_response_feedback.json','rgb_mask_lift','Ours + image response + 3D (N04)'),
                                   ('nvos_partial_response_feedback.json','positive_core_lift','Ours + partial image response + 3D (N05)'),
                                   ('nvos_unboxed_response_feedback.json','unboxed_sparse_lift','Ours + unboxed image response + 3D (N06)'),
                                   ('nvos_memory_response_feedback.json','sequence_memory_lift','Ours + reference memory + 3D (N07)'),
                                   ('nvos_native_writeback_feedback.json','sequence_memory_native','Ours + native memory writeback (N08)'),
                                   ('nvos_signed_affinity_feedback.json','signed_native','Ours + signed reference affinity (N09)'),
                                   ('nvos_direct_affinity_feedback.json','signed_direct_native','Ours + direct signed affinity (N10)'),
                                   ('nvos_instance_extent_feedback.json','scribble_anchor_native','Ours + instance-constrained extent (N11)'),
                                   ('nvos_reference_completion_feedback.json','constrained_memory_native','Ours + signed reference completion (N12)'),
                                   ('nvos_reference_delta_feedback.json','probability_native','Ours + paired reference delta (N13)'),
                                   ('nvos_reference_admission_feedback.json','feasible_native','Ours + source feasibility admission (N14)'),
                                   ('nvos_multiview_completion_feedback.json','completed_joint_native','Ours + completed multiview fusion (N15)'),
                                   ('nvos_target_redecode_feedback.json','crop_native','Ours + query-local target decoding (N16)')]:
        path=PAPER/filename
        if not path.exists():continue
        candidate=read(bind(path))
        if not candidate['complete'] or candidate['default_changed'] or candidate['primary_arm']!=primary:
            raise ValueError('NVOS image-response scope changed')
        for p,h in candidate['receipts'].items():bind(p,h)
        values={s:r[primary]['retained_gt_gt127'] for s,r in candidate['rows'].items()}
        add('nvos',label,'policy',[100*checked_mean(values,tasks,'iou')],source_path=str(path.relative_to(REPO)),
            input_contract='observed target RGB; fixed all-eight prompt and actual 3D writeback; original v10 geometry; view-conditioned, not a query-global field; no target GT calibration; not combined with L11'+
            ('; official negative reference scribbles and frozen DINOv2 are added inputs; policy controls share these inputs' if filename in ['nvos_signed_affinity_feedback.json','nvos_direct_affinity_feedback.json','nvos_instance_extent_feedback.json','nvos_reference_completion_feedback.json','nvos_reference_delta_feedback.json','nvos_reference_admission_feedback.json','nvos_multiview_completion_feedback.json','nvos_target_redecode_feedback.json'] else ''))
        if filename=='nvos_target_redecode_feedback.json':
            values={s:r['full_native']['retained_gt_gt127'] for s,r in candidate['rows'].items()}
            add('nvos','Ours + full-frame target decoding (N16 control)','policy',
                [100*checked_mean(values,tasks,'iou')],source_path=str(path.relative_to(REPO)),
                input_contract='one full-frame decode of the N14 complete response; same SAM3, DINOv2, official negative scribbles, target RGB and actual native 3D writeback as the local primary; fixed all-eight development control')
        if filename=='nvos_reference_admission_feedback.json':
            values={s:r['quality_native']['retained_gt_gt127'] for s,r in candidate['rows'].items()}
            add('nvos','Ours + source quality admission (N14 secondary)','policy',
                [100*checked_mean(values,tasks,'iou')],source_path=str(path.relative_to(REPO)),
                input_contract='fixed source-quality selector for all eight tasks; actual native 3D writeback; DINOv2, SAM3, official negative scribbles and target RGB; development, not an independent SOTA certification')
        if filename=='nvos_native_writeback_feedback.json':
            values={s:r['static_image_native']['retained_gt_gt127'] for s,r in candidate['rows'].items()}
            add('nvos','Ours + static native writeback (N08 secondary)','policy',
                [100*checked_mean(values,tasks,'iou')],source_path=str(path.relative_to(REPO)),
                input_contract='predeclared N08 secondary arm; exact N04 static teacher evidence; all eight tasks use native 3D writeback; observed target RGB and original geometry; view-conditioned; no per-task winners; development comparison, not independent test')
    inherited_path=REPO/'output/tmm_response_20261016/study/nvos_v10_readout_audit_v1/summary.json'
    if inherited_path.exists():
        inherited=read(bind(inherited_path))
        if not inherited['complete'] or inherited['new_method_gain'] or inherited['per_task_winners']:
            raise ValueError('NVOS inherited readout audit changed scope')
        values={s:r['independent_set_projection']['retained_gt_gt127'] for s,r in inherited['rows'].items()}
        add('nvos','Ours (v10, selected set)','provider',[100*checked_mean(values,tasks,'iou')],
            source_path=str(inherited_path.relative_to(REPO)),
            input_contract='same inherited v10, inputs and GT; existing independent set and alpha projection readout; no new model')
    for bench in ('lerf2d','lerf3d','nvos'):
        add(bench,'Shared replacement (pending)','pending',[None]*len(config['metrics'][bench]))

    residual = read(bind(config['residual_feedback']))
    for p,h in residual['receipts'].items(): bind(p,h)
    transfer = residual['transfer']
    if transfer['freshly_scored_category_scenes']!=10 or transfer['freshly_scored_uqis_queries']!=268:
        raise ValueError('incomplete latest transfer')
    region = read(bind(system_path.parent/'uqis/identity_region/result.json'))
    uqis_scenes = [r['job']['name'] for r in system['rows'] if r['job']['benchmark']=='uqis']
    if region['query_count']!=268 or region['scene_count']!=9:
        raise ValueError('incomplete retained UQIS regional provider')
    region_values=[]
    for modality in ('text','image','point_2d','point_3d'):
        query_rows=[r for r in region['per_query'] if r['modality']==modality]
        set_rows=[r for r in region['explicit_selection']['per_query'] if r['modality']==modality]
        if len(query_rows)!=67 or {r['query_id'] for r in query_rows}!={r['query_id'] for r in set_rows}:
            raise ValueError('UQIS ranking/selection cohort mismatch')
        for source,metric in ((set_rows,'iou'),(query_rows,'average_precision')):
            per_scene={s:{metric:mean(r[metric] for r in source if r['scene_id']==s)}
                       for s in {r['scene_id'] for r in source}}
            region_values.append(100*checked_mean(per_scene,uqis_scenes,metric))
    add('uqis','Ours (regional readout)','provider',region_values,'uqis9_map_only',
        source_path=str((system_path.parent/'uqis/identity_region/result.json').relative_to(REPO)),
        input_contract='historical regional provider; complete-decoder difference, not a selector ablation')
    for arm,method,kind in [('baseline','Ours (v10)','provider'),
                            ('full','Ours + replacement (L11)','policy')]:
        add('scannet',method,kind,[100*transfer['category_macro'][arm]['miou_'+s]
            if m=='miou' else None for s in ('19','15','10') for m in ('miou','macc')],
            'internal10_mesh', input_contract='provided mesh; source-assisted; original 10 scenes')
        add('uqis',method,kind,[100*transfer['uqis_macro'][arm][k] for k in config['metrics']['uqis']],
            'uqis9_map_only',input_contract='fixed map; no source RGB/depth at query time; points retained')

    bridge_path = REPO/config['bridge']
    if bridge_path.exists():
        bridge = read(bind(bridge_path))
        if not bridge['complete'] or bridge['scenes'] != paper8:
            raise ValueError('incomplete or mismatched metric bridge')
        seal = read(bind(bridge_path.parent/'prediction_seal.json',bridge['prediction_seal_sha256']))
        if not seal['predictions_before_labels']: raise ValueError('unsealed metric bridge')
        for p,h in seal['inputs'].items(): bind(p,h)
        for scene,record in bridge['rows'].items():
            bind(record['label_file'],record['label_sha256'])
            if read(bind(bridge_path.parent/(scene+'.json'))) != record:
                raise ValueError('bridge scene report mismatch')
        values = [100*checked_mean({s:r['splits'][split] for s,r in bridge['rows'].items()},paper8,m)
                  for split in ('19','15','10') for m in ('miou','macc')]
        add('scannet','Ours (v10, metric bridge)','provider',values,'paper8_gaussian',
            input_contract='provided-mesh carrier, different from VALA RGB geometry; metric aligned only')

    for row in rows:
        row['ownership'] = 'ours' if row['kind'] in ('provider','policy','pending') else 'external'
        validate_row(row,config)
    for correction in config['corrections']:
        vala = next(r for r in rows if r['benchmark']=='lerf3d' and r['method']=='VALA' and r['kind']=='published')
        if list(vala['metrics_percent'].values()) != correction['correct_values']:
            raise ValueError('VALA/Dr. Splat paper attribution regression')

    tables = {}
    headers = {'lerf2d':['Method','Origin','mIoU','Loc. acc.'],
        'lerf3d':['Method','Origin','mIoU','Acc.@0.25'],
        'scannet':['Method','Origin','IoU19','Acc19','IoU15','Acc15','IoU10','Acc10'],
        'nvos':['Method','Origin','IoU'],
        'uqis':['Method','Origin','T-IoU','T-AP','I-IoU','I-AP','P2-IoU','P2-AP','P3-IoU','P3-AP']}
    for bench in config['metrics']:
        selected = [r for r in rows if r['benchmark']==bench and r['kind']!='pending']
        if bench=='scannet':
            selected.sort(key=lambda r:r['group']=='internal10_mesh')
        rendered = []
        group = None
        for row in selected:
            if bench=='scannet' and row['group'] != group:
                group=row['group']
                title=('Eight scenes: Gaussian scoring (input differences disclosed)' if group=='paper8_gaussian'
                       else 'Ten scenes: mesh scoring (separate evaluation track)')
                rendered.append(r'\midrule\multicolumn{8}{l}{\textit{'+title+r'}} \\')
            label = row['method']
            if row['kind']=='published':
                label += r' \cite{'+config['sources'][row['source']]['citation']+'}'
            if row['group']=='internal10_mesh': label += ' (10-mesh)'
            rendered.append([label,KINDS[row['kind']]]+
                ['--' if v is None else f'{v:.2f}' for v in row['metrics_percent'].values()])
        tables['comparison_'+bench] = tex_table(headers[bench],rendered)
    ablations = []
    for arm,label in [('baseline','Incumbent only'),('full','Full shared replacement'),
                      ('query_blind','Without query coordinates'),('permuted','Permuted query correspondence'),
                      ('unregularized','Without null correction constraint'),('unrestricted','Without positive margin')]:
        s,u=transfer['category_macro'][arm],transfer['uqis_macro'][arm]
        change=residual['changed_actual_sets'].get(arm,dict(improved=0,harmed=0))
        ablations.append(dict(arm=arm,label=label,category=s,uqis=u,changed_actual_sets=change))
    tables['mechanism_ablations'] = tex_table(
        ['Policy','Repair / harm','S19','S15','S10','T-IoU','I-IoU','T-AP','I-AP'],
        [[a['label'],f"{a['changed_actual_sets']['improved']} / {a['changed_actual_sets']['harmed']}"]+
         [f'{100*v:.2f}' for v in [a['category']['miou_'+s] for s in ('19','15','10')]+
          [a['uqis'][k] for k in ('text_selected_iou','image_selected_iou','text_ap','image_ap')]] for a in ablations])
    for name,content in tables.items():
        path=PAPER/'tables'/f'{name}.tex'; path.write_text(content); bind(path)
    catalog=dict(schema='radio_gs.tmm_paper_tables.v1',rows=rows,ablations=ablations,
        corrections=config['corrections'],coverage=config['coverage'],sources=config['sources'],
        local_reproduction_artifacts_verified=sum(len(sources[k]) for k in sources),
        external_baselines_retrained=False,all_five_mechanism_complete=False,
        implemented_method_all_five_evaluated=True,
        ownership_note='v10 and its component controls belong to our method; replacement is an optional extension',
        receipts=receipts)
    (PAPER/'paper_experiments.json').write_text(json.dumps(catalog,indent=2)+'\n')
    with (PAPER/'artifacts/paper_comparisons.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['benchmark','group','method','origin','metric','percent','source'])
        for row in rows:
            source=config['sources'][row['source']]['url'] if row.get('source') else row.get('receipt_key',config['system_summary'])
            if row['kind']=='policy':source=config['residual_feedback']
            if row.get('source_path'):source=row['source_path']
            if 'bridge' in row['method']:source=config['bridge']
            if row['kind']=='pending':source='pending'
            for metric,value in row['metrics_percent'].items():
                writer.writerow([row['benchmark'],row['group'],row['method'],row['kind'],metric,value,source])
    print(json.dumps(dict(rows=len(rows),tables=len(tables),verified_files=len(receipts),
        baseline_training=False,corrections=len(config['corrections'])),indent=2))


if __name__ == '__main__':
    build()
