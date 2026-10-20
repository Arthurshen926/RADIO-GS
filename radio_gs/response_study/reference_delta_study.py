"""Validate matched reference-intervention residuals and exact parent controls."""
import hashlib
import json
from statistics import mean
from .reference_study import read,sha,require,check_binary_metrics

ARMS=['v10','static_native','constrained_memory_native','probability_image','probability_native','log_odds_image','log_odds_native']
PROVIDERS=['probability','log_odds']
INPUTS=['static_native','constrained_memory_native','static_image','legacy_memory_image','constrained_memory_image']


def check_reference_delta_study(repo,study,item,jobs):
    root=study/item['run']
    if not (root/'summary.json').exists():
        return dict(**item,complete=False,status='failed' if (root/'failure.json').exists() else 'in_progress' if root.exists() else 'planned')
    m,s,seal,snapshot,binding=[read(root/p) for p in ('manifest.json','summary.json','predictions_complete.json','source_snapshot.json','parent_binding.json')]
    c=m['config'];parent=study/c['parent_run'];pm,ps,pseal=[read(parent/p) for p in ('manifest.json','summary.json','predictions_complete.json')]
    inherited=read(parent/'source_snapshot.json');require(sha(parent/'source_snapshot.json')==m['inherited_source_snapshot_sha256'],'N13 inherited source snapshot changed')
    snapshot={**inherited,**snapshot}
    expected=[j for j in jobs if j['benchmark']=='nvos'];names={j['name'] for j in expected}
    require(len(names)==8 and m['jobs']==pm['jobs']==expected and c['arms']==ARMS,'N13 cohort/arms changed')
    require(c==json.loads(snapshot['configs/research/nvos_reference_delta.json']['text']) and c['solver_steps']==120
            and c['footprint_backend']=='gsplat_cuda' and c['native_wh']==[1008,756] and c['source_hw']==[384,512] and c['threshold']==seal['threshold']==.5
            and c['primary_arm']==s['primary_arm']=='probability_native','N13 design changed')
    require(s['complete'] and ps['complete'] and not s['target_gt_used_for_prediction'] and not s['per_task_winners']
            and not s['independent_test'] and not s['default_changed'] and s['average_precision_measured'] and s['negative_scribble_used']
            and s['durable_dense_prediction_bytes']==0 and s['target_rgb_used'] and s['parent_controls_exact'],'N13 scope changed')
    require(seal['jobs']==8 and seal['images']==56 and not seal['target_gt_read'] and not binding['target_gt_read']
            and binding['images']==40 and set(seal['report_hashes'])==set(seal['predictions'])==set(s['rows'])==names,'N13 incomplete seal')
    require(sha(root/'predictions_complete.json')==s['prediction_seal_sha256'] and sha(parent/'manifest.json')==m['parent_manifest_sha256']
            and sha(parent/'predictions_complete.json')==binding['parent_seal_sha256'] and sha(parent/'summary.json')==s['parent_summary_sha256'],'N13 parent binding changed')
    require(binding['exact_array_digests']=={n+'_'+a:v['sha256'] for n,r in pseal['predictions'].items() for a,v in r.items() if a in INPUTS},'N13 evidence digests changed')
    receipts={str(root/p):sha(root/p) for p in ('manifest.json','summary.json','predictions_complete.json','source_snapshot.json','parent_binding.json')}
    for p in ('manifest.json','summary.json','predictions_complete.json','source_snapshot.json'):receipts[str(parent/p)]=sha(parent/p)
    for p,h in m['source_hashes'].items():require(hashlib.sha256(snapshot[p]['text'].encode()).hexdigest()==h==snapshot[p]['sha256'],'N13 source archive changed')
    inputs=dict(s['gt_hashes']);source_checks={};checks={}
    for name in names:
        rp=root/(name+'_prediction.json');ep=root/(name+'_evaluation.json');pp=parent/(name+'_prediction.json');r=read(rp);pr=read(pp);row=s['rows'][name]
        require(sha(rp)==seal['report_hashes'][name] and sha(pp)==r['parent_report_sha256']==pseal['report_hashes'][name]
                and r['job'] in expected and r['job']['name']==name and not r['target_gt_read']
                and r['predictions']==seal['predictions'][name] and set(r['predictions'])==set(ARMS),'N13 prediction changed')
        require(read(ep)==row and set(row)==set(ARMS),'N13 evaluation changed')
        checks[name]=r['implementation_checks'];cc=checks[name]
        require(cc['v10_binary_exact'] and cc['parent_inputs_exact'] and cc['native_grid_wh']==[1008,756]
                and all(0<=cc[k]<1e-4 for k in ('v10_render_max_error','compositor_alpha_max_error','normalized_operator_max_error')),'N13 renderer check failed')
        native={p+'_native' for p in PROVIDERS};require(set(r['lifting'])==native,'N13 missing native branch')
        for v in r['lifting'].values():require(v['unsupported_changed']==0 and v['source_image_loss_after']<=v['source_image_loss_before']+1e-5,'N13 fit/support check failed')
        require(set(r['residual'])==set(PROVIDERS) and all(v['unaltered_teacher_exact_noop'] for v in r['residual'].values()), 'N13 unchanged teacher must be exact no-op')
        source_checks[name]=r['source_teacher_checks'];require(source_checks[name] and all(set(x['confidence_weighted_mean_huber'])=={'v10'}|native for x in source_checks[name]),'N13 missing source checks')
        for arm,tracks in row.items():
            require(set(tracks)==set(c['gt_tracks']),'N13 GT tracks changed')
            if arm in ps['rows'][name]:
                require(tracks==ps['rows'][name][arm] and r['predictions'][arm]==pseal['predictions'][name][arm],'N13 parent control changed')
            for track,v in tracks.items():
                check_binary_metrics(v);count=sum(v[k] for k in ('tp','fp','fn','tn'));old=row['v10'][track]
                require(type(v['tn']) is int and v['tn']>=0 and v['independent_confusion_exact'] and 0<=v['average_precision']<=1 and abs(v['accuracy']-(v['tp']+v['tn'])/count)<1e-12
                        and v['tp']+v['fn']==old['tp']+old['fn'] and v['tn']+v['fp']==old['tn']+old['fp'],'N13 confusion/GT check failed')
        grand=repo/'output/tmm_response_20261016/study'/pm['config']['parent_run']/(name+'_prediction.json')
        require(sha(grand)==pr['parent_report_sha256'],'N13 grandparent binding changed')
        receipts[str(grand)]=sha(grand)
        for p,h in read(grand)['input_hashes'].items():require(p not in inputs or inputs[p]==h,'N13 inconsistent parent inputs');inputs[p]=h
        for p in (rp,ep,pp):receipts[str(p)]=sha(p)
    require(set(s['macro'])==set(c['gt_tracks']),'N13 missing macro GT track')
    for t,arms in s['macro'].items():
        require(set(arms)==set(ARMS),'N13 missing macro arms')
        for a,metrics in arms.items():
            require(set(metrics)==set(c['metrics']),'N13 metrics changed')
            for k,v in metrics.items():require(abs(v-mean(row[a][t][k] for row in s['rows'].values()))<1e-12,'N13 mean differs')
    require(s['other_benchmarks']==ps['other_benchmarks'] and not s['other_benchmarks_freshly_evaluated'],'N13 other outputs changed')
    for p,h in inputs.items():require(sha(p)==h,'N13 input changed: '+p);receipts[p]=h
    spatial=study/'nvos_reference_delta_spatial_v1'; spatial_record=None
    if (spatial/'summary.json').exists():
        spatial_record=read(spatial/'summary.json');src=read(spatial/'source_snapshot.json')
        require(spatial_record['complete'] and spatial_record['diagnostic_only'] and spatial_record['tasks']==8 and
                spatial_record['target_gt_after_parent_scoring'] and spatial_record['parent_summary_sha256']==sha(root/'summary.json') and
                spatial_record['parent_seal_sha256']==sha(root/'predictions_complete.json') and
                spatial_record['figure_sha256']==sha(spatial/'errors.png') and
                spatial_record['source_snapshot_sha256']==sha(spatial/'source_snapshot.json') and
                hashlib.sha256(src['text'].encode()).hexdigest()==src['sha256'], 'N13 spatial binding differs')
        for path in spatial.iterdir():
            if path.is_file(): receipts[str(path)]=sha(path)
    previous=repo/'output/tmm_response_20261016/study/nvos_instance_extent_v2/summary.json'
    incumbent=read(previous);receipts[str(previous)]=sha(previous)
    require(read(study/'storage_policy.json')==c['storage_policy'] and c['storage_policy']['study_bytes']==8*2**20 and
            c['maximum_parent_link_bytes']==128*2**20, 'N13 budget differs')
    return dict(**item,complete=True,status='completed_not_promoted',fresh_prediction_jobs=8,images=56,spatial=spatial_record,previous_candidate_rows=incumbent['rows'],previous_candidate_macro=incumbent['macro'],average_precision_measured=True,negative_scribble_used=True,rows=s['rows'],macro=s['macro'],
        receipts=receipts,source_teacher_checks=source_checks,implementation_checks=checks,primary_arm=c['primary_arm'],
        target_rgb_arms=ARMS[1:],independent_test=False,other_jobs_fresh_inference=False,default_changed=False)
