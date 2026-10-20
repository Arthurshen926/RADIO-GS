"""Validate the fixed N16 intervention and its exact N14 controls."""
import hashlib
import json
from statistics import mean
from .reference_study import read, sha, require, check_binary_metrics

ARMS=['v10','quality_image','quality_native','full_image','full_native','crop_image','crop_native']


def check_target_redecode_study(repo, study, item, jobs):
    root=study/item['run'];old=repo/'output/tmm_response_20261016/study'
    if not (root/'summary.json').exists():
        return dict(**item,complete=False,status='failed' if (root/'failure.json').exists()
                    else 'in_progress' if root.exists() else 'planned')
    m,s,seal,snapshot=[read(root/p) for p in ['manifest.json','summary.json','predictions_complete.json','source_snapshot.json']]
    c=m['config'];parent=old/c['parent_run'];candidate=study/c['candidate_run'];cs=read(candidate/'summary.json')
    expected=[j for j in jobs if j['benchmark']=='nvos'];names={j['name'] for j in expected}
    require(m['jobs']==read(parent/'manifest.json')['jobs']==expected and len(names)==8 and c['arms']==ARMS,'N16 cohort differs')
    require(c==json.loads(snapshot['configs/research/nvos_target_redecode.json']['text']) and
            c['primary_arm']==s['primary_arm']=='crop_native' and c['context_fraction']==.25 and
            c['solver_steps']==120 and c['native_wh']==[1008,756] and c['source_hw']==[384,512] and
            c['threshold']==seal['threshold']==.5 and c['points_per_sign']==16,'N16 recipe differs')
    require(read(study/'storage_policy.json')==c['storage_policy'] and c['storage_policy']['study_bytes']==8*2**20 and
            c['maximum_new_durable_bytes']==2**20 and c['maximum_volatile_prediction_bytes']==256*2**20,'N16 budget differs')
    require(s['complete'] and not s['default_changed'] and not s['target_gt_used_for_prediction'] and
            not s['per_task_winners'] and not s['independent_test'] and s['target_rgb_used'] and
            s['negative_scribble_used'] and s['average_precision_measured'] and s['durable_dense_prediction_bytes']==0 and
            not s['other_benchmarks_freshly_evaluated'],'N16 scope differs')
    require(seal['jobs']==8 and seal['images']==56 and not seal['target_gt_read'] and
            set(seal['report_hashes'])==set(seal['predictions'])==set(s['rows'])==names and
            s['prediction_seal_sha256']==sha(root/'predictions_complete.json'),'N16 seal differs')
    require(m['parent_manifest_sha256']==sha(parent/'manifest.json') and s['parent_summary_sha256']==sha(parent/'summary.json') and
            m['candidate_manifest_sha256']==sha(candidate/'manifest.json') and
            s['candidate_summary_sha256']==s['previous_candidate_summary_sha256']==sha(candidate/'summary.json'),'N16 lineage differs')
    receipts={str(p):sha(p) for p in root.iterdir() if p.is_file()};inherited={}
    for directory,key in [(parent,'inherited_source_snapshot_sha256'),
                          (old/'nvos_memory_response_v2','inherited_memory_snapshot_sha256'),
                          (old/'nvos_instance_extent_v2','inherited_instance_snapshot_sha256'),
                          (candidate,'inherited_completion_snapshot_sha256')]:
        require(m[key]==sha(directory/'source_snapshot.json'),'N16 archive binding differs')
        inherited.update(read(directory/'source_snapshot.json'))
        for file in ['source_snapshot.json','manifest.json','summary.json']:receipts[str(directory/file)]=sha(directory/file)
    for run,digest in read(root/'completion_lineage.json').items():
        path=study/run/'source_snapshot.json';require(sha(path)==digest,'N16 completion ancestry differs')
        inherited.update(read(path));receipts[str(path)]=digest
    inherited.update(snapshot)
    for path,entry in inherited.items():
        require(hashlib.sha256(entry['text'].encode()).hexdigest()==entry['sha256'],'N16 archive text differs')
    for path,digest in m['source_hashes'].items():require(snapshot[path]['sha256']==digest,'N16 source binding differs')
    for path,digest in m['worker_source_hashes'].items():
        require(path in inherited and inherited[path]['sha256']==digest,'N16 worker source missing: '+path)
    inputs=dict(s['gt_hashes']);decoders={};checks={};sources={}
    for name in names:
        path=root/(name+'_prediction.json');r=read(path);row=s['rows'][name]
        pp=parent/(name+'_prediction.json');cp=candidate/(name+'_prediction.json');pr,cr=read(pp),read(cp)
        require(r['name']==name and sha(path)==seal['report_hashes'][name] and not r['target_gt_read'] and
                r['predictions']==seal['predictions'][name] and set(r['predictions'])==set(row)==set(ARMS) and
                row==read(root/(name+'_evaluation.json')),'N16 task seal differs')
        require(r['parent_report_sha256']==sha(pp) and r['candidate_report_sha256']==sha(cp) and
                r['parent_arrays']=={a:cr['predictions'][a]['sha256'] for a in ['v10','quality_image']},'N16 replay differs')
        cc=r['implementation_checks'];checks[name]=cc
        require(cc['v10_binary_exact'] and all(0<=cc[k]<1e-4 for k in
                ['v10_render_max_error','target_alpha_max_error','normalized_operator_max_error']),'N16 renderer differs')
        d=r['decoder'];decoders[name]=d
        require(d['source_hashes']==m['worker_source_hashes'] and not d['target_gt_available'] and
                not d['reference_gt_available'] and d['sam_eval'] and d['dino_eval'] and d['memory_selection'] and
                d['negative_scribble_used'] and not d['target_dino_features_used'] and
                d['choices']==cr['decoder']['choices'] and d['context_fraction']==c['context_fraction'] and
                d['decoder']==cr['decoder']['decoder'] and set(d['refinement'])=={'full','crop'},'N16 decoder contract differs')
        for arm,v in d['refinement'].items():
            require(v['iterations']==1 and v['outside_changed']==0 and
                    v['input_sha256']==r['predictions']['quality_image']['sha256'] and
                    v['output_sha256']==r['predictions'][arm+'_image']['sha256'],'N16 refinement binding differs')
            box=v['box']
            if box is not None:
                x0,y0,x1,y1=box;require(0<=x0<x1<=1008 and 0<=y0<y1<=756,'N16 invalid crop')
            else:require(v['empty_fallback'],'N16 missing crop without fallback')
        require(d['refinement']['full']['box']==[0,0,1008,756] and r['transient_bytes']<64*2**20,'N16 image bounds differ')
        if d['refinement']['crop']['box']==[0,0,1008,756]:
            require(r['predictions']['crop_image']['sha256']==r['predictions']['full_image']['sha256'],
                    'N16 identical full-frame inputs disagree')
        fields={'quality_native','full_native','crop_native'}
        require(set(r['lifting'])==fields and all(v['unsupported_changed']==0 and
                v['source_image_loss_after']<=v['source_image_loss_before']+1e-5 for v in r['lifting'].values()),'N16 fit differs')
        sources[name]=r['source_teacher_checks']
        require(sources[name] and all(set(v['confidence_weighted_mean_huber'])==fields|{'v10'} for v in sources[name]),'N16 source diagnostics missing')
        for arm,tracks in row.items():
            require(set(tracks)==set(c['gt_tracks']),'N16 GT track missing')
            for track,v in tracks.items():
                check_binary_metrics(v);size=sum(v[k] for k in ['tp','fp','fn','tn']);control=row['v10'][track]
                require(type(v['tn']) is int and v['tn']>=0 and v['independent_confusion_exact'] and
                        0<=v['average_precision']<=1 and abs(v['accuracy']-(v['tp']+v['tn'])/size)<1e-12 and
                        v['tp']+v['fn']==control['tp']+control['fn'] and v['tn']+v['fp']==control['tn']+control['fp'],'N16 confusion differs')
                if arm in ['v10','quality_image','quality_native']:
                    require(all(v[k]==cs['rows'][name][arm][track][k] for k in ['tp','fp','fn','tn']),'N16 control differs')
        for arm,v in s['transitions_diagnostic_only'][name].items():
            a,b=row[arm]['retained_gt_gt127'],row['quality_native']['retained_gt_gt127']
            require(v['recovered_foreground']-v['lost_foreground']==a['tp']-b['tp'] and
                    v['added_false_positive']-v['removed_false_positive']==a['fp']-b['fp'],'N16 transitions differ')
        for p in [pp,cp]:receipts[str(p)]=sha(p)
        for p,h in pr['input_hashes'].items():
            require(p not in inputs or inputs[p]==h,'N16 conflicting input');inputs[p]=h
    require(s['other_benchmarks']==cs['other_benchmarks'] and set(s['macro'])==set(c['gt_tracks']),'N16 unchanged outputs differ')
    for track,arms in s['macro'].items():
        require(set(arms)==set(ARMS),'N16 macro arm missing')
        for arm,metrics in arms.items():
            require(set(metrics)==set(c['metrics']),'N16 macro metric missing')
            for k,v in metrics.items():require(abs(v-mean(row[arm][track][k] for row in s['rows'].values()))<1e-12,'N16 macro differs')
    for p,h in inputs.items():require(sha(p)==h,'N16 input changed: '+p);receipts[p]=h
    spatial=study/'nvos_target_redecode_spatial_v1';spatial_record=None
    if (spatial/'summary.json').exists():
        spatial_record=read(spatial/'summary.json');src=read(spatial/'source_snapshot.json')
        require(spatial_record['complete'] and spatial_record['tasks']==8 and spatial_record['diagnostic_only'] and
                spatial_record['target_gt_after_parent_scoring'] and spatial_record['parent_summary_sha256']==sha(root/'summary.json') and
                spatial_record['parent_seal_sha256']==sha(root/'predictions_complete.json') and
                spatial_record['figure_sha256']==sha(spatial/'errors.png') and
                spatial_record['source_snapshot_sha256']==sha(spatial/'source_snapshot.json') and
                hashlib.sha256(src['text'].encode()).hexdigest()==src['sha256'],'N16 spatial binding differs')
        for p in spatial.iterdir():
            if p.is_file():receipts[str(p)]=sha(p)
    return dict(**item,complete=True,status='completed_not_promoted',fresh_prediction_jobs=8,images=56,
        primary_arm=c['primary_arm'],rows=s['rows'],macro=s['macro'],receipts=receipts,decoder=decoders,
        implementation_checks=checks,source_teacher_checks=sources,transitions=s['transitions_diagnostic_only'],
        spatial=spatial_record,independent_test=False,other_jobs_fresh_inference=False,default_changed=False)
