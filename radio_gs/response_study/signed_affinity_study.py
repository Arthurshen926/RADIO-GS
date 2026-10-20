"""Validate N09 full-cohort signed evidence transport and actual 3D outputs."""
import hashlib,json,math
from statistics import mean
from .reference_study import read,sha,require,check_binary_metrics

ARMS=['v10','static_image','static_native','signed_affinity','pseudo_affinity','signed_image','pseudo_image','signed_native','pseudo_native']


def check_signed_affinity_study(repo,study,item,jobs):
    root=study/item['run']
    if not (root/'spatial_diagnosis.json').exists():
        return dict(**item,complete=False,status='failed' if (root/'failure.json').exists() else 'in_progress' if root.exists() else 'planned')
    m,s,seal,snapshot,spatial=[read(root/p) for p in ['manifest.json','summary.json','predictions_complete.json','source_snapshot.json','spatial_diagnosis.json']]
    c=m['config'];expected=[j for j in jobs if j['benchmark']=='nvos'];names={j['name'] for j in expected}
    require(len(names)==8 and m['jobs']==expected and c['arms']==ARMS,'N09 cohort/arms changed')
    require(c==json.loads(snapshot['configs/research/nvos_signed_affinity.json']['text']) and c['solver_steps']==120
            and c['footprint_backend']=='gsplat_cuda' and c['native_wh']==[1008,756] and c['source_hw']==[384,512]
            and c['threshold']==seal['threshold']==.5 and c['primary_arm']==s['primary_arm']=='signed_native','N09 design changed')
    require(s['complete'] and not s['target_gt_used_for_prediction'] and not s['per_task_winners'] and not s['independent_test']
            and not s['default_changed'] and s['average_precision_measured'] and s['negative_scribble_used'] and s['target_rgb_used']
            and s['durable_dense_prediction_bytes']==0,'N09 access/scope changed')
    require(seal['jobs']==8 and seal['images']==72 and not seal['target_gt_read']
            and set(seal['report_hashes'])==set(seal['predictions'])==set(s['rows'])==names
            and sha(root/'predictions_complete.json')==s['prediction_seal_sha256'],'N09 incomplete seal')
    require(spatial['complete'] and spatial['diagnostic_only'] and set(spatial['transitions'])==names
            and spatial['prediction_seal_sha256']==sha(root/'predictions_complete.json')
            and spatial['summary_sha256']==sha(root/'summary.json') and spatial['figure_sha256']==sha(root/'errors.png'),'N09 spatial binding changed')
    receipts={str(p):sha(p) for p in root.iterdir() if p.is_file()}
    for p,h in m['source_hashes'].items():require(hashlib.sha256(snapshot[p]['text'].encode()).hexdigest()==h==snapshot[p]['sha256'],'N09 source archive changed')
    inputs=dict(s['gt_hashes']);checks={};source_checks={};decoders={}
    prior=study/'nvos_native_writeback_v3';old=read(prior/'summary.json');oldseal=read(prior/'predictions_complete.json')
    parent=study/'nvos_memory_response_v2';pseal=read(parent/'predictions_complete.json')
    replay={}
    for name in names:
        r=read(root/(name+'_prediction.json'));row=s['rows'][name]
        require(sha(root/(name+'_prediction.json'))==seal['report_hashes'][name] and r['job'] in expected and r['job']['name']==name
                and not r['target_gt_read'] and r['negative_scribble_used'] and r['target_rgb_used']
                and r['predictions']==seal['predictions'][name] and set(r['predictions'])==set(ARMS),'N09 prediction changed')
        require(read(root/(name+'_evaluation.json'))==row and set(row)==set(ARMS),'N09 evaluation changed')
        checks[name]=cc=r['implementation_checks']
        require(cc['v10_binary_exact'] and all(0<=cc[k]<1e-4 for k in ['v10_render_max_error','compositor_alpha_max_error','normalized_operator_max_error']),'N09 renderer check failed')
        require(set(r['lifting'])=={'static_native','signed_native','pseudo_native'} and all(v['unsupported_changed']==0 and v['source_image_loss_after']<=v['source_image_loss_before']+1e-5 for v in r['lifting'].values()),'N09 unsupported/fit check failed')
        decoders[name]=d=r['decoder'];require(d['negative_scribble_used'] and not d['target_gt_available'] and d['sam_eval'] and d['dino_eval'] and d['feature_grid']==[54,72],'N09 worker contract changed')
        require(all(m['source_hashes'][p]==h for p,h in d['source_hashes'].items()),'N09 worker source differs')
        for a in ['signed','pseudo']:
            v=d['decoder'][a];require(v['source_only_scale'] and not v['reference_extent_gate'] and v['positive_patches']+v['negative_patches']+v['unknown_patches']==3888,'N09 patch authority differs')
            # The worker divides float32 tensors; receipt arithmetic is float64.
            if v['fallback'] is None:require(v['temperature']>0 and abs(v['temperature']-v['median_signed_loo_margin']/math.log(9))<1e-7,'N09 source scale differs')
        source_checks[name]=r['source_teacher_checks'];require(source_checks[name] and all(set(v['confidence_weighted_mean_huber'])=={'v10','static_native','signed_native','pseudo_native'} for v in source_checks[name]),'N09 source checks missing')
        require(r['predictions']['static_image']['sha256']==pseal['predictions'][name]['static_image']['sha256'],'N09 static teacher differs from N08')
        replay[name]=dict(static_native_float_exact=r['predictions']['static_native']['sha256']==oldseal['predictions'][name]['static_image_native']['sha256'],
            static_native_confusion_exact=all(row['static_native'][t][k]==old['rows'][name]['static_image_native'][t][k]
                for t in c['gt_tracks'] for k in ['tp','fp','fn','tn']),
            static_native_iou_delta_pp={t:100*(row['static_native'][t]['iou']-old['rows'][name]['static_image_native'][t]['iou']) for t in c['gt_tracks']})
        require(replay[name]['static_native_confusion_exact'],'N09 static native binary counts differ from N08')
        require(all(abs(v)<.001 for v in replay[name]['static_native_iou_delta_pp'].values()),'N09 static native replay changed materially')
        for arm,tracks in row.items():
            require(set(tracks)==set(c['gt_tracks']),'N09 GT tracks missing')
            for track,v in tracks.items():
                check_binary_metrics(v);count=sum(v[k] for k in ['tp','fp','fn','tn']);base=row['v10'][track]
                require(type(v['tn']) is int and v['tn']>=0 and v['independent_confusion_exact'] and 0<=v['average_precision']<=1
                        and abs(v['accuracy']-(v['tp']+v['tn'])/count)<1e-12 and v['tp']+v['fn']==base['tp']+base['fn']
                        and v['tn']+v['fp']==base['tn']+base['fp'],'N09 independent counts/ranking invalid')
                if arm=='v10':require(all(v[k]==old['rows'][name]['v10'][track][k] for k in ['tp','fp','fn','tn']),'N09 v10 control changed')
        for p,h in r['input_hashes'].items():require(p not in inputs or inputs[p]==h,'N09 inconsistent input');inputs[p]=h
    for key in ['image_checkpoint','dino_checkpoint']:
        inputs[c[key]]=m['checkpoint_sha256' if key=='image_checkpoint' else 'dino_checkpoint_sha256']
    inputs['/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/manifests/nvos_strict_unseen_v1.json']=m['dataset_manifest_sha256']
    require(set(s['macro'])==set(c['gt_tracks']),'N09 missing means')
    for t,arms in s['macro'].items():
        require(set(arms)==set(ARMS),'N09 macro arms missing')
        for a,metrics in arms.items():
            require(set(metrics)==set(c['metrics']),'N09 metrics changed')
            for k,v in metrics.items():require(abs(v-mean(r[a][t][k] for r in s['rows'].values()))<1e-12,'N09 mean differs')
    require(s['other_benchmarks']==old['other_benchmarks'] and not s['other_benchmarks_freshly_evaluated'],'N09 other benchmarks changed')
    for p,h in inputs.items():require(sha(p)==h,'N09 input changed: '+p);receipts[p]=h
    for p in [prior/'summary.json',prior/'predictions_complete.json',parent/'predictions_complete.json']:receipts[str(p)]=sha(p)
    return dict(**item,complete=True,status='completed_not_promoted',fresh_prediction_jobs=8,images=72,rows=s['rows'],macro=s['macro'],
        receipts=receipts,implementation_checks=checks,source_teacher_checks=source_checks,decoders=decoders,static_replay=replay,
        spatial_diagnosis=spatial,primary_arm='signed_native',negative_scribble_used=True,average_precision_measured=True,
        independent_test=False,other_jobs_fresh_inference=False,default_changed=False)
