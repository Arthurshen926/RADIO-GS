"""Retained-evidence checks for the N16 numerical diagnostic (A24)."""
import hashlib
import json
from pathlib import Path
from statistics import mean
from .reference_study import read, sha, require, check_binary_metrics


def check_score_precision_audit(repo, study, item, jobs):
    root = study/item['run']
    if not (root/'summary.json').exists():
        return dict(**item,complete=False,status='failed' if (root/'failure.json').exists()
            else 'in_progress' if root.exists() else 'planned')
    m,s,seal,snapshot = [read(root/p) for p in ['manifest.json','summary.json','predictions_complete.json','source_snapshot.json']]
    c=m['config'];parent=study/c['parent_run'];ps=read(parent/'summary.json')
    expected=[j for j in jobs if j['benchmark']=='nvos'];names={j['name'] for j in expected}
    require(m['jobs']==expected==read(parent/'manifest.json')['jobs'] and len(names)==8,'A24 cohort differs')
    require(c==json.loads(snapshot['configs/research/nvos_score_precision.json']['text']) and
        c['arms']==['quality_image','full_image','crop_image'] and c['diagnostic_only'],'A24 recipe differs')
    require(s['complete'] and s['diagnostic_only'] and not s['actual_3d_rerun'] and not s['default_changed'] and
        not s['independent_test'] and not s['other_benchmarks_freshly_evaluated'] and
        s['durable_dense_prediction_bytes']==0 and s['source_hashes_unchanged'],'A24 scope differs')
    require(m['parent_manifest_sha256']==sha(parent/'manifest.json') and
        m['parent_summary_sha256']==sha(parent/'summary.json'),'A24 parent differs')
    require(seal['jobs']==8 and seal['probabilities']==seal['logit_maps']==24 and not seal['target_gt_read'] and
        set(seal['report_hashes'])==set(s['rows'])==names and
        s['prediction_seal_sha256']==sha(root/'predictions_complete.json'),'A24 seal differs')
    require(read(study/'storage_policy.json')==c['storage_policy'] and c['storage_policy']['study_bytes']==8*2**20 and
        c['maximum_new_durable_bytes']==2**20 and c['maximum_volatile_prediction_bytes']==256*2**20,'A24 budget differs')
    receipts={str(p):sha(p) for p in root.iterdir() if p.is_file()}
    for p,h in m['source_hashes'].items():
        require(snapshot[p]['sha256']==h==hashlib.sha256(snapshot[p]['text'].encode()).hexdigest(),'A24 archive differs')
    for p,h in {**m['external_interface_hashes'],**s['gt_hashes']}.items():
        require(sha(p)==h,'A24 input/interface changed');receipts[p]=h
    for name in names:
        p=root/(name+'_prediction.json');r=read(p);pr=read(parent/(name+'_prediction.json'))
        require(sha(p)==seal['report_hashes'][name] and r['parent_report_sha256']==sha(parent/(name+'_prediction.json')) and
            not r['target_gt_read'] and r['decoder_exact'] and r['all_probability_hashes_exact'] and
            r['scratch_bytes']<128*2**20,'A24 replay differs')
        require(set(r['entries'])==set(s['rows'][name])==set(c['arms']) and
            s['rows'][name]==read(root/(name+'_evaluation.json')),'A24 task rows differ')
        for a,v in r['entries'].items():
            require(v['probability']==pr['predictions'][a] and v['reconstruction_exact'] and
                0<=v['whole_canvas_sigmoid_max_delta']<=2e-7 and v['logits']['shape']==[762048,1] and
                v['logits']['dtype']=='float32' and v['min_logit']<=v['max_logit'],'A24 exact array differs')
        for a,row in s['rows'][name].items():
            require(set(row['tracks'])==set(s['macro']),'A24 track missing')
            for t,v in row['tracks'].items():
                require(set(v['benchmark_grid'])=={'original_float32','float64','stable_mixture_rank'},'A24 readout missing')
                for mode,value in v['benchmark_grid'].items():
                    check_binary_metrics(value)
                    require(0<=value['average_precision']<=1 and value['tn']>=0,'A24 invalid metric')
                before,old=v['benchmark_grid']['original_float32'],ps['rows'][name][a][t]
                require(all(before[k]==old[k] for k in ['tp','fp','fn','tn','average_precision']),'A24 scoring replay differs')
                require(all(0<=ap<=1 for ap in v['native_grid_diagnostic_ap'].values()),'A24 invalid native AP')
    for t,arms in s['macro'].items():
        for a,variants in arms.items():
            for v,values in variants.items():
                for k,x in values.items():
                    require(abs(x-mean(r[a]['tracks'][t]['benchmark_grid'][v][k] for r in s['rows'].values()))<1e-12,'A24 aggregate differs')
    cleanup=read(root/'cleanup.json');require(cleanup['bank_released'] and cleanup['scratch_released'],'A24 scratch not released')
    require(not Path(m['volatile_directory']).exists(),'A24 bank still exists')
    failed=study/'nvos_score_precision_v1'
    if failed.exists():
        f=read(failed/'failure.json');clean=read(failed/'cleanup.json');archive=read(failed/'source_snapshot.json')
        require(not f['target_gt_opened'] and f['completed_predictions']==1 and clean['bank_released'] and
            clean['scratch_released'] and not (failed/'summary.json').exists(),'A24 failed-attempt scope differs')
        for v in archive.values():require(hashlib.sha256(v['text'].encode()).hexdigest()==v['sha256'],'A24 failure archive differs')
        receipts.update({str(p):sha(p) for p in failed.iterdir() if p.is_file()})
    return dict(**item,complete=True,status='completed_diagnostic_only',rows=s['rows'],macro=s['macro'],receipts=receipts,
        actual_3d_rerun=False,default_changed=False,independent_test=False)
