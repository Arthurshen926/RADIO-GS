"""Collect completed evaluations without comparing run-specific PNG paths."""
from scripts.research.run_query_field_scope_topology import ROOT,PARENT,ARMS,design
from scripts.research.run_query_field_relations import read,write


def main():
    design();cfg=read(ROOT/'prediction_config.json');per={a:{} for a in ARMS}
    assert read(ROOT/'predictions_complete.json')['queries']==68
    for j in cfg['jobs']:
        m=read(ROOT/'predictions'/j['name']/'evaluation/evaluation.json')['metrics']
        previous=read(PARENT/'predictions'/j['name']/'evaluation/evaluation.json')['metrics']
        for a in ARMS:
            for k in ['iou','ap','localization','observations']:assert m[a][k]==previous['seed_addition'][k]
            if a in ARMS[:2]:
                for k in ['miou','count','missing']:assert m[a]['selected'][k]==previous[a]['selected'][k]
                left={(r['frame'],r['query']):r for r in m[a]['selected']['objects']}
                right={(r['frame'],r['query']):r for r in previous[a]['selected']['objects']}
                assert left.keys()==right.keys()
                for k in left:
                    for value in ['iou','gt_path','missing']:assert left[k][value]==right[k][value]
            per[a][j['scene']]=m[a]['selected']['miou']
    write(ROOT/'summary.json',dict(complete=True,per_scene=per,macro={a:sum(v.values())/4 for a,v in per.items()},
        matched_positive_scope_privileged=True,native_complete_scope_source_free=True,
        predictions_before_scoring=True,default_promoted=False,whole_five_task=False,
        collector_fix='compare invariant metrics and GT bindings; prediction PNG paths necessarily differ by run'))


if __name__=='__main__':main()
