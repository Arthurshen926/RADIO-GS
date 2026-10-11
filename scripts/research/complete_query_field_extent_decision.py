"""Resume the frozen experiment after independently scheduled scene predictions."""
from scripts.research.run_query_field_extent_decision import ROOT, PARENT, ARMS, design
from scripts.research.run_query_field_relations import read, write, receipt, check, execute, lanes


def main():
    d = design(); cfg = read(d['configuration']['path']); records = []
    for j in cfg['jobs']:
        path = ROOT/'predictions'/j['name']/'prediction.json'; r = read(path)
        assert r['configuration'] == cfg and r['job'] == j and not r['target_labels_opened']
        check(dict(path=str(path.with_name('predictions.pt')), sha256=r['predictions_sha256']))
        records.append(receipt(path))
    barrier = dict(records=records, queries=sum(read(r['path'])['queries'] for r in records), controls=len(ARMS))
    assert barrier['queries'] == 68
    if (ROOT/'predictions_complete.json').exists(): assert read(ROOT/'predictions_complete.json') == barrier
    else: write(ROOT/'predictions_complete.json', barrier)
    def score(j, gpu):
        path = ROOT/'predictions'/j['name']/'evaluation/evaluation.json'
        if path.exists(): return
        execute(['-m', 'radio_gs.query_field.public_lerf', 'score', '--config', str(ROOT/'prediction_config.json'),
            '--job', j['name'], '--output', str(ROOT/'predictions')], ROOT/'logs'/('score_'+j['scene']+'.log'), gpu)
    lanes(cfg['jobs'], [0, 2], score)
    per = {a: {} for a in ARMS}
    for j in cfg['jobs']:
        metrics = read(ROOT/'predictions'/j['name']/'evaluation/evaluation.json')['metrics']
        previous = read(PARENT/'predictions'/j['name']/'evaluation/evaluation.json')['metrics']
        for a, m in metrics.items():
            for k in ['iou', 'ap', 'localization', 'observations']: assert m[k] == previous['semantic_initial'][k]
            if a != 'extent_discrete':
                name = dict(dense_joint='admitted_joint', source_initial='admitted_fusion').get(a, a)
                for k in ['miou', 'count', 'missing']: assert m['selected'][k] == previous[name]['selected'][k]
            per[a][j['scene']] = m['selected']['miou']
    report = dict(complete=True, per_scene=per, macro={a: sum(v.values())/4 for a, v in per.items()},
        privileged_scope=True, transient_query_fitting=False, standard_deployment=False,
        predictions_before_scoring=True, default_promoted=False, whole_five_task=False)
    write(ROOT/'summary.json', report); print(report, flush=True)


if __name__ == '__main__': main()
