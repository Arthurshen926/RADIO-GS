"""Paired post-prediction domain and full-source diagnostics, never model inputs."""
import json
from pathlib import Path
from scripts.research.run_query_field_joint_exclusion_domain import ROOT,OLD_JOINT,ARMS,read,write,receipt


def main():
    cfg=read(ROOT/'prediction_config.json');per={};domain=[];conditions=[];inputs=[]
    for job in cfg['jobs']:
        scene=job['scene'];path=ROOT/'diagnostics'/('source_'+scene+'.json')
        source=read(path);old=read(OLD_JOINT/'diagnostics'/('source_'+scene+'.json'))
        dpath=ROOT/'diagnostics'/('domain_'+scene+'.json');d=read(dpath)
        inputs.extend([receipt(path),receipt(dpath)])
        assert source['query_view_pairs']==old['query_view_pairs']
        assert source['positive']==old['positive'] and source['negative']==old['negative']
        for a,b in zip(source['records'],old['records']):
            for k in ['query','text','frame','positive','negative']:assert a[k]==b[k]
            for current,previous in [('semantic_initial','semantic_initial'),('observed_fixed','frozen_visibility'),('observed_joint','joint_coverage')]:
                assert a['arms'][current]==b['arms'][previous]
        for a in ['frozen_visibility','joint_coverage']:
            assert source['totals'][a]['sampled_positive_lost']==0
        per[scene]=dict(query_view_pairs=source['query_view_pairs'],positive=source['positive'],negative=source['negative'],
            totals=source['totals'],partial_known_iou={a:source['totals'][a]['positive_covered']/max(source['positive']+source['totals'][a]['negative_covered'],1) for a in ARMS})
        domain.extend(d['rows'])
        conditions.extend(read(ROOT/'predictions'/job['name']/'prediction.json')['diagnostics'])
    source_totals={a:{k:sum(s['totals'][a][k] for s in per.values()) for k in next(iter(per.values()))['totals'][a]} for a in ARMS}
    keys=['initial_selected','initial_factor_touched','initial_semantic_only','removed','removed_observed','removed_semantic_only','remaining_untouched']
    report=dict(complete=True,per_scene=per,source_totals=source_totals,
        source_query_view_pairs=sum(s['query_view_pairs'] for s in per.values()),
        known_positive_occurrences=sum(s['positive'] for s in per.values()),
        known_negative_occurrences=sum(s['negative'] for s in per.values()),
        old_controls_render_exact=True,source_labels_unchanged=True,partial_pseudo_labels_not_GT=True,
        occurrences_not_independent_samples=True,inference_inputs_changed=False,
        domain_totals={k:sum(r[k] for r in domain) for k in keys},
        query_conditions={k:sum(x['controls']['joint_coverage'][k]==0 for x in conditions) for k in ['sampled_positive','positive_covered_before','negative_covered_before']},
        partial_known_iou_scene_macro={a:sum(s['partial_known_iou'][a] for s in per.values())/4 for a in ARMS},
        inputs=inputs,model_promotion=False,whole_five_task=False)
    write(ROOT/'diagnostics/summary.json',report)
    write('docs/experiments/2026-10-11-available-coverage-diagnostics.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['per_scene','inputs']},indent=2))


if __name__=='__main__':main()
