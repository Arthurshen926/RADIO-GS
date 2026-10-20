"""Q02 evidence and finite-proposal regret decomposition; no inference or fitting."""
import json
from collections import Counter

import numpy as np

from radio_gs.response_study.registry import (
    REPO,REGISTRY,read,sha,source_index,check_targeted_study,close,
)


def main():
    registry=read(REPO/REGISTRY);study=REPO/registry['study_root'];paper=REPO/registry['paper_root']
    item=next(e for e in registry['targeted_studies'] if e['id']=='Q02')
    root=study/item['run'];summary=read(root/'summary.json')
    verification,rows=check_targeted_study(REPO,study,item,read(REPO/registry['recipe']),
        read(REPO/registry['reference_summary']),source_index(study))
    capacity=read(root/'proposal_capacity.json');diagnosis=read(root/'anchor_diagnosis.json')['rows']
    queries=[r for p in root.glob('scene*_prediction.json') for r in read(p)['queries']]
    scenes=sorted({r['scene'] for r in capacity['rows']})
    def macro(field):
        return float(np.mean([np.mean([field(r) for r in capacity['rows'] if r['scene']==s]) for s in scenes]))
    baseline=macro(lambda r:r['proposal_iou']['joint'])
    ceiling=macro(lambda r:r['oracle_iou']);actual=macro(lambda r:r['verified_iou'])
    close(baseline,summary['uqis_macro']['baseline']['image_selected_iou'],'capacity baseline')
    close(actual,summary['uqis_macro']['verified']['image_selected_iou'],'capacity automatic')
    close(ceiling,summary['finite_pool_oracle_iou'],'capacity ceiling')
    decomposition=dict(baseline_iou=baseline,finite_pool_oracle_iou=ceiling,automatic_iou=actual,
        available_gain_pp=100*(ceiling-baseline),selection_regret_pp=100*(ceiling-actual),
        realized_gain_pp=100*(actual-baseline),oracle_is_not_a_method=True,
        identity='automatic = baseline + finite-pool available gain - selection regret',
        queries_with_better_proposal=sum(r['oracle_iou']>r['proposal_iou']['joint']+1e-12 for r in capacity['rows']),
        proposal_iou_scene_macro={a:macro(lambda r:r['proposal_iou'][a]) for a in ['joint','siglip','dino']})
    previous_path=study/'query_witness_anchor_v1/anchor_diagnosis.json'
    previous=read(previous_path)['rows']
    previous_gain=float(np.mean([np.mean([max([r['baseline_iou']]+
        [r['arms'][a]['iou'] for a in ['query_witness','permuted_witness']])-r['baseline_iou']
        for r in previous if r['scene']==s]) for s in sorted({r['scene'] for r in previous})]))
    strata={}
    for arm in item['arms']:
        strata[arm]={}
        for correct in [True,False]:
            group=[r for r in diagnosis if r['baseline_anchor_correct']==correct]
            strata[arm]['baseline_correct' if correct else 'baseline_incorrect']=dict(
                queries=len(group),new_correct=sum(r['arms'][arm]['anchor_correct'] for r in group),
                original_iou_query_mean=float(np.mean([r['baseline_iou'] for r in group])),
                new_iou_query_mean=float(np.mean([r['arms'][arm]['iou'] for r in group])),
                diagnostic_conditional_not_causal=True)
    record=dict(summary,summary_sha256=sha(root/'summary.json'),verification=verification,
        rows=rows,finite_pool_decomposition=decomposition,fixed_baseline_strata=strata,
        previous_pool_diagnosis=dict(available_gain_pp=100*previous_gain,
            source=str(previous_path),source_sha256=sha(previous_path),oracle_is_not_a_method=True),
        choices={a:dict(Counter(r['arms'][a]['chosen_proposal'] for r in queries)) for a in item['arms']},
        common_views_quantiles=np.quantile([len(r['arms']['verified']['common_views']) for r in queries],[0,.25,.5,.75,1]).tolist(),
        build_runs_inference=False,source_capacity_sha256=sha(root/'proposal_capacity.json'))
    (paper/'complete_response_verification_feedback.json').write_text(json.dumps(record,indent=2)+'\n')
    lines=[r'\begin{tabular}{lrrr}',r'\toprule',
        r'UQIS image output & v10 & Verified & Permuted \\',r'\midrule']
    for metric,label in [('image_ap',r'AP (\%)'),('image_selected_iou',r'Actual-set IoU (\%)')]:
        values=[100*summary['uqis_macro'][a][metric] for a in ['baseline']+item['arms']]
        lines.append(label+' & '+' & '.join(f'{v:.4f}' for v in values)+r' \\')
    lines += ['Correct anchors / 67 & '+' & '.join(str(summary['anchor_correct_counts'][a])
        for a in ['baseline']+item['arms'])+r' \\',r'\bottomrule',r'\end{tabular}']
    (paper/'tables/complete_response_verification.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(verified=True,finite_pool_decomposition=decomposition,
        fixed_baseline_strata=strata,choices=record['choices']),indent=2))


if __name__=='__main__':main()
