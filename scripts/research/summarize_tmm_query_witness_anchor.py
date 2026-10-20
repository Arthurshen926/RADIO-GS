"""Generate targeted feedback from sealed Q01 results, never run new inference."""
import json
from pathlib import Path

import numpy as np

from radio_gs.response_study.registry import (
    REPO, REGISTRY, read, sha, source_index, check_targeted_study,
)


def main():
    registry=read(REPO/REGISTRY);study=REPO/registry['study_root'];paper=REPO/registry['paper_root']
    item=next(e for e in registry['targeted_studies'] if e['id']=='Q01')
    root=study/item['run'];summary=read(root/'summary.json')
    verification,rows=check_targeted_study(REPO,study,item,read(REPO/registry['recipe']),
        read(REPO/registry['reference_summary']),source_index(study))
    diagnosis=read(root/'anchor_diagnosis.json')['rows']
    attribution={}
    for arm in item['arms']:
        attribution[arm]={}
        for correct in [True,False]:
            group=[r for r in diagnosis if r['baseline_anchor_correct']==correct]
            # Fixed baseline strata; their conditional means are not full-cohort effects.
            attribution[arm]['baseline_correct' if correct else 'baseline_incorrect']=dict(
                queries=len(group),new_correct=sum(r['arms'][arm]['anchor_correct'] for r in group),
                original_iou_query_mean=float(np.mean([r['baseline_iou'] for r in group])),
                new_iou_query_mean=float(np.mean([r['arms'][arm]['iou'] for r in group])),
                descriptive_conditional_not_causal=True)
    record=dict(summary,summary_sha256=sha(root/'summary.json'),verification=verification,
        fixed_baseline_strata=attribution,rows=rows,
        build_runs_inference=False,source_diagnosis_sha256=sha(root/'anchor_diagnosis.json'))
    (paper/'query_witness_anchor_feedback.json').write_text(json.dumps(record,indent=2)+'\n')
    lines=[r'\begin{tabular}{lrrr}',r'\toprule',
        r'UQIS image output & v10 & Query witness & Permuted witness \\',r'\midrule']
    for metric,label in [('image_ap','AP (\%)'),('image_selected_iou','Actual-set IoU (\%)')]:
        values=[100*summary['uqis_macro'][a][metric] for a in ['baseline']+item['arms']]
        lines.append(label+' & '+' & '.join(f'{v:.4f}' for v in values)+r' \\')
    lines += ['Correct anchors / 67 & '+' & '.join(str(summary['anchor_correct_counts'][a])
        for a in ['baseline']+item['arms'])+r' \\',r'\bottomrule',r'\end{tabular}']
    (paper/'tables/query_witness_anchor.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(verified=True,targeted=True,intervened_queries=67,scored_queries=268,
        new_all31_inference=False,effects=summary['effects']),indent=2))


if __name__=='__main__':main()
