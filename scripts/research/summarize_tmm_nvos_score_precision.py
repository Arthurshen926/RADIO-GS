"""Publish A24 as a score audit, keeping N16's actual 3D results unchanged."""
import csv
import json
from pathlib import Path
from radio_gs.response_study.reference_study import read, require
from radio_gs.response_study.score_precision_audit import check_score_precision_audit

REPO=Path(__file__).resolve().parents[2];PAPER=REPO/'paper/tmm_response'


def main():
    registry=read(REPO/'configs/research/tmm_response.json');item=next(x for x in registry['audits'] if x['id']=='A24')
    r=check_score_precision_audit(REPO,REPO/item['study_root'],item,read(REPO/registry['recipe'])['jobs'])
    require(r['complete'],'A24 incomplete')
    r['effects']={t:{a:dict(
        stable_minus_original_ap_pp=100*(v['stable_mixture_rank']['average_precision']-v['original_float32']['average_precision']),
        float64_minus_original_ap_pp=100*(v['float64']['average_precision']-v['original_float32']['average_precision']),
        original_vs_quality_ap_pp=100*(v['original_float32']['average_precision']-arms['quality_image']['original_float32']['average_precision']),
        stable_vs_quality_ap_pp=100*(v['stable_mixture_rank']['average_precision']-arms['quality_image']['stable_mixture_rank']['average_precision']))
        for a,v in arms.items()} for t,arms in r['macro'].items()}
    r['hard_mask_changes']={a:dict(float64=sum(x[a]['float64_hard_mask_changes'] for x in r['rows'].values()),
        stable=sum(x[a]['stable_hard_mask_changes'] for x in r['rows'].values())) for a in r['macro']['retained_gt_gt127']}
    (PAPER/'nvos_score_precision_feedback.json').write_text(json.dumps(r,indent=2)+'\n')
    rows=[dict(task=n,arm=a,gt_track=t,readout=v,**value) for n,row in r['rows'].items()
        for a,record in row.items() for t,track in record['tracks'].items() for v,value in track['benchmark_grid'].items()]
    with (PAPER/'artifacts/nvos_score_precision.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    table=[r'\begin{tabular}{lrrr}\toprule',r'Image response & Original AP & Float64 AP & Stable mixture AP \\\midrule']
    for a,label in [('quality_image','N14 transport'),('full_image','N16 full frame'),('crop_image','N16 local')]:
        v=r['macro']['retained_gt_gt127'][a]
        table.append(label+' & '+' & '.join(f"{100*v[k]['average_precision']:.4f}" for k in ['original_float32','float64','stable_mixture_rank'])+r' \\')
    table += [r'\bottomrule\end{tabular}']
    (PAPER/'tables/nvos_score_precision.tex').write_text('\n'.join(table)+'\n')
    lines=[r'\section{Response-score numerical audit}',
        'A24 exactly replays all 24 N14/N16 image probability arrays and decoder decisions, '
        'while observing raw logits. All eight tasks are sealed before scoring on both GT conventions. '
        'The original float32 probability readout is reproduced, not replaced. Float64 evaluates '
        'the same sigmoid and bilinear operation at higher precision. The stable-mixture score '
        'independently interpolates positive and complementary probabilities, then takes their log ratio. '
        'It preserves fine upper-tail differences without changing to logit interpolation. '
        'Native-grid AP uses separately nearest-resized GT and remains diagnostic only.',
        r'\begin{center}\small\input{tables/nvos_score_precision.tex}\end{center}',
        'These are numerical diagnostics of image responses, not new 3D method scores. '
        'Actual 3D predictions are not recomputed; the N16 native AP loss and predeclared stopping decision remain unchanged.']
    for a,e in r['effects']['retained_gt_gt127'].items():
        lines.append(a.replace('_',' ')+f": stable minus original AP {e['stable_minus_original_ap_pp']:+.4f} points; "
            f"relative to matched N14 image AP, original {e['original_vs_quality_ap_pp']:+.4f}, stable {e['stable_vs_quality_ap_pp']:+.4f} points. "
            f"Stable-readout hard-mask changes: {r['hard_mask_changes'][a]['stable']} pixels across all tasks.")
    lines += [r'\begin{center}\small\begin{tabular}{lrrr}\toprule',
        r'Task & N14 AP correction & Full AP correction & Local AP correction \\\midrule']
    for n,row in r['rows'].items():
        values=[]
        for a in ['quality_image','full_image','crop_image']:
            v=row[a]['tracks']['retained_gt_gt127']['benchmark_grid']
            values.append(100*(v['stable_mixture_rank']['average_precision']-v['original_float32']['average_precision']))
        lines.append(n.replace('_',r'\_')+' & '+' & '.join(f'{v:+.4f}' for v in values)+r' \\')
    lines += [r'\bottomrule\end{tabular}\end{center}']
    (PAPER/'sections/nvos_score_precision.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(macro=r['macro'],effects=r['effects'],hard_mask_changes=r['hard_mask_changes']),indent=2))


if __name__=='__main__':main()
